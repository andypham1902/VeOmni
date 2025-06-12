import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import wandb
from tqdm import trange
import pandas as pd
from openai import OpenAI
from omegaconf import DictConfig
import numpy as np
from multiprocessing import Pool
import functools

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce, all_gather_defaultdict_v1
from veomni.utils.validation_utils import ValidationVLLMManager

from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device
import logging

logger = helper.create_logger(__name__)


def extract_content(solution_str):
    """Remove reasoning content from the solution string.
    Args:
        solution_str (str): The solution string containing reasoning content.
    Returns:
        str: The solution string with reasoning content removed.
    """
    reasoning_tags = ["</think>", "</thinking>"]
    for tag in reasoning_tags:
        if tag in solution_str:
            solution_str = solution_str.split(tag)[1].strip()
    return solution_str


def compute_score(generation_result: Dict[str, Any], ground_truth: str, data_source: str) -> float:
    generation_result = extract_content(generation_result)
    if data_source in ['hoanganh/Medical-Train', 'TsinghuaC3I/MedXpertQA', 'hoanganh/MedQA-Test']:
        from veomni.utils.reward_score.medical import compute_score
        res = compute_score(generation_result, ground_truth)
    elif data_source in ['google/IFEval']:
        from veomni.utils.reward_score.ifeval import ifeval
        res = ifeval.compute_score(generation_result, ground_truth)
    elif data_source in ['openai/HealthBench']:
        from veomni.utils.reward_score.healthbench import healthbench
        res = healthbench.compute_score(generation_result, ground_truth)
    else:
        raise NotImplementedError(f"Unsupported dataset: {data_source}. Please implement compute_score for this dataset.")
    return res

def compute_score_wrapper(args):
    generation_result, ground_truth, dataset_name = args
    return compute_score(generation_result['response'], ground_truth, dataset_name)


@dataclass
class ExtendedDataArguments(DataArguments):
    """Extended data arguments with validation support"""
    val_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of the validation data. Use comma to separate multiple datasets."},
    )
    val_size: int = field(
        default=1000,
        metadata={"help": "Number of validation samples to evaluate."},
    )
    dataset_metrics_config: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string mapping dataset names to metric types. E.g., '{\"medqa\": \"mcqa\", \"healthbench\": \"health\"}'"},
    )


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    """Extended training arguments with validation support"""
    validation_steps: int = field(
        default=100,
        metadata={"help": "Run validation every N training steps."},
    )
    validation_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of validation batches. None means use all."},
    )
    

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "ExtendedDataArguments" = field(default_factory=ExtendedDataArguments)
    train: "ExtendedTrainingArguments" = field(default_factory=ExtendedTrainingArguments)


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    global_step: int,
    args: Arguments,
    model_fwd_context,
    tokenizer,
    validation_manager: Optional[ValidationVLLMManager] = None,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run validation and compute metrics with text generation
    
    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        global_step: Current global training step
        args: Training arguments
        model_fwd_context: Forward context for model execution
        tokenizer: Tokenizer for decoding generated text
        limit_batches: Limit number of batches to process 
        
    Returns:
        Dictionary of validation metrics
    """
    logger.info(f"GPU {args.train.global_rank}: Starting validation at step {global_step}")
    logger.info(f"GPU {args.train.global_rank}: Validation dataloader has {len(val_dataloader)} batches")
    
    # Log sampler information for each GPU
    if hasattr(val_dataloader, 'sampler') and val_dataloader.sampler is not None:
        logger.info(f"GPU {args.train.global_rank}: Using distributed sampler - rank {val_dataloader.sampler.rank}/{val_dataloader.sampler.num_replicas}")
    elif hasattr(val_dataloader, '_dataloader') and hasattr(val_dataloader._dataloader, 'sampler'):
        # For DynamicBatchSizeDataLoader
        sampler = val_dataloader._dataloader.sampler
        if sampler is not None:
            logger.info(f"GPU {args.train.global_rank}: Using distributed sampler (dynamic) - rank {sampler.rank}/{sampler.num_replicas}")
        else:
            logger.info(f"GPU {args.train.global_rank}: No distributed sampler found in dynamic dataloader!")
    else:
        logger.info(f"GPU {args.train.global_rank}: No distributed sampler found!")
    
    model.eval()
    
    total_batches = 0
    
    # Initialize per-dataset metric tracking
    dataset_metrics = defaultdict(lambda: {
        'scores': [],
        'prompt_lengths': [],
        'response_lengths': [],
        'total_samples': 0
    })
    
    # Create iterator and progress bar
    val_iterator = iter(val_dataloader)
    max_batches = limit_batches if limit_batches is not None else len(val_dataloader)
    
    validation_tqdm = trange(
        max_batches,
        desc="Validation",
        disable=args.train.local_rank != 0,
    )

    # Get model dtype from training arguments, not from parameters
    if args.train.enable_mixed_precision:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    
    logger.info_rank0(f"Using validation model dtype: {model_dtype} (mixed_precision: {args.train.enable_mixed_precision})")


    with validation_manager.validation_context():  
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.train.enable_mixed_precision, dtype=model_dtype):
                for batch_idx in validation_tqdm:
                    micro_batches: List[Dict[str, Any]] = next(val_iterator)
                    for micro_batch in micro_batches:
                        # Extract and remove non-tensor fields BEFORE moving to GPU and collation
                        dataset_names = micro_batch['data_source']
                        reward_models = micro_batch['reward_model']['ground_truth']
                        ids = micro_batch['id']
                        assert len(ids) == len(reward_models) == len(dataset_names), "Inconsistent batch sizes"

                        # Handle cases where these fields might be tensors, lists, or single values
                        batch_size = micro_batch['input_ids'].size(0) if 'input_ids' in micro_batch else 1
                        
                        micro_batch = {
                            k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in micro_batch.items()
                        }
                        
                        # Get original inputs
                        input_ids = micro_batch['input_ids']
                        attention_mask = micro_batch['attention_mask']
                        position_ids = micro_batch['position_ids']

                        # Debug logging for first sample
                        if batch_idx == 0:
                        # if dataset_names[i] == "openai/HealthBench":
                            question_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                            logger.info(f"GPU {args.train.global_rank} - Sample 0 - Question: {question_text[:200]}...")
                            logger.info(f"GPU {args.train.global_rank} - Sample 0 - Data Source: {dataset_names[0]}")
                            logger.info(f"GPU {args.train.global_rank} - Sample 0 - ID: {ids[0]}")
                            logger.info(f"GPU {args.train.global_rank} - Using validation manager for generation.")
                            
                        generation_kwargs = {
                            "bos_token_id": 151643,
                            "do_sample": False,
                            "validate": True,
                            "eos_token_id": [
                                151645,
                                151643
                            ],
                            "pad_token_id": 151643,
                            "temperature": 0.0,
                        }
                        generation_results = validation_manager.generate_responses(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **generation_kwargs
                        )

                        if batch_idx == 0:  # Log first sample of first batch
                            logger.info(f"GPU {args.train.global_rank} - Sample 0 - Generated: {generation_results[0]['response'][:200]}...")
                        # if dataset_names[i] == "openai/HealthBench":
                            # logger.info(f"GPU {args.train.global_rank} - Batch {batch_idx}, Sample {i} - Ground Truth: {reward_models[i][:200]}...")
                        assert len(generation_results) == len(reward_models), "Mismatch in generated results size"
                        # Update metrics per dataset
                        # Use multiprocessing to parallelize score computation
                        
                        # Prepare arguments for multiprocessing
                        score_args = [(gen_result, gt, ds_name) for gen_result, gt, ds_name in zip(generation_results, reward_models, dataset_names)]
                        
                        # Use multiprocessing to compute scores in parallel
                        with Pool(processes=2) as pool:  # Limit to 2 processes to avoid overhead
                            scores = pool.map(compute_score_wrapper, score_args)
                        
                        # Update metrics per dataset
                        for (generation_result, ground_truth, dataset_name), score in zip(zip(generation_results, reward_models, dataset_names), scores):
                            # Track metrics per dataset
                            dataset_metrics[dataset_name]['scores'].append(score)
                            dataset_metrics[dataset_name]['prompt_lengths'].append(generation_result['prompt_length'])
                            dataset_metrics[dataset_name]['response_lengths'].append(generation_result['response_length'])
                            dataset_metrics[dataset_name]['total_samples'] += 1

                    total_batches += 1
                    
                    # Update progress bar
                    validation_tqdm.set_postfix_str(
                        f"Processed {total_batches} batches, "
                    )
                    
                validation_tqdm.close()
    time.sleep(2)  # Allow time for vLLM to release memory
    # Aggregate per-dataset metrics across all GPUs
    val_metrics = all_gather_defaultdict_v1(dataset_metrics)
    logger.info_rank0(f"Gathered metrics from datasets across all GPUs: {val_metrics}")

    val_metrics_summary = {}
    if args.train.global_rank == 0:
        for dataset_name, metrics in val_metrics.items():
            # Extract scalar values safely
            val_metrics_summary[f'{dataset_name}_accuracy'] = np.mean(metrics['scores']) if metrics['scores'] else 0.0
            val_metrics_summary[f'{dataset_name}_avg_prompt_length'] = np.mean(metrics['prompt_lengths']) if metrics['prompt_lengths'] else 0.0
            val_metrics_summary[f'{dataset_name}_avg_response_length'] = np.mean(metrics['response_lengths']) if metrics['response_lengths'] else 0.0
            val_metrics_summary[f'{dataset_name}_total_samples'] = metrics['total_samples']
            logger.info_rank0(f"Gathered '{dataset_name}' successfully - {metrics['total_samples']} samples, ")

    model.train()  # Switch back to training mode
    return val_metrics_summary


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(args.train.data_parallel_size, args.train.tensor_parallel_size), mesh_dim_names=["dp", "infer_tp"])

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
        chat_template_val = build_chat_template("chatml_val", tokenizer)
        transform_val = partial(
            process_sft_example,
            chat_template=chat_template_val,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    # Build training dataset
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    # Build validation dataset if provided
    val_dataloader = None
    if args.data.val_path:
        logger.info_rank0("Building validation dataset")
        val_dataset = build_mapping_dataset(
            args.data.val_path,
            transform=transform_val,
        )
        
        from veomni.data.data_collator import DataCollatorWithPadding, MakeMicroBatchCollator
        from veomni.distributed.parallel_state import get_parallel_state
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader

        
        # Use simple padding collator without any packing/concatenation
        val_collate_fn = DataCollatorWithPadding()
        
        # Add micro-batch wrapper for consistency with training loop expectations
        val_collate_fn = MakeMicroBatchCollator(
            num_micro_batch=1,  # Single micro-batch for validation
            internal_data_collator=val_collate_fn
        )
        
        # Create distributed sampler for validation
        parallel_state = get_parallel_state()
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=parallel_state.dp_size,
            rank=parallel_state.dp_rank,
            shuffle=False,
            drop_last=False,  # This ensures all samples are used
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            sampler=val_sampler,
            num_workers=args.data.num_workers,
            collate_fn=val_collate_fn,
            pin_memory=args.data.pin_memory,
            drop_last=False,  # Don't drop last batch in validation
            prefetch_factor=args.data.prefetch_factor,
        )

        logger.info_rank0(f"Validation dataloader built with {len(val_dataset)} samples")
        logger.info_rank0(f"Validation dataloader has {len(val_dataloader)} batches per GPU")
        logger.info_rank0(f"Validation dataloader type: {type(val_dataloader)}")
        
        # Log sampler information for debugging
        if hasattr(val_dataloader, 'sampler') and val_dataloader.sampler is not None:
            logger.info_rank0(f"Validation sampler type: {type(val_dataloader.sampler)}")
            logger.info_rank0(f"Validation sampler rank: {val_dataloader.sampler.rank}/{val_dataloader.sampler.num_replicas}")
        else:
            logger.info_rank0("Validation dataloader has no distributed sampler - this may cause duplicate data!")
        logger.info(f"GPU {args.train.global_rank}: Validation dataloader has {len(val_dataloader)} batches")

    logger.info_rank0("Prepare model")
    # FIXED: Correct the inverted logic - mixed precision should use bfloat16, not float32
    if args.train.enable_mixed_precision:
        torch_dtype = "bfloat16"
    else:
        torch_dtype = "float32"
    
    logger.info_rank0(f"Building model with torch_dtype: {torch_dtype} (enable_mixed_precision: {args.train.enable_mixed_precision})")
    
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )
    

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        # save model_assets before training
        model_assets = [model_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()

        # vLLM configuration
    vllm_config = DictConfig({
        "tensor_model_parallel_size": 1,
        "max_num_batched_tokens": 8448 * 8,
        "dtype": "bfloat16",
        "enforce_eager": False,
        "free_cache_engine": False,
        "gpu_memory_utilization": 0.85,
        "response_length": 8192,
        "prompt_length": 128,
        "max_model_len": 8448,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "load_format": "dummy_hf",
    })
    
    validation_manager = ValidationVLLMManager(
        fsdp_model=model,  # Your FSDP wrapped model
        model_path=args.model.model_path,  # Path to the model for vLLM
        tokenizer=tokenizer,
        model_hf_config=model_config,
        vllm_config=vllm_config,
        device_mesh=rollout_device_mesh,  # Your device mesh
        full_params=False,
        offload_param=True,  # Offload parameters to CPU when not in use
        load_format='dummy_hf',  # or 'safetensors' if base model is preloaded
        layered_summon=True
    )
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        
        # Also set epoch for validation dataloader to ensure different data distribution
        if hasattr(val_dataloader, "set_epoch"):
            val_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()
            log_gpu_memory_usage(f"Before backward step {global_step}", logger=logger, level=logging.INFO)
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss.mean() / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            # Run validation periodically
            validation_metrics = {}
            if val_dataloader and (global_step % args.train.validation_steps == 0 or global_step == 1):
                logger.info_rank0(f"Starting validation at step {global_step}")
                # offload_fsdp_model_to_cpu(model)
                # offload_fsdp_optimizer(optimizer=optimizer)
                validation_metrics = run_validation(
                    model=model,
                    val_dataloader=val_dataloader,  # Use the properly distributed dataloader
                    global_step=global_step,
                    args=args,
                    model_fwd_context=model_fwd_context,
                    tokenizer=tokenizer,
                    validation_manager=validation_manager,
                    limit_batches=args.train.validation_limit,
                )
                load_fsdp_model_to_gpu(model.cuda())
                # load_fsdp_optimizer(optimizer=optimizer, device_id=get_torch_device().current_device())
                
                # Log validation metrics
                if args.train.global_rank == 0:
                    logger.info_rank0(f"Validation metrics at step {global_step}:")
                    for key, value in validation_metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info_rank0(f"  {key}: {value:.4f}")
                        else:
                            logger.info_rank0(f"  {key}: {value}")

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    # Combine training and validation metrics
                    all_metrics = train_metrics.copy()
                    all_metrics.update({
                        "training/loss": total_loss, 
                        "training/grad_norm": grad_norm, 
                        "training/lr": lr
                    })
                    
                    # Add validation metrics with proper dataset-specific grouping for wandb
                    for key, value in validation_metrics.items():
                        if key.endswith('_accuracy'):
                            dataset_name = key.replace('_accuracy', '')
                            all_metrics[f"validation/{dataset_name}/accuracy"] = value
                        elif key.endswith('_avg_prompt_length'):
                            dataset_name = key.replace('_avg_prompt_length', '') 
                            all_metrics[f"validation/{dataset_name}/avg_prompt_length"] = value
                        elif key.endswith('_avg_response_length'):
                            dataset_name = key.replace('_avg_response_length', '')
                            all_metrics[f"validation/{dataset_name}/avg_response_length"] = value
                        elif key.endswith('_total_samples'):
                            dataset_name = key.replace('_total_samples', '')
                            all_metrics[f"validation/{dataset_name}/total_samples"] = value
                        else:
                            # General validation metrics
                            all_metrics[f"validation/{key}"] = value
                    
                    wandb.log(all_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()