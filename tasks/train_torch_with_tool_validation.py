"""
Training script with tool-based validation for VeOmni.

This script extends the existing validation training script to support
tool-based agent validation during training.
"""

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
import gc
import psutil

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

# Import tool validation components
from veomni.utils.tool_validation_utils import ToolValidationVLLMManager, ToolServer, ToolCallResult
from veomni.utils.agent_manager import AgentActorManager, AgentConfig
from veomni.utils.reward_score.tool_based import compute_tool_score

from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device
import logging
# Create custom tool server for web_search and web_visit
from veomni.utils.search_tools import web_search, web_visit

class WebToolServer(ToolServer):
    async def execute_tool(self, tool_name: str, tool_input: str, **kwargs) -> ToolCallResult:
        import time
        start_time = time.time()
        
        try:
            if tool_name == "web_search":
                # Parse parameters from tool_input
                import json
                try:
                    params = json.loads(tool_input)
                    query = params.get("query", tool_input)
                    top_k = params.get("top_k", 3)
                    preview_char = params.get("preview_char", 256)
                except:
                    query = tool_input
                    top_k = 3
                    preview_char = 256
                
                result = web_search(query=query, top_k=top_k, preview_char=preview_char)
                execution_time = time.time() - start_time
                return ToolCallResult(
                    success=True, 
                    output=result, 
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            
            elif tool_name == "web_visit":
                # Parse URL from tool_input
                try:
                    params = json.loads(tool_input)
                    url = params.get("url", tool_input)
                except:
                    url = tool_input
                
                result = web_visit(url=url)
                execution_time = time.time() - start_time
                return ToolCallResult(
                    success=True, 
                    output=result, 
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            
            else:
                execution_time = time.time() - start_time
                return ToolCallResult(
                    success=False, 
                    error=f"Unknown tool: {tool_name}", 
                    tool_name=tool_name,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolCallResult(
                success=False, 
                error=str(e), 
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    def get_available_tools(self) -> List[str]:
        return ["web_search", "web_visit"]

logger = helper.create_logger(__name__)

# Global multiprocessing pool for scoring
SCORE_POOL = None


def get_score_pool():
    global SCORE_POOL
    if SCORE_POOL is None:
        SCORE_POOL = Pool(processes=2)
    return SCORE_POOL


def cleanup_score_pool():
    global SCORE_POOL
    if SCORE_POOL is not None:
        SCORE_POOL.close()
        SCORE_POOL.join()
        SCORE_POOL = None


def extract_content(solution_str):
    """Remove reasoning content from the solution string."""
    reasoning_tags = ["</think>", "</thinking>"]
    for tag in reasoning_tags:
        if tag in solution_str:
            solution_str = solution_str.split(tag)[1].strip()
    return solution_str


def log_memory_usage(prefix=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info_rank0(f"{prefix} Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


@dataclass
class ToolDataArguments(DataArguments):
    """Extended data arguments with tool validation support"""
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
class ToolTrainingArguments(TrainingArguments):
    """Extended training arguments with tool validation support"""
    validation_steps: int = field(
        default=100,
        metadata={"help": "Run validation every N training steps."},
    )
    validation_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of validation batches. None means use all."},
    )
    # Tool validation specific arguments
    enable_tool_validation: bool = field(default=False, metadata={"help": "Enable tool-based validation"})
    tool_server_url: Optional[str] = field(default=None, metadata={"help": "URL for external tool server"})
    max_agent_turns: int = field(default=10, metadata={"help": "Maximum turns for agent interaction"})
    tool_call_timeout: int = field(default=30, metadata={"help": "Timeout for tool calls in seconds"})
    agent_total_timeout: int = field(default=300, metadata={"help": "Total timeout for agent interaction"})
    supported_tools: List[str] = field(default_factory=list, metadata={"help": "List of supported tools"})


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: ToolDataArguments = field(default_factory=ToolDataArguments)
    train: ToolTrainingArguments = field(default_factory=ToolTrainingArguments)


def run_tool_validation(validation_manager: ToolValidationVLLMManager, 
                             val_dataloader,
                             args: Arguments,
                             step: int) -> Dict[str, float]:
    """
    Run tool-based validation.
    
    Args:
        validation_manager: Tool validation manager
        val_dataloader: Validation dataloader
        args: Training arguments
        step: Current training step
        
    Returns:
        Dictionary of validation metrics
    """
    logger.info_rank0(f"Starting tool validation at step {step}")
    log_memory_usage("Before tool validation")
    

    # Use local tool servers
    # Set default supported tools to match simple_llm_agent_test.py
    if not args.train.supported_tools:
        args.train.supported_tools = ["web_search", "web_visit"]
    
    tool_server = WebToolServer()
    
    # Create agent manager
    agent_config = AgentConfig(
        max_turns=args.train.max_agent_turns,
        tool_call_timeout=args.train.tool_call_timeout,
        total_timeout=args.train.agent_total_timeout,
        enable_tool_validation=args.train.enable_tool_validation,
    )
    
    agent_manager = AgentActorManager(
        validation_manager=validation_manager,
        tool_server=tool_server,
        config=agent_config
    )
    
    # Track metrics
    tool_metrics = defaultdict(lambda: defaultdict(list))
    regular_metrics = defaultdict(lambda: defaultdict(list))
    
    total_batches = 0
    
    # Validation loop - limit to first few batches for memory safety
    max_val_batches = min(len(val_dataloader), 3)  # Limit validation batches
    
    if args.train.global_rank == 0:
        validation_tqdm = trange(max_val_batches, desc="Tool Validation", leave=False)
        use_tqdm = True
    else:
        validation_tqdm = range(max_val_batches)
        use_tqdm = False
    
    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= max_val_batches:
            break
            
        if use_tqdm:
            validation_tqdm.set_description(f"Processing batch {batch_idx + 1}/{max_val_batches}")
        
        try:
            # Handle micro-batch structure from validation dataloader
            micro_batch = batch[0] if isinstance(batch, list) else batch
            
            # Move batch to device
            input_ids = micro_batch["input_ids"].to(f"cuda:{args.train.local_rank}")
            attention_mask = micro_batch["attention_mask"].to(f"cuda:{args.train.local_rank}")
            position_ids = micro_batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(f"cuda:{args.train.local_rank}")
            
            # Get batch metadata - ground truth is in reward_model column
            batch_reward_models = micro_batch.get("reward_model", {})
            
            # Extract ground truth from batched reward_model structure
            batch_ground_truths = []
            if isinstance(batch_reward_models, dict) and "ground_truth" in batch_reward_models:
                # reward_model is batched: {'ground_truth': ['answer1', 'answer2', ...], 'style': ['ruel', 'ruel', ...]}
                ground_truth_list = batch_reward_models["ground_truth"]
                if isinstance(ground_truth_list, list):
                    batch_ground_truths = ground_truth_list
                else:
                    batch_ground_truths = [ground_truth_list]  # Single item
            else:
                logger.info_rank0(f"Warning: Unexpected reward_model structure: {batch_reward_models}")
                raise ValueError(f"Unexpected reward_model structure: {batch_reward_models}")
            
            # Validate that we have ground truths
            if not batch_ground_truths or any(not gt.strip() for gt in batch_ground_truths):
                logger.info_rank0(f"Warning: Empty ground_truth found in batch: {batch_ground_truths}")
                raise ValueError("Empty ground_truth found - stopping validation")
            
            batch_data_sources = micro_batch.get("data_source", [])
            batch_task_types = micro_batch.get("task_type", ["general"] * input_ids.size(0))
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": 4096,  # Increased for better reasoning
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "eos_token_id": validation_manager.tokenizer.eos_token_id,
                "pad_token_id": validation_manager.tokenizer.pad_token_id,
            }
            
            # Run agent interactions (using asyncio within validation context)
            import asyncio
            agent_results = asyncio.run(agent_manager.run_batch_interactions(
                input_ids,
                attention_mask,
                position_ids,
                **generation_kwargs
            ))
            
            # Evaluate results
            for i, result in enumerate(agent_results):
                idx = i
                ground_truth = batch_ground_truths[idx] if idx < len(batch_ground_truths) else ""
                data_source = batch_data_sources[idx] if idx < len(batch_data_sources) else "unknown"
                task_type = batch_task_types[idx] if idx < len(batch_task_types) else "general"
                
                # Log full conversation history in message format
                if args.train.global_rank == 0:  # Only log from rank 0 to avoid duplicates
                    logger.info_rank0(f"=== Validation Sample {batch_idx}_{idx} ({data_source}) ===")
                    logger.info_rank0(f"Ground Truth: {ground_truth}")
                    logger.info_rank0(f"Turns: {result.total_turns}, Tool Calls: {len(result.tool_calls)}, Success: {result.success}")
                    
                    # Log conversation history in message format
                    if hasattr(result, 'messages') and result.messages:
                        import json
                        logger.info_rank0("Conversation History:")
                        logger.info_rank0(json.dumps(result.messages, indent=2, ensure_ascii=False))
                    
                    logger.info_rank0(f"Final Response: {result.final_response[:300]}...")
                    logger.info_rank0(f"=== End Sample {batch_idx}_{idx} ===\n")
                
                # Compute tool-based score
                tool_score_result = compute_tool_score(
                    result, ground_truth, task_type,
                    question=f"Sample {idx} from {data_source}"
                )
                
                # Log detailed scoring information
                if args.train.global_rank == 0:  # Only log from rank 0 to avoid duplicates
                    detailed_metrics = tool_score_result.get('detailed_metrics', {})
                    
                    logger.info_rank0(f"=== SCORING DETAILS Sample {batch_idx}_{idx} ===")
                    logger.info_rank0(f"Ground Truth: {ground_truth}")
                    logger.info_rank0(f"Full Model Response (first 500 chars): {result.final_response[:500]}{'...' if len(result.final_response) > 500 else ''}")
                    logger.info_rank0(f"Main Answer (reasoning removed): {detailed_metrics.get('main_answer', 'N/A')}")
                    logger.info_rank0(f"Short Answer (GPT-4.1 extracted): {detailed_metrics.get('short_answer', 'N/A')}")
                    logger.info_rank0(f"Judgment (GPT-4o): {detailed_metrics.get('judgment', 'N/A')} (A=correct, B/C=incorrect)")
                    logger.info_rank0(f"Final Answer Correctness Score: {tool_score_result['overall_score']} (1.0=correct, 0.0=incorrect)")
                    logger.info_rank0(f"=== END SCORING DETAILS ===\n")
                
                # Track metrics (only overall_score now)
                dataset_name = f"{data_source}_tool"
                tool_metrics[dataset_name]['scores'].append(tool_score_result['overall_score'])
                tool_metrics[dataset_name]['total_turns'].append(result.total_turns)
                tool_metrics[dataset_name]['total_tool_calls'].append(len(result.tool_calls))
                tool_metrics[dataset_name]['interaction_success'].append(1.0 if result.success else 0.0)
                # Initialize total_samples if not exists, then increment
                if 'total_samples' not in tool_metrics[dataset_name]:
                    tool_metrics[dataset_name]['total_samples'] = 0
                tool_metrics[dataset_name]['total_samples'] += 1
                
                # Skip detailed wandb logging for individual samples (removed per user request)
        
        except Exception as e:
            logger.info_rank0(f"Error in validation batch {batch_idx}: {e}")
            continue
        finally:
            # Clear batch from GPU memory
            torch.cuda.empty_cache()
    
        total_batches += 1
        
        if use_tqdm:
            validation_tqdm.set_postfix_str(f"Processed {total_batches} batches")
    
    if use_tqdm:
        validation_tqdm.close()
    
    # Aggregate metrics across GPUs
    all_tool_metrics = all_gather_defaultdict_v1(tool_metrics)
    
    # Clear accumulated data structures
    for dataset_name in tool_metrics:
        for metric_name in tool_metrics[dataset_name]:
            if isinstance(tool_metrics[dataset_name][metric_name], list):
                tool_metrics[dataset_name][metric_name].clear()
    
    del tool_metrics
    
    # Compute final metrics
    final_metrics = {}
    
    if args.train.global_rank == 0:
        # Process tool metrics (only overall_score now)
        for dataset_name, metrics in all_tool_metrics.items():
            if metrics['scores']:
                final_metrics[f'{dataset_name}_overall_score'] = np.mean(metrics['scores'])
                final_metrics[f'{dataset_name}_avg_turns'] = np.mean(metrics['total_turns'])
                final_metrics[f'{dataset_name}_avg_tool_calls'] = np.mean(metrics['total_tool_calls'])
                final_metrics[f'{dataset_name}_interaction_success_rate'] = np.mean(metrics['interaction_success'])
                final_metrics[f'{dataset_name}_total_samples'] = metrics['total_samples']
        # Add agent manager statistics
        agent_stats = agent_manager.get_interaction_statistics()
        for key, value in agent_stats.items():
            final_metrics[f'agent_{key}'] = value
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    log_memory_usage("After tool validation")
    logger.info_rank0(f"Completed tool validation at step {step}")
    
    return final_metrics


def main():
    """Main training function with tool validation."""
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    
    # Log tool validation settings
    if args.train.enable_tool_validation:
        logger.info_rank0("Tool validation enabled")
        logger.info_rank0(f"Supported tools: {args.train.supported_tools}")
        logger.info_rank0(f"Max agent turns: {args.train.max_agent_turns}")
        logger.info_rank0(f"Tool call timeout: {args.train.tool_call_timeout}s")
        logger.info_rank0(f"Agent total timeout: {args.train.agent_total_timeout}s")
    
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
    
    # Build data transforms
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

    # Build datasets and dataloaders
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            train_dataset = build_iterative_dataset(
                args.data.train_path,
                transform=transform,
                seed=args.train.seed,
            )
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        else:
            train_dataset = build_mapping_dataset(
                args.data.train_path,
                transform=transform,
            )
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
        
        # Validation dataset
        if args.data.val_path:
            val_dataset = build_mapping_dataset(
                args.data.val_path,
                transform=transform_val if args.data.data_type == "conversation" else transform,
            )
            from veomni.data.data_collator import DataCollatorWithPadding, MakeMicroBatchCollator
            from veomni.distributed.parallel_state import get_parallel_state
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data import DataLoader
            
            # Get parallel state for distributed sampling
            parallel_state = get_parallel_state()
            
            # Create collator for validation
            val_collate_fn = DataCollatorWithPadding()
            
            # Add micro-batch wrapper
            val_collate_fn = MakeMicroBatchCollator(
                num_micro_batch=1,  # Single micro-batch for validation
                internal_data_collator=val_collate_fn
            )
            
            # Create distributed sampler for validation
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=parallel_state.dp_size,
                rank=parallel_state.dp_rank,
                shuffle=False,
                drop_last=False,
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=32,  # Increased from 8 for better throughput
                sampler=val_sampler,
                num_workers=args.data.num_workers,
                collate_fn=val_collate_fn,
                pin_memory=args.data.pin_memory,
                drop_last=False,
                prefetch_factor=args.data.prefetch_factor,
            )
        else:
            val_dataloader = None
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Build model")
    # Determine torch dtype based on mixed precision setting
    if args.train.enable_mixed_precision:
        torch_dtype = "bfloat16"
    else:
        torch_dtype = "float32"
    
    # Prepare config kwargs for rope scaling if specified
    config_kwargs = {}
    if hasattr(args.model, 'rope_scaling') and args.model.rope_scaling:
        config_kwargs['rope_scaling'] = args.model.rope_scaling
        logger.info_rank0(f"Using RoPE scaling configuration: {args.model.rope_scaling}")
    
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        config_kwargs=config_kwargs,
    )
    model.to(f"cuda:{args.train.local_rank}")
    model.train()

    logger.info_rank0("Build optimizer and lr scheduler")
    optimizer = build_optimizer(
        model=model,
        lr=args.train.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.train.weight_decay,
        optimizer_type=args.train.optimizer,
    )
    
    # Add environment meter for periodic cache clearing
    environ_meter = helper.EnvironMeter(
        config=model.config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        train_steps=args.train.train_steps,
        lr=args.train.lr,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_min=args.train.lr_min,
        lr_start=args.train.lr_start,
    )

    if args.train.load_checkpoint_path:
        logger.info_rank0(f"Load checkpoint from {args.train.load_checkpoint_path}")
        checkpoint = Checkpointer.load_checkpoint(args.train.load_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_step = checkpoint["step"] + 1
        logger.info_rank0(f"Loaded checkpoint from step {start_step - 1}")
    else:
        start_step = 0

    logger.info_rank0("Apply model parallelism")
    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        basic_modules=args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        init_device=args.train.init_device,
    )

    # Initialize validation manager
    validation_manager = None
    tool_server = None
    vllm_config = None
    
    if args.data.val_path and args.train.validation_steps > 0:
        logger.info_rank0("Initialize validation manager")
        
        # vLLM configuration optimized for 32k context
        vllm_config = DictConfig({
            "tensor_model_parallel_size": 1,
            "max_num_batched_tokens": args.data.max_seq_len * 2,  # Conservative multiplier for 32k context
            "dtype": "bfloat16",
            "enforce_eager": False,
            "free_cache_engine": False,
            "gpu_memory_utilization": 0.6,  # Reduced to prevent OOM after validation
            "response_length": 4096,  # Increased for better reasoning
            "prompt_length": 128,
            "max_model_len": args.data.max_seq_len,
            "disable_log_stats": True,
            "enable_chunked_prefill": True,  # Enable for long context
            "enable_prefix_caching": True,   # Enable for efficiency
            "load_format": "dummy_hf",
        })
        
        # Initialize tool server for validation
        tool_server = WebToolServer()
        
        # Create tool validation manager
        validation_manager = ToolValidationVLLMManager(
            fsdp_model=model,
            model_path=args.model.model_path,
            tokenizer=tokenizer,
            model_hf_config=model.config,
            vllm_config=vllm_config,
            device_mesh=rollout_device_mesh,
            full_params=False,
            offload_param=True,
            load_format='dummy_hf',
            layered_summon=True,
            tool_server=tool_server,
            max_turns=args.train.max_agent_turns,
            tool_call_timeout=args.train.tool_call_timeout,
            enable_tool_validation=args.train.enable_tool_validation,
        )

    # Initialize wandb if needed
    if args.train.global_rank == 0 and args.train.use_wandb:
        import wandb
        wandb.init(
            project=args.train.wandb_project,
            name=args.train.wandb_name,
            config={**vars(args.model), **vars(args.data), **vars(args.train)},
        )

    logger.info_rank0("Start training")
    
    # Training loop
    step = start_step
    
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    
    # Use proper epoch-based training loop like the working script
    for epoch in range(args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        
        data_iterator = iter(train_dataloader)
        
        for batch_idx in range(args.train.train_steps):
            if args.train.max_steps and step >= args.train.max_steps:
                break
            
            try:
                micro_batches = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished")
                break
            
            total_loss = 0
            
            # Start timing for this step
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Process each micro-batch
            for micro_batch in micro_batches:
                # Add to environ meter for tracking
                environ_meter.add(micro_batch)
                
                # Move batch to device
                micro_batch = {k: v.to(f"cuda:{args.train.local_rank}") if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}
                
                # Forward pass
                with model_fwd_context:
                    outputs = model(**micro_batch)
                    loss = outputs.loss / len(micro_batches)  # Scale loss by number of micro-batches
                    total_loss += loss.item()
                
                # Backward pass
                with model_bwd_context:
                    loss.backward()
                
                # Clear intermediate tensors to prevent memory accumulation
                del outputs, loss
            
            # Gradient clipping
            if args.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Calculate timing and update environment meter
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            train_metrics = environ_meter.step(delta_time=delta_time, global_step=step)
            
            # Average loss across micro-batches
            avg_loss = total_loss
            
            # Clear micro_batches reference to prevent memory accumulation
            del micro_batches
        
            # Logging
            if step % 10 == 0:
                logger.info_rank0(f"Step {step}, Loss: {avg_loss:.6f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                
                if args.train.global_rank == 0 and args.train.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": step,
                    }, step=step)
            
            # Validation
            if (validation_manager is not None and 
                val_dataloader is not None and 
                step % args.train.validation_steps == 0):
                
                logger.info_rank0(f"Running validation at step {step}")
                
                # Clear GPU cache before validation
                torch.cuda.empty_cache()
                
                try:
                    with validation_manager.validation_context():
                        logger.info_rank0("Entered validation context - weights should be synced to vLLM")
                        
                        if args.train.enable_tool_validation:
                            # Run tool validation (synchronously inside validation context)
                            val_metrics = run_tool_validation(
                                validation_manager, val_dataloader, args, step
                            )
                        else:
                            # Run regular validation (implement if needed)
                            val_metrics = {}
                    
                    # Sleep to allow vLLM to release memory (from working script)
                    time.sleep(2)
                    
                    # Force model back to training state after validation
                    model.train()
                    logger.info_rank0(f"Restoring FSDP model to training state after validation")
                    
                    # Periodic full cleanup of validation manager every 100 validation steps
                    if step % (args.train.validation_steps * 100) == 0:
                        logger.info_rank0(f"Performing full validation manager cleanup at step {step}")
                        validation_manager.cleanup()
                        # Reinitialize validation manager
                        validation_manager = ToolValidationVLLMManager(
                            fsdp_model=model,
                            model_path=args.model.model_path,
                            tokenizer=tokenizer,
                            model_hf_config=model.config,
                            vllm_config=vllm_config,
                            device_mesh=rollout_device_mesh,
                            full_params=False,
                            offload_param=True,
                            load_format='dummy_hf',
                            layered_summon=True,
                            tool_server=tool_server,
                            max_turns=args.train.max_agent_turns,
                            tool_call_timeout=args.train.tool_call_timeout,
                            enable_tool_validation=args.train.enable_tool_validation,
                        )
                    
                    # Properly reload FSDP model to GPU (this is the key fix from working script!)
                    load_fsdp_model_to_gpu(model.cuda())
                    
                    # Clear GPU cache after validation
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Log validation metrics
                    if args.train.global_rank == 0 and val_metrics:
                        logger.info_rank0(f"Validation metrics at step {step}: {val_metrics}")
                        
                        if args.train.use_wandb:
                            import wandb
                            wandb_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                            wandb_metrics["train/step"] = step
                            wandb.log(wandb_metrics, step=step)
                except Exception as e:
                    logger.info_rank0(f"Validation failed at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Ensure model is back in training mode
                    model.train()
                    
                    # Try to reload FSDP model to GPU even on error
                    try:
                        load_fsdp_model_to_gpu(model.cuda())
                    except Exception as e:
                        logger.info_rank0(f"Failed to reload FSDP model: {e}")
                    
                    # Clear GPU cache after validation
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Checkpointing
            if step % args.train.save_steps == 0 and step > 0:
                logger.info_rank0(f"Saving checkpoint at step {step}")
                save_checkpoint_path = os.path.join(args.train.output_dir, f"global_step_{step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(save_checkpoint_path, state, global_steps=step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            
            step += 1
    
    logger.info_rank0("Training completed")
    
    # Final checkpoint
    if args.train.global_rank == 0:
        logger.info_rank0("Saving final checkpoint")
        save_checkpoint_path = os.path.join(args.train.output_dir, f"global_step_{step}")
        state = {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {
                "global_step": step,
                "lr_scheduler": lr_scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        Checkpointer.save(save_checkpoint_path, state, global_steps=step)
        dist.barrier()
        logger.info_rank0(f"Final checkpoint saved at {save_checkpoint_path} successfully!")
    
    # Cleanup
    cleanup_score_pool()
    
    if validation_manager:
        validation_manager.cleanup()
    
    # Proper distributed cleanup
    dist.barrier()
    dist.destroy_process_group()
    
    logger.info_rank0("Training finished")


if __name__ == "__main__":
    main()