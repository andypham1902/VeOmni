"""
Training script with tool-based validation for VeOmni.

This script extends the existing validation training script to support
tool-based agent validation during training.
"""

import json
import os
import re
import time
import asyncio
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
from veomni.utils.tool_validation_utils import ToolValidationVLLMManager
from veomni.utils.tool_servers import MultiToolServer, PythonToolServer, BashToolServer, SearchToolServer
from veomni.utils.agent_manager import AgentActorManager, AgentConfig
from veomni.utils.reward_score.tool_based import compute_tool_score

# Import existing validation components
from veomni.utils.reward_score.medical import compute_score as medical_compute_score
from veomni.utils.reward_score.ifeval import ifeval
from veomni.utils.reward_score.healthbench import healthbench

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


def compute_score(generation_result: Dict[str, Any], ground_truth: str, data_source: str) -> float:
    """Compute score for regular (non-tool) validation."""
    generation_result = extract_content(generation_result)
    if data_source in ['hoanganh/Medical-Train', 'TsinghuaC3I/MedXpertQA', 'hoanganh/MedQA-Test']:
        res = medical_compute_score(generation_result, ground_truth)
    elif data_source in ['google/IFEval']:
        res = ifeval.compute_score(generation_result, ground_truth)
    elif data_source in ['openai/HealthBench']:
        res = healthbench.compute_score(generation_result, ground_truth)
    else:
        raise NotImplementedError(f"Unsupported dataset: {data_source}. Please implement compute_score for this dataset.")
    return res


def log_memory_usage(prefix=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info_rank0(f"{prefix} Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


@dataclass
class Arguments:
    model: ModelArguments
    data: DataArguments
    train: TrainingArguments

    # Tool validation specific arguments
    enable_tool_validation: bool = field(default=False, metadata={"help": "Enable tool-based validation"})
    tool_validation_ratio: float = field(default=0.3, metadata={"help": "Ratio of samples to use for tool validation"})
    tool_server_url: Optional[str] = field(default=None, metadata={"help": "URL for external tool server"})
    max_agent_turns: int = field(default=10, metadata={"help": "Maximum turns for agent interaction"})
    tool_call_timeout: int = field(default=30, metadata={"help": "Timeout for tool calls in seconds"})
    agent_total_timeout: int = field(default=300, metadata={"help": "Total timeout for agent interaction"})
    supported_tools: List[str] = field(default_factory=lambda: ["python", "bash", "search"], metadata={"help": "List of supported tools"})


async def run_tool_validation(validation_manager: ToolValidationVLLMManager, 
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
    
    # Create tool server
    if args.tool_server_url:
        from veomni.utils.tool_validation_utils import HTTPToolServer
        tool_server = HTTPToolServer(args.tool_server_url)
    else:
        # Use local tool servers
        tool_servers = {}
        if "python" in args.supported_tools:
            tool_servers["python"] = PythonToolServer(timeout=args.tool_call_timeout)
        if "bash" in args.supported_tools:
            tool_servers["bash"] = BashToolServer(timeout=args.tool_call_timeout)
        if "search" in args.supported_tools:
            tool_servers["search"] = SearchToolServer(timeout=args.tool_call_timeout)
        
        tool_server = MultiToolServer(tool_servers)
    
    # Create agent manager
    agent_config = AgentConfig(
        max_turns=args.max_agent_turns,
        tool_call_timeout=args.tool_call_timeout,
        total_timeout=args.agent_total_timeout,
        enable_tool_validation=args.enable_tool_validation,
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
    
    # Validation loop
    if args.train.global_rank == 0:
        validation_tqdm = trange(len(val_dataloader), desc="Tool Validation", leave=False)
    else:
        validation_tqdm = range(len(val_dataloader))
    
    for batch_idx, batch in enumerate(val_dataloader):
        if isinstance(validation_tqdm, trange):
            validation_tqdm.set_description(f"Processing batch {batch_idx + 1}/{len(val_dataloader)}")
        
        # Move batch to device
        input_ids = batch["input_ids"].to(get_torch_device())
        attention_mask = batch["attention_mask"].to(get_torch_device())
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(get_torch_device())
        
        # Get batch metadata
        batch_ground_truths = batch.get("ground_truth", [])
        batch_data_sources = batch.get("data_source", [])
        batch_task_types = batch.get("task_type", ["general"] * input_ids.size(0))
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": validation_manager.tokenizer.eos_token_id,
            "pad_token_id": validation_manager.tokenizer.pad_token_id,
        }
        
        # Determine which samples to use for tool validation
        batch_size = input_ids.size(0)
        tool_indices = []
        regular_indices = []
        
        for i in range(batch_size):
            if i < batch_size * args.tool_validation_ratio:
                tool_indices.append(i)
            else:
                regular_indices.append(i)
        
        # Process tool validation samples
        if tool_indices:
            tool_input_ids = input_ids[tool_indices]
            tool_attention_mask = attention_mask[tool_indices]
            tool_position_ids = position_ids[tool_indices] if position_ids is not None else None
            
            # Run agent interactions
            agent_results = await agent_manager.run_batch_interactions(
                tool_input_ids,
                tool_attention_mask,
                tool_position_ids,
                **generation_kwargs
            )
            
            # Evaluate results
            for i, result in enumerate(agent_results):
                idx = tool_indices[i]
                ground_truth = batch_ground_truths[idx] if idx < len(batch_ground_truths) else ""
                data_source = batch_data_sources[idx] if idx < len(batch_data_sources) else "unknown"
                task_type = batch_task_types[idx] if idx < len(batch_task_types) else "general"
                
                # Compute tool-based score
                tool_score_result = compute_tool_score(
                    result, ground_truth, task_type
                )
                
                # Track metrics
                dataset_name = f"{data_source}_tool"
                tool_metrics[dataset_name]['scores'].append(tool_score_result['overall_score'])
                tool_metrics[dataset_name]['task_completion_scores'].append(tool_score_result['task_completion_score'])
                tool_metrics[dataset_name]['tool_usage_efficiency'].append(tool_score_result['tool_usage_efficiency'])
                tool_metrics[dataset_name]['reasoning_quality'].append(tool_score_result['reasoning_quality'])
                tool_metrics[dataset_name]['final_answer_correctness'].append(tool_score_result['final_answer_correctness'])
                tool_metrics[dataset_name]['total_turns'].append(result.total_turns)
                tool_metrics[dataset_name]['total_tool_calls'].append(len(result.tool_calls))
                tool_metrics[dataset_name]['interaction_success'].append(1.0 if result.success else 0.0)
                tool_metrics[dataset_name]['total_samples'] += 1
        
        # Process regular validation samples
        if regular_indices:
            regular_input_ids = input_ids[regular_indices]
            regular_attention_mask = attention_mask[regular_indices]
            regular_position_ids = position_ids[regular_indices] if position_ids is not None else None
            
            # Generate regular responses
            regular_responses = validation_manager.generate_responses(
                regular_input_ids,
                regular_attention_mask,
                regular_position_ids,
                **generation_kwargs
            )
            
            # Evaluate regular responses
            for i, response in enumerate(regular_responses):
                idx = regular_indices[i]
                ground_truth = batch_ground_truths[idx] if idx < len(batch_ground_truths) else ""
                data_source = batch_data_sources[idx] if idx < len(batch_data_sources) else "unknown"
                
                # Compute regular score
                try:
                    score = compute_score(response["response"], ground_truth, data_source)
                except Exception as e:
                    logger.warning(f"Error computing score for {data_source}: {e}")
                    score = 0.0
                
                # Track metrics
                dataset_name = f"{data_source}_regular"
                regular_metrics[dataset_name]['scores'].append(score)
                regular_metrics[dataset_name]['prompt_lengths'].append(response["prompt_length"])
                regular_metrics[dataset_name]['response_lengths'].append(response["response_length"])
                regular_metrics[dataset_name]['total_samples'] += 1
        
        total_batches += 1
        
        if isinstance(validation_tqdm, trange):
            validation_tqdm.set_postfix_str(f"Processed {total_batches} batches")
    
    if isinstance(validation_tqdm, trange):
        validation_tqdm.close()
    
    # Aggregate metrics across GPUs
    all_tool_metrics = all_gather_defaultdict_v1(tool_metrics)
    all_regular_metrics = all_gather_defaultdict_v1(regular_metrics)
    
    # Clear accumulated data structures
    for dataset_name in tool_metrics:
        for metric_name in tool_metrics[dataset_name]:
            if isinstance(tool_metrics[dataset_name][metric_name], list):
                tool_metrics[dataset_name][metric_name].clear()
    for dataset_name in regular_metrics:
        for metric_name in regular_metrics[dataset_name]:
            if isinstance(regular_metrics[dataset_name][metric_name], list):
                regular_metrics[dataset_name][metric_name].clear()
    
    del tool_metrics, regular_metrics
    
    # Compute final metrics
    final_metrics = {}
    
    if args.train.global_rank == 0:
        # Process tool metrics
        for dataset_name, metrics in all_tool_metrics.items():
            if metrics['scores']:
                final_metrics[f'{dataset_name}_overall_score'] = np.mean(metrics['scores'])
                final_metrics[f'{dataset_name}_task_completion'] = np.mean(metrics['task_completion_scores'])
                final_metrics[f'{dataset_name}_tool_efficiency'] = np.mean(metrics['tool_usage_efficiency'])
                final_metrics[f'{dataset_name}_reasoning_quality'] = np.mean(metrics['reasoning_quality'])
                final_metrics[f'{dataset_name}_answer_correctness'] = np.mean(metrics['final_answer_correctness'])
                final_metrics[f'{dataset_name}_avg_turns'] = np.mean(metrics['total_turns'])
                final_metrics[f'{dataset_name}_avg_tool_calls'] = np.mean(metrics['total_tool_calls'])
                final_metrics[f'{dataset_name}_interaction_success_rate'] = np.mean(metrics['interaction_success'])
                final_metrics[f'{dataset_name}_total_samples'] = metrics['total_samples']
        
        # Process regular metrics
        for dataset_name, metrics in all_regular_metrics.items():
            if metrics['scores']:
                final_metrics[f'{dataset_name}_accuracy'] = np.mean(metrics['scores'])
                final_metrics[f'{dataset_name}_avg_prompt_length'] = np.mean(metrics['prompt_lengths'])
                final_metrics[f'{dataset_name}_avg_response_length'] = np.mean(metrics['response_lengths'])
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
    if args.enable_tool_validation:
        logger.info_rank0(f"Tool validation enabled with ratio: {args.tool_validation_ratio}")
        logger.info_rank0(f"Supported tools: {args.supported_tools}")
        logger.info_rank0(f"Max agent turns: {args.max_agent_turns}")
        logger.info_rank0(f"Tool call timeout: {args.tool_call_timeout}s")
        logger.info_rank0(f"Agent total timeout: {args.agent_total_timeout}s")
    
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
                args.data.train_paths,
                transform=transform,
                batch_size=args.train.train_batch_size,
                seed=args.train.seed,
                num_workers=args.data.num_workers,
                distributed=True,
                infinite=True,
            )
        else:
            train_dataset = build_mapping_dataset(
                args.data.train_paths,
                transform=transform,
                distributed=True,
                seed=args.train.seed,
                shuffle=True,
                num_workers=args.data.num_workers,
                infinite=True,
            )
        
        train_dataloader = build_dataloader(
            train_dataset,
            batch_size=args.train.train_batch_size,
            num_workers=args.data.num_workers,
            distributed=True,
            shuffle=True,
            infinite=True,
        )
        
        # Validation dataset
        if args.data.val_path:
            val_dataset = build_mapping_dataset(
                [args.data.val_path],
                transform=transform_val if args.data.data_type == "conversation" else transform,
                distributed=True,
                seed=args.train.seed,
                shuffle=False,
                num_workers=args.data.num_workers,
                infinite=False,
            )
            val_dataloader = build_dataloader(
                val_dataset,
                batch_size=args.train.eval_batch_size,
                num_workers=args.data.num_workers,
                distributed=True,
                shuffle=False,
                infinite=False,
            )
        else:
            val_dataloader = None
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Build model")
    model = build_foundation_model(args.model)
    model.to(get_torch_device())
    model.train()

    logger.info_rank0("Build optimizer and lr scheduler")
    optimizer = build_optimizer(model, args.train)
    lr_scheduler = build_lr_scheduler(optimizer, args.train)

    if args.train.checkpoint_path:
        logger.info_rank0(f"Load checkpoint from {args.train.checkpoint_path}")
        checkpoint = Checkpointer.load_checkpoint(args.train.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_step = checkpoint["step"] + 1
        logger.info_rank0(f"Loaded checkpoint from step {start_step - 1}")
    else:
        start_step = 0

    logger.info_rank0("Apply model parallelism")
    model = build_parallelize_model(model, args.train)

    # Initialize validation manager
    validation_manager = None
    if args.data.val_path and args.train.validation_steps > 0:
        logger.info_rank0("Initialize validation manager")
        
        # vLLM configuration
        vllm_config = DictConfig({
            "model": args.model.model_name,
            "max_model_len": args.data.max_seq_len,
            "tensor_parallel_size": args.train.tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "disable_log_stats": True,
            "disable_log_requests": True,
            "gpu_memory_utilization": 0.8,
        })
        
        # Create tool validation manager
        validation_manager = ToolValidationVLLMManager(
            fsdp_model=model,
            model_path=args.model.model_name,
            tokenizer=tokenizer,
            model_hf_config=model.config,
            vllm_config=vllm_config,
            device_mesh=rollout_device_mesh,
            enable_tool_validation=args.enable_tool_validation,
            max_turns=args.max_agent_turns,
            tool_call_timeout=args.tool_call_timeout,
        )

    logger.info_rank0("Start training")
    
    # Training loop
    step = start_step
    
    with build_activation_offloading_context(args.train.activation_offloading):
        for batch in train_dataloader:
            if step >= args.train.max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(get_torch_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.train.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.gradient_clipping)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if step % args.train.log_interval == 0:
                logger.info_rank0(f"Step {step}, Loss: {loss.item():.6f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                
                if args.train.global_rank == 0 and args.train.enable_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": step,
                    }, step=step)
            
            # Validation
            if (validation_manager is not None and 
                val_dataloader is not None and 
                step % args.train.validation_steps == 0 and 
                step > 0):
                
                logger.info_rank0(f"Running validation at step {step}")
                model.eval()
                
                with validation_manager.validation_context():
                    if args.enable_tool_validation:
                        # Run tool validation
                        val_metrics = asyncio.run(run_tool_validation(
                            validation_manager, val_dataloader, args, step
                        ))
                    else:
                        # Run regular validation (implement if needed)
                        val_metrics = {}
                
                # Log validation metrics
                if args.train.global_rank == 0 and val_metrics:
                    logger.info_rank0(f"Validation metrics at step {step}: {val_metrics}")
                    
                    if args.train.enable_wandb:
                        wandb_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                        wandb_metrics["train/step"] = step
                        wandb.log(wandb_metrics, step=step)
                
                model.train()
            
            # Checkpointing
            if step % args.train.checkpoint_interval == 0 and step > 0:
                logger.info_rank0(f"Saving checkpoint at step {step}")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": step,
                }
                Checkpointer.save_checkpoint(checkpoint, step)
            
            step += 1
    
    logger.info_rank0("Training completed")
    
    # Final checkpoint
    if args.train.global_rank == 0:
        logger.info_rank0("Saving final checkpoint")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": step,
        }
        Checkpointer.save_checkpoint(checkpoint, step)
    
    # Cleanup
    cleanup_score_pool()
    
    if validation_manager:
        validation_manager.cleanup()
    
    logger.info_rank0("Training finished")


if __name__ == "__main__":
    main()