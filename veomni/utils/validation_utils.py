import os
import tempfile
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from omegaconf import DictConfig
import logging
import time
import gc

from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from verl.protocol import DataProto
from verl.utils.device import get_device_name, get_torch_device
from tensordict import TensorDict

logger = logging.getLogger(__name__)

class ValidationVLLMManager:
    """
    Manager class to handle switching between FSDP model training and vLLM inference
    for fast validation during training using FSDPVLLMShardingManager.
    """
    
    def __init__(self, 
                 fsdp_model,
                 model_path: str,
                 tokenizer, 
                 model_hf_config,
                 vllm_config: Dict[str, Any],
                 device_mesh=None,
                 offload_param: bool = False,
                 load_format: str = 'dummy_hf',
                 layered_summon: bool = True,
                 full_params: bool = False,
                 **kwargs):
        """
        Initialize the validation manager.
        
        Args:
            fsdp_model: FSDP wrapped model used for training
            model_path: Path to the model for vLLM
            tokenizer: Model tokenizer
            model_hf_config: HuggingFace model config
            vllm_config: vLLM configuration dictionary
            device_mesh: Device mesh for distributed training (optional)
            offload_param: Whether to offload parameters (default: False)
            load_format: Load format for vLLM (default: 'dummy_hf')
            layered_summon: Whether to use layered summon for LoRA (default: True)
            full_params: Whether to use full parameters (default: False)
            **kwargs: Additional arguments for vLLMRollout
        """
        self.fsdp_model = fsdp_model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        
        # Convert dict to DictConfig if needed
        if isinstance(vllm_config, dict):
            self.vllm_config = DictConfig(vllm_config)
        else:
            self.vllm_config = vllm_config
            
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon
        self.full_params = full_params
        self.kwargs = kwargs
        
        # Initialize vLLM rollout and sharding manager
        self.vllm_rollout = None
        self.sharding_manager = None
        self._vllm_initialized = False
        self._generation_count = 0
        
    def _initialize_vllm(self):
        """Initialize vLLM rollout and sharding manager lazily."""
        if self._vllm_initialized:
            return
            
        try:
            # Filter out parameters that should only go to sharding manager
            vllm_kwargs = {k: v for k, v in self.kwargs.items() 
                          if k not in ['load_format', 'layered_summon', 'full_params', 'offload_param']}
            
            # Initialize vLLM rollout with correct parameters
            self.vllm_rollout = vLLMRollout(
                model_path=self.model_path,
                config=self.vllm_config,
                tokenizer=self.tokenizer,
                model_hf_config=self.model_hf_config,
                **vllm_kwargs
            )
            
            # Initialize FSDP-vLLM sharding manager
            self.sharding_manager = FSDPVLLMShardingManager(
                module=self.fsdp_model,
                inference_engine=self.vllm_rollout.inference_engine,
                model_config=self.model_hf_config,
                full_params=self.full_params,
                device_mesh=self.device_mesh,
                offload_param=self.offload_param,
                load_format=self.load_format,
                layered_summon=self.layered_summon
            )
            
            self._vllm_initialized = True
            logger.info("vLLM rollout and sharding manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM rollout and sharding manager: {e}")
            raise
    
    @contextmanager
    def validation_context(self):
        """
        Context manager for validation that uses FSDPVLLMShardingManager:
        1. Initialize vLLM rollout and sharding manager
        2. Enter sharding manager context (handles model switching)
        3. Yield for validation
        4. Exit sharding manager context (restores training state)
        """
        try:
            # Initialize vLLM and sharding manager
            self._initialize_vllm()
            
            # Use sharding manager context for model switching
            with self.sharding_manager:
                yield self
            
            # Periodic cleanup every 10 validations to prevent memory accumulation
            self._generation_count += 1
            if self._generation_count % 10 == 0:
                logger.info(f"Performing periodic vLLM cleanup after {self._generation_count} validations")
                self._periodic_cleanup()
                
        except Exception as e:
            logger.error(f"Error in validation context: {e}")
            raise
    
    def generate_responses(self, 
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          position_ids: Optional[torch.Tensor] = None,
                          **generation_kwargs) -> List[str]:
        """
        Generate responses using vLLM rollout and return decoded strings.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated response strings
        """
        # start = time.time()
        if not self._vllm_initialized:
            raise RuntimeError("vLLM is not initialized. Use validation_context() first.")
        
        # Create DataProto input for vLLM rollout
        batch_size = input_ids.size(0)

        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare batch data
        batch = TensorDict({
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "position_ids": position_ids.contiguous()
        }, batch_size=batch_size)
        
        # Prepare meta info
        meta_info = {
            "eos_token_id": generation_kwargs["eos_token_id"],
        }
        
        # Create DataProto
        prompts = DataProto(
            batch=batch,
            non_tensor_batch={},
            meta_info=meta_info
        )
        # end = time.time() - start
        # logger.info(f"Prepared prompts in {end:.4f} seconds")
        
        # start = time.time()
        # Generate using vLLM rollout
        results = self.vllm_rollout.generate_sequences(prompts, **generation_kwargs)
        # end = time.time() - start
        # logger.info(f"vLLM generated responses in {end:.4f} seconds")

        # start = time.time()
        # Extract and decode responses
        responses = results.batch["responses"]  # Shape: (batch_size, response_length)
        decoded_responses = []
        for i in range(responses.size(0)):
            response_ids = responses[i]
            # Remove padding tokens
            if self.vllm_rollout.pad_token_id is not None:
                response_ids = response_ids[response_ids != self.vllm_rollout.pad_token_id]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            decoded_responses.append({
                "response": response_text,
                "prompt_length": attention_mask[i].sum().item(),
                "response_length": response_ids.size(0)
            })
        # clear kv cache
        get_torch_device().empty_cache()
        # end = time.time() - start
        # logger.info(f"Decoded responses in {end:.4f} seconds")
        return decoded_responses
    
    def _periodic_cleanup(self):
        """Periodically reinitialize vLLM to prevent memory accumulation"""
        try:
            # Cleanup existing vLLM resources
            if self.vllm_rollout and hasattr(self.vllm_rollout, 'inference_engine'):
                if hasattr(self.vllm_rollout.inference_engine, 'cleanup'):
                    self.vllm_rollout.inference_engine.cleanup()
            
            # Reset initialization flag
            self._vllm_initialized = False
            self.vllm_rollout = None
            self.sharding_manager = None
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Periodic vLLM cleanup completed")
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.vllm_rollout and hasattr(self.vllm_rollout, 'inference_engine'):
            # Clean up vLLM inference engine if it has cleanup method
            if hasattr(self.vllm_rollout.inference_engine, 'cleanup'):
                self.vllm_rollout.inference_engine.cleanup()
        if self.sharding_manager:
            # Sharding manager cleanup is handled by its context manager
            pass
        
        # Reset state
        self._vllm_initialized = False
        self.vllm_rollout = None
        self.sharding_manager = None
        self._generation_count = 0
