import os
import json
import asyncio
import tempfile
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from omegaconf import DictConfig
import logging
import time
import gc
import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod

from veomni.utils.validation_utils import ValidationVLLMManager
from verl.protocol import DataProto
from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result of a tool call execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""


@dataclass
class AgentInteractionResult:
    """Result of a complete agent interaction loop."""
    final_response: str
    tool_calls: List[ToolCallResult]
    total_turns: int
    success: bool
    total_time: float
    reasoning_trace: List[str]


class ToolServer(ABC):
    """Abstract base class for tool servers."""
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute a tool action and return the result."""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        pass


class HTTPToolServer(ToolServer):
    """HTTP-based tool server client."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute a tool via HTTP request."""
        start_time = time.time()
        
        try:
            payload = {
                "tool_name": tool_name,
                "action": action,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.base_url}/execute",
                json=payload,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return ToolCallResult(
                    success=True,
                    output=result.get("output", ""),
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            else:
                return ToolCallResult(
                    success=False,
                    output="",
                    error=f"HTTP {response.status_code}: {response.text}",
                    tool_name=tool_name,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolCallResult(
                success=False,
                output="",
                error=str(e),
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    def get_available_tools(self) -> List[str]:
        """Get available tools from the server."""
        try:
            response = self.session.get(f"{self.base_url}/tools", timeout=self.timeout)
            if response.status_code == 200:
                return response.json().get("tools", [])
            else:
                logger.error(f"Failed to get tools: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []


class LocalToolServer(ToolServer):
    """Local tool server for testing and development."""
    
    def __init__(self):
        self.tools = {
            "python": self._execute_python,
            "bash": self._execute_bash,
            "search": self._execute_search,
        }
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute a tool locally."""
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolCallResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not available",
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )
        
        try:
            output = await self.tools[tool_name](action, **kwargs)
            return ToolCallResult(
                success=True,
                output=output,
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolCallResult(
                success=False,
                output="",
                error=str(e),
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )
    
    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return list(self.tools.keys())
    
    async def _execute_python(self, code: str, **kwargs) -> str:
        """Execute Python code safely."""
        # Simple mock implementation
        return f"Executed Python code: {code[:100]}..."
    
    async def _execute_bash(self, command: str, **kwargs) -> str:
        """Execute bash command safely."""
        # Simple mock implementation
        return f"Executed bash command: {command[:100]}..."
    
    async def _execute_search(self, query: str, **kwargs) -> str:
        """Execute search query."""
        # Simple mock implementation
        return f"Search results for: {query[:100]}..."


class ToolValidationVLLMManager(ValidationVLLMManager):
    """
    Extended ValidationVLLMManager that supports tool-based agent validation.
    
    This manager extends the base ValidationVLLMManager to support multi-turn
    agent interactions with tools during validation.
    """
    
    def __init__(self, 
                 tool_server: Optional[ToolServer] = None,
                 max_turns: int = 10,
                 tool_call_timeout: int = 30,
                 enable_tool_validation: bool = True,
                 tool_call_template: str = "default",
                 **kwargs):
        """
        Initialize the tool validation manager.
        
        Args:
            tool_server: Tool server instance for executing tools
            max_turns: Maximum number of agent turns per interaction
            tool_call_timeout: Timeout for individual tool calls
            enable_tool_validation: Whether to enable tool validation
            tool_call_template: Template for parsing tool calls
            **kwargs: Arguments passed to parent ValidationVLLMManager
        """
        super().__init__(**kwargs)
        
        self.tool_server = tool_server or LocalToolServer()
        self.max_turns = max_turns
        self.tool_call_timeout = tool_call_timeout
        self.enable_tool_validation = enable_tool_validation
        self.tool_call_template = tool_call_template
        
        # Tool call parsing patterns
        self.tool_patterns = {
            "default": r"<tool_call>\s*(\w+)\s*</tool_call>\s*<tool_input>\s*(.*?)\s*</tool_input>",
            "function": r"```(\w+)\s*(.*?)```",
            "json": r"```json\s*(\{.*?\})\s*```"
        }
        
        logger.info(f"ToolValidationVLLMManager initialized with {len(self.tool_server.get_available_tools())} tools")
    
    def parse_tool_calls(self, response: str) -> List[Tuple[str, str]]:
        """
        Parse tool calls from model response.
        
        Args:
            response: Model response text
            
        Returns:
            List of (tool_name, tool_input) tuples
        """
        import re
        
        pattern = self.tool_patterns.get(self.tool_call_template, self.tool_patterns["default"])
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        tool_calls = []
        for match in matches:
            if len(match) == 2:
                tool_name, tool_input = match
                tool_calls.append((tool_name.strip(), tool_input.strip()))
        
        return tool_calls
    
    async def execute_agent_loop(self, 
                                 input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 position_ids: Optional[torch.Tensor] = None,
                                 **generation_kwargs) -> AgentInteractionResult:
        """
        Execute a complete agent interaction loop with tools.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            **generation_kwargs: Generation parameters
            
        Returns:
            AgentInteractionResult with complete interaction history
        """
        if not self.enable_tool_validation:
            # Fallback to regular generation
            responses = self.generate_responses(
                input_ids, attention_mask, position_ids, **generation_kwargs
            )
            return AgentInteractionResult(
                final_response=responses[0]["response"] if responses else "",
                tool_calls=[],
                total_turns=1,
                success=True,
                total_time=0.0,
                reasoning_trace=[]
            )
        
        start_time = time.time()
        tool_calls = []
        reasoning_trace = []
        
        # Current conversation state
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        if position_ids is not None:
            current_position_ids = position_ids.clone()
        else:
            current_position_ids = None
        
        turn = 0
        while turn < self.max_turns:
            turn += 1
            
            # Generate response
            responses = self.generate_responses(
                current_input_ids, 
                current_attention_mask, 
                current_position_ids, 
                **generation_kwargs
            )
            
            if not responses:
                break
                
            response_text = responses[0]["response"]
            reasoning_trace.append(f"Turn {turn}: {response_text}")
            
            # Parse tool calls
            parsed_tool_calls = self.parse_tool_calls(response_text)
            
            if not parsed_tool_calls:
                # No tool calls, conversation is complete
                return AgentInteractionResult(
                    final_response=response_text,
                    tool_calls=tool_calls,
                    total_turns=turn,
                    success=True,
                    total_time=time.time() - start_time,
                    reasoning_trace=reasoning_trace
                )
            
            # Execute tool calls
            tool_results = []
            for tool_name, tool_input in parsed_tool_calls:
                try:
                    result = await asyncio.wait_for(
                        self.tool_server.execute_tool(tool_name, tool_input),
                        timeout=self.tool_call_timeout
                    )
                    tool_calls.append(result)
                    tool_results.append(result)
                except asyncio.TimeoutError:
                    result = ToolCallResult(
                        success=False,
                        output="",
                        error="Tool call timeout",
                        tool_name=tool_name,
                        execution_time=self.tool_call_timeout
                    )
                    tool_calls.append(result)
                    tool_results.append(result)
            
            # Prepare next turn input with tool results
            tool_results_text = self._format_tool_results(tool_results)
            next_input = response_text + "\n\n" + tool_results_text + "\n\nPlease continue:"
            
            # Tokenize next input
            next_tokens = self.tokenizer.encode(next_input, return_tensors="pt")
            
            # Update conversation state
            current_input_ids = next_tokens.to(input_ids.device)
            current_attention_mask = torch.ones_like(current_input_ids)
            if position_ids is not None:
                current_position_ids = torch.arange(
                    current_input_ids.size(1), 
                    device=current_input_ids.device
                ).unsqueeze(0)
        
        # Max turns reached
        return AgentInteractionResult(
            final_response=reasoning_trace[-1] if reasoning_trace else "",
            tool_calls=tool_calls,
            total_turns=turn,
            success=False,  # Max turns reached
            total_time=time.time() - start_time,
            reasoning_trace=reasoning_trace
        )
    
    def _format_tool_results(self, tool_results: List[ToolCallResult]) -> str:
        """Format tool results for the next turn."""
        formatted_results = []
        for result in tool_results:
            if result.success:
                formatted_results.append(f"Tool {result.tool_name} output: {result.output}")
            else:
                formatted_results.append(f"Tool {result.tool_name} error: {result.error}")
        return "\n".join(formatted_results)
    
    async def generate_agent_responses(self, 
                                       input_ids: torch.Tensor,
                                       attention_mask: torch.Tensor,
                                       position_ids: Optional[torch.Tensor] = None,
                                       **generation_kwargs) -> List[AgentInteractionResult]:
        """
        Generate agent responses for a batch of inputs.
        
        Args:
            input_ids: Batch of input token IDs
            attention_mask: Batch of attention masks
            position_ids: Batch of position IDs
            **generation_kwargs: Generation parameters
            
        Returns:
            List of AgentInteractionResult objects
        """
        batch_size = input_ids.size(0)
        results = []
        
        for i in range(batch_size):
            single_input_ids = input_ids[i:i+1]
            single_attention_mask = attention_mask[i:i+1]
            single_position_ids = position_ids[i:i+1] if position_ids is not None else None
            
            result = await self.execute_agent_loop(
                single_input_ids,
                single_attention_mask,
                single_position_ids,
                **generation_kwargs
            )
            results.append(result)
        
        return results
    
    def get_tool_validation_metrics(self, results: List[AgentInteractionResult]) -> Dict[str, float]:
        """
        Compute tool validation metrics from agent results.
        
        Args:
            results: List of agent interaction results
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {}
        
        total_interactions = len(results)
        successful_interactions = sum(1 for r in results if r.success)
        total_tool_calls = sum(len(r.tool_calls) for r in results)
        successful_tool_calls = sum(
            sum(1 for tc in r.tool_calls if tc.success) for r in results
        )
        
        avg_turns = np.mean([r.total_turns for r in results])
        avg_time = np.mean([r.total_time for r in results])
        
        # Tool usage statistics
        tool_usage = {}
        for result in results:
            for tool_call in result.tool_calls:
                tool_name = tool_call.tool_name
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {"total": 0, "successful": 0}
                tool_usage[tool_name]["total"] += 1
                if tool_call.success:
                    tool_usage[tool_name]["successful"] += 1
        
        metrics = {
            "interaction_success_rate": successful_interactions / total_interactions,
            "tool_call_success_rate": successful_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0,
            "avg_turns_per_interaction": avg_turns,
            "avg_time_per_interaction": avg_time,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls_per_interaction": total_tool_calls / total_interactions,
        }
        
        # Add per-tool metrics
        for tool_name, stats in tool_usage.items():
            metrics[f"tool_{tool_name}_success_rate"] = stats["successful"] / stats["total"]
            metrics[f"tool_{tool_name}_usage_count"] = stats["total"]
        
        return metrics