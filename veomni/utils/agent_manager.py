"""
Agent manager for tool-based validation.

This module provides the agent management functionality for executing
multi-turn tool interactions during validation, inspired by the verl-tool
AgentActorManager implementation.
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np
from omegaconf import DictConfig

from veomni.utils.tool_validation_utils import (
    ToolServer, ToolCallResult, AgentInteractionResult, ToolValidationVLLMManager
)
from veomni.utils.tool_servers import MultiToolServer

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of agent interaction."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_TURNS = "max_turns"


@dataclass
class AgentConfig:
    """Configuration for agent execution."""
    max_turns: int = 10
    tool_call_timeout: int = 30
    total_timeout: int = 300  # 5 minutes
    enable_tool_validation: bool = True
    tool_call_template: str = "default"
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Tool execution parameters
    max_tool_calls_per_turn: int = 3
    allow_parallel_tool_calls: bool = False
    
    # Parsing parameters
    stop_sequences: List[str] = field(default_factory=lambda: ["<|end|>", "<|stop|>"])
    tool_call_patterns: Dict[str, str] = field(default_factory=lambda: {
        "default": r"<tool_call>\s*(\w+)\s*</tool_call>\s*<tool_input>\s*(.*?)\s*</tool_input>",
        "function": r"```(\w+)\s*(.*?)```",
        "json": r"```json\s*(\{.*?\})\s*```"
    })


@dataclass
class AgentTurn:
    """Represents a single turn in agent interaction."""
    turn_number: int
    prompt: str
    response: str
    tool_calls: List[ToolCallResult]
    status: AgentStatus
    execution_time: float
    tokens_used: int = 0


class AgentActorManager:
    """
    Agent actor manager for executing tool-based validation.
    
    This class manages the multi-turn interaction loop between the agent
    and tool servers, handling tool calls, responses, and state management.
    """
    
    def __init__(self, 
                 validation_manager: ToolValidationVLLMManager,
                 tool_server: Optional[ToolServer] = None,
                 config: Optional[AgentConfig] = None):
        """
        Initialize the agent actor manager.
        
        Args:
            validation_manager: ToolValidationVLLMManager instance
            tool_server: Tool server for executing tools
            config: Agent configuration
        """
        self.validation_manager = validation_manager
        self.tool_server = tool_server or MultiToolServer()
        self.config = config or AgentConfig()
        
        # State tracking
        self.current_interactions: Dict[str, List[AgentTurn]] = {}
        self.interaction_stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
            "timeout_interactions": 0,
            "total_turns": 0,
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
        }
        
        logger.info(f"AgentActorManager initialized with {len(self.tool_server.get_available_tools())} tools")
    
    async def run_agent_interaction(self, 
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   position_ids: Optional[torch.Tensor] = None,
                                   interaction_id: Optional[str] = None,
                                   **generation_kwargs) -> AgentInteractionResult:
        """
        Run a complete agent interaction with tools.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            interaction_id: Unique identifier for this interaction
            **generation_kwargs: Additional generation parameters
            
        Returns:
            AgentInteractionResult with complete interaction history
        """
        if interaction_id is None:
            interaction_id = f"interaction_{int(time.time() * 1000)}"
        
        import json  # Move import to top of function
        
        start_time = time.time()
        turns = []
        
        # Initialize interaction tracking
        self.current_interactions[interaction_id] = []
        self.interaction_stats["total_interactions"] += 1
        
        try:
            # Prepare generation parameters
            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample,
                **generation_kwargs
            }
            
            # Decode initial prompt to extract messages
            initial_text = self.validation_manager.tokenizer.decode(
                input_ids[0], skip_special_tokens=False
            )
            
            # Initialize messages list (similar to simple_llm_agent_test.py)
            messages = self._extract_messages_from_prompt(initial_text)
            conversation_log = []
            tool_results = []
            
            status = AgentStatus.RUNNING
            turn_number = 0
            
            # Main interaction loop
            while (status == AgentStatus.RUNNING and 
                   turn_number < self.config.max_turns and
                   time.time() - start_time < self.config.total_timeout):
                
                turn_number += 1
                turn_start_time = time.time()
                
                # Convert messages to chat template
                input_text = self.validation_manager.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize for generation
                inputs = self.validation_manager.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.validation_manager.tokenizer.model_max_length
                )
                
                # Generate response
                responses = self.validation_manager.generate_responses(
                    inputs["input_ids"].to(input_ids.device),
                    inputs["attention_mask"].to(attention_mask.device),
                    None,  # position_ids will be auto-generated
                    **gen_kwargs
                )
                
                if not responses:
                    status = AgentStatus.FAILED
                    break
                
                raw_response = responses[0]["response"]
                tokens_used = responses[0].get("response_length", len(raw_response.split()))  # Fallback to word count
                
                # Parse the 3 response types (similar to simple_llm_agent_test.py)
                reasoning_content = None
                regular_content = raw_response
                function_call = None
                
                # Extract reasoning content if present
                if hasattr(responses[0], 'reasoning_content') and responses[0].get('reasoning_content'):
                    reasoning_content = responses[0]['reasoning_content']
                    regular_content = responses[0]['response']
                else:
                    # Check for <think> tags in text
                    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
                    if think_match:
                        reasoning_content = think_match.group(1).strip()
                        regular_content = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
                
                # Print reasoning if present (before adding to messages)
                if reasoning_content:
                    logger.info(json.dumps({
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": ""
                    }))
                
                # Check for function calls - handle the JSON format the model is actually generating
                function_call = None
                
                # Try to find and parse the entire JSON function call
                json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}'
                json_matches = re.findall(json_pattern, regular_content, re.DOTALL)
                
                if json_matches:
                    try:
                        # Parse the entire JSON object
                        json_str = json_matches[0]
                        func_call_obj = json.loads(json_str)
                        
                        function_call = {
                            'name': func_call_obj['name'],
                            'arguments': func_call_obj['arguments']
                        }
                    except Exception as e:
                        # If JSON parsing fails, try regex extraction
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                        args_match = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', json_str)
                        
                        if name_match and args_match:
                            func_name = name_match.group(1)
                            try:
                                func_args = json.loads(args_match.group(1))
                            except:
                                func_args = {'input': args_match.group(1)}
                            
                            function_call = {
                                'name': func_name,
                                'arguments': func_args
                            }
                else:
                    # Try XML format as fallback
                    xml_pattern = r"<tool_call>\s*(\w+)\s*</tool_call>\s*<tool_input>\s*(.*?)\s*</tool_input>"
                    xml_matches = re.findall(xml_pattern, regular_content, re.DOTALL | re.IGNORECASE)
                    
                    if xml_matches:
                        func_name, func_args_str = xml_matches[0]
                        try:
                            func_args = json.loads(func_args_str)
                        except:
                            func_args = {'input': func_args_str.strip()}
                            
                        function_call = {
                            'name': func_name,
                            'arguments': func_args
                        }
                    else:
                        # Try function call format as last resort
                        func_pattern = r'(web_search|web_visit|python|bash|search)\s*\(([^)]*)\)'
                        func_matches = re.findall(func_pattern, regular_content)
                        
                        if func_matches:
                            func_name, func_args_str = func_matches[0]
                            try:
                                json_match = re.search(r'\{.*\}', func_args_str)
                                if json_match:
                                    func_args = json.loads(json_match.group())
                                else:
                                    func_args = self._parse_function_args(func_name, func_args_str)
                            except:
                                func_args = {'input': func_args_str.strip()}
                            
                            function_call = {
                                'name': func_name,
                                'arguments': func_args
                            }
                
                conversation_log.append({
                    'turn': turn_number,
                    'type': 'assistant',
                    'content': regular_content,
                    'reasoning_content': reasoning_content,
                    'function_call': function_call
                })
                
                # Handle function calls
                if function_call:
                    func_name = function_call.get('name', '')
                    func_args = function_call.get('arguments', {})
                    
                    # Print conversation format for function call
                    logger.info(json.dumps({
                        "role": "assistant", 
                        "content": regular_content,
                        "function_call": {
                            "name": func_name,
                            "arguments": json.dumps(func_args)
                        }
                    }))
                    
                    # Execute function call
                    tool_input = json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)
                    result = await self.tool_server.execute_tool(func_name, tool_input)
                    
                    # Print conversation format for function result
                    formatted_response = result.output[:200] + ('...' if len(result.output) > 200 else '')
                    logger.info(json.dumps({
                        "role": "function",
                        "name": func_name,
                        "content": formatted_response
                    }))
                    
                    # Add function call to messages (assistant with tool_call)
                    messages.append({
                        "role": "assistant",
                        "content": regular_content  # This includes the <tool_call>...</tool_call>
                    })
                    
                    # Add function result to messages as user message
                    messages.append({
                        "role": "user",
                        "content": result.output  # Tool result as user message
                    })
                    
                    tool_results.append(result)
                    
                    # Update statistics
                    self.interaction_stats["total_tool_calls"] += 1
                    if result.success:
                        self.interaction_stats["successful_tool_calls"] += 1
                
                    # Continue conversation after function call
                    continue
                else:
                    # Final answer: content without reasoning or function calls
                    # This is the stopping condition
                    logger.info(json.dumps({
                        "role": "assistant", 
                        "content": regular_content
                    }))
                    
                    # Add final assistant message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": regular_content
                    })
                    
                    # No function calls and has content - this is the final answer
                    status = AgentStatus.COMPLETED
                    break
                
                # Create turn record
                turn = AgentTurn(
                    turn_number=turn_number,
                    prompt=input_text,
                    response=regular_content,
                    tool_calls=tool_results if function_call else [],
                    status=status,
                    execution_time=time.time() - turn_start_time,
                    tokens_used=tokens_used
                )
                
                turns.append(turn)
                self.current_interactions[interaction_id].append(turn)
                self.interaction_stats["total_turns"] += 1
            
            # Determine final status
            if status == AgentStatus.RUNNING:
                if turn_number >= self.config.max_turns:
                    status = AgentStatus.MAX_TURNS
                elif time.time() - start_time >= self.config.total_timeout:
                    status = AgentStatus.TIMEOUT
            
            # Update statistics
            if status == AgentStatus.COMPLETED:
                self.interaction_stats["successful_interactions"] += 1
            elif status == AgentStatus.TIMEOUT:
                self.interaction_stats["timeout_interactions"] += 1
            else:
                self.interaction_stats["failed_interactions"] += 1
            
            # Extract final answer from last assistant message
            final_answer = ""
            for log_entry in reversed(conversation_log):
                if log_entry['type'] == 'assistant' and not log_entry.get('function_call'):
                    final_answer = log_entry['content']
                    break
            
            # Create final result
            result = AgentInteractionResult(
                final_response=final_answer,
                tool_calls=[tc for turn in turns for tc in turn.tool_calls],
                total_turns=turn_number,
                success=(status == AgentStatus.COMPLETED),
                total_time=time.time() - start_time,
                reasoning_trace=[f"Turn {turn.turn_number}: {turn.response}" for turn in turns]
            )
            
            # Add messages field for logging
            result.messages = messages
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agent interaction {interaction_id}: {e}")
            self.interaction_stats["failed_interactions"] += 1
            
            return AgentInteractionResult(
                final_response="",
                tool_calls=[],
                total_turns=turn_number,
                success=False,
                total_time=time.time() - start_time,
                reasoning_trace=[f"Error: {str(e)}"]
            )
        
        finally:
            # Clean up interaction tracking
            if interaction_id in self.current_interactions:
                del self.current_interactions[interaction_id]
    
    def _extract_messages_from_prompt(self, prompt_text: str) -> List[Dict[str, Any]]:
        """Extract messages from initial prompt text."""
        messages = []
        
        # Try to parse chat format markers
        # This is a simplified version - adjust based on your tokenizer format
        if "<|im_start|>system" in prompt_text:
            # Extract system prompt
            system_match = re.search(r'<\|im_start\|>system\s*(.+?)\s*<\|im_end\|>', prompt_text, re.DOTALL)
            if system_match:
                messages.append({"role": "system", "content": system_match.group(1).strip()})
            
            # Extract user prompt
            user_match = re.search(r'<\|im_start\|>user\s*(.+?)\s*<\|im_end\|>', prompt_text, re.DOTALL)
            if user_match:
                messages.append({"role": "user", "content": user_match.group(1).strip()})
        else:
            # Fallback: treat entire prompt as user message
            messages = [
                {"role": "user", "content": prompt_text.strip()}
            ]
        
        return messages
    
    def _parse_function_args(self, func_name: str, func_args_str: str) -> Dict[str, Any]:
        """Parse function arguments based on function name."""
        if func_name == 'web_search':
            query_match = re.search(r'["\']([^"\']+)["\']', func_args_str)
            func_args = {'query': query_match.group(1) if query_match else func_args_str.strip()}
            # Extract other parameters
            num_match = re.search(r'(?:num_results|top_k)\s*=\s*(\d+)', func_args_str)
            if num_match:
                func_args['top_k'] = int(num_match.group(1))
            preview_match = re.search(r'preview_char\s*=\s*(\d+)', func_args_str)
            if preview_match:
                func_args['preview_char'] = int(preview_match.group(1))
        elif func_name == 'web_visit':
            url_match = re.search(r'["\']([^"\']+)["\']', func_args_str)
            func_args = {'url': url_match.group(1) if url_match else func_args_str.strip()}
        elif func_name in ['python', 'bash']:
            # For code execution tools, the entire string is the code
            func_args = {'code': func_args_str.strip()}
        else:
            func_args = {'input': func_args_str.strip()}
        
        return func_args
    
    async def _execute_tool_calls(self, tool_calls: List[Tuple[str, str]]) -> List[ToolCallResult]:
        """Execute a list of tool calls."""
        if self.config.allow_parallel_tool_calls:
            # Execute tool calls in parallel
            tasks = []
            for tool_name, tool_input in tool_calls:
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.tool_server.execute_tool(tool_name, tool_input),
                        timeout=self.config.tool_call_timeout
                    )
                )
                tasks.append(task)
            
            results = []
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                except asyncio.TimeoutError:
                    results.append(ToolCallResult(
                        success=False,
                        output="",
                        error="Tool call timeout",
                        tool_name="unknown",
                        execution_time=self.config.tool_call_timeout
                    ))
                except Exception as e:
                    results.append(ToolCallResult(
                        success=False,
                        output="",
                        error=str(e),
                        tool_name="unknown",
                        execution_time=0.0
                    ))
            
            return results
        else:
            # Execute tool calls sequentially
            results = []
            for tool_name, tool_input in tool_calls:
                try:
                    result = await asyncio.wait_for(
                        self.tool_server.execute_tool(tool_name, tool_input),
                        timeout=self.config.tool_call_timeout
                    )
                    results.append(result)
                except asyncio.TimeoutError:
                    results.append(ToolCallResult(
                        success=False,
                        output="",
                        error="Tool call timeout",
                        tool_name=tool_name,
                        execution_time=self.config.tool_call_timeout
                    ))
                except Exception as e:
                    results.append(ToolCallResult(
                        success=False,
                        output="",
                        error=str(e),
                        tool_name=tool_name,
                        execution_time=0.0
                    ))
            
            return results
    
    def _contains_stop_sequence(self, text: str) -> bool:
        """Check if text contains any stop sequences."""
        for stop_seq in self.config.stop_sequences:
            if stop_seq in text:
                return True
        return False
    
    
    def _format_tool_results(self, tool_results: List[ToolCallResult]) -> str:
        """Format tool results for display."""
        if not tool_results:
            return ""
        
        formatted_results = []
        for result in tool_results:
            if result.success:
                formatted_results.append(f"✓ {result.tool_name}: {result.output}")
            else:
                formatted_results.append(f"✗ {result.tool_name}: {result.error}")
        
        return "\n".join(formatted_results)
    
    async def run_batch_interactions(self, 
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   position_ids: Optional[torch.Tensor] = None,
                                   **generation_kwargs) -> List[AgentInteractionResult]:
        """
        Run agent interactions for a batch of inputs.
        
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
        
        # Process each sample in the batch
        for i in range(batch_size):
            single_input_ids = input_ids[i:i+1]
            single_attention_mask = attention_mask[i:i+1]
            single_position_ids = position_ids[i:i+1] if position_ids is not None else None
            
            interaction_id = f"batch_{i}_{int(time.time() * 1000)}"
            
            result = await self.run_agent_interaction(
                single_input_ids,
                single_attention_mask,
                single_position_ids,
                interaction_id=interaction_id,
                **generation_kwargs
            )
            results.append(result)
        
        return results
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get interaction statistics."""
        stats = self.interaction_stats.copy()
        
        # Calculate derived statistics
        if stats["total_interactions"] > 0:
            stats["success_rate"] = stats["successful_interactions"] / stats["total_interactions"]
            stats["failure_rate"] = stats["failed_interactions"] / stats["total_interactions"]
            stats["timeout_rate"] = stats["timeout_interactions"] / stats["total_interactions"]
            stats["avg_turns_per_interaction"] = stats["total_turns"] / stats["total_interactions"]
        
        if stats["total_tool_calls"] > 0:
            stats["tool_success_rate"] = stats["successful_tool_calls"] / stats["total_tool_calls"]
            stats["avg_tool_calls_per_interaction"] = stats["total_tool_calls"] / stats["total_interactions"]
        
        return stats
    
    def reset_statistics(self):
        """Reset interaction statistics."""
        self.interaction_stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
            "timeout_interactions": 0,
            "total_turns": 0,
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tool_server.get_available_tools()