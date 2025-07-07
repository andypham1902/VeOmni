"""
Tool-based evaluation metrics for agent validation.

This module provides evaluation metrics for tool-based agent interactions,
including task completion rates, tool usage efficiency, and reasoning quality.
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

from veomni.utils.tool_validation_utils import AgentInteractionResult, ToolCallResult

logger = logging.getLogger(__name__)


@dataclass
class ToolEvaluationResult:
    """Result of tool-based evaluation."""
    task_completion_score: float
    tool_usage_efficiency: float
    reasoning_quality: float
    final_answer_correctness: float
    overall_score: float
    detailed_metrics: Dict[str, Any]


class ToolBasedEvaluator(ABC):
    """Abstract base class for tool-based evaluators."""
    
    @abstractmethod
    def evaluate(self, 
                 interaction_result: AgentInteractionResult,
                 ground_truth: Any,
                 **kwargs) -> ToolEvaluationResult:
        """Evaluate an agent interaction result."""
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        """Get the task type this evaluator handles."""
        pass


class MathToolEvaluator(ToolBasedEvaluator):
    """Evaluator for math problem solving with tools."""
    
    def __init__(self, required_tools: Optional[List[str]] = None):
        """
        Initialize the math tool evaluator.
        
        Args:
            required_tools: List of tools that should be used for this task
        """
        self.required_tools = required_tools or ["python", "calculator"]
    
    def evaluate(self, 
                 interaction_result: AgentInteractionResult,
                 ground_truth: str,
                 **kwargs) -> ToolEvaluationResult:
        """
        Evaluate math problem solving with tools.
        
        Args:
            interaction_result: Result of agent interaction
            ground_truth: Expected answer
            **kwargs: Additional evaluation parameters
            
        Returns:
            ToolEvaluationResult with evaluation metrics
        """
        # Extract final answer
        final_answer = self._extract_final_answer(interaction_result.final_response)
        
        # Check answer correctness
        answer_correct = self._check_answer_correctness(final_answer, ground_truth)
        
        # Evaluate tool usage
        tool_usage_score = self._evaluate_tool_usage(interaction_result.tool_calls)
        
        # Evaluate reasoning quality
        reasoning_score = self._evaluate_reasoning_quality(interaction_result.reasoning_trace)
        
        # Calculate task completion score
        task_completion_score = self._calculate_task_completion_score(
            interaction_result, answer_correct
        )
        
        # Calculate overall score
        overall_score = (
            answer_correct * 0.4 +
            tool_usage_score * 0.2 +
            reasoning_score * 0.2 +
            task_completion_score * 0.2
        )
        
        detailed_metrics = {
            "final_answer": final_answer,
            "answer_correct": answer_correct,
            "ground_truth": ground_truth,
            "tools_used": [tc.tool_name for tc in interaction_result.tool_calls],
            "successful_tool_calls": sum(1 for tc in interaction_result.tool_calls if tc.success),
            "total_tool_calls": len(interaction_result.tool_calls),
            "total_turns": interaction_result.total_turns,
            "interaction_success": interaction_result.success,
            "total_time": interaction_result.total_time,
        }
        
        return ToolEvaluationResult(
            task_completion_score=task_completion_score,
            tool_usage_efficiency=tool_usage_score,
            reasoning_quality=reasoning_score,
            final_answer_correctness=answer_correct,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics
        )
    
    def get_task_type(self) -> str:
        """Get task type."""
        return "math"
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Look for common answer patterns
        patterns = [
            r"the answer is\s*([^\n.]+)",
            r"final answer:\s*([^\n.]+)",
            r"answer:\s*([^\n.]+)",
            r"=\s*([^\n.]+)",
            r"therefore,?\s*([^\n.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return the last line
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def _check_answer_correctness(self, final_answer: str, ground_truth: str) -> float:
        """Check if the final answer is correct."""
        # Normalize answers
        final_answer_norm = self._normalize_answer(final_answer)
        ground_truth_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if final_answer_norm == ground_truth_norm:
            return 1.0
        
        # Try to evaluate as numbers
        try:
            final_num = float(final_answer_norm)
            ground_num = float(ground_truth_norm)
            
            # Check if they're close (within 1e-6)
            if abs(final_num - ground_num) < 1e-6:
                return 1.0
            
            # Check percentage error
            if ground_num != 0:
                error_rate = abs(final_num - ground_num) / abs(ground_num)
                if error_rate < 0.01:  # Within 1%
                    return 0.9
                elif error_rate < 0.05:  # Within 5%
                    return 0.7
                elif error_rate < 0.1:  # Within 10%
                    return 0.5
        except ValueError:
            pass
        
        # Check if answers contain similar key information
        if self._contains_similar_info(final_answer_norm, ground_truth_norm):
            return 0.3
        
        return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove extra whitespace
        answer = answer.strip()
        
        # Remove common prefixes/suffixes
        answer = re.sub(r'^(the answer is|answer:?|final answer:?)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'[.!?]+$', '', answer)
        
        # Normalize mathematical expressions
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.replace('=', '').strip()
        
        return answer.lower()
    
    def _contains_similar_info(self, answer1: str, answer2: str) -> bool:
        """Check if answers contain similar key information."""
        # Extract numbers from both answers
        numbers1 = re.findall(r'-?\d+\.?\d*', answer1)
        numbers2 = re.findall(r'-?\d+\.?\d*', answer2)
        
        # Check if they have common numbers
        return len(set(numbers1) & set(numbers2)) > 0
    
    def _evaluate_tool_usage(self, tool_calls: List[ToolCallResult]) -> float:
        """Evaluate tool usage efficiency."""
        if not tool_calls:
            return 0.0
        
        # Check if required tools were used
        tools_used = set(tc.tool_name for tc in tool_calls)
        required_used = sum(1 for tool in self.required_tools if tool in tools_used)
        
        # Calculate success rate
        successful_calls = sum(1 for tc in tool_calls if tc.success)
        success_rate = successful_calls / len(tool_calls)
        
        # Calculate efficiency (fewer calls is better, but not too few)
        efficiency = min(1.0, 3.0 / len(tool_calls)) if len(tool_calls) > 0 else 0.0
        
        # Combine metrics
        tool_score = (
            (required_used / len(self.required_tools)) * 0.4 +
            success_rate * 0.4 +
            efficiency * 0.2
        )
        
        return tool_score
    
    def _evaluate_reasoning_quality(self, reasoning_trace: List[str]) -> float:
        """Evaluate the quality of reasoning."""
        if not reasoning_trace:
            return 0.0
        
        # Check for mathematical reasoning keywords
        reasoning_keywords = [
            "calculate", "compute", "solve", "equation", "formula",
            "step", "first", "then", "next", "therefore", "because",
            "substitute", "simplify", "evaluate"
        ]
        
        keyword_score = 0.0
        total_text = " ".join(reasoning_trace).lower()
        
        for keyword in reasoning_keywords:
            if keyword in total_text:
                keyword_score += 1.0
        
        # Normalize by number of keywords
        keyword_score = min(1.0, keyword_score / len(reasoning_keywords))
        
        # Check for logical structure
        structure_score = 0.0
        if len(reasoning_trace) > 1:
            structure_score = 0.5  # Multiple turns show iterative reasoning
        
        # Check for tool integration
        integration_score = 0.0
        for trace in reasoning_trace:
            if any(word in trace.lower() for word in ["tool", "calculate", "compute", "result"]):
                integration_score = 1.0
                break
        
        return (keyword_score * 0.4 + structure_score * 0.3 + integration_score * 0.3)
    
    def _calculate_task_completion_score(self, 
                                       interaction_result: AgentInteractionResult,
                                       answer_correct: float) -> float:
        """Calculate task completion score."""
        # Base score from interaction success
        base_score = 1.0 if interaction_result.success else 0.5
        
        # Adjust based on efficiency
        if interaction_result.total_turns <= 3:
            efficiency_bonus = 0.2
        elif interaction_result.total_turns <= 5:
            efficiency_bonus = 0.1
        else:
            efficiency_bonus = 0.0
        
        # Combine with answer correctness
        completion_score = base_score * 0.7 + answer_correct * 0.3 + efficiency_bonus
        
        return min(1.0, completion_score)


class CodeToolEvaluator(ToolBasedEvaluator):
    """Evaluator for coding tasks with tools."""
    
    def __init__(self, required_tools: Optional[List[str]] = None):
        """
        Initialize the code tool evaluator.
        
        Args:
            required_tools: List of tools that should be used for this task
        """
        self.required_tools = required_tools or ["python", "bash"]
    
    def evaluate(self, 
                 interaction_result: AgentInteractionResult,
                 ground_truth: Dict[str, Any],
                 **kwargs) -> ToolEvaluationResult:
        """
        Evaluate coding task with tools.
        
        Args:
            interaction_result: Result of agent interaction
            ground_truth: Expected result (dict with 'expected_output', 'test_cases', etc.)
            **kwargs: Additional evaluation parameters
            
        Returns:
            ToolEvaluationResult with evaluation metrics
        """
        # Extract code from interaction
        code = self._extract_code(interaction_result.final_response, interaction_result.tool_calls)
        
        # Evaluate code correctness
        correctness_score = self._evaluate_code_correctness(
            code, ground_truth, interaction_result.tool_calls
        )
        
        # Evaluate tool usage
        tool_usage_score = self._evaluate_tool_usage(interaction_result.tool_calls)
        
        # Evaluate code quality
        quality_score = self._evaluate_code_quality(code)
        
        # Calculate task completion score
        task_completion_score = self._calculate_task_completion_score(
            interaction_result, correctness_score
        )
        
        # Calculate overall score
        overall_score = (
            correctness_score * 0.4 +
            tool_usage_score * 0.2 +
            quality_score * 0.2 +
            task_completion_score * 0.2
        )
        
        detailed_metrics = {
            "code_extracted": code,
            "correctness_score": correctness_score,
            "tools_used": [tc.tool_name for tc in interaction_result.tool_calls],
            "successful_tool_calls": sum(1 for tc in interaction_result.tool_calls if tc.success),
            "total_tool_calls": len(interaction_result.tool_calls),
            "total_turns": interaction_result.total_turns,
            "interaction_success": interaction_result.success,
            "total_time": interaction_result.total_time,
        }
        
        return ToolEvaluationResult(
            task_completion_score=task_completion_score,
            tool_usage_efficiency=tool_usage_score,
            reasoning_quality=quality_score,
            final_answer_correctness=correctness_score,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics
        )
    
    def get_task_type(self) -> str:
        """Get task type."""
        return "code"
    
    def _extract_code(self, response: str, tool_calls: List[ToolCallResult]) -> str:
        """Extract code from response and tool calls."""
        # Look for code blocks in response
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1]  # Return the last code block
        
        # Look for code in tool calls
        for tool_call in tool_calls:
            if tool_call.tool_name.lower() == "python" and tool_call.success:
                return tool_call.output
        
        return ""
    
    def _evaluate_code_correctness(self, 
                                 code: str, 
                                 ground_truth: Dict[str, Any],
                                 tool_calls: List[ToolCallResult]) -> float:
        """Evaluate code correctness."""
        if not code:
            return 0.0
        
        # Check if code ran successfully
        execution_success = any(
            tc.success for tc in tool_calls if tc.tool_name.lower() == "python"
        )
        
        if not execution_success:
            return 0.0
        
        # Get expected output
        expected_output = ground_truth.get("expected_output", "")
        
        # Find actual output from tool calls
        actual_output = ""
        for tool_call in tool_calls:
            if tool_call.tool_name.lower() == "python" and tool_call.success:
                actual_output = tool_call.output
                break
        
        # Compare outputs
        if expected_output and actual_output:
            if expected_output.strip() == actual_output.strip():
                return 1.0
            elif expected_output.strip() in actual_output.strip():
                return 0.8
            else:
                return 0.3
        
        # If no expected output, just check execution success
        return 0.7 if execution_success else 0.0
    
    def _evaluate_tool_usage(self, tool_calls: List[ToolCallResult]) -> float:
        """Evaluate tool usage efficiency."""
        if not tool_calls:
            return 0.0
        
        # Check if required tools were used
        tools_used = set(tc.tool_name for tc in tool_calls)
        required_used = sum(1 for tool in self.required_tools if tool in tools_used)
        
        # Calculate success rate
        successful_calls = sum(1 for tc in tool_calls if tc.success)
        success_rate = successful_calls / len(tool_calls)
        
        # Calculate efficiency
        efficiency = min(1.0, 5.0 / len(tool_calls)) if len(tool_calls) > 0 else 0.0
        
        # Combine metrics
        tool_score = (
            (required_used / len(self.required_tools)) * 0.4 +
            success_rate * 0.4 +
            efficiency * 0.2
        )
        
        return tool_score
    
    def _evaluate_code_quality(self, code: str) -> float:
        """Evaluate code quality."""
        if not code:
            return 0.0
        
        quality_score = 0.0
        
        # Check for proper structure
        if "def " in code or "class " in code:
            quality_score += 0.3
        
        # Check for comments
        if "#" in code:
            quality_score += 0.2
        
        # Check for error handling
        if "try:" in code and "except:" in code:
            quality_score += 0.2
        
        # Check for imports
        if "import " in code:
            quality_score += 0.1
        
        # Check for proper variable names
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=', code):
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _calculate_task_completion_score(self, 
                                       interaction_result: AgentInteractionResult,
                                       correctness_score: float) -> float:
        """Calculate task completion score."""
        # Base score from interaction success
        base_score = 1.0 if interaction_result.success else 0.5
        
        # Adjust based on efficiency
        if interaction_result.total_turns <= 3:
            efficiency_bonus = 0.2
        elif interaction_result.total_turns <= 5:
            efficiency_bonus = 0.1
        else:
            efficiency_bonus = 0.0
        
        # Combine with correctness
        completion_score = base_score * 0.6 + correctness_score * 0.4 + efficiency_bonus
        
        return min(1.0, completion_score)


class GeneralToolEvaluator(ToolBasedEvaluator):
    """General-purpose tool evaluator for various tasks."""
    
    def __init__(self):
        """Initialize the general tool evaluator."""
        pass
    
    def evaluate(self, 
                 interaction_result: AgentInteractionResult,
                 ground_truth: Any,
                 **kwargs) -> ToolEvaluationResult:
        """
        Evaluate general tool usage.
        
        Args:
            interaction_result: Result of agent interaction
            ground_truth: Expected result (flexible format)
            **kwargs: Additional evaluation parameters
            
        Returns:
            ToolEvaluationResult with evaluation metrics
        """
        # Basic evaluation based on interaction success
        task_completion_score = 1.0 if interaction_result.success else 0.5
        
        # Evaluate tool usage
        tool_usage_score = self._evaluate_tool_usage(interaction_result.tool_calls)
        
        # Evaluate response quality
        response_quality = self._evaluate_response_quality(
            interaction_result.final_response, ground_truth
        )
        
        # Calculate overall score
        overall_score = (
            task_completion_score * 0.4 +
            tool_usage_score * 0.3 +
            response_quality * 0.3
        )
        
        detailed_metrics = {
            "tools_used": [tc.tool_name for tc in interaction_result.tool_calls],
            "successful_tool_calls": sum(1 for tc in interaction_result.tool_calls if tc.success),
            "total_tool_calls": len(interaction_result.tool_calls),
            "total_turns": interaction_result.total_turns,
            "interaction_success": interaction_result.success,
            "total_time": interaction_result.total_time,
        }
        
        return ToolEvaluationResult(
            task_completion_score=task_completion_score,
            tool_usage_efficiency=tool_usage_score,
            reasoning_quality=response_quality,
            final_answer_correctness=response_quality,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics
        )
    
    def get_task_type(self) -> str:
        """Get task type."""
        return "general"
    
    def _evaluate_tool_usage(self, tool_calls: List[ToolCallResult]) -> float:
        """Evaluate general tool usage."""
        if not tool_calls:
            return 0.0
        
        # Calculate success rate
        successful_calls = sum(1 for tc in tool_calls if tc.success)
        success_rate = successful_calls / len(tool_calls)
        
        # Calculate efficiency (not too many calls)
        efficiency = min(1.0, 3.0 / len(tool_calls)) if len(tool_calls) > 0 else 0.0
        
        return success_rate * 0.7 + efficiency * 0.3
    
    def _evaluate_response_quality(self, response: str, ground_truth: Any) -> float:
        """Evaluate response quality."""
        if not response:
            return 0.0
        
        # Basic quality metrics
        quality_score = 0.0
        
        # Check response length (not too short, not too long)
        if 10 <= len(response) <= 1000:
            quality_score += 0.3
        
        # Check for complete sentences
        if response.endswith('.') or response.endswith('!') or response.endswith('?'):
            quality_score += 0.2
        
        # Check for coherence (basic)
        if len(response.split()) > 5:
            quality_score += 0.2
        
        # Check against ground truth if provided
        if ground_truth and isinstance(ground_truth, str):
            if ground_truth.lower() in response.lower():
                quality_score += 0.3
        
        return min(1.0, quality_score)


def get_evaluator(task_type: str, **kwargs) -> ToolBasedEvaluator:
    """
    Get appropriate evaluator for task type.
    
    Args:
        task_type: Type of task ("math", "code", "general")
        **kwargs: Additional arguments for evaluator
        
    Returns:
        ToolBasedEvaluator instance
    """
    if task_type.lower() == "math":
        return MathToolEvaluator(**kwargs)
    elif task_type.lower() == "code":
        return CodeToolEvaluator(**kwargs)
    else:
        return GeneralToolEvaluator(**kwargs)


def compute_tool_score(interaction_result: AgentInteractionResult, 
                      ground_truth: Any,
                      task_type: str = "general",
                      **kwargs) -> Dict[str, Any]:
    """
    Compute tool-based evaluation score.
    
    Args:
        interaction_result: Result of agent interaction
        ground_truth: Expected result
        task_type: Type of task
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = get_evaluator(task_type, **kwargs)
    result = evaluator.evaluate(interaction_result, ground_truth, **kwargs)
    
    return {
        "overall_score": result.overall_score,
        "task_completion_score": result.task_completion_score,
        "tool_usage_efficiency": result.tool_usage_efficiency,
        "reasoning_quality": result.reasoning_quality,
        "final_answer_correctness": result.final_answer_correctness,
        "detailed_metrics": result.detailed_metrics
    }