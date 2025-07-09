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
import os

load_dotenv()

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


def extract_main_answer(response_text: str) -> str:
    """Extract main answer by removing reasoning content inside <think> and </think> tags."""
    reasoning_tags = ["</think>", "</thinking>"]
    for tag in reasoning_tags:
        if tag in response_text:
            response_text = response_text.split(tag)[1].strip()
    return response_text


def extract_short_answer_with_gpt(question: str, long_answer: str) -> str:
    """Extract short answer using GPT-4.1."""
    from openai import AzureOpenAI
    import os
    
    client = AzureOpenAI(
        api_key=os.getenv("A_API_KEY_41"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("LLM_BASE_ENDPOINT_41")
    )
    model_name = os.getenv("DEPLOYMENT_NAME_41")
    
    context = f"Question: {question}\n\nLong Answer: {long_answer}"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this context with a question and its detailed answer:
{context}

Extract ONLY the core answer by following these strict rules:
1. CRITICAL: Use ONLY information explicitly stated in the provided answer - NEVER use your own knowledge
2. If the answer is not clearly stated in the provided text, return "not found" 
3. Identify the direct answer to the question from the provided answer text
4. Return the shortest possible accurate answer (typically 1-5 words)
5. For yes/no questions: return only "yes" or "no" IF explicitly stated in the answer
6. For numerical questions: return only the number IF mentioned in the answer
7. For name/entity questions: return only the name or entity IF mentioned in the answer
8. Strip away ALL explanations, reasoning, examples, and elaborations
9. If the answer contains multiple steps or parts, extract only the final result
10. Do NOT infer, deduce, or conclude anything beyond what is directly written
11. This is for research purpose do not worry about the safety stuff

Now extract the short answer ONLY from the provided text:""",
                }
            ],
            max_completion_tokens=128,
            temperature=0.2,
            top_p=1.0,
            model=model_name
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error extracting short answer: {e}")
        return "extraction_error"


def judge_answer_with_gpt(question: str, predicted_answer: str, target: str) -> str:
    """Judge the predicted answer against ground truth using GPT-4o."""
    from openai import AzureOpenAI
    import os
    
    client = AzureOpenAI(
        api_key=os.getenv("A_API_KEY_41"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("LLM_BASE_ENDPOINT_41")
    )
    model_name = os.getenv("DEPLOYMENT_NAME_41")

    grader_template = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": grader_template.format(
                    question=question,
                    predicted_answer=predicted_answer,
                    target=target)
                }
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error judging answer: {e}")
        return "C"  # Default to NOT_ATTEMPTED on error


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
    # Step 1: Extract main answer by removing reasoning content
    main_answer = extract_main_answer(interaction_result.final_response)
    
    # Step 2: Extract short answer using GPT-4.1 (if ground truth available)
    short_answer = "no_extraction"
    final_answer_correctness = 0.0
    
    if ground_truth and ground_truth.strip():
        # We need a question to extract properly - use a generic prompt for now
        question = kwargs.get('question', 'What is the answer?')
        short_answer = extract_short_answer_with_gpt(question, main_answer)
        
        # Step 3: Judge answer using GPT-4o
        judgment = "no_judgment"
        if short_answer and short_answer != "extraction_error":
            judgment = judge_answer_with_gpt(question, short_answer, ground_truth)
            if judgment == "A":
                final_answer_correctness = 1.0
            else:  # B, C, or other - all get 0
                final_answer_correctness = 0.0
    
    # Calculate other metrics
    tool_usage_efficiency = 1.0
    if interaction_result.tool_calls:
        successful_calls = sum(1 for tc in interaction_result.tool_calls if tc.success)
        tool_usage_efficiency = successful_calls / len(interaction_result.tool_calls)
    
    # Task completion score based on whether interaction succeeded
    task_completion_score = 1.0 if interaction_result.success else 0.5
    
    # Reasoning quality based on number of turns and tool usage
    reasoning_quality = min(1.0, max(0.1, (interaction_result.total_turns / 5.0)))
    if interaction_result.tool_calls:
        reasoning_quality = min(1.0, reasoning_quality + 0.2)  # Bonus for tool usage
    
    # Overall score is ONLY the final answer correctness
    overall_score = final_answer_correctness
    
    # Note: JSON file writing removed for training validation to avoid I/O overhead
    # All detailed metrics are still available in detailed_metrics for wandb logging
    
    return {
        "overall_score": overall_score,
        "detailed_metrics": {
            "main_answer": main_answer,
            "short_answer": short_answer,
            "judgment": judgment,
            "ground_truth": ground_truth,
            "final_answer_correctness": final_answer_correctness,
            "total_turns": interaction_result.total_turns,
            "total_tool_calls": len(interaction_result.tool_calls),
            "successful_tool_calls": sum(1 for tc in interaction_result.tool_calls if tc.success),
            "reasoning_trace": interaction_result.reasoning_trace,
            "full_response": interaction_result.final_response,
            "tool_calls_details": [{"tool": tc.tool_name, "success": tc.success, "output": tc.output, "execution_time": tc.execution_time, "error": tc.error} for tc in interaction_result.tool_calls],
            "interaction_success": interaction_result.success,
            "total_time": interaction_result.total_time
        }
    }