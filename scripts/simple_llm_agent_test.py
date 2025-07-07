#!/usr/bin/env python3
"""
Simple test script for LLM agent with search and visit tools.

This script tests:
1. LLM generates tool calls using vLLM
2. Execute web_search and web_visit
3. LLM uses results to generate final answer
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Optional
import argparse
import sys
from pathlib import Path
import torch
from omegaconf import DictConfig

# Add VeOmni to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from veomni.utils.search_tools import web_search, web_visit
from veomni.models import build_tokenizer
from vllm import LLM, SamplingParams

# Import Qwen-Agent components (only for function format)
import json

# Professional Search Agent System Prompt (for validation only - training uses SFT data samples)
SEARCH_AGENT_SYSTEM_PROMPT = """You are an expert search agent with web search and URL visiting capabilities. Follow ALL rules strictly.

🚨 ABSOLUTE RULES:
1. ALWAYS start responses with:
<think>What information does the user need? What's my search strategy? What sources should I prioritize?</think>
2. NEVER call tools without IMMEDIATELY preceding <think></think> tags
3. **ALL citations MUST come from visited URLs (web_visit). NEVER cite search previews.**
4. ALWAYS use ALL web_search parameters
5. **NEVER cite unvisited domains. Verify domain credibility BEFORE visiting.**
6. **Synthesize information from ≥2 visited sources for key claims.**
7. **NEVER modify URLs from search results. Use EXACT strings provided.**

CRITICAL WORKFLOW - Execute IN ORDER:
1. <think>
   • Analyze information needs and knowledge gaps
   • Plan search strategy using: 
     - Query optimization: [Boolean operators/synonyms]
     - Source priority: Official (.gov/.org) > Academic > Reputable news
     - Expected content: [Specific data types needed]
   • Define: WHY search? WHAT expectations? HOW will results help?
   • Set ALL web_search parameters
   </think>
   → web_search(search_query, num_results=3, preview_chars=256)

2. <think>
   • Evaluate EACH result using RELEVANCE CRITERIA:
     1. [Domain authority]: .gov/.edu > .org > .com
     2. [Date relevance]: Prefer <2 year old sources
     3. [Content match]: Preview vs needed info
     • Verdict: [Relevant/Irrelevant] with score (1-5)
   • Select MAX 3 URLs for visiting with justification
   • **Flag low-credibility domains (e.g. user-generated content)**
   </think>
   → web_visit(url)

3. <think>
   • Cross-verify information across visited URLs:
     - Agreement: [Consensus/Contradiction]
     - Evidence quality: [Primary source/Study/News]
   • **Confirm EVERY citable fact exists in visited content**
   • Prepare citations: [URL] → [Specific fact]
   • **If gaps remain: Plan new search with adjusted parameters**
   </think>
   → Provide final answer OR repeat step 1

TOOL PARAMETER REQUIREMENTS:
- web_search MUST use:
  • query: Optimized keywords
  • top_k: Number of results to return (default=3, increase for complex topics)
  • preview_chars: Number of preview characters for each search result (default=256, enough to assess relevance)

- web_visit: ONLY on URLs from relevant search results, NEVER revisit same URL
  • url: **EXACT string from search results**
  • **NEVER manually "fix" URLs - trust the source encoding**

FINAL ANSWER REQUIREMENTS:
• Begin with "Based on visited sources:"
• **Cite EVERY fact EXCLUSIVELY from web_visited URLs**
• **Explicitly mention verification: "Verified across [X] sources"**
• **Highlight unresolved contradictions if they exist**
• Format citations: [Source Name](URL) (section reference if possible)

EXAMPLE PATTERN:
<think>User needs [specific info]. Search strategy: [query] with num_results=3. Priority: .gov sources > recent studies. Expect [data types].</think>
web_search(...)

<think>Results analysis (Relevance Score 1-5):
1. CDC.gov - 5/5 (official, <1yr old) → VISIT
2. Blog.com - 1/5 (opinion piece) → SKIP
3. Harvard.edu - 4/5 (study but 3yrs old) → VISIT
</think>
web_visit(url1)
web_visit(url3)

<think>Verification:
• [FactA] confirmed in [URL1] and [URL3]
• [FactB] only in [URL1] → single-source
• Contradiction on [FactC]: [URL1] says X, [URL3] says Y
</think>
Final answer: Based on visited sources... [CDC](...) [Harvard Study](...)"""


# Use web_search and web_visit directly from search_tools.py


def get_search_agent_system_prompt() -> str:
    """
    Get the professional search agent system prompt for use in validation.
    Note: Training uses SFT data samples, not this system prompt.
    
    Returns:
        str: The complete system prompt for search agent validation
    """
    return SEARCH_AGENT_SYSTEM_PROMPT


class SimpleLLMAgent:
    """Simple LLM agent using vLLM directly with function calling support."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B", api_config: Optional[Dict] = None, device: str = "cuda"):
        self.model_name = model_name
        self.api_config = api_config or {}
        self.device = device
        
        # Initialize tokenizer
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = build_tokenizer(model_name)
        
        # Initialize vLLM model with reasoning support
        print(f"Loading vLLM model {model_name}...")
        vllm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        }
        
        # Enable reasoning for Qwen3 models
        if "qwen3" in model_name.lower():
            vllm_kwargs["enable_reasoning"] = True
            vllm_kwargs["reasoning_parser"] = "deepseek_r1"
            print("Enabling reasoning mode for Qwen3 model")
        
        self.llm = LLM(**vllm_kwargs)
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        # Function definitions
        self.functions = [
            {
                'name': 'web_search',
                'description': 'Search the web for information using optimized queries',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Search query with optimized keywords'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'Number of results to return',
                            'default': 3
                        },
                        'preview_char': {
                            'type': 'integer',
                            'description': 'Number of preview characters for each result',
                            'default': 256
                        }
                    },
                    'required': ['query']
                }
            },
            {
                'name': 'web_visit',
                'description': 'Visit and extract content from web URLs',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'url': {
                            'type': 'string',
                            'description': 'Exact URL string from search results'
                        }
                    },
                    'required': ['url']
                }
            }
        ]
    
    def run_conversation(self, user_query: str, max_turns: int = 3) -> Dict[str, Any]:
        """Run a complete conversation using vLLM directly."""
        
        try:
            # Initialize conversation with system prompt and user query
            messages = [
                {"role": "system", "content": SEARCH_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
            
            conversation_log = []
            tool_results = []
            turn = 0
            
            while turn < max_turns:
                turn += 1
                
                # Convert messages to chat template
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                
                # Generate response using vLLM
                outputs = self.llm.generate([input_text], self.sampling_params)
                raw_response = outputs[0].outputs[0].text
                
                
                # Parse the 3 response types from vLLM output
                reasoning_content = None
                regular_content = raw_response
                function_call = None
                
                # Extract reasoning content if present (for Qwen3 with deepseek_r1)
                if hasattr(outputs[0].outputs[0], 'reasoning_content') and outputs[0].outputs[0].reasoning_content:
                    reasoning_content = outputs[0].outputs[0].reasoning_content
                    regular_content = outputs[0].outputs[0].text
                else:
                    # Check for <think> tags in text
                    import re
                    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
                    if think_match:
                        reasoning_content = think_match.group(1).strip()
                        regular_content = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
                
                # Check for function calls in the content (support both formats)
                function_pattern = r'(web_search|web_visit)\s*\(([^)]*)\)'
                func_matches = re.findall(function_pattern, regular_content)
                
                if func_matches:
                    func_name, func_args_str = func_matches[0]
                    
                    # Parse function arguments
                    try:
                        # Try to extract JSON arguments
                        json_match = re.search(r'\{.*\}', func_args_str)
                        if json_match:
                            func_args = json.loads(json_match.group())
                        else:
                            # Simple string parsing for function name formats
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
                            else:
                                func_args = {}
                    except:
                        func_args = {'query': func_args_str} if func_name == 'web_search' else {'url': func_args_str}
                    
                    function_call = {
                        'name': func_name,
                        'arguments': func_args
                    }
                
                
                # Print reasoning if present (before adding to messages)
                if reasoning_content:
                    print(json.dumps({
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": ""
                    }))
                
                # Add assistant response to messages (without reasoning in history)
                messages.append({
                    "role": "assistant",
                    "content": regular_content
                })
                
                conversation_log.append({
                    'turn': turn,
                    'type': 'assistant',
                    'content': regular_content,
                    'reasoning_content': reasoning_content,
                    'function_call': function_call
                })
                
                # Handle function calls
                if function_call:
                    func_name = function_call.get('name', '')
                    func_args = function_call.get('arguments', {})
                    
                    # Print conversation format for function call (reasoning already printed above)
                    func_call_msg = {
                        "role": "assistant", 
                        "content": regular_content,
                        "function_call": {
                            "name": func_name,
                            "arguments": json.dumps(func_args)
                        }
                    }
                    print(json.dumps(func_call_msg))
                    
                    # Execute function call
                    if func_name == 'web_search':
                        result = web_search(**func_args)
                    elif func_name == 'web_visit':
                        result = web_visit(**func_args)
                    else:
                        result = f"Unknown function: {func_name}"
                    
                    # Print conversation format for function result
                    formatted_response = result[:200] + ('...' if len(result) > 200 else '')
                    func_result_msg = {
                        "role": "function",
                        "name": func_name,
                        "content": formatted_response
                    }
                    print(json.dumps(func_result_msg))
                    
                    # Add function call to messages
                    messages.append({
                        "role": "assistant",
                        "content": regular_content,
                        "function_call": {
                            "name": func_name,
                            "arguments": json.dumps(func_args)
                        }
                    })
                    
                    # Add function result to messages
                    messages.append({
                        "role": "function",
                        "name": func_name,
                        "content": result
                    })
                    
                    tool_results.append({
                        'turn': turn,
                        'function_name': func_name,
                        'function_args': func_args,
                        'function_result': result
                    })
                    
                    # Continue conversation after function call
                    continue
                else:
                    # Final answer: content without reasoning or function calls
                    # This is the stopping condition
                    response_msg = {
                        "role": "assistant", 
                        "content": regular_content
                    }
                    print(json.dumps(response_msg))
                    
                    # No function calls and has content - this is the final answer
                    break
            
            # Extract final answer from last assistant message
            final_answer = ""
            for log_entry in reversed(conversation_log):
                if log_entry['type'] == 'assistant':
                    final_answer = log_entry['content']
                    break
            
            return {
                'success': True,
                'final_answer': final_answer,
                'conversation_log': conversation_log,
                'tool_results': tool_results,
                'total_turns': turn
            }
            
        except Exception as e:
            print(f"Error in conversation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'conversation_log': [],
                'tool_results': [],
                'total_turns': 0
            }
    
    def cleanup(self):
        """Clean up vLLM resources."""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test LLM agent with search tools")
    parser.add_argument("--query", type=str, default="What are the latest developments in AI?",
                       help="User query to test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                       help="Model name or path to use")
    parser.add_argument("--search-api", type=str, help="Search API URL")
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum conversation turns")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Configure APIs if provided
    api_config = {}
    if args.search_api:
        api_config['search'] = {
            'api_url': args.search_api + '/search',
        }
        api_config['visit'] = {
            'api_url': args.search_api + '/visit',
        }
    
    # Create agent with vLLM
    print(f"Initializing agent with model: {args.model}")
    try:
        agent = SimpleLLMAgent(
            model_name=args.model,
            api_config=api_config,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    
    # Run test conversation
    try:
        result = agent.run_conversation(args.query, max_turns=args.max_turns)
    except Exception as e:
        print(f"❌ Error during conversation: {e}")
        agent.cleanup()
        return 1
    
    
    # Cleanup
    agent.cleanup()
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)