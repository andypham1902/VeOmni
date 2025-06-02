# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def extract_boxed_text(text, extract_mc_letter=True):
    """
    Extract content from LaTeX \boxed{} command.
    
    Args:
        text: The text to extract boxed content from
        extract_mc_letter: If True, extract just the letter from multiple choice answers
                          like \boxed{A}, \boxed{A. 123}, \boxed{(A) 123}
    
    Returns:
        The extracted content or empty string if no match
    """
    # Fix the pattern to include the backslash
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    
    if not matches:
        return ""
    
    for match in matches[::-1]:
        if match == "":
            continue
            
        if extract_mc_letter:
            # Try to extract just the letter for multiple choice
            # Match patterns like: A, A., (A), A:, A), etc.
            mc_pattern = r'^([A-Z])[\.:\)\s]|^\(([A-Z])\)|^([A-Z])$'
            mc_match = re.search(mc_pattern, match.strip())
            if mc_match:
                # Return the first non-None group
                return next((g for g in mc_match.groups() if g is not None), "")
        
        return match
    
    return ""

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        final_answer = extract_boxed_text(solution_str)
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer.lower() == ground_truth.lower():
            return score
        else:
            return format_score

if __name__ == "__main__":
    # Example usage
    solution_str = "\\boxed{(A) 123}"
    ground_truth = "A"
    print(compute_score(solution_str, ground_truth, method='strict'))
