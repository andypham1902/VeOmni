
import numpy as np

def calculate_rubrics_scores(data):
    total_scores = 0
    scores = 0
    if data is not None:
        for rubric in data:
            if "grading_response" in rubric:
                grading_response = rubric["grading_response"]
                if "criteria_met" in grading_response:
                    if grading_response["criteria_met"]:
                        scores += rubric["points"]
                    if rubric["points"] > 0:
                        total_scores += rubric["points"]
        avg_score = max(0, scores) / total_scores
    else:
        avg_score = 0.0
    return avg_score
