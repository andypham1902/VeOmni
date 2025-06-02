import pandas as pd
from openai import OpenAI, AzureOpenAI
import os
import re
from dotenv import load_dotenv
from veomni.utils.reward_score.healthbench.avg_healthbench import calculate_rubrics_scores
# Initialize pandarallel
import json

load_dotenv()


client = AzureOpenAI(
    api_key=os.getenv("A_API_KEY_41"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("LLM_BASE_ENDPOINT_41")
)
model_name = os.getenv("DEPLOYMENT_NAME_41")


INSTRUCTION = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def convo_to_str(prompt, completion):
    convo_with_response = prompt # Create a copy of the list
    if hasattr(convo_with_response, 'tolist'):  # Check if it's a NumPy array
        convo_with_response = convo_with_response.tolist()
    else:
        convo_with_response = list(convo_with_response)
    convo_with_response.append(
        dict(content=completion, role="assistant")
    )
    prompt_str = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in convo_with_response]
    )
    return prompt_str


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


def get_openai_response(conversation, rubrics):
    rubrics_with_result = []
    for rubric in rubrics:
        rubric_with_result = rubric.copy()
        rubric_item = rubric["criterion"]
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": INSTRUCTION.replace("<<conversation>>", conversation).replace("<<rubric_item>>", rubric_item)},
                    ],
                    temperature=0.5,
                    # top_p=0.8,
                    max_tokens=2048,
                )
                generated = response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                return None
            if generated is not None:
                grading_response_dict = parse_json_to_dict(generated)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        break
            print("Grading failed due to bad JSON output, retrying...")
        rubric_with_result["grading_response"] = grading_response_dict
        rubrics_with_result.append(rubric_with_result)
    # print(rubrics_with_result)
    return rubrics_with_result

# def compute_score(solution_str, prompt_id):
#     with open('/home/andrew/VeOmni/veomni/utils/reward_score/healthbench/scoring.json', 'r') as f:
#         scoring = json.load(f)
#     data = scoring[prompt_id]
#     conversation = convo_to_str(data['prompt'], solution_str)
#     # print(conversation)
#     result = get_openai_response(conversation, data['rubrics'])
#     return calculate_rubrics_scores(result)

def compute_score(solution_str, prompt_id):
    return 0.5


if __name__ == "__main__":
    solution_str = """
### **3-Month Plan for Mild Postpartum Depression**  \n**Current Status**: 6 weeks postpartum, no harmful thoughts, good support system, open to talk therapy.  \n\n---\n\n### **How Talk Therapy Might Help**  \n1. **Emotional Processing**: Safely explore feelings of guilt, anxiety, or sadness without judgment.  \n2. **Cognitive Restructuring**: Challenge negative thought patterns (e.g., "I’m a bad mom") and replace them with balanced perspectives.  \n3. **Skill-Building**: Learn coping strategies for stress, sleep deprivation, and parenting challenges.  \n4. **Validation**: Normalize your experience and reduce isolation by sharing with a trained therapist.  \n\n---\n\n### **3-Month Plan**  \n\n#### **Month 1: Establish Routine & Build Foundation**  \n**Weeks 1–4**:  \n- **Talk Therapy**:  \n  - Schedule 1–2 weekly sessions to set goals (e.g., managing anxiety around baby’s sleep, reconnecting with self-care).  \n  - Share specific concerns (e.g., "I feel overwhelmed when my baby cries").  \n- **Daily Practices**:  \n  - **Self-Care**: 20-minute walks, hydrate, eat balanced meals.  \n  - **Connection**: Call a friend or partner daily; use a "gratitude list" for small wins.  \n  - **Sleep**: Negotiate 1–2 hours of uninterrupted rest nightly (e.g., swap nighttime feeding for rocking).  \n\n**Weeks 5–8**:  \n- **Talk Therapy**:  \n  - Identify 1–2 recurring stressors (e.g., loneliness) and practice grounding techniques (e.g., breathing exercises).  \n  - Introduce "cognitive restructuring" homework (e.g., journal negative thoughts + rebuttals).  \n- **Daily Practices**:  \n  - **Mindfulness**: 5-minute meditation (apps like Calm or Headspace).  \n  - **Support**: Join a postpartum group (virtual or in-person) once a week.  \n\n**Weeks 9–12**:  \n- **Talk Therapy**:  \n  - Reframe challenges as opportunities for growth (e.g., "This sleepless night teaches me my baby’s needs").  \n  - Address residual self-criticism (e.g., "It’s okay to need help").  \n- **Daily Practices**:  \n  - **Routine**: Block 1 hour/day for "me time" (e.g., reading, bath).  \n  - **Progress Check**: Reflect on 1 thing you’re proudest of each day.  \n\n---\n\n#### **Month 2: Deepen Coping Strategies**  \n- **Talk Therapy**:  \n  - Role-play scenarios (e.g., handling toddler tantrums).  \n  - Explore root causes (e.g., past trauma, hormonal shifts).  \n- **Daily Practices**:  \n  - **Problem-Solving**: Break down overwhelming tasks into 5-minute steps (e.g., "I need to clean the kitchen → wipe counters first").  \n  - **Movement**: Gentle yoga or stretching to reduce physical tension.  \n\n---\n\n#### **Month 3: Consolidate Progress & Plan Ahead**  \n- **Talk Therapy**:  \n  - Review progress; adjust strategies as needed.  \n  - Prepare for future stressors (e.g., potty training, work transitions).  \n- **Daily Practices**:  \n  - **Sustainability**: Assign supportive roles to family (e.g., "Mom needs quiet time after 8 PM").  \n  - **Long-Term Goals**: Draft a "wellness plan" for busy seasons (e.g., monthly check-ins with your therapist).  \n\n---\n\n### **Practical Steps to Start Talk Therapy**  \n1. **Find a Therapist**:  \n   - Use platforms like Psychology Today or ask your OB/GYN for recommendations.  \n   - Prioritize therapists experienced with perinatal mental health (look for "PPD" in their specialties).  \n2. **Prepare for Sessions**:  \n   - Journal ahead to note triggers, emotions, or questions.  \n   - Bring a loved one for support (if desired).  \n3. **Insurance/Accessibility**:  \n   - Check if your insurance covers therapy; explore sliding-scale clinics if needed.  \n\n---\n\n### **Key Reminders**  \n- **Progress is gradual**: Celebrate small wins (e.g., crying less over a minor setback).  \n- **Adjust as needed**: If certain strategies feel ineffective, revisit your therapist together.  \n- **You’re not alone**: Mild PPD is common; seek help without shame.  \n\nThis plan balances professional support with actionable self-care, leveraging your existing strengths and support system. Consistency over 3 months can foster lasting resilience. 🌸
""".strip()
    import time
    start_time = time.time()
    print(compute_score(solution_str, "0"))
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    print(compute_score(solution_str, "0"))
    print("--- %s seconds ---" % (time.time() - start_time))
