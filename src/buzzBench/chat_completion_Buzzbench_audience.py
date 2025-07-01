import pandas as pd
import requests
import time
import json
from prompt import few_shot_audience_examples

INPUT_PATH = "../../converted_dataset/buzzbench_converted.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_model_phi-4-mini-audience.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "microsoft/Phi-4-mini-instruct"

headers = {"Content-Type": "application/json"}

def make_audience_prompt(text):
    return f"""
You are a strict humor evaluator representing a general audience.

Do NOT include any tags or tokens such as <think>, <thought>, or similar.  
Do NOT include any internal monologue, reasoning process, or explanation of what you're doing.  
Your response must ONLY contain the final formatted answer.

You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Intro").  
If you mention or refer to any other characters, your answer is invalid.

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How the audience might react>

** Funniness Rating (Audience) **  
Audience: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Do NOT mention comedy writers or any other characters.

Here are 5 formatted examples:

{few_shot_audience_examples}

Now evaluate:

{text}
"""

df = pd.read_csv(INPUT_PATH)
attempted_answers = []

for idx, row in df.iterrows():
    question_text = row["question"]

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": make_audience_prompt(question_text)}],
            "max_tokens": 2048,
            "temperature": 0.5
        }
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        audience_output = res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[{idx}] Audience Error: {e}")
        audience_output = ""

    attempted_answers.append(audience_output)
    print(f"[{idx}] Done")
    time.sleep(0.5)

df["attempted_answer"] = attempted_answers
df.to_csv(OUTPUT_PATH, index=False, quoting=1)