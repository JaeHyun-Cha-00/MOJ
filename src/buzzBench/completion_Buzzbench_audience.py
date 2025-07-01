import pandas as pd
import requests
import time
import json

INPUT_PATH = "../../converted_dataset/buzzbench_converted.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_with_model_llama-3.1-8B-audience.csv"

VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

df = pd.read_csv(INPUT_PATH)
headers = {"Content-Type": "application/json"}
attempted_answers = []

for idx, row in df.iterrows():
    prompt_text = f"""
You are a fair and thoughtful humor evaluator representing a general audience.

You MUST follow the format **exactly as shown** below. 
You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Name's intro").  
If you mention or refer to any other characters, your answer is invalid.

You MUST NOT include comedy writer opinion or comedy writer ratings.

Use this format for your output:

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How an audience might evaluate it>

** Funniness Rating (Audience) **
Audience: <1–5> (must be one of: 1, 2, 3, 4, or 5 — no decimals)

Here is the full introduction text:

{row['question']}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": 1024,
        "temperature": 0.5,
        "stop": ["</s>"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        completion = res.json()["choices"][0]["text"].strip()

        print(f"[{idx}] Completion:\n{completion}\n")
        attempted_answers.append(completion)

    except Exception as e:
        print(f"[{idx}] Error: {e}")
        attempted_answers.append("")

    time.sleep(0.5)

df["attempted_answer"] = attempted_answers
df.to_csv(OUTPUT_PATH, index=False, quoting=1)