import pandas as pd
import requests
import re
import time
import json

INPUT_PATH = "../converted_dataset/buzzbench_converted.csv"
OUTPUT_PATH = "../converted_dataset/buzzbench_with_model_single_score.csv"

VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

df = pd.read_csv(INPUT_PATH)
headers = {"Content-Type": "application/json"}

attempted_answers = []
model_scores = []

for idx, row in df.iterrows():
    prompt_text = f"""

You are a humor evaluator. Give a humor score between 1 and 5 for the following intro.

Do not add explanation or repeat the score.

Respond with in this exact format:

**Score:** X

Intro:
{row['question']}

Now begin your evaluation:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": 512,
        "temperature": 0.5,
        "stop": ["</s>"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        completion = res.json()["choices"][0]["text"].strip()

        print(f"[{idx}] Completion:\n{completion}\n")

        match = re.search(r"\*\*Score:\*\*\s*([1-5])", completion)
        model_score = int(match.group(1)) if match else None

        attempted_answers.append(completion)
        model_scores.append(model_score)

        print(f"[{idx}] Model Score: {model_score}")

    except Exception as e:
        print(f"[{idx}] Error: {e}")
        attempted_answers.append("")
        model_scores.append(None)

    time.sleep(0.5)

df["attempted_answer"] = attempted_answers
df["model_score"] = model_scores
df.to_csv(OUTPUT_PATH, index=False, quoting=1)
