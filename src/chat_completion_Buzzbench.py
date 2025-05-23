import pandas as pd
import requests
import time
import json

INPUT_PATH = "../converted_dataset/buzzbench_converted.csv"
OUTPUT_PATH = "../converted_dataset/buzzbench_model_response_qwen(2.5).csv"
# OUTPUT_PATH = "../converted_dataset/buzzbench_model_response_phi-4-mini-instruct.csv"
# OUTPUT_PATH = "../converted_dataset/buzzbench_model_response_gemma-3-4b-it.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "microsoft/Phi-4-mini-instruct"
# MODEL_NAME = "google/gemma-3-4b-it"

df = pd.read_csv(INPUT_PATH)
headers = {"Content-Type": "application/json"}

attempted_answers = []

for idx, row in df.iterrows():

    prompt_text = f"""
You are a fair and thoughtful humor evaluator.

Here is the intro:

{row['question']}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 1024,
        "temperature": 0.5
    }

    try:
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        completion = res.json()["choices"][0]["message"]["content"].strip()

        print(f"[{idx}] Completion:\n{completion}\n")
        attempted_answers.append(completion)

    except Exception as e:
        print(f"[{idx}] Error: {e}")
        attempted_answers.append("")

    time.sleep(0.5)

df["attempted_answer"] = attempted_answers
df.to_csv(OUTPUT_PATH, index=False, quoting=1)