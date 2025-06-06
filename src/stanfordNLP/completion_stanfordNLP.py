import pandas as pd
import requests
import json

INPUT_PATH = "../../converted_dataset/shp_converted.csv"
OUTPUT_PATH = "../../converted_dataset/shp_with_model_llama-3.1-8B-Instruct.csv"

VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HEADERS = {"Content-Type": "application/json"}

df = pd.read_csv(INPUT_PATH)
results = []

for idx, row in df.iterrows():
    prompt = (
        "This is a Reddit post with two user responses (A and B).\n"
        "We aim to identify which response is more helpful based on collective Reddit user preferences.\n\n"
        f"{row['question']}\n\n"
        "Please answer only with 'A' or 'B', followed by a brief justification.\n"
        "Answer:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.5,
        "stop": ["</s>"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        answer = res.json()["choices"][0]["text"].strip()
        print(f"[{idx}] Answer: {answer}")
        row["attempted_answer"] = answer
    except Exception as e:
        print(f"[{idx}] Failed: {e}")
        row["attempted_answer"] = ""

    results.append(row)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
