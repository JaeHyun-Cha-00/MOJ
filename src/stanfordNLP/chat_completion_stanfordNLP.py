import pandas as pd
import requests
import json

INPUT_PATH = "../../converted_dataset/shp_converted.csv"
OUTPUT_PATH = "../../converted_dataset/shp_with_model_deepseek_chat.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
HEADERS = {"Content-Type": "application/json"}

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-8B"

df = pd.read_csv(INPUT_PATH)
results = []

for idx, row in df.iterrows():
    system_prompt = (
        "You are a helpful assistant tasked with evaluating Reddit comment quality.\n"
        "Given two user responses to a Reddit post, identify which one is more helpful "
        "based on collective Reddit user preferences.\n"
        "Respond only with 'A' or 'B', followed by a brief justification."
    )

    user_prompt = (
        f"{row['question']}\n\n"
        "Please answer only with 'A' or 'B', followed by a brief justification.\n"
        "Answer:"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.5,
        "stop": ["</s>"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        answer = res.json()["choices"][0]["message"]["content"].strip()
        print(f"[{idx}] Answer: {answer}")
        row["attempted_answer"] = answer
    except Exception as e:
        print(f"[{idx}] Failed: {e}")
        row["attempted_answer"] = ""

    results.append(row)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)