import pandas as pd
import requests
import json

INPUT_PATH = "../../converted_dataset/shp_converted.csv"
OUTPUT_PATH = "../../converted_dataset/shp_with_model_qwen2.5-7B.csv"
# OUTPUT_PATH = "../../converted_dataset/shp_with_model_phi-4-multimodal-instruct.csv"
# OUTPUT_PATH = "../../converted_dataset/shp_with_model_deepSeek-R1-0528-Qwen3-8B.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-8B"

df = pd.read_csv(INPUT_PATH)
results = []

for idx, row in df.iterrows():
    system_prompt = (
        "You are a helpful assistant tasked with evaluating Reddit comment quality.\n"
        "Given two user responses to a Reddit post, your job is to decide which one is more helpful "
        "based on collective Reddit user preferences.\n\n"
        "Always respond in this exact two-line format:\n"
        "Answer: A or B\n"
        "Explanation: <brief justification>\n"
        "Both lines must be present. Keep it clear and concise."
    )

    user_prompt = (
        f"{row['question']}\n\n"
        "Please follow the exact format:\n"
        "Answer: A or B\n"
        "Explanation: <brief justification>"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.5,
        "repetition_penalty": 1.2,
        "stop": ["\n\n"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        answer = res.json()["choices"][0]["message"]["content"].strip()
        print(f"[{idx}] {answer}")
        row["attempted_answer"] = answer
    except Exception as e:
        print(f"[{idx}] Failed: {e}")
        row["attempted_answer"] = ""

    results.append(row)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)