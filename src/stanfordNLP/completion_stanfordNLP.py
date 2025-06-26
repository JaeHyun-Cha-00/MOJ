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
        "Your task is to choose which response is more helpful, based on Reddit user preferences.\n\n"
        f"{row['question']}\n\n"
        "Respond in exactly this format:\n"
        "Answer: A or B\n"
        "Explanation: <brief justification>\n"
        "Both 'Answer' and 'Explanation' must be included on separate lines.\n"
        "Avoid repeating. Be concise and clear.\n"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.5,
        "repetition_penalty": 1.2,
        "stop": ["\n\n"]
    }

    try:
        res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
        res.raise_for_status()
        answer = res.json().get("choices", [{}])[0].get("text", "").strip()

        print(f"[{idx}] {answer}")
        row["attempted_answer"] = answer
    except Exception as e:
        print(f"[{idx}] Failed: {e}")
        row["attempted_answer"] = ""

    results.append(row)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
