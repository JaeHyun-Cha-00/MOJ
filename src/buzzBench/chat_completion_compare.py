import pandas as pd
import requests
import json
import re
from tqdm import tqdm

VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

INPUT_PATH = "../../converted_dataset/pedant/fewshot/buzzbench_converted_llama3.1-8B-Instruct_fewshot_writer_with_pedant.csv"
OUTPUT_PATH = "../../converted_dataset/compared_llama_writer_fewshot_with_reasoning.csv"

def make_meta_eval_prompt(question, golden, attempted):
    return f"""
You are an evaluator.

Your task is to evaluate how well the model's answer aligns with the golden answer for a given question.

You should assign a similarity score between 0.000 and 1.000 (3 decimal places).

Scoring should be based on:
- Logical accuracy
- Coverage of key ideas
- Relevance to the question

Output format (exactly this format):
Score: <float between 0.000 and 1.000>

---

Question:
{question}

Golden Answer:
{golden}

Model's Answer:
{attempted}
""".strip()

def remove_audience_rating(text):
    """Remove the 'Audience:' line from funniness ratings block."""
    lines = text.splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("Audience:")]
    return "\n".join(cleaned)

df = pd.read_csv(INPUT_PATH)
scores = []
reasonings = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    question = str(row.get("question", "")).strip()
    golden = remove_audience_rating(str(row.get("golden_answer", "")).strip())
    attempted = str(row.get("attempted_answer", "")).strip()

    if not question or not golden or not attempted:
        scores.append(None)
        reasonings.append("")
        continue

    prompt = make_meta_eval_prompt(question, golden, attempted)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        response = requests.post(VLLM_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["text"].strip()

        # 점수 추출
        match = re.search(r"Score:\s*(0(?:\.\d{1,3})?|1(?:\.000)?)", text)
        if match:
            score = round(float(match.group(1)), 4)
        else:
            raise ValueError(f"Could not extract score from: {text}")

        scores.append(score)
        reasonings.append(text)

    except Exception as e:
        print(f"[Row {i}] Error: {e}")
        scores.append(None)
        reasonings.append("")

df["meta_eval_score"] = scores
df["meta_eval_reasoning"] = reasonings
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
