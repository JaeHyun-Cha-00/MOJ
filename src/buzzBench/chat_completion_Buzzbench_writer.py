import pandas as pd
import requests
import time
import json
from prompt import few_shot_writer_examples

INPUT_PATH = "../../converted_dataset/buzzbench_converted.csv"

# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_writer_only_phi-4-mini.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_DeepSeek-R1-0528-Qwen3-8B.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_Qwen3-8B.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-8B"

headers = {"Content-Type": "application/json"}

df = pd.read_csv(INPUT_PATH)
attempted_answers = []

def make_writer_prompt(text):
    return f"""
You are a strict humor evaluator representing a professional comedy writer.

You MUST follow the format **exactly as shown** below.  
You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Name's intro").  
If you mention or refer to any other characters, your answer is invalid.

You MUST NOT include audience opinions or audience ratings.

Use this format for your output:

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How a comedy writer might evaluate it>

** Funniness Rating (Comedy Writer) **  
Comedy writer: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

DO NOT DEVIATE FROM THIS FORMAT.

Here are 5 formatted examples:

{few_shot_writer_examples}

Now evaluate:

{text}
"""


for idx, row in df.iterrows():
    question_text = row["question"]

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": make_writer_prompt(question_text)}],
            "max_tokens": 2048,
            "temperature": 0.5
        }
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        writer_output = res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[{idx}] Writer Error: {e}")
        writer_output = ""

    time.sleep(0.5)
    attempted_answers.append(writer_output)
    print(f"[{idx}] Done")

df["attempted_answer"] = attempted_answers
df.to_csv(OUTPUT_PATH, index=False, quoting=1)