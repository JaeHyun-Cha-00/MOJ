import pandas as pd
import requests
import time
import json

INPUT_PATH = "../../converted_dataset/buzzbench_converted.csv"

# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_model_phi-4-multimodal-instruct.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_DeepSeek-R1-0528-Qwen3-8B.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_Qwen3-8B.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"


# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-8B"

df = pd.read_csv(INPUT_PATH)
headers = {"Content-Type": "application/json"}

attempted_answers = []

for idx, row in df.iterrows():

    prompt_text = f"""
You are a strict humor evaluator.

Do NOT include any tags or tokens such as <think>, <thought>, or similar.  
Do NOT include any internal monologue, reasoning process, or explanation of what you're doing.  
Your response must ONLY contain the final formatted answer.

You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Intro").  
If you mention or refer to any other characters, your answer is invalid.

Your response must exactly follow this format:

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation of the humor>

** How it Lands **  
<Brief explanation of how the humor might be received>

** Funniness Ratings **  
Audience: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)
Comedy writer: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Do NOT include anything before or after this structure.

{row['question']}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 2048,
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

# You are a fair and thoughtful humor evaluator.

# Only analyze the character whose intro heading appears below (e.g., "# Character Intro").
# Do not mention or evaluate *any* other characters, even if their names appear in the intro text.
# Your response must only include one character analysis.

# If you mention more than one character, your answer is invalid.

# At the end of your response, include the ratings section in this exact format:

# ** Funniness Ratings **
# Audience: <1–5> (<description from scale>)
# Comedy writer: <1–5> (<description from scale>)

# Use this exact structure. Do not change the heading or labels.

# Here is the full introduction text: