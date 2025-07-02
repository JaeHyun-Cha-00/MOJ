import pandas as pd
import requests
import time
import json
from prompt import few_shot_audience_examples

INPUT_PATH = "../../converted_dataset/buzzbench_converted_without_fewshot.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_with_model_llama-3.1-8B-audience.csv"

VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

df = pd.read_csv(INPUT_PATH)
headers = {"Content-Type": "application/json"}
attempted_answers = []

for idx, row in df.iterrows():
    # 마지막 캐릭터 이름 추출
    intro_lines = [line for line in row["question"].split("\n") if line.strip().startswith("# ")]
    target_line = intro_lines[-1] if intro_lines else "# Unknown's intro"
    target_character = target_line.replace("#", "").replace("'s intro", "").strip()

    prompt_text = f"""
You are a strict humor evaluator representing a general audience.

Your response must ONLY contain the final formatted answer.

ONLY evaluate the one character whose intro appears **after the heading** like:

# <Character Name>'s intro

DO NOT analyze or mention any other character. If you do, your answer is INVALID.

Target character: **{target_character}**

Here are 5 examples:

{few_shot_audience_examples}

---

Now evaluate only the character below. Use only the character intro marked by `# {target_character}'s intro`:

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
