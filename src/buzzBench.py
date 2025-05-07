import pandas as pd
import re
import csv
from datasets import load_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

ds = load_dataset("sam-paech/BuzzBench-v0.60", split="test")

rows = []

for row in ds:
    try:
        prompt = row["prompt"].strip()
        golden_answer = row["gold_answer"].strip()

        blocks = re.split(r"(?=^# )", golden_answer, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]

        for block in blocks:

            matches = re.findall(r"(Audience|Comedy writer):\s*(\d+)", block)
            scores = [int(s) for _, s in matches]
            human_score = sum(scores) / len(scores) if scores else None

            block_cleaned = re.sub(r"\*\*Funniness ratings:\*\*.*", "", block, flags=re.DOTALL | re.IGNORECASE)
            block_cleaned = re.sub(r"\*\*Ratings:\*\*.*", "", block_cleaned, flags=re.DOTALL | re.IGNORECASE)

            rows.append({
                "question": prompt,
                "question_type": "humor-eval",
                "golden_answer": block_cleaned.strip(),
                "attempted_answer": None,
                "answer_type": "no-attempt",
                "human_score": human_score
            })

    except Exception as e:
        print("Skipped row due to error:", e)
        continue

output_path = "../converted_dataset/buzzbench_converted.csv"
df = pd.DataFrame(rows)
df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

df2 = pd.read_csv(output_path)
print(df2.head())