import pandas as pd
import re
import csv
from datasets import load_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

ds = load_dataset("sam-paech/BuzzBench-v0.60", split="test")
ds = ds.shuffle(seed=42).select(range(10))

rows = []

for row in ds:
    try:
        full_prompt = row["prompt"].strip()
        golden_answer = row["gold_answer"].strip()

        blocks = re.split(r"(?=^# )", golden_answer, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]

        for block in blocks:
            matches = re.findall(r"(Home Audience|Audience|Comedy writer|Writer):\s*([1-5])", block, re.IGNORECASE)
            scores = [int(score) for _, score in matches]
            human_score = sum(scores) / len(scores) if scores else None

            block_cleaned = re.sub(r"\*\*(Funniness\s+Ratings|Ratings)\*\*.*", "", block, flags=re.DOTALL | re.IGNORECASE)

            match_title = re.match(r"# (.+?)'s intro", block)
            character_intro = match_title.group(0) if match_title else "[Unknown Character]"

            question_combined = f"{full_prompt}\n\n{character_intro}"

            rows.append({
                "question": question_combined.strip(),
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

print(df.head())