import pandas as pd
import re
import csv
from datasets import load_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Load BuzzBench dataset
ds = load_dataset("sam-paech/BuzzBench-v0.60", split="test")

rows = []

for row in ds:
    try:
        # Clean prompt
        full_prompt = row["prompt"].strip()

        # Remove template placeholder from prompt
        full_prompt = re.sub(
            r"# \[Character 2 name\]'s intro\s+etc\.",
            "",
            full_prompt,
            flags=re.IGNORECASE
        ).strip()

        # Clean golden answer
        golden_answer = row["gold_answer"].strip()

        # Split golden answer into blocks
        blocks = re.split(r"(?=^# )", golden_answer, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]

        for block in blocks:

            if block.strip().startswith("# Overall Assessment"):
                continue

            # Extract human scores
            matches = re.findall(r"(Home Audience|Audience|Comedy writer|Writer):\s*([1-5])", block, re.IGNORECASE)
            scores = [int(score) for _, score in matches]
            human_score = sum(scores) / len(scores) if scores else None

            # Remove funniness ratings section
            block_cleaned = re.sub(
                r"\*\*(Funniness\s+Ratings|Ratings)\*\*.*",
                "",
                block,
                flags=re.DOTALL | re.IGNORECASE
            )

            # Extract character intro
            match_title = re.match(r"# (.+?)['’]s? intro", block, re.IGNORECASE)
            character_intro = match_title.group(0) if match_title else "[Unknown Character]"

            # Construct final question
            question_combined = f"{full_prompt}\n\n{character_intro}"

            # Append to result
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

# Save to CSV
output_path = "../../converted_dataset/buzzbench_converted.csv"
df = pd.DataFrame(rows)
df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print(df.head())
