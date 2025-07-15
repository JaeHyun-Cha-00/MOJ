import pandas as pd
import re
import csv
from datasets import load_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Load BuzzBench dataset
ds = load_dataset("sam-paech/BuzzBench-v0.60", split="test")

# Audience-only prompt
audience_prompt = """For each introduction, you need to:

1. Identify if it contains one or more jokes
2. If it contains a joke:
   - Explain the intended humor: how the joke works & what makes it funny. Break the joke down **in detail**. Aim to capture all the **intended** comedic elements that are present, while including nothing extraneous.
   - Analyze how well the joke 'lands' considering:
     * The show's typical audience
     * How funny it is (be specific in your analysis!)
   - Give it funniness ratings from the perspective of the audience watching at home, using this scale:
   1: Crickets
   2: A minor exhale out the nose
   3. An audible snort
   4. LOL
   5. ROFL
3. Or, if it doesn't contain a joke:
   - Explain why it's not a joke (e.g., purely informational, etc)

Respond with your thorough, in-depth analysis in this format:

# [Character 1 name]'s intro
** Intended Humour **
...

** How it Lands ** 
...

** Funniness Ratings **
..."""

rows = []

for row in ds:
    try:
        # Clean prompt
        prompt_text = row["prompt"].strip()

        # Remove existing instruction block if present
        prompt_text = re.sub(
            r"Task: Analyzing Humor.*?examine this intro:\n*",
            "",
            prompt_text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # Remove placeholder text
        prompt_text = re.sub(
            r"# \[Character 2 name\]'s intro\s+etc\.",
            "",
            prompt_text,
            flags=re.IGNORECASE
        ).strip()

        # Clean golden answer
        golden_answer = row["gold_answer"].strip()

        # Split golden answer into character blocks
        blocks = re.split(r"(?=^# )", golden_answer, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]

        for block in blocks:
            if block.startswith("# Overall Assessment"):
                continue

            # Extract audience + comedy writer scores
            matches = re.findall(r"(Audience|Comedy writer):\s*([1-5])", block, re.IGNORECASE)
            scores = [int(score) for _, score in matches]
            human_score = sum(scores) / len(scores) if scores else None

            block_cleaned = block

            # Remove writer lines (optional)
            block_cleaned = re.sub(
                r"(?m)^\s*(Comedy writer|Writer):.*$",
                "",
                block_cleaned,
                flags=re.IGNORECASE
            )

            # Remove funniness ratings section (optional)
            block_cleaned = re.sub(
                r"\*\*(Funniness\s+Ratings|Ratings)\*\*.*",
                "",
                block_cleaned,
                flags=re.DOTALL | re.IGNORECASE
            )

            # Extract character intro title
            match_title = re.match(r"# (.+?)['’]s? intro", block, re.IGNORECASE)
            character_intro = match_title.group(0) if match_title else "[Unknown Character]"

            # Construct question with emphasis
            question_combined = f"{audience_prompt}\n\nNow, examine this intro:\n\n{prompt_text}\n\nIMPORTANT: Only analyze the following character's introduction:\n\n{character_intro}"

            rows.append({
                "question": question_combined.strip(),
                "question_type": "humor-eval-audience",
                "golden_answer": block_cleaned.strip(),
                "attempted_answer": None,
                "answer_type": "no-attempt",
                "human_score": human_score
            })

    except Exception as e:
        print("Skipped row due to error:", e)
        continue

# Save to CSV
output_path = "../../converted_dataset/buzzbench/default/buzzbench_converted_audience.csv"
df = pd.DataFrame(rows)
df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print(df.head())