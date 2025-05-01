# In BuzzBench, the `prompt` is the instruction given to a language model
# to generate a humor analysis—explaining how a joke works and judging its funniness.
# 
# The `judge_prompt`, on the other hand, is given to a separate model(Claude 3.5 Sonnet) acting as a judge,
# which compares the model’s response to a human-written gold answer and evaluates how well the model captured the joke and its intended effect.
#
# While the `prompt` guides generation, the `judge_prompt` guides evaluation,
# using Claude 3.5 Sonnet's scoring to assess humor understanding from both audience and writer perspectives.
# "We picked Sonnet 3.5 to act as the judge partly because it scores highest on the Judgemark leaderboard, 
# and partly because it seems least biased to favour longwinded, over-analysed, over-reaching responses." -- from the paper

import pandas as pd
import re

file_path = "/home/j4cha/moj-project/dataset/train-00000-of-00001.csv"
df = pd.read_csv(file_path, engine="python")

rows = []

for _, row in df.iterrows():
    try:
        prompt = row["prompt"]
        intro_text = row["intro_text"]
        golden_answer = row["gold_answer"]

        # Extract funniness ratings
        audience_match = re.search(r"Audience:\s*(\d+)", golden_answer)
        writer_match = re.search(r"Comedy writer:\s*(\d+)", golden_answer)

        # Compute the average human score
        if audience_match and writer_match:
            audience_score = int(audience_match.group(1))
            writer_score = int(writer_match.group(1))
            human_score = (audience_score + writer_score) / 2
        else:
            human_score = None

        # (prompt) + (intro text) into full question
        full_question = f"{str(prompt).strip()}\n\nIntro:\n{str(intro_text).strip()}"

        # Append the row
        rows.append({
            "question": full_question,
            "question_type": "humor-eval",
            "golden_answer": str(golden_answer),
            "attempted_answer": None,
            "answer_type": "no-attempt",
            "human_score": human_score
        })

    except Exception as e:
        # Skip rows with error
        print(f"Skipped by error:", e)
        continue

output_path = "../converted_dataset/buzzbench_converted.csv"
pd.DataFrame(rows).to_csv(output_path, index=False)