import pandas as pd
from datasets import load_dataset

# Load SHP dataset
ds = load_dataset("stanfordnlp/SHP", split="train").select(range(50))

rows = []

for row in ds:
    try:
        # Extract fields
        history = row["history"]
        human_ref_A = row["human_ref_A"]
        human_ref_B = row["human_ref_B"]
        score_A = row["score_A"]
        score_B = row["score_B"]
        label = row["labels"]
        score_ratio = row["score_ratio"]

        # Determine which candidate is preferred
        if label == 0:
            golden_answer = "B"
            score_b = 1.0
            score_a = 1.0 / score_ratio
        elif label == 1:
            golden_answer = "A"
            score_a = 1.0
            score_b = 1.0 / score_ratio
        else:
            golden_answer = "tie"
            score_a = score_b = 0.5

        # Format human_score to single string
        score_str = f"A:{round(score_a, 4)}, B:{round(score_b, 4)}"

        # Build single-row question with both candidates
        full_question = (
            f"History:\n{str(history).strip()}\n\n"
            f"Candidate A:\n{str(human_ref_A).strip()}\n\n"
            f"Candidate B:\n{str(human_ref_B).strip()}\n\n"
            "Question: Which candidate provides a better response?"
        )

        # Append row
        rows.append({
            "question": full_question,
            "question_type": "preference-eval",
            "golden_answer": golden_answer,
            "attempted_answer": None,
            "answer_type": "comment-ranking",
            "human_score": score_str
        })

    except Exception as e:
        print(f"Skipped by error: {e}")
        continue

output_path = "../../converted_dataset/shp_converted.csv"
pd.DataFrame(rows).to_csv(output_path, index=False)