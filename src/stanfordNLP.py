import pandas as pd
from datasets import load_dataset

# Load SHP dataset
ds = load_dataset("stanfordnlp/SHP", split="train")

rows = []

for row in ds:
    try:
        # Extract fields
        history = row["history"]
        human_ref_A = row["human_ref_A"]
        human_ref_B = row["human_ref_B"]
        score_A = row["score_A"]
        score_B = row["score_B"]

        # Determine golden_answer
        if score_A > score_B:
            golden_answer = human_ref_A
        elif score_B > score_A:
            golden_answer = human_ref_B
        else:
            golden_answer = human_ref_A, human_ref_B  # score_A == score_B

        # Build full question
        full_question = (
            f"History:\n{str(history).strip()}\n\n"
            f"Candidate A:\n{str(human_ref_A).strip()}\n\n"
            f"Candidate B:\n{str(human_ref_B).strip()}\n\n"
            "Question: Which candidate provides a better response?"
        )

        # Append the row
        rows.append({
            "question": full_question,
            "question_type": "preference-eval",
            "golden_answer": golden_answer,
            "attempted_answer": None,
            "answer_type": "multiple-choice",
            "human_score": None                      
            
            # How to calculate human_score? 
            # 1. 5 * {(human_score_A or B) / (human_score_A + human_score_B)} 
            # 2. Just use label? Using 0 and 1 to show who won?
        })

    except Exception as e:
        print(f"Skipped by error: {e}")
        continue

output_path = "../converted_dataset/shp_converted.csv"
pd.DataFrame(rows).to_csv(output_path, index=False)
