import pandas as pd
from datasets import load_dataset

ds = load_dataset("lmsys/mt_bench_human_judgments", split="train")

rows_comparative_evaluation = []

for row in ds:
    try:
        # Extract conversations
        conversation_a = row["conversation_a"]
        conversation_b = row["conversation_b"]

        # Get prompt and assistant responses
        prompt = next(item["content"] for item in conversation_a if item["role"] == "user")
        answer_a = next(item["content"] for item in conversation_a if item["role"] == "assistant")
        answer_b = next(item["content"] for item in conversation_b if item["role"] == "assistant")

        # Determine winner
        winner = str(row.get("winner", "")).strip().lower()
        if winner == "model_a":
            score_a, score_b = 1, 0
        elif winner == "model_b":
            score_a, score_b = 0, 1
        else:
            score_a, score_b = 0.5, 0.5  # Tie

        # Build comparative evaluation prompt
        comparative_question = (
            "You are given two sets of conversations. Evaluate which response is better. "
            "Answer either 'conversation A' or 'conversation B'.\n\n"
            f"Conversation A: {conversation_a}\n\n"
            f"Conversation B: {conversation_b}"
        )

        rows_comparative_evaluation.append({
            "question": comparative_question,
            "question_type": "comparative-evaluation",
            "golden_answer": winner,
            "attempted_answer": "",
            "answer_type": "comparative-decision",
            "human_score": ""
        })

    except Exception as e:
        print("Skipped by error:", e)
        continue

# Save
output_path = "../converted_dataset/mtbench_converted.csv"
pd.DataFrame(rows_comparative_evaluation).to_csv(output_path, index=False)