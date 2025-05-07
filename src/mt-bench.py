import pandas as pd
from datasets import load_dataset

ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")

rows_comparative_evaluation = []

for row in ds:
    try:
        # Extract conversation components
        conversation_a = row["conversation_a"]
        conversation_b = row["conversation_b"]

        # Get prompt and assistant answers
        prompt = next(item["content"] for item in conversation_a if item["role"] == "user")
        answer_a = next(item["content"] for item in conversation_a if item["role"] == "assistant")
        answer_b = next(item["content"] for item in conversation_b if item["role"] == "assistant")

        # Determine winner
        winner = str(row.get("winner", "")).strip().lower()
        if winner == "model_a":
            golden_answer = "conversation A"
            human_score = 1.0
        elif winner == "model_b":
            golden_answer = "conversation B"
            human_score = 1.0
        else:
            golden_answer = "tie"
            human_score = 0.5

        # Build clean comparison-style question (no duplicate prompt)
        comparative_question = (
            "You are given a user prompt and two candidate answers. Evaluate which response is better. "
            "Answer either 'conversation A' or 'conversation B'.\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Conversation A:\nAssistant: {answer_a.strip()}\n\n"
            f"Conversation B:\nAssistant: {answer_b.strip()}"
        )

        # Append to list
        rows_comparative_evaluation.append({
            "question": comparative_question,
            "question_type": "comparative-evaluation",
            "golden_answer": golden_answer,
            "attempted_answer": None,
            "answer_type": "comparative-decision",
            "human_score": human_score
        })

    except Exception as e:
        print("Skipped by error:", e)
        continue

# Save the result
output_path = "../converted_dataset/mtbench_converted.csv"
pd.DataFrame(rows_comparative_evaluation).to_csv(output_path, index=False)