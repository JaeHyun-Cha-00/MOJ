import pandas as pd
import ast

file_path = "/home/j4cha/moj-project/dataset/human-00000-of-00001-25f4910818759289-1.csv"
df = pd.read_csv(file_path, engine="python")

rows_comparative_evaluation = []
rows_assistance_response_quality = []

# Iterate over each row in the dataset
for _, row in df.iterrows():
    try:
        # Parse the conversation fields
        conversation_a = list(ast.literal_eval(row["conversation_a"]))
        conversation_b = list(ast.literal_eval(row["conversation_b"]))

        # Extract the user prompt and assistant responses from conversations
        prompt = next(content for content, role in conversation_a if role == "user")
        answer_a = next(content for content, role in conversation_a if role == "assistant")
        answer_b = next(content for content, role in conversation_b if role == "assistant")

        # Determine the winner and assign scores
        winner = str(row.get("winner", "")).strip().lower()
        if winner == "model_a":
            score_a, score_b = 1, 0
        elif winner == "model_b":
            score_a, score_b = 0, 1
        else:
            score_a, score_b = 0.5, 0.5  # Tie case

        # full_question = f"{str(conversation_a[:-1])}" # expect the LLM to generate like the last output of the "assistant"
        comparative_question = f"You are given two sets of conversation. Evaluate which response is better. Answer either conversation A or conversation B.: \n\nConversation A: {conversation_a}\nConversation B: {conversation_b}"

        # User ABC
        # Asssistant DEF
        # User GHI
        # Assistant JK

        # ABC, DEF -> one data point
        # GHI, JK -> one data point
        # -> this could be problemabtic because the winner column is based on "ABC + DEF + GHI + JK".
        # -> if we split up the two turn converstaion into two data points, then the winnder column may not be correct.
        # "ABC + DEF" -> we don't have the correct preference for just this conversation turn.

        rows_comparative_evaluation.append({
            "question": comparative_question,
            "question_type": "comparative-evaluation",
            "golden_answer": winner,
            "attempted_answer": "",
            "answer_type": "comparative-decision (sth like that...)",
            "human_score": "",
        })

        # # Append Model A's response
        # rows_assistance_response_quality.append({
        #     "question": full_question,
        #     "question_type": "comparative-evaluation",
        #     "golden_answer": None,
        #     "attempted_answer": str(answer_a),
        #     "answer_type": "open-ended QA",
        #     "human_score": score_a
        # })

        # # Append Model B's response
        # rows_assistance_response_quality.append({
        #     "question": full_question,
        #     "question_type": "comparative-evaluation",
        #     "golden_answer": None,
        #     "attempted_answer": str(answer_b),
        #     "answer_type": "open-ended QA",
        #     "human_score": score_b
        # })

    except Exception as e:
        print("Skipped by error:", e)
        continue

output_path = "../converted_dataset/mtbench_converted.csv"
pd.DataFrame(rows_comparative_evaluation).to_csv(output_path, index=False)