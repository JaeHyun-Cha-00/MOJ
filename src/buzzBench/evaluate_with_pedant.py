import pandas as pd
from qa_metrics.pedant import PEDANT
from tqdm import tqdm

INPUT_PATH = "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_deepseek-qwen3-8B_fewshot_audience_results.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_converted_writer_with_pedant.csv"

df = pd.read_csv(INPUT_PATH)

pedant = PEDANT()

scores = []


for i, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    attempted = row.get("attempted_answer", "")
    gold = row.get("golden_answer") or row.get("answer", "")

    try:
        if not isinstance(attempted, str) or not isinstance(gold, str) or not attempted.strip():
            print(f"[Row {i}] Skipped due to invalid input.")
            scores.append(None)
            continue

        score = pedant.get_score(gold, attempted, question)
    except Exception as e:
        print(f"[Row {i}] Grading error: {e}")
        score = None

    scores.append(score)

df["pedant_score"] = scores
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")