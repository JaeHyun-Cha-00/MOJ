import pandas as pd
from qa_metrics.pedant import PEDANT
from tqdm import tqdm
import os

# input_files = [
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_deepseek-qwen3-8B_fewshot_audience_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_deepseek-qwen3-8B_fewshot_writer_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_llama3.1-8B-Instruct_fewshot_audience.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_llama3.1-8B-Instruct_fewshot_writer.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_phi4-mini-instruct_fewshot_audience_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_phi4-mini-instruct_fewshot_writer_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_qwen2.5-7B_fewshot_writer_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_qwen2.5-7B-Instruct_fewshot_audience_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_qwen3-8B_fewshot_audience_results.csv",
#     "../../converted_dataset/buzzbench/fewshot/buzzbench_converted_qwen3-8B_fewshot_writer_results.csv"
# ]

input_files = [
    "../../converted_dataset/buzzbench/general/deepseek-qwen-8B_general_audience.csv",
    "../../converted_dataset/buzzbench/general/deepseek-qwen-8B_general_writer.csv",
    "../../converted_dataset/buzzbench/general/llama-3.1-8B_general_audience.csv",
    "../../converted_dataset/buzzbench/general/llama-3.1-8B_general_writer.csv",
    "../../converted_dataset/buzzbench/general/phi-4-mini-instruct_general_audience.csv",
    "../../converted_dataset/buzzbench/general/phi-4-mini-instruct_general_writer.csv",
    "../../converted_dataset/buzzbench/general/qwen2.5_7b_general_audience.csv",
    "../../converted_dataset/buzzbench/general/qwen2.5_7B_general_writer.csv",
    "../../converted_dataset/buzzbench/general/qwen3-8B_general_audience.csv",
    "../../converted_dataset/buzzbench/general/qwen3-8B_general_writer.csv"
]

pedant = PEDANT()

for INPUT_PATH in input_files:
    filename = os.path.basename(INPUT_PATH)
    name, ext = os.path.splitext(filename)
    OUTPUT_PATH = os.path.join("../../converted_dataset", f"{name}_with_pedant{ext}")

    print(f"📂 Processing: {filename}")

    df = pd.read_csv(INPUT_PATH)
    scores = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {filename}"):
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
    print(f"Saved to {OUTPUT_PATH}\n")