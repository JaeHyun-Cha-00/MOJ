import pandas as pd
import numpy as np
import re
import os

def extract_audience_score(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"Audience[:：]?\s*(\d)", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return float(score)
    return np.nan

file_paths = {
    "phi-4-mini-instruct": "../../../converted_dataset/pedant/fewshot/buzzbench_converted_phi4-mini-instruct_fewshot_audience_results_with_pedant.csv",
    "qwen2.5-7B-Instruct": "../../../converted_dataset/pedant/fewshot/buzzbench_converted_qwen2.5-7B-Instruct_fewshot_audience_results_with_pedant.csv",
    "llama-3.1-8B-Instruct": "../../../converted_dataset/pedant/fewshot/buzzbench_converted_llama3.1-8B-Instruct_fewshot_audience_with_pedant.csv",
    "DeepSeek-R1-0528-Qwen3-8B": "../../../converted_dataset/pedant/fewshot/buzzbench_converted_deepseek-qwen3-8B_fewshot_audience_results_with_pedant.csv",
    "Qwen3-8B": "../../../converted_dataset/pedant/fewshot/buzzbench_converted_qwen3-8B_fewshot_audience_results_with_pedant.csv"
}

results = []

for model_name, file_path in file_paths.items():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df["audience_score"] = df["attempted_answer"].apply(extract_audience_score)
    df["scaled_pedant"] = df["pedant_score"] * 5

    total_rows = len(df)
    na_audience = df["audience_score"].isna().sum()
    na_pedant = df["pedant_score"].isna().sum()

    valid_df = df.dropna(subset=["audience_score", "pedant_score"])
    valid_rows = len(valid_df)

    mse = np.mean((valid_df["scaled_pedant"] - valid_df["audience_score"]) ** 2)

    results.append({
        "model": model_name,
        "MSE": round(mse, 4),
        "Total Rows": total_rows,
        "Valid Rows": valid_rows,
        "Audience NaN": na_audience,
        "Pedant NaN": na_pedant
    })

df_results = pd.DataFrame(results)
print(df_results)

df_results.to_csv("fewshot_audience_mse_summary.csv", index=False)
print("\nSaved as fewshot_audience_mse_summary.csv")

#                        model     MSE  Total Rows  Valid Rows  Audience NaN  Pedant NaN
# 0        phi-4-mini-instruct  2.0962         133         133             0           0
# 1        qwen2.5-7B-Instruct  0.9676         133         133             0           0
# 2      llama-3.1-8B-Instruct  1.4272         133         133             0           0
# 3  DeepSeek-R1-0528-Qwen3-8B  1.3343         133         131             2           0
# 4                   Qwen3-8B  1.5018         133         132             1           0