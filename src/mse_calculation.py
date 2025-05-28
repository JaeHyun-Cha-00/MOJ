import pandas as pd
import numpy as np
import re

file_paths = {
    "phi-4-multimodal-instruct": "../converted_dataset/buzzbench_model_phi-4-multimodal-instruct.csv",
    "qwen2.5-7B-Instruct": "../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv",
    "llama-3.1-8B-Instruct": "../converted_dataset/buzzbench_with_model_llama-3.1-8B-Instruct.csv"
}

# Extract model_score from attempted_answer
def extract_model_score(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"Audience:\s*(\d(?:\.\d)?)\s*\(.*?\)\s*Comedy writer:\s*(\d(?:\.\d)?)", text)
    if match:
        audience = float(match.group(1))
        writer = float(match.group(2))
        return (audience + writer) / 2
    return np.nan

mse_dict = {}

for model_name, path in file_paths.items():
    df = pd.read_csv(path)
    df["model_score"] = df["attempted_answer"].apply(extract_model_score)
    
    # Dropping rows where there are no scores
    df = df.dropna(subset=["model_score", "human_score"])

    # Avoiding DeprecationWarning: Added [["model_score", "human_score"]]
    grouped = df.groupby("question_type")[["model_score", "human_score"]]
    mse_per_dataset = grouped.apply(lambda g: np.mean((g["model_score"] - g["human_score"]) ** 2))

    mse_dict[model_name] = mse_per_dataset.to_dict()

# Create MSE table
mse_table = pd.DataFrame.from_dict(mse_dict, orient="index")

# Columns : Different kind of Datasets
mse_table.columns = ["BuzzBench"]
print(mse_table)

# Result
#                            BuzzBench
# phi-4-multimodal-instruct   2.304104
# qwen2.5-7B-Instruct         2.075487
# llama-3.1-8B-Instruct       2.388060