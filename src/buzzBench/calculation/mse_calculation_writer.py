import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

file_paths = {
    "phi-4-mini-instruct": "../../../converted_dataset/buzzbench/general/phi-4-mini-instruct_general_writer.csv",
    "qwen2.5-7B-Instruct": "../../../converted_dataset/buzzbench/general/qwen2.5_7B_general_writer.csv",
    "llama-3.1-8B-Instruct": "../../../converted_dataset/buzzbench/general/llama-3.1-8B_general_writer.csv",
    "DeepSeek-R1-0528-Qwen3-8B": "../../../converted_dataset/buzzbench/general/deepseek-qwen-8B_general_writer.csv",
    "Qwen3-8B": "../../../converted_dataset/buzzbench/general/qwen3-8B_general_writer.csv"
}

def extract_writer_score(text):
    if not isinstance(text, str):
        return np.nan

    # 1. Comedy writer: 3
    match = re.search(r"Comedy\s*writer[:：]?\s*(\d)", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return float(score)

    # 2. ** Funniness Ratings **\n4
    match2 = re.search(r"\*\*\s*Funniness Rating[s]?\s*\*\*[\s\n\r]*([1-5])(?:\s*\(.*?\))?", text, re.IGNORECASE)
    if match2:
        return float(match2.group(1))

    return np.nan


mse_dict = {}
na_stats = {}
plot_data = []

for model_name, path in file_paths.items():
    df = pd.read_csv(path)

    df["writer_score"] = df["attempted_answer"].apply(extract_writer_score)
    total_rows = len(df)
    na_rows = df["writer_score"].isna().sum()
    na_ratio = na_rows / total_rows
    na_stats[model_name] = {
        "NA_count": na_rows,
        "Total_rows": total_rows,
        "NA_ratio": na_ratio
    }

    df = df.dropna(subset=["writer_score", "human_score"])
    df["model_score"] = df["writer_score"]

    for _, row in df.iterrows():
        plot_data.append({
            "Model": model_name,
            "Evaluator": "Comedy Writer",
            "Human Score": int(row["human_score"]),
            "Model Score": int(row["writer_score"])
        })

    mse = np.mean((df["model_score"] - df["human_score"]) ** 2)
    mse_dict[model_name] = {"BuzzBench": mse}

mse_table = pd.DataFrame.from_dict(mse_dict, orient="index")
print("=== MSE Table ===")
print(mse_table, end="\n\n")

na_table = pd.DataFrame.from_dict(na_stats, orient="index")
print("=== Score Missing Stats ===")
print(na_table, end="\n\n")

plot_df = pd.DataFrame(plot_data)
models = plot_df["Model"].unique()

os.makedirs("bar_plot", exist_ok=True)

for model in models:
    subset_df = plot_df[plot_df["Model"] == model]
    if subset_df.empty:
        continue

    human_scores = subset_df["Human Score"]
    model_scores = subset_df["Model Score"]
    counter = Counter(zip(human_scores, model_scores))

    heatmap_data = np.zeros((6, 6))
    for (h, m), count in counter.items():
        if 0 <= h <= 5 and 0 <= m <= 5:
            heatmap_data[h][m] = count

    vmax = int(max(counter.values())) if counter else 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        square=True,
        xticklabels=range(6),
        yticklabels=range(6)
    )
    plt.xlabel("Model Score")
    plt.ylabel("Human Score")
    plt.title(f"Heatmap: {model} (Comedy Writer)")

    filename = f"bar_plot/heatmap_writer_{model.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# === MSE Table ===
#                            BuzzBench
# phi-4-mini-instruct         2.173358
# qwen2.5-7B-Instruct         1.552920
# llama-3.1-8B-Instruct       1.881387
# DeepSeek-R1-0528-Qwen3-8B   1.825926
# Qwen3-8B                    1.887868

# === Score Missing Stats ===
#                            NA_count  Total_rows  NA_ratio
# phi-4-mini-instruct               0         143  0.000000
# qwen2.5-7B-Instruct               0         143  0.000000
# llama-3.1-8B-Instruct             0         143  0.000000
# DeepSeek-R1-0528-Qwen3-8B         3         143  0.020979
# Qwen3-8B                          2         143  0.013986