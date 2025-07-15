import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

file_paths = {
    "phi-4-mini-instruct": "../../../converted_dataset/buzzbench/general/phi-4-mini-instruct_general_audience.csv",
    "qwen2.5-7B-Instruct": "../../../converted_dataset/buzzbench/general/qwen2.5_7b_general_audience.csv",
    "llama-3.1-8B-Instruct": "../../../converted_dataset/buzzbench/general/llama-3.1-8B_general_audience.csv",
    "DeepSeek-R1-0528-Qwen3-8B": "../../../converted_dataset/buzzbench/general/deepseek-qwen-8B_general_audience.csv",
    "Qwen3-8B": "../../../converted_dataset/buzzbench/general/qwen3-8B_general_audience.csv"
}

def extract_audience_score(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"Audience[:：]?\s*(\d)", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return float(score)
    return np.nan

mse_dict = {}
na_stats = {}
plot_data = []

for model_name, path in file_paths.items():
    df = pd.read_csv(path)

    df["audience_score"] = df["attempted_answer"].apply(extract_audience_score)
    total_rows = len(df)
    na_rows = df["audience_score"].isna().sum()
    na_ratio = na_rows / total_rows
    na_stats[model_name] = {
        "NA_count": na_rows,
        "Total_rows": total_rows,
        "NA_ratio": na_ratio
    }

    df = df.dropna(subset=["audience_score", "human_score"])
    df["model_score"] = df["audience_score"]

    for _, row in df.iterrows():
        plot_data.append({
            "Model": model_name,
            "Evaluator": "Audience",
            "Human Score": int(row["human_score"]),
            "Model Score": int(row["audience_score"])
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

    heatmap_data = heatmap_data[::-1]
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
        yticklabels=range(5, -1, -1)
    )
    plt.xlabel("Model Score")
    plt.ylabel("Human Score")
    plt.title(f"Heatmap: {model} (Audience)")

    filename = f"../heatmap/buzzbench/general/heatmap_audience_{model.replace('/', '_')}_general.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# === MSE Table ===
#                            BuzzBench
# phi-4-mini-instruct         2.050000
# qwen2.5-7B-Instruct         1.640511
# llama-3.1-8B-Instruct       1.582117
# DeepSeek-R1-0528-Qwen3-8B   1.577206
# Qwen3-8B                    1.583333

# === Score Missing Stats ===
#                            NA_count  Total_rows  NA_ratio
# phi-4-mini-instruct               2         143  0.013986
# qwen2.5-7B-Instruct               0         143  0.000000
# llama-3.1-8B-Instruct             0         143  0.000000
# DeepSeek-R1-0528-Qwen3-8B         1         143  0.006993
# Qwen3-8B                          3         143  0.020979