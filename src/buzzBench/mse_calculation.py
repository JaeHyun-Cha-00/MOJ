import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

file_paths = {
    "phi-4-multimodal-instruct": "../../converted_dataset/buzzbench_model_phi-4-multimodal-instruct.csv",
    "qwen2.5-7B-Instruct": "../../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv",
    "llama-3.1-8B-Instruct": "../../converted_dataset/buzzbench_with_model_llama-3.1-8B-Instruct.csv",
    "DeepSeek-R1-0528-Qwen3-8B": "../../converted_dataset/buzzbench_model_DeepSeek-R1-0528-Qwen3-8B.csv",
    "Qwen3-8B": "../../converted_dataset/buzzbench_model_Qwen3-8B.csv"
}

def extract_scores(text):
    if not isinstance(text, str):
        return np.nan, np.nan
    match_aud = re.search(r"Audience:\s*(\d+)", text, re.IGNORECASE)
    match_wr = re.search(r"Comedy\s+writer:\s*(\d+)", text, re.IGNORECASE)
    if not (match_aud and match_wr):
        return np.nan, np.nan

    audience = int(match_aud.group(1))
    writer = int(match_wr.group(1))

    if 1 <= audience <= 5 and 1 <= writer <= 5:
        return float(audience), float(writer)
    else:
        return np.nan, np.nan

mse_dict = {}
na_stats = {}
plot_data = []

for model_name, path in file_paths.items():
    df = pd.read_csv(path)

    # Extracting scores
    df[["audience_score", "writer_score"]] = df["attempted_answer"].apply(
        lambda x: pd.Series(extract_scores(x))
    )

    # Mean Calculation
    df["model_score"] = df[["audience_score", "writer_score"]].mean(axis=1)

    # Counting NAs
    total_rows = len(df)
    na_rows = df["model_score"].isna().sum()
    na_ratio = na_rows / total_rows
    na_stats[model_name] = {
        "NA_count": na_rows,
        "Total_rows": total_rows,
        "NA_ratio": na_ratio
    }

    # Removing rows that is NA
    df = df.dropna(subset=["audience_score", "writer_score", "human_score"])

    for _, row in df.iterrows():
        plot_data.append({
            "Model": model_name,
            "Evaluator": "Audience",
            "Human Score": int(row["human_score"]),
            "Model Score": int(row["audience_score"])
        })
        plot_data.append({
            "Model": model_name,
            "Evaluator": "Comedy Writer",
            "Human Score": int(row["human_score"]),
            "Model Score": int(row["writer_score"])
        })

    grouped = df.groupby("question_type")[["model_score", "human_score"]]
    mse_per_dataset = grouped.apply(lambda g: np.mean((g["model_score"] - g["human_score"]) ** 2))
    mse_dict[model_name] = mse_per_dataset.to_dict()

# MSE Table
mse_table = pd.DataFrame.from_dict(mse_dict, orient="index")
mse_table.columns = ["BuzzBench"]
print("=== MSE Table ===")
print(mse_table, end="\n\n")

# NA Table
na_table = pd.DataFrame.from_dict(na_stats, orient="index")
print("=== Score Missing Stats ===")
print(na_table, end="\n\n")

# 3D Bar Plot
plot_df = pd.DataFrame(plot_data)
models = plot_df["Model"].unique()
evaluators = ["Audience", "Comedy Writer"]

for model in models:
    for evaluator in evaluators:
        subset_df = plot_df[
            (plot_df["Model"] == model) & (plot_df["Evaluator"] == evaluator)
        ]
        if subset_df.empty:
            continue

        human_scores = subset_df["Human Score"]
        model_scores = subset_df["Model Score"]
        counter = Counter(zip(human_scores, model_scores))

        # x, y, z-axis
        xs, ys, zs = [], [], []
        for (x, y), count in counter.items():
            xs.append(x)
            ys.append(y)
            zs.append(int(count))

        norm = colors.Normalize(vmin=1, vmax=5)
        cmap = plt.colormaps["coolwarm"]
        bar_colors = [cmap(norm(score)) for score in ys]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        dx = dy = 0.5
        dz = zs

        ax.bar3d(xs, ys, np.zeros_like(zs), dx, dy, dz, color=bar_colors, alpha=0.3)

        ax.set_xlabel('Human Score')
        ax.set_ylabel('Model Score')
        ax.set_zlabel('Count')
        ax.set_title(f'3D Score: {model} - {evaluator}')
        ax.set_xlim(0, 5.8)
        ax.set_ylim(-0.8, 6)
        ax.set_xticks(range(0, 6))
        ax.set_yticks(range(0, 6))
        ax.set_zlim(0, max(zs) + 1)
        ax.set_zticks(range(0, max(zs) + 2, 4))

        plt.tight_layout()
        filename = f"3d_barplot_{model.replace('/', '_')}_{evaluator.replace(' ', '_')}.png"
        
        # Save in Image
        plt.savefig(filename, dpi=300)
        plt.close()

# === MSE Table ===
#                            BuzzBench
# phi-4-multimodal-instruct   1.886861
# qwen2.5-7B-Instruct         1.757299
# llama-3.1-8B-Instruct       2.358456
# DeepSeek-R1-0528-Qwen3-8B   1.849237
# Qwen3-8B                    1.722015

# === Score Missing Stats ===
#                            NA_count  Total_rows  NA_ratio
# phi-4-multimodal-instruct         0         144  0.000000
# qwen2.5-7B-Instruct               0         144  0.000000
# llama-3.1-8B-Instruct             1         144  0.006944
# DeepSeek-R1-0528-Qwen3-8B         7         144  0.048611
# Qwen3-8B                          3         144  0.020833