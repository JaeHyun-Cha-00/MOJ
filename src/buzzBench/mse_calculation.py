import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# import seaborn as sns

file_paths = {
    # "phi-4-multimodal-instruct": "../converted_dataset/buzzbench_model_phi-4-multimodal-instruct.csv",
    # "qwen2.5-7B-Instruct": "../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv",
    # "llama-3.1-8B-Instruct": "../converted_dataset/buzzbench_with_model_llama-3.1-8B-Instruct.csv",
    "DeepSeek-R1-0528-Qwen3-8B": "../converted_dataset/buzzbench_model_DeepSeek-R1-0528-Qwen3-8B.csv",
    "Qwen3-8B": "../converted_dataset/buzzbench_model_Qwen3-8B.csv"
}

# Score extraction function
def extract_model_score(text):
    if not isinstance(text, str):
        return np.nan
    # Loosen the matching conditions
    match = re.search(r"Audience:\s*(\d(?:\.\d)?)", text, re.IGNORECASE)
    match2 = re.search(r"Comedy\s+writer:\s*(\d(?:\.\d)?)", text, re.IGNORECASE)
    if match and match2:
        audience = float(match.group(1))
        writer = float(match2.group(1))
        return (audience + writer) / 2
    return np.nan

mse_dict = {}
na_stats = {}
plot_data = [] 

for model_name, path in file_paths.items():
    df = pd.read_csv(path)

    # Extract score
    df["model_score"] = df["attempted_answer"].apply(extract_model_score)

    # Track NA stats
    total_rows = len(df)
    na_rows = df["model_score"].isna().sum()
    na_ratio = na_rows / total_rows
    na_stats[model_name] = {
        "NA_count": na_rows,
        "Total_rows": total_rows,
        "NA_ratio": na_ratio
    }

    # Drop rows with missing score or human label
    df = df.dropna(subset=["model_score", "human_score"])

    # Append data for scatterplot here
    for _, row in df.iterrows():
        plot_data.append({
            "Model": model_name,
            "Human Score": row["human_score"],
            "Model Score": row["model_score"]
        })

    # Calculate MSE per dataset
    grouped = df.groupby("question_type")[["model_score", "human_score"]]
    mse_per_dataset = grouped.apply(lambda g: np.mean((g["model_score"] - g["human_score"]) ** 2))
    mse_dict[model_name] = mse_per_dataset.to_dict()

# Create and rename MSE table
mse_table = pd.DataFrame.from_dict(mse_dict, orient="index")
mse_table.columns = ["BuzzBench"]

# Create NA statistics table
na_table = pd.DataFrame.from_dict(na_stats, orient="index")

# Output
print("=== MSE Table ===")
print(mse_table, end="\n\n")

print("=== Score Missing Stats ===")
print(na_table)

# Create scatterplot
# scatter_df = pd.DataFrame(plot_data)
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=scatter_df, x="Human Score", y="Model Score", hue="Model", alpha=0.2, palette="Set1")
# plt.title("Human Score vs Model Score by Model")
# plt.grid(True)
# plt.tight_layout()

# # Replace plt.show() with this
# plt.savefig("scatterplot.png", dpi=300)
# print("Scatterplot saved as scatterplot.png")

#=== MSE Table ===
#                           BuzzBench
#phi-4-multimodal-instruct   2.290146
#qwen2.5-7B-Instruct         1.722886
#llama-3.1-8B-Instruct       2.087786
#DeepSeek-R1-0528-Qwen3-8B   1.971415
#Qwen3-8B                    1.849237

#=== Score Missing Stats ===
#                           NA_count  Total_rows  NA_ratio
#phi-4-multimodal-instruct         0         144  0.000000
#qwen2.5-7B-Instruct               1         144  0.006944
#llama-3.1-8B-Instruct             6         144  0.041667
#DeepSeek-R1-0528-Qwen3-8B         8         144  0.055556
#Qwen3-8B                          6         144  0.041667