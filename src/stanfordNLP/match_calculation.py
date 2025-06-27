import pandas as pd
import re
import matplotlib.pyplot as plt
import os

output_dir = os.path.join(os.path.dirname(__file__), "bar_plot")
os.makedirs(output_dir, exist_ok=True)

model_files = {
    "llama-3.1-8B": "../../converted_dataset/shp_with_model_llama-3.1-8B-Instruct.csv",
    "qwen2.5-7B-Instruct": "../../converted_dataset/shp_with_model_qwen2.5-7B.csv",
    "phi-4-multimodel-Instruct": "../../converted_dataset/shp_with_model_phi-4-multimodal-instruct.csv"
}

def extract_answer(text):
    match = re.search(r'Answer:\s*([AB])', str(text))
    return match.group(1) if match else None

results = []

for model_name, filepath in model_files.items():
    df = pd.read_csv(filepath)
    df["predicted_label"] = df["attempted_answer"].apply(extract_answer)
    df["gold_label"] = df["golden_answer"].str.strip()

    total = len(df)
    correct = (df["predicted_label"] == df["gold_label"]).sum()
    accuracy = correct / total if total > 0 else 0
    NA = df["predicted_label"].isnull().sum()

    results.append({
        "model": model_name,
        "total": total,
        "correct": correct,
        "NA": NA,
        "accuracy": round(accuracy * 100, 2)
    })

result_df = pd.DataFrame(results)
print("=== Accuracy Table ===")
print(result_df)

plt.figure(figsize=(8, 6))
plt.bar(result_df["model"], result_df["accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy by Model on SHP")
plt.ylim(0, 100)

for i, acc in enumerate(result_df["accuracy"]):
    plt.text(i, acc + 1, f"{acc}%", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plot_path = os.path.join(output_dir, "accuracy_barplot.png")
plt.savefig(plot_path)

# === Accuracy Table ===
#                        model  total  correct   NA  accuracy
# 0               llama-3.1-8B   2000     1156   59     57.80
# 1        qwen2.5-7B-Instruct   2000     1125    2     56.25
# 2  phi-4-multimodel-Instruct   2000      647  906     32.35