import pandas as pd
import re

model_files = {
    "llama-3.1-8B": "../../converted_dataset/shp_with_model_llama-3.1-8B-Instruct.csv"
    ,"qwen2.5-7B-Instruct": "../../converted_dataset/shp_with_model_qwen2.5-7B.csv"
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
print(result_df)

#                  model  total  correct  NA  accuracy
# 0         llama-3.1-8B   2000     1156  59     57.80
# 1  qwen2.5-7B-Instruct   2000     1125   2     56.25