import pandas as pd
import re

model_files = {
    "DeepSeek-R1-0528-Qwen3-8B": "../converted_dataset/shp_with_model_deepseek_chat.csv",
    "Llama-3.1-8B-Instruct": "../converted_dataset/shp_with_model_llama-3.1-8B-Instruct.csv"
}

accuracy_stats = {}

# Checking "Ties" in golden answer label (human preference)
def check_golden_answer(ans):
    if not isinstance(ans, str):
        return None
    ans = ans.strip()
    if ans in {"A", "B"}:
        return ans
    elif "A" in ans and "B" in ans:
        return "tie"
    return None

def extract_model_choice(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # For </think>, extract A. or B.
    if "</think>" in text:
        match = re.search(r"</think>\s*([AB])\.", text)
        if match:
            return match.group(1)

    # short A/B response
    for line in text.splitlines():
        line = line.strip().upper()
        if line in {"A", "B"}:
            return line

    return None

# Does model's answer match with the human's answer
def is_correct(gold, pred):
    if gold == "tie" or gold is None or pred is None:
        return None
    return gold == pred

for model_name, filepath in model_files.items():
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
        continue

    # extract labels
    df["gold_label"] = df["golden_answer"].apply(check_golden_answer)
    df["model_choice"] = df["attempted_answer"].apply(extract_model_choice)
    df["is_correct"] = df.apply(lambda row: is_correct(row["gold_label"], row["model_choice"]), axis=1)

    # Calculate accuracy
    total = len(df)
    evaluable = df["is_correct"].notna().sum()
    correct = df["is_correct"].sum()
    not_evaluable = total - evaluable
    accuracy = correct / evaluable

    # Store results
    accuracy_stats[model_name] = {
        "Accuracy": round(accuracy, 4),
        "Evaluable_Samples": evaluable,
        "Total_Samples": total,
        "NA": not_evaluable
    }

# Display final accuracy table
accuracy_table = pd.DataFrame.from_dict(accuracy_stats, orient="index")
print("=== Accuracy Table ===")
print(accuracy_table)

# === Accuracy Table ===
#                            Accuracy  Evaluable_Samples  Total_Samples  NA
# DeepSeek-R1-0528-Qwen3-8B    0.6327                 49             50   1
# Llama-3.1-8B-Instruct        0.4800                 50             50   0