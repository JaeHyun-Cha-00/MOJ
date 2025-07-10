import os
import pandas as pd
import re

base_path = "../../converted_dataset/pedant"
subdirs = ["fewshot", "general"]

results = []

for subdir in subdirs:
    folder_path = os.path.join(base_path, subdir)
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        if "pedant_score" not in df.columns:
            print(f"No pedant_score in {filename}")
            continue

        mean_score = df["pedant_score"].dropna().mean()

        evaluator = "audience" if "audience" in filename else "writer"

        model_match = re.search(r"converted_?(.*?)(?:_fewshot|_general)?_", filename)
        model = model_match.group(1) if model_match else filename.split("_")[0]

        results.append({
            "model": model,
            "evaluator": evaluator,
            "prompt_version": subdir,
            "mean_pedant_score": round(mean_score, 4)
        })

df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by=["prompt_version", "model", "evaluator"])
print(df_result)


df_result.to_csv("pedant_mean_scores.csv", index=False)

#                    model evaluator prompt_version  mean_pedant_score
# 4      deepseek-qwen3-8B  audience        fewshot             0.5700
# 3      deepseek-qwen3-8B    writer        fewshot             0.6002
# 9   llama3.1-8B-Instruct  audience        fewshot             0.5394
# 1   llama3.1-8B-Instruct    writer        fewshot             0.5664
# 2     phi4-mini-instruct  audience        fewshot             0.4550
# 5     phi4-mini-instruct    writer        fewshot             0.4904
# 6    qwen2.5-7B-Instruct    writer        fewshot             0.4980
# 7    qwen2.5-7B-Instruct  audience        fewshot             0.4465
# 8               qwen3-8B  audience        fewshot             0.6168
# 0               qwen3-8B    writer        fewshot             0.6474
# 17     deepseek-qwen3-8B  audience        general             0.6217
# 15     deepseek-qwen3-8B    writer        general             0.5785
# 11  llama3.1-8B-Instruct  audience        general             0.5105
# 13  llama3.1-8B-Instruct    writer        general             0.5328
# 18   phi-4-mini-instruct  audience        general             0.4578
# 10   phi-4-mini-instruct    writer        general             0.4714
# 19   qwen2.5-7B-Instruct  audience        general             0.4916
# 16   qwen2.5-7B-Instruct    writer        general             0.5013
# 12              qwen3-8B  audience        general             0.6195
# 14              qwen3-8B    writer        general             0.6425
