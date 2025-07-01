import pandas as pd
import csv

input_path = "../../converted_dataset/buzzbench_converted.csv"
output_path = "../../converted_dataset/buzzbench_converted_without_fewshot.csv"

excluded_row_indices = [29, 27, 42, 45, 44, 0, 3, 18, 19, 12]

df = pd.read_csv(input_path, quoting=csv.QUOTE_ALL)

filtered_df = df.drop(index=excluded_row_indices).reset_index(drop=True)

filtered_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print(f"{len(excluded_row_indices)} rows removed by index")
