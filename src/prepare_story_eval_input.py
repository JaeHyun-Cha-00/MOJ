import pandas as pd

def convert_to_eval_format(input_path, output_path):
    # Load your converted story benchmark dataset
    df = pd.read_csv(input_path)

    # Extract prompt and story text
    eval_df = df[["question", "attempted_answer"]].copy()
    eval_df.columns = ["prompt", "story_text"]

    # Add required columns for evaluation
    eval_df["target_word_count"] = eval_df["story_text"].apply(lambda x: len(str(x).split()))
    eval_df["language"] = "en"

    # Save to evaluation format
    eval_df.to_csv(output_path, index=False, float_format="%.2f")


if __name__ == "__main__":
    input_csv = "../converted_dataset/story_benchmark_converted_50.csv"
    output_csv = "../converted_dataset/story_eval_input.csv"

    convert_to_eval_format(input_csv, output_csv)