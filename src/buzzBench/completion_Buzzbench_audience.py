import pandas as pd
import requests
import time
import json
import logging
import click
from prompt import few_shot_audience_examples

VLLM_API_URL = "http://localhost:8000/v1/completions"
HEADERS = {"Content-Type": "application/json"}

def make_prompt(text):
    return f"""
Only analyze the character whose intro heading appears below (e.g., "# Character Intro").
Do not mention or evaluate *any* other characters, even if their names appear in the intro text.
Your response must only include one character analysis.

If you mention more than one character, your answer is invalid.

At the end of your response, include the ratings section in this exact format:

** Funniness Ratings **
Audience: <1–5> (<description from scale>)
Comedy writer: <1–5> (<description from scale>)

Use this exact structure. Do not change the heading or labels.

--

Here are 5 formatted examples as a reference:

{few_shot_audience_examples}

--

Here is the full introduction text:

{text}
"""

@click.command()
@click.option("--input_path", type=click.Path(exists=True), required=True, help="Input CSV path")
@click.option("--output_path", type=click.Path(), required=True, help="Output CSV path")
@click.option("--model_name", type=str, required=True, help="Model name to be passed to VLLM API")
@click.option("--verbose", "-v", count=True, help="Verbosity level. Use -v, -vv, etc.")
def main(input_path, output_path, model_name, verbose):
    # Logging config
    logging_level = logging.INFO
    if verbose == 1:
        logging_level = logging.DEBUG
    elif verbose == 2:
        logging_level = logging.INFO
    elif verbose == 3:
        logging_level = logging.WARNING
    elif verbose == 4:
        logging_level = logging.ERROR

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler("mmlu.log", mode="a"))
    logger.addHandler(logging.StreamHandler())

    df = pd.read_csv(input_path)
    attempted_answers = []

    for idx, row in df.iterrows():
        prompt_text = make_prompt(row["question"])
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": 2048,
            "temperature": 0.5,
            "stop": ["</s>"]
        }

        try:
            res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
            res.raise_for_status()
            completion = res.json()["choices"][0]["text"].strip()
            logger.debug(f"[{idx}] Completion:\n{completion}\n")
            attempted_answers.append(completion)
        except Exception as e:
            logger.error(f"[{idx}] Error: {e}")
            attempted_answers.append("")

        time.sleep(0.5)

    df["attempted_answer"] = attempted_answers
    df.to_csv(output_path, index=False, quoting=1)
    logger.info(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()