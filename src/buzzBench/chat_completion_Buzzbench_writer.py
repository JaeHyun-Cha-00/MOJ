import pandas as pd
import requests
import time
import json
import logging
import click
from prompt import few_shot_writer_examples

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def make_writer_prompt(text):
    return f"""
You are a strict humor evaluator representing a professional comedy writer.

You MUST follow the format **exactly as shown** below.  
You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Name's intro").  
If you mention or refer to any other characters, your answer is invalid.

You MUST NOT include audience opinions or audience ratings.

Use this format for your output:

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How a comedy writer might evaluate it>

** Funniness Rating (Comedy Writer) **  
Comedy writer: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

DO NOT DEVIATE FROM THIS FORMAT.

Here are 5 formatted examples:

{few_shot_writer_examples}

Now evaluate:

{text}
"""

@click.command()
@click.option("--input_path", type=click.Path(exists=True), required=True, help="Path to input CSV file")
@click.option("--output_path", type=click.Path(), required=True, help="Path to save output CSV")
@click.option("--model_name", type=str, required=True, help="Model name (e.g., microsoft/Phi-4-mini-instruct)")
@click.option("--verbose", "-v", count=True, help="Verbosity level (e.g., -v, -vv)")
def main(input_path, output_path, model_name, verbose):
    # Set up logging
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
        prompt = make_writer_prompt(row["question"])
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.5
        }

        try:
            res = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload))
            res.raise_for_status()
            response = res.json()
            content = response["choices"][0]["message"]["content"].strip()
            attempted_answers.append(content)
            logger.debug(f"[{idx}] Completion:\n{content}\n")
        except Exception as e:
            logger.error(f"[{idx}] Writer Error: {e}")
            attempted_answers.append("")

        logger.info(f"[{idx}] Done")
        time.sleep(0.5)

    df["attempted_answer"] = attempted_answers
    df.to_csv(output_path, index=False, quoting=1)
    logger.info(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()