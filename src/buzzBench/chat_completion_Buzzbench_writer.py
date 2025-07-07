import pandas as pd
import requests
import time
import json
import logging
import click
from prompt import few_shot_writer_examples

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# Qwen2.5-7B-Instruct (O), Phi-4-mini-instruct (), Qwen3-8B (), DeepSeek-R1-0528-Qwen3-8B (), Llama-3.1-8B-Instruct (O) 

def make_writer_prompt(text):
    return f"""
You are a humor evaluator representing a general comedy writer.

Your response must ONLY contain the final formatted answer.

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How the comedy writer might react>

** Funniness Rating **  
Comedy writer: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Here are 5 examples:

{few_shot_writer_examples}

Now evaluate:

{text}
"""

@click.command()
@click.option("--input_path", type=click.Path(exists=True), required=True, help="Input CSV path")
@click.option("--output_path", type=click.Path(), required=True, help="Output CSV path")
@click.option("--model_name", type=str, required=True, help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
@click.option("--verbose", "-v", count=True, help="Verbosity level")
def main(input_path, output_path, model_name, verbose):
    # Logging
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

            res_json = res.json()

            content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.warning(f"[{idx}] Empty content in response.")

            attempted_answers.append(content)
            logger.debug(f"[{idx}] Completion:\n{content}\n")

        except Exception as e:
            logger.error(f"[{idx}] Audience Error: {e}")
            attempted_answers.append("")

        logger.info(f"[{idx}] Done")
        time.sleep(0.5)

    logger.info(f"Total answers collected: {len(attempted_answers)} vs {len(df)} rows in DataFrame")
    df["attempted_answer"] = attempted_answers
    df.to_csv(output_path, index=False, quoting=1)
    logger.info(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()