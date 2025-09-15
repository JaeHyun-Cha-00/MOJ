import pandas as pd
import requests
import time
import json
import logging
import os
import click

VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
HEADERS = {"Content-Type": "application/json"}

def make_audience_prompt(text):
    return f"""
You are a humor evaluator representing a general audience.

Your response must ONLY contain the final formatted answer.

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How the audience might react>

** Funniness Rating **  
Audience: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Now evaluate:

{text}
"""

@click.command()
@click.option("--input_path", type=click.Path(exists=True), required=True, help="Input CSV path")
@click.option("--output_path", type=click.Path(), required=True, help="Output CSV path")
@click.option("--model_name", type=str, required=True, help="Model name (e.g., Qwen/Qwen3-8B)")
@click.option("--verbose", "-v", count=True, help="Verbosity level")
def main(input_path, output_path, model_name, verbose):
    os.makedirs("/app/logs", exist_ok=True)
    log_file = f"/app/logs/{model_name.replace('/', '_')}.log"

    logging_level = logging.DEBUG if verbose >= 1 else logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    logger.addHandler(logging.FileHandler(log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())

    logger.info(f"[START] Model: {model_name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"VLLM endpoint: {VLLM_API_URL}")

    df = pd.read_csv(input_path)
    attempted_answers = []

    for idx, row in df.iterrows():
        prompt = make_audience_prompt(row["question"])
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

    logger.info(f"Collected answers: {len(attempted_answers)} / {len(df)} rows")
    df["attempted_answer"] = attempted_answers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, quoting=1)
    logger.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()