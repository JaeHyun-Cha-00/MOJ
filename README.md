### File Descriptions

| File Name                                      | Description |
|------------------------------------------------|-------------|
| `mse_calculation_audience.py`                 | Calculates MSE between audience scores and human scores for all models. |
| `mse_calculation_writer.py`                   | Calculates MSE between comedy writer scores and human scores for all models. |
| `mse_calculation_fewshot_audience.py`         | Calculates audience MSE for few-shot prompts. |
| `mse_calculation_fewshot_writer.py`           | Calculates writer MSE for few-shot prompts. |
| `mse_calculation_pedant_general_audience.py`  | Calculates MSE between PEDANT and audience scores for 0-shot (general) prompts. |
| `mse_calculation_pedant_general_writer.py`    | Calculates MSE between PEDANT and writer scores for 0-shot (general) prompts. |
| `mse_calculation_pedant_fewshot_audience.py`  | Calculates MSE between PEDANT and audience scores for 5-shot (fewshot) prompts. |
| `mse_calculation_pedant_fewshot_writer.py`    | Calculates MSE between PEDANT and writer scores for 5-shot (fewshot) prompts. |
| `buzzBench_audience.py`                       | Converts the BuzzBench dataset into a cleaned audience-focused CSV. |
| `buzzBench_writer.py`                         | Converts the BuzzBench dataset into a cleaned writer-focused CSV. |
| `chat_completion_buzzbench_audience.py`       | Sends audience-evaluation prompts to a local VLLM API and saves completions. |
| `chat_completion_buzzbench_writer.py`         | Sends comedy-writer-evaluation prompts to a local VLLM API and saves completions. |
| `evaluate_with_pedant.py`                     | Applies the PEDANT metric to BuzzBench completions and outputs with `pedant_score`. |
| `exclude_fewshot.py`                          | Removes few-shot rows from BuzzBench dataset (by index). |
| `fewshot_example.py`                          | Example script showing how few-shot prompts are constructed. |

---

### How to Run "chat_completion_buzzbench_audience.py" (Example)

```bash
python chat_completion_buzzbench_audience.py \
  --input_path ../../converted_dataset/buzzbench/general/buzzbench_converted_audience.csv \
  --output_path ../../converted_dataset/qwen3-8B_general_audience.csv \
  --model_name Qwen/Qwen3-8B
