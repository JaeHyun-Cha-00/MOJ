# MOJ – Mixture-of-Grader

This project provides a framework for evaluating large language models (LLMs) on **human preference datasets**.  
It includes dataset conversion, model inference (via vLLM), metric calculations, and visualization.

## Project Structure
```
.
├─ src/
│  ├─ nautilus_audience.py
│  ├─ buzzBench/
│  │  ├─ buzzBench_audience.py
│  │  ├─ buzzBench_writer.py
│  │  ├─ chat_completion_Buzzbench_audience.py
│  │  ├─ chat_completion_Buzzbench_writer.py
│  │  ├─ chat_completion_compare.py
│  │  ├─ evaluate_with_pedant.py
│  │  ├─ exclude_fewshot.py
│  │  ├─ fewshot_example.py
│  │  ├─ calculation/
│  │  │  ├─ mse_calculation_audience.py
│  │  │  └─ mse_calculation_writer.py
│  │  └─ heatmap/  (pre-generated figures)
│  ├─ mt-bench/
│  │  └─ mt-bench.py
│  └─ stanfordNLP/
│     ├─ chat_completion_stanfordNLP.py
│     ├─ completion_stanfordNLP.py
│     ├─ correct_prediction_stanfordNLP.py
│     ├─ match_calculation.py
│     └─ bar_plot/
├─ .gitignore
├─ .gitlab-ci.yml
├─ Dockerfile
├─ job.yaml
├─ requirements.txt
└─ README.md
```
