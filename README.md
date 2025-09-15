# MOJ – Mixture-of-Grader

This project provides a framework for evaluating large language models (LLMs) on **human preference datasets**.  
It includes dataset conversion, model inference (via vLLM), metric calculations, and visualization.

## Project Structure
├─ .gitignore
├─ .gitlab-ci.yml # CI/CD with Kaniko
├─ Dockerfile # Container build
├─ job.yaml # Kubernetes job spec
├─ requirements.txt # Python dependencies
├─ src/
├─ nautilus_audience.py # Inference runner on Nautilus
├─ buzzBench/ # BuzzBench evaluation
│ ├─ buzzBench_audience.py
│ ├─ buzzBench_writer.py
│ ├─ chat_completion_*.py
│ ├─ evaluate_with_pedant.py
│ ├─ calculation/ # MSE calculation scripts
│ └─ heatmap/ # Heatmap outputs
├─ mt-bench/ # MT-Bench evaluation
│ └─ mt-bench.py
└─ stanfordNLP/ # Stanford NLP dataset evaluation
├─ chat_completion_stanfordNLP.py
├─ completion_stanfordNLP.py
├─ correct_prediction_stanfordNLP.py
├─ match_calculation.py
└─ bar_plot/
