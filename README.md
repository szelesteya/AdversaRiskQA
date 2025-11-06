# AdversaRiskQA
## Overview
AdversaRiskQA is a benchmark and evaluation framework designed to assess the resilience of Large Language Models (LLMs) to adversarial misinformation. 
The framework focuses on high-risk domains:
- Finance
- Health
- Law

The core task evaluates an LLM's ability to identify and correct a fallacious premise embedded within a user's question, rather than answering the question based on the incorrect information.

This repository provides:
- A dataset of questions containing factual knowledge and a corresponding "modified" (incorrect) version.
- A Python script (answer_generation.py) to generate model responses using vLLM.
- A Python script (evaluation.py) to evaluate the generated responses using an LLM-based judge (e.g., GPT-4o-mini).

## Features
- Targeted Dataset: Includes basic and advanced question sets for finance, health, and law.
- High-Performance Generation: Utilizes vLLM for fast and efficient LLM inference.
- Automated Evaluation: Employs an LLM judge to programmatically score a model's ability to refute misinformation.
- HPC Ready: Includes sample SLURM batch scripts (.sh) for running experiments on A100 and H100 GPU clusters.

## Environment Setup
This project uses uv for package management, as defined in pyproject.toml and uv.lock.

### Linux / WSL

1. Clone the repository
```bash
git clone https://github.com/szelesteya/AdversaRiskQA.git
cd AdversaRiskQA
```

2. Install uv:
```bash
pip install uv
```

3. Create and sync the virtual environment:
```bash
uv venv
uv sync
```

4. Setting up Hugging Face token for restricted models
```bash
pip install huggingface_hub
hf auth login
```

5. Run scripts
```bash
# Replace Qwen/Qwen3-0.6B with the model of your choice
# The second argument is the number of samples to run
python answer_generation.py law.json Qwen/Qwen3-0.6B 20
```

### Windows
The vLLM package is only available for Linux systems so it is suggested to install WSL.

Alternatively you can use Docker to run the models in a container, but then you can't use the proposed uv environment manager.

Example to run vLLM in docker container:
```bash
# Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
export HF_TOKEN="your_hf_token_here"

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

Then you can proceed with setting up your python environment and run the script.

```bash
python script.py Qwen/Qwen3-0.6B 20
```


## Workflow
1. Set Up Hugging Face Token (Optional)

If you are using gated models (like those from Meta or Mistral), you must log in to the Hugging Face CLI:
```bash
pip install huggingface_hub
hf auth login
```

2. Generate AnswersUse the answer_generation.py script to run your model of choice against one of the provided datasets.

Usage:
```bash
python answer_generation.py <data_file_name> <model_name> [num_samples]
```

Arguments:
- <data_file_name>: The name of the dataset file in the data/ directory (e.g., law_advanced_golden.json).
- <model_name>: The model identifier from Hugging Face (e.g., Qwen/Qwen3-0.6B).
- [num_samples] (Optional): The number of samples to run from the dataset.

Example:
```bash
# Run the Qwen-0.6B model on 20 samples from the law.json dataset
python answer_generation.py law_basic_golden.json Qwen/Qwen3-0.6B 20
```
Results will be saved as a JSON file in the out/ directory (e.g., out/Qwen-Qwen3-0.6B-law_basic_golden.json).

3. Evaluate Results

Use the evaluation.py script to score the generated answers. This script requires an OpenAI API key for the LLM judge.

Execute the script:
```bash
python evaluation.py
```
This will create a new file (e.g., out/Qwen-Qwen3-0.6B-law_basic_golden_evaluation.json) containing the original data along with a new "evaluation" field ("Correct" or "Incorrect") for each sample.

## Dataset Structure

Each item in the dataset JSON files follows this structure:
```json
{
  "knowledge": "The correct, factual statement.",
  "modified_knowledge": "The incorrect, adversarial premise.",
  "query": "The user's underlying question without the false premise.",
  "prompt": "The full prompt given to the LLM, which embeds the 'modified_knowledge' as a fact."
}
```
An evaluation of "Correct" means the model successfully identified and refuted the modified_knowledge.An evaluation of "Incorrect" means the model either accepted the false premise or failed to correct it.

## Running on HPC (SLURM)

The repository includes several batch scripts (.sh) for running the answer_generation.py script on HPC clusters using SLURM. These are pre-configured for different GPU partitions (A100, H100) and GPU counts.
Example usage:

```bash
sbatch answer_generation_a100.sh law_advanced_golden.json Qwen/Qwen3-0.6B 50
```
