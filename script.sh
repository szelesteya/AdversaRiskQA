#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=CheckEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt
#SBATCH --reservation=terv92681

# Load and install necessary modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install uv

# Move to project directory CHANGE THIS TO YOUR PROJECT DIRECTORY
cd $HOME/project/adversarisk-qa

# Creating and activating virtual environment
uv venv
source .venv/bin/activate  # activate your virtual environment

# Starting LLM in container
# define your hugging face token in HF_TOKEN environment variable and hugging face llm model in LLM_MODEL environment variable
# or add them as an argument for the job submission command: sbatch --export HF_TOKEN=your_token,LLM_MODEL=your_model script.sh
docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"     -p 8000:8000     --ipc=host     vllm/vllm-openai:latest     --model "$LLM_MODEL"





