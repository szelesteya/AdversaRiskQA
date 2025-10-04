#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=CheckEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=out/slurm_output_%A.txt
#SBATCH --error=out/slurm_error_%A.txt
#SBATCH --reservation=terv92681

# Load and install necessary modules
module purge
pip install uv

# Clone git repository if not present
if [ ! -d "$HOME/projects/AdversaRiskQA" ]; then
    git clone git@github.com:szelesteya/AdversaRiskQA.git

# Move to project directory
cd $HOME/projects/AdversaRiskQA

# Creating and activating virtual environment
uv venv
uv sync
source .venv/bin/activate  # activate your virtual environment

# Running the script with the specified model and dataset
uv run pyhton script.py $MODEL $NO_SAMPLES


