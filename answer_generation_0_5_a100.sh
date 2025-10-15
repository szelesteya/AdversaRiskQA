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


DATA=$1
MODEL=$2
NO_SAMPLES=$3

# Load and install necessary modules
module purge
pip install uv

# Clone git repository if not present
if [ ! -d "$HOME/projects/AdversaRiskQA" ]; then
  git clone git@github.com:szelesteya/AdversaRiskQA.git
fi
 echo "Repository is already cloned."

# Move to project directory
cd $HOME/projects/AdversaRiskQA

# Create and activate virtual environment only if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating new virtual environment..."
  uv venv
  uv sync
else
  echo "Using existing virtual environment."
fi

source .venv/bin/activate  # activate your virtual environment

echo "Running $DATA$ dataset with $NO_SAMPLES samples on $MODEL model"
# Running the script with the specified model and dataset
uv run python answer_generation.py $DATA $MODEL $NO_SAMPLES
