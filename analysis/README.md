# Analysis Notebooks

This directory contains Jupyter notebooks for analyzing AdversaRiskQA evaluation results.

## Notebooks

### `adversarial_analysis.ipynb`

Exploratory data analysis on adversarial evaluation results. This notebook:

- Loads evaluation results from `data/results/adversarial_evaluations/`
- Creates visualizations comparing model performance across:
  - **Domains**: Finance, Health, Law
  - **Difficulty levels**: Basic and Advanced
  - **Models**: Various LLM models tested
- Generates interactive plots using Plotly, including:
  - Accuracy comparisons by domain and difficulty
  - Model performance heatmaps
  - Analysis of successful adversarial attacks

### `factuality_analysis.ipynb`

Analysis of factuality evaluation results. This notebook:

- Loads factuality evaluation results from both adversarial and non-adversarial prompts
- Builds dataframes with factuality metrics:
  - Number of facts extracted from responses
  - Count of correct facts
  - F1@K scores (factuality precision/recall)
- Compares factuality performance between:
  - Adversarial vs non-adversarial prompts
  - Different domains and difficulty levels
- Aggregates metrics by dataset for summary statistics

## Usage

1. Ensure evaluation results are available in `data/results/`
2. Open the notebooks in Jupyter Lab/Notebook
3. Run cells sequentially to generate analyses and visualizations
4. Modify paths or parameters as needed for your specific results structure

