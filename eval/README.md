# Evaluation Scripts

This directory contains evaluation scripts for assessing model performance on two key dimensions: **factuality** and **adversality**.

## Overview

### Factuality Evaluation

The `factuality.py` script implements a **Search-Augmented Factuality Evaluator (SAFE)** pipeline inspired by the methodology described in [Long-form factuality in large language models](https://arxiv.org/abs/2403.18802) (Wei et al., 2024).

**Key Features:**
- Extracts individual, verifiable facts from model responses
- Validates each fact using web search and authoritative sources
- Provides detailed citations and reasoning for each fact check
- Supports batch processing with parallel execution

**Academic Background:**
The SAFE approach uses LLM agents as automated evaluators that:
1. Break down long-form responses into atomic facts
2. Validate each fact through multi-step reasoning with web search
3. Aggregate results using an extended F1 score that balances precision (supported facts) and recall (completeness)

### Adversality Evaluation

The `adversality.py` script implements an **LLM-as-Judge** approach to evaluate how well models correct misinformation, similar to the methodology in [Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models](https://arxiv.org/abs/2503.10690) (Sakib et al., 2025).

**Key Features:**
- Evaluates whether models successfully correct adversarial misinformation
- Uses a strict LLM judge to determine if incorrect premises are addressed
- Provides binary classification (Correct/Incorrect) for each evaluation

**Academic Background:**
This evaluator assesses how models respond when presented with deliberately incorrect information in prompts. The LLM-as-judge approach uses a separate language model to evaluate whether the target model's response:
- Explicitly or implicitly corrects the misinformation
- Successfully steers users toward accurate information
- Avoids accepting or reinforcing false premises

## Installation

Ensure you have the required dependencies installed:

```bash
pip install openai python-dotenv pydantic
```

Or use a project manager and sync it with `pyproject.toml` and activate environment:

```bash
uv venv
uv sync
source .venv/bin/activate # Win: source .venv/Scripts/activate
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Factuality Evaluation

Evaluate the factuality of model responses:

```bash
python -m eval.factuality <input_file> [options]
```

**Arguments:**
- `input_file`: Path to JSON file containing model responses

**Options:**
- `--output-file PATH`: Custom output path (default: `{input_file}_factuality.json`)
- `--limit N`: Limit evaluation to first N items
- `--debug`: Enable verbose logging and detailed error traces
- `--log-output PATH`: Write logs to a file
- `--smoke`: Run a single-question smoke test

**Example:**
```bash
python -m eval.factuality data/results/model_responses.json --limit 10 --log-output logs/factuality.log
```

**Input Format:**
The input JSON should have the following structure:

```json
{
  "model": "model-name",
  "dataset": "dataset-name",
  "results": [
    {
      "question": "What is the capital of France?",
      "response": {
        "success": true,
        "answer": "The capital of France is Paris..."
      }
    }
  ]
}
```

**Output Format:**
The output includes extracted facts and validation results:

```json
{
  "results": [
    {
      "question": "...",
      "response": {...},
      "factuality": {
        "answer": "...",
        "facts": ["Fact 1", "Fact 2"],
        "fact_check": {
          "decisions": [
            {
              "fact": "Fact 1",
              "correct": true,
              "rational": [
                {
                  "link": {
                    "title": "Source Title",
                    "hyperlink": "https://..."
                  },
                  "supporting_information": "..."
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

**Note:** The factuality evaluation API calls may include empty responses in some cases (e.g., when no facts can be extracted, validation fails, or the model response is empty). It's worth checking the output to ensure all expected evaluations are present. Items with failed responses or missing answers will have empty factuality results with `"facts": []` and `"fact_check": {"decisions": []}`.

**Environment Variables:**
- `FACT_EXTRACTION_MODEL`: Model for fact extraction (default: `gpt-5`)
- `FACT_VALIDATION_MODEL`: Model for fact validation (default: `gpt-5-mini-2025-08-07`)
- `DATA_LIBRARY`: Directory for output files (default: `data`)
- `FACT_MAX_OUTPUT_TOKENS`: Maximum output tokens (default: `2048`)

### Adversality Evaluation

Evaluate how well models correct adversarial misinformation:

```bash
python -m eval.adversality <input_file> [options]
```

**Arguments:**
- `input_file`: Path to JSON file containing model responses

**Options:**
- `--output-file PATH`: Custom output path (default: `{input_file}_evaluation.json`)
- `--limit N`: Limit evaluation to first N items
- `--debug`: Enable verbose logging and detailed error traces
- `--log-output PATH`: Write logs to a file

**Example:**
```bash
python -m eval.adversality data/results/model_responses.json --limit 10 --log-output logs/adversality.log
```

**Input Format:**
The input JSON should have the following structure:

```json
{
  "results": [
    {
      "knowledge": "Correct fact about the topic",
      "modified_knowledge": "Incorrect premise inserted into question",
      "question": "Question containing the incorrect premise",
      "response": {
        "answer": "Model's response to the question"
      }
    }
  ]
}
```

**Output Format:**
The output includes evaluation results:

```json
{
  "results": [
    {
      "knowledge": "...",
      "modified_knowledge": "...",
      "question": "...",
      "response": {...},
      "evaluation": "Correct"  // or "Incorrect", "UnknownError", "EvaluationError"
    }
  ]
}
```

**Environment Variables:**
- `ADVERSALITY_JUDGE_MODEL`: Model used as judge (default: `gpt-5-mini`)

## Logging

Both scripts use structured logging with the following format:

```
YYYY-MM-DD HH:MM:SS | LEVEL | logger_name | message | key=value | key2=value2
```

Logs include:
- Timestamps
- Log levels (INFO, WARNING, ERROR, DEBUG)
- Contextual information via extra fields
- Token usage and cost estimates

## Cost Considerations

### Factuality Evaluation
- **Fact Extraction**: Uses `gpt-5` (higher cost, better extraction)
- **Fact Validation**: Uses `gpt-5-mini` with web search (cost per search query)
- Costs are logged for each operation

### Adversality Evaluation
- **Judge Model**: Uses `gpt-5-mini` with high-effort reasoning
- Single API call per question-answer pair
- Costs are logged for each evaluation

## Error Handling

Both scripts handle errors gracefully:
- **Validation Errors**: Logged and skipped
- **API Errors**: Retried with exponential backoff
- **Critical Errors**: Logged and marked in output
- Failed evaluations are marked with error codes in the output

## Code Structure

```
eval/
├── __init__.py
├── logger.py              # Shared logging utilities
├── adversality.py         # Adversality evaluation script
├── factuality.py          # Factuality evaluation script
├── openai_response_models.py  # Pydantic models for structured outputs
└── README.md              # This file
```

## References

1. Wei, J., Yang, C., Song, X., et al. (2024). Long-form factuality in large language models. *arXiv preprint arXiv:2403.18802*. https://arxiv.org/abs/2403.18802

2. Sakib, S. K., Das, A. B., & Ahmed, S. (2025). Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models. *arXiv preprint arXiv:2503.10690*. https://arxiv.org/abs/2503.10690

