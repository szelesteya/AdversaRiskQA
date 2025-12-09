# Answer Generation Script

This directory contains the `execute_prompts.py` script for generating answers from open-source Large Language Models (LLMs) on the AdversaRiskQA dataset.

## Overview

The `execute_prompts.py` script is responsible for:
- Loading prompts from dataset JSON files
- Generating model responses using either vLLM (Python package) or a Docker container
- Applying chat templates appropriate for each model when available
- Sanitizing and cleaning model responses
- Saving results in a structured JSON format

## Important Note: Different Requirements

**This script has different requirements from other modules in the project** (such as the evaluation scripts in `eval/`). Unlike the evaluation modules which primarily use OpenAI API calls, this script requires:

1. **Hardware**: GPU access (CUDA-compatible) for running open-source models
2. **Infrastructure**: Either vLLM installed natively or a Docker container running vLLM
3. **Architecture-specific configuration**: The script must be adjusted to match the exact architecture where open-source models are executed, including:
   - GPU configuration (tensor parallelism, device placement)
   - Model loading parameters
   - Container networking (if using Docker)
   - System-specific paths and permissions

**Before using this script, ensure your environment is properly configured for running open-source LLMs.**

## Installation

### Option 1: Native vLLM (Linux/WSL)

Install vLLM and required dependencies:

```bash
pip install vllm transformers requests
```

**Note**: vLLM is only available for Linux systems. On Windows, use WSL or Option 2 (Docker).

### Option 2: Docker Container (Windows/Cross-platform)

If vLLM is not installed, the script will automatically detect this and use a Docker container instead. You'll need to run a vLLM container separately:

```bash
# Get your Hugging Face token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_hf_token_here"

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model <model_name>
```

The script will connect to the container via `http://localhost:8000/v1/chat/completions`.

## Usage

```bash
python execute_prompts/execute_prompts.py <data_file> <model_name> [num_samples]
```

### Arguments

- `<data_file>`: Name of the dataset file in the `data/prompts` directory (e.g., `prompts/law_basic_golden.json`)
- `<model_name>`: Model identifier from Hugging Face (e.g., `Qwen/Qwen3-0.6B`)
- `[num_samples]` (Optional): Number of samples to process from the dataset. If omitted, processes all samples.

### Examples

```bash
# Process all samples from law_basic_golden.json using Qwen3-0.6B
python execute_prompts/execute_prompts.py prompts/law_basic_golden.json Qwen/Qwen3-0.6B

# Process only the first 20 samples
python execute_prompts/execute_prompts.py prompts/law_basic_golden.json Qwen/Qwen3-0.6B 20

# Using a different model
python execute_prompts/execute_prompts.py prompts/health_advanced_golden.json facebook/opt-125m 10
```

## Configuration

### Model Parameters

The script uses the following default parameters:
- **Temperature**: 0.0 (deterministic outputs)
- **Max Tokens**: 8192
- **Tensor Parallelism**: 2 (for native vLLM)

**To adjust these parameters**, modify the script:
- For native vLLM: Edit `SamplingParams` in `generate_answers_vllm()` and `tensor_parallel_size` in `LLM()` initialization
- For Docker: Modify the request body in `generate_answers_container()`

### System Prompt

The script applies a system prompt that instructs models to:
- Write in complete, grammatically correct sentences
- Limit responses to a maximum of five sentences
- Use plain, professional language

The system prompt can be modified in the `SYSTEM_PROMPT` constant at the top of the script.

### Output Location

Results are saved to the `out/` directory with the filename format:
```
out/<model_name_formatted>-<dataset_name>.json
```

For example: `out/Qwen-Qwen3-0.6B-law_basic_golden.json`

## Input Format

The script expects JSON files in the `data/` directory with the following structure:

```json
[
  {
    "knowledge": "The correct, factual statement.",
    "modified_knowledge": "The incorrect, adversarial premise.",
    "query": "The user's underlying question without the false premise.",
    "prompt": "The full prompt given to the LLM, which embeds the 'modified_knowledge' as a fact."
  }
]
```

## Output Format

The script generates JSON files with the following structure:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "dataset": "prompts/law_basic_golden.json",
  "results": [
    {
      "knowledge": "The correct fact...",
      "modified_knowledge": "The incorrect premise...",
      "query": "The underlying question...",
      "question": "The full prompt...",
      "response": {
        "success": true,
        "answer": "The model's generated response..."
      }
    }
  ]
}
```

## Response Sanitization

The script includes a `sanitize_llm_response()` function that:
1. Removes chain-of-thought artifacts (e.g., "assistantfinal" markers)
2. Strips common prefixes and artifacts (e.g., "Assistant:", "me:", URL fragments)
3. Validates that responses are not empty

This sanitization is only applied when using native vLLM. Docker container responses are returned as-is from the API.

## Architecture-Specific Adjustments

When deploying this script to a specific architecture, you may need to adjust:

1. **GPU Configuration**:
   - `tensor_parallel_size` in `LLM()` initialization (line 126)
   - Docker GPU flags (`--gpus all`, `--runtime nvidia`)

2. **Model Loading**:
   - `trust_remote_code=True` flag (line 126)
   - Model cache directory paths
   - Hugging Face token authentication

3. **Network Configuration** (Docker):
   - Port mapping (`-p 8000:8000`)
   - Container networking mode
   - API endpoint URL (line 108)

4. **Resource Limits**:
   - Max tokens per request
   - Batch size for parallel requests
   - Memory allocation

5. **Chat Templates**:
   - The script automatically applies model-specific chat templates
   - Falls back to simple concatenation if no template exists
   - May need adjustment for custom or fine-tuned models

## Troubleshooting

### vLLM Not Found
If you see "vLLM is not installed. Using docker vllm.":
- Install vLLM: `pip install vllm` (Linux only)
- Or ensure Docker container is running on port 8000

### Docker Connection Errors
- Verify the container is running: `docker ps`
- Check port 8000 is accessible: `curl http://localhost:8000/health`
- Ensure firewall allows localhost connections

### GPU Out of Memory
- Reduce `tensor_parallel_size`
- Use a smaller model
- Reduce batch size (process fewer samples at once)

### Model Loading Errors
- Verify Hugging Face token is set: `hf auth login`
- Check model name is correct and accessible
- Ensure sufficient disk space for model cache

## Differences from Evaluation Scripts

Unlike the evaluation scripts in `eval/`:
- **No OpenAI API key required** (uses open-source models)
- **Requires GPU access** (for model inference)
- **Architecture-dependent** (must match deployment environment)
- **Different dependencies** (vLLM vs. OpenAI SDK)
- **Local execution** (models run on your hardware/container)

## See Also

- Main project README: `../README.md`
- Evaluation scripts: `../eval/README.md`
- Dataset files: `../data/prompts/`

