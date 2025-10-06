# AdversaRiskQA

## Environment setup
### Windows

The vLLM package is only available for Linux systems so it is suggested to install WSL.

Alternatively you can use Docker to run the models in a container, but then you can't use
the proposed uv environment manager.

Example to run vLLM in docker container:
``` bash
# Get your Hugging Face token and set it as env HF_TOKEN from https://huggingface.co/settings/tokens
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

Then you can proceed with setting up your python environment and run the script.

```python
python script.py Qwen/Qwen3-0.6B 20
```

### Linux/Unix/WSL

1. Clone the repository

```bash
   git clone git@github.com:szelesteya/AdversaRiskQA.git
   cd AdversaRiskQA
```

2. Install uv

```bash
   pip install uv
```

3. Create, sync and activate virtual environment

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