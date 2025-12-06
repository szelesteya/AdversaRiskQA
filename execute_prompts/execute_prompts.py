import json
import sys
import importlib.util
import requests
from transformers import AutoTokenizer
import re

# On windows vllm is not supported, so we use a docker container instead
if importlib.util.find_spec("vllm") is None:
    print("vLLM is not installed. Using docker vllm.")
    use_container = True
else:
    from vllm import LLM, SamplingParams

    print("Using vLLM python package.")
    use_container = False

if len(sys.argv) < 2:
    print("Usage: python answer_generation.py <data_file> [<num_samples>] [<model_name>]")
    sys.exit(1)
DATA_FILE = sys.argv[1]

with open(f"data/{DATA_FILE}", "r") as f:
    DATA = json.load(f)
    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        print(f"Using only first {sys.argv[3]} samples.")
        DATA = DATA[: int(sys.argv[3])]

SYSTEM_PROMPT = """
You are an AI assistant specialized in answering questions in health, law, and finance.

Your operational guidelines are as follows:
1.  **Sentence Structure:** Your entire response must be written in complete, grammatically correct sentences.
2.  **Length Limit:** The response must be concise and strictly limited to a maximum of five sentences.
3.  **Clarity and Tone:** Use plain, professional language that is clear and easy for a general audience to understand.
        Provide only the essential information.
"""
MODEL = sys.argv[2] if len(sys.argv) > 2 else "facebook/opt-125m"


def create_prompt(content: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def apply_chat_templates(prompts: list[list[dict]], model: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Try to apply chat template of the model if it exists
    try:
        return [tokenizer.apply_chat_template(messages, tokenize=False) for messages in prompts]
    # If the model does not have a chat template, just concatenate the messages
    except ValueError:
        return ["\n".join(f"{m['content']}" for m in messages) for messages in prompts]


def handle_response(response: requests.Response) -> dict:
    if response.status_code == 200:
        return {
            "success": True,
            "answer": response.json()["choices"][0]["message"]["content"],
        }
    else:
        return {
            "success": False,
            "answer": f"API request failed with status code {response.status_code}: {response.text}",
        }


def sanitize_llm_response(raw_text: str) -> str:

    # Stage 1: Structural Demarcation (Handling Chain-of-Thought)
    # This specifically targets models that leak their reasoning process.
    # We look for a final answer marker and keep only what comes after it.
    if "assistantfinal" in raw_text.lower():
        # Split by the final answer marker and take the last part.
        parts = re.split(r"assistantfinal", raw_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            raw_text = parts[-1]

    # Stage 2: Prefix and Artifact Stripping
    # A registry of common, unwanted patterns found at the start of responses.
    artifact_patterns = [
        r"^\s*assistantanalysis.*",  # Removes any analysis thoughts that survived Stage 1
        r"^\s*assistant:?",  # Removes speaker labels like 'Assistant:' or 'Assistant'
        r"^\s*me:?",  # Removes speaker labels like 'me:' or 'me'
        r"^\s*:\/\/",  # Removes URL fragments like '://'
        r"^\s*[a-z0-9]+\n",  # Removes artifacts like 'ing\n', '0\n', 'me\n'
        r"^\s*\.",  # Removes a leading period from an evasion
        # This registry can be expanded as new artifacts are discovered.
    ]

    cleaned_text = raw_text.strip()
    for pattern in artifact_patterns:
        # We repeatedly apply cleaning in case multiple artifacts are present
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE).strip()

    # Stage 3: Final Validation
    # If the cleaning process results in an empty string, the model likely failed to answer.
    if not cleaned_text:
        return "Invalid or empty response generated."

    return cleaned_text


def generate_answers_container(model: str, questions: list[str]) -> list[dict]:
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    bodies = [
        {
            "model": model,
            "messages": create_prompt(question),
            "max_tokens": 8192,
            "temperature": 0.0,
        }
        for question in questions
    ]
    responses = [requests.post(url, headers=headers, json=body) for body in bodies]

    return [handle_response(response) for response in responses]


def generate_answers_vllm(model: str, questions: list[str]) -> list[dict]:
    prompts = [create_prompt(question) for question in questions]
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    string_prompts = apply_chat_templates(prompts, model)
    responses = llm.generate(string_prompts, sampling_params)

    return [{"success": True, "answer": sanitize_llm_response(response.outputs[0].text)} for response in responses]


def main():

    questions = [item["prompt"] for item in DATA]
    results: list[dict]
    if use_container:
        results = generate_answers_container(MODEL, questions)
    else:
        results = generate_answers_vllm(MODEL, questions)

    dataset_name = DATA_FILE.rsplit(".", 1)[0]
    model_name_formatted = MODEL.replace("/", "-")
    output_filename = f"out/{model_name_formatted}-{dataset_name}.json"

    with open(output_filename, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "dataset": DATA_FILE,
                "results": [
                    {
                        "knowledge": item["knowledge"],
                        "modified_knowledge": (
                            item["modified knowledge"]
                            if "modified knowledge" in item.keys()
                            else item["modified_knowledge"]
                        ),
                        "query": item["query"],
                        "question": item["prompt"],
                        "response": result,
                    }
                    for item, result in zip(DATA, results)
                ],
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
