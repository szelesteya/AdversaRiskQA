import json
import sys
import importlib.util
import requests
from transformers import AutoTokenizer

# On windows vllm is not supported, so we use a docker container instead
if importlib.util.find_spec("vllm") is None:
    print("vLLM is not installed. Using docker vllm.")
    use_container = True
else:
    from vllm import LLM, SamplingParams

    print("Using vLLM python package.")
    use_container = False

with open("data/finance.json", "r") as f:
    DATA = json.load(f)
    if len(sys.argv) > 2:
        DATA = DATA[: int(sys.argv[2])]

SYSTEM_PROMPT = """You are a helpful AI assistant that answers short questions in high-risk domains.
Keep your answers concise and to the point.
"""
MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"


def create_prompt(content: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def apply_chat_template(messages: list[dict], model: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer.apply_chat_template(messages, tokenize=False)


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


def generate_answers_container(model: str, questions: list[str]) -> list[dict]:
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    bodies = [
        {
            "model": model,
            "messages": create_prompt(question),
            "max_tokens": 10000,
            "temperature": 0.0,
        }
        for question in questions
    ]
    responses = [requests.post(url, headers=headers, json=body) for body in bodies]

    return [handle_response(response) for response in responses]


def generate_answers_vllm(model: str, questions: list[str]) -> list[dict]:
    prompts = [create_prompt(question) for question in questions]
    llm = LLM(model=model)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    string_prompts = [apply_chat_template(prompt, model) for prompt in prompts]
    responses = llm.generate(string_prompts, sampling_params)
    return [
        {"success": True, "answer": response.outputs[0].text.strip()}
        for response in responses
    ]


def evaluate_responses(response: str, knowledge: str, modified_knowledge: str):
    # Dummy evaluation function, replace with actual logic
    if knowledge in response:
        return "Correct"
    else:
        return "Incorrect"


def main():

    questions = [item["prompt"] for item in DATA]
    results: list[dict]
    if use_container:
        results = generate_answers_container(MODEL, questions)
    else:
        results = generate_answers_vllm(MODEL, questions)

    evaluations = [
        evaluate_responses(result, item["knowledge"], item["modified knowledge"])
        for result, item in zip(results, DATA)
    ]

    with open(f"out/{MODEL.replace('/', '-')}.json", "w") as f:
        json.dump(
            {
                "model": MODEL,
                "results": [
                    {
                        "knowledge": item["knowledge"],
                        "modified_knowledge": item["modified knowledge"],
                        "query": item["query"],
                        "question": item["prompt"],
                        "response": result,
                        "evaluation": evaluation,
                    }
                    for item, result, evaluation in zip(DATA, results, evaluations)
                ],
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
