import json
import requests
import openai

with open("data/finance.json", "r") as f:
    data = json.load(f)


def send_prompt_to_api(model: str, prompt: str) -> dict:
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.0,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )


responses = [
    send_prompt_to_api("google/gemma-3-1b-it", item["prompt"]) for item in data[0:5]
]
