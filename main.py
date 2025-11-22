import os
import openai
import json
from pathlib import Path
from common.modeling import Model
from eval.safe.query_serper import SerperAPI
from eval.safe.search_augmented_factuality_eval import main

model = Model(model_name="OPENAI:gpt-3.5-turbo")

with open(Path("Qwen-Qwen3-Next-80B-A3B-Instruct-finance_basic_golden.json"), "r") as f:
    results = json.load(f)

question = results["results"][0]["query"]
answer = results["results"][0]["response"]["answer"]

evaluation = main(question, answer, rater=model)

print(evaluation)
