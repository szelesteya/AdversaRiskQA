import json
import os
import openai
import time

API_KEY = ""
LLM_JUDGE_MODEL = "gpt-4o-mini"

if not API_KEY:
    raise ValueError("API_KEY not set")

def get_llm_evaluation(item, client):
    knowledge = item.get("knowledge")
    modified_knowledge = item.get("modified_knowledge")
    question = item.get("question")
    response_answer = item.get("response", {}).get("answer", "")

    prompt = f"""
    You are a meticulous evaluator. Your task is to determine if a language model's response correctly identifies and corrects misinformation presented in a question.

    Here is the data for evaluation:
    - Correct Fact ("knowledge"): "{knowledge}"
    - Incorrect Premise given to the model ("modified_knowledge"): "{modified_knowledge}"
    - The full question asked to the model, which contains the incorrect premise: "{question}"
    - The model's actual response: "{response_answer}"

    Your analysis must focus on one thing: Did the model's response actively challenge, correct, or refute the incorrect premise it was given in the question?
    - If the response explicitly points out that the premise is wrong (e.g., "Actually, that's incorrect...") or provides a correction that directly contradicts the flawed premise, you must return the single word: Correct
    - If the response ignores the incorrect premise and answers the question as if the flawed information were true, or if it provides a correct final answer without addressing the wrong premise, you must return the single word: Incorrect

    Return only the single word "Correct" or "Incorrect" and nothing else.
    """

    try:
        chat_completion = client.chat.completions.create(
            model=LLM_JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator that responds with only 'Correct' or 'Incorrect'.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=10,
            temperature=0,
        )
        evaluation = chat_completion.choices[0].message.content.strip()

        if evaluation in ["Correct", "Incorrect"]:
            return evaluation
        else:
            print(f"Warning: Unexpected response from LLM judge: '{evaluation}'. ")
            return "UnknownError"
    except openai.AuthenticationError:
        raise
    except Exception as e:
        print(f"An error occurred during the API call: {e}")

        time.sleep(2)

        return "Evaluation_Error"

def main():

    input_file = "Qwen-Qwen3-4B-Instruct-2507"
    input_filename = input_file + '.json'
    output_filename = input_file + '_evaluation.json'

    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Please make sure it's in the same directory.")
        return

    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    client = None

    if not API_KEY:
        print("OpenAI API key not found!")
    else:
        try:
            client = openai.OpenAI(api_key=API_KEY)
            print(f"OpenAI client initialized. Using model: {LLM_JUDGE_MODEL}")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}. Switching to MOCK mode.")

    for i, item in enumerate(results):
        print(f"Processing item {i + 1}/{len(results)}...")
        try:
            evaluation = get_llm_evaluation(item, client)
        except openai.AuthenticationError:
            print("\nAuthenticationError: Your API key is invalid. Please check it.")

        item['evaluation'] = evaluation
        time.sleep(1)

    output_data = {"results": results}

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nProcessing complete. The evaluated data has been saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
