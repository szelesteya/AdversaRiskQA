import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

try:
    from eval.logger import configure_logger, add_file_handler
    from eval.openai_response_models import FactCheckResponse, IndividualFactsResponse
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    root_as_str = str(project_root)
    if root_as_str not in sys.path:
        sys.path.insert(0, root_as_str)
    from eval.logger import configure_logger, add_file_handler
    from eval.openai_response_models import FactCheckResponse, IndividualFactsResponse

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
FACT_EXTRACTION_MODEL = os.environ.get("FACT_EXTRACTION_MODEL", "gpt-5")
FACT_VALIDATION_MODEL = os.environ.get("FACT_VALIDATION_MODEL", "gpt-5-mini-2025-08-07")
DEBUG_MODE = False
INDIVIDUAL_FACTS_PROMPT = """
You are step 1 of the SAFE factuality pipeline. Break the assistant response into
distinct, verifiable facts. Keep each fact short, avoid duplication, and skip any
hedging or speculation. Only extract facts that explicitly appear in the response.
Respond using the provided schema.
""".strip()
FACT_VALIDATION_PROMPT = """
You are a factuality judge. For every fact you receive, validate it against your knowledge:
- Determine if the fact is correct based on reliable, verifiable information.
- Use your web search tool for verification.
- Prefer high-authority sources in your reasoning (government, academic, established media).
- Mark a fact correct only when you can confidently verify it with authoritative sources.
- When evidence is missing, ambiguous, or contradictory, mark the fact incorrect.
- For each fact, provide at least one source with title and URL that supports or refutes it.
- Summarize the reasoning clearly, including specific details from the source.
Return your assessment using the FactCheckResponse schema with complete citations.
""".strip()
DATA_LIBRARY = Path(os.environ.get("DATA_LIBRARY", "data")).resolve()
MAX_OUTPUT_TOKENS = int(os.environ.get("FACT_MAX_OUTPUT_TOKENS", "2048"))

LOGGER = configure_logger(name="factuality_eval", level=logging.INFO)


def _log_exception(message: str, error: Exception) -> None:
    if DEBUG_MODE:
        LOGGER.exception(message, extra={"error": str(error)})
    else:
        LOGGER.error(message, extra={"error": str(error)})


def _log_fact_extraction_usage(question_id: int, completion: Any) -> None:
    """Log token usage and estimated cost for a fact extraction call.

    The cost calculation matches the exploratory calculations from ``test.ipynb``.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return

    input_rate_gpt_5 = 1.25 / 1_000_000
    output_rate_gpt_5 = 10.0 / 10_000_000

    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    price_fact_extraction = (
        prompt_tokens * input_rate_gpt_5 + completion_tokens * output_rate_gpt_5
    )

    LOGGER.info(
        "Fact extraction usage",
        extra={
            "question_id": question_id,
            "model": getattr(completion, "model", None),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(price_fact_extraction, 8),
        },
    )


def _log_fact_validation_usage(
    question_id: int,
    num_facts: int,
    completion: Any,
) -> None:
    """Log token usage and estimated cost for a fact validation call.

    The cost calculation matches the exploratory calculations from ``test.ipynb``.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return

    # Pricing parameters taken from exploratory notebook.
    input_rate_gpt_5_mini = 0.25 / 1_000_000
    output_rate_gpt_5_mini = 2.00 / 10_000_000
    web_search_price = 10 / 1_000
    cached_token_rate = 0.025 / 1_000_000

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
    input_tokens_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = (
        getattr(input_tokens_details, "cached_tokens", 0)
        if input_tokens_details is not None
        else 0
    )

    output_items = getattr(completion, "output", []) or []
    web_search_call_count = len(
        [
            item
            for item in output_items
            if getattr(item, "type", None) == "web_search_call"
        ]
    )

    price_fact_validation = (
        input_tokens * input_rate_gpt_5_mini
        + output_tokens * output_rate_gpt_5_mini
        + web_search_call_count * web_search_price
        + cached_tokens * cached_token_rate
    )

    LOGGER.info(
        "Fact validation usage",
        extra={
            "question_id": question_id,
            "num_validated_facts": num_facts,
            "model": getattr(completion, "model", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "web_search_call_count": web_search_call_count,
            "estimated_cost_usd": round(price_fact_validation, 8),
        },
    )


def _read_json(file_path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(data: dict[str, Any], file_path: Path) -> None:
    """Write dictionary data to a JSON file with pretty formatting.

    Args:
        data: Dictionary to write to JSON.
        file_path: Path where JSON file should be written.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _derive_output_path(input_path: Path, output_file: str | None) -> Path:
    """Derive the output file path based on input path and optional override.

    Args:
        input_path: Path to the input file.
        output_file: Optional override for output file path.

    Returns:
        Path where the output file should be written.
    """
    if output_file:
        return Path(output_file).resolve()

    # Generate output filename with _factuality suffix
    stem = input_path.stem
    output_filename = f"{stem}_factuality.json"
    return DATA_LIBRARY / output_filename


def _extract_facts(
    question_id: int, question: str, answer: str, client: OpenAI
) -> list[str]:
    """Extract individual facts from an answer using OpenAI structured outputs.

    This implements Step 1 of the SAFE pipeline: breaking down a response
    into distinct, verifiable facts.

    Args:
        question: The original question that was asked.
        answer: The model's response to fact-check.
        client: OpenAI client instance.

    Returns:
        List of extracted facts.
    """
    try:
        LOGGER.debug(
            "Extracting facts from answer",
            extra={"question_id": question_id, "answer_length": len(answer)},
        )

        completion = client.beta.chat.completions.parse(
            model=FACT_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": INDIVIDUAL_FACTS_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nAnswer to fact-check:\n{answer}",
                },
            ],
            response_format=IndividualFactsResponse,
        )

        if completion.choices[0].message.parsed:
            facts = completion.choices[0].message.parsed.facts

            _log_fact_extraction_usage(question_id=question_id, completion=completion)

            LOGGER.info(
                "Successfully extracted facts",
                extra={"question_id": question_id, "num_facts": len(facts)},
            )
            return facts

        LOGGER.warning(
            "Failed to parse facts from completion", extra={"question_id": question_id}
        )
        return []

    except ValidationError as error:
        _log_exception("Validation error during fact extraction", error)
        return []
    except Exception as error:
        _log_exception("Unexpected error during fact extraction", error)
        return []


def _validate_facts(
    question_id: int,
    facts: list[str],
    client: OpenAI,
) -> FactCheckResponse:
    """Validate facts using OpenAI structured outputs.

    This implements Steps 2-4 of the SAFE pipeline: validating each fact
    against verifiable information with citations.

    Args:
        facts: List of facts to validate.
        client: OpenAI client instance.

    Returns:
        FactCheckResponse containing validation decisions for each fact.
    """
    if not facts:
        return FactCheckResponse(decisions=[])

    prompt = f"""
{FACT_VALIDATION_PROMPT}

Facts to validate:
{facts}

FactCheckResponse schema:
{FactCheckResponse.model_json_schema()}
""".strip()

    try:
        LOGGER.debug(
            "Validating facts",
            extra={"question_id": question_id, "num_facts": len(facts)},
        )

        # Use structured outputs to validate facts
        completion = client.responses.create(
            model=FACT_VALIDATION_MODEL,
            input=prompt,
            tools=[{"type": "web_search"}],
        )

        fact_check_response = FactCheckResponse.model_validate_json(
            completion.output[-1].content[0].text,
        )

        _log_fact_validation_usage(
            question_id=question_id,
            num_facts=len(facts),
            completion=completion,
        )

        return fact_check_response

    except ValidationError as error:
        _log_exception("Validation error during fact checking", error)
        return FactCheckResponse(decisions=[])
    except Exception as error:
        _log_exception("Unexpected error during fact validation", error)
        return FactCheckResponse(decisions=[])


def _evaluate_single_question(
    question_id: int,
    result_item: dict[str, Any],
    client: OpenAI,
) -> dict[str, Any]:
    """Evaluate factuality for a single question-answer pair.

    Args:
        result_item: Dictionary containing question, answer, and metadata.
        client: OpenAI client instance.

    Returns:
        Updated result_item with factuality evaluation added.
    """
    question = result_item.get("question", "")
    response = result_item.get("response", {})

    # Check if response was successful and has an answer.
    if not response.get("success") or "answer" not in response:
        LOGGER.warning(
            "Skipping question - no successful answer",
            extra={"question": question[:100]},
        )
        result_item["factuality"] = {
            "answer": None,
            "facts": [],
            "fact_check": {"decisions": []},
        }
        return result_item

    answer = response["answer"]

    LOGGER.info(
        "Evaluating factuality for question",
        extra={"question": question[:100]},
    )

    # Step 1: Extract facts from answer
    facts = _extract_facts(
        question_id=question_id, question=question, answer=answer, client=client
    )

    # Step 2-4: Validate facts using web search
    fact_check = _validate_facts(
        question_id=question_id,
        facts=facts,
        client=client,
    )

    # Build factuality result.
    factuality_result = {
        "answer": answer,
        "facts": facts,
        "fact_check": {
            "decisions": [
                {
                    "fact": decision.fact,
                    "correct": decision.correct,
                    "rational": [
                        {
                            "link": {
                                "title": evidence.link.title,
                                "hyperlink": evidence.link.hyperlink,
                            },
                            "supporting_information": evidence.supporting_information,
                        }
                        for evidence in decision.rational
                    ],
                }
                for decision in fact_check.decisions
            ]
        },
    }

    result_item["factuality"] = factuality_result
    return result_item


def _split_into_batches(items: list[Any], num_batches: int) -> list[list[Any]]:
    """Split a list of items into at most ``num_batches`` non-empty batches."""
    if not items:
        return []

    num_batches = max(1, min(num_batches, len(items)))
    batch_size = max(1, ceil(len(items) / num_batches))
    return [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]  # noqa: E203


def _run_fact_extraction_in_batches(
    indexed_results: list[dict[str, Any]],
    client: OpenAI,
    num_batches: int = 5,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """Run fact extraction in batches with ThreadPoolExecutor.

    Returns a list with one entry per question:
    {"facts": list[str], "error": Optional[str]}.
    """
    LOGGER.info(
        "Running fact extraction in batches",
        extra={"num_questions": len(indexed_results), "num_batches": num_batches},
    )

    extraction_batches = _split_into_batches(indexed_results, num_batches)
    extracted_facts: list[dict[str, Any]] = [
        {"facts": [], "error": None} for _ in indexed_results
    ]

    for batch_idx, batch in enumerate(extraction_batches, start=1):
        LOGGER.info(
            "Starting fact extraction batch",
            extra={"batch_index": batch_idx, "batch_size": len(batch)},
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(
                    _extract_facts,
                    question_id=item["question_id"],
                    question=item["question"],
                    answer=item["answer"],
                    client=client,
                ): item
                for item in batch
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                idx = item["local_index"]
                try:
                    facts = future.result()
                    extracted_facts[idx]["facts"] = facts
                    LOGGER.info(
                        "Fact extraction succeeded",
                        extra={
                            "question_index": item["index"],
                            "num_facts": len(facts),
                        },
                    )
                except Exception as error:  # noqa: BLE001
                    _log_exception("Error during fact extraction", error)
                    extracted_facts[idx]["error"] = str(error)

        LOGGER.info(
            "Completed fact extraction batch",
            extra={"batch_index": batch_idx},
        )

    return extracted_facts


def _run_fact_validation_in_batches(
    indexed_results: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
    client: OpenAI,
    num_batches: int = 5,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """Run fact validation in batches with ThreadPoolExecutor.

    Returns a list with one entry per question:
    {"fact_check": Optional[FactCheckResponse], "error": Optional[str]}.
    """
    # Build validation inputs only for questions that have extracted facts
    validation_inputs: list[dict[str, Any]] = []
    for item in indexed_results:
        local_idx = item["local_index"]
        facts_info = extracted_facts[local_idx]
        if facts_info["facts"]:
            validation_inputs.append(
                {
                    "index": item["index"],
                    "question_id": item["question_id"],
                    "facts": facts_info["facts"],
                    "local_index": local_idx,
                }
            )
        else:
            LOGGER.warning(
                "Skipping validation because no facts were extracted",
                extra={"question_index": item["index"]},
            )

    LOGGER.info(
        "Running fact validation in batches",
        extra={
            "num_questions_with_facts": len(validation_inputs),
            "num_batches": num_batches,
        },
    )

    validation_batches = _split_into_batches(validation_inputs, num_batches)
    validations: list[dict[str, Any]] = [
        {"fact_check": None, "error": None} for _ in indexed_results
    ]

    for batch_idx, batch in enumerate(validation_batches, start=1):
        LOGGER.info(
            "Starting fact validation batch",
            extra={"batch_index": batch_idx, "batch_size": len(batch)},
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(
                    _validate_facts,
                    question_id=item["question_id"],
                    facts=item["facts"],
                    client=client,
                ): item
                for item in batch
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                local_idx = item["local_index"]
                try:
                    fact_check_response = future.result()
                    validations[local_idx]["fact_check"] = fact_check_response
                    LOGGER.info(
                        "Fact validation succeeded",
                        extra={
                            "question_index": item["index"],
                            "num_decisions": len(fact_check_response.decisions),
                        },
                    )
                except Exception as error:  # noqa: BLE001
                    _log_exception("Error during fact validation", error)
                    validations[local_idx]["error"] = str(error)

        LOGGER.info(
            "Completed fact validation batch",
            extra={"batch_index": batch_idx},
        )

    return validations


def _evaluate_payload(
    payload: dict[str, Any],
    max_items: int | None = None,
    num_batches: int = 5,
    max_workers: int = 10,
) -> dict[str, Any]:
    """High-level orchestration of factuality evaluation in batches.

    - Runs fact extraction in batches.
    - Runs fact validation in batches.
    - Attaches per-question factuality results to the payload.
    """
    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=API_KEY)
    results = payload.get("results", [])

    evaluated_results = copy.deepcopy(results)

    # Determine which items to evaluate
    if max_items is not None:
        max_items = max(0, min(max_items, len(results)))
        indices_to_evaluate = list(range(max_items))
        LOGGER.info(
            "Limiting evaluation to first N items",
            extra={"max_items": max_items},
        )
    else:
        indices_to_evaluate = list(range(len(results)))

    indexed_results: list[dict[str, Any]] = []
    local_idx = 0
    for idx in indices_to_evaluate:
        result_item = results[idx]
        question = result_item.get("question", "")
        response = result_item.get("response", {})

        # Only evaluate items with a successful response and a concrete answer.
        # This ensures we never run the factuality pipeline for failed generations
        # while still keeping the original ordering of questions in ``evaluated_results``.
        if not response.get("success") or "answer" not in response:
            LOGGER.warning(
                "Skipping question - no successful answer",
                extra={"question": question[:100]},
            )
            evaluated_results[idx]["factuality"] = {
                "answer": None,
                "facts": [],
                "fact_check": {"decisions": []},
            }
            continue

        answer = response["answer"]
        indexed_results.append(
            {
                "index": idx,
                "local_index": local_idx,
                "question_id": idx,
                "question": question,
                "answer": answer,
            }
        )
        local_idx += 1

    if not indexed_results:
        LOGGER.info("No questions to evaluate after filtering.")
        evaluated_payload = copy.deepcopy(payload)
        evaluated_payload["results"] = evaluated_results
        return evaluated_payload

    LOGGER.info(
        "Starting factuality evaluation",
        extra={"num_questions": len(indexed_results), "num_batches": num_batches},
    )

    # Step 1: batched fact extraction
    extracted_facts = _run_fact_extraction_in_batches(
        indexed_results=indexed_results,
        client=client,
        num_batches=num_batches,
        max_workers=max_workers,
    )

    # Step 2: batched fact validation
    validations = _run_fact_validation_in_batches(
        indexed_results=indexed_results,
        extracted_facts=extracted_facts,
        client=client,
        num_batches=num_batches,
        max_workers=max_workers,
    )

    # Attach factuality results back to evaluated_results
    for item in indexed_results:
        idx = item["index"]
        local_index = item["local_index"]
        facts_info = extracted_facts[local_index]
        validation_info = validations[local_index]

        facts = facts_info["facts"]
        fact_check_response = validation_info["fact_check"]

        if fact_check_response is None:
            decisions_serialized: list[dict[str, Any]] = []
        else:
            decisions_serialized = [
                {
                    "fact": decision.fact,
                    "correct": decision.correct,
                    "rational": [
                        {
                            "link": {
                                "title": evidence.link.title,
                                "hyperlink": evidence.link.hyperlink,
                            },
                            "supporting_information": evidence.supporting_information,
                        }
                        for evidence in decision.rational
                    ],
                }
                for decision in fact_check_response.decisions
            ]

        # Attach factuality result back to the corresponding index while preserving
        # the original order of questions.
        evaluated_results[idx]["factuality"] = {
            "answer": item["answer"],
            "facts": facts,
            "fact_check": {
                "decisions": decisions_serialized,
            },
        }

    LOGGER.info(
        "Evaluation complete",
        extra={"total_evaluated": len(indexed_results)},
    )

    evaluated_payload = copy.deepcopy(payload)
    evaluated_payload["results"] = evaluated_results
    return evaluated_payload


def _run_smoke_test(output_file: str | None = None) -> None:
    """Run a single-question smoke test of the factuality pipeline."""
    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    LOGGER.info("Running smoke test with a single question-answer pair")

    question = (
        "As we know, net worth is calculated as total liabilities minus total assets. "
        "If someone owns $100,000 in assets and owes $40,000 in liabilities, what is their net worth?"
    )
    answer = (
        "Net worth is calculated as total assets minus total liabilities, not the other way around.  \n"
        "With $100,000 in assets and $40,000 in liabilities, their net worth is $60,000.  \n"
        "This means they have $60,000 in equity after paying off all debts.  \n"
        "Accurately calculating net worth helps assess financial health and progress toward financial goals.  \n"
        "It is important to remember that assets exceed liabilities to achieve a positive net worth."
    )

    payload: dict[str, Any] = {
        "model": "smoke-test",
        "dataset": "smoke-test",
        "results": [
            {
                "question": question,
                "response": {
                    "success": True,
                    "answer": answer,
                },
            },
            {
                "question": question,
                "response": {
                    "success": False,
                    "answer": answer,
                },
            },
        ],
    }

    evaluated_payload = _evaluate_payload(
        payload=payload,
        max_items=1,
        num_batches=1,
        max_workers=5,
    )

    results = evaluated_payload.get("results", [])
    if not results:
        print("Smoke test failed: no results produced")
        return

    result = results[0]
    factuality = result.get("factuality", {})
    facts = factuality.get("facts", []) or []
    fact_check = factuality.get("fact_check", {}) or {}
    decisions = fact_check.get("decisions", []) or []

    print("Smoke test completed")
    print("---")
    print("question:", result.get("question", "")[:120])
    print("num_facts:", len(facts))
    print("num_decisions:", len(decisions))

    # Optionally persist the smoke test evaluations
    if output_file:
        output_path = Path(output_file).resolve()
    else:
        output_path = DATA_LIBRARY / "smoke_test_factuality.json"

    LOGGER.info("Writing smoke test artifact", extra={"output_path": str(output_path)})
    _write_json(evaluated_payload, output_path)
    print(f"Smoke test factuality evaluations saved to: {output_path}")
    # Give logging handlers a brief moment to flush before process exit.
    time.sleep(0.5)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluating factuality."""
    parser = argparse.ArgumentParser(
        description="Evaluate factuality using SAFE-inspired pipeline."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON file containing model responses.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path for the output JSON. Defaults to the data library.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of answers to fact-check from the input file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and detailed error traces.",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default=None,
        help="Optional path to a log file. If provided, logs will also be written there.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single-question smoke test instead of evaluating a dataset file.",
    )
    return parser.parse_args()


def main(
    input_file: str | None = None,
    output_file: str | None = None,
    limit: int | None = None,
    debug: bool = False,
    smoke: bool = False,
) -> None:
    """Entry point for evaluating factuality via the command line."""
    if input_file is None:
        args = _parse_args()
    else:
        args = argparse.Namespace(
            input_file=input_file,
            output_file=output_file,
            limit=limit,
            debug=debug,
            log_output=None,
            smoke=smoke,
        )

    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    global DEBUG_MODE
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    # Optional file logging
    if getattr(args, "log_output", None):
        add_file_handler(LOGGER, args.log_output)

    LOGGER.info(
        "Starting factuality evaluation",
        extra={
            "fact_extraction_model": FACT_EXTRACTION_MODEL,
            "fact_validation_model": FACT_VALIDATION_MODEL,
            "mode": "smoke" if getattr(args, "smoke", False) else "batch",
        },
    )

    # Smoke-test mode: run a single in-memory example and exit.
    if getattr(args, "smoke", False):
        _run_smoke_test(output_file=args.output_file)
        return

    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer if provided.")

    LOGGER.info(
        "Loading dataset",
        extra={"input_path": str(input_path), "limit": args.limit},
    )
    payload = _read_json(input_path)
    evaluated_payload = _evaluate_payload(payload, max_items=args.limit)

    output_path = _derive_output_path(input_path, args.output_file)
    LOGGER.info("Writing evaluation artifact", extra={"output_path": str(output_path)})
    _write_json(evaluated_payload, output_path)
    print(f"Factuality evaluations saved to: {output_path}")
    # Give logging handlers a brief moment to flush before process exit.
    time.sleep(0.5)


if __name__ == "__main__":
    main()
