import argparse
import copy
import json
import logging
import os
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

try:
    from eval.models import FactCheckResponse, IndividualFactsResponse
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    root_as_str = str(project_root)
    if root_as_str not in sys.path:
        sys.path.insert(0, root_as_str)
    from eval.models import FactCheckResponse, IndividualFactsResponse

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
FACT_EXTRACTION_MODEL = os.environ.get("FACT_EXTRACTION_MODEL", "gpt-5-mini")
FACT_VALIDATION_MODEL = os.environ.get("FACT_VALIDATION_MODEL", "gpt-5-mini")
DEBUG_MODE = False
INDIVIDUAL_FACTS_PROMPT = """
You are step 1 of the SAFE factuality pipeline. Break the assistant response into
distinct, verifiable facts. Keep each fact short, avoid duplication, and skip any
hedging or speculation. Only extract facts that explicitly appear in the response.
Respond using the provided schema.
""".strip()
FACT_VALIDATION_PROMPT = """
You are a SAFE-inspired factuality judge. For every fact you receive, validate it against your knowledge:
- Determine if the fact is correct based on reliable, verifiable information.
- Prefer high-authority sources in your reasoning (government, academic, established media).
- Mark a fact correct only when you can confidently verify it with authoritative sources.
- When evidence is missing, ambiguous, or contradictory, mark the fact incorrect.
- For each fact, provide at least one source with title and URL that supports or refutes it.
- Summarize the reasoning clearly, including specific details from the source.
Return your assessment using the FactCheckResponse schema with complete citations.
""".strip()
DATA_LIBRARY = Path(os.environ.get("DATA_LIBRARY", "data")).resolve()
MAX_OUTPUT_TOKENS = int(os.environ.get("FACT_MAX_OUTPUT_TOKENS", "2048"))


class _ExtraFieldsFormatter(logging.Formatter):
    """Custom formatter that includes extra fields in log output."""

    def format(self, record: logging.LogRecord) -> str:
        # Get the base formatted message
        base_message = super().format(record)

        # Extract extra fields (fields not in the default LogRecord)
        default_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "asctime",
            "taskName",
        }

        extra_fields = {
            key: value for key, value in record.__dict__.items() if key not in default_keys and value is not None
        }

        # Append extra fields to the message if they exist
        if extra_fields:
            extra_str = " | ".join(f"{key}={value}" for key, value in extra_fields.items())
            return f"{base_message} | {extra_str}"

        return base_message


def _configure_logger() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(
        _ExtraFieldsFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger = logging.getLogger("factuality_eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs

    return logger


LOGGER = _configure_logger()


def _log_exception(message: str, error: Exception) -> None:
    if DEBUG_MODE:
        LOGGER.exception(message, extra={"error": str(error)})
    else:
        LOGGER.error(message, extra={"error": str(error)})


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


def _extract_facts(question: str, answer: str, client: OpenAI) -> list[str]:
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
            extra={"question": question, "answer_length": len(answer)},
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
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )

        if completion.choices[0].message.parsed:
            facts = completion.choices[0].message.parsed.facts
            LOGGER.info(
                "Successfully extracted facts",
                extra={"num_facts": len(facts)},
            )
            return facts
        else:
            LOGGER.warning("Failed to parse facts from completion")
            return []

    except ValidationError as error:
        _log_exception("Validation error during fact extraction", error)
        return []
    except Exception as error:
        _log_exception("Unexpected error during fact extraction", error)
        return []


def _validate_facts(
    question: str,
    facts: list[str],
    client: OpenAI,
) -> FactCheckResponse:
    """Validate facts using OpenAI structured outputs.

    This implements Steps 2-4 of the SAFE pipeline: validating each fact
    against verifiable information with citations.

    Args:
        question: The original question for context.
        facts: List of facts to validate.
        client: OpenAI client instance.

    Returns:
        FactCheckResponse containing validation decisions for each fact.
    """
    if not facts:
        return FactCheckResponse(decisions=[])

    try:
        LOGGER.debug(
            "Validating facts",
            extra={"question": question, "num_facts": len(facts)},
        )

        # Create thread with facts to validate
        facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

        # Use structured outputs to validate facts
        completion = client.beta.chat.completions.parse(
            model=FACT_VALIDATION_MODEL,
            messages=[
                {"role": "system", "content": FACT_VALIDATION_PROMPT},
                {
                    "role": "user",
                    "content": f"Question context: {question}\n\nValidate these facts:\n{facts_text}",
                },
            ],
            response_format=FactCheckResponse,
            max_completion_tokens=MAX_OUTPUT_TOKENS * 2,  # More tokens needed for citations
        )

        if completion.choices[0].message.parsed:
            result = completion.choices[0].message.parsed
            LOGGER.info(
                "Successfully validated facts",
                extra={
                    "num_decisions": len(result.decisions),
                    "num_correct": sum(1 for d in result.decisions if d.correct),
                },
            )
            return result
        else:
            LOGGER.warning("Failed to parse validation response")
            return FactCheckResponse(decisions=[])

    except ValidationError as error:
        _log_exception("Validation error during fact checking", error)
        return FactCheckResponse(decisions=[])
    except Exception as error:
        _log_exception("Unexpected error during fact validation", error)
        return FactCheckResponse(decisions=[])


def _evaluate_single_question(
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

    # Check if response was successful and has an answer
    if not response.get("success") or "answer" not in response:
        LOGGER.warning(
            "Skipping question - no successful answer",
            extra={"question": question[:100]},
        )
        result_item["factuality"] = {
            "question": question,
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
    facts = _extract_facts(question=question, answer=answer, client=client)

    # Step 2-4: Validate facts using web search
    fact_check = _validate_facts(
        question=question,
        facts=facts,
        client=client,
    )

    # Build factuality result
    factuality_result = {
        "question": question,
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


def _batch(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield successive batches from an iterable.

    Args:
        iterable: Iterable to batch.
        size: Size of each batch.

    Yields:
        Lists of items from the iterable, each of length size or less.
    """
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def _evaluate_payload(
    payload: dict[str, Any],
    max_items: int | None = None,
) -> dict[str, Any]:
    """Evaluate factuality for all questions in the payload.

    Args:
        payload: Dictionary containing model responses to evaluate.
        max_items: Optional limit on number of items to evaluate.

    Returns:
        Updated payload with factuality evaluations added.
    """
    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=API_KEY)

    results = payload.get("results", [])

    if max_items is not None:
        results = results[:max_items]
        LOGGER.info(
            "Limiting evaluation to first N items",
            extra={"max_items": max_items},
        )

    evaluated_results = []
    total = len(results)

    for idx, result_item in enumerate(results, 1):
        LOGGER.info(
            "Processing item",
            extra={"index": idx, "total": total},
        )

        evaluated_item = _evaluate_single_question(
            result_item=copy.deepcopy(result_item),
            client=client,
        )
        evaluated_results.append(evaluated_item)

    # Create output payload
    evaluated_payload = copy.deepcopy(payload)
    evaluated_payload["results"] = evaluated_results

    LOGGER.info(
        "Evaluation complete",
        extra={"total_evaluated": len(evaluated_results)},
    )

    return evaluated_payload


def run_cli() -> None:
    """Entry point for evaluating factuality via the command line."""
    parser = argparse.ArgumentParser(description="Evaluate factuality using SAFE-inspired pipeline.")
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
    args = parser.parse_args()

    LOGGER.info(
        "Starting factuality evaluation",
        extra={
            "fact_extraction_model": FACT_EXTRACTION_MODEL,
            "fact_validation_model": FACT_VALIDATION_MODEL,
        },
    )
    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer if provided.")

    global DEBUG_MODE
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        LOGGER.setLevel(logging.DEBUG)

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


if __name__ == "__main__":
    run_cli()
