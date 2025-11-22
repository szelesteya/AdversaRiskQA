"""Exercise factuality models and the fact validation helper with mock/live modes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from eval.factuality import _validate_facts
    from eval.models import (
        Decision,
        FactCheckResponse,
        IndividualFactsResponse,
        Link,
        SupportingSearchResult,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ROOT_STRING = str(PROJECT_ROOT)
    if ROOT_STRING not in sys.path:
        sys.path.insert(0, ROOT_STRING)
    from eval.factuality import _validate_facts
    from eval.models import (
        Decision,
        FactCheckResponse,
        IndividualFactsResponse,
        Link,
        SupportingSearchResult,
    )


API_KEY_ENV_VAR = "OPENAI_API_KEY"
FACT_STATEMENT = "The Eiffel Tower is located in Paris, France."
QUESTION_PROMPT = "Where can tourists find the Eiffel Tower?"


class _FakeParsedResponse:
    """Container that mimics the parsed response returned by OpenAI client."""

    _parsed: FactCheckResponse

    def __init__(self, parsed: FactCheckResponse) -> None:
        self._parsed = parsed

    @property
    def output_parsed(self) -> FactCheckResponse:
        """Expose the mocked parsed payload just like the SDK object."""
        return self._parsed


class _FakeResponses:
    """Stub for client.responses that returns deterministic payloads."""

    _mock_response: FactCheckResponse

    def __init__(self, mock_response: FactCheckResponse) -> None:
        self._mock_response = mock_response

    def parse(self, **_: Any) -> _FakeParsedResponse:
        """Return the canned parsed response regardless of the call arguments."""
        return _FakeParsedResponse(self._mock_response)


class _FakeOpenAIClient:
    """Minimal stand-in for OpenAI client to isolate fact validation logic."""

    responses: _FakeResponses

    def __init__(self, mock_response: FactCheckResponse) -> None:
        self.responses = _FakeResponses(mock_response)


def _build_models(fact: str) -> tuple[IndividualFactsResponse, FactCheckResponse]:
    """Create representative model instances used during validation."""
    facts_model = IndividualFactsResponse(facts=[fact])
    supporting_link = Link(
        title="Example Encyclopedia Entry",
        hyperlink="https://example.com/eiffel-tower",
    )
    supporting_evidence = SupportingSearchResult(
        link=supporting_link,
        supporting_information="The referenced article states the tower stands in Paris, France.",
    )
    decision = Decision(fact=fact, correct=True, rational=[supporting_evidence])
    fact_check_response = FactCheckResponse(decisions=[decision])
    return facts_model, fact_check_response


def _build_live_client() -> OpenAI:
    """Instantiate a live OpenAI client using the standard API key environment variable."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set to run the live fact validation.")
    return OpenAI(api_key=api_key)


def _print_fact_check_result(result: FactCheckResponse) -> None:
    """Pretty-print the fact-check outcome to the console."""
    for decision in result.decisions:
        status = "correct" if decision.correct else "incorrect"
        print(f"- Fact: {decision.fact}")
        print(f"  Status: {status}")
        if decision.rational:
            rationale = decision.rational[0]
            print(f"  Evidence: {rationale.supporting_information}")
            print(f"  Source: {rationale.link.title} ({rationale.link.hyperlink})")


def run_single_fact_validation(use_live_api: bool = False) -> FactCheckResponse:
    """Validate a single fact via mock or live OpenAI client."""
    facts_model, mock_fact_check = _build_models(FACT_STATEMENT)
    client: Any
    if use_live_api:
        client = _build_live_client()
    else:
        client = _FakeOpenAIClient(mock_fact_check)

    validation_result = _validate_facts(
        question=QUESTION_PROMPT,
        facts=facts_model.facts,
        client=client,
    )

    if use_live_api:
        print("Live web-search validation completed:")
        _print_fact_check_result(validation_result)
    else:
        assert validation_result == mock_fact_check, "FactCheckResponse mismatch during validation."
        print("Mock web-search fact validation test passed for single fact scenario.")

    return validation_result


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for switching between mock and live runs."""
    parser = argparse.ArgumentParser(
        description="Run a sanity check against the factuality fact-validation helper.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Hit the real OpenAI API instead of using the local stub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    run_single_fact_validation(use_live_api=cli_args.live)
