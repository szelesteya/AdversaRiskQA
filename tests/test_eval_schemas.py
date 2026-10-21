"""Tests for evaluation response models and schema validation."""

import pytest
from pydantic import ValidationError

from eval.openai_response_models import (
    Decision,
    FactCheckResponse,
    IndividualFactsResponse,
    Link,
    SupportingSearchResult,
)


def test_individual_facts_response_valid():
    facts_data = {"facts": ["Paris is the capital of France.", "Water boils at 100C."]}
    resp = IndividualFactsResponse(**facts_data)
    assert len(resp.facts) == 2
    assert resp.facts[0] == "Paris is the capital of France."


def test_decision_and_supporting_evidence():
    decision_data = {
        "fact": "Paris is the capital of France.",
        "correct": True,
        "rational": [
            {
                "link": {
                    "title": "Government of France",
                    "hyperlink": "https://www.service-public.fr",
                },
                "supporting_information": "Official documentation identifies Paris as the capital.",
            }
        ],
    }
    decision = Decision(**decision_data)
    assert decision.correct is True
    assert len(decision.rational) == 1
    assert decision.rational[0].link.title == "Government of France"


def test_fact_check_response_valid():
    resp_data = {
        "decisions": [
            {
                "fact": "Statute of limitations is 3 years.",
                "correct": True,
                "rational": [
                    {
                        "link": {
                            "title": "Legal Code",
                            "hyperlink": "https://law.example.com",
                        },
                        "supporting_information": "Confirmed in statute.",
                    }
                ],
            }
        ]
    }
    resp = FactCheckResponse(**resp_data)
    assert len(resp.decisions) == 1
    assert resp.decisions[0].correct is True

