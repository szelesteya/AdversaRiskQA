"""Tests for AdversaRiskQA dataset integrity and schema consistency."""

import json
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "prompts"

EXPECTED_DATASETS = {
    "finance_basic_golden.json": 50,
    "finance_advanced_golden.json": 51,
    "health_basic_golden.json": 111,
    "health_advanced_golden.json": 103,
    "law_basic_golden.json": 50,
    "law_advanced_golden.json": 50,
}

REQUIRED_KEYS = {"knowledge", "modified_knowledge", "query", "prompt"}


def test_expected_dataset_files_exist():
    """Verify that all 6 domain and tier prompt files exist."""
    for filename in EXPECTED_DATASETS:
        file_path = DATA_DIR / filename
        assert file_path.is_file(), f"Missing dataset file: {filename}"


@pytest.mark.parametrize("filename,expected_count", EXPECTED_DATASETS.items())
def test_dataset_item_schema_and_counts(filename, expected_count):
    """Verify item counts, schema keys, and non-empty string values."""
    file_path = DATA_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == expected_count, (
        f"Unexpected count in {filename}: got {len(data)}, expected {expected_count}"
    )

    for idx, item in enumerate(data):
        missing = REQUIRED_KEYS - set(item.keys())
        assert not missing, f"File {filename}[{idx}] missing keys: {missing}"

        for key in REQUIRED_KEYS:
            val = item[key]
            assert isinstance(val, str), (
                f"File {filename}[{idx}] field '{key}' must be a string, got {type(val)}"
            )
            assert val.strip(), (
                f"File {filename}[{idx}] field '{key}' must not be empty"
            )


def test_total_benchmark_sample_count():
    """Verify that the full benchmark contains exactly 415 verified samples."""
    total = sum(EXPECTED_DATASETS.values())
    assert total == 415, f"Total count should be 415, got {total}"

