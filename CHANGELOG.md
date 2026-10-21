# Changelog

All notable changes to the AdversaRiskQA project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-21

### Added
- **Hugging Face Hub Integration**: Released the benchmark on Hugging Face ([`dudderr/adversarisk-qa`](https://huggingface.co/datasets/dudderr/adversarisk-qa)) featuring typed Parquet splits (`finance_basic`, `finance_advanced`, `health_basic`, `health_advanced`, `law_basic`, `law_advanced`) and unified benchmark table (`adversarisk_qa_test.parquet`).
- **Standardized Evaluation CLI**: Added standalone `evaluate.py` entrypoint in the export bundle for running both LLM-as-a-judge adversality evaluation and SAFE factuality pipelines.
- **Unit Test Suite**: Added automated tests under `tests/` (`test_benchmark_dataset.py`, `test_eval_schemas.py`) validating prompt schema integrity, non-empty fields, and Pydantic evaluation models.
- **Academic Citation**: Added `CITATION.cff` for GitHub 1-click citation and Zenodo integration.
- **License**: Added formal MIT license file for open scientific use.
- **Automated Tooling**: Reusable dataset preparation and Hugging Face upload scripts in `scripts/`.

### Changed
- **Schema Normalization**: Standardized all prompt JSON files in `data/prompts/` to use the unified key `modified_knowledge` (normalizing previous occurrences of `"modified knowledge"`).
- **Robustness in Evaluator**: Updated `eval/adversality.py` to gracefully support both `modified_knowledge` and legacy `modified knowledge` keys.
- **Documentation**: Updated `README.md` with arXiv badges, baseline results leaderboard table, Hugging Face quickstart, and reproducibility guide.

---

## [0.1.0] - 2026-01-22 (arXiv Submission Snapshot)

### Added
- Original research codebase accompanying the arXiv preprint:
  > **AdversaRiskQA: An Adversarial Factuality Benchmark for High-Risk Domains**  
  > *Adam Szelestey, Sofie van Engelen, Tianhao Huang, Justin Snelders, Qintao Zeng, Songgaojun Deng*  
  > [arXiv:2601.15511](https://arxiv.org/abs/2601.15511)
- 415 verified benchmark prompts across **Health** (214), **Finance** (101), and **Law** (100) at **Basic** and **Advanced** difficulty tiers.
- vLLM model answer generation scripts (`execute_prompts/`).
- Automated evaluators: LLM-as-a-Judge adversality evaluator and Search-Augmented Factuality Evaluator (SAFE).
- Analysis notebooks comparing Qwen (4B, 30B, 80B), GPT-OSS (20B, 120B), and GPT-5 model families.
- *Tag: `v0.1.0-arxiv` points to the exact repository state at submission.*

