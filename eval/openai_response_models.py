from pydantic import BaseModel, Field


class IndividualFactsResponse(BaseModel):
    """Response model for extracting individual facts from a model response.

    This represents Step 1 of the SAFE factuality pipeline: breaking down
    a response into atomic, verifiable facts.
    """

    facts: list[str] = Field(
        ...,
        description=(
            "Distinct, non-overlapping factual statements extracted from the model's "
            "response. Each fact should be: (1) atomic and self-contained, (2) verifiable "
            "through external sources, (3) free of hedging language like 'may' or 'could', "
            "(4) faithful to the original wording without adding interpretation. "
            "Exclude opinions, speculation, or redundant statements."
        ),
    )


class Link(BaseModel):
    """Metadata for a source citation used to verify a fact."""

    title: str = Field(
        ...,
        description=(
            "Human-readable title for the cited source. Use the actual page/article "
            "title from the website, not a generic description."
        ),
    )
    hyperlink: str = Field(
        ...,
        description="Direct URL pointing to the evidence. Must be a complete, valid URL.",
    )


class SupportingSearchResult(BaseModel):
    """Evidence from web search that supports or refutes a fact."""

    link: Link = Field(
        ...,
        description=(
            "Metadata that allows citing the evidence. Should reference high-authority "
            "sources such as government websites, academic institutions, established media, "
            "or domain experts."
        ),
    )
    supporting_information: str = Field(
        ...,
        description=(
            "One or two sentences summarizing how the cited source supports or "
            "refutes the fact. Include relevant statistics, direct quotes, or specific "
            "details from the source when possible. Be precise and avoid vague summaries."
        ),
    )


class Decision(BaseModel):
    """Verdict on whether a single fact is supported by reliable evidence."""

    fact: str = Field(
        ...,
        description=("Original fact under evaluation. Must match exactly one of the facts " "provided for validation."),
    )
    correct: bool = Field(
        ...,
        description=(
            "True when at least one reputable source explicitly confirms the fact, "
            "false when sources refute it or when evidence is missing/inconclusive. "
            "Be conservative: if evidence is ambiguous or contradictory, mark as false. "
            "Prefer authoritative sources (e.g., .gov, .edu, established news) over "
            "low-quality ones."
        ),
    )
    rational: list[SupportingSearchResult] = Field(
        ...,
        description=(
            "Chain of supporting evidence produced after running targeted web searches. "
            "Include at least one source that directly addresses the fact. If the fact is "
            "marked correct, include sources that confirm it. If marked incorrect, include "
            "sources that refute it or explain why it cannot be verified. Leave empty only "
            "if absolutely no relevant sources exist after exhaustive search."
        ),
    )


class FactCheckResponse(BaseModel):
    """Complete factuality assessment for all facts in a response.

    This represents Steps 2-4 of the SAFE pipeline: validating each fact
    against web search results and determining correctness.
    """

    decisions: list[Decision] = Field(
        ...,
        description=(
            "Ordered factuality verdicts corresponding to each extracted fact. "
            "The number of decisions must match the number of facts provided for validation. "
            "Each decision should have the same fact text as provided in the input."
        ),
    )
