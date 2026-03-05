"""
PII Detector
=============
Detects and redacts Personally Identifiable Information (PII) using
Microsoft Presidio.  Supports names, emails, SSNs, phone numbers,
credit cards, IP addresses, and more.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PIIEntity(BaseModel):
    """A single detected PII entity."""

    entity_type: str
    text: str
    start: int
    end: int
    score: float


class PIIDetectionResult(BaseModel):
    """Result of PII detection on a text."""

    has_pii: bool
    entities: list[PIIEntity] = Field(default_factory=list)
    redacted_text: Optional[str] = None
    anonymized_text: Optional[str] = None
    entity_counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regex-based fallback detector (no Presidio required)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "US_PASSPORT": re.compile(r"\b[A-Z]\d{8}\b"),
    "DATE_OF_BIRTH": re.compile(
        r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
    ),
}


def _regex_detect(text: str) -> list[PIIEntity]:
    """Detect PII using regex patterns (fallback when Presidio unavailable)."""
    entities: list[PIIEntity] = []
    for entity_type, pattern in _REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.85,
                )
            )
    return entities


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Detects and redacts PII from text.

    Uses Microsoft Presidio when available, otherwise falls back to regex
    patterns.  Supports configurable entity types and confidence thresholds.

    Example
    -------
    >>> detector = PIIDetector(confidence_threshold=0.8)
    >>> result = detector.detect("Call me at 555-123-4567 or john@example.com")
    >>> assert result.has_pii
    >>> print(result.redacted_text)
    Call me at <PHONE_NUMBER> or <EMAIL_ADDRESS>
    """

    # Default entity types to detect
    DEFAULT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "LOCATION",
        "DATE_TIME",
        "US_PASSPORT",
    ]

    def __init__(
        self,
        entities: Optional[list[str]] = None,
        confidence_threshold: float = 0.5,
        language: str = "en",
    ) -> None:
        self.entities = entities or self.DEFAULT_ENTITIES
        self.confidence_threshold = confidence_threshold
        self.language = language

        if PRESIDIO_AVAILABLE:
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        else:
            self._analyzer = None
            self._anonymizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> PIIDetectionResult:
        """Detect PII entities in *text*."""
        if PRESIDIO_AVAILABLE and self._analyzer is not None:
            return self._detect_with_presidio(text)
        return self._detect_with_regex(text)

    def redact(self, text: str) -> str:
        """Return *text* with all PII replaced by entity-type placeholders."""
        result = self.detect(text)
        return result.redacted_text or text

    def anonymize(self, text: str) -> str:
        """Return *text* with PII replaced by fake / masked values."""
        result = self.detect(text)
        return result.anonymized_text or text

    # ------------------------------------------------------------------
    # Presidio-based detection
    # ------------------------------------------------------------------

    def _detect_with_presidio(self, text: str) -> PIIDetectionResult:
        results: list[RecognizerResult] = self._analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.confidence_threshold,
        )

        entities = [
            PIIEntity(
                entity_type=r.entity_type,
                text=text[r.start : r.end],
                start=r.start,
                end=r.end,
                score=r.score,
            )
            for r in results
        ]

        # Redact (replace with <ENTITY_TYPE>)
        redacted = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                **{
                    entity.entity_type: OperatorConfig(
                        "replace",
                        {"new_value": f"<{entity.entity_type}>"},
                    )
                    for entity in entities
                },
            },
        )

        # Anonymize (mask with asterisks)
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("mask", {
                    "masking_char": "*",
                    "chars_to_mask": 100,
                    "from_end": False,
                }),
            },
        )

        entity_counts: dict[str, int] = {}
        for e in entities:
            entity_counts[e.entity_type] = entity_counts.get(e.entity_type, 0) + 1

        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            entities=entities,
            redacted_text=redacted.text,
            anonymized_text=anonymized.text,
            entity_counts=entity_counts,
        )

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    def _detect_with_regex(self, text: str) -> PIIDetectionResult:
        entities = _regex_detect(text)

        # Redact by replacing matches (process from end to preserve offsets)
        redacted = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            redacted = (
                redacted[: entity.start]
                + f"<{entity.entity_type}>"
                + redacted[entity.end :]
            )

        # Anonymize with asterisks
        anonymized = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            anonymized = (
                anonymized[: entity.start]
                + "*" * len(entity.text)
                + anonymized[entity.end :]
            )

        entity_counts: dict[str, int] = {}
        for e in entities:
            entity_counts[e.entity_type] = entity_counts.get(e.entity_type, 0) + 1

        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            entities=entities,
            redacted_text=redacted,
            anonymized_text=anonymized,
            entity_counts=entity_counts,
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = PIIDetector(confidence_threshold=0.5)

    sample = (
        "John Smith's SSN is 123-45-6789. "
        "Reach him at john.smith@company.com or call 555-867-5309. "
        "His credit card is 4111-1111-1111-1111."
    )

    print("Original:")
    print(sample)

    result = detector.detect(sample)
    print(f"\nPII found: {result.has_pii}")
    print(f"Entity counts: {result.entity_counts}")
    for entity in result.entities:
        print(f"  [{entity.entity_type}] '{entity.text}' (score: {entity.score:.2f})")

    print(f"\nRedacted:\n{result.redacted_text}")
    print(f"\nAnonymized:\n{result.anonymized_text}")
