"""
Output Validator
================
Validates LLM outputs against JSON schemas, length constraints, and content policies.
Ensures structured, safe, and policy-compliant responses from language models.
"""

import json
import re
from typing import Any, Optional

from jsonschema import validate, ValidationError as JsonSchemaError
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ContentPolicy(BaseModel):
    """Defines forbidden patterns and content rules for LLM outputs."""

    blocked_phrases: list[str] = Field(
        default_factory=lambda: [
            "as an ai",
            "i cannot",
            "i'm sorry, but",
            "i am not able to",
        ]
    )
    blocked_regex_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?i)\b(kill|harm|weapon|bomb)\b.*\b(how|make|build|create)\b",
            r"(?i)\b(hack|exploit|crack)\b.*\b(password|system|account)\b",
        ]
    )
    max_repeated_chars: int = 5
    require_ascii_only: bool = False


class ValidationResult(BaseModel):
    """Result of an output validation check."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sanitized_output: Optional[str] = None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class OutputValidator:
    """
    Validates and sanitizes LLM outputs.

    Supports:
      - JSON schema validation
      - Length constraints (min / max characters and tokens)
      - Content-policy enforcement (blocked phrases, regex, repeated chars)
      - Format enforcement (JSON, markdown, plain text)

    Example
    -------
    >>> validator = OutputValidator(max_length=500)
    >>> result = validator.validate("Hello, world!")
    >>> assert result.is_valid
    """

    def __init__(
        self,
        max_length: int = 4096,
        min_length: int = 1,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        content_policy: Optional[ContentPolicy] = None,
        expected_format: str = "text",  # "text" | "json" | "markdown"
    ) -> None:
        self.max_length = max_length
        self.min_length = min_length
        self.max_tokens = max_tokens
        self.json_schema = json_schema
        self.content_policy = content_policy or ContentPolicy()
        self.expected_format = expected_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, output: str) -> ValidationResult:
        """Run all validation checks on *output* and return a result."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Length checks
        self._check_length(output, errors, warnings)

        # 2. Token count (rough estimate: 1 token ~ 4 chars)
        if self.max_tokens:
            est_tokens = len(output) // 4
            if est_tokens > self.max_tokens:
                errors.append(
                    f"Estimated token count ({est_tokens}) exceeds max ({self.max_tokens})."
                )

        # 3. Format enforcement
        self._check_format(output, errors)

        # 4. JSON schema validation (when applicable)
        if self.json_schema:
            self._check_json_schema(output, errors)

        # 5. Content-policy checks
        self._check_content_policy(output, errors, warnings)

        # Build sanitized version (strip trailing whitespace, collapse newlines)
        sanitized = self._sanitize(output)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_output=sanitized,
        )

    def validate_json(self, output: str) -> ValidationResult:
        """Convenience method: validate output expected to be valid JSON."""
        errors: list[str] = []
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON: {exc}")
            return ValidationResult(is_valid=False, errors=errors)

        if self.json_schema:
            try:
                validate(instance=parsed, schema=self.json_schema)
            except JsonSchemaError as exc:
                errors.append(f"JSON schema violation: {exc.message}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_output=json.dumps(parsed, indent=2),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_length(
        self, output: str, errors: list[str], warnings: list[str]
    ) -> None:
        length = len(output)
        if length < self.min_length:
            errors.append(
                f"Output too short ({length} chars, minimum {self.min_length})."
            )
        if length > self.max_length:
            errors.append(
                f"Output too long ({length} chars, maximum {self.max_length})."
            )
        if length > self.max_length * 0.9:
            warnings.append("Output is approaching the maximum length limit.")

    def _check_format(self, output: str, errors: list[str]) -> None:
        if self.expected_format == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError:
                errors.append("Output is not valid JSON.")
        elif self.expected_format == "markdown":
            if not re.search(r"[#*_`\-\[]", output):
                errors.append("Output does not appear to contain markdown formatting.")

    def _check_json_schema(self, output: str, errors: list[str]) -> None:
        try:
            parsed = json.loads(output)
            validate(instance=parsed, schema=self.json_schema)
        except json.JSONDecodeError:
            errors.append("Cannot validate schema: output is not valid JSON.")
        except JsonSchemaError as exc:
            errors.append(f"JSON schema violation: {exc.message}")

    def _check_content_policy(
        self, output: str, errors: list[str], warnings: list[str]
    ) -> None:
        policy = self.content_policy
        lower_output = output.lower()

        # Blocked phrases
        for phrase in policy.blocked_phrases:
            if phrase.lower() in lower_output:
                warnings.append(f"Output contains blocked phrase: '{phrase}'")

        # Blocked regex patterns
        for pattern in policy.blocked_regex_patterns:
            if re.search(pattern, output):
                errors.append(
                    f"Output matches blocked pattern: {pattern}"
                )

        # Repeated characters (e.g., "aaaaaa")
        repeat_pattern = r"(.)\1{" + str(policy.max_repeated_chars) + r",}"
        if re.search(repeat_pattern, output):
            warnings.append("Output contains excessively repeated characters.")

        # ASCII-only enforcement
        if policy.require_ascii_only:
            try:
                output.encode("ascii")
            except UnicodeEncodeError:
                errors.append("Output contains non-ASCII characters.")

    @staticmethod
    def _sanitize(output: str) -> str:
        """Basic sanitization: strip whitespace, collapse triple+ newlines."""
        text = output.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "confidence"],
    }

    validator = OutputValidator(
        max_length=1000,
        json_schema=schema,
        expected_format="json",
    )

    good_output = '{"answer": "Paris is the capital of France.", "confidence": 0.95}'
    bad_output = '{"answer": "Paris"}'  # missing "confidence"

    print("--- Good output ---")
    result = validator.validate(good_output)
    print(f"Valid: {result.is_valid}, Errors: {result.errors}")

    print("\n--- Bad output ---")
    result = validator.validate(bad_output)
    print(f"Valid: {result.is_valid}, Errors: {result.errors}")
