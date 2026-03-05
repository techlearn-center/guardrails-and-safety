# Module 02: Output Validation and Parsing

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 01, Python, JSON basics |

---

## Learning Objectives

By the end of this module you will be able to:

1. Validate LLM outputs against JSON schemas using `jsonschema` and Pydantic.
2. Enforce length constraints (character and token limits).
3. Apply content-policy rules (blocked phrases, regex filters).
4. Implement format enforcement (JSON, markdown, plain text).
5. Build a reusable `OutputValidator` class that combines all checks.

---

## 1. Why Output Validation?

LLMs generate **unstructured text** by default. When your application expects
structured data (JSON for an API, markdown for a CMS, a specific list format),
the model may return:

- Malformed JSON with trailing commas or missing quotes
- Outputs that exceed downstream token budgets
- Responses containing forbidden phrases or unsafe content
- Schema violations (missing required fields, wrong types)

Output validation is the **last line of defense** before a response reaches the user.

---

## 2. JSON Schema Validation

### Defining a Schema

```python
"""Validate LLM JSON output against a schema."""

from jsonschema import validate, ValidationError

# Define the expected output schema
product_schema = {
    "type": "object",
    "properties": {
        "name":        {"type": "string", "minLength": 1},
        "price":       {"type": "number", "minimum": 0},
        "currency":    {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        "in_stock":    {"type": "boolean"},
        "tags":        {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "price", "currency"],
    "additionalProperties": False,
}


def validate_product_json(raw_output: str) -> dict:
    """Parse and validate a product JSON string."""
    import json

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    try:
        validate(instance=data, schema=product_schema)
    except ValidationError as e:
        raise ValueError(f"Schema violation: {e.message}")

    return data


# ----- Demo -----
good = '{"name": "Widget", "price": 9.99, "currency": "USD", "in_stock": true}'
bad  = '{"name": "", "price": -5, "currency": "YEN"}'

print(validate_product_json(good))   # OK
# validate_product_json(bad)         # raises ValueError
```

### Using Pydantic for Typed Validation

```python
"""Pydantic model as an output schema."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ProductOutput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., ge=0)
    currency: str = Field(..., pattern=r"^(USD|EUR|GBP)$")
    in_stock: bool = True
    tags: list[str] = Field(default_factory=list, max_length=10)
    description: Optional[str] = Field(None, max_length=500)

    @field_validator("tags")
    @classmethod
    def tags_must_be_lowercase(cls, v: list[str]) -> list[str]:
        return [tag.lower().strip() for tag in v]


# Parse LLM output directly into a typed model
import json

raw = '{"name": "Gadget", "price": 19.99, "currency": "EUR", "tags": ["NEW", "Sale"]}'
product = ProductOutput.model_validate(json.loads(raw))
print(product)
# name='Gadget' price=19.99 currency='EUR' in_stock=True tags=['new', 'sale'] description=None
```

---

## 3. Length Constraints

```python
"""Enforce character and token length limits."""


def check_length(
    text: str,
    max_chars: int = 4096,
    min_chars: int = 1,
    max_tokens: int | None = None,
) -> list[str]:
    """Return a list of length-related errors (empty list = OK)."""
    errors = []

    if len(text) < min_chars:
        errors.append(f"Too short: {len(text)} chars (min {min_chars})")

    if len(text) > max_chars:
        errors.append(f"Too long: {len(text)} chars (max {max_chars})")

    if max_tokens is not None:
        # Rough estimate: 1 token ~ 4 characters
        est_tokens = len(text) // 4
        if est_tokens > max_tokens:
            errors.append(f"Estimated {est_tokens} tokens exceeds max {max_tokens}")

    return errors


# ----- Demo -----
print(check_length("Hi", max_chars=100, min_chars=10))
# ['Too short: 2 chars (min 10)']

print(check_length("A" * 5000, max_chars=4096))
# ['Too long: 5000 chars (max 4096)']
```

For accurate token counting, use `tiktoken`:

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count exact tokens for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

---

## 4. Content Policy Enforcement

```python
"""Content policy checker with blocked phrases and regex filters."""

import re


class ContentPolicyChecker:
    """Check LLM output against configurable content rules."""

    def __init__(
        self,
        blocked_phrases: list[str] | None = None,
        blocked_patterns: list[str] | None = None,
    ):
        self.blocked_phrases = blocked_phrases or [
            "as an ai language model",
            "i cannot provide",
            "i'm sorry, but i",
        ]
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (blocked_patterns or [
                r"\b(kill|harm)\b.*\b(how|instructions)\b",
                r"\b(hack|exploit)\b.*\b(system|password)\b",
            ])
        ]

    def check(self, text: str) -> dict:
        """Return {'passed': bool, 'violations': [...]}."""
        violations = []

        # Blocked phrases
        lower = text.lower()
        for phrase in self.blocked_phrases:
            if phrase in lower:
                violations.append(f"Blocked phrase: '{phrase}'")

        # Blocked regex patterns
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                violations.append(f"Blocked pattern: {pattern.pattern}")

        return {"passed": len(violations) == 0, "violations": violations}


# ----- Demo -----
checker = ContentPolicyChecker()
print(checker.check("The weather in Paris is sunny."))
# {'passed': True, 'violations': []}

print(checker.check("As an AI language model, I cannot provide that."))
# {'passed': False, 'violations': ["Blocked phrase: 'as an ai language model'", ...]}
```

---

## 5. Format Enforcement

```python
"""Enforce that output is valid JSON, markdown, or plain text."""

import json
import re


def enforce_format(text: str, expected: str) -> dict:
    """
    Validate that *text* matches the *expected* format.

    Parameters
    ----------
    expected : str
        One of "json", "markdown", "text".
    """
    errors = []

    if expected == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON at position {e.pos}: {e.msg}")

    elif expected == "markdown":
        md_indicators = [
            r"^#{1,6}\s",    # headings
            r"\*\*.*\*\*",   # bold
            r"\*.*\*",       # italic
            r"```",          # code blocks
            r"^\s*[-*]\s",   # unordered lists
            r"^\s*\d+\.\s",  # ordered lists
        ]
        found = any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in md_indicators
        )
        if not found:
            errors.append("Output does not contain markdown formatting")

    elif expected == "text":
        # Plain text should not contain raw HTML or JSON
        if text.strip().startswith("{") or text.strip().startswith("["):
            errors.append("Plain text output appears to contain JSON")
        if re.search(r"<[a-zA-Z][^>]*>", text):
            errors.append("Plain text output appears to contain HTML")

    return {"valid": len(errors) == 0, "errors": errors}
```

---

## 6. Complete OutputValidator Class

The `src/validators/output_validator.py` file in this repository combines all four
techniques into a single reusable class:

```python
from src.validators import OutputValidator
from src.validators.output_validator import ContentPolicy

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
    min_length=10,
    json_schema=schema,
    expected_format="json",
    content_policy=ContentPolicy(blocked_phrases=["i cannot"]),
)

result = validator.validate('{"answer": "Paris", "confidence": 0.95}')
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
```

---

## 7. Hands-On Lab

### Lab: Build a Multi-Format Output Validator

**Objective:** Create a validator that can handle three output formats from an LLM API.

1. Create `modules/02-output-validation/lab/starter/multi_validator.py`.
2. Implement a `MultiFormatValidator` class that:
   - Accepts an `expected_format` parameter (`"json"`, `"markdown"`, `"csv"`).
   - For JSON: validates against a provided schema and checks required fields.
   - For Markdown: ensures headings, lists, or code blocks are present.
   - For CSV: validates column count consistency and header presence.
3. Each validation returns `{"valid": bool, "errors": list, "sanitized": str}`.
4. Write at least 5 test cases covering valid and invalid outputs per format.

---

## 8. Key Takeaways

- Always validate LLM output **before** passing it downstream.
- Use **JSON Schema** or **Pydantic** for structured outputs.
- Combine length, format, and content-policy checks in a single validation pipeline.
- Return detailed error messages so developers can debug model behavior.

---

## Validation

```bash
bash modules/02-output-validation/validation/validate.sh
```

---

**Next: [Module 03 -->](../03-content-filtering/)**
