# Module 04: PII Detection and Redaction

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 01-03, Python |

---

## Learning Objectives

By the end of this module you will be able to:

1. Explain what PII is and why detecting it in LLM outputs matters.
2. Use Microsoft Presidio to detect named entities (names, emails, SSNs, phone numbers, credit cards).
3. Implement redaction (replace PII with type labels) and anonymization (replace with masked values).
4. Configure confidence thresholds and custom entity recognizers.
5. Build a regex-based fallback detector for environments without spaCy.

---

## 1. Why PII Detection?

LLMs can inadvertently leak PII in several ways:

- **Training data memorization** -- regurgitating names, emails, or phone numbers from training data.
- **Context window leakage** -- including PII from one user's context in another user's response.
- **Over-helpful responses** -- generating realistic-looking PII (e.g., example SSNs that happen to be real).

**Regulatory frameworks** like GDPR, HIPAA, and CCPA require that PII be handled carefully.
A PII detection layer on LLM outputs is a compliance requirement for most production systems.

### Common PII Entity Types

| Entity | Pattern | Example |
|---|---|---|
| PERSON | Named entity recognition | "John Smith" |
| EMAIL_ADDRESS | user@domain.tld | "john@company.com" |
| PHONE_NUMBER | Various formats | "(555) 123-4567" |
| US_SSN | NNN-NN-NNNN | "123-45-6789" |
| CREDIT_CARD | 16-digit patterns | "4111-1111-1111-1111" |
| IP_ADDRESS | IPv4/IPv6 | "192.168.1.1" |
| LOCATION | Named entity recognition | "123 Main St, NYC" |
| DATE_TIME | Various date formats | "March 15, 1990" |

---

## 2. Microsoft Presidio: Setup and Basics

```bash
# Install Presidio and spaCy model
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

### Basic Detection

```python
"""Detect PII using Microsoft Presidio."""

from presidio_analyzer import AnalyzerEngine


def detect_pii(text: str, language: str = "en") -> list[dict]:
    """
    Detect PII entities in text.

    Returns list of dicts with entity_type, start, end, and score.
    """
    analyzer = AnalyzerEngine()

    results = analyzer.analyze(
        text=text,
        entities=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IP_ADDRESS",
            "LOCATION",
        ],
        language=language,
    )

    return [
        {
            "entity_type": r.entity_type,
            "text": text[r.start : r.end],
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
        }
        for r in results
    ]


# ----- Demo -----
if __name__ == "__main__":
    sample = (
        "Please contact John Smith at john.smith@example.com "
        "or call (555) 867-5309. His SSN is 123-45-6789."
    )

    entities = detect_pii(sample)
    for e in entities:
        print(f"  [{e['entity_type']}] '{e['text']}' (score: {e['score']})")
```

---

## 3. Redaction and Anonymization

### Redaction: Replace PII with Labels

```python
"""Redact PII by replacing entities with type labels."""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


def redact_pii(text: str) -> str:
    """Replace all PII with <ENTITY_TYPE> labels."""
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language="en")

    operators = {}
    for r in results:
        operators[r.entity_type] = OperatorConfig(
            "replace", {"new_value": f"<{r.entity_type}>"}
        )

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    return anonymized.text


# ----- Demo -----
text = "Email sarah@corp.com or call 555-123-4567. SSN: 123-45-6789."
print(redact_pii(text))
# Email <EMAIL_ADDRESS> or call <PHONE_NUMBER>. SSN: <US_SSN>.
```

### Anonymization: Replace with Masked Values

```python
"""Anonymize PII by masking characters."""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


def anonymize_pii(text: str, masking_char: str = "*") -> str:
    """Replace PII with masked characters."""
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language="en")

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig("mask", {
                "masking_char": masking_char,
                "chars_to_mask": 100,
                "from_end": False,
            }),
        },
    )

    return anonymized.text


print(anonymize_pii("Call John at 555-123-4567"))
# Call **** at ************
```

---

## 4. Configuring Confidence Thresholds

```python
"""Tune PII detection sensitivity with score thresholds."""

from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

text = "Dr. Smith will see you on Monday at the clinic."

# Low threshold (catches more, including false positives)
results_low = analyzer.analyze(text=text, language="en", score_threshold=0.3)

# High threshold (fewer results, higher precision)
results_high = analyzer.analyze(text=text, language="en", score_threshold=0.85)

print(f"Low threshold:  {len(results_low)} entities found")
print(f"High threshold: {len(results_high)} entities found")

# In production, start with threshold=0.5 and tune based on your data
```

---

## 5. Custom Entity Recognizers

Add domain-specific PII detection (e.g., internal employee IDs):

```python
"""Add a custom recognizer for internal employee IDs."""

from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
)

# Define custom pattern: employee IDs like "EMP-12345"
emp_id_pattern = Pattern(
    name="employee_id_pattern",
    regex=r"\bEMP-\d{5}\b",
    score=0.9,
)

emp_id_recognizer = PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    patterns=[emp_id_pattern],
    supported_language="en",
)

# Add to analyzer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(emp_id_recognizer)

# Detect
text = "Assign ticket to EMP-42195 in the engineering department."
results = analyzer.analyze(text=text, language="en")

for r in results:
    print(f"  [{r.entity_type}] '{text[r.start:r.end]}' (score: {r.score})")
```

---

## 6. Regex Fallback (No spaCy Required)

When Presidio / spaCy cannot be installed, use the regex fallback from
`src/validators/pii_detector.py`:

```python
from src.validators.pii_detector import PIIDetector

# Create detector (automatically uses regex if Presidio is unavailable)
detector = PIIDetector(confidence_threshold=0.5)

text = (
    "John's SSN is 123-45-6789. "
    "Email: john@company.com. "
    "Card: 4111-1111-1111-1111."
)

result = detector.detect(text)

print(f"PII found: {result.has_pii}")
print(f"Entity counts: {result.entity_counts}")
print(f"Redacted: {result.redacted_text}")
print(f"Anonymized: {result.anonymized_text}")
```

---

## 7. Hands-On Lab

### Lab: PII-Safe LLM Response Handler

**Objective:** Build a middleware that scans every LLM response for PII before it reaches the user.

1. Create `modules/04-pii-detection/lab/starter/pii_middleware.py`.
2. Implement a `PIIMiddleware` class with:
   - `scan(text) -> PIIScanResult` -- detect all PII entities with scores.
   - `redact(text) -> str` -- replace PII with entity labels.
   - `anonymize(text) -> str` -- replace PII with realistic fake values.
3. Add a custom recognizer for a domain-specific entity (e.g., project codes like `PROJ-XXXX`).
4. Write test cases covering: emails, SSNs, phone numbers, credit cards, names, and your custom entity.
5. Bonus: add a `report()` method that returns statistics on PII detected across multiple calls.

---

## 8. Key Takeaways

- **Microsoft Presidio** provides a production-grade PII detection engine with NER and pattern matching.
- Always **redact or anonymize** PII before logging, storing, or displaying LLM outputs.
- Tune the **confidence threshold** for your use case: high for legal/medical, lower for customer support.
- Add **custom recognizers** for domain-specific sensitive data patterns.
- Keep a **regex fallback** for lightweight deployments where spaCy is not available.

---

## Validation

```bash
bash modules/04-pii-detection/validation/validate.sh
```

---

**Next: [Module 05 -->](../05-hallucination-detection/)**
