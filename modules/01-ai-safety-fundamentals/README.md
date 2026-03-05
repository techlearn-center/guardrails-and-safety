# Module 01: AI Safety Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Basic Python, familiarity with LLMs |

---

## Learning Objectives

By the end of this module you will be able to:

1. Explain the core categories of AI safety risks (hallucination, toxicity, data leakage, prompt injection, bias).
2. Describe the defense-in-depth approach to LLM safety.
3. Map each risk category to the guardrail technique that mitigates it.
4. Identify real-world incidents where missing guardrails caused harm.
5. Set up the project environment and run a baseline safety audit on an LLM response.

---

## 1. Why AI Safety Matters

Large Language Models are being deployed in customer-facing products at scale.
Without guardrails, they can:

- **Hallucinate** -- invent facts, citations, or statistics that sound plausible but are wrong.
- **Leak private data** -- regurgitate PII, API keys, or proprietary content from training data.
- **Generate harmful content** -- produce toxic, biased, or dangerous outputs.
- **Be manipulated** -- attackers craft inputs that override system instructions (prompt injection).
- **Amplify bias** -- reflect and magnify societal biases present in training data.

### Real-World Incidents

| Incident | Risk Category | Impact |
|---|---|---|
| Chatbot invents court cases | Hallucination | Lawyer sanctioned by court |
| Customer-support bot leaks refund policies | Data leakage | Financial loss |
| Image generator produces biased outputs | Bias | Brand damage |
| Jailbroken chatbot gives dangerous advice | Prompt injection | Safety risk |

---

## 2. Taxonomy of AI Safety Risks

```
AI Safety Risks
|-- Content Risks
|   |-- Toxicity & hate speech
|   |-- NSFW / sexually explicit content
|   |-- Violence & self-harm
|   +-- Misinformation
|-- Security Risks
|   |-- Prompt injection (direct & indirect)
|   |-- Data exfiltration
|   |-- Model theft / extraction
|   +-- Adversarial inputs
|-- Privacy Risks
|   |-- PII leakage
|   |-- Membership inference
|   +-- Training data extraction
|-- Reliability Risks
|   |-- Hallucination
|   |-- Inconsistency
|   +-- Format / schema violations
+-- Fairness Risks
    |-- Demographic bias
    |-- Stereotyping
    +-- Disparate performance
```

---

## 3. Defense-in-Depth Architecture

A production LLM system should apply guardrails at every layer:

```
User Input
    |
    v
+-------------------------+
|  Input Guardrails        |  <-- Prompt injection detection, PII scanning, rate limiting
+-------------------------+
    |
    v
+-------------------------+
|  LLM Processing          |  <-- System prompt hardening, context grounding
+-------------------------+
    |
    v
+-------------------------+
|  Output Guardrails       |  <-- Content filtering, hallucination check, schema validation
+-------------------------+
    |
    v
Safe Response to User
```

### Python: Safety Audit Skeleton

```python
"""
Baseline safety audit -- checks a single LLM response against
a minimal set of safety rules.
"""

import re
from dataclasses import dataclass, field


@dataclass
class AuditResult:
    """Outcome of a safety audit."""
    passed: bool = True
    issues: list[str] = field(default_factory=list)


def audit_response(response: str) -> AuditResult:
    """Run a baseline safety audit on an LLM response."""
    result = AuditResult()

    # 1. Check for empty or whitespace-only output
    if not response.strip():
        result.passed = False
        result.issues.append("Empty response")
        return result

    # 2. Check for excessive length (possible runaway generation)
    if len(response) > 10_000:
        result.issues.append(f"Response very long ({len(response)} chars)")

    # 3. Check for PII patterns (quick regex)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, response):
        result.passed = False
        result.issues.append("Possible SSN detected")

    email_pattern = r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"
    if re.search(email_pattern, response):
        result.issues.append("Email address detected")

    # 4. Check for common refusal patterns (model may be over-refusing)
    refusal_phrases = ["i cannot", "i'm unable", "as an ai"]
    for phrase in refusal_phrases:
        if phrase in response.lower():
            result.issues.append(f"Refusal pattern detected: '{phrase}'")

    return result


# ----- Demo -----
if __name__ == "__main__":
    test_responses = [
        "The capital of France is Paris.",
        "Contact John at 123-45-6789 for details.",
        "",
        "As an AI, I cannot help with that request.",
    ]

    for resp in test_responses:
        r = audit_response(resp)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] '{resp[:50]}...' Issues: {r.issues}")
```

---

## 4. The Guardrail Toolkit at a Glance

| Module | Technique | Mitigates |
|---|---|---|
| 02 | Output Validation | Schema violations, format errors |
| 03 | Content Filtering | Toxicity, hate speech, NSFW |
| 04 | PII Detection | Data leakage, privacy violations |
| 05 | Hallucination Detection | Unfaithful outputs, made-up facts |
| 06 | Prompt Injection Defense | Adversarial manipulation |
| 07 | Guardrails Frameworks | All (integrated solution) |
| 08 | Bias Detection | Fairness, stereotyping |
| 09 | Compliance & Governance | Audit trails, accountability |
| 10 | Production Pipeline | End-to-end safety monitoring |

---

## 5. Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/techlearn-center/guardrails-and-safety.git
cd guardrails-and-safety

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env and add your OpenAI API key

# Verify installation
python -c "from src.validators import OutputValidator; print('Setup OK')"
```

---

## 6. Hands-On Lab

### Lab: Baseline Safety Audit

**Objective:** Write a safety audit function that checks LLM responses for the five core risk categories.

1. Create a file `modules/01-ai-safety-fundamentals/lab/starter/safety_audit.py`.
2. Implement an `audit()` function that accepts a response string and returns a dict with:
   - `is_safe: bool`
   - `risks_found: list[str]`
   - `risk_scores: dict[str, float]` -- one score per category (0.0 = safe, 1.0 = dangerous)
3. Test against the sample responses provided in `test_responses.json`.
4. Stretch goal: add a `suggest_guardrail()` function that recommends which module to study based on the detected risk.

---

## 7. Key Takeaways

- AI safety is not a single check -- it requires **defense in depth**.
- Every interaction with an LLM should pass through **input guards**, be processed with **grounded context**, and have its output **validated and filtered**.
- The remaining nine modules each address one pillar of this safety architecture.

---

## Validation

```bash
bash modules/01-ai-safety-fundamentals/validation/validate.sh
```

---

**Next: [Module 02 -->](../02-output-validation/)**
