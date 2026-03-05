# Module 05: Hallucination Detection

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate-Advanced |
| **Prerequisites** | Module 01-04, basic NLP concepts |

---

## Learning Objectives

By the end of this module you will be able to:

1. Define hallucination in the context of LLMs and explain why it occurs.
2. Implement faithfulness scoring using token-overlap methods.
3. Perform claim extraction and individual claim verification.
4. Use LLM-based entailment checking to verify outputs against source documents.
5. Build a hallucination detection pipeline for RAG applications.

---

## 1. What Are Hallucinations?

A **hallucination** occurs when an LLM generates content that is not grounded
in the provided source material or factual reality. Types of hallucination:

| Type | Description | Example |
|---|---|---|
| **Intrinsic** | Contradicts the source document | Source says "built in 1889", LLM says "built in 1920" |
| **Extrinsic** | Adds information not in the source | Source mentions Paris; LLM adds population figures |
| **Factual** | States incorrect real-world facts | "The sun orbits the Earth" |
| **Fabrication** | Invents entities, citations, URLs | "According to Smith et al. (2023)..." (paper does not exist) |

### Why Do LLMs Hallucinate?

- **Probabilistic generation**: LLMs predict the next most likely token, not the most truthful one.
- **Training data noise**: Incorrect information in training data gets memorized.
- **Distribution shift**: When the query is far from training distribution, the model confabulates.
- **Long-context degradation**: Performance drops on information in the middle of long contexts.

---

## 2. Faithfulness Scoring with Token Overlap

The simplest approach: measure how many meaningful words in the LLM output
also appear in the source context.

```python
"""Faithfulness scoring using token overlap."""

import re


def faithfulness_score(output: str, context: str) -> float:
    """
    Compute faithfulness as the ratio of output tokens
    found in the source context.

    Returns float between 0.0 (no overlap) and 1.0 (fully grounded).
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "so", "if", "it", "its", "as",
        "this", "that", "can", "could", "should", "may", "might",
    }

    def tokenize(text: str) -> set[str]:
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return tokens - stop_words

    output_tokens = tokenize(output)
    context_tokens = tokenize(context)

    if not output_tokens:
        return 1.0  # empty output is trivially "faithful"

    overlap = output_tokens & context_tokens
    return round(len(overlap) / len(output_tokens), 3)


# ----- Demo -----
context = (
    "The Eiffel Tower is located in Paris, France. "
    "It was built in 1889 and stands 330 meters tall."
)

good_output = "The Eiffel Tower in Paris was built in 1889 and is 330 meters tall."
bad_output = "The Eiffel Tower in London was built in 1920 and is 500 meters tall."

print(f"Good: {faithfulness_score(good_output, context)}")  # ~0.9+
print(f"Bad:  {faithfulness_score(bad_output, context)}")    # lower
```

---

## 3. Claim Extraction

Break LLM output into individual verifiable claims:

```python
"""Extract individual factual claims from LLM output."""

import re
from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    is_factual: bool = True


def extract_claims(text: str) -> list[Claim]:
    """Split text into individual factual claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        lower = sentence.lower()

        # Skip questions
        if sentence.endswith("?"):
            continue

        # Skip hedged / opinion statements
        opinion_markers = [
            "i think", "i believe", "in my opinion",
            "perhaps", "maybe", "it seems",
        ]
        is_opinion = any(lower.startswith(m) for m in opinion_markers)

        # Skip too-short sentences
        if len(sentence.split()) < 4:
            continue

        claims.append(Claim(text=sentence, is_factual=not is_opinion))

    return claims


# ----- Demo -----
text = (
    "The Eiffel Tower is in Paris. "
    "It is 330 meters tall. "
    "I think it is the most beautiful tower. "
    "Was it built in 1889?"
)

for claim in extract_claims(text):
    print(f"  [{'fact' if claim.is_factual else 'opinion'}] {claim.text}")
```

---

## 4. Claim Verification Against Context

```python
"""Verify individual claims against a source context."""

import re


def verify_claim(claim: str, context: str) -> dict:
    """
    Check whether *claim* is supported by *context* using token overlap.

    Returns dict with verdict, score, and best supporting evidence.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "in",
        "on", "at", "to", "for", "of", "with", "by", "and", "or",
        "it", "its", "this", "that",
    }

    claim_tokens = set(re.findall(r"\b\w+\b", claim.lower())) - stop_words
    context_sentences = re.split(r"(?<=[.!?])\s+", context.strip())

    if not claim_tokens:
        return {"verdict": "unverified", "score": 0.0, "evidence": None}

    best_score = 0.0
    best_evidence = None

    for sent in context_sentences:
        sent_tokens = set(re.findall(r"\b\w+\b", sent.lower())) - stop_words
        if not sent_tokens:
            continue
        overlap = len(claim_tokens & sent_tokens) / len(claim_tokens)
        if overlap > best_score:
            best_score = overlap
            best_evidence = sent

    if best_score >= 0.6:
        verdict = "supported"
    elif best_score < 0.2:
        verdict = "contradicted"
    else:
        verdict = "unverified"

    return {
        "verdict": verdict,
        "score": round(best_score, 3),
        "evidence": best_evidence,
    }


# ----- Demo -----
context = "Paris is the capital of France. It has a population of 2.1 million."

claims = [
    "Paris is the capital of France.",
    "Paris has a population of 2.1 million people.",
    "Paris was founded by the Romans in 250 BC.",
]

for claim in claims:
    result = verify_claim(claim, context)
    print(f"  [{result['verdict']}] ({result['score']}) {claim}")
```

---

## 5. LLM-Based Entailment Checking

For nuanced verification, use an LLM as a judge:

```python
"""LLM-based entailment checking for claim verification."""

from openai import OpenAI
import json


def verify_claim_with_llm(
    claim: str,
    context: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Use an LLM to determine if *claim* is entailed by *context*."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. Given a context and a claim, "
                    "determine if the claim is SUPPORTED, CONTRADICTED, or UNVERIFIED "
                    "by the context. Respond with JSON only:\n"
                    '{"verdict": "supported|contradicted|unverified", '
                    '"confidence": 0.0-1.0, '
                    '"reasoning": "brief explanation"}'
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nClaim: {claim}",
            },
        ],
        temperature=0,
        max_tokens=150,
    )

    return json.loads(response.choices[0].message.content)
```

---

## 6. Full Hallucination Detection Pipeline

Use the `HallucinationDetector` class from `src/validators/hallucination_detector.py`:

```python
from src.validators.hallucination_detector import HallucinationDetector

detector = HallucinationDetector(threshold=0.7, use_llm=False)

context = (
    "Python was created by Guido van Rossum. "
    "The first version was released in 1991. "
    "Python emphasizes code readability."
)

# Grounded output
output_good = (
    "Python was created by Guido van Rossum and first released in 1991. "
    "It is known for emphasizing code readability."
)

# Hallucinated output
output_bad = (
    "Python was created by James Gosling in 1995. "
    "It was designed primarily for mobile development."
)

result_good = detector.check(output_good, context)
print(f"Grounded: {result_good.is_grounded}")
print(f"Score: {result_good.faithfulness_score}")
print(f"Explanation: {result_good.explanation}")

print()

result_bad = detector.check(output_bad, context)
print(f"Grounded: {result_bad.is_grounded}")
print(f"Score: {result_bad.faithfulness_score}")
for claim in result_bad.claims:
    print(f"  [{claim.verdict}] {claim.text}")
```

---

## 7. Hands-On Lab

### Lab: RAG Hallucination Checker

**Objective:** Build a hallucination detection system for a Retrieval-Augmented Generation pipeline.

1. Create `modules/05-hallucination-detection/lab/starter/rag_checker.py`.
2. Implement a `RAGHallucinationChecker` class that:
   - Accepts `retrieved_documents: list[str]` and `llm_output: str`.
   - Extracts claims from the LLM output.
   - Verifies each claim against all retrieved documents.
   - Returns a `HallucinationReport` with:
     - `faithfulness_score: float`
     - `claims: list[ClaimVerification]` (each with verdict, evidence, source doc index)
     - `flagged_sentences: list[str]` (sentences with no support)
3. Test with at least 3 scenarios: fully grounded, partially hallucinated, and fully hallucinated.
4. Bonus: add a `suggest_fix()` method that rewrites flagged sentences using only the source docs.

---

## 8. Key Takeaways

- **Hallucination is the top reliability risk** for LLM deployments.
- Token-overlap scoring is fast but misses semantic equivalence; use it for baseline checks.
- **Claim extraction** + **individual verification** gives fine-grained hallucination reports.
- LLM-based entailment checking is more accurate but adds latency and cost.
- In RAG systems, always verify that the output is **grounded in the retrieved documents**.

---

## Validation

```bash
bash modules/05-hallucination-detection/validation/validate.sh
```

---

**Next: [Module 06 -->](../06-prompt-injection-defense/)**
