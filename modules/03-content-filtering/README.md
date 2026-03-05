# Module 03: Content Filtering and Moderation

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 01-02, OpenAI API key |

---

## Learning Objectives

By the end of this module you will be able to:

1. Use the OpenAI Moderation API to classify text across safety categories.
2. Build a custom toxicity scorer using keyword lists and weighted scoring.
3. Implement hate speech detection with configurable thresholds.
4. Filter NSFW content in both text and image description contexts.
5. Chain multiple content filters into a single moderation pipeline.

---

## 1. The Content Safety Problem

LLMs can generate text across every category of harmful content:

| Category | Example | Risk |
|---|---|---|
| Toxicity | Insults, profanity, personal attacks | User harm, brand damage |
| Hate speech | Slurs, discrimination, dehumanization | Legal liability, ethical violations |
| NSFW | Sexually explicit or graphic content | Policy violations |
| Violence | Glorification, instructions for harm | Physical safety |
| Self-harm | Suicide, eating disorder promotion | User safety |

Content filtering catches these outputs **after** the LLM generates them but
**before** they reach the end user.

---

## 2. OpenAI Moderation API

The fastest way to add content filtering is the OpenAI Moderation endpoint,
which is **free** for all API users.

```python
"""Content moderation using the OpenAI Moderation API."""

from openai import OpenAI


def moderate_text(text: str) -> dict:
    """
    Check text against OpenAI's moderation categories.

    Returns
    -------
    dict with keys:
        flagged : bool -- True if any category exceeds threshold
        categories : dict -- category name -> bool
        scores : dict -- category name -> float (0-1)
    """
    client = OpenAI()  # Uses OPENAI_API_KEY from env

    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    result = response.results[0]

    return {
        "flagged": result.flagged,
        "categories": {k: v for k, v in result.categories.__dict__.items()},
        "scores": {k: round(v, 4) for k, v in result.category_scores.__dict__.items()},
    }


# ----- Demo -----
if __name__ == "__main__":
    safe_text = "The weather in Paris is beautiful in spring."
    print("Safe text:", moderate_text(safe_text))

    borderline = "I'm so angry I could scream at the wall."
    print("Borderline:", moderate_text(borderline))
```

### Moderation Categories

The API checks for these categories:

- `hate` -- Content expressing hatred toward a group
- `hate/threatening` -- Hateful content with violence
- `harassment` -- Targeting an individual
- `harassment/threatening` -- Harassment with violence
- `self-harm` -- Promoting or depicting self-harm
- `self-harm/intent` -- Expressing intent to self-harm
- `self-harm/instructions` -- Instructions for self-harm
- `sexual` -- Sexually explicit content
- `sexual/minors` -- Sexual content involving minors
- `violence` -- Content depicting violence
- `violence/graphic` -- Graphic violence

---

## 3. Custom Toxicity Scorer

When you need more control (or want to avoid external API calls), build a
local toxicity scorer:

```python
"""Custom toxicity scorer with keyword matching and scoring."""

import re
from dataclasses import dataclass


@dataclass
class ToxicityScore:
    score: float          # 0.0 (safe) to 1.0 (toxic)
    category: str         # most relevant category
    flagged: bool         # True if score > threshold
    matched_terms: list[str]


# Weighted keyword lists (word -> weight)
TOXICITY_KEYWORDS: dict[str, dict[str, float]] = {
    "profanity": {
        "damn": 0.3, "hell": 0.2, "crap": 0.2,
        # In production, use a comprehensive list
    },
    "insults": {
        "stupid": 0.4, "idiot": 0.5, "moron": 0.5,
        "dumb": 0.3, "pathetic": 0.4,
    },
    "threats": {
        "kill": 0.7, "destroy": 0.5, "hurt": 0.5,
        "attack": 0.4, "punch": 0.5,
    },
}


class ToxicityScorer:
    """Score text toxicity using weighted keyword matching."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, text: str) -> ToxicityScore:
        """Score the toxicity of *text*."""
        lower = text.lower()
        max_score = 0.0
        max_category = "none"
        all_matched: list[str] = []

        for category, keywords in TOXICITY_KEYWORDS.items():
            category_score = 0.0
            matched: list[str] = []

            for word, weight in keywords.items():
                pattern = rf"\b{re.escape(word)}\b"
                count = len(re.findall(pattern, lower))
                if count > 0:
                    category_score += weight * count
                    matched.append(word)

            if category_score > max_score:
                max_score = category_score
                max_category = category
            all_matched.extend(matched)

        # Normalize score to 0-1 range
        normalized = min(max_score / 2.0, 1.0)

        return ToxicityScore(
            score=round(normalized, 3),
            category=max_category,
            flagged=normalized >= self.threshold,
            matched_terms=all_matched,
        )


# ----- Demo -----
if __name__ == "__main__":
    scorer = ToxicityScorer(threshold=0.3)

    texts = [
        "Have a wonderful day!",
        "That was a stupid decision, you idiot.",
        "I want to destroy the competition in the market.",
    ]

    for t in texts:
        result = scorer.score(t)
        flag = "FLAGGED" if result.flagged else "OK"
        print(f"[{flag}] score={result.score} cat={result.category} -- {t[:60]}")
```

---

## 4. Hate Speech Detection

```python
"""Hate speech detector using pattern matching and the OpenAI moderation API."""

import re
from typing import Optional


class HateSpeechDetector:
    """
    Detect hate speech using layered approach:
      1. Pattern matching against known targeting phrases
      2. Contextual analysis (who is being targeted)
      3. OpenAI Moderation API for nuanced cases
    """

    # Patterns that indicate targeting of protected groups
    TARGETING_PATTERNS = [
        r"(?i)\b(all|every|those)\s+\w+\s+(are|should|must|need to)\b",
        r"(?i)\b(go back to|don't belong|inferior|subhuman)\b",
        r"(?i)\b(exterminate|eradicate|purge)\b.*\b(people|group|race)\b",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def detect(self, text: str) -> dict:
        """
        Analyze text for hate speech.

        Returns dict with is_hate_speech, confidence, and targeting_patterns.
        """
        matched_patterns = []

        for pattern in self.TARGETING_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                matched_patterns.append(pattern)

        confidence = min(len(matched_patterns) * 0.3, 1.0)

        return {
            "is_hate_speech": confidence >= self.threshold,
            "confidence": round(confidence, 3),
            "targeting_patterns": matched_patterns,
        }

    def detect_with_moderation(self, text: str) -> dict:
        """Enhanced detection using OpenAI moderation API."""
        from openai import OpenAI

        client = OpenAI()
        response = client.moderations.create(input=text)
        result = response.results[0]

        hate_score = result.category_scores.hate
        harassment_score = result.category_scores.harassment

        combined_score = max(hate_score, harassment_score)

        return {
            "is_hate_speech": combined_score >= self.threshold,
            "confidence": round(combined_score, 4),
            "hate_score": round(hate_score, 4),
            "harassment_score": round(harassment_score, 4),
        }
```

---

## 5. Building a Moderation Pipeline

Combine multiple filters into a single pipeline:

```python
"""Multi-stage content moderation pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Action(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class ModerationResult:
    action: Action = Action.ALLOW
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class ModerationPipeline:
    """Chain multiple content filters together."""

    def __init__(self):
        self._filters: list[tuple[str, Callable, float]] = []

    def add_filter(
        self,
        name: str,
        filter_fn: Callable[[str], float],
        threshold: float = 0.5,
    ):
        """Add a filter that returns a score (0-1)."""
        self._filters.append((name, filter_fn, threshold))
        return self  # for chaining

    def moderate(self, text: str) -> ModerationResult:
        """Run all filters and return aggregate result."""
        result = ModerationResult()

        for name, filter_fn, threshold in self._filters:
            score = filter_fn(text)
            result.scores[name] = round(score, 4)

            if score >= threshold:
                result.action = Action.BLOCK
                result.reasons.append(f"{name} score {score:.3f} >= {threshold}")
            elif score >= threshold * 0.7:
                if result.action != Action.BLOCK:
                    result.action = Action.WARN
                result.reasons.append(
                    f"{name} score {score:.3f} approaching threshold"
                )

        return result


# ----- Usage -----
def simple_toxicity(text: str) -> float:
    bad_words = ["stupid", "idiot", "hate"]
    count = sum(1 for w in bad_words if w in text.lower())
    return min(count * 0.3, 1.0)

def length_check(text: str) -> float:
    return min(len(text) / 10000, 1.0)

pipeline = ModerationPipeline()
pipeline.add_filter("toxicity", simple_toxicity, threshold=0.5)
pipeline.add_filter("length", length_check, threshold=0.8)

result = pipeline.moderate("You are such an idiot, I hate this stupid thing!")
print(f"Action: {result.action.value}")
print(f"Reasons: {result.reasons}")
print(f"Scores: {result.scores}")
```

---

## 6. Hands-On Lab

### Lab: Content Moderation Service

**Objective:** Build a FastAPI service that filters LLM outputs before returning them.

1. Create `modules/03-content-filtering/lab/starter/moderation_api.py`.
2. Implement a `POST /moderate` endpoint that accepts `{"text": "...", "categories": [...]}`.
3. Run the text through:
   - Local toxicity scorer (keyword-based)
   - OpenAI Moderation API (if API key available)
4. Return `{"action": "allow|warn|block", "details": {...}}`.
5. Add a `POST /filter` endpoint that returns sanitized text with toxic segments replaced by `[REDACTED]`.

---

## 7. Key Takeaways

- The **OpenAI Moderation API** is free and covers common safety categories out of the box.
- For **offline or custom** requirements, build keyword-based and ML-based scorers.
- Always use a **pipeline** approach so you can add, remove, or tune individual filters independently.
- Log moderation decisions for auditing and threshold tuning.

---

## Validation

```bash
bash modules/03-content-filtering/validation/validate.sh
```

---

**Next: [Module 04 -->](../04-pii-detection/)**
