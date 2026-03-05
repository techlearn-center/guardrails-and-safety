"""
Hallucination Detector
=======================
Checks whether LLM output is grounded in provided context / source documents.
Uses claim extraction, entailment scoring, and source-span matching to
identify unsupported or contradicted statements.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    """A single factual claim extracted from LLM output."""

    text: str
    supported: bool = False
    support_score: float = 0.0
    supporting_evidence: Optional[str] = None
    verdict: str = "unverified"  # "supported" | "contradicted" | "unverified"


class HallucinationResult(BaseModel):
    """Result of hallucination detection on LLM output."""

    is_grounded: bool
    faithfulness_score: float = 0.0
    claims: list[Claim] = Field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    unverified_claims: int = 0
    explanation: str = ""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """
    Detects hallucinations by verifying LLM output against source context.

    Supports two modes:
      1. **Keyword overlap** -- fast, no API calls, compares token overlap
      2. **LLM-based entailment** -- uses GPT to verify each claim

    Example
    -------
    >>> detector = HallucinationDetector(threshold=0.7)
    >>> context = "Paris is the capital of France. It has a population of 2.1 million."
    >>> output = "Paris is the capital of France with 2.1 million people."
    >>> result = detector.check(output, context)
    >>> assert result.is_grounded
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_llm: bool = False,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.threshold = threshold
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self.openai_model = openai_model

        if self.use_llm:
            self._client = OpenAI()
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, output: str, context: str) -> HallucinationResult:
        """
        Check whether *output* is grounded in *context*.

        Parameters
        ----------
        output : str
            The LLM-generated text to verify.
        context : str
            The source / reference material that the output should be based on.

        Returns
        -------
        HallucinationResult
        """
        # Step 1: Extract claims from output
        claims = self._extract_claims(output)

        if not claims:
            return HallucinationResult(
                is_grounded=True,
                faithfulness_score=1.0,
                explanation="No factual claims detected in output.",
            )

        # Step 2: Verify each claim
        if self.use_llm:
            verified_claims = self._verify_claims_llm(claims, context)
        else:
            verified_claims = self._verify_claims_overlap(claims, context)

        # Step 3: Compute scores
        supported = sum(1 for c in verified_claims if c.verdict == "supported")
        contradicted = sum(1 for c in verified_claims if c.verdict == "contradicted")
        unverified = sum(1 for c in verified_claims if c.verdict == "unverified")
        total = len(verified_claims)

        faithfulness = supported / total if total > 0 else 0.0

        return HallucinationResult(
            is_grounded=faithfulness >= self.threshold,
            faithfulness_score=round(faithfulness, 3),
            claims=verified_claims,
            total_claims=total,
            supported_claims=supported,
            contradicted_claims=contradicted,
            unverified_claims=unverified,
            explanation=self._build_explanation(
                faithfulness, supported, contradicted, unverified, total
            ),
        )

    def is_faithful(self, output: str, context: str) -> bool:
        """Quick boolean check: is the output grounded in the context?"""
        return self.check(output, context).is_grounded

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(self, text: str) -> list[Claim]:
        """
        Split text into individual factual claims.

        Uses sentence splitting with heuristics to filter out non-factual
        sentences (questions, greetings, hedged language).
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims: list[Claim] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip non-factual sentences
            if self._is_non_factual(sentence):
                continue

            claims.append(Claim(text=sentence))

        return claims

    @staticmethod
    def _is_non_factual(sentence: str) -> bool:
        """Return True if the sentence is likely not a factual claim."""
        lower = sentence.lower()

        # Questions
        if sentence.strip().endswith("?"):
            return True

        # Greetings / filler
        non_factual_starts = [
            "hello", "hi ", "hey ", "sure", "certainly",
            "of course", "great question", "thank you",
            "you're welcome", "let me", "i think", "i believe",
            "in my opinion",
        ]
        for start in non_factual_starts:
            if lower.startswith(start):
                return True

        # Too short to be a claim
        if len(sentence.split()) < 4:
            return True

        return False

    # ------------------------------------------------------------------
    # Verification: keyword overlap (no API required)
    # ------------------------------------------------------------------

    def _verify_claims_overlap(
        self, claims: list[Claim], context: str
    ) -> list[Claim]:
        """Verify claims using token overlap with the source context."""
        context_lower = context.lower()
        context_tokens = set(re.findall(r"\b\w+\b", context_lower))

        # Remove stop words for more meaningful comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "but", "not", "no", "so", "if", "then", "than",
            "that", "this", "it", "its", "as",
        }
        context_meaningful = context_tokens - stop_words

        verified: list[Claim] = []
        for claim in claims:
            claim_tokens = set(re.findall(r"\b\w+\b", claim.text.lower()))
            claim_meaningful = claim_tokens - stop_words

            if not claim_meaningful:
                claim.verdict = "unverified"
                claim.support_score = 0.0
                verified.append(claim)
                continue

            overlap = claim_meaningful & context_meaningful
            score = len(overlap) / len(claim_meaningful)

            # Find supporting span in context
            evidence = self._find_best_span(claim.text, context)

            claim.support_score = round(score, 3)
            claim.supporting_evidence = evidence

            if score >= 0.6:
                claim.supported = True
                claim.verdict = "supported"
            elif score < 0.2:
                claim.verdict = "contradicted"
            else:
                claim.verdict = "unverified"

            verified.append(claim)

        return verified

    @staticmethod
    def _find_best_span(claim: str, context: str) -> Optional[str]:
        """Find the sentence in *context* most relevant to *claim*."""
        context_sentences = re.split(r"(?<=[.!?])\s+", context.strip())
        claim_words = set(re.findall(r"\b\w+\b", claim.lower()))

        best_score = 0.0
        best_span = None

        for sent in context_sentences:
            sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
            if not sent_words:
                continue
            overlap = len(claim_words & sent_words) / max(len(claim_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_span = sent

        return best_span if best_score > 0.3 else None

    # ------------------------------------------------------------------
    # Verification: LLM-based entailment
    # ------------------------------------------------------------------

    def _verify_claims_llm(
        self, claims: list[Claim], context: str
    ) -> list[Claim]:
        """Verify claims using LLM entailment checking."""
        if not self._client:
            return self._verify_claims_overlap(claims, context)

        verified: list[Claim] = []
        for claim in claims:
            try:
                response = self._client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fact-checking assistant. Given a context and a "
                                "claim, determine if the claim is supported by the context. "
                                "Respond with ONLY a JSON object:\n"
                                '{"verdict": "supported|contradicted|unverified", '
                                '"confidence": 0.0-1.0, '
                                '"evidence": "relevant quote from context or null"}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context[:2000]}\n\n"
                                f"Claim: {claim.text}"
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=200,
                )

                import json

                content = response.choices[0].message.content or ""
                data = json.loads(content)

                claim.verdict = data.get("verdict", "unverified")
                claim.support_score = float(data.get("confidence", 0.5))
                claim.supported = claim.verdict == "supported"
                claim.supporting_evidence = data.get("evidence")

            except Exception:
                claim.verdict = "unverified"
                claim.support_score = 0.0

            verified.append(claim)

        return verified

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        faithfulness: float,
        supported: int,
        contradicted: int,
        unverified: int,
        total: int,
    ) -> str:
        parts = [
            f"Faithfulness score: {faithfulness:.1%}.",
            f"{supported}/{total} claims supported by context.",
        ]
        if contradicted:
            parts.append(f"{contradicted} claim(s) contradicted by context.")
        if unverified:
            parts.append(f"{unverified} claim(s) could not be verified.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = HallucinationDetector(threshold=0.7, use_llm=False)

    context = (
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 for the World's Fair. "
        "The tower is 330 meters tall and is made of iron. "
        "It was designed by Gustave Eiffel's engineering company."
    )

    # Grounded output
    output_good = (
        "The Eiffel Tower stands in Paris, France. "
        "It was constructed in 1889 and reaches 330 meters in height."
    )

    # Hallucinated output
    output_bad = (
        "The Eiffel Tower is in London, England. "
        "It was built in 1920 and is 500 meters tall. "
        "It was designed by Leonardo da Vinci."
    )

    print("=== Grounded Output ===")
    result = detector.check(output_good, context)
    print(f"Grounded: {result.is_grounded}")
    print(f"Score: {result.faithfulness_score}")
    print(f"Explanation: {result.explanation}")
    for c in result.claims:
        print(f"  [{c.verdict}] {c.text}")

    print("\n=== Hallucinated Output ===")
    result = detector.check(output_bad, context)
    print(f"Grounded: {result.is_grounded}")
    print(f"Score: {result.faithfulness_score}")
    print(f"Explanation: {result.explanation}")
    for c in result.claims:
        print(f"  [{c.verdict}] {c.text}")
