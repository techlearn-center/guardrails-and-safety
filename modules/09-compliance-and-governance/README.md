# Module 09: Compliance and AI Governance

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 01-08 |

---

## Learning Objectives

By the end of this module you will be able to:

1. Explain key regulatory frameworks for AI (EU AI Act, NIST AI RMF, GDPR, SOC 2).
2. Implement structured audit logging for all LLM interactions.
3. Create model cards that document model capabilities, limitations, and risks.
4. Build a responsible AI checklist and review process.
5. Design a governance workflow for deploying LLM features to production.

---

## 1. The Regulatory Landscape

Organizations deploying LLMs must comply with an evolving set of regulations:

| Framework | Region | Key Requirements |
|---|---|---|
| **EU AI Act** | EU | Risk classification, transparency, human oversight |
| **NIST AI RMF** | US | Risk management framework, trustworthiness characteristics |
| **GDPR** | EU | Data protection, right to explanation, consent |
| **SOC 2** | Global | Security, availability, processing integrity |
| **CCPA/CPRA** | California | Consumer data rights, disclosure requirements |
| **HIPAA** | US | Protected health information, minimum necessary standard |

### Risk Classification (EU AI Act)

```
Unacceptable Risk    --> Banned (social scoring, real-time biometric ID)
High Risk            --> Strict requirements (hiring, credit, medical)
Limited Risk         --> Transparency obligations (chatbots, deepfakes)
Minimal Risk         --> Voluntary codes of conduct
```

---

## 2. Audit Logging

Every LLM interaction should be logged for compliance and debugging:

```python
"""Structured audit logging for LLM interactions."""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class AuditLogEntry:
    """A single auditable LLM interaction."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str = ""
    session_id: str = ""
    model: str = ""
    prompt_hash: str = ""       # Hash of the prompt (not the raw text for privacy)
    input_length: int = 0
    output_length: int = 0
    latency_ms: float = 0.0
    safety_checks: dict = field(default_factory=dict)
    safety_passed: bool = True
    blocked: bool = False
    block_reason: str = ""
    pii_detected: bool = False
    pii_redacted: bool = False
    hallucination_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class AuditLogger:
    """Append-only audit logger for LLM interactions."""

    def __init__(self, log_dir: str = "./audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditLogEntry) -> str:
        """Write an audit log entry and return the request ID."""
        log_file = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        return entry.request_id

    def query(
        self,
        date: str,
        filters: Optional[dict] = None,
    ) -> list[AuditLogEntry]:
        """Query audit logs for a specific date with optional filters."""
        log_file = self.log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return []

        entries = []
        with open(log_file) as f:
            for line in f:
                data = json.loads(line.strip())
                if filters:
                    if all(data.get(k) == v for k, v in filters.items()):
                        entries.append(AuditLogEntry(**data))
                else:
                    entries.append(AuditLogEntry(**data))

        return entries

    def get_compliance_report(self, date: str) -> dict:
        """Generate a compliance report for the given date."""
        entries = self.query(date)

        total = len(entries)
        blocked = sum(1 for e in entries if e.blocked)
        pii_detected = sum(1 for e in entries if e.pii_detected)
        safety_failures = sum(1 for e in entries if not e.safety_passed)

        return {
            "date": date,
            "total_interactions": total,
            "blocked_requests": blocked,
            "block_rate": round(blocked / total, 4) if total else 0,
            "pii_incidents": pii_detected,
            "safety_failures": safety_failures,
            "safety_pass_rate": round(
                (total - safety_failures) / total, 4
            ) if total else 0,
        }


# ----- Demo -----
logger = AuditLogger(log_dir="./audit_logs")

entry = AuditLogEntry(
    user_id="user-123",
    model="gpt-4o-mini",
    input_length=150,
    output_length=320,
    latency_ms=450.5,
    safety_checks={
        "pii_scan": "passed",
        "toxicity_check": "passed",
        "injection_check": "passed",
    },
    safety_passed=True,
    toxicity_score=0.02,
)

request_id = logger.log(entry)
print(f"Logged request: {request_id}")
```

---

## 3. Model Cards

A model card documents a model's intended use, limitations, and safety profile:

```python
"""Model card generator for LLM deployments."""

from dataclasses import dataclass, field
from datetime import date
import json


@dataclass
class ModelCard:
    """Structured model card following the Mitchell et al. format."""

    # Model details
    model_name: str = ""
    model_version: str = ""
    model_type: str = ""
    provider: str = ""
    release_date: str = ""

    # Intended use
    primary_use_cases: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    intended_users: list[str] = field(default_factory=list)

    # Safety
    known_limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    safety_guardrails: list[str] = field(default_factory=list)

    # Performance
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    bias_evaluation: dict[str, str] = field(default_factory=dict)

    # Governance
    owner: str = ""
    review_date: str = ""
    next_review_date: str = ""
    approval_status: str = "pending"  # pending | approved | deprecated
    reviewers: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate a markdown-formatted model card."""
        sections = [
            f"# Model Card: {self.model_name}",
            f"\n**Version:** {self.model_version}  ",
            f"**Provider:** {self.provider}  ",
            f"**Type:** {self.model_type}  ",
            f"**Release Date:** {self.release_date}  ",
            f"**Status:** {self.approval_status}",
            "\n## Intended Use",
            "\n### Primary Use Cases",
        ]

        for use in self.primary_use_cases:
            sections.append(f"- {use}")

        sections.append("\n### Out-of-Scope Uses")
        for use in self.out_of_scope_uses:
            sections.append(f"- {use}")

        sections.append("\n## Known Limitations")
        for lim in self.known_limitations:
            sections.append(f"- {lim}")

        sections.append("\n## Safety Guardrails")
        for guard in self.safety_guardrails:
            sections.append(f"- {guard}")

        sections.append("\n## Ethical Considerations")
        for eth in self.ethical_considerations:
            sections.append(f"- {eth}")

        sections.append("\n## Evaluation Metrics")
        for metric, value in self.evaluation_metrics.items():
            sections.append(f"- **{metric}:** {value}")

        sections.append("\n## Governance")
        sections.append(f"- **Owner:** {self.owner}")
        sections.append(f"- **Review Date:** {self.review_date}")
        sections.append(f"- **Next Review:** {self.next_review_date}")
        sections.append(f"- **Reviewers:** {', '.join(self.reviewers)}")

        return "\n".join(sections)


# ----- Demo -----
card = ModelCard(
    model_name="Customer Support Assistant v2",
    model_version="2.1.0",
    model_type="Fine-tuned GPT-4o-mini",
    provider="OpenAI",
    release_date="2026-01-15",
    primary_use_cases=[
        "Answer product questions",
        "Process return requests",
        "Provide order status updates",
    ],
    out_of_scope_uses=[
        "Medical advice",
        "Legal advice",
        "Financial planning",
    ],
    known_limitations=[
        "May hallucinate product details not in knowledge base",
        "Cannot process payments directly",
        "Limited to English language only",
    ],
    safety_guardrails=[
        "PII detection and redaction on all outputs",
        "Toxicity filtering with 0.8 threshold",
        "Hallucination check against product knowledge base",
        "Prompt injection defense (pattern + LLM classifier)",
    ],
    ethical_considerations=[
        "Model may exhibit bias in product recommendations",
        "Automated responses should be clearly labeled as AI-generated",
    ],
    evaluation_metrics={
        "accuracy": 0.94,
        "hallucination_rate": 0.03,
        "demographic_parity": 0.92,
        "avg_latency_ms": 320,
    },
    owner="ML Platform Team",
    review_date="2026-01-15",
    next_review_date="2026-04-15",
    approval_status="approved",
    reviewers=["Jane Smith", "Bob Johnson"],
)

print(card.to_markdown())
```

---

## 4. Responsible AI Checklist

```python
"""Responsible AI deployment checklist."""

RESPONSIBLE_AI_CHECKLIST = {
    "pre_deployment": [
        {"item": "Model card completed and reviewed", "required": True},
        {"item": "Bias audit passed (demographic parity > 0.8)", "required": True},
        {"item": "Safety guardrails configured and tested", "required": True},
        {"item": "PII detection enabled with redaction", "required": True},
        {"item": "Prompt injection defense enabled", "required": True},
        {"item": "Hallucination detection configured for RAG", "required": True},
        {"item": "Content filtering enabled", "required": True},
        {"item": "Audit logging configured", "required": True},
        {"item": "Rate limiting configured", "required": True},
        {"item": "Human escalation path defined", "required": True},
    ],
    "post_deployment": [
        {"item": "Monitoring dashboards set up", "required": True},
        {"item": "Alerting configured for safety threshold breaches", "required": True},
        {"item": "Weekly bias audit scheduled", "required": False},
        {"item": "Monthly compliance report automated", "required": True},
        {"item": "Incident response plan documented", "required": True},
        {"item": "User feedback collection enabled", "required": False},
        {"item": "Model performance degradation monitoring", "required": True},
    ],
}


def evaluate_checklist(completed: dict[str, list[bool]]) -> dict:
    """Evaluate a completed checklist against requirements."""
    results = {}

    for phase, items in RESPONSIBLE_AI_CHECKLIST.items():
        phase_completed = completed.get(phase, [False] * len(items))
        total = len(items)
        done = sum(phase_completed[:total])
        required_items = [i for i in items if i["required"]]
        required_done = sum(
            1 for i, c in zip(items, phase_completed) if i["required"] and c
        )

        results[phase] = {
            "total": total,
            "completed": done,
            "required_total": len(required_items),
            "required_completed": required_done,
            "ready": required_done == len(required_items),
        }

    return results
```

---

## 5. Governance Workflow

```python
"""AI governance workflow for feature deployment."""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class ReviewStatus(Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


@dataclass
class GovernanceReview:
    """Track the governance review process for an AI feature."""
    feature_name: str
    description: str
    risk_level: str  # "minimal" | "limited" | "high"
    status: ReviewStatus = ReviewStatus.DRAFT
    model_card_url: str = ""
    bias_audit_passed: bool = False
    safety_tests_passed: bool = False
    compliance_review_passed: bool = False
    reviewers: list[str] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    approved_at: str = ""

    def add_comment(self, reviewer: str, comment: str) -> None:
        self.comments.append({
            "reviewer": reviewer,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        })

    def can_approve(self) -> tuple[bool, list[str]]:
        """Check if all required reviews are complete."""
        blockers = []

        if not self.model_card_url:
            blockers.append("Model card not submitted")
        if not self.bias_audit_passed:
            blockers.append("Bias audit not passed")
        if not self.safety_tests_passed:
            blockers.append("Safety tests not passed")
        if self.risk_level == "high" and not self.compliance_review_passed:
            blockers.append("Compliance review required for high-risk features")

        return len(blockers) == 0, blockers

    def approve(self, reviewer: str) -> bool:
        """Approve the feature if all checks pass."""
        can_approve, blockers = self.can_approve()
        if not can_approve:
            return False

        self.status = ReviewStatus.APPROVED
        self.approved_at = datetime.now().isoformat()
        self.add_comment(reviewer, "Feature approved for deployment")
        return True
```

---

## 6. Hands-On Lab

### Lab: Compliance Dashboard

**Objective:** Build a compliance monitoring system that tracks AI safety metrics.

1. Create `modules/09-compliance-and-governance/lab/starter/compliance_dashboard.py`.
2. Implement:
   - `AuditLogger` class that writes structured JSON logs for every LLM call.
   - `ComplianceReporter` class that generates daily/weekly reports with:
     - Total interactions, block rate, PII incident count
     - Average toxicity and hallucination scores
     - Top block reasons
   - `ModelCardGenerator` that produces a markdown model card from config.
   - `GovernanceWorkflow` that tracks feature reviews through draft/review/approved states.
3. Create a `GET /compliance/report` FastAPI endpoint that returns the latest report.
4. Bonus: add a `GET /compliance/alerts` endpoint that returns any metric that breaches its threshold.

---

## 7. Key Takeaways

- **Audit logging** is mandatory for compliance -- log every LLM interaction with safety check results.
- **Model cards** document intended use, limitations, and safety measures for accountability.
- **Responsible AI checklists** ensure all safety measures are in place before deployment.
- **Governance workflows** add human oversight to AI feature deployment.
- Compliance is not a one-time task -- it requires **continuous monitoring and reporting**.

---

## Validation

```bash
bash modules/09-compliance-and-governance/validation/validate.sh
```

---

**Next: [Module 10 -->](../10-production-safety-pipeline/)**
