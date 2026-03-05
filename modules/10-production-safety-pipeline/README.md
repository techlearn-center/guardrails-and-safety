# Module 10: Production Safety Pipeline

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 01-09 |

---

## Learning Objectives

By the end of this module you will be able to:

1. Design an end-to-end safety pipeline that combines all guardrails from Modules 02-09.
2. Implement a `SafetyPipeline` class that orchestrates input guards, LLM processing, and output guards.
3. Add rate limiting, caching, and circuit breakers for production resilience.
4. Set up monitoring and alerting for safety metrics.
5. Build a FastAPI safety proxy that sits between clients and the LLM.

---

## 1. Production Safety Architecture

```
Client Request
     |
     v
+---------------------------+
|  Rate Limiter (Redis)      |  --> 429 if exceeded
+---------------------------+
     |
     v
+---------------------------+
|  INPUT GUARDS              |
|  1. Prompt injection scan  |
|  2. PII detection          |
|  3. Input length check     |
|  4. Content policy check   |
+---------------------------+
     |
     v
+---------------------------+
|  LLM API Call              |  --> Circuit breaker on failure
|  (with timeout)            |
+---------------------------+
     |
     v
+---------------------------+
|  OUTPUT GUARDS             |
|  1. Schema validation      |
|  2. Content filtering      |
|  3. PII redaction          |
|  4. Hallucination check    |
|  5. Bias scan              |
+---------------------------+
     |
     v
+---------------------------+
|  AUDIT LOG                 |  --> Every interaction logged
+---------------------------+
     |
     v
Safe Response to Client
```

---

## 2. The SafetyPipeline Class

```python
"""End-to-end safety pipeline combining all guardrails."""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PipelineAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class GuardResult:
    """Result from a single guard."""
    guard_name: str
    passed: bool
    score: float = 0.0
    details: str = ""
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result from the full safety pipeline."""
    action: PipelineAction = PipelineAction.ALLOW
    input_guards: list[GuardResult] = field(default_factory=list)
    output_guards: list[GuardResult] = field(default_factory=list)
    llm_response: str = ""
    safe_response: str = ""
    total_latency_ms: float = 0.0
    blocked_by: Optional[str] = None


class SafetyPipeline:
    """
    Production safety pipeline that orchestrates all guardrails.

    Guards are executed in order. If any guard with `blocking=True`
    fails, the pipeline short-circuits and returns a blocked response.
    """

    def __init__(self):
        self._input_guards: list[tuple[str, callable, bool]] = []
        self._output_guards: list[tuple[str, callable, bool]] = []

    def add_input_guard(
        self, name: str, guard_fn: callable, blocking: bool = True
    ):
        """Add a guard that runs on user input."""
        self._input_guards.append((name, guard_fn, blocking))
        return self

    def add_output_guard(
        self, name: str, guard_fn: callable, blocking: bool = True
    ):
        """Add a guard that runs on LLM output."""
        self._output_guards.append((name, guard_fn, blocking))
        return self

    def run(
        self,
        user_input: str,
        llm_fn: callable,
        context: Optional[str] = None,
    ) -> PipelineResult:
        """Execute the full safety pipeline."""
        result = PipelineResult()
        pipeline_start = time.time()

        # ---- Input Guards ----
        for name, guard_fn, blocking in self._input_guards:
            start = time.time()
            try:
                guard_result = guard_fn(user_input)
                latency = (time.time() - start) * 1000

                gr = GuardResult(
                    guard_name=name,
                    passed=guard_result.get("passed", True),
                    score=guard_result.get("score", 0.0),
                    details=guard_result.get("details", ""),
                    latency_ms=round(latency, 2),
                )
                result.input_guards.append(gr)

                if not gr.passed and blocking:
                    result.action = PipelineAction.BLOCK
                    result.blocked_by = name
                    result.safe_response = (
                        "Your request could not be processed due to "
                        "safety policy restrictions."
                    )
                    result.total_latency_ms = round(
                        (time.time() - pipeline_start) * 1000, 2
                    )
                    return result

            except Exception as e:
                gr = GuardResult(
                    guard_name=name,
                    passed=False,
                    details=f"Guard error: {str(e)}",
                    latency_ms=round((time.time() - start) * 1000, 2),
                )
                result.input_guards.append(gr)

        # ---- LLM Call ----
        try:
            llm_start = time.time()
            llm_response = llm_fn(user_input)
            result.llm_response = llm_response
        except Exception as e:
            result.action = PipelineAction.BLOCK
            result.blocked_by = "llm_error"
            result.safe_response = (
                "An error occurred processing your request. "
                "Please try again later."
            )
            result.total_latency_ms = round(
                (time.time() - pipeline_start) * 1000, 2
            )
            return result

        # ---- Output Guards ----
        safe_response = llm_response

        for name, guard_fn, blocking in self._output_guards:
            start = time.time()
            try:
                guard_input = {"output": safe_response}
                if context:
                    guard_input["context"] = context

                guard_result = guard_fn(guard_input)
                latency = (time.time() - start) * 1000

                gr = GuardResult(
                    guard_name=name,
                    passed=guard_result.get("passed", True),
                    score=guard_result.get("score", 0.0),
                    details=guard_result.get("details", ""),
                    latency_ms=round(latency, 2),
                )
                result.output_guards.append(gr)

                # Apply sanitization if the guard provides it
                if "sanitized" in guard_result:
                    safe_response = guard_result["sanitized"]

                if not gr.passed and blocking:
                    result.action = PipelineAction.BLOCK
                    result.blocked_by = name
                    safe_response = (
                        "The response was filtered due to safety concerns."
                    )
                    break

            except Exception as e:
                gr = GuardResult(
                    guard_name=name,
                    passed=False,
                    details=f"Guard error: {str(e)}",
                    latency_ms=round((time.time() - start) * 1000, 2),
                )
                result.output_guards.append(gr)

        result.safe_response = safe_response
        if result.action == PipelineAction.ALLOW:
            # Check if any non-blocking guard warned
            all_passed = all(
                g.passed
                for g in result.input_guards + result.output_guards
            )
            if not all_passed:
                result.action = PipelineAction.WARN

        result.total_latency_ms = round(
            (time.time() - pipeline_start) * 1000, 2
        )
        return result


# ----- Usage -----
def injection_guard(text: str) -> dict:
    """Simple prompt injection check."""
    from src.validators.prompt_injection import PromptInjectionDetector
    detector = PromptInjectionDetector()
    result = detector.analyze(text)
    return {
        "passed": not result.is_injection,
        "score": result.confidence,
        "details": result.explanation,
    }


def pii_output_guard(data: dict) -> dict:
    """PII check on LLM output."""
    from src.validators.pii_detector import PIIDetector
    detector = PIIDetector()
    result = detector.detect(data["output"])
    return {
        "passed": not result.has_pii,
        "score": len(result.entities) / 10.0,
        "details": f"Found {len(result.entities)} PII entities",
        "sanitized": result.redacted_text or data["output"],
    }


pipeline = SafetyPipeline()
pipeline.add_input_guard("injection_check", injection_guard, blocking=True)
pipeline.add_output_guard("pii_redaction", pii_output_guard, blocking=False)

# result = pipeline.run(
#     user_input="What is the weather in Paris?",
#     llm_fn=lambda x: "The weather in Paris is sunny and 22 degrees.",
# )
```

---

## 3. Rate Limiting with Redis

```python
"""Redis-based rate limiter for LLM API endpoints."""

import time
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RateLimiter:
    """
    Token-bucket rate limiter using Redis.

    Tracks requests per user/IP with configurable limits.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        if REDIS_AVAILABLE:
            self.redis = redis.from_url(redis_url)
        else:
            self.redis = None
            self._local_store: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> tuple[bool, dict]:
        """
        Check if a request from *identifier* is allowed.

        Returns (allowed: bool, info: dict with remaining, reset_at).
        """
        if self.redis:
            return self._check_redis(identifier)
        return self._check_local(identifier)

    def _check_redis(self, identifier: str) -> tuple[bool, dict]:
        key = f"ratelimit:{identifier}"
        now = time.time()

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - self.window_seconds)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window_seconds)
        results = pipe.execute()

        current_count = results[2]
        allowed = current_count <= self.max_requests

        return allowed, {
            "current": current_count,
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - current_count),
            "reset_at": now + self.window_seconds,
        }

    def _check_local(self, identifier: str) -> tuple[bool, dict]:
        """Fallback: in-memory rate limiting."""
        now = time.time()
        if identifier not in self._local_store:
            self._local_store[identifier] = []

        # Remove expired entries
        self._local_store[identifier] = [
            t for t in self._local_store[identifier]
            if t > now - self.window_seconds
        ]

        self._local_store[identifier].append(now)
        current = len(self._local_store[identifier])

        return current <= self.max_requests, {
            "current": current,
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - current),
        }
```

---

## 4. Circuit Breaker for LLM Calls

```python
"""Circuit breaker pattern for LLM API calls."""

import time
from enum import Enum
from dataclasses import dataclass


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for LLM API calls.

    Opens after `failure_threshold` consecutive failures.
    Moves to half-open after `recovery_timeout` seconds.
    Closes again after a successful call in half-open state.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN: allow one request through
        return True

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def call(self, fn: callable, *args, **kwargs):
        """Execute a function through the circuit breaker."""
        if not self.can_execute():
            raise RuntimeError(
                f"Circuit breaker is {self.state.value}. "
                f"Retry after {self.recovery_timeout}s."
            )

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
```

---

## 5. FastAPI Safety Proxy

```python
"""FastAPI safety proxy that wraps an LLM API with guardrails."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import time


app = FastAPI(title="LLM Safety Proxy")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    context: Optional[str] = None
    model: str = "gpt-4o-mini"


class ChatResponse(BaseModel):
    response: str
    request_id: str
    safety_action: str
    guard_results: list[dict]
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/chat", response_model=ChatResponse)
async def safe_chat(request: ChatRequest):
    """Process a chat request through the safety pipeline."""

    # 1. Rate limiting (pseudo-code, use RateLimiter class)
    # allowed, info = rate_limiter.is_allowed(request.user_id)
    # if not allowed:
    #     raise HTTPException(429, "Rate limit exceeded")

    # 2. Run through safety pipeline
    # pipeline_result = safety_pipeline.run(
    #     user_input=request.message,
    #     llm_fn=lambda x: call_llm(x, request.model),
    #     context=request.context,
    # )

    # 3. Log to audit system
    # audit_logger.log(AuditLogEntry(...))

    # 4. Return safe response
    return ChatResponse(
        response="Pipeline response here",
        request_id="req-12345",
        safety_action="allow",
        guard_results=[],
        latency_ms=0.0,
    )


@app.get("/metrics")
async def safety_metrics():
    """Return current safety metrics for monitoring."""
    return {
        "total_requests_24h": 0,
        "blocked_rate": 0.0,
        "avg_latency_ms": 0.0,
        "pii_incidents": 0,
        "injection_attempts": 0,
    }
```

---

## 6. Monitoring and Alerting

```python
"""Safety monitoring and alerting configuration."""

from dataclasses import dataclass


@dataclass
class SafetyAlert:
    """Define alert thresholds for safety metrics."""
    metric: str
    threshold: float
    comparison: str  # "gt" (greater than) or "lt" (less than)
    severity: str    # "warning" | "critical"
    message: str


# Define production alerting rules
ALERT_RULES = [
    SafetyAlert(
        metric="block_rate_1h",
        threshold=0.1,
        comparison="gt",
        severity="warning",
        message="Block rate exceeds 10% in the last hour",
    ),
    SafetyAlert(
        metric="block_rate_1h",
        threshold=0.25,
        comparison="gt",
        severity="critical",
        message="Block rate exceeds 25% -- possible attack or model degradation",
    ),
    SafetyAlert(
        metric="pii_incident_rate_1h",
        threshold=0.05,
        comparison="gt",
        severity="critical",
        message="PII leakage rate exceeds 5% -- immediate investigation required",
    ),
    SafetyAlert(
        metric="avg_hallucination_score",
        threshold=0.3,
        comparison="gt",
        severity="warning",
        message="Average hallucination score rising -- check RAG pipeline",
    ),
    SafetyAlert(
        metric="injection_attempts_1h",
        threshold=50,
        comparison="gt",
        severity="warning",
        message="High volume of injection attempts -- possible coordinated attack",
    ),
    SafetyAlert(
        metric="avg_latency_ms",
        threshold=5000,
        comparison="gt",
        severity="warning",
        message="Safety pipeline latency exceeding 5s -- performance degradation",
    ),
    SafetyAlert(
        metric="safety_pass_rate",
        threshold=0.95,
        comparison="lt",
        severity="critical",
        message="Safety pass rate below 95% -- model may need retraining",
    ),
]


def check_alerts(current_metrics: dict[str, float]) -> list[SafetyAlert]:
    """Check current metrics against alert thresholds."""
    triggered = []

    for rule in ALERT_RULES:
        value = current_metrics.get(rule.metric)
        if value is None:
            continue

        if rule.comparison == "gt" and value > rule.threshold:
            triggered.append(rule)
        elif rule.comparison == "lt" and value < rule.threshold:
            triggered.append(rule)

    return triggered
```

---

## 7. Hands-On Lab

### Lab: Production Safety API

**Objective:** Build a complete safety proxy API that can be deployed to protect any LLM endpoint.

1. Create `modules/10-production-safety-pipeline/lab/starter/safety_api.py`.
2. Implement a FastAPI application with:
   - `POST /chat` -- full safety pipeline (input guards + LLM + output guards).
   - `GET /health` -- health check endpoint.
   - `GET /metrics` -- return current safety statistics.
   - Rate limiting using Redis (or in-memory fallback).
   - Circuit breaker for LLM API calls.
3. Configure these guards:
   - Input: prompt injection detection, input length limit.
   - Output: PII redaction, content filtering, schema validation.
4. Add audit logging for every request.
5. Run the Docker Compose setup (`docker-compose.yml`) to test with Redis.

---

## 8. Key Takeaways

- A production safety pipeline **combines all guardrails** into a single request flow.
- **Rate limiting** prevents abuse; **circuit breakers** handle API failures gracefully.
- **Monitoring and alerting** catch safety regressions before they reach users.
- **Audit logging** is essential for compliance and incident investigation.
- Deploy the safety proxy as a **sidecar or middleware** in front of your LLM API.

---

## Validation

```bash
bash modules/10-production-safety-pipeline/validation/validate.sh
```

---

**Next: [Capstone Project -->](../../capstone/)**
