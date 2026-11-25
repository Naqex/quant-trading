"""Typed, offline evidence for a future paper-soak run.

The contract makes the departure gate executable without pretending that an
offline fixture is a 30--60 day observation.  It validates daily accounting,
health pauses and restart duplicate checks, then produces a deterministic
receipt.  It never contacts a broker/provider, promotes a strategy, or grants
execution capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

PAPER_SOAK_SCHEMA = "paper-soak-evidence-v1"
PAPER_SOAK_HEALTH_SCHEMA = "paper-soak-health-report-v1"
PAPER_SOAK_SCOPE = "PREP_ONLY_OFFLINE"
MINIMUM_SOAK_DAYS = 30
MAXIMUM_SOAK_DAYS = 60
_MAX_MISSING_DATES_REPORTED = 60
_SYNTHETIC_EVIDENCE_LEVELS = frozenset({"template_only", "offline_fixture", "offline_smoke_fixture"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ReconciliationStatus = Literal["reconciled", "mismatch", "unknown"]
HealthStatus = Literal["healthy", "degraded", "paused", "unknown"]
RestartStatus = Literal["passed", "failed", "unknown"]
SoakStatus = Literal["NOT_RUN", "OBSERVED", "ACCEPTED", "REJECTED"]
HealthVerdict = Literal[
    "NOT_RUN",
    "IN_PROGRESS",
    "READY_FOR_OWNER_REVIEW",
    "ACCEPTED_OFFLINE_ONLY",
    "REJECTED",
]


class PaperSoakError(ValueError):
    """Raised when paper-soak evidence is malformed or unsafe."""


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperSoakError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise PaperSoakError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PaperSoakError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise PaperSoakError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperSoakError(f"{name} must be finite") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise PaperSoakError(f"{name} must be finite{f' and >= {minimum}' if minimum is not None else ''}")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PaperSoakError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PaperSoakError("paper-soak receipt must contain finite JSON data") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperSoakDay:
    """One UTC calendar day of paper accounting and health evidence."""

    date_utc: str
    input_cutoff_utc: int
    candidate_count: int
    paper_intents: int
    fills: int
    realized_pnl_vnd: float
    unrealized_pnl_vnd: float
    fees_vnd: float
    slippage_bps: float
    equity_hash: str
    ledger_hash: str
    reconciliation_status: ReconciliationStatus
    health_status: HealthStatus
    pause_or_stop_reason: str | None = None

    def __post_init__(self) -> None:
        date_value = _text(self.date_utc, "date_utc", maximum=10)
        try:
            parsed_date = date.fromisoformat(date_value)
        except ValueError as exc:
            raise PaperSoakError("date_utc must use YYYY-MM-DD") from exc
        if parsed_date.isoformat() != date_value:
            raise PaperSoakError("date_utc must use canonical YYYY-MM-DD")
        object.__setattr__(self, "date_utc", date_value)
        _strict_int(self.input_cutoff_utc, "input_cutoff_utc")
        for name in ("candidate_count", "paper_intents", "fills"):
            _strict_int(getattr(self, name), name)
        for name in ("realized_pnl_vnd", "unrealized_pnl_vnd"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("fees_vnd", "slippage_bps"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))
        _sha256(self.equity_hash, "equity_hash")
        _sha256(self.ledger_hash, "ledger_hash")
        if self.reconciliation_status not in {"reconciled", "mismatch", "unknown"}:
            raise PaperSoakError("reconciliation_status is unsupported")
        if self.health_status not in {"healthy", "degraded", "paused", "unknown"}:
            raise PaperSoakError("health_status is unsupported")
        if self.pause_or_stop_reason is not None:
            object.__setattr__(
                self,
                "pause_or_stop_reason",
                _text(self.pause_or_stop_reason, "pause_or_stop_reason", maximum=1024),
            )
        requires_reason = self.reconciliation_status != "reconciled" or self.health_status != "healthy"
        if requires_reason and self.pause_or_stop_reason is None:
            raise PaperSoakError("pause_or_stop_reason is required for mismatch or unhealthy day")
        if not requires_reason and self.pause_or_stop_reason is not None:
            raise PaperSoakError("healthy reconciled day cannot contain pause_or_stop_reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "date_utc": self.date_utc,
            "input_cutoff_utc": self.input_cutoff_utc,
            "candidate_count": self.candidate_count,
            "paper_intents": self.paper_intents,
            "fills": self.fills,
            "realized_pnl_vnd": self.realized_pnl_vnd,
            "unrealized_pnl_vnd": self.unrealized_pnl_vnd,
            "fees_vnd": self.fees_vnd,
            "slippage_bps": self.slippage_bps,
            "equity_hash": self.equity_hash,
            "ledger_hash": self.ledger_hash,
            "reconciliation_status": self.reconciliation_status,
            "health_status": self.health_status,
            "pause_or_stop_reason": self.pause_or_stop_reason,
        }


@dataclass(frozen=True)
class PaperSoakRestartCheck:
    """One restart/recovery observation used by the no-duplicate gate."""

    observed_at_utc: int
    attempt_no: int
    checkpoint: str
    duplicate_intents: int
    status: RestartStatus
    evidence_hash: str
    reason: str | None = None

    def __post_init__(self) -> None:
        _strict_int(self.observed_at_utc, "observed_at_utc")
        _strict_int(self.attempt_no, "attempt_no", minimum=1)
        _text(self.checkpoint, "checkpoint")
        duplicate_intents = _strict_int(self.duplicate_intents, "duplicate_intents")
        if self.status not in {"passed", "failed", "unknown"}:
            raise PaperSoakError("restart status is unsupported")
        _sha256(self.evidence_hash, "evidence_hash")
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason", maximum=1024))
        if self.status == "passed" and duplicate_intents != 0:
            raise PaperSoakError("passed restart check cannot contain duplicate intents")
        if self.status != "passed" and self.reason is None:
            raise PaperSoakError("failed or unknown restart check requires reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_at_utc": self.observed_at_utc,
            "attempt_no": self.attempt_no,
            "checkpoint": self.checkpoint,
            "duplicate_intents": self.duplicate_intents,
            "status": self.status,
            "evidence_hash": self.evidence_hash,
            "reason": self.reason,
        }


def _core(receipt: PaperSoakReceipt) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "scope": receipt.scope,
        "status": receipt.status,
        "evidence_level": receipt.evidence_level,
        "run_identity": dict(receipt.run_identity),
        "mode": receipt.mode,
        "window": {
            "minimum_days": MINIMUM_SOAK_DAYS,
            "maximum_days": MAXIMUM_SOAK_DAYS,
            "observed_days": len(receipt.daily_reconciliation),
            "start_date_utc": (
                receipt.daily_reconciliation[0].date_utc if receipt.daily_reconciliation else None
            ),
            "end_date_utc": (
                receipt.daily_reconciliation[-1].date_utc if receipt.daily_reconciliation else None
            ),
        },
        "daily_reconciliation": [day.as_dict() for day in receipt.daily_reconciliation],
        "restart_checks": [check.as_dict() for check in receipt.restart_checks],
        "owner_reviewed": receipt.owner_reviewed,
        "promotion_allowed": receipt.promotion_allowed,
        "execution_capability": receipt.execution_capability,
        "provider_access": receipt.provider_access,
        "broker_access": receipt.broker_access,
        "oauth_access": receipt.oauth_access,
    }


@dataclass(frozen=True)
class PaperSoakReceipt:
    """Immutable, promotion-safe paper-soak evidence."""

    daily_reconciliation: tuple[PaperSoakDay, ...] = ()
    restart_checks: tuple[PaperSoakRestartCheck, ...] = ()
    run_id: str | None = None
    attempt_no: int | None = None
    fence_token: str | None = None
    config_sha256: str | None = None
    strategy_sha256: str | None = None
    risk_budget_sha256: str | None = None
    owner_reviewed: bool = False
    status: SoakStatus = "NOT_RUN"
    evidence_level: str = "template_only"
    schema: str = PAPER_SOAK_SCHEMA
    scope: str = PAPER_SOAK_SCOPE
    mode: Literal["paper"] = "paper"
    promotion_allowed: bool = False
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    oauth_access: bool = False
    receipt_sha256: str | None = None

    @property
    def run_identity(self) -> dict[str, Any]:
        """Return the stable identity fields used by the receipt hash."""

        return {
            "run_id": self.run_id,
            "attempt_no": self.attempt_no,
            "fence_token": self.fence_token,
            "config_sha256": self.config_sha256,
            "strategy_sha256": self.strategy_sha256,
            "risk_budget_sha256": self.risk_budget_sha256,
        }

    def __post_init__(self) -> None:
        if self.schema != PAPER_SOAK_SCHEMA or self.scope != PAPER_SOAK_SCOPE:
            raise PaperSoakError("unsupported paper-soak schema or scope")
        if self.mode != "paper":
            raise PaperSoakError("paper-soak evidence must use paper mode")
        for name in (
            "execution_capability",
            "provider_access",
            "broker_access",
            "oauth_access",
            "promotion_allowed",
        ):
            if getattr(self, name) is not False:
                raise PaperSoakError(f"{name} must remain false for offline paper soak")
        days = tuple(self.daily_reconciliation)
        checks = tuple(self.restart_checks)
        if any(not isinstance(item, PaperSoakDay) for item in days):
            raise PaperSoakError("daily_reconciliation must contain PaperSoakDay values")
        if any(not isinstance(item, PaperSoakRestartCheck) for item in checks):
            raise PaperSoakError("restart_checks must contain PaperSoakRestartCheck values")
        ordered_dates = [item.date_utc for item in days]
        if ordered_dates != sorted(set(ordered_dates)):
            raise PaperSoakError("daily_reconciliation dates must be unique and sorted")
        cutoff_timestamps = [item.input_cutoff_utc for item in days]
        if any(
            current <= previous
            for previous, current in zip(
                cutoff_timestamps, cutoff_timestamps[1:], strict=False
            )
        ):
            raise PaperSoakError(
                "daily_reconciliation input_cutoff_utc values must be strictly increasing"
            )
        if len(days) > MAXIMUM_SOAK_DAYS:
            raise PaperSoakError("paper-soak evidence cannot exceed 60 observed days")
        if self.status not in {"NOT_RUN", "OBSERVED", "ACCEPTED", "REJECTED"}:
            raise PaperSoakError("status is unsupported")
        if self.status == "NOT_RUN" and (days or checks):
            raise PaperSoakError("NOT_RUN evidence cannot contain observations")
        if self.status == "NOT_RUN" and any(
            value is not None for value in (self.run_id, self.attempt_no, self.fence_token)
        ):
            raise PaperSoakError("NOT_RUN evidence cannot contain a run identity")
        if self.run_id is not None:
            object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=256))
        if self.attempt_no is not None:
            _strict_int(self.attempt_no, "attempt_no", minimum=1)
        for name in ("fence_token",):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _text(getattr(self, name), name, maximum=256))
        identity_hashes = (self.config_sha256, self.strategy_sha256, self.risk_budget_sha256)
        if any(value is not None for value in identity_hashes) and not all(
            value is not None for value in identity_hashes
        ):
            raise PaperSoakError("run identity hashes must be provided together")
        for name in ("config_sha256", "strategy_sha256", "risk_budget_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if type(self.owner_reviewed) is not bool:
            raise PaperSoakError("owner_reviewed must be a boolean")
        if self.status == "ACCEPTED":
            if len(days) < MINIMUM_SOAK_DAYS or not self.owner_reviewed:
                raise PaperSoakError("ACCEPTED evidence requires 30 days and explicit owner review")
            if self.run_id is None or self.attempt_no is None or self.fence_token is None:
                raise PaperSoakError("ACCEPTED evidence requires a complete run identity")
            if any(value is None for value in identity_hashes):
                raise PaperSoakError("ACCEPTED evidence requires config, strategy and risk hashes")
            if any(
                day.reconciliation_status != "reconciled" or day.health_status != "healthy"
                for day in days
            ):
                raise PaperSoakError("ACCEPTED evidence requires healthy reconciled days")
            if not checks or any(
                check.status != "passed" or check.duplicate_intents != 0 for check in checks
            ):
                raise PaperSoakError("ACCEPTED evidence requires passed restart checks with no duplicates")
        if self.status == "OBSERVED" and len(days) == 0:
            raise PaperSoakError("OBSERVED evidence requires at least one day")
        object.__setattr__(self, "daily_reconciliation", days)
        object.__setattr__(self, "restart_checks", checks)
        object.__setattr__(self, "evidence_level", _text(self.evidence_level, "evidence_level", maximum=64))
        expected = _hash(_core(self))
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise PaperSoakError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def observed_days(self) -> int:
        return len(self.daily_reconciliation)

    def as_dict(self) -> dict[str, Any]:
        result = _core(self)
        result["receipt_sha256"] = self.receipt_sha256
        return result

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PaperSoakReceipt:
        if not isinstance(payload, Mapping):
            raise PaperSoakError("paper-soak receipt must be a mapping")
        try:
            daily = tuple(PaperSoakDay(**item) for item in payload.get("daily_reconciliation", ()))
            restarts = tuple(PaperSoakRestartCheck(**item) for item in payload.get("restart_checks", ()))
            identity = payload.get("run_identity", {})
            if not isinstance(identity, Mapping):
                raise PaperSoakError("run_identity must be a mapping")
            receipt = cls(
                daily_reconciliation=daily,
                restart_checks=restarts,
                run_id=identity.get("run_id"),
                attempt_no=identity.get("attempt_no"),
                fence_token=identity.get("fence_token"),
                config_sha256=identity.get("config_sha256"),
                strategy_sha256=identity.get("strategy_sha256"),
                risk_budget_sha256=identity.get("risk_budget_sha256"),
                owner_reviewed=payload.get("owner_reviewed", False),
                status=payload.get("status", "NOT_RUN"),
                evidence_level=payload.get("evidence_level", "template_only"),
                schema=payload.get("schema", PAPER_SOAK_SCHEMA),
                scope=payload.get("scope", PAPER_SOAK_SCOPE),
                mode=payload.get("mode", "paper"),
                promotion_allowed=payload.get("promotion_allowed", False),
                execution_capability=payload.get("execution_capability", False),
                provider_access=payload.get("provider_access", False),
                broker_access=payload.get("broker_access", False),
                oauth_access=payload.get("oauth_access", False),
                receipt_sha256=payload.get("receipt_sha256"),
            )
            # A supplied summary must describe the actual observations; discarding
            # it would make an unchanged hash appear to endorse a tampered window.
            if "window" in payload and _canonical(payload["window"]) != _canonical(_core(receipt)["window"]):
                raise PaperSoakError("window does not match daily reconciliation observations")
            return receipt
        except (KeyError, TypeError) as exc:
            raise PaperSoakError("paper-soak receipt is malformed") from exc


@dataclass(frozen=True)
class PaperSoakHealthReport:
    """Deterministic, capability-free health and reconciliation audit for a PaperSoakReceipt."""

    receipt_sha256: str
    receipt_status: SoakStatus
    evidence_level: str
    verdict: HealthVerdict
    observed_days: int
    expected_calendar_days: int
    calendar_gap_days: int
    missing_dates_utc: tuple[str, ...]
    healthy_reconciled_days: int
    unhealthy_days: int
    mismatch_days: int
    excess_fill_days: int
    unfilled_intent_days: int
    total_paper_intents: int
    total_fills: int
    total_realized_pnl_vnd: float
    latest_unrealized_pnl_vnd: float
    total_fees_vnd: float
    mean_slippage_bps: float | None
    max_observed_slippage_bps: float | None
    restart_checks_total: int
    restart_checks_passed: int
    total_duplicate_intents: int
    owner_reviewed: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = PAPER_SOAK_HEALTH_SCHEMA
    scope: str = PAPER_SOAK_SCOPE
    promotion_allowed: bool = False
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    oauth_access: bool = False
    report_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != PAPER_SOAK_HEALTH_SCHEMA or self.scope != PAPER_SOAK_SCOPE:
            raise PaperSoakError("unsupported paper-soak health report schema or scope")
        for name in (
            "promotion_allowed",
            "execution_capability",
            "provider_access",
            "broker_access",
            "oauth_access",
        ):
            if getattr(self, name) is not False:
                raise PaperSoakError(f"{name} must remain false for offline health report")
        _sha256(self.receipt_sha256, "receipt_sha256")
        if self.verdict not in {
            "NOT_RUN",
            "IN_PROGRESS",
            "READY_FOR_OWNER_REVIEW",
            "ACCEPTED_OFFLINE_ONLY",
            "REJECTED",
        }:
            raise PaperSoakError("verdict is unsupported")
        expected = _hash(self._core_dict())
        if self.report_sha256 is not None and self.report_sha256 != expected:
            raise PaperSoakError("report_sha256 does not match canonical health report")
        object.__setattr__(self, "report_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scope": self.scope,
            "receipt_sha256": self.receipt_sha256,
            "receipt_status": self.receipt_status,
            "evidence_level": self.evidence_level,
            "verdict": self.verdict,
            "observed_days": self.observed_days,
            "expected_calendar_days": self.expected_calendar_days,
            "calendar_gap_days": self.calendar_gap_days,
            "missing_dates_utc": list(self.missing_dates_utc),
            "healthy_reconciled_days": self.healthy_reconciled_days,
            "unhealthy_days": self.unhealthy_days,
            "mismatch_days": self.mismatch_days,
            "excess_fill_days": self.excess_fill_days,
            "unfilled_intent_days": self.unfilled_intent_days,
            "total_paper_intents": self.total_paper_intents,
            "total_fills": self.total_fills,
            "total_realized_pnl_vnd": self.total_realized_pnl_vnd,
            "latest_unrealized_pnl_vnd": self.latest_unrealized_pnl_vnd,
            "total_fees_vnd": self.total_fees_vnd,
            "mean_slippage_bps": self.mean_slippage_bps,
            "max_observed_slippage_bps": self.max_observed_slippage_bps,
            "restart_checks_total": self.restart_checks_total,
            "restart_checks_passed": self.restart_checks_passed,
            "total_duplicate_intents": self.total_duplicate_intents,
            "owner_reviewed": self.owner_reviewed,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "promotion_allowed": self.promotion_allowed,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
            "oauth_access": self.oauth_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["report_sha256"] = self.report_sha256
        return result


def audit_paper_soak_receipt(
    receipt: PaperSoakReceipt,
    *,
    max_slippage_bps: float | None = None,
    max_calendar_gap_days: int = 0,
) -> PaperSoakHealthReport:
    """Audit a PaperSoakReceipt for calendar continuity, accounting health, and gate readiness."""
    if not isinstance(receipt, PaperSoakReceipt):
        raise PaperSoakError("receipt must be a PaperSoakReceipt")
    _strict_int(max_calendar_gap_days, "max_calendar_gap_days", minimum=0)
    if max_slippage_bps is not None:
        max_slippage_bps = _finite(max_slippage_bps, "max_slippage_bps", minimum=0.0)
    if receipt.receipt_sha256 != _hash(_core(receipt)):
        raise PaperSoakError("receipt_sha256 does not match canonical receipt")
    assert receipt.receipt_sha256 is not None

    days = receipt.daily_reconciliation
    restarts = receipt.restart_checks
    observed_days = len(days)

    missing_dates: list[str] = []
    expected_calendar_days = 0
    calendar_gap_days = 0
    if observed_days > 0:
        start_d = date.fromisoformat(days[0].date_utc)
        end_d = date.fromisoformat(days[-1].date_utc)
        expected_calendar_days = (end_d - start_d).days + 1
        calendar_gap_days = expected_calendar_days - observed_days
        if calendar_gap_days > 0:
            present = {day.date_utc for day in days}
            cursor = start_d
            while cursor <= end_d and len(missing_dates) < _MAX_MISSING_DATES_REPORTED:
                iso = cursor.isoformat()
                if iso not in present:
                    missing_dates.append(iso)
                cursor += timedelta(days=1)

    healthy_reconciled_days = sum(
        1 for day in days if day.reconciliation_status == "reconciled" and day.health_status == "healthy"
    )
    unhealthy_days = sum(1 for day in days if day.health_status != "healthy")
    mismatch_days = sum(1 for day in days if day.reconciliation_status != "reconciled")
    excess_fill_days = sum(1 for day in days if day.fills > day.paper_intents)
    unfilled_intent_days = sum(1 for day in days if day.fills < day.paper_intents)
    total_paper_intents = sum(day.paper_intents for day in days)
    total_fills = sum(day.fills for day in days)
    total_realized_pnl_vnd = round(sum(day.realized_pnl_vnd for day in days), 6)
    latest_unrealized_pnl_vnd = round(days[-1].unrealized_pnl_vnd, 6) if days else 0.0
    total_fees_vnd = round(sum(day.fees_vnd for day in days), 6)
    mean_slippage_bps = (
        round(sum(day.slippage_bps for day in days) / observed_days, 6) if observed_days > 0 else None
    )
    max_observed_slippage_bps = max((day.slippage_bps for day in days), default=None)

    restart_checks_total = len(restarts)
    restart_checks_passed = sum(
        1 for check in restarts if check.status == "passed" and check.duplicate_intents == 0
    )
    failed_restart_checks = sum(1 for check in restarts if check.status != "passed")
    total_duplicate_intents = sum(check.duplicate_intents for check in restarts)

    hard_blockers: list[str] = []
    readiness_blockers: list[str] = []
    warnings: list[str] = []

    if receipt.status == "REJECTED":
        hard_blockers.append("receipt_status_rejected")
    if mismatch_days > 0:
        hard_blockers.append(f"reconciliation_mismatch_days:{mismatch_days}")
    if unhealthy_days > 0:
        hard_blockers.append(f"unhealthy_days:{unhealthy_days}")
    if excess_fill_days > 0:
        hard_blockers.append(f"excess_fills_over_intents:{excess_fill_days}")
    if failed_restart_checks > 0:
        hard_blockers.append(f"failed_restart_checks:{failed_restart_checks}")
    if total_duplicate_intents > 0:
        hard_blockers.append(f"duplicate_intents_detected:{total_duplicate_intents}")
    if calendar_gap_days > max_calendar_gap_days:
        hard_blockers.append(f"calendar_gap_days_exceeded:{calendar_gap_days}>{max_calendar_gap_days}")
    elif calendar_gap_days > 0:
        warnings.append(f"calendar_gap_days_within_budget:{calendar_gap_days}")
    if max_slippage_bps is not None and max_observed_slippage_bps is not None:
        if max_observed_slippage_bps > max_slippage_bps:
            hard_blockers.append(
                f"slippage_limit_exceeded:{max_observed_slippage_bps:.4f}>{max_slippage_bps:.4f}"
            )
    if unfilled_intent_days > 0:
        warnings.append(f"unfilled_paper_intent_days:{unfilled_intent_days}")
    if calendar_gap_days > _MAX_MISSING_DATES_REPORTED:
        warnings.append(f"missing_dates_truncated_to:{_MAX_MISSING_DATES_REPORTED}")

    if receipt.status != "NOT_RUN":
        if observed_days < MINIMUM_SOAK_DAYS:
            readiness_blockers.append(f"insufficient_soak_days:{observed_days}/{MINIMUM_SOAK_DAYS}")
        if restart_checks_passed == 0:
            readiness_blockers.append("missing_passed_restart_check")
        if any(value is None for value in receipt.run_identity.values()):
            readiness_blockers.append("incomplete_run_identity")
        if receipt.evidence_level in _SYNTHETIC_EVIDENCE_LEVELS:
            readiness_blockers.append(f"non_live_evidence_level:{receipt.evidence_level}")
        if not receipt.owner_reviewed:
            readiness_blockers.append("awaiting_owner_review")

    verdict: HealthVerdict
    if receipt.status == "NOT_RUN":
        verdict = "NOT_RUN"
    elif hard_blockers:
        verdict = "REJECTED"
    elif not readiness_blockers and receipt.status == "ACCEPTED":
        verdict = "ACCEPTED_OFFLINE_ONLY"
    elif readiness_blockers == ["awaiting_owner_review"]:
        verdict = "READY_FOR_OWNER_REVIEW"
    else:
        if not readiness_blockers and receipt.status != "ACCEPTED":
            readiness_blockers.append("receipt_not_marked_accepted")
        verdict = "IN_PROGRESS"

    return PaperSoakHealthReport(
        receipt_sha256=receipt.receipt_sha256,
        receipt_status=receipt.status,
        evidence_level=receipt.evidence_level,
        verdict=verdict,
        observed_days=observed_days,
        expected_calendar_days=expected_calendar_days,
        calendar_gap_days=calendar_gap_days,
        missing_dates_utc=tuple(missing_dates),
        healthy_reconciled_days=healthy_reconciled_days,
        unhealthy_days=unhealthy_days,
        mismatch_days=mismatch_days,
        excess_fill_days=excess_fill_days,
        unfilled_intent_days=unfilled_intent_days,
        total_paper_intents=total_paper_intents,
        total_fills=total_fills,
        total_realized_pnl_vnd=total_realized_pnl_vnd,
        latest_unrealized_pnl_vnd=latest_unrealized_pnl_vnd,
        total_fees_vnd=total_fees_vnd,
        mean_slippage_bps=mean_slippage_bps,
        max_observed_slippage_bps=max_observed_slippage_bps,
        restart_checks_total=restart_checks_total,
        restart_checks_passed=restart_checks_passed,
        total_duplicate_intents=total_duplicate_intents,
        owner_reviewed=receipt.owner_reviewed,
        blockers=tuple(hard_blockers + readiness_blockers),
        warnings=tuple(warnings),
    )


__all__ = [
    "MAXIMUM_SOAK_DAYS",
    "MINIMUM_SOAK_DAYS",
    "PAPER_SOAK_HEALTH_SCHEMA",
    "PAPER_SOAK_SCHEMA",
    "PAPER_SOAK_SCOPE",
    "PaperSoakDay",
    "PaperSoakError",
    "PaperSoakHealthReport",
    "PaperSoakReceipt",
    "PaperSoakRestartCheck",
    "audit_paper_soak_receipt",
]
