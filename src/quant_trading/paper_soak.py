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
from datetime import date
from typing import Any, Literal

PAPER_SOAK_SCHEMA = "paper-soak-evidence-v1"
PAPER_SOAK_SCOPE = "PREP_ONLY_OFFLINE"
MINIMUM_SOAK_DAYS = 30
MAXIMUM_SOAK_DAYS = 60
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ReconciliationStatus = Literal["reconciled", "mismatch", "unknown"]
HealthStatus = Literal["healthy", "degraded", "paused", "unknown"]
RestartStatus = Literal["passed", "failed", "unknown"]
SoakStatus = Literal["NOT_RUN", "OBSERVED", "ACCEPTED", "REJECTED"]


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
            return cls(
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
                receipt_sha256=payload.get("receipt_sha256"),
            )
        except (KeyError, TypeError) as exc:
            raise PaperSoakError("paper-soak receipt is malformed") from exc


__all__ = [
    "MAXIMUM_SOAK_DAYS",
    "MINIMUM_SOAK_DAYS",
    "PAPER_SOAK_SCHEMA",
    "PAPER_SOAK_SCOPE",
    "PaperSoakDay",
    "PaperSoakError",
    "PaperSoakReceipt",
    "PaperSoakRestartCheck",
]
