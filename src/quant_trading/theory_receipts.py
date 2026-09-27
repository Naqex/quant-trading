"""Typed, provider-free receipts for theory and scenario research.

The receipt is a small audit boundary shared by diagnostic theories (for
example Bayesian, Kalman, chaos, SOC and TDA) and scenario methods (for
example game theory, reflexivity, adaptive search and ABM).  It records what
was evaluated at a point-in-time cutoff without turning a research result into
an execution instruction.  Inputs are already captured locally; this module
does not contact a provider, broker, wallet or model service.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

THEORY_RECEIPT_SCHEMA = "theory-hypothesis-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ResultStatus = Literal["accepted", "rejected", "inconclusive", "pending"]


class TheoryReceiptError(ValueError):
    """Raised when a theory receipt violates the offline research contract."""


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TheoryReceiptError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise TheoryReceiptError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TheoryReceiptError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TheoryReceiptError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise TheoryReceiptError(f"{name} must be a finite number")
    return value


def _freeze(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _text(str(key), f"{name} key", maximum=256): _freeze(item, f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{name} item") for item in value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, name)
    raise TheoryReceiptError(f"{name} must be finite JSON data")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TheoryReceiptError("receipt values must be finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TheoryReceiptError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _core(receipt: TheoryHypothesisReceiptV1) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "theory_id": receipt.theory_id,
        "theory_version": receipt.theory_version,
        "hypothesis": receipt.hypothesis,
        "source_hash": receipt.source_hash,
        "data_hash": receipt.data_hash,
        "model_hash": receipt.model_hash,
        "cutoff_timestamp": receipt.cutoff_timestamp,
        "known_at": receipt.known_at,
        "seed": receipt.seed,
        "assumptions": _thaw(receipt.assumptions),
        "metrics": _thaw(receipt.metrics),
        "baseline": _thaw(receipt.baseline),
        "result_status": receipt.result_status,
        "execution_capability": receipt.execution_capability,
        "provider_access": receipt.provider_access,
        "broker_access": receipt.broker_access,
        "rollback_ref": receipt.rollback_ref,
    }


@dataclass(frozen=True)
class TheoryHypothesisReceiptV1:
    """Immutable evidence for one offline theory/scenario evaluation.

    ``known_at`` is the earliest timestamp at which the complete receipt input
    was available.  It must be no later than ``cutoff_timestamp`` so a receipt
    cannot silently use information from after its research boundary.
    """

    theory_id: str
    theory_version: str
    hypothesis: str
    source_hash: str
    data_hash: str
    model_hash: str
    cutoff_timestamp: int
    known_at: int
    seed: int
    assumptions: Mapping[str, Any]
    metrics: Mapping[str, Any]
    baseline: Mapping[str, Any]
    result_status: ResultStatus
    rollback_ref: str
    schema: str = THEORY_RECEIPT_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_hash: str | None = None

    def __post_init__(self) -> None:
        if self.schema != THEORY_RECEIPT_SCHEMA:
            raise TheoryReceiptError(f"unsupported theory receipt schema: {self.schema!r}")
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(self, "theory_version", _text(self.theory_version, "theory_version"))
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "hypothesis", maximum=4096))
        object.__setattr__(self, "source_hash", _sha256(self.source_hash, "source_hash"))
        object.__setattr__(self, "data_hash", _sha256(self.data_hash, "data_hash"))
        object.__setattr__(self, "model_hash", _sha256(self.model_hash, "model_hash"))
        cutoff = _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        known_at = _strict_int(self.known_at, "known_at")
        if known_at > cutoff:
            raise TheoryReceiptError("known_at cannot be later than cutoff_timestamp")
        object.__setattr__(self, "cutoff_timestamp", cutoff)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "seed", _strict_int(self.seed, "seed"))
        if self.result_status not in {"accepted", "rejected", "inconclusive", "pending"}:
            raise TheoryReceiptError("result_status is unsupported")
        object.__setattr__(self, "rollback_ref", _text(self.rollback_ref, "rollback_ref", maximum=1024))
        for name in ("assumptions", "metrics", "baseline"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TheoryReceiptError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze(value, name))
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise TheoryReceiptError(f"{name} must remain false for offline research")
        expected = _sha256_json(_core(self))
        if self.receipt_hash is not None and self.receipt_hash != expected:
            raise TheoryReceiptError("receipt_hash does not match canonical receipt")
        object.__setattr__(self, "receipt_hash", expected)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TheoryHypothesisReceiptV1:
        if not isinstance(payload, Mapping):
            raise TheoryReceiptError("theory receipt must be a mapping")
        try:
            return cls(
                schema=payload.get("schema", THEORY_RECEIPT_SCHEMA),
                theory_id=payload["theory_id"],
                theory_version=payload["theory_version"],
                hypothesis=payload["hypothesis"],
                source_hash=payload["source_hash"],
                data_hash=payload["data_hash"],
                model_hash=payload["model_hash"],
                cutoff_timestamp=payload["cutoff_timestamp"],
                known_at=payload["known_at"],
                seed=payload["seed"],
                assumptions=payload["assumptions"],
                metrics=payload["metrics"],
                baseline=payload["baseline"],
                result_status=payload["result_status"],
                rollback_ref=payload["rollback_ref"],
                execution_capability=payload.get("execution_capability", False),
                provider_access=payload.get("provider_access", False),
                broker_access=payload.get("broker_access", False),
                receipt_hash=payload.get("receipt_hash"),
            )
        except KeyError as exc:
            raise TheoryReceiptError(f"theory receipt is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        result = _core(self)
        result["receipt_hash"] = self.receipt_hash
        return result


def json_sha256(value: Any) -> str:
    """Return a canonical SHA-256 for a finite JSON-like input fixture."""

    return _sha256_json(_freeze(value, "value"))
