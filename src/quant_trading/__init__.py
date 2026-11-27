"""Offline research contracts and statistical diagnostics for Quant Lab."""

from quant_trading import event_envelope, paper_soak, theory_receipts, walk_forward
from quant_trading.event_envelope import (
    EVENT_ENVELOPE_SCHEMA,
    EventBatch,
    EventEnvelope,
    EventEnvelopeError,
    collect_events,
)
from quant_trading.paper_soak import (
    PAPER_SOAK_HEALTH_SCHEMA,
    PAPER_SOAK_SCHEMA,
    PAPER_SOAK_SCOPE,
    PaperSoakDay,
    PaperSoakError,
    PaperSoakHealthReport,
    PaperSoakReceipt,
    PaperSoakRestartCheck,
    audit_paper_soak_receipt,
)
from quant_trading.theory_receipts import (
    THEORY_RECEIPT_SCHEMA,
    TheoryHypothesisReceiptV1,
    TheoryReceiptError,
    json_sha256,
)
from quant_trading.walk_forward import (
    WALK_FORWARD_SCHEMA,
    FoldEvaluation,
    StressCase,
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardFold,
    WalkForwardReceipt,
    build_walk_forward_folds,
    run_walk_forward_stress,
)

__all__ = [
    "EVENT_ENVELOPE_SCHEMA",
    "PAPER_SOAK_HEALTH_SCHEMA",
    "PAPER_SOAK_SCHEMA",
    "PAPER_SOAK_SCOPE",
    "THEORY_RECEIPT_SCHEMA",
    "WALK_FORWARD_SCHEMA",
    "EventBatch",
    "EventEnvelope",
    "EventEnvelopeError",
    "FoldEvaluation",
    "PaperSoakDay",
    "PaperSoakError",
    "PaperSoakHealthReport",
    "PaperSoakReceipt",
    "PaperSoakRestartCheck",
    "StressCase",
    "TheoryHypothesisReceiptV1",
    "TheoryReceiptError",
    "WalkForwardConfig",
    "WalkForwardError",
    "WalkForwardFold",
    "WalkForwardReceipt",
    "audit_paper_soak_receipt",
    "build_walk_forward_folds",
    "collect_events",
    "event_envelope",
    "json_sha256",
    "paper_soak",
    "run_walk_forward_stress",
    "theory_receipts",
    "walk_forward",
]
