from __future__ import annotations

from dataclasses import replace

import pytest

from quant_trading.paper_soak import (
    PaperSoakDay,
    PaperSoakError,
    PaperSoakReceipt,
    PaperSoakRestartCheck,
)

HASH = "a" * 64


def _day(index: int, *, health: str = "healthy", reconciliation: str = "reconciled") -> PaperSoakDay:
    reason = None if health == "healthy" and reconciliation == "reconciled" else "fixture health gate"
    return PaperSoakDay(
        date_utc=f"2026-01-{index:02d}",
        input_cutoff_utc=index,
        candidate_count=2,
        paper_intents=1,
        fills=1,
        realized_pnl_vnd=100.0,
        unrealized_pnl_vnd=50.0,
        fees_vnd=2.0,
        slippage_bps=1.5,
        equity_hash=HASH,
        ledger_hash=HASH,
        reconciliation_status=reconciliation,
        health_status=health,
        pause_or_stop_reason=reason,
    )


def _restart() -> PaperSoakRestartCheck:
    return PaperSoakRestartCheck(
        observed_at_utc=100,
        attempt_no=2,
        checkpoint="cp-001",
        duplicate_intents=0,
        status="passed",
        evidence_hash=HASH,
    )


def _identity() -> dict[str, str | int]:
    return {
        "run_id": "paper-run-1",
        "attempt_no": 1,
        "fence_token": "fence-1",
        "config_sha256": HASH,
        "strategy_sha256": HASH,
        "risk_budget_sha256": HASH,
    }


def test_not_run_receipt_is_deterministic_and_never_promotes() -> None:
    receipt = PaperSoakReceipt()
    roundtrip = PaperSoakReceipt.from_mapping(receipt.as_dict())

    assert receipt.status == "NOT_RUN"
    assert receipt.observed_days == 0
    assert receipt.promotion_allowed is False
    assert receipt.execution_capability is False
    assert receipt.as_dict() == roundtrip.as_dict()
    assert receipt.receipt_sha256 == roundtrip.receipt_sha256


def test_day_requires_reason_for_unhealthy_or_unreconciled_state() -> None:
    with pytest.raises(PaperSoakError, match="pause_or_stop_reason"):
        paused = _day(1, health="paused").as_dict()
        paused["pause_or_stop_reason"] = None
        PaperSoakDay(**paused)

    with pytest.raises(PaperSoakError, match="healthy reconciled"):
        healthy = _day(1).as_dict()
        healthy["pause_or_stop_reason"] = "unnecessary"
        PaperSoakDay(**healthy)


def test_observed_receipt_is_allowed_but_not_a_soak_pass() -> None:
    receipt = PaperSoakReceipt(
        daily_reconciliation=(_day(1),),
        restart_checks=(_restart(),),
        run_id="paper-run-1",
        attempt_no=1,
        fence_token="fence-1",
        config_sha256=HASH,
        strategy_sha256=HASH,
        risk_budget_sha256=HASH,
        status="OBSERVED",
        evidence_level="offline_fixture",
    )

    assert receipt.observed_days == 1
    assert receipt.promotion_allowed is False
    assert PaperSoakReceipt.from_mapping(receipt.as_dict()).receipt_sha256 == receipt.receipt_sha256


def test_accepted_receipt_requires_full_window_review_and_restart_proof() -> None:
    receipt = PaperSoakReceipt(
        daily_reconciliation=tuple(_day(index) for index in range(1, 31)),
        restart_checks=(_restart(),),
        run_id="paper-run-1",
        attempt_no=1,
        fence_token="fence-1",
        config_sha256=HASH,
        strategy_sha256=HASH,
        risk_budget_sha256=HASH,
        owner_reviewed=True,
        status="ACCEPTED",
        evidence_level="paper_observation",
    )

    assert receipt.observed_days == 30
    assert receipt.promotion_allowed is False
    assert receipt.receipt_sha256 == PaperSoakReceipt.from_mapping(receipt.as_dict()).receipt_sha256

    with pytest.raises(PaperSoakError, match="30 days"):
        PaperSoakReceipt(
            daily_reconciliation=tuple(_day(index) for index in range(1, 30)),
            restart_checks=(_restart(),),
            run_id="paper-run-1",
            attempt_no=1,
            fence_token="fence-1",
            config_sha256=HASH,
            strategy_sha256=HASH,
            risk_budget_sha256=HASH,
            owner_reviewed=True,
            status="ACCEPTED",
        )


def test_accepted_rejects_unhealthy_days_and_tampered_receipt() -> None:
    with pytest.raises(PaperSoakError, match="healthy reconciled"):
        days = tuple(_day(index, health="paused") if index == 1 else _day(index) for index in range(1, 31))
        PaperSoakReceipt(
            daily_reconciliation=days,
            restart_checks=(_restart(),),
            run_id="paper-run-1",
            attempt_no=1,
            fence_token="fence-1",
            config_sha256=HASH,
            strategy_sha256=HASH,
            risk_budget_sha256=HASH,
            owner_reviewed=True,
            status="ACCEPTED",
        )

    accepted = PaperSoakReceipt(
        daily_reconciliation=tuple(_day(index) for index in range(1, 31)),
        restart_checks=(_restart(),),
        run_id="paper-run-1",
        attempt_no=1,
        fence_token="fence-1",
        config_sha256=HASH,
        strategy_sha256=HASH,
        risk_budget_sha256=HASH,
        owner_reviewed=True,
        status="ACCEPTED",
    )
    with pytest.raises(PaperSoakError, match="does not match"):
        replace(accepted, receipt_sha256="0" * 64)


def test_duplicate_dates_and_restart_duplicates_fail_closed() -> None:
    with pytest.raises(PaperSoakError, match="unique and sorted"):
        PaperSoakReceipt(daily_reconciliation=(_day(1), _day(1)), status="OBSERVED")
    with pytest.raises(PaperSoakError, match="duplicate intents"):
        PaperSoakRestartCheck(
            observed_at_utc=100,
            attempt_no=2,
            checkpoint="cp-001",
            duplicate_intents=1,
            status="passed",
            evidence_hash=HASH,
        )


def test_daily_cutoffs_must_advance_with_observation_dates() -> None:
    first = _day(1)
    second_payload = _day(2).as_dict()
    second_payload["input_cutoff_utc"] = first.input_cutoff_utc
    with pytest.raises(PaperSoakError, match="input_cutoff_utc.*strictly increasing"):
        PaperSoakReceipt(
            daily_reconciliation=(first, PaperSoakDay(**second_payload)),
            status="OBSERVED",
        )


@pytest.mark.parametrize(
    "field",
    [
        "promotion_allowed", "execution_capability", "provider_access",
        "broker_access", "oauth_access",
    ],
)
@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_mapping_rejects_non_false_capabilities(field: str, value: object) -> None:
    payload = PaperSoakReceipt().as_dict()
    payload[field] = value

    with pytest.raises(PaperSoakError, match=field):
        PaperSoakReceipt.from_mapping(payload)


def test_mapping_rejects_live_mode_instead_of_normalizing_to_paper() -> None:
    payload = PaperSoakReceipt().as_dict()
    payload["mode"] = "live"

    with pytest.raises(PaperSoakError, match="paper mode"):
        PaperSoakReceipt.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_days", 30), ("observed_days", False),
        ("start_date_utc", "2026-01-01"), ("end_date_utc", "2026-01-30"),
        ("minimum_days", 1), ("maximum_days", 365),
    ],
)
def test_mapping_rejects_window_claims_that_disagree_with_observations(
    field: str, value: object,
) -> None:
    payload = PaperSoakReceipt().as_dict()
    payload["window"][field] = value

    with pytest.raises(PaperSoakError, match="window"):
        PaperSoakReceipt.from_mapping(payload)


def test_mapping_can_read_legacy_receipt_without_optional_window_or_capabilities() -> None:
    payload = {"status": "NOT_RUN"}
    receipt = PaperSoakReceipt.from_mapping(payload)

    assert receipt.as_dict() == PaperSoakReceipt().as_dict()

