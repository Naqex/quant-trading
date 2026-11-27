"""Transparent tools for small-scale quantitative research."""

from quant_lab import causal_universe, funding_carry_v2, unified_engine
from quant_lab.causal_universe import (
    build_sleeve2_paper_soak_day,
    build_sleeve2_paper_soak_receipt,
    causal_rolling_top_n_mask,
    f3_donchian_breakout_weights,
    f4_ema_trend_btc_weights,
    run_multi_asset_spot_backtest,
    run_sleeve2_offline_smoke,
)
from quant_lab.funding_carry_v2 import (
    FundingCarryConfig,
    carry_btc_eth_cons_weights,
    carry_top_k_hysteresis_weights,
    carry_v2_blend_weights,
    compute_carry_btc_eth_live_step,
    run_funding_carry_backtest,
)
from quant_lab.unified_engine import (
    DEFAULT_SHADOW_4H_KLINE_LIMIT,
    HOLDOUT_CUTOFF_DATE,
    MIN_RECOMMENDED_4H_KLINE_LIMIT,
    F4EmaConfig,
    UnifiedSpotEngine,
    apply_deadband_step,
    causal_ema_1d,
    causal_ema_matrix,
    causal_rolling_vol_1d,
    causal_rolling_vol_matrix,
    compute_f4_ema_step,
    enforce_pre_holdout_dates,
    enforce_pre_holdout_frame,
    update_shadow_spot_accounting,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SHADOW_4H_KLINE_LIMIT",
    "F4EmaConfig",
    "FundingCarryConfig",
    "HOLDOUT_CUTOFF_DATE",
    "MIN_RECOMMENDED_4H_KLINE_LIMIT",
    "UnifiedSpotEngine",
    "__version__",
    "apply_deadband_step",
    "build_sleeve2_paper_soak_day",
    "build_sleeve2_paper_soak_receipt",
    "carry_btc_eth_cons_weights",
    "carry_top_k_hysteresis_weights",
    "carry_v2_blend_weights",
    "causal_ema_1d",
    "causal_ema_matrix",
    "causal_rolling_top_n_mask",
    "causal_rolling_vol_1d",
    "causal_rolling_vol_matrix",
    "causal_universe",
    "compute_carry_btc_eth_live_step",
    "compute_f4_ema_step",
    "enforce_pre_holdout_dates",
    "enforce_pre_holdout_frame",
    "f3_donchian_breakout_weights",
    "f4_ema_trend_btc_weights",
    "funding_carry_v2",
    "run_funding_carry_backtest",
    "run_multi_asset_spot_backtest",
    "run_sleeve2_offline_smoke",
    "unified_engine",
    "update_shadow_spot_accounting",
]
