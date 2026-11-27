"""Transparent tools for small-scale quantitative research."""

from quant_lab import causal_universe
from quant_lab.causal_universe import (
    build_sleeve2_paper_soak_day,
    build_sleeve2_paper_soak_receipt,
    causal_rolling_top_n_mask,
    f3_donchian_breakout_weights,
    f4_ema_trend_btc_weights,
    run_multi_asset_spot_backtest,
    run_sleeve2_offline_smoke,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_sleeve2_paper_soak_day",
    "build_sleeve2_paper_soak_receipt",
    "causal_rolling_top_n_mask",
    "causal_universe",
    "f3_donchian_breakout_weights",
    "f4_ema_trend_btc_weights",
    "run_multi_asset_spot_backtest",
    "run_sleeve2_offline_smoke",
]
