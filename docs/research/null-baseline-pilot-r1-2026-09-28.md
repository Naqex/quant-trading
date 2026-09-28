# Offline null-baseline pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm `quant_trading.null_baseline.run_null_baselines` để mọi review
walk-forward có cùng hai mốc tham chiếu:

- `cash`: signal luôn bằng `0`, đại diện cho null/no-signal;
- `buy_hold`: signal luôn bằng `1`, đại diện cho benchmark nắm giữ thụ động.

Cả hai chạy qua `run_walk_forward_stress`, nên dùng cùng timing next-bar, fee,
slippage, fold layout và cost multipliers với strategy candidate. Receipt có
hash canonical, input hash và tất cả capability flags đều `false`.

Đây là baseline accounting, không phải permutation test, p-value, edge claim,
model selection hay quyền paper/live execution. Một kết quả tốt hơn baseline
chỉ là lý do để review thêm, không phải quyết định promote.

## Kiểm chứng trên fixture local

Fixture đã capture tại `data/raw/BTCUSDT_1d.csv` (3156 rows, 2018-01-01 đến
2026-08-22, SHA-256
`2f3dcf1f154627a961c1990b5e6d83302ee0f440083a3cfe28b14bba6b9a5398`). Dữ liệu
không được commit; hash giúp phát hiện fixture drift.

Config đã đăng ký trước: `train_bars=730`, `test_bars=180`, `step_bars=180`,
13 folds, chi phí cơ sở `10 bps + 5 bps`, stress `1.0x/1.5x/2.0x`.

| Reference | Median OOS total return | Median OOS max drawdown | Positive-CAGR folds | Receipt |
| --- | ---: | ---: | ---: | --- |
| cash | 0.0000 | 0.0000 | 0/13 | `92eaf2690c0a3830cf8d17a96016701b9e3d94c4b7148f87835fd37def329ab6` |
| buy_hold | 0.3628 | -0.6268 | 9/13 | cùng receipt trên |

Một SMA-200 candidate chạy cùng fixture có receipt
`024d8949030ed3f21c0de7ae72c9f0b976b7225e44795f05d6b4f1236f05e45f`. Median
OOS total return là `0.0000`, max drawdown `-0.4720`, positive-CAGR fraction
`6/13`; stress 2x làm Sharpe median giảm từ `1.0032` xuống `0.9208` và
max drawdown xấu hơn đến `-0.4744`.

Neighbourhood SMA-150/200/250 được chạy qua parameter stability receipt
`5815a59dda20f77f81ca133e0dddfd3e5a9ac012174182281ce840d8877aab26`. Baseline
Sharpe range là `1.2290`, max-drawdown range `0.2146`; stress Sharpe range là
`1.1936`. Đây là dispersion evidence, không chọn winner.

## Khoảng trống P0 còn lại

`run_walk_forward_stress` hiện là expanding chronological OOS và không có
`purge_bars`/`embargo_bars` trong contract. Vì vậy chưa được gọi là **purged
OOS** khi nhãn/horizon có thể chồng lấn. Không nên tự suy ra purge từ
`test_bars`; caller phải khai báo horizon và quy tắc purge/embargo trước khi
thêm implementation.

Các gate tiếp theo trước promotion:

1. pre-register horizon, `purge_bars`, `embargo_bars` và locked holdout;
2. thêm receipt fold metadata để chứng minh các vùng bị loại và kiểm tra không
   chồng lấn label;
3. chạy multi-asset cùng fixture/cost/null baselines;
4. thêm null permutation hoặc block-bootstrap nếu cần claim thống kê, giữ nó
   tách khỏi cash/buy-and-hold accounting baseline.

## Validation

```powershell
uv run pytest -q tests/quant_trading/test_null_baseline.py tests/quant_trading/test_walk_forward.py tests/quant_trading/test_parameter_stability.py tests/quant_trading/test_cost_monte_carlo.py
uv run ruff check src/quant_trading/null_baseline.py tests/quant_trading/test_null_baseline.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Kết quả: `27 passed`, Ruff pass, compileall pass. Toàn bộ slice vẫn
`PREP_ONLY`; không provider, broker, wallet, network hay execution.
