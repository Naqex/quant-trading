# Offline robust risk metrics pilot — 2026-09-27

## Quyết định

Quant Lab có primitive `quant_trading.risk_metrics.compute_risk_metrics` cho
nghiên cứu crypto spot offline. Primitive nhận một chuỗi simple returns đã
capture và các timestamp theo thứ tự thời gian; nó không suy ra position,
leverage, sizing, chi phí khớp lệnh, broker hoặc provider.

Tail risk dùng return-space lower tail với `alpha` được truyền tường minh:

- `var_return` là empirical lower-tail quantile, dùng nội suy tuyến tính;
- `expected_shortfall_return`/`cvar_return` là trung bình của đúng `alpha`
  probability mass ở đuôi xấu nhất; order statistic cuối nhận fractional
  weight khi `n * alpha` không nguyên;
- `var_loss` và `expected_shortfall_loss` là cùng kết quả biểu diễn thành loss
  magnitude không âm.

Downside deviation dùng zero target và annualization `sqrt(periods_per_year)`
giống semantics hiện có trong `quant_lab.metrics`. Khi sample không có return
âm, Sortino ratio về mặt toán học không xác định; receipt giữ trường số hữu
hạn `sortino_zero_target=0.0` và ghi rõ `sortino_defined=false`, tránh NaN hoặc
infinity làm hỏng pipeline.

## Contract

- `alpha` phải thỏa `0 < alpha < 1`; `periods_per_year >= 1`.
- `returns` hữu hạn, không rỗng; timestamp là integer, tăng nghiêm ngặt và số
  dòng phải khớp.
- Mặc định chỉ nhận **test window** `validation_end_utc < timestamp <=
  test_end_utc`; train/validation chỉ được mở bằng tham số `window` tường minh.
- Receipt lưu `data_sha256`, digest returns + timestamps, cutoff, khoảng thời
  gian đánh giá, sample count, tail mass và method. `status=PREP_ONLY`;
  `execution_capability`, `provider_access`, `broker_access` luôn `false`.
- Tính toán tail, mean và downside RMS có scaling để tránh overflow trung gian
  với các return hữu hạn lớn; mọi output public đều finite.

## Fixture tái lập

Fixture dùng returns `[-0.10, -0.04, 0.02, 0.08]`, `alpha=0.50`,
`periods_per_year=1`, timestamps `[220, 230, 240, 250]` và cutoff
`[100, 200, 300]`. Kết quả chính là `var_return=-0.01`,
`expected_shortfall_return=-0.07`, `expected_shortfall_loss=0.07`.
Receipt JSON nằm tại
`risk-metrics-pilot-r1-2026-09-27.json` và có canonical SHA-256
`4b4c94f824a006c78a7508010d592828225232d701f52482e12d70683742d8a5`.

Đã kiểm chứng:

```text
uv run pytest -q tests/quant_trading/test_risk_metrics.py
13 passed

uv run ruff check src/quant_trading/risk_metrics.py tests/quant_trading/test_risk_metrics.py
PASS

git diff --check -- src/quant_trading/risk_metrics.py tests/quant_trading/test_risk_metrics.py
PASS
```

Đây là risk measurement primitive, không phải bằng chứng về alpha hay khuyến
nghị đầu tư. Promotion cần cùng fixture với baseline, walk-forward/OOS, phí,
stress, sample sufficiency và calibration; không dùng receipt này để mở
execution.
