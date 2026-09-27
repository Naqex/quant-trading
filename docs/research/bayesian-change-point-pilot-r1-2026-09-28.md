# Bayesian online change-point pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm một diagnostic BOCPD (Bayesian Online Change-Point Detection)
nhỏ, chạy offline và causal. Mỗi run-length giả thuyết dùng prior
Normal-Inverse-Gamma; predictive phân phối Student-t; posterior được cập nhật
theo thứ tự `(known_at, sequence, event_time)` bằng log-space recursion.
Primitive này chỉ mô tả xác suất một change-point tại bước hiện tại. Nó không
phải regime label đã được calibrate, forecast, alpha hoặc quyền giao dịch.

## Hợp đồng

- Module: `src/quant_trading/bayesian_change_point.py`
- Entry point: `run_bocpd(observations, config, cutoff_timestamp=...)`.
- Observation dùng lại `state_primitives.CausalObservation`; mọi dòng có
  `known_at` sau cutoff bị reject, không bị âm thầm bỏ qua.
- `BOCPDConfig` ghi rõ `prior_mean`, `prior_kappa`, `prior_alpha`,
  `prior_beta` và constant `hazard` trong `(0, 1]`.
- Posterior mỗi bước giữ run-length probabilities, `change_probability`, MAP
  run length và prequential `predictive_log_score`.
- `max_run_length` nếu được đặt là safety bound fail-closed: detector raise khi
  bound sắp làm mất posterior mass; implementation không drop hoặc merge tail.
- Log density, posterior parameters và normalization có finite guards; các
  fixture cực trị hữu hạn được clamp score theo giới hạn float thay vì sinh
  `NaN`/`inf` trong receipt.
- `execution_capability`, `provider_access` và `broker_access` luôn `false`;
  module không gọi network, provider, wallet hay broker.

## Vì sao chọn lát này

CUSUM hiện tại hữu ích để chẩn đoán crossing nhưng dùng baseline và threshold
cố định. BOCPD bổ sung uncertainty có cấu trúc về độ dài run và variance mà
không tạo abstraction mới trong backtest/execution. Nó là bước phù hợp hơn
TDA/ABM/chaos ở giai đoạn này vì không thêm dependency và có thể kiểm tra
causal, deterministic trên cùng fixture.

## Kết quả fixture

Fixture `[0, 0, 5, 5]`, prior `(mu=0, kappa=1, alpha=1, beta=1)`, hazard `0.25`
cho change probabilities `[0.25, 0.1848213, 0.6965556, 0.2844957]`. Đây chỉ là
kiểm tra cơ chế: xác suất tăng khi jump xuất hiện, không phải bằng chứng edge.
Receipt bất biến nằm tại
`docs/research/bayesian-change-point-pilot-r1-2026-09-28.json`.

## Giới hạn và gate promotion

Constant hazard và prior có thể làm thay đổi kết quả mạnh; Student-t ở đây là
predictive của model conjugate, không phải bằng chứng thị trường có phân phối
đó. Trước khi nối vào chart/regime filter cần fixture đa tài sản, prior/hazard
được chốt trước split, so sánh với no-change/CUSUM/volatility baseline, purged
walk-forward/OOS, detection delay, false-alarm budget, stress/overflow và
kiểm tra drift. Nếu không thắng baseline theo mục tiêu định trước, giữ
research-only hoặc rollback.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_bayesian_change_point.py
uv run ruff check src/quant_trading/bayesian_change_point.py tests/quant_trading/test_bayesian_change_point.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Focused tests kiểm tra causal cutoff, known-time deterministic ordering, hash,
posterior normalization, synthetic shift, invalid config, explicit run-length
bound, extreme finite values và deny-only flags.

Nguồn phương pháp: Adams & MacKay, *Bayesian Online Changepoint Detection*
(2007), và Murphy, *Conjugate Bayesian analysis of the Gaussian distribution*
(2007). Nguồn chỉ định hướng model; receipt này không phải khuyến nghị tài
chính hay cam kết lợi nhuận.
