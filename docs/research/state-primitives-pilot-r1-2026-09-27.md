# State primitives pilot r1 — 2026-09-27

## Quyết định

Quant Lab nhận một lớp primitive offline, deterministic cho các đặc trưng
trạng thái. Lớp này chỉ nhận observation đã được capture; nó không gọi provider,
không tải dữ liệu, không sửa holdout và không cấp quyền execution.

`execution_capability=false` là một invariant được kiểm tra ở từng receipt.
Observation phải có `event_time <= known_at`. Khi truyền `cutoff_timestamp`, mọi
observation có `known_at` sau cutoff đều bị reject để lỗi point-in-time không bị
che bằng việc âm thầm bỏ dòng dữ liệu.

## Phạm vi đã triển khai

- `CausalObservation` và `run_kalman_1d`: bộ lọc Kalman scalar chạy forward-only.
  Covariance dùng Joseph form, kiểm tra finite/PSD sau từng prediction/update.
  Receipt giữ innovation, gain, filtered mean/variance, model parameters và
  SHA-256 của observation canonical.
- `CategoricalObservation` và `compute_shannon_entropy`: Shannon entropy theo
  categorical sample, đơn vị nats. Miller–Madow correction là lựa chọn explicit
  và được ghi lại trong receipt; category/count/probability được sắp xếp ổn định.
- `BernoulliObservation` và `update_beta_bernoulli`: posterior Beta-Bernoulli
  tuần tự cho các outcome nhị phân, có prior, mean/variance và cutoff/provenance.

Các model không dùng smoothing, dữ liệu tương lai hoặc implicit fallback. Duplicate
`(known_at, sequence)` bị reject vì thứ tự quan sát tại cùng một thời điểm phải
được định danh rõ ràng.

## Chưa promote

BOCPD được giữ thành module/pilot riêng tại
`docs/research/bayesian-change-point-pilot-r1-2026-09-28.md`, không trộn vào
primitive state cũ. Regime label và mọi signal giao dịch vẫn nằm ngoài phạm vi;
BOCPD chỉ được promotion sau fixture dữ liệu, calibration/OOS split, drift
checks và so sánh với baseline. Primitive state không phải alpha claim và không
được dùng để suy ra lợi nhuận tương lai.

## Kiểm chứng

Focused suite:

```powershell
uv run pytest -q tests/quant_trading/test_state_primitives.py
uv run ruff check src/quant_trading/state_primitives.py tests/quant_trading/test_state_primitives.py
```

Suite kiểm tra deterministic hash, causal ordering, cutoff rejection, finite input,
Joseph covariance không âm, entropy correction, Beta posterior calibration và
`execution_capability=false`.

Nguồn nền tảng tham khảo: [Maybeck, *Stochastic Models, Estimation, and
Control*](https://www.cs.unc.edu/~welch/kalman/) cho Kalman filtering và [Paninski,
*Estimation of entropy and mutual information*](https://people.math.binghamton.edu/qiao/math605/paninski.pdf)
cho bias correction của entropy. Các nguồn này chỉ định hướng phương pháp; chúng
không phải bằng chứng về edge giao dịch.
