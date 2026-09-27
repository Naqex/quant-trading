# Ornstein–Uhlenbeck spread/residual pilot — 2026-09-27

## Quyết định

Quant Lab có một primitive OU offline tại
`quant_trading.ou_process`. Primitive này chỉ nhận chuỗi đã được gắn nhãn
`spread` hoặc `residual`; constructor từ chối `price` và các vai trò khác để
không biến hồi quy về trung bình thành giả định cho raw price. Nó không gọi
provider, broker, wallet, model service và không sinh tín hiệu giao dịch.

Fit dùng transition OU chính xác theo thời gian không đều:

```text
x[t] = μ + exp(-κ Δt) (x[t-1] - μ) + ε[t]
Var(ε[t]) = σ² (1 - exp(-2κ Δt)) / (2κ)
```

`κ`, `μ` và `σ²` được ước lượng bằng profile likelihood xác định: grid
log-space rồi golden-section refinement. Tính `1 - exp(-2κΔt)` dùng
`expm1` để tránh mất chính xác khi `κΔt` nhỏ. Fit trả half-life, stationary
variance, khoảng `Δt`, log-likelihood và SHA-256 của observation canonical.

## Contract point-in-time và stationarity

- Mỗi observation phải có `event_time <= known_at`, timestamp sự kiện tăng
  nghiêm ngặt và key `(known_at, sequence)` duy nhất.
- `cutoff_timestamp` chặn observation chưa biết tại thời điểm nghiên cứu;
  dòng vượt cutoff bị reject thay vì âm thầm bỏ qua.
- Fit cần tối thiểu 12 observation theo mặc định, cùng một `series_kind`,
  stationary variance dương và hữu hạn.
- Nếu likelihood chạm biên `κ` thấp/cao hoặc transition gần unit-root tại
  khoảng `Δt` nhỏ nhất, fit bị reject. Đây là guard chống gắn nhãn OU cho
  random walk; nó không phải một kiểm định stationarity đầy đủ.
- `filter_ou_process` chỉ dùng quan sát trước đó và prior stationary cho dòng
  đầu tiên. Nó trả conditional mean, conditional variance, innovation và
  standardized innovation, không dùng smoothing hay dữ liệu tương lai.
- `execution_capability`, `provider_access`, `broker_access` đều bị khóa
  `false` trong fit và filter result.

## Validation fixture

Fixture deterministic gồm 72 điểm spread có `Δt` tuần hoàn 1–4, mean level
gần 0.4 và mean-reverting transition. Fit ổn định qua thứ tự input đảo ngược;
stationarity guard từ chối chuỗi random walk; filter giữ provenance và cutoff.
Fixture khoảng thời gian rất lớn cũng đi qua `expm1` mà không sinh NaN/Inf.

Đã kiểm chứng:

```text
uv run pytest -q tests/quant_trading/test_ou_process.py
5 passed

uv run ruff check src/quant_trading/ou_process.py tests/quant_trading/test_ou_process.py
PASS
```

## Giới hạn và bước promotion

Đây là state/diagnostic primitive, không phải alpha claim. OU chỉ nên trở
thành một feature trong pipeline khi có spread/residual definition được ghi
nguồn, kiểm tra leakage, calibration theo walk-forward, baseline naive,
chi phí/slippage, OOS nhiều regime và drift monitor. Không áp dụng fit này
trực tiếp cho giá BTC/ETH. Nếu residual không stationary hoặc fit thường xuyên
chạm biên, phải giữ trạng thái `rejected` và không tự fallback sang raw price.

Nguồn nền tảng: [Uhlenbeck & Ornstein, 1930](https://doi.org/10.1103/PhysRev.36.823)
cho quá trình OU; [Aït-Sahalia, 2002](https://doi.org/10.1111/1468-0262.00305)
cho ước lượng khuếch tán với transition thời gian liên tục. Các nguồn này
chỉ định hướng mô hình; receipt này không chứng minh lợi nhuận.
