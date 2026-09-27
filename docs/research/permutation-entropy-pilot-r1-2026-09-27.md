# Permutation entropy pilot r1 — 2026-09-27

## Quyết định

Thêm `compute_permutation_entropy` như một diagnostic offline nhỏ cho độ phức
tạp của chuỗi scalar. Đây là lát có rủi ro thấp hơn TDA, BDS hoặc ABM vì không
cần dependency mới, không fit tham số, không mô phỏng tác nhân và không phát
sinh signal. Kết quả chỉ mô tả mức đa dạng của ordinal pattern trong một cửa sổ
đã biết tại cutoff.

## Hợp đồng causal

- Input là `CausalObservation` đã capture; thứ tự là `(known_at, sequence,
  event_time)` và observation sau `cutoff_timestamp` bị reject.
- `order` được giới hạn 2–8, `delay` phải dương; mẫu ngắn hơn cửa sổ bị reject
  thay vì âm thầm trả số liệu không ổn định.
- Pattern dùng tie-break theo vị trí sớm hơn trong cửa sổ. Số cửa sổ có ties
  được ghi ở `tie_window_count`, để người dùng không nhầm entropy do dữ liệu
  lặp với bằng chứng chaos.
- Receipt có hash observation, pattern counts/probabilities, entropy nats,
  entropy chuẩn hóa theo `log(order!)`, cutoff và
  `execution_capability=false`.

## Vì sao chọn lát này

Shannon entropy hiện có đo phân bố category rời rạc. Permutation entropy bổ
sung cấu trúc thứ tự cục bộ của returns/residual mà vẫn giữ một seam dữ liệu và
cutoff giống nhau. Nó có thể được dùng để nghiên cứu noise/regime và abstention,
nhưng chưa được nối vào chiến lược, sizing hay execution.

## Giới hạn và gate tiếp theo

Permutation entropy nhạy với `order`, `delay`, ties, cửa sổ và non-stationarity.
Không được gọi là fractal/chaos, không suy ra Hurst hay Lyapunov, và không được
dùng làm directional alpha. Trước khi promotion cần fixture đa tài sản, purged
walk-forward/OOS, block-permutation null, so sánh với Shannon/volatility/drawdown
baseline, stability theo tham số và false-alarm budget. Nếu không thắng baseline
về một mục tiêu định trước, giữ ở trạng thái research-only hoặc rollback.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_state_primitives.py
uv run ruff check src/quant_trading/state_primitives.py tests/quant_trading/test_state_primitives.py
```

Focused suite kiểm tra pattern đơn điệu, deterministic hash bất kể thứ tự input,
tie accounting, cutoff rejection, mẫu ngắn và giới hạn tham số.

Đây là diagnostic receipt offline; không phải bằng chứng lợi nhuận hay quyền
giao dịch thật.
