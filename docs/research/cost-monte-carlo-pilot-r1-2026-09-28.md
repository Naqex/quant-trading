# Deterministic cost Monte Carlo pilot r1 — 2026-09-28

Quant Lab thêm `quant_trading.cost_monte_carlo.run_cost_monte_carlo` để kiểm
tra độ nhạy của một signal fixture trước các giả định phí và slippage khác
nhau. Runner giữ nguyên close/signal, chạy một baseline và một số draw được
định trước trong cấu hình. Mỗi draw dùng hai multiplier độc lập trong khoảng
đã đăng ký trước; giá trị được sinh từ SHA-256 của `seed`, index và component
thay vì phụ thuộc phiên bản RNG của môi trường. Receipt lưu input hash, seed,
multiplier, chi phí và metric distribution nên có thể tái tạo và audit.

Đây là stress cho **mô hình chi phí**, không phải mô phỏng market impact,
spread, latency, partial fill hay regime change. Signal phải được tạo causal
trước đó (ví dụ từ walk-forward); module này không tự chứng minh không có
look-ahead. Không tải dữ liệu, gọi provider/broker, đặt lệnh hay mở capability.

## Contract và giới hạn

- `draws` nằm trong `1..5000`, seed là số nguyên không âm.
- Multiplier phải hữu hạn và dương; high không nhỏ hơn low.
- Tổng fee + slippage của mọi draw phải dưới 100% turnover để equity không bị
  âm bởi giả định chi phí.
- Receipt luôn có `status=PREP_ONLY`, `execution_capability=false`,
  `provider_access=false` và `broker_access=false`.
- Metric summary chỉ là phân phối diagnostic (`min`, `q05`, `median`, `q95`,
  `max`); không có winner, promotion hoặc lệnh giao dịch.

## Validation

```powershell
uv run pytest -q tests/quant_trading/test_cost_monte_carlo.py
uv run ruff check src/quant_trading/cost_monte_carlo.py tests/quant_trading/test_cost_monte_carlo.py
```

Kết quả fixture hiện tại: test kiểm tra hash/draw deterministic, chi phí khác
nhau nhưng cùng input, giới hạn equity, validation cấu hình và tamper hash.

## Bước tiếp theo

Muốn dùng receipt làm bằng chứng paper promotion vẫn cần cùng fixture với
walk-forward/OOS, holdout độc lập, spread/latency/partial-fill stress, drift
monitor và paper soak. Pilot này không tạo edge claim và không thay thế các
gate đó.
