# Chronological walk-forward + cost stress pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm `quant_trading.walk_forward.run_walk_forward_stress` để đóng
một khoảng trống trong quy trình nghiên cứu: kết quả một train/test split đơn
không đủ để đánh giá độ ổn định theo thời gian. Runner tạo các expanding
chronological folds và chạy lại cùng signal factory dưới nhiều mức phí và
slippage định trước.

Lát này chỉ là diagnostic offline. Nó không chọn tham số, không huấn luyện
model, không gọi network/provider/broker và không phát tín hiệu hoặc lệnh.

## Contract

- Mỗi fold có `train_bars` trước `test_bars`; `step_bars` mặc định bằng độ dài
  test nên các test window không chồng nhau.
- Signal factory của fold được replay trên từng prefix: khi tính OOS bar `t`,
  runner chỉ truyền dữ liệu đến `t`, rồi lấy giá trị signal cuối prefix. Vì vậy
  centered window hoặc negative shift trong factory không thể nhìn thấy bar OOS
  tương lai qua runner; factory vẫn phải là pure function.
- `run_backtest` vẫn chịu trách nhiệm next-bar execution, spot range
  `[0, 1]`, fee và slippage. Boundary entry được cộng lại vào số lần vào lệnh
  của OOS metrics để không mất một entry ở ranh giới train/test.
- Stress cases bắt buộc có baseline `1.0`; các multiplier là finite, dương,
  duy nhất và tổng chi phí sau stress phải nhỏ hơn 100% turnover.
- Receipt ghi input SHA-256, layout fold, metrics từng fold, median aggregate,
  worst drawdown/tail return và `positive_cagr_fraction`. Receipt hash là
  canonical và tất cả capability flags đều `false`.

## Vì sao chọn expanding walk-forward

Expanding history giữ lại toàn bộ dữ liệu đã biết khi bước qua thời gian, phù
hợp với chiến lược khung ngày không reset dữ liệu mỗi tháng. Đây không phải là
cam kết rằng signal factory đã được fit tối ưu trong từng train window; việc
fit/chọn tham số phải nằm trong caller với cutoff riêng và cần được kiểm tra
thêm trước khi promotion.

Cost stress ở đây là perturbation tối thiểu, dễ tái lập: giữ nguyên signal rồi
nhân đồng thời fee/slippage. Nó giúp phát hiện kết quả chỉ sống nhờ chi phí
quá lạc quan, nhưng chưa mô phỏng spread theo thanh, partial fill, latency,
market impact, missing candles hay regime shift. Các stress đó vẫn là gate
riêng.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_walk_forward.py
uv run ruff check src/quant_trading/walk_forward.py tests/quant_trading/test_walk_forward.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Test bao phủ fold boundary, không nhìn tương lai, deterministic receipt/hash,
baseline/adverse cost, boundary entry, malformed index, thiếu fold, signal sai
index, receipt tamper, chi phí vượt giới hạn và giá non-finite.

## Gate tiếp theo

Chỉ dùng receipt này làm evidence cho review offline. Promotion vẫn cần fixture
đã capture, parameter policy được pre-register, OOS/holdout độc lập, calibration,
data provenance, cost/latency stress và human review. Không suy ra edge,
profitability hay quyền paper/live execution từ pilot này.
