# Parameter stability pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm `quant_trading.parameter_stability.run_parameter_stability` để
đánh giá độ nhạy của một neighbourhood tham số đã pre-register. Mỗi variant
được đưa qua cùng `walk_forward.run_walk_forward_stress`, cùng fixture, cùng
cutoff, cùng fee/slippage và cùng cost-stress multipliers. Receipt báo min,
median, max và range của các metric giữa variant ở baseline và mức stress xấu
nhất.

Lát này chỉ là diagnostic offline. Nó không grid-search, không chọn variant tốt
nhất, không huấn luyện, không gọi provider/broker/network và không phát tín
hiệu hoặc lệnh.

## Contract

- Có ít nhất hai factory được đặt tên; variant được sort theo id để output và
  hash deterministic, không phụ thuộc thứ tự mapping.
- Mỗi factory được replay bằng runner causal: OOS bar chỉ nhận prefix đến đúng
  bar đó, sau đó `run_backtest` thực hiện next-bar spot accounting với chi phí.
- Tất cả variant phải giữ cùng input hash và cùng `WalkForwardConfig`; mismatch
  bị reject thay vì gộp kết quả không so sánh được.
- Receipt giữ toàn bộ receipt walk-forward con, summary metric giữa variants,
  input hash và canonical SHA-256; `status=PREP_ONLY` cùng mọi capability flag
  đều `false`.
- Summary chỉ mô tả độ phân tán. Không có ngưỡng promotion hay logic chọn
  champion để tránh biến robustness check thành data-snooping.

## Hạn chế và gate tiếp theo

Đây không phải bằng chứng edge hay lợi nhuận. Các variant phải được đăng ký
trước khi nhìn holdout, có policy tham số và nguồn dữ liệu rõ ràng. Promotion
cần holdout/OOS độc lập, cost/latency/spread stress, drift/OOD, paper soak và
human review. Module chưa mô phỏng partial fill, market impact, missing data
hoặc regime shift ngoài các stress đã cấu hình.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_parameter_stability.py tests/quant_trading/test_walk_forward.py
uv run ruff check src/quant_trading/parameter_stability.py tests/quant_trading/test_parameter_stability.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Test bao phủ thứ tự deterministic, input/config/cost reuse, offline flags,
metric dispersion, malformed variant set và tamper receipt hash.
