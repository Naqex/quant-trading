# OHLCV data-quality preflight pilot r1

## Quyết định

Quant Lab có thêm một preflight offline trước walk-forward. Preflight kiểm tra
artifact nến đã capture nhưng không sửa dữ liệu: không sort, forward-fill,
interpolate hay tự bỏ dòng. Nến thiếu được ghi là warning mặc định vì dữ liệu
crypto có thể có gap hợp lệ; owner có thể đặt `max_gap_count` để fail-closed
khi fixture yêu cầu lịch liên tục.

`audit_ohlcv(...)` trả receipt bất biến với fingerprint SHA-256, khoảng thời
gian, interval quan sát, gap count, warning/error codes và policy. Luồng bắt
buộc dùng `require_ohlcv_quality(...)`; receipt bị reject sẽ raise
`DataQualityError` trước khi được đưa vào walk-forward. Tất cả capability
provider/broker/execution vẫn là `false`, trạng thái receipt là `PREP_ONLY`.

## Các kiểm tra

- required columns: `open_time`, `open`, `high`, `low`, `close`, `volume`;
- timestamp parse được, duy nhất và tăng dần;
- OHLCV hữu hạn, giá dương, volume không âm;
- `high`/`low` bao phủ open và close;
- median/min/max interval và gap count khi policy khai báo interval;
- canonical input hash không phụ thuộc DataFrame index;
- gap budget chỉ được áp dụng khi policy khai báo rõ.

Không có giả định gap-free mặc định và không có claim về edge, forecast hay
profitability. Đây là lớp provenance/data-quality cho phép walk-forward từ
chối fixture hỏng sớm hơn.

## Receipt và kiểm chứng

Receipt mẫu synthetic (4 nến UTC, 1 giờ, gap budget bằng 0) nằm tại
`ohlcv-data-quality-pilot-r1-2026-09-28.json`. Nó chỉ chứng minh schema và
canonicalization, không phải dữ liệu thị trường thật.

```text
uv run pytest -q tests/quant_trading/test_data_quality.py
9 passed

uv run ruff check src/quant_trading/data_quality.py tests/quant_trading/test_data_quality.py
uv run python -m compileall -q src/quant_trading
```

