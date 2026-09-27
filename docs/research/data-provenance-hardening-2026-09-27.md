# Quant Lab data provenance hardening — 2026-09-27

## Mục tiêu

Chặn dữ liệu OHLCV hỏng trước khi dữ liệu được dùng trong backtest hoặc
walk-forward, đồng thời ghi được fingerprint để báo cáo có thể truy ngược về
đúng tập nến đã chạy.

## Thay đổi

- `validate_candles` yêu cầu OHLCV tối thiểu, timestamp hợp lệ và tăng dần,
  giá hữu hạn/dương, volume không âm, và quan hệ high/low nhất quán với
  open/close.
- `load_candles` không còn âm thầm sort hoặc drop duplicate. Dữ liệu hỏng bị
  từ chối để người vận hành sửa nguồn trước khi nghiên cứu tiếp.
- `candle_fingerprint` tạo SHA-256 ổn định từ biểu diễn chuẩn hóa của các cột
  OHLCV; `build_provenance` trả metadata versioned gồm nguồn, symbol, interval,
  khoảng thời gian, số dòng và digest.

## Giới hạn

Không kiểm chứng tính đúng của nguồn Binance, không phát hiện mọi dạng
survivorship bias, không chứng minh edge, và không kết nối broker/live. Missing
calendar days vẫn được cho phép vì crypto có thể có khoảng trống dữ liệu; bước
walk-forward vẫn phải ghi rõ khoảng thời gian và data-quality report.
