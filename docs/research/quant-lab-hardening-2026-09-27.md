# Quant Lab hardening receipt — 2026-09-27

## Mục tiêu

Đóng một khoảng trống an toàn trong backtest trước khi mở rộng nghiên cứu edge: dữ liệu giá hoặc chi phí không hợp lệ không được âm thầm biến thành lợi nhuận, drawdown hay equity không có ý nghĩa.

## Thay đổi

- `run_backtest` từ chối chi phí không hữu hạn, chi phí âm và tổng chi phí từ 100% turnover trở lên.
- Giá đóng cửa phải là số hữu hạn, lớn hơn 0; series không được rỗng.
- Index phải khớp giữa `close` và `signal`, duy nhất và tăng dần. Điều này giữ thứ tự thời gian rõ ràng, tránh một fixture bị sắp sai làm sai kết luận timing.
- Tính phần trăm thay đổi dùng `fill_method=None`; không forward-fill giá thiếu một cách ngầm định.
- Giữ nguyên contract của spot lab: signal `NaN` vẫn được xem là cash (`0`), signal hữu hạn chỉ được nằm trong `[0, 1]`, và tín hiệu ngày `t` chỉ được giữ từ ngày `t+1`.

## Kiểm chứng

```powershell
uv run pytest -q
```

Kết quả: `17 passed`.

Các test mới bao phủ chi phí `NaN/Inf`, chi phí có thể làm equity âm, giá `NaN/0`, và index không tăng dần. Đây là hardening ở lớp mô phỏng offline; chưa phải bằng chứng về edge, chưa chạy dữ liệu mới, broker, paper trading hay live execution.

## Quyết định phạm vi

Không thêm dependency, không đổi spot/no-short, không đổi chiến lược hoặc tham số đã freeze, và không dùng kết quả test để promote chiến lược. Bước kế tiếp phù hợp là bổ sung kiểm tra provenance/data-quality cho artifact dữ liệu trước walk-forward, sau đó mới xem xét các kiểm định thống kê như bootstrap/CPCV/PBO.
