# Quant Lab

Một phòng thí nghiệm quant trading nhỏ, minh bạch và ưu tiên kiểm chứng trước khi dùng tiền thật.

## Phạm vi bản đầu

- Crypto spot (mua tài sản thật, không dùng đòn bẩy) trên khung ngày.
- Dữ liệu nến công khai từ Binance; không cần API key.
- Benchmark buy-and-hold (mua và nắm giữ) để làm mốc so sánh.
- Chiến lược trend following (đi theo xu hướng) bằng hai đường trung bình động.
- Tín hiệu hôm nay chỉ được thực hiện ở nến kế tiếp để tránh look-ahead bias (vô tình dùng dữ liệu tương lai).
- Backtest có phí giao dịch và slippage (trượt giá).
- Chia train/test theo thời gian: phần đầu để nghiên cứu, phần sau để kiểm tra ngoài mẫu.

Đây là công cụ nghiên cứu, không phải khuyến nghị đầu tư hay cam kết lợi nhuận.

## Cài đặt

```powershell
uv sync
```

## Chạy nhanh

Tải dữ liệu BTC/USDT từ năm 2018:

```powershell
uv run quant-lab download --symbol BTCUSDT --start 2018-01-01
```

Chạy chiến lược với 10 điểm cơ bản phí và 5 điểm cơ bản trượt giá mỗi lần đổi vị thế (`100 bps = 1%`):

```powershell
uv run quant-lab backtest --symbol BTCUSDT --fast 50 --slow 200 --fee-bps 10 --slippage-bps 5
```

Kết quả được in ra màn hình và lưu trong `reports/`.

## Cách dùng chiến lược trên mạng

Chỉ lấy **ý tưởng và quy tắc**, không sao chép kết quả lợi nhuận. Mỗi ý tưởng phải đi qua cùng một quy trình:

1. Viết quy tắc chính xác, không diễn giải mơ hồ.
2. Ghi nguồn và ngày tiếp cận.
3. Kiểm tra lỗi dùng dữ liệu tương lai hoặc indicator repaint (chỉ báo tự thay đổi tín hiệu cũ).
4. Tính phí, trượt giá và thời điểm khớp lệnh thực tế.
5. So sánh với buy-and-hold trên dữ liệu ngoài mẫu.
6. Thử vùng tham số lân cận; loại chiến lược chỉ tốt ở một bộ số "thần kỳ".
7. Paper trade trước khi cân nhắc vốn thật.

Ưu tiên các họ chiến lược lâu đời, dễ giải thích như trend following, breakout (phá vỡ vùng giá), momentum (động lượng) và mean reversion (hồi quy về trung bình). Không bắt đầu bằng chiến lược AI dự đoán giá hoặc code không giải thích được.
