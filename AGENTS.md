# AGENTS.md

## Mục tiêu project

Quant Lab là môi trường nghiên cứu crypto spot khung ngày. Ưu tiên tính đúng, khả năng tái lập và quản trị rủi ro trước lợi nhuận backtest. Không coi kết quả quá khứ là cam kết lợi nhuận tương lai.

## Môi trường và lệnh chuẩn

- Windows + PowerShell.
- Python 3.12, dependency và virtual environment do `uv` quản lý.
- Cài đặt: `uv sync`.
- Chạy test: `uv run pytest -q`.
- Tải dữ liệu mẫu: `uv run quant-lab download --symbol BTCUSDT --start 2018-01-01`.
- Backtest mẫu: `uv run quant-lab backtest --symbol BTCUSDT`.

Không thêm dependency hoặc MCP nếu công cụ hiện có giải quyết được nhiệm vụ với độ phức tạp thấp hơn.

## Cấu trúc và ownership

- `src/quant_lab/data.py`: tải, lưu và đọc dữ liệu thị trường.
- `src/quant_lab/strategies.py`: quy tắc tạo target position (tỷ trọng mục tiêu) từ dữ liệu đã biết.
- `src/quant_lab/backtest.py`: mô phỏng position, turnover, chi phí và equity.
- `src/quant_lab/metrics.py`: chỉ số hiệu suất và rủi ro.
- `src/quant_lab/cli.py`: command-line interface, không chứa logic chiến lược cốt lõi.
- `tests/`: kiểm thử accounting, timing và quy tắc chiến lược.
- `docs/research/`: giả thuyết, nguồn, phương pháp và báo cáo nghiên cứu.
- `data/raw/` và `reports/`: artifact có thể tạo lại, không commit.

Tìm và mở rộng implementation hiện có trước khi tạo module hoặc abstraction mới.

## Quy tắc bắt buộc cho backtest

- Một tín hiệu tính bằng giá đóng cửa ngày `t` chỉ được giữ vị thế từ ngày `t+1`.
- Luôn tính phí và slippage khi target position thay đổi.
- Spot lab chỉ cho phép target position từ 0 đến 1; không short, không leverage.
- Không sửa tham số dựa trên test/out-of-sample. Train dùng để phát triển; test chỉ dùng để đánh giá.
- Mỗi chiến lược phải so với buy-and-hold và báo cáo ít nhất: CAGR, volatility, Sharpe, max drawdown, turnover, số lần vào lệnh và time in market.
- Báo cáo rõ survivorship bias, data-snooping/overfitting và hạn chế dữ liệu khi chúng tồn tại.
- Với dữ liệu nhiều tài sản, tránh vô tình dùng ngày tài sản chưa được niêm yết hoặc forward-fill giá trước niêm yết.

## Quy trình thêm chiến lược

1. Viết giả thuyết kinh tế/hành vi và quy tắc không mơ hồ.
2. Ghi nguồn gốc ý tưởng trong `docs/research/`.
3. Viết test cho timing, trạng thái vào/thoát và edge case.
4. Chạy trên cùng dữ liệu, chi phí và split với benchmark.
5. Kiểm tra một vùng tham số nhỏ được định trước; không grid-search để săn bộ số đẹp.
6. Chỉ promote sang paper trading khi kết quả ngoài mẫu và đa tài sản đủ ổn định.

## Ranh giới an toàn

- Không kết nối tài khoản sàn, tạo API key, đặt lệnh, dùng tiền thật, futures hoặc leverage nếu người dùng chưa yêu cầu rõ và xác nhận bước có rủi ro.
- Không lưu secret trong repo. Nếu sau này cần credential, dùng environment variable và cung cấp `.env.example` không chứa giá trị thật.
- Không diễn đạt kết quả research như khuyến nghị tài chính cá nhân hoặc đảm bảo lợi nhuận.
