# Báo cáo chọn chiến lược crypto spot v1

Ngày nghiên cứu: 23/08/2026. Dữ liệu kết thúc: 22/08/2026.

## Quyết định đề xuất

Chọn **Donchian breakout 100/50 + volatility target 30%, chia đều BTC và ETH** làm ứng viên đầu tiên để tiếp tục kiểm chứng bằng walk-forward và paper trading.

- **Donchian breakout 100/50:** sau khi giá đóng cửa vượt mức cao nhất của 100 ngày trước thì chuyển sang nắm giữ; khi giá đóng cửa xuống dưới mức thấp nhất của 50 ngày trước thì về tiền mặt.
- **Volatility target 30%:** khi biến động 30 ngày quá cao, tự giảm tỷ trọng; không vay và không nâng tỷ trọng quá 100%.
- Tín hiệu tạo ở cuối ngày chỉ có hiệu lực từ ngày kế tiếp.

Đây chưa phải chiến lược đủ điều kiện dùng tiền thật. Nó được chọn vì cân bằng tốt nhất giữa bằng chứng, độ đơn giản, độ bền tham số và kiểm soát thua lỗ; không phải vì có lợi nhuận backtest cao nhất.

## Nguồn và điều rút ra

1. [A Century of Evidence on Trend-Following Investing](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing), Hurst, Ooi và Pedersen: trend following/time-series momentum có bằng chứng dài hạn trên nhiều thị trường và nhiều điều kiện kinh tế. Đây là bằng chứng cho **họ chiến lược**, không phải bằng chứng rằng đúng bộ số Donchian 100/50 sẽ tiếp tục có lãi trên crypto.
2. [Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877), Liu và Tsyvinski: lợi nhuận crypto khác các yếu tố truyền thống và nghiên cứu ghi nhận hiệu ứng time-series momentum mạnh. Nguồn này hỗ trợ việc ưu tiên momentum/trend thay vì bắt đầu bằng mô hình dự đoán phức tạp.
3. [Meb Faber timing model](https://mebfaber.com/timing-model/): mô hình giá so với trung bình động dài hạn được công bố như một ví dụ đơn giản; tác giả nói độ ổn định tham số khá rộng và cũng cảnh báo mô hình công bố chỉ mang tính hướng dẫn. Điều này ủng hộ kiểm tra **vùng tham số**, không săn một con số tối ưu.
4. [Volatility-Managed Portfolios](https://www.nber.org/papers/w22208), Moreira và Muir: giảm rủi ro khi biến động cao đã cải thiện Sharpe trong nhiều danh mục yếu tố truyền thống. Nghiên cứu không trực tiếp về crypto; lớp volatility target trong báo cáo này được coi là giả thuyết quản trị rủi ro và đã được test riêng trên dữ liệu crypto.
5. [The Probability of Backtest Overfitting](https://ideas.repec.org/a/rsk/journ0/2471206.html), Bailey và cộng sự: khi thử nhiều cấu hình rồi chọn người thắng, backtest có thể bị overfit; một lần hold-out đơn giản cũng có giới hạn. Vì vậy báo cáo chỉ kiểm tra một vùng tham số nhỏ được định trước.
6. [Binance Spot REST API – Market](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market): nguồn dữ liệu nến ngày `GET /api/v3/klines`; mốc thời gian được xử lý theo UTC.

## Phương pháp test

- Thị trường: spot, long/cash; không short, không futures, không leverage.
- Tài sản: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT và ADAUSDT.
- Dữ liệu: nến ngày Binance từ lúc cặp có dữ liệu đến 22/08/2026.
- Train: trước 01/01/2023. Test: 01/01/2023–22/08/2026.
- Chi phí cơ sở: 10 bps phí + 5 bps slippage cho mỗi lần thay đổi toàn bộ một đơn vị tỷ trọng. `100 bps = 1%`.
- Tiền mặt sinh lãi 0%; chưa tính thuế, funding, spread biến đổi hoặc hạn chế khớp lệnh thực tế.
- Các chỉ số:
  - **CAGR:** tốc độ tăng trưởng kép trung bình năm.
  - **Sharpe:** lợi nhuận trung bình so với độ biến động; ở đây dùng lãi suất phi rủi ro bằng 0.
  - **Max drawdown:** mức sụt giảm lớn nhất từ đỉnh xuống đáy.
  - **Calmar:** CAGR chia cho độ lớn max drawdown; cao hơn thường tốt hơn.

## Các ứng viên

| Ứng viên | Quy tắc | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Buy and hold | Luôn giữ tài sản | Đơn giản, bắt trọn thị trường tăng, ít phí | Chịu toàn bộ bear market; drawdown rất lớn |
| Price/SMA 200 | Giữ khi giá trên SMA 200 ngày | Dễ hiểu, có nền tảng nghiên cứu, lợi nhuận test khá | Dễ whipsaw – mua/bán liên tục khi giá đi ngang; ADA thất bại nặng |
| MA 50/200 | Giữ khi MA 50 trên MA 200 | Ít giao dịch | Phản ứng chậm; kết quả test giảm mạnh so với train |
| Momentum 180 | Giữ khi lợi nhuận 180 ngày dương | Bám sát hiệu ứng momentum trong nghiên cứu crypto | Nhạy lookback, giao dịch khá nhiều, ETH test âm |
| Donchian 100/50 | Mua khi phá đỉnh 100 ngày, thoát khi phá đáy 50 ngày | Quy tắc rõ, vùng tham số ổn định nhất, 5/5 tài sản test dương | Bỏ lỡ đoạn đầu xu hướng; vẫn có false breakout – phá vỡ giả |
| RSI mean reversion | Mua khi RSI quá bán trong xu hướng dài hạn | Thời gian giữ ngắn, drawdown thấp do ít vào lệnh | Gần như không tạo lợi nhuận; không có bằng chứng đủ để tiếp tục cấu hình này |

## Kết quả ngoài mẫu theo từng tài sản

Các số dưới đây là median – trung vị của 5 tài sản, không phải kết quả một tài khoản duy nhất.

| Chiến lược | CAGR trung vị | Sharpe trung vị | Drawdown trung vị | Drawdown xấu nhất | Số tài sản CAGR dương |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 32,37% | 0,805 | -67,55% | -88,29% | 4/5 |
| Price/SMA 200 | 25,10% | 0,725 | -53,61% | -82,23% | 4/5 |
| Donchian 100/50 | 17,55% | 0,617 | -43,08% | -52,04% | 5/5 |
| Momentum 180 | 13,14% | 0,500 | -59,27% | -77,37% | 3/5 |
| MA 50/200 + vol 30% | 6,34% | 0,375 | -29,06% | -36,79% | 4/5 |
| MA 50/200 | 5,87% | 0,410 | -61,51% | -73,06% | 4/5 |
| RSI mean reversion | 0,00% | 0,083 | -6,71% | -20,26% | 2/5 |

Buy and hold thắng về lợi nhuận vì test chứa giai đoạn tăng mạnh, nhưng mức sụt giảm rất khó chịu. Price/SMA 200 đứng đầu theo Calmar trung vị, nhưng có hai dấu hiệu yếu: turnover trung vị khoảng 11,24 lần vốn/năm và drawdown ADA -82,23%. Donchian kiếm ít hơn nhưng nhất quán hơn giữa các tài sản.

## Kiểm tra vùng tham số

Không chọn Donchian 100/50 vì nó là bộ số tốt nhất. Ba biến thể được định trước đều có kết quả tương đối gần nhau:

| Biến thể | CAGR trung vị | Sharpe trung vị | Drawdown trung vị | Tài sản CAGR dương |
|---|---:|---:|---:|---:|
| Donchian 60/30 | 22,88% | 0,675 | -40,73% | 5/5 |
| Donchian 100/50 | 17,55% | 0,617 | -43,08% | 5/5 |
| Donchian 150/75 | 15,82% | 0,587 | -44,26% | 4/5 |

Đây là dấu hiệu tốt hơn một đỉnh lợi nhuận đơn lẻ. Tôi chọn 100/50 ở giữa vùng thay vì 60/30 có kết quả đẹp hơn, nhằm giảm quyết định dựa trên dữ liệu test.

## Danh mục có thể vận hành

Để tránh biến phiên bản đầu thành hệ thống năm coin khó quản lý, đề xuất chỉ dùng BTC và ETH, mỗi tài sản nhận 50% vốn chiến lược.

Kết quả test của **BTC + ETH Donchian 100/50 + vol 30%**:

- CAGR: **11,56%/năm**.
- Sharpe: **0,684**.
- Max drawdown: **-17,30%**.
- Calmar: **0,668**.
- Tỷ trọng đầu tư trung bình toàn danh mục: **25,42%**; phần còn lại là tiền mặt.
- Stress test tăng tổng chi phí từ 15 lên 40 bps mỗi lần đổi tỷ trọng: CAGR còn **10,28%**, drawdown **-17,89%**.

Để tham chiếu, danh mục năm coin cùng chiến lược có CAGR 12,64%, Sharpe 0,841 và drawdown -14,73%. Kết quả đẹp hơn nhưng chưa đáng đánh đổi sự phức tạp và survivorship bias ở phiên bản đầu.

## Vì sao chưa chọn phương án khác

- **Buy and hold:** phù hợp nếu mục tiêu là tối đa hóa upside và chấp nhận tài khoản giảm 50–80%; không phù hợp làm hệ thống đầu tiên với người chưa trải qua drawdown thật.
- **Price/SMA 200:** đáng giữ làm challenger – đối thủ so sánh, nhưng chưa chọn vì giao dịch dày và thất bại nặng trên ADA.
- **MA 50/200:** train đẹp nhưng test yếu; đây là dấu hiệu suy giảm hiệu quả rõ.
- **Momentum 180:** bằng chứng học thuật tốt nhưng cấu hình đơn lẻ này thiếu ổn định giữa BTC, ETH và altcoin.
- **Mean reversion:** loại khỏi vòng tiếp theo; thêm indicator để cứu nó lúc này dễ dẫn đến overfit.

## Hạn chế cần hiểu trước khi tin kết quả

1. **Survivorship bias:** năm coin được chọn vì hiện vẫn lớn và giao dịch được. Backtest không chứa các coin từng lớn rồi chết hoặc bị hủy niêm yết.
2. **Test set đã bị “tiêu thụ”:** sau báo cáo này, giai đoạn 2023–2026 không còn là dữ liệu hoàn toàn chưa nhìn thấy. Không được tiếp tục sửa chiến lược rồi gọi cùng giai đoạn là out-of-sample sạch.
3. **Mẫu ngắn:** crypto Binance chỉ có khoảng 6–9 năm; SOL ngắn hơn. Một vài chu kỳ là chưa đủ để chứng minh edge – lợi thế giao dịch bền vững.
4. **Close-to-close approximation:** mô hình dùng lợi nhuận đóng cửa sang đóng cửa và chi phí cố định. Khớp lệnh thật, spread lúc biến động và downtime có thể xấu hơn.
5. **Không có lãi tiền mặt:** đây là giả định thận trọng cho lợi nhuận, nhưng chưa mô phỏng stablecoin depeg, rủi ro sàn hay yield.
6. **Không phải kiểm định thống kê hoàn chỉnh:** chưa chạy CPCV/PBO, bootstrap hay walk-forward. Báo cáo là vòng sàng lọc ứng viên, chưa phải chứng minh khoa học cuối cùng.

## Bước tiếp theo được phép

1. Đóng băng quy tắc Donchian 100/50 + vol 30%; không chỉnh tiếp theo kết quả 2023–2026.
2. Thêm walk-forward theo năm và kiểm tra data quality từ một nguồn độc lập.
3. Paper trade BTC/ETH tối thiểu 3 tháng, ghi mọi tín hiệu và giá có thể khớp thực tế.
4. Chỉ bàn đến tiền thật rất nhỏ nếu paper trade khớp hành vi backtest; chưa dùng leverage.

Các CSV đầy đủ được tạo trong `reports/research/` bằng lệnh:

```powershell
uv run quant-research --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT --test-start 2023-01-01 --fee-bps 10 --slippage-bps 5
```
