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

## Primitive nghiên cứu offline

Các pilot P0 được giữ tách khỏi backtest/execution để có thể tái lập và
rollback độc lập:

- `quant_trading.state_primitives`: Kalman forward-only, Shannon entropy có
  Miller–Madow explicit, permutation entropy ordinal-pattern và Beta–Bernoulli
  Bayesian update.
- `quant_trading.conformal`: split-conformal interval cho point forecast đã có,
  với calibration/validation/test cutoff theo thời gian.
- `quant_trading.change_detection`: CUSUM hai phía để chẩn đoán change-point,
  với baseline/threshold explicit và reset policy tái lập.
- `quant_trading.bayesian_change_point`: BOCPD Normal-Inverse-Gamma causal,
  predictive Student-t và posterior run-length; safety bound fail-closed.
- `quant_trading.ou_process`: exact irregular-time Ornstein–Uhlenbeck fit và
  forward-only conditional diagnostics cho stationary spread/residual; raw
  price, near-unit-root fit và provider/broker/execution access đều bị chặn.
- `quant_trading.risk_metrics`: empirical VaR/CVaR, downside deviation và
  Sortino có trạng thái defined rõ ràng trên từng research window.
- `quant_trading.theory_receipts`: receipt bất biến cho hypothesis/scenario,
  ghi provenance, hash, cutoff và baseline.
- `quant_trading.distribution_drift`: Population Stability Index (PSI) với
  bin boundary chốt trước, smoothing explicit và fail-closed OOD/drift flag;
  chỉ là diagnostic governance, không phải regime label hay signal.
- `quant_trading.selective_prediction`: uncertainty threshold abstention với coverage/risk receipt thuần offline;
- `quant_trading.model_governance`: receipt champion/challenger thuần offline;
  yêu cầu cùng fixture/cutoff, holdout, calibration, decay và PSI preflight
  trước khi recommendation promote; rollback chỉ là review recommendation.
- `quant_trading.feature_registry`: registry DAG bất biến cho feature numeric;
  pin source/rule/data hash, dependency lineage và explicit event/known/available
  timestamps. Snapshot chỉ nhận dữ liệu đã khả dụng trước cutoff, loại duplicate,
  stale/invalidated rows và luôn giữ execution/provider/broker capability false.
- `quant_trading.feature_drift`: nối `FeatureSnapshot` vào PSI/OOD với registry
  hash, snapshot hashes và cutoff; chỉ nhận hàng `accepted`, dùng bin đã chốt
  trước và trả receipt `PREP_ONLY`, không biến drift thành regime hay signal.
- `quant_trading.feature_governance`: giữ nguyên feature OOD lineage khi nối
  PSI vào champion/challenger review; mismatch PSI, fixture, sample hoặc cutoff
  đều fail-closed và quyết định chỉ là recommendation offline.
- `quant_trading.walk_forward`: expanding chronological OOS folds với prefix
  replay point-in-time và cost stress theo multiplier; receipt canonical,
  boundary entry được giữ đúng và mọi capability vẫn `false`.

Các primitive này chỉ nhận fixture đã capture, không gọi provider/broker/network
và luôn giữ `execution_capability=false`. BOCPD/change-point, regime label,
ABM, game/reflexivity, chaos/SOC/TDA và mọi candidate signal chỉ được thêm sau
calibration, walk-forward/OOS, cost/stress và baseline evidence tương ứng.

Causal effect hiện có pilot typed offline tại `quant_trading.causal_effects`; chỉ fixture synthetic/local, research-only, chưa có effect claim trên thị trường. Chi tiết gate và promotion nằm tại `docs/research/causal-effect-diagnostic-r1-2026-09-28.md`.

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

## Chạy nghiên cứu so sánh chiến lược

Tải dữ liệu cho các tài sản cần so sánh, sau đó chạy:

```powershell
uv run quant-research --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT --test-start 2023-01-01 --fee-bps 10 --slippage-bps 5
```

Quy trình sẽ so sánh buy-and-hold, moving average, momentum, Donchian breakout, RSI mean reversion và lớp volatility target; đồng thời kiểm tra các vùng tham số lân cận. Báo cáo quyết định hiện tại nằm tại [docs/research/strategy-report-2026-08-23.md](docs/research/strategy-report-2026-08-23.md).

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
