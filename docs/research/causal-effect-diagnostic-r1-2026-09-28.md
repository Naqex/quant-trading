# Causal effect diagnostic — research gate r1 (2026-09-28)

## Quyết định

Quant Lab chưa thêm bộ ước lượng ATE/CATE vào runtime ở vòng này. Repo hiện
chỉ có OHLCV/feature state và các diagnostic forward-only; chưa có fixture ghi rõ
một **treatment/intervention** (ví dụ tín hiệu được bật ở thời điểm nào), outcome
được quan sát sau intervention bao lâu, hay các covariate đã biết trước treatment.
Nếu thêm một ATE module ngay bây giờ, con số “effect” sẽ dễ bị gọi nhầm là causal
edge trong khi thực chất chỉ là chênh lệch có điều kiện hoặc backtest hậu nghiệm.

Đây là quyết định có chủ đích: giữ research-only và fail-closed cho tới khi có
fixture intervention có provenance. Không thêm dependency causal-ML nặng, không
kết nối provider/broker, và không nối kết quả vào chart, signal hay execution.

## Seam v1 đã đăng ký

Khi có fixture phù hợp, lát triển khai nhỏ nhất sẽ nhận mỗi dòng:

- `event_time`: thời điểm thị trường xảy ra;
- `known_at`: thời điểm toàn bộ feature/treatment của dòng có thể biết;
- `treatment`: nhị phân, được chốt tại `known_at`;
- `outcome`: outcome kỳ hạn tương lai, chỉ trở nên quan sát được sau horizon;
- `covariates`: chỉ biến đã có trước treatment, không chứa giá tương lai hoặc
  post-treatment mediator;
- `sequence`: khóa thứ tự khi nhiều dòng có cùng timestamp.

Contract phải reject `known_at > cutoff`, `event_time > known_at`, treatment
không phải 0/1, outcome/covariate không finite, duplicate point-in-time keys,
và mọi covariate có `known_at` sau thời điểm treatment. Dữ liệu không được âm
thầm sort, drop hoặc forward-fill.

Phương pháp mặc định là **temporal cross-fitted doubly robust ATE**, nhưng chỉ
trên covariate dimension nhỏ và đã pre-register:

1. chia theo thời gian thành train, calibration và holdout; tuyệt đối không
   random split và không fit scaler/model trên holdout;
2. fit outcome nuisance `m_a(x)` và propensity `e(x)` trên train ở từng fold;
3. trim/guard overlap với ngưỡng đã chốt trước, ghi rõ số dòng bị loại và
   effective sample size;
4. tính pseudo-outcome doubly robust trên calibration/holdout, kèm bootstrap
   theo block thời gian hoặc influence-function standard error;
5. so sánh với baseline difference-in-means, placebo treatment và outcome-only
   model; báo sensitivity khi thay cutoff, horizon, overlap và covariate set.

Ở vòng đầu, implementation nên dùng NumPy/Pandas đã có, ridge/logistic solver
bounded và deterministic seed; không kéo `econml`, `dowhy` hay framework model
lớn trước khi fixture chứng minh giá trị. Nếu overlap yếu hoặc số mẫu hiệu dụng
thấp, trạng thái phải là `inconclusive`, không trả effect như tín hiệu.

## Promotion gate

Chỉ được chuyển từ memo sang code khi người vận hành cung cấp fixture local có:

- treatment policy và decision timestamp rõ ràng;
- outcome horizon và cost/slippage mapping rõ ràng;
- ít nhất một covariate set pre-treatment có provenance/hash;
- temporal train/calibration/holdout đủ lớn, không overlap và không dùng holdout
  để chọn tham số;
- null/placebo fixture cùng cấu trúc để kiểm tra false positive;
- OOS đa tài sản, purged/embargoed boundary nếu horizon chồng lấn;
- stress cho propensity cực đoan, missing/duplicate/late rows và số học;
- receipt ghi `execution_capability=false`, `provider_access=false`,
  `broker_access=false`.

Một effect chỉ được xem là **research evidence** nếu vượt qua baseline đã đăng
ký, ổn định qua OOS/horizon/covariate sensitivity và không bị placebo phát hiện
như edge. Nó không tự cấp quyền đặt lệnh.

## Vì sao chưa chọn CATE/ABM/causal RL

CATE/causal forests, ABM và causal RL đều đòi hỏi treatment semantics, policy
support và sample size mạnh hơn dữ liệu hiện có. Chúng sẽ được xem xét sau khi
ATE v1 có fixture và null/OOS evidence; thêm code trước mốc đó chỉ làm tăng
complexity và rủi ro data-snooping mà chưa tạo quyết định tốt hơn.

## Nguồn phương pháp

- Hernán & Robins, *Causal Inference: What If* — phân biệt intervention,
  time-varying information và target estimand:
  <https://www.hsph.harvard.edu/magenta/wp-content/uploads/sites/1268/2024/02/hernanrobins_v2.0.1.pdf>
- Chernozhukov et al., *Double/Debiased Machine Learning for Treatment and
  Structural Parameters* — orthogonal score và cross-fitting:
  <https://academic.oup.com/ectj/article/21/1/C1/5056401>

Các nguồn trên định hướng thiết kế estimator; chúng không phải bằng chứng edge
trên thị trường crypto và không thay thế fixture/OOS của Quant Lab.

## Trạng thái

`research_only`, `implementation_pending_fixture`, `execution_capability=false`.

Kiểm chứng vòng này chỉ là review contract và full test hiện có; không có ATE
result nào được công bố hoặc nối vào backtest.
