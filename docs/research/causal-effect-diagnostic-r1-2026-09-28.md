# Causal effect diagnostic — research pilot r1 (2026-09-28)

## Quyết định

Quant Lab hiện có một pilot causal-effect **offline, provider-free và research-only**
ở `src/quant_trading/causal_effects.py`. Pilot này không nối vào chart,
signal, backtest hay execution. Nó chỉ kiểm tra cơ chế trên fixture synthetic/local;
chưa có kết quả effect trên dữ liệu thị trường và không tạo cam kết lợi nhuận.

Không thêm `econml`, `dowhy` hoặc framework causal-ML lớn. Pilot dùng NumPy đã có,
solver ridge/logistic bounded và hash deterministic để giữ dependency/rollback nhỏ.

## Hợp đồng point-in-time

Mỗi `TreatmentObservation` phải có:

- `unit_id`: tài sản hoặc đơn vị nghiên cứu;
- `event_time`: thời điểm thị trường/treatment event;
- `known_at`: thời điểm treatment và toàn bộ covariate pre-treatment đã biết;
- `outcome_known_at`: thời điểm outcome kỳ hạn đã quan sát đầy đủ;
- `treatment`: binary `0/1`, chốt tại `known_at`;
- `outcome`: số hữu hạn, chỉ dùng sau `outcome_known_at`;
- `covariates`: mapping hữu hạn, giới hạn 16 biến và phải có trước treatment;
- `sequence`: khóa ổn định khi replay.

Contract reject `known_at < event_time`, `outcome_known_at <= known_at`, treatment
không phải 0/1, giá trị không finite/quá lớn, covariate set thay đổi, duplicate
`(unit_id,event_time,sequence)` hoặc `(unit_id,known_at,sequence)`. Input được
sort deterministic cho replay nhưng không âm thầm drop, forward-fill hay sửa dòng.

`TemporalCausalSplit` chia theo **outcome availability**, không random split:

1. `outcome_known_at <= train_end`: fit nuisance models;
2. `train_end < outcome_known_at <= calibration_end`: calibration/evaluation;
3. `outcome_known_at > calibration_end` (tùy chọn `holdout_end`): holdout.

Vì outcome horizon được ghi rõ, outcome tương lai không thể lọt vào train. Khi
fixture thật có horizon chồng lấn giữa các event, caller phải pre-register purge/
embargo đủ dài trước khi tạo split; module không tự đoán horizon.

## Estimator bounded v1

Entry point là `run_causal_effect_pilot(...)`. Pilot:

- robust-scale covariates bằng median/MAD fit chỉ trên train;
- fit hai outcome nuisance ridge `m0(x)`, `m1(x)` theo từng treatment arm trên
  train;
- fit propensity logistic có ridge trên train, clip theo `propensity_epsilon`;
- tính augmented inverse-propensity (AIPW) trên calibration và holdout mà không
  fit lại bằng outcome ở các split sau;
- báo `baseline_difference_in_means` và placebo treatment shift deterministic;
- báo overlap fraction và weighted effective sample size cho holdout.

Nếu train/holdout thiếu một arm, overlap dưới ngưỡng, ESS thấp hoặc split chưa đủ,
result là `status="inconclusive"` và không trả effect estimate. Receipt giữ
`data_sha256`, `model_sha256`, split, estimator version và luôn khóa:
`execution_capability=false`, `provider_access=false`, `broker_access=false`.

Đây là temporal out-of-sample nuisance fit, chưa phải cross-fitting nhiều fold
hoặc CATE. AIPW vẫn cần giả định consistency, no unmeasured confounding và
positivity; receipt cơ học không chứng minh các giả định đó.

## Kiểm chứng hiện có

Synthetic tests tại `tests/quant_trading/test_causal_effects.py` kiểm tra:

- deterministic hash khi input bị đảo thứ tự;
- split theo thời điểm outcome và không leak nhãn tương lai;
- positive synthetic treatment effect chỉ để kiểm tra plumbing;
- thiếu treatment arm trả `inconclusive`;
- event/known/outcome timestamp, binary treatment, finite bound;
- duplicate PIT keys, covariate drift, split/config boundary;
- capability flags luôn deny-only.

Lát này không chạy provider, broker, wallet, network hay live account. Receipt mechanics-only nằm tại `docs/research/causal-effect-pilot-r1-2026-09-28.json`; trường `market_edge_claim=false` là invariant của artifact.

## Promotion gate cho dữ liệu thật

Chỉ dùng pilot trên fixture thị trường khi đã có:

- treatment policy và decision timestamp rõ ràng;
- outcome horizon, missing/censored policy, cost/slippage mapping;
- covariate pre-treatment có provenance/hash;
- temporal train/calibration/holdout đủ lớn, purge/embargo khi horizon overlap;
- null/placebo fixture cùng temporal structure;
- OOS đa tài sản, sensitivity theo horizon/cutoff/covariate/overlap;
- stress cho propensity cực đoan, missing/duplicate/late rows và số học;
- baseline đã đăng ký trước và receipt tương ứng.

Một effect chỉ là research evidence khi ổn định ngoài mẫu, vượt baseline theo tiêu
chí đã chốt và không bị placebo phát hiện như edge. Nó không tự cấp quyền đặt lệnh.

## Vì sao chưa chọn CATE/ABM/causal RL

CATE/causal forests, ABM và causal RL cần treatment semantics, policy support và
sample size mạnh hơn dữ liệu hiện có. Chúng chỉ được xem xét sau khi ATE pilot có
fixture thật, null/OOS evidence và một quyết định cụ thể mà ATE không giải quyết
được; thêm code sớm sẽ tăng complexity và data-snooping risk.

## Nguồn phương pháp

- Hernán & Robins, *Causal Inference: What If* — intervention, time-varying
  information và estimand:
  <https://www.hsph.harvard.edu/magenta/wp-content/uploads/sites/1268/2024/02/hernanrobins_v2.0.1.pdf>
- Chernozhukov et al., *Double/Debiased Machine Learning for Treatment and
  Structural Parameters* — orthogonal score và nuisance separation:
  <https://academic.oup.com/ectj/article/21/1/C1/5056401>

Nguồn trên định hướng phương pháp; chúng không phải bằng chứng edge crypto và
không thay thế fixture/OOS của Quant Lab.

## Trạng thái

`research_only`, `synthetic_mechanics_verified`, `execution_capability=false`.
Không có effect claim trên dữ liệu thị trường.
