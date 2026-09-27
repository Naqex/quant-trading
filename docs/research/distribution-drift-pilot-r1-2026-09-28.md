# Distribution drift / PSI pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm một diagnostic Population Stability Index (PSI) nhỏ, chạy
offline và causal. PSI so sánh phân bố của một feature giữa sample tham chiếu
và sample hiện tại trên **bin edges đã chốt trước**. Nó phục vụ Adaptive
Markets/model governance và OOD review: phát hiện khi dữ liệu đã rời khỏi
phạm vi calibration để có thể abstain hoặc mở review. Nó không gắn nhãn regime,
không dự báo giá và không cấp quyền giao dịch.

## Hợp đồng

- Module: `src/quant_trading/distribution_drift.py`.
- Entry point: `compute_psi(reference, current, config, cutoff_timestamp=...)`.
- Input dùng lại `CausalObservation`; rows có `known_at` sau cutoff bị reject,
  không âm thầm bỏ qua. Mỗi sample phải có key `(known_at, sequence)` duy nhất.
- `PSIConfig.bin_edges` phải hữu hạn và tăng nghiêm ngặt. Giá trị ngoài support
  bị reject thay vì bị kẹp vào tail bucket; nhờ vậy drift ở tail không bị che.
- `epsilon` được cộng vào mọi bin trước khi chuẩn hóa để bin rỗng không sinh
  `log(0)`. Threshold chỉ là ngưỡng review có provenance, không phải tín hiệu.
- Receipt ghi count/probability/contribution từng bin, sample hashes, cutoff và
  `execution_capability=false`, `provider_access=false`, `broker_access=false`.

## Vì sao chọn lát này

Workspace đã có Kalman/Bayesian/entropy/CUSUM/BOCPD/conformal; PSI bổ sung một
kiểm tra drift đơn giản trước khi dùng các primitive đó ngoài miền calibration.
Nó không thêm dependency, có null “identical samples ⇒ PSI≈0”, và dễ so sánh
với KS/energy/MMD sau này. Đây là bước có giá trị vận hành cao hơn việc nhảy
thẳng vào TDA, GNN, RL hoặc chaos diagnostics khi chưa có fixture đủ lớn.

## Gate promotion

Trước khi nối vào chart/regime filter hoặc model selection cần:

1. chốt bin edges và threshold trên train, không học từ OOS;
2. so sánh PSI với KS/energy hoặc block-bootstrap null trên nhiều asset;
3. đo false-alarm budget, detection delay và stability theo cửa sổ/scale;
4. kiểm tra purged walk-forward, locked holdout, cost/stress và missing/tail data;
5. nếu drift flag không cải thiện abstention/calibration hoặc làm tăng churn,
   giữ research-only hoặc rollback.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_distribution_drift.py
uv run ruff check src/quant_trading/distribution_drift.py tests/quant_trading/test_distribution_drift.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Focused tests kiểm tra shift synthetic, identical-sample null, deterministic
hash, cutoff/duplicate rejection, out-of-support rejection, empty input và
finite/config guards. Đây là cơ chế drift trên fixture, không phải bằng chứng
edge hay lợi nhuận.
