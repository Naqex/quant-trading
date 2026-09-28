# Feature provenance → OOD/PSI pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm `quant_trading.feature_drift.compute_feature_snapshot_psi` để
nối hai boundary đã có:

- `FeatureSnapshot` xác nhận feature nào đã khả dụng tại cutoff, cùng registry
  DAG, revision/quality và snapshot hash;
- `distribution_drift.compute_psi` tính Population Stability Index (PSI) với
  bin cố định, smoothing explicit và cutoff causal.

Adapter này chỉ chọn một `feature_id` từ hai snapshot đã capture. Nó giữ lại
registry hash, reference/current snapshot hash và hiệu lực cutoff trong một
`FeaturePSIResult`. Registry phải giống hệt ở hai phía để không so sánh hai
định nghĩa feature khác nhau. Hàng `suspect` hoặc `invalidated` bị chặn trước
khi tính toán; không âm thầm bỏ qua chất lượng kém.

## Luồng dữ liệu

1. Caller tạo hai `FeatureSnapshot` từ artifact local đã có provenance.
2. Adapter kiểm tra registry, feature ID, hàng được chấp nhận và cutoff.
3. `available_at_utc` trở thành `known_at` của observation PSI; thứ tự local
   được gán deterministic để xử lý nhiều hàng cùng thời điểm khả dụng.
4. PSI dùng `PSIConfig.bin_edges` đã đăng ký trước. Bin không được suy ra từ
   current sample và giá trị ngoài support bị từ chối.
5. Kết quả trả về receipt `PREP_ONLY`, hash ổn định và luôn giữ
   `execution_capability=false`, `provider_access=false`, `broker_access=false`.

Cutoff mặc định là biên mới hơn của hai snapshot, cho phép reference window cũ
được so với current window mới. Cutoff chỉ định rõ không được vượt biên mới
nhất; hàng có `available_at_utc` sau cutoff bị reject bởi PSI contract.

## Ý nghĩa và giới hạn

PSI chỉ là màn hình drift/OOD để quyết định cần review hoặc abstain. Nó không
phải regime label, signal, causal effect, edge hay lệnh giao dịch. Ngưỡng PSI
vẫn phải được pre-register và kiểm tra với baseline, calibration, holdout,
stress và cost trước khi dùng trong governance. Lát này không đọc provider,
broker, RPC, wallet hoặc database.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_feature_drift.py
uv run ruff check src/quant_trading/feature_drift.py tests/quant_trading/test_feature_drift.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Test tập trung kiểm tra lineage/hash ổn định, registry mismatch, feature không
tồn tại, suspect rows, cutoff tương lai và reject dữ liệu sau cutoff.
