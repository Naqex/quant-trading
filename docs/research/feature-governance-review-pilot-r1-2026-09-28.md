# Feature OOD → champion/challenger governance pilot r1 — 2026-09-28

## Quyết định

Quant Lab thêm `quant_trading.feature_governance.review_feature_drift` để
đưa `FeaturePSIResult` vào vòng review đã có của
`quant_trading.model_governance`, đồng thời giữ nguyên lineage của feature.
Receipt kết quả chứa:

- registry hash và reference/current snapshot hashes của feature OOD;
- hai model evaluation và PSI hash mà chúng thật sự dùng;
- quyết định champion/challenger hiện có cùng policy, reason codes và cutoff.

Review bắt buộc candidate và champion cùng mang đúng `PSIResult` của
`FeaturePSIResult` theo canonical hash. Mismatch, thiếu PSI, khác fixture,
khác sample count hoặc lệch cutoff đều fail-closed. Như vậy outer controller
không thể đính một drift receipt thuận tiện vào model evidence khác nguồn.

## Hành vi

- PSI không drift và challenger qua các gate còn lại: trả recommendation
  `promote_challenger` theo governance hiện có.
- PSI drift: health gate hiện có đánh dấu candidate/champion không khỏe; nếu
  champion có rollback reference, receipt trả `rollback_review`. Đây chỉ là
  review recommendation, không tự rollback hay đổi model.
- Mọi capability flag vẫn là `false`; không có provider, broker, wallet,
  network hay execution path.

Lát này không thay đổi threshold, không huấn luyện model, không claim edge và
không biến PSI thành regime label hoặc lệnh giao dịch. Promotion thực tế vẫn
cần fixture cố định, chronological walk-forward/OOS, calibration, cost,
stress và holdout evidence.

## Kiểm chứng

```powershell
uv run pytest -q tests/quant_trading/test_feature_governance.py
uv run ruff check src/quant_trading/feature_governance.py tests/quant_trading/test_feature_governance.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Test kiểm tra promotion khi không drift, rollback review khi drift, mismatch
PSI/missing PSI, cutoff mismatch, tampered receipt và deterministic JSON/hash.
