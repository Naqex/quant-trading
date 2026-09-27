# Split-conformal interval pilot — 2026-09-27

## Quyết định

Quant Lab có một primitive split-conformal offline, không phụ thuộc provider:
`quant_trading.conformal.fit_split_conformal`. Primitive này nhận point
forecast đã có, dùng residual tuyệt đối trên một calibration window theo thời
gian, rồi trả interval đối xứng với finite-sample rank
`ceil((n + 1) * (1 - alpha))`. Nó không huấn luyện predictor và không phải
bằng chứng về alpha hay lợi nhuận.

## Contract đã khóa

- Calibration timestamp phải nằm sau `train_end_utc` và không vượt
  `validation_end_utc`; prediction timestamp phải nằm sau validation và không
  vượt `test_end_utc`.
- `y_true`, `y_pred`, bounds và residual phải là số hữu hạn; residual overflow,
  interval overflow, timestamp trùng/đảo thứ tự và SHA-256 sai đều bị từ chối.
- Receipt chứa `alpha`, quantile, số mẫu, cutoff, digest residual, digest toàn
  bộ calibration (timestamps + residuals) và digest dữ liệu.
  `execution_capability`, `provider_access`, `broker_access` đều bị khóa
  `false`.
- Coverage chỉ là phép đo evaluation-only qua `empirical_coverage`; không dùng
  test để hiệu chỉnh hoặc tự động promote model.

## Validation fixture

Với residual `[1, 2, 1, 3]`, `alpha=0.25`, rank là `ceil(5 * 0.75)=4`, nên
quantile bằng `3.0`. Receipt mẫu nằm ở
`conformal-interval-pilot-r1-2026-09-27.json`.

Đã kiểm chứng:

```text
uv run pytest -q tests/quant_trading/test_conformal.py
9 passed

uv run ruff check src/quant_trading/conformal.py tests/quant_trading/test_conformal.py
PASS
```

## Giới hạn và bước promotion

Split conformal chỉ cho coverage marginal dưới điều kiện trao đổi (exchangeability)
phù hợp; thị trường có drift/regime shift nên phải đánh giá coverage theo từng
regime, thời gian và horizon. Bước tiếp theo là coverage monitor có calibration
window rolling và change-point gate, vẫn giữ holdout khóa cứng. Chưa thêm
adaptive conformal hay dependency nặng trong slice này.

Nguồn phương pháp: [A Gentle Introduction to Conformal Prediction and
Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
