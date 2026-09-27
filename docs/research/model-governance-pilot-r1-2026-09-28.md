# Champion/challenger governance pilot r1 — 2026-09-28

## Quyết định

Quant Lab có thêm `quant_trading.model_governance`, một contract thuần
offline cho vòng review Adaptive Markets. Contract này nhận hai evaluation đã
được capture trên **cùng fixture và cùng cutoff**: champion đang giữ và
challenger đề xuất. Nó trả về một receipt bất biến với một trong bốn hành động:

- `promote_challenger`: challenger vượt ngưỡng score và qua toàn bộ health gate;
- `retain_champion`: champion còn khỏe hoặc challenger chưa đủ bằng chứng;
- `rollback_review`: champion không còn khỏe, challenger chưa đủ điều kiện, và
  có `rollback_ref` để outer controller/user xem xét khôi phục;
- `inconclusive`: phép so sánh không đáng tin (fixture/cutoff lệch, evidence
  chưa hoàn chỉnh hoặc không có rollback target).

Đây là recommendation có provenance, **không tự đổi model, không tự rollback,
không gọi provider/broker và không mở execution capability**.

## Contract và gate

`GovernancePolicy` chốt trước các tham số: số mẫu tối thiểu, mức score cải thiện
tối thiểu, calibration error tối đa, decay tối đa, yêu cầu holdout, yêu cầu PSI
preflight và rollback reference khi promote. `ModelEvaluation` ghi model/data
hash, cutoff, score hiện tại, score reference, calibration error, holdout
pass/fail và `PSIResult` hiện có.

Decay dùng `reference_score - score` thay vì tỷ lệ. Cách này ổn định hơn khi
score gần 0 hoặc âm, đồng thời giữ rõ đơn vị utility đã đăng ký. PSI được nối
trực tiếp: thiếu PSI là fail-closed khi policy yêu cầu; `drift_flag=true` không
được promote.

Không dùng một metric tổng hợp bí ẩn. Candidate chỉ được promote khi:

1. fixture hash và cutoff trùng champion;
2. evidence status là `accepted`, đủ sample và holdout pass;
3. calibration error không vượt ngưỡng;
4. decay không vượt ngưỡng;
5. PSI preflight tồn tại và không cảnh báo drift;
6. score candidate tăng **strictly greater** `minimum_score_improvement`;
7. candidate có rollback reference nếu policy yêu cầu.

Nếu incumbent fail health gate, receipt chỉ ghi `rollback_review`; outer
controller phải thực hiện review/restore riêng. Không có online parameter
mutation và không biến kết quả này thành paper/live authority.

## Vì sao chọn lát này

Workspace đã có primitive drift/PSI, conformal/calibration và receipt hash,
nhưng chưa có boundary nối chúng vào policy champion/challenger. Lát này nhỏ,
không thêm dependency, reuse `PSIResult`, dễ rollback và tạo được evidence
cho bước shadow/challenger sau này. Nó không phải backtest mới và không claim
market edge.

## Validation và giới hạn

Focused tests kiểm tra promotion đủ gate, retain khi calibration fail,
rollback review khi champion drift, fixture/cutoff mismatch, PSI missing,
count/policy guard và deterministic receipt. Chạy:

```powershell
uv run pytest -q tests/quant_trading/test_model_governance.py
uv run ruff check src/quant_trading/model_governance.py tests/quant_trading/test_model_governance.py
uv run python -m compileall -q src/quant_trading tests/quant_trading
```

Contract chưa có rolling-window scheduler, probability calibration estimator,
WFO/locked-holdout runner, PBO/DSR/reality-check hay persistence catalog. Các
phần đó phải được bổ sung ở slice riêng với fixture thật và giữ `PREP_ONLY`.

