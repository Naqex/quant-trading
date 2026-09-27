# Feature provenance registry pilot — 2026-09-28

## Vấn đề và quyết định

`quant_lab.provenance` đã pin được dữ liệu OHLCV, còn chart research contract
đã pin rule version và walk-forward ranges. Khoảng trống còn lại là lineage của
từng feature numeric: lúc nào feature xảy ra, lúc nào nguồn biết nó, lúc nào
consumer được phép dùng nó, và feature phụ thuộc vào artifact nào. Pilot này
thêm `quant_trading.feature_registry` làm boundary offline nhỏ, không thay thế
chart engine hoặc backtest.

## Contract

- `FeatureSpec` pin `feature_id/version`, instrument/timeframe, source schema và
  SHA-256, rule version, source references, parameters và dependency IDs.
- `FeatureRegistry` canonicalize feature DAG và từ chối dependency thiếu hoặc
  cycle; registry hash trở thành lineage anchor.
- `FeatureObservation` giữ `event_time_utc`, `known_at_utc` và
  `available_at_utc` riêng biệt, cùng source row, revision/supersedes,
  quality status và optional validity expiry.
- `FeatureSnapshot` chỉ nhận observation có `available_at_utc <= cutoff`, đủ
  consumer delay, quality không bị `invalidated`, chưa stale và không trùng
  logical `(feature_id, event_time_utc)`. Input đảo thứ tự cho cùng một snapshot
  vẫn cho cùng receipt hash.
- Hash/JSON canonical và ba capability flags đều fail-closed:
  `execution_capability=false`, `provider_access=false`, `broker_access=false`.

Revision correction không được âm thầm ghi đè lịch sử: observation revision
phải nêu `supersedes_source_row_id`, còn một snapshot không nhận hai giá trị cho
cùng feature/event. Caller phải chọn revision hiệu lực trước khi tạo snapshot.

## Evidence và giới hạn

- Synthetic fixture chỉ kiểm tra contract, không tuyên bố edge, alpha, market
  performance hoặc live/provider integration.
- Không có network, broker, API key, database migration hay execution adapter.
- Mốc tiếp theo nếu cần là adapter đọc artifact thật và quality receipt riêng;
  adapter đó phải giữ nguyên cutoff/holdout và re-check hash trước promotion.

Validation: `uv run pytest -q tests/quant_trading/test_feature_registry.py`.
