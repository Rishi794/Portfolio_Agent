#!/usr/bin/env bash
# =============================================================================
# Local, fully offline demo: deterministic synthetic feature store -> pipeline
# -> trade ticket + Plotly dashboard + walk-forward validation.
#
# No network. No API keys. Uses the dependency-free hashing embedding backend.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/demo_output}"
mkdir -p "$OUT"

cd "$ROOT"
echo "==> uv sync (dev extra only — no ONNX download needed for the demo)"
uv sync --extra dev

echo "==> running the V4 reducer on a synthetic feature store"
uv run python -m src.main_v4 \
  --synthetic \
  --data-dir "$OUT/data_lake" \
  --state-dir "$OUT/state" \
  --output-issue "$OUT/issue.md" \
  --dashboard-html "$OUT/pages/index.html" \
  --backend hashing \
  --synthetic-days 320 \
  --synthetic-tickers 40

echo "==> verifying lakehouse integrity"
uv run python -m src.storage.verify_parquet --data-dir "$OUT/data_lake"

echo "==> PIT walk-forward validation (both markets)"
uv run python -m src.backtest.walk_forward_v4 \
  --lake-dir "$OUT/data_lake" \
  --state-dir "$OUT/state" \
  --market BOTH

echo
echo "artifacts:"
echo "  trade ticket : $OUT/issue.md"
echo "  dashboard    : $OUT/pages/index.html   (open in a browser)"
echo "  state branch : $OUT/state/*.json, nav_history.jsonl"
echo
echo "Try the approval flow locally:"
echo "  uv run python -m src.execution.approve_rebalance \\"
echo "      --state-dir $OUT/state --comment-body '/approve' \\"
echo "      --issue-number 1 --actor local-demo --receipt-out $OUT/receipt.md"
