#!/usr/bin/env bash
# =============================================================================
# One-time bootstrap of the 'data-state' orphan branch.
#
# Creates the initial event-sourced ledgers (India ₹1,000,000 · US $100,000),
# the empty order queues, the event journal and the NAV history, pushes the
# orphan branch, wires GitHub Pages to Actions, and kicks the first DAG run.
#
# Usage:
#   bash scripts/bootstrap_state_branch.sh [REPO]        # REPO defaults to origin
#
# Requires: git, gh (authenticated with repo + workflow scope), jq optional.
# =============================================================================
set -euo pipefail

REPO="${1:-$(git remote get-url origin 2>/dev/null || true)}"
if [ -z "$REPO" ]; then
  echo "error: no git remote 'origin' found; pass the repo explicitly:" >&2
  echo "  bash scripts/bootstrap_state_branch.sh <owner/repo>" >&2
  exit 1
fi

BRANCH="data-state"
WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/pms-state-XXXXXX")"
trap 'rm -rf "$WORKTREE"' EXIT

echo "==> bootstrapping orphan branch '$BRANCH' in $WORKTREE"
git -C "$WORKTREE" init -q -b "$BRANCH"

now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------- ledgers -----
cat > "$WORKTREE/portfolio_ledger_india.json" <<JSON
{
  "market": "INDIA",
  "as_of": null,
  "starting_cash": 1000000.0,
  "cash": 1000000.0,
  "cash_buffer_target": 0.0,
  "positions": {},
  "realized_pnl": 0.0,
  "fees_paid": 0.0,
  "rebalance_count": 0,
  "last_run_id": null,
  "reconciled_ex_dates": []
}
JSON

cat > "$WORKTREE/portfolio_ledger_us.json" <<JSON
{
  "market": "US",
  "as_of": null,
  "starting_cash": 100000.0,
  "cash": 100000.0,
  "cash_buffer_target": 0.0,
  "positions": {},
  "realized_pnl": 0.0,
  "fees_paid": 0.0,
  "rebalance_count": 0,
  "last_run_id": null,
  "reconciled_ex_dates": []
}
JSON

cat > "$WORKTREE/pending_orders_india.json"  <<JSON
{ "market": "INDIA", "created_at": "$now", "run_id": null, "orders": [] }
JSON

cat > "$WORKTREE/pending_orders_us.json"  <<JSON
{ "market": "US", "created_at": "$now", "run_id": null, "orders": [] }
JSON

: > "$WORKTREE/ledger_events.jsonl"
: > "$WORKTREE/nav_history.jsonl"

cat > "$WORKTREE/README.md" <<'MD'
# data-state

Event-sourced portfolio state for Autonomous PMS V4. Single-writer: every workflow
serializes on the `pms-v4-global-state-lock` concurrency group.

| File | Contents |
| :-- | :-- |
| `portfolio_ledger_<market>.json` | cash, positions, realised PnL, fees, reconciled ex-dates |
| `pending_orders_<market>.json` | orders awaiting `/approve` |
| `ledger_events.jsonl` | append-only audit journal (source of truth) |
| `nav_history.jsonl` | equity/cash marks appended by every run |

Never edit these by hand: mutate them through `src/main_v4.py` (recommendations) or
`src/execution/approve_rebalance.py` (approved fills).
MD

git -C "$WORKTREE" add -A
git -C "$WORKTREE" -c user.name="pms-bootstrap" -c user.email="pms-bootstrap@users.noreply.github.com" \
  commit -q -m "chore(state): bootstrap ledgers (INDIA 1,000,000 INR · US 100,000 USD)"
git -C "$WORKTREE" remote add origin "$REPO"
git -C "$WORKTREE" push -u origin "$BRANCH"

# ------------------------------------------------------------- pages + run ----
echo "==> enabling GitHub Pages (Actions source)"
gh api -X POST "repos/${REPO#*github.com[:/]}/pages" -f build_type=workflow >/dev/null 2>&1 \
  || gh api -X PUT "repos/${REPO#*github.com[:/]}/pages" -f build_type=workflow >/dev/null 2>&1 \
  || echo "    (Pages already configured or insufficient scope — set Settings → Pages → GitHub Actions)"

echo "==> triggering the first parallel DAG run"
gh workflow run parallel_dag_pipeline.yml -f target_market=BOTH || {
  echo "    (trigger manually: gh workflow run parallel_dag_pipeline.yml -f target_market=BOTH)" >&2
}

echo
echo "done. Next:"
echo "  1. gh run list --limit 3            # follow the first DAG run"
echo "  2. the pipeline opens 'V4 Rebalance Recommendation' issues"
echo "  3. comment '/approve' (or '/approve TICKER=shares@price ...') to execute"
