# Portfolio Agent — data-state branch

Event-sourced portfolio state, committed automatically by the fan-in stage of
`.github/workflows/parallel_dag_pipeline.yml` after every run.

| file | purpose |
| --- | --- |
| `portfolio_ledger_us.json` | US book snapshot (starting cash $100,000) |
| `portfolio_ledger_india.json` | India book snapshot (starting cash ₹1,000,000) |
| `ledger_events.jsonl` | append-only journal of every portfolio mutation |
| `nav_history.jsonl` | daily NAV/equity marks for the dashboard |
| `pending_orders_us.json` / `pending_orders_india.json` | queues awaiting `/approve` |

Do not hand-edit: the reducer and the `/approve` reconciler own this branch.
