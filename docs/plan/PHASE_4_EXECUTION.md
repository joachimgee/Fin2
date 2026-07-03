# Phase 4 — Execution (Alpaca broker + StreamManager)

> Context to load: this file + `src/execution/*.py` (its CLAUDE.md loads
> automatically) + `src/shared/interfaces.py` + `src/risk/__init__.py`.
> Prerequisite: Phase 3 GATE green.

## Objective

The only module that knows Alpaca exists. Order pipeline wired to the risk
gate, one WebSocket fanned out over Redis, positions re-synced on every
reconnect.

## Allowed imports

`execution/` → `risk` + `shared` + alpaca-py + redis. `OrderIntent` and
`ValidationResult` come from `src.risk` (re-export). Never alpaca-trade-api.

## Files

### 1. `src/execution/broker.py` — implement `AlpacaBrokerClient`

- `__init__(api_key, secret_key, paper: bool = True)`: build
  `TradingClient(api_key, secret_key, paper=paper)`. The `paper` argument is
  passed by the composition root as `not config["execution"]["live_mode"]` —
  never computed here, never read from env names.
- `submit_order(request)`: try/except `alpaca.common.exceptions.APIError` →
  `log.error("order_submit_failed", extra={symbol, code})` then raise
  `OrderRejectedError` with context. On success:
  `log.info("order_submitted", extra={order_id, symbol, side, qty, strategy_id})`,
  return `order_id` as str.
- `get_positions()` / `get_account()`: convert SDK objects to plain dicts
  (`symbol`, `qty`, `avg_entry_price`, `market_value` / `equity`, `cash`).
  SDK types never cross the module boundary.
- Helper `build_order_request(intent: OrderIntent, adjusted_qty: float)`:
  market/limit request from the VALIDATED quantity — the function signature
  makes it impossible to forget `result.adjusted_qty`.

### 2. `src/execution/stream_manager.py` — implement `StreamManager`

- `__init__(api_key, secret_key, paper, redis_client, broker, tracker, config)`.
- One `TradingStream` + one `StockDataStream` — a second connection to the
  same endpoint silently kills the first; nothing outside this class may open one.
- `_on_bar`: serialize to JSON, `PUBLISH channel:bars:{symbol}`, return.
  `_on_trade_update`: explicit dispatch, never a silent catch-all:

  | event | action |
  |---|---|
  | `fill` | `tracker.on_fill()` (via risk manager) + `log.info("order_filled", ...)` |
  | `partial_fill` | partial update + `log.warning("order_partial_fill", ...)` |
  | `canceled` / `expired` | `log.info` |
  | `new` / `pending_new` | `log.debug` |
  | `rejected` | `log.error` + injected alert callback |
  | anything else | `log.warning("unhandled_trade_event", extra={event})` |

  Callbacks publish/log and return (< 5 ms) — no feature computation, no
  awaited network calls other than the Redis publish.
- Reconnect loop: backoff `min(1 * 2**attempt, 60)` seconds
  (`config stream.*`); after EVERY reconnect and before consuming any event:
  `positions = await broker.get_positions()`, `account = await broker.get_account()`,
  `tracker.sync_from_api(positions, float(account["equity"]))`,
  `log.info("position_sync_completed", ...)`.

## Tests (mock alpaca SDK + fakeredis; zero network)

`tests/test_broker.py`: `test_paper_defaults_true`,
`test_paper_flag_forwarded_to_sdk`, `test_submit_success_returns_id_and_logs`,
`test_api_error_wrapped_in_order_rejected`, `test_positions_are_plain_dicts`,
`test_build_request_uses_adjusted_qty_param`.

`tests/test_stream_manager.py`: `test_fill_updates_tracker`,
`test_partial_fill_logs_warning`, `test_rejected_triggers_alert_callback`,
`test_unknown_event_logs_unhandled`, `test_bar_published_to_redis_channel`,
`test_backoff_sequence_1_2_4_capped_60`, `test_resync_called_after_reconnect`,
`test_no_second_stream_instantiated`.

## GATE 4

```bash
make check
pytest tests/test_broker.py tests/test_stream_manager.py -v
```

Optional (needs paper keys in `.env`, marked `@pytest.mark.integration`):
one round-trip `get_account()` against the paper endpoint.

## Definition of done

- [ ] No `NotImplementedError` left in `src/execution/`
- [ ] All tests above pass; `make check` green
- [ ] Progress ticked; committed `Phase 4: execution layer`

## Pitfalls

- Fills arrive via TradingStream, never synchronously after submit — do not
  poll order status after submitting.
- Backoff that restarts from 1 s only after a SUCCESSFUL reconnect; a failed
  attempt continues doubling.
- Importing `alpaca` anywhere else in `src/` fails `test_architecture.py` —
  if another module "needs" broker data, it receives it as plain dicts
  through injection instead.
