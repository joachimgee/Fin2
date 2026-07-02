# Phase 7 — Monitoring (JSON logging + Telegram alerts)

> Context to load: this file + `src/monitoring/*.py`.
> Prerequisite: Phase 6 GATE green.

## Objective

Every event queryable in a log aggregator; critical events pushed to Telegram.
Monitoring can observe everything and break nothing.

## Allowed imports

`monitoring/` → `shared` only (+ httpx). It never imports risk/execution —
they reach it exclusively through injected callbacks (DIP, wired in Phase 8).

## Files

### 1. `src/monitoring/logging_setup.py` — implement `setup_logging`

- Custom `logging.Formatter` emitting one JSON object per line:
  `ts` (ISO-8601 UTC), `level`, `logger`, `event` (= the message), plus every
  key from `extra` flattened at top level (collisions with reserved
  `LogRecord` attrs get an `x_` prefix).
- Root logger → stdout handler, level from `config monitoring.log_level`.
- Called exactly once per process, by the entrypoint. Idempotent (second
  call must not duplicate handlers).

### 2. `src/monitoring/alerts.py` — implement `send_telegram_alert`

- `httpx.AsyncClient` POST
  `https://api.telegram.org/bot{token}/sendMessage` with `chat_id`, `text`;
  token/chat_id are parameters, resolved via `require_env` at composition root.
- Timeout ≈ 5 s. `except httpx.HTTPError`: `log.error("telegram_alert_failed",
  extra={...})` and RETURN — fire-and-forget. An alerting outage must never
  propagate into the trading path.
- Add sync wrapper `dispatch_alert(message)` suitable as the `on_trip` /
  `rejected` callback (schedules the coroutine on the running loop).

### 3. Daily report (small, same file or `report.py`)

- `build_daily_report(tracker, storage) -> str`: P&L, exposure, drawdown,
  trades count. Scheduling (16:05 ET) belongs to the Phase 8 entrypoint,
  not here.

## Tests

`tests/test_logging_setup.py`: `test_each_line_is_valid_json`,
`test_extra_fields_at_top_level`, `test_reserved_key_collision_prefixed`,
`test_double_setup_no_duplicate_handlers`, `test_level_from_config`.

`tests/test_alerts.py` (httpx MockTransport): `test_posts_expected_payload`,
`test_http_error_swallowed_and_logged`, `test_timeout_swallowed`,
`test_no_exception_ever_escapes` (raise inside transport → call returns).

## GATE 7

```bash
make check
python -c "from src.monitoring.logging_setup import setup_logging; import logging, json; \
setup_logging('INFO'); logging.getLogger('x').info('smoke_event', extra={'k': 1})" \
  | python -c "import sys, json; json.loads(sys.stdin.readline()); print('json ok')"
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/monitoring/`
- [ ] All tests pass; `make check` green
- [ ] Progress ticked; committed `Phase 7: monitoring`

## Pitfalls

- The message IS the event name (`"order_filled"`), snake_case, no f-strings —
  variable data goes in `extra` or it is unqueryable.
- Never `await send_telegram_alert(...)` inside a stream callback's critical
  path — dispatch and move on.
