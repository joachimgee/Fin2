# CLAUDE.md — src/execution/

<!-- Chargé automatiquement par Claude Code quand tu travailles dans src/execution/
     Le CLAUDE.md racine est toujours chargé en plus de celui-ci. -->

<role>
Sole module allowed to import alpaca-py and submit orders.
Every other module uses the AbstractBrokerClient abstraction — they don't know Alpaca exists.
</role>

<constraints>
- paper=True by default. paper=False only from YAML config (execution.live_mode: true).
- 1 WebSocket connection per Alpaca endpoint per account — never open a second.
- Fills arrive asynchronously via TradingStream, never synchronously after submit_order().
- Re-sync positions from REST API after every WebSocket reconnection.
</constraints>

<order_pipeline>
Execute in this exact order. Never shortcut.

  1. result = risk_manager.validate(intent)          ← mandatory, raises if rejected
  2. Use result.adjusted_qty (not intent.qty)        ← risk may have reduced it
  3. client.submit_order(request)
  4. log.info("order_submitted", extra={order_id, symbol, side, qty, strategy_id})
  5. return order_id
</order_pipeline>

<websocket_reconnect>
On every reconnection, before accepting any signals:
  positions = await broker.get_positions()
  account   = await broker.get_account()
  tracker.sync_from_api(positions, float(account["equity"]))
  log.info("position_sync_completed", extra={...})
</websocket_reconnect>

<event_handling>
Handle all TradingStream events explicitly. Never a catch-all that silently drops events.

  "fill"           → ExposureTracker.on_fill() + log order_filled
  "partial_fill"   → partial update + log WARNING order_partial_fill
  "canceled"|"expired" → log INFO
  "new"|"pending_new"  → log DEBUG
  "rejected"       → log ERROR + Telegram alert
  _                → log WARNING("unhandled_trade_event", ...)
</event_handling>

<backoff>
Reconnection: exponential backoff starting at 1s, doubling each attempt, capped at 60s.
Never instant reconnect — risks rate-limit ban during Alpaca outages.
</backoff>

<warn_about>
- paper=False set outside YAML config
- submit_order() called without validate() first
- intent.qty used instead of result.adjusted_qty
- Second WebSocket connection opened to same endpoint
- Stream callback contains blocking logic (> 5ms)
- alpaca-trade-api (deprecated) imported anywhere
- Position state not re-synced after WebSocket reconnection
</warn_about>
