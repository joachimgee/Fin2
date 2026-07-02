# CLAUDE.md — src/risk/

<!-- Chargé automatiquement par Claude Code quand tu travailles dans src/risk/ -->

<role>
Mandatory gatekeeper. Every order passes validate() or it doesn't exist.
When in doubt: reject the order. A missed trade is recoverable. A blown account is not.
validate() is synchronous, < 1ms, zero network calls, zero DB reads.
</role>

<validate_pipeline>
Execute in this exact order. First failure = immediate rejection, stop evaluating.

  1. Circuit breaker active? → reject if trading_halted=True
  2. Kelly sizing           → compute adjusted_qty from signal_strength
  3. Risk-per-trade cap 2%  → reduce qty if needed (don't reject)
  4. Position cap 10%       → reduce qty if needed (don't reject)
  5. Total exposure cap 80% → reject if breach even after reduction
  6. Sector cap 25%         → reject if sector would breach
  7. Correlation guard 0.80 → reject if > 0.80 correlated with any existing position
</validate_pipeline>

<kelly>
Default: quarter-Kelly (fraction=0.25). Hard maximum: half-Kelly (fraction=0.50).
Full Kelly leads to catastrophic drawdowns in practice due to estimation error.
Raise InvalidKellyFractionError if fraction > 0.50.
Return 0.0 (do not trade) if edge ≤ 0 or avg_loss ≤ 0.
</kelly>

<circuit_breakers>
Three independent breakers. Any one halts all new orders.
  daily_loss_pct:       P&L < −3% of start-of-day equity
  max_drawdown_pct:     equity < 85% of peak equity
  consecutive_losses:   5 losing trades in a row

When tripped:
  ✓ Set _trading_halted = True
  ✓ Log CRITICAL circuit_breaker_tripped with reason + current values
  ✓ Send Telegram alert
  ✗ Do NOT auto-cancel open orders (may be protecting positions)
  ✗ Do NOT auto-reset (requires manual reset_circuit_breaker() call)
</circuit_breakers>

<exposure_tracker>
Source of truth for validate(). Updated from fill events, not by polling REST API.
Re-sync from REST API only:
  - At system startup
  - After every WebSocket reconnection
Thread-safe: all public methods use threading.Lock().
</exposure_tracker>

<warn_about>
- validate() contains any async operation, network call, or DB read
- Circuit breaker has auto-reset logic anywhere
- Open orders auto-canceled when circuit breaker trips
- MAX_RISK_PER_TRADE_PCT set above 0.05 (5%)
- Kelly fraction set above 0.50 (half-Kelly)
- Exposure tracked by polling REST API instead of fill events
- Any code path allows submit_order() without validate() first
</warn_about>
