"""First concrete strategy: LightGBM momentum, gated by HMM regime.

TODO(Phase 4): implement against AbstractStrategy:
  - universe / warmup thresholds from YAML config only.
  - on_bar(): append bar -> if not is_ready return None ->
    compute_features (all shifted) -> regime gate (skip hostile regimes) ->
    signal = self._signal_generator.generate(features) ->
    build OrderIntent(signal_strength=signal) or None below threshold.
  - on_trade_update(): maintain _in_position from fill events only.
  - reset(): clear bar buffer and position state.
"""

from __future__ import annotations
