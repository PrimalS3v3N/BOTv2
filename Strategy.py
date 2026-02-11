"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import numpy as np
import Config

# Export for use by other modules
__all__ = ['TakeProfitMilestones', 'MilestoneTracker', 'MomentumPeak', 'MomentumPeakDetector']


# =============================================================================
# INTERNAL - Take Profit Milestone Strategy
# =============================================================================

class TakeProfitMilestones:
    """
    Milestone-based take profit strategy with ratcheting trailing stops.

    When a position's gain reaches a milestone percentage, a trailing stop
    is set at the corresponding trailing level. Milestones only ratchet
    upward — once reached, the trailing stop never drops below that level.

    If the price falls to or below the trailing stop, the position is exited.

    Config: BACKTEST_CONFIG['take_profit_milestones']
        {
            'enabled': True,
            'milestones': [
                {'gain_pct': 10, 'trailing_pct': 0},   # Breakeven
                {'gain_pct': 20, 'trailing_pct': 5},   # 5% above entry
                ...
            ]
        }
    """

    def __init__(self, config=None):
        tp_config = config or Config.get_setting('backtest', 'take_profit_milestones', {})
        self.enabled = tp_config.get('enabled', False)
        self.milestones = sorted(
            tp_config.get('milestones', []),
            key=lambda m: m['gain_pct']
        )

    def create_tracker(self, entry_price):
        """Create a new MilestoneTracker for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return MilestoneTracker(self.milestones, entry_price)


class MilestoneTracker:
    """
    Tracks milestone state for a single position.

    Maintains the highest milestone reached and corresponding trailing
    exit price. The trailing price can only ratchet upward.
    """

    def __init__(self, milestones, entry_price):
        self.milestones = milestones
        self.entry_price = entry_price
        self.current_milestone_index = -1
        self.trailing_exit_price = 0.0

    def update(self, current_price):
        """
        Update milestone state and check for trailing stop exit.

        Args:
            current_price: Current option price

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        if not self.milestones or self.entry_price <= 0:
            return False, None

        gain_pct = ((current_price - self.entry_price) / self.entry_price) * 100

        # Ratchet up through milestones (can skip multiple in one update)
        for i, milestone in enumerate(self.milestones):
            if i > self.current_milestone_index and gain_pct >= milestone['gain_pct']:
                self.current_milestone_index = i
                self.trailing_exit_price = self.entry_price * (1 + milestone['trailing_pct'] / 100)

        # Check if price has fallen to or below trailing stop
        if self.current_milestone_index >= 0 and current_price <= self.trailing_exit_price:
            milestone_pct = self.milestones[self.current_milestone_index]['gain_pct']
            return True, f'Take Profit - {milestone_pct}%'

        return False, None

    @property
    def current_milestone_pct(self):
        """Current highest milestone percentage, or None if none reached."""
        if self.current_milestone_index >= 0:
            return self.milestones[self.current_milestone_index]['gain_pct']
        return None

    @property
    def current_trailing_pct(self):
        """Current trailing stop percentage above entry, or None if none reached."""
        if self.current_milestone_index >= 0:
            return self.milestones[self.current_milestone_index]['trailing_pct']
        return None


# =============================================================================
# INTERNAL - Momentum Peak Detection Strategy
# =============================================================================

class MomentumPeak:
    """
    Factory for creating MomentumPeakDetector instances.

    Detects momentum exhaustion peaks using RSI overbought reversal,
    EWO decline, and RSI-below-average confirmation. Triggers 1-2 bars
    after a peak, before the bulk of the reversal.

    Config: BACKTEST_CONFIG['momentum_peak']
        {
            'enabled': True,
            'min_profit_pct': 15,          # Minimum option profit %
            'rsi_overbought': 80,          # RSI overbought threshold
            'rsi_lookback': 5,             # Bars to check for recent overbought
            'rsi_drop_threshold': 10,      # Min RSI drop from peak
            'ewo_declining_bars': 1,       # Consecutive declining EWO bars
            'require_rsi_below_avg': True, # RSI < RSI_10min_avg
        }
    """

    def __init__(self, config=None):
        mp_config = config or Config.get_setting('backtest', 'momentum_peak', {})
        self.enabled = mp_config.get('enabled', False)
        self.config = mp_config

    def create_detector(self):
        """Create a new MomentumPeakDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return MomentumPeakDetector(self.config)


class MomentumPeakDetector:
    """
    Tracks momentum state for a single position and detects peaks.

    Conditions (ALL must be met to trigger exit):
    1. Option profit >= min_profit_pct
    2. RSI reached overbought level within lookback window
    3. RSI dropped by >= rsi_drop_threshold from its recent peak
    4. EWO declining for N consecutive bars
    5. RSI crossed below RSI_10min_avg (optional)
    """

    def __init__(self, config):
        self.min_profit_pct = config.get('min_profit_pct', 15)
        self.rsi_overbought = config.get('rsi_overbought', 80)
        self.rsi_lookback = config.get('rsi_lookback', 5)
        self.rsi_drop_threshold = config.get('rsi_drop_threshold', 10)
        self.ewo_declining_bars = config.get('ewo_declining_bars', 1)
        self.require_rsi_below_avg = config.get('require_rsi_below_avg', True)

        self.rsi_history = []
        self.ewo_history = []

    def update(self, pnl_pct, rsi, rsi_avg, ewo):
        """
        Check for momentum peak exit signal.

        Args:
            pnl_pct: Current P&L percentage
            rsi: Current RSI value
            rsi_avg: Current RSI 10-min average
            ewo: Current EWO value

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.rsi_history.append(rsi)
        self.ewo_history.append(ewo)

        # Need at least 2 bars of history
        if len(self.rsi_history) < 2 or len(self.ewo_history) < 2:
            return False, None

        # Skip if critical values are NaN
        if np.isnan(pnl_pct) or np.isnan(rsi) or np.isnan(ewo):
            return False, None

        # 1. Minimum profit threshold
        if pnl_pct < self.min_profit_pct:
            return False, None

        # 2. RSI was overbought recently (within lookback window)
        lookback = min(self.rsi_lookback, len(self.rsi_history))
        recent_rsi = self.rsi_history[-lookback:]
        valid_recent = [r for r in recent_rsi if not np.isnan(r)]
        if not valid_recent:
            return False, None
        rsi_peak = max(valid_recent)

        if rsi_peak < self.rsi_overbought:
            return False, None

        # 3. RSI has dropped significantly from recent peak
        rsi_drop = rsi_peak - rsi
        if rsi_drop < self.rsi_drop_threshold:
            return False, None

        # 4. EWO is declining for N consecutive bars
        ewo_check_len = self.ewo_declining_bars + 1
        if len(self.ewo_history) < ewo_check_len:
            return False, None

        recent_ewo = self.ewo_history[-ewo_check_len:]
        for j in range(len(recent_ewo) - 1):
            if np.isnan(recent_ewo[j]) or np.isnan(recent_ewo[j + 1]):
                return False, None
            if recent_ewo[j + 1] >= recent_ewo[j]:
                return False, None

        # 5. RSI below its 10-min average (optional confirmation)
        if self.require_rsi_below_avg and not np.isnan(rsi_avg):
            if rsi >= rsi_avg:
                return False, None

        return True, 'Momentum Peak'
