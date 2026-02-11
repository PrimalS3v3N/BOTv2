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

import Config

# Export for use by other modules
__all__ = ['TakeProfitMilestones', 'MilestoneTracker']


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
