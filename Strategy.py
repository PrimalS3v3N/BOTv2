"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""

import numpy as np


class DynamicStopLoss:
    """
    Dynamic Stop Loss Manager for Options Contracts.

    Implements a three-phase stop loss strategy:
    1. INITIAL: Fixed stop loss at (entry_price - stop_loss_pct * entry_price)
       Default 30% below entry if no stop loss specified in signal.

    2. BREAKEVEN: When price rises above breakeven_threshold, move stop to entry price.
       Breakeven threshold = entry_price / (1 - stop_loss_pct)
       Example: Entry $1.00, 30% SL -> threshold = $1.00 / 0.70 = $1.43
       NOTE: Only transitions to breakeven after minimum hold time (default 30 mins).

    3. TRAILING: When price reaches 50% above entry, switch to trailing stop
       at 30% below the highest price since entry.
    """

    # Stop loss modes
    MODE_INITIAL = 'initial'
    MODE_BREAKEVEN = 'breakeven'
    MODE_TRAILING = 'trailing'

    def __init__(self, entry_price, stop_loss_pct=None, trailing_trigger_pct=0.50,
                 trailing_stop_pct=0.30, breakeven_min_minutes=30):
        """
        Initialize dynamic stop loss manager.

        Args:
            entry_price: Contract entry price
            stop_loss_pct: Initial stop loss percentage (default: 0.30 = 30%)
            trailing_trigger_pct: Profit % to trigger trailing mode (default: 0.50 = 50%)
            trailing_stop_pct: Trailing stop % below high (default: 0.30 = 30%)
            breakeven_min_minutes: Minimum minutes held before allowing breakeven (default: 30)
        """
        self.entry_price = entry_price
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else 0.30
        self.trailing_trigger_pct = trailing_trigger_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.breakeven_min_minutes = breakeven_min_minutes

        # Current state
        self.mode = self.MODE_INITIAL
        self.highest_price_since_entry = entry_price
        self.stop_loss_price = self._calculate_initial_stop()

        # Calculate thresholds
        # Breakeven threshold: entry / (1 - stop_loss_pct)
        self.breakeven_threshold = entry_price / (1.0 - self.stop_loss_pct)

        # Trailing trigger: entry * (1 + trailing_trigger_pct)
        self.trailing_trigger = entry_price * (1.0 + self.trailing_trigger_pct)

    def _calculate_initial_stop(self):
        """Calculate initial stop loss price."""
        return self.entry_price * (1.0 - self.stop_loss_pct)

    def update(self, current_price, minutes_held=0):
        """
        Update stop loss based on current price and time held.

        Args:
            current_price: Current contract price
            minutes_held: Minutes since entry (default: 0)

        Returns:
            dict with:
                - stop_loss: Current stop loss price
                - mode: Current stop loss mode
                - triggered: True if stop loss was hit
        """
        # Track highest price since entry (for trailing stop)
        if current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = current_price

        # Check mode transitions (only forward, never backward)
        if self.mode == self.MODE_INITIAL:
            # Check if we should move to breakeven
            # Must meet BOTH conditions: price threshold AND minimum time held
            if current_price >= self.breakeven_threshold and minutes_held >= self.breakeven_min_minutes:
                self.mode = self.MODE_BREAKEVEN
                self.stop_loss_price = self.entry_price  # Move stop to breakeven

        if self.mode in [self.MODE_INITIAL, self.MODE_BREAKEVEN]:
            # Check if we should start trailing
            if current_price >= self.trailing_trigger:
                self.mode = self.MODE_TRAILING

        # Update stop loss based on mode
        if self.mode == self.MODE_TRAILING:
            # Trailing stop: 30% below highest price since entry
            new_stop = self.highest_price_since_entry * (1.0 - self.trailing_stop_pct)
            # Only move stop up, never down
            if new_stop > self.stop_loss_price:
                self.stop_loss_price = new_stop

        # Check if stop loss triggered
        triggered = current_price <= self.stop_loss_price

        return {
            'stop_loss': self.stop_loss_price,
            'mode': self.mode,
            'triggered': triggered,
            'highest_price': self.highest_price_since_entry,
            'breakeven_threshold': self.breakeven_threshold,
            'trailing_trigger': self.trailing_trigger
        }

    def get_state(self):
        """Get current stop loss state."""
        return {
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss_price,
            'mode': self.mode,
            'highest_price': self.highest_price_since_entry,
            'stop_loss_pct': self.stop_loss_pct,
            'trailing_trigger_pct': self.trailing_trigger_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'breakeven_threshold': self.breakeven_threshold,
            'trailing_trigger': self.trailing_trigger,
            'breakeven_min_minutes': self.breakeven_min_minutes
        }


def check_stop_loss(entry_price, current_price, highest_price_since_entry,
                    stop_loss_pct=0.30, trailing_trigger_pct=0.50,
                    trailing_stop_pct=0.30, current_mode='initial',
                    minutes_held=0, breakeven_min_minutes=30):
    """
    Stateless stop loss check function for use in backtesting.

    Args:
        entry_price: Contract entry price
        current_price: Current contract price
        highest_price_since_entry: Highest price since position was opened
        stop_loss_pct: Stop loss percentage (default: 30%)
        trailing_trigger_pct: Profit % to trigger trailing mode (default: 50%)
        trailing_stop_pct: Trailing stop % below high (default: 30%)
        current_mode: Current stop loss mode ('initial', 'breakeven', 'trailing')
        minutes_held: Minutes since entry (default: 0)
        breakeven_min_minutes: Minimum minutes before allowing breakeven (default: 30)

    Returns:
        dict with:
            - stop_loss: Current stop loss price
            - mode: Updated stop loss mode
            - triggered: True if stop loss was hit
    """
    # Calculate thresholds
    initial_stop = entry_price * (1.0 - stop_loss_pct)
    breakeven_threshold = entry_price / (1.0 - stop_loss_pct)
    trailing_trigger = entry_price * (1.0 + trailing_trigger_pct)

    # Determine mode
    mode = current_mode
    # Must meet BOTH conditions: price threshold AND minimum time held
    if mode == 'initial' and current_price >= breakeven_threshold and minutes_held >= breakeven_min_minutes:
        mode = 'breakeven'
    if mode in ['initial', 'breakeven'] and current_price >= trailing_trigger:
        mode = 'trailing'

    # Calculate stop loss based on mode
    if mode == 'initial':
        stop_loss = initial_stop
    elif mode == 'breakeven':
        stop_loss = entry_price
    else:  # trailing
        stop_loss = highest_price_since_entry * (1.0 - trailing_stop_pct)
        # Don't let trailing stop go below breakeven once we've reached that mode
        stop_loss = max(stop_loss, entry_price)

    # Check if triggered
    triggered = current_price <= stop_loss

    return {
        'stop_loss': stop_loss,
        'mode': mode,
        'triggered': triggered,
        'highest_price': highest_price_since_entry,
        'breakeven_threshold': breakeven_threshold,
        'trailing_trigger': trailing_trigger
    }


class TieredProfitExit:
    """
    Tiered Profit Exit Strategy for Options Contracts.

    Implements a dynamic profit target system with tiered contract management:

    Contract Tiers:
    - Tier 1: 1 contract - sell all at exit signal
    - Tier 2: 3 contracts - sell 1 at each profit tier
    - Tier 3: 5+ contracts - sell progressively at profit tiers

    Profit Target Tiers (when stock price > EMA 30):
    - 35% profit: move target to 10% above entry
    - 50% profit: move target to 20% above entry
    - 75% profit: move target to 35% above entry
    - 100% profit: move target to 50% above entry, start trailing
    - 125% profit: trailing = (1 - stop_loss_pct) * max_option_price
    - 200% profit: sell immediately (can be modified later)

    Sell Signals:
    - Option price drops below profit target
    - Option price drops below EMA 30 (when active)
    """

    # Profit tier modes
    MODE_INITIAL = 'initial'           # No profit target set
    MODE_TIER_35 = 'tier_35'           # 35% profit reached, target = 10%
    MODE_TIER_50 = 'tier_50'           # 50% profit reached, target = 20%
    MODE_TIER_75 = 'tier_75'           # 75% profit reached, target = 35%
    MODE_TIER_100 = 'tier_100'         # 100% profit reached, target = 50%, trailing starts
    MODE_TIER_125 = 'tier_125'         # 125% profit reached, advanced trailing
    MODE_TIER_200 = 'tier_200'         # 200% profit reached, sell signal

    # Contract tiers
    TIER_1_CONTRACTS = 1
    TIER_2_CONTRACTS = 3
    TIER_3_CONTRACTS = 5

    def __init__(self, entry_price, contracts=1, stop_loss_pct=0.30):
        """
        Initialize tiered profit exit manager.

        Args:
            entry_price: Contract entry price
            contracts: Number of contracts held (default: 1)
            stop_loss_pct: Stop loss percentage for trailing calculation (default: 0.30)
        """
        self.entry_price = entry_price
        self.contracts = contracts
        self.stop_loss_pct = stop_loss_pct

        # Current state
        self.mode = self.MODE_INITIAL
        self.max_option_price = entry_price
        self.profit_target = None  # No initial profit target
        self.is_trailing = False
        self.is_active = False  # Only active when stock > EMA 30

        # Track contracts sold at each tier
        self.contracts_remaining = contracts
        self.contracts_sold = 0

        # Determine contract tier
        if contracts >= self.TIER_3_CONTRACTS:
            self.contract_tier = 3
        elif contracts >= self.TIER_2_CONTRACTS:
            self.contract_tier = 2
        else:
            self.contract_tier = 1

    def _calculate_profit_pct(self, current_price):
        """Calculate current profit percentage."""
        if self.entry_price <= 0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100

    def _calculate_profit_target(self, target_pct):
        """Calculate profit target price from percentage above entry."""
        return self.entry_price * (1.0 + target_pct)

    def _get_contracts_to_sell(self):
        """
        Determine how many contracts to sell based on tier.

        Returns:
            Number of contracts to sell at current profit tier
        """
        if self.contract_tier == 1:
            return self.contracts_remaining
        elif self.contract_tier == 2:
            # Tier 2: sell 1 contract at each major tier
            return min(1, self.contracts_remaining)
        else:
            # Tier 3 (5+): sell progressively
            # Sell 1-2 contracts at each tier
            return min(2, self.contracts_remaining)

    def update(self, current_option_price, stock_price, ema_30):
        """
        Update profit target based on current prices and EMA condition.

        Args:
            current_option_price: Current option contract price
            stock_price: Current stock price
            ema_30: Current EMA 30 value

        Returns:
            dict with:
                - profit_target: Current profit target price (None if not set)
                - mode: Current profit tier mode
                - triggered: True if sell signal triggered
                - sell_reason: Reason for sell signal (if triggered)
                - contracts_to_sell: Number of contracts to sell
                - max_price: Highest option price seen
                - is_active: Whether profit targeting is active
        """
        # Update max option price
        if current_option_price > self.max_option_price:
            self.max_option_price = current_option_price

        # Calculate current profit percentage
        profit_pct = self._calculate_profit_pct(current_option_price)

        # Check if stock is above EMA 30 (activates profit targeting)
        self.is_active = stock_price > ema_30 if not np.isnan(ema_30) else False

        triggered = False
        sell_reason = None
        contracts_to_sell = 0

        # Only apply profit targeting when stock > EMA 30
        if self.is_active:
            # Mode transitions based on profit percentage
            previous_mode = self.mode

            if profit_pct >= 200:
                self.mode = self.MODE_TIER_200
                self.profit_target = self._calculate_profit_target(0.50)
                self.is_trailing = True
            elif profit_pct >= 125:
                self.mode = self.MODE_TIER_125
                # Trailing = (1 - stop_loss_pct) * max_option_price
                self.profit_target = (1.0 - self.stop_loss_pct) * self.max_option_price
                self.is_trailing = True
            elif profit_pct >= 100:
                self.mode = self.MODE_TIER_100
                self.profit_target = self._calculate_profit_target(0.50)
                self.is_trailing = True
            elif profit_pct >= 75:
                self.mode = self.MODE_TIER_75
                self.profit_target = self._calculate_profit_target(0.35)
            elif profit_pct >= 50:
                self.mode = self.MODE_TIER_50
                self.profit_target = self._calculate_profit_target(0.20)
            elif profit_pct >= 35:
                self.mode = self.MODE_TIER_35
                self.profit_target = self._calculate_profit_target(0.10)

            # If in trailing mode (100%+), update profit target to trail max price
            if self.is_trailing and self.mode in [self.MODE_TIER_100, self.MODE_TIER_125, self.MODE_TIER_200]:
                if self.mode == self.MODE_TIER_125:
                    # Advanced trailing: (1 - stop_loss_pct) * max_price
                    trailing_target = (1.0 - self.stop_loss_pct) * self.max_option_price
                else:
                    # Standard trailing: 50% above entry as floor, trail max
                    floor_target = self._calculate_profit_target(0.50)
                    trailing_target = max(floor_target, (1.0 - self.stop_loss_pct) * self.max_option_price)

                # Only move target up, never down
                if self.profit_target is None or trailing_target > self.profit_target:
                    self.profit_target = trailing_target

            # Check sell triggers
            # Trigger 1: 200% profit - immediate sell
            if self.mode == self.MODE_TIER_200:
                triggered = True
                sell_reason = 'profit_200_pct'
                contracts_to_sell = self.contracts_remaining

            # Trigger 2: Price dropped below profit target
            elif self.profit_target is not None and current_option_price <= self.profit_target:
                triggered = True
                sell_reason = f'profit_target_{self.mode}'
                contracts_to_sell = self._get_contracts_to_sell()

        # Trigger 3: Price dropped below EMA 30 (regardless of stock/EMA relationship)
        # This is a protective sell when option price itself falls below EMA 30 equivalent
        # Note: This would need option EMA, but we use stock EMA as proxy signal
        # If stock drops below EMA 30, it's a bearish signal
        if not self.is_active and self.profit_target is not None:
            # Stock below EMA 30 - bearish, check if we should protect profits
            if current_option_price <= self.profit_target:
                triggered = True
                sell_reason = 'ema_30_bearish'
                contracts_to_sell = self._get_contracts_to_sell()

        return {
            'profit_target': self.profit_target,
            'mode': self.mode,
            'triggered': triggered,
            'sell_reason': sell_reason,
            'contracts_to_sell': contracts_to_sell,
            'max_price': self.max_option_price,
            'is_active': self.is_active,
            'profit_pct': profit_pct,
            'is_trailing': self.is_trailing,
            'contracts_remaining': self.contracts_remaining
        }

    def execute_sell(self, contracts_sold):
        """
        Record contracts sold.

        Args:
            contracts_sold: Number of contracts that were sold
        """
        self.contracts_sold += contracts_sold
        self.contracts_remaining -= contracts_sold

    def get_state(self):
        """Get current profit target state."""
        return {
            'entry_price': self.entry_price,
            'profit_target': self.profit_target,
            'mode': self.mode,
            'max_price': self.max_option_price,
            'is_active': self.is_active,
            'is_trailing': self.is_trailing,
            'contracts': self.contracts,
            'contracts_remaining': self.contracts_remaining,
            'contract_tier': self.contract_tier,
            'stop_loss_pct': self.stop_loss_pct
        }


def check_profit_target(entry_price, current_option_price, max_option_price,
                        stock_price, ema_30, stop_loss_pct=0.30, current_mode='initial'):
    """
    Stateless profit target check function for use in backtesting.

    Args:
        entry_price: Contract entry price
        current_option_price: Current option contract price
        max_option_price: Highest option price seen since entry
        stock_price: Current stock price
        ema_30: Current EMA 30 value
        stop_loss_pct: Stop loss percentage for trailing (default: 0.30)
        current_mode: Current profit tier mode

    Returns:
        dict with:
            - profit_target: Current profit target price
            - mode: Updated profit tier mode
            - triggered: True if sell signal triggered
            - sell_reason: Reason for sell (if triggered)
            - max_price: Updated max option price
            - is_active: Whether profit targeting is active
    """
    # Update max price
    max_price = max(max_option_price, current_option_price)

    # Calculate profit percentage
    profit_pct = ((current_option_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

    # Check if stock > EMA 30
    is_active = stock_price > ema_30 if not np.isnan(ema_30) else False

    # Determine mode and profit target
    mode = current_mode
    profit_target = None
    is_trailing = False

    if is_active:
        if profit_pct >= 200:
            mode = 'tier_200'
            profit_target = entry_price * 1.50
            is_trailing = True
        elif profit_pct >= 125:
            mode = 'tier_125'
            profit_target = (1.0 - stop_loss_pct) * max_price
            is_trailing = True
        elif profit_pct >= 100:
            mode = 'tier_100'
            floor_target = entry_price * 1.50
            trailing_target = (1.0 - stop_loss_pct) * max_price
            profit_target = max(floor_target, trailing_target)
            is_trailing = True
        elif profit_pct >= 75:
            mode = 'tier_75'
            profit_target = entry_price * 1.35
        elif profit_pct >= 50:
            mode = 'tier_50'
            profit_target = entry_price * 1.20
        elif profit_pct >= 35:
            mode = 'tier_35'
            profit_target = entry_price * 1.10

    # Check triggers
    triggered = False
    sell_reason = None

    if mode == 'tier_200':
        triggered = True
        sell_reason = 'profit_200_pct'
    elif profit_target is not None and current_option_price <= profit_target:
        triggered = True
        sell_reason = f'profit_target_{mode}'
    elif not is_active and profit_target is not None and current_option_price <= profit_target:
        triggered = True
        sell_reason = 'ema_30_bearish'

    return {
        'profit_target': profit_target,
        'mode': mode,
        'triggered': triggered,
        'sell_reason': sell_reason,
        'max_price': max_price,
        'is_active': is_active,
        'profit_pct': profit_pct,
        'is_trailing': is_trailing
    }


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['DynamicStopLoss', 'check_stop_loss', 'TieredProfitExit', 'check_profit_target']
