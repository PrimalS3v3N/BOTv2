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

    # Profit tier modes (ordered from lowest to highest)
    MODE_INITIAL = 'initial'           # No profit target set
    MODE_TIER_35 = 'tier_35'           # 35% profit reached, target = 10%
    MODE_TIER_50 = 'tier_50'           # 50% profit reached, target = 20%
    MODE_TIER_75 = 'tier_75'           # 75% profit reached, target = 35%
    MODE_TIER_100 = 'tier_100'         # 100% profit reached, target = 50%, trailing starts
    MODE_TIER_125 = 'tier_125'         # 125% profit reached, advanced trailing
    MODE_TIER_200 = 'tier_200'         # 200% profit reached, sell signal

    # Mode hierarchy (higher index = higher tier, cannot go backwards)
    MODE_HIERARCHY = [MODE_INITIAL, MODE_TIER_35, MODE_TIER_50, MODE_TIER_75,
                      MODE_TIER_100, MODE_TIER_125, MODE_TIER_200]

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
            # Determine potential new mode based on current profit percentage
            potential_mode = self.MODE_INITIAL
            if profit_pct >= 200:
                potential_mode = self.MODE_TIER_200
            elif profit_pct >= 125:
                potential_mode = self.MODE_TIER_125
            elif profit_pct >= 100:
                potential_mode = self.MODE_TIER_100
            elif profit_pct >= 75:
                potential_mode = self.MODE_TIER_75
            elif profit_pct >= 50:
                potential_mode = self.MODE_TIER_50
            elif profit_pct >= 35:
                potential_mode = self.MODE_TIER_35

            # Only upgrade mode, never downgrade (tier cannot go backwards)
            current_tier_idx = self.MODE_HIERARCHY.index(self.mode)
            potential_tier_idx = self.MODE_HIERARCHY.index(potential_mode)

            if potential_tier_idx > current_tier_idx:
                self.mode = potential_mode

            # Update profit target based on current mode (use highest tier reached)
            if self.mode == self.MODE_TIER_200:
                self.profit_target = self._calculate_profit_target(0.50)
                self.is_trailing = True
            elif self.mode == self.MODE_TIER_125:
                # Trailing = (1 - stop_loss_pct) * max_option_price
                self.profit_target = (1.0 - self.stop_loss_pct) * self.max_option_price
                self.is_trailing = True
            elif self.mode == self.MODE_TIER_100:
                self.profit_target = self._calculate_profit_target(0.50)
                self.is_trailing = True
            elif self.mode == self.MODE_TIER_75:
                self.profit_target = self._calculate_profit_target(0.35)
            elif self.mode == self.MODE_TIER_50:
                self.profit_target = self._calculate_profit_target(0.20)
            elif self.mode == self.MODE_TIER_35:
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

        # Trigger 3: Stock dropped below EMA 30 - immediate bearish exit
        # When stock drops below EMA 30, it's a bearish signal - exit immediately
        # to protect profits rather than waiting for option price to fall further
        if not self.is_active and self.profit_target is not None:
            # Stock below EMA 30 - bearish, exit immediately to protect profits
            triggered = True
            sell_reason = 'X_EMA'
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

    # Only upgrade mode when stock is above EMA (is_active)
    if is_active:
        if profit_pct >= 200:
            mode = 'tier_200'
        elif profit_pct >= 125:
            mode = 'tier_125'
        elif profit_pct >= 100:
            mode = 'tier_100'
        elif profit_pct >= 75:
            mode = 'tier_75'
        elif profit_pct >= 50:
            mode = 'tier_50'
        elif profit_pct >= 35:
            mode = 'tier_35'

    # Calculate profit target based on mode (regardless of is_active)
    # This ensures we have a profit target for EMA exit checks
    if mode == 'tier_200':
        profit_target = entry_price * 1.50
        is_trailing = True
    elif mode == 'tier_125':
        profit_target = (1.0 - stop_loss_pct) * max_price
        is_trailing = True
    elif mode == 'tier_100':
        floor_target = entry_price * 1.50
        trailing_target = (1.0 - stop_loss_pct) * max_price
        profit_target = max(floor_target, trailing_target)
        is_trailing = True
    elif mode == 'tier_75':
        profit_target = entry_price * 1.35
    elif mode == 'tier_50':
        profit_target = entry_price * 1.20
    elif mode == 'tier_35':
        profit_target = entry_price * 1.10

    # Check triggers
    triggered = False
    sell_reason = None

    if mode == 'tier_200':
        triggered = True
        sell_reason = 'profit_200_pct'
    elif is_active and profit_target is not None and current_option_price <= profit_target:
        # Active: price fell below profit target while stock above EMA
        triggered = True
        sell_reason = f'profit_target_{mode}'
    elif not is_active and profit_target is not None:
        # Bearish: stock dropped below EMA - exit immediately to protect profits
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


class TestPeakExit:
    """
    TEST Exit Strategy - Peak Detection for Options Contracts.

    Implements momentum-based peak detection using EWO (Elliott Wave Oscillator) signals:

    Detection Logic:
    1. EWO Overbought: EWO fast exceeds threshold (default: 0.5)
    2. EWO Extended: EWO spread (fast - slow) exceeds threshold (default: 0.1)
    3. EWO Velocity Reversal: EWO fast starts declining (negative velocity)
    4. Confirmation: Multiple consecutive declining EWO readings

    The strategy aims to identify price peaks in real-time by detecting
    when momentum is reversing from overbought conditions.

    Modes:
    - WATCHING: Monitoring for peak conditions
    - PEAK_DETECTED: Initial peak signal detected
    - CONFIRMING: Waiting for confirmation
    - TRIGGERED: Exit signal confirmed

    Design Philosophy:
    - Only uses data available at the current moment (no lookahead)
    - Requires multiple confirmations to reduce false signals
    - Only triggers when position is profitable (minimum profit threshold)
    """

    # Exit modes (cannot go backward)
    MODE_WATCHING = 'watching'
    MODE_PEAK_DETECTED = 'peak_detected'
    MODE_CONFIRMING = 'confirming'
    MODE_TRIGGERED = 'triggered'

    MODE_HIERARCHY = [MODE_WATCHING, MODE_PEAK_DETECTED, MODE_CONFIRMING, MODE_TRIGGERED]

    def __init__(self, entry_price,
                 ewo_overbought_threshold=0.5,
                 ewo_spread_threshold=0.1,
                 min_profit_pct=0.35,
                 confirmation_bars=2,
                 velocity_lookback=3):
        """
        Initialize TEST peak detection exit manager.

        Args:
            entry_price: Contract entry price
            ewo_overbought_threshold: EWO fast value considered overbought (default: 0.5)
            ewo_spread_threshold: Minimum EWO spread (fast - slow) for extended condition (default: 0.1)
            min_profit_pct: Minimum profit % before considering exit (default: 0.35 = 35%)
            confirmation_bars: Number of declining bars needed to confirm (default: 2)
            velocity_lookback: Number of bars for velocity calculation (default: 3)
        """
        self.entry_price = entry_price
        self.ewo_overbought_threshold = ewo_overbought_threshold
        self.ewo_spread_threshold = ewo_spread_threshold
        self.min_profit_pct = min_profit_pct
        self.confirmation_bars = confirmation_bars
        self.velocity_lookback = velocity_lookback

        # Current state
        self.mode = self.MODE_WATCHING
        self.max_option_price = entry_price
        self.max_ewo_fast = 0.0

        # EWO history for velocity calculation
        self.ewo_history = []
        self.declining_count = 0
        self.peak_ewo_value = 0.0
        self.peak_price = entry_price

    def _calculate_profit_pct(self, current_price):
        """Calculate current profit percentage."""
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def _calculate_ewo_velocity(self):
        """
        Calculate EWO velocity (rate of change).

        Returns:
            float: Average change in EWO over lookback period, or None if insufficient data
        """
        if len(self.ewo_history) < 2:
            return None

        lookback = min(self.velocity_lookback, len(self.ewo_history) - 1)
        recent = self.ewo_history[-lookback:]

        # Calculate average velocity
        velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return sum(velocities) / len(velocities) if velocities else None

    def _is_overbought(self, ewo_fast, ewo_slow):
        """Check if EWO indicates overbought conditions."""
        ewo_spread = ewo_fast - ewo_slow

        return (ewo_fast >= self.ewo_overbought_threshold and
                ewo_spread >= self.ewo_spread_threshold)

    def update(self, current_option_price, ewo_fast, ewo_slow):
        """
        Update peak detection based on current prices and EWO values.

        Args:
            current_option_price: Current option contract price
            ewo_fast: Current EWO fast value
            ewo_slow: Current EWO slow value (EWO average)

        Returns:
            dict with:
                - triggered: True if exit signal triggered
                - sell_reason: Reason for exit (if triggered)
                - mode: Current detection mode
                - max_price: Highest option price seen
                - ewo_velocity: Current EWO velocity
                - ewo_spread: Current EWO spread
                - declining_count: Number of consecutive declining bars
        """
        # Update max prices
        if current_option_price > self.max_option_price:
            self.max_option_price = current_option_price

        if ewo_fast > self.max_ewo_fast:
            self.max_ewo_fast = ewo_fast

        # Add to EWO history
        self.ewo_history.append(ewo_fast)
        if len(self.ewo_history) > 20:  # Keep last 20 values
            self.ewo_history.pop(0)

        # Calculate metrics
        profit_pct = self._calculate_profit_pct(current_option_price)
        ewo_spread = ewo_fast - ewo_slow
        ewo_velocity = self._calculate_ewo_velocity()

        triggered = False
        sell_reason = None

        # Check if EWO is declining
        if len(self.ewo_history) >= 2:
            if self.ewo_history[-1] < self.ewo_history[-2]:
                self.declining_count += 1
            else:
                self.declining_count = 0

        # State machine for peak detection
        if self.mode == self.MODE_WATCHING:
            # Check for initial peak conditions
            if (profit_pct >= self.min_profit_pct and
                    self._is_overbought(ewo_fast, ewo_slow) and
                    ewo_velocity is not None and ewo_velocity < 0):
                # Peak detected - EWO overbought and starting to decline
                self.mode = self.MODE_PEAK_DETECTED
                self.peak_ewo_value = self.max_ewo_fast
                self.peak_price = self.max_option_price

        elif self.mode == self.MODE_PEAK_DETECTED:
            # Wait for confirmation
            if self.declining_count >= self.confirmation_bars:
                self.mode = self.MODE_CONFIRMING

            # Reset if EWO makes new high (not a real peak)
            if ewo_fast > self.peak_ewo_value * 1.02:  # 2% buffer
                self.mode = self.MODE_WATCHING
                self.declining_count = 0

        elif self.mode == self.MODE_CONFIRMING:
            # Confirmed peak - check for exit signal
            # Exit when price starts following EWO down
            if current_option_price < self.peak_price * 0.995:  # 0.5% below peak
                self.mode = self.MODE_TRIGGERED
                triggered = True
                sell_reason = 'test_peak_momentum_reversal'

            # Reset if EWO makes new high
            if ewo_fast > self.peak_ewo_value * 1.02:
                self.mode = self.MODE_WATCHING
                self.declining_count = 0

        elif self.mode == self.MODE_TRIGGERED:
            # Already triggered
            triggered = True
            sell_reason = 'test_peak_momentum_reversal'

        return {
            'triggered': triggered,
            'sell_reason': sell_reason,
            'mode': self.mode,
            'max_price': self.max_option_price,
            'ewo_velocity': ewo_velocity,
            'ewo_spread': ewo_spread,
            'declining_count': self.declining_count,
            'peak_ewo_value': self.peak_ewo_value,
            'peak_price': self.peak_price,
            'profit_pct': profit_pct,
            'is_overbought': self._is_overbought(ewo_fast, ewo_slow)
        }

    def get_state(self):
        """Get current peak detection state."""
        return {
            'entry_price': self.entry_price,
            'mode': self.mode,
            'max_option_price': self.max_option_price,
            'max_ewo_fast': self.max_ewo_fast,
            'peak_ewo_value': self.peak_ewo_value,
            'peak_price': self.peak_price,
            'declining_count': self.declining_count,
            'ewo_overbought_threshold': self.ewo_overbought_threshold,
            'ewo_spread_threshold': self.ewo_spread_threshold,
            'min_profit_pct': self.min_profit_pct,
            'confirmation_bars': self.confirmation_bars
        }


def check_test_peak_exit(entry_price, current_option_price, ewo_fast, ewo_slow,
                         ewo_history, max_option_price, max_ewo_fast,
                         current_mode='watching', declining_count=0,
                         peak_ewo_value=0.0, peak_price=None,
                         ewo_overbought_threshold=0.5, ewo_spread_threshold=0.1,
                         min_profit_pct=0.35, confirmation_bars=2,
                         velocity_lookback=3):
    """
    Stateless TEST peak exit check function for use in backtesting.

    Args:
        entry_price: Contract entry price
        current_option_price: Current option contract price
        ewo_fast: Current EWO fast value
        ewo_slow: Current EWO slow value
        ewo_history: List of recent EWO fast values (for velocity calculation)
        max_option_price: Highest option price seen since entry
        max_ewo_fast: Highest EWO fast value seen
        current_mode: Current detection mode
        declining_count: Current count of declining EWO bars
        peak_ewo_value: EWO value at detected peak
        peak_price: Option price at detected peak
        ewo_overbought_threshold: EWO value considered overbought
        ewo_spread_threshold: Minimum EWO spread for extended condition
        min_profit_pct: Minimum profit before considering exit
        confirmation_bars: Number of declining bars for confirmation
        velocity_lookback: Number of bars for velocity calculation

    Returns:
        dict with:
            - triggered: True if exit signal triggered
            - sell_reason: Reason for exit (if triggered)
            - mode: Updated detection mode
            - max_price: Updated max option price
            - max_ewo_fast: Updated max EWO fast
            - ewo_velocity: Current EWO velocity
            - ewo_spread: Current EWO spread
            - declining_count: Updated declining count
            - peak_ewo_value: Updated peak EWO value
            - peak_price: Updated peak price
    """
    # Update max values
    max_price = max(max_option_price, current_option_price)
    max_ewo = max(max_ewo_fast, ewo_fast)
    peak_p = peak_price if peak_price is not None else entry_price

    # Calculate metrics
    profit_pct = ((current_option_price - entry_price) / entry_price) if entry_price > 0 else 0
    ewo_spread = ewo_fast - ewo_slow

    # Calculate velocity from history
    ewo_velocity = None
    if len(ewo_history) >= 2:
        lookback = min(velocity_lookback, len(ewo_history) - 1)
        recent = ewo_history[-lookback:]
        velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        ewo_velocity = sum(velocities) / len(velocities) if velocities else None

    # Check if declining
    new_declining_count = declining_count
    if len(ewo_history) >= 2:
        if ewo_history[-1] < ewo_history[-2]:
            new_declining_count += 1
        else:
            new_declining_count = 0

    # Check overbought condition
    is_overbought = (ewo_fast >= ewo_overbought_threshold and
                     ewo_spread >= ewo_spread_threshold)

    triggered = False
    sell_reason = None
    mode = current_mode
    new_peak_ewo = peak_ewo_value
    new_peak_price = peak_p

    # State machine
    if mode == 'watching':
        if (profit_pct >= min_profit_pct and is_overbought and
                ewo_velocity is not None and ewo_velocity < 0):
            mode = 'peak_detected'
            new_peak_ewo = max_ewo
            new_peak_price = max_price

    elif mode == 'peak_detected':
        if new_declining_count >= confirmation_bars:
            mode = 'confirming'
        if ewo_fast > new_peak_ewo * 1.02:
            mode = 'watching'
            new_declining_count = 0

    elif mode == 'confirming':
        if current_option_price < new_peak_price * 0.995:
            mode = 'triggered'
            triggered = True
            sell_reason = 'test_peak_momentum_reversal'
        if ewo_fast > new_peak_ewo * 1.02:
            mode = 'watching'
            new_declining_count = 0

    elif mode == 'triggered':
        triggered = True
        sell_reason = 'test_peak_momentum_reversal'

    return {
        'triggered': triggered,
        'sell_reason': sell_reason,
        'mode': mode,
        'max_price': max_price,
        'max_ewo_fast': max_ewo,
        'ewo_velocity': ewo_velocity,
        'ewo_spread': ewo_spread,
        'declining_count': new_declining_count,
        'peak_ewo_value': new_peak_ewo,
        'peak_price': new_peak_price,
        'profit_pct': profit_pct,
        'is_overbought': is_overbought
    }


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['DynamicStopLoss', 'check_stop_loss', 'TieredProfitExit', 'check_profit_target',
           'TestPeakExit', 'check_test_peak_exit']
