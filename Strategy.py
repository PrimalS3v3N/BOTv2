"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""

import numpy as np


class StopLoss:
    """
    Stop Loss Manager for Options Contracts.

    Implements a three-phase stop loss strategy:
    1. INITIAL: Fixed stop loss at (entry_price - stop_loss_pct * entry_price)
       Default 30% below entry if no stop loss specified in signal.

    2. BREAKEVEN: When price rises above breakeven_threshold, move stop to entry price.
       Breakeven threshold = entry_price / (1 - stop_loss_pct)
       Example: Entry $1.00, 30% SL -> threshold = $1.00 / 0.70 = $1.43
       NOTE: Only transitions to breakeven after minimum hold time (default 30 mins).

    3. TRAILING: When price reaches 50% above entry, switch to trailing stop
       at 30% below the highest price since entry.

    Reversal detection is direction-aware:
    - CALL: reversal when true_price < VWAP OR true_price < EMAVWAP (stock dropping = bad for calls)
    - PUT:  reversal when true_price > VWAP OR true_price > EMAVWAP (stock rising = bad for puts)

    Downtrend detection is direction-aware:
    - CALL: downtrend when true_price < vwap_ema_avg AND ema < vwap_ema_avg
    - PUT:  downtrend when true_price > vwap_ema_avg AND ema > vwap_ema_avg
    """

    # Stop loss modes
    MODE_INITIAL = 'initial'
    MODE_BREAKEVEN = 'breakeven'
    MODE_TRAILING = 'trailing'

    def __init__(self, entry_price, stop_loss_pct=None, trailing_trigger_pct=0.50,
                 trailing_stop_pct=0.30, breakeven_min_minutes=30, option_type='CALL'):
        """
        Initialize stop loss manager.

        Args:
            entry_price: Contract entry price
            stop_loss_pct: Initial stop loss percentage (default: 0.30 = 30%)
            trailing_trigger_pct: Profit % to trigger trailing mode (default: 0.50 = 50%)
            trailing_stop_pct: Trailing stop % below high (default: 0.30 = 30%)
            breakeven_min_minutes: Minimum minutes held before allowing breakeven (default: 30)
            option_type: 'CALL' or 'PUT' - determines reversal detection direction
        """
        self.entry_price = entry_price
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else 0.30
        self.is_put = option_type.upper() in ('PUT', 'PUTS', 'P')
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

    def update(self, current_price, minutes_held=0, true_price=None, vwap=None, ema=None,
               emavwap=None, vwap_ema_avg=None):
        """
        Update stop loss based on current price and time held.

        Args:
            current_price: Current contract price
            minutes_held: Minutes since entry (default: 0)
            true_price: True Price of the stock (avg of high, low, close)
            vwap: Current VWAP value
            ema: Current EMA value
            emavwap: Current EMAVWAP value ((EMA + VWAP) / 2)
            vwap_ema_avg: Current VWAP/EMA average value

        Returns:
            dict with:
                - stop_loss: Current stop loss price
                - mode: Current stop loss mode
                - triggered: True if stop loss was hit
                - reversal: True if True Price crossed VWAP or EMAVWAP against the position
                            (CALL: true_price < VWAP or true_price < EMAVWAP,
                             PUT: true_price > VWAP or true_price > EMAVWAP)
                - bearish_signal: True if EMA crossed VWAP against the position
                            (CALL: EMA < VWAP, PUT: EMA > VWAP)
                - downtrend: True if True Price and EMA are both below vwap_ema_avg
                            (CALL: both below, PUT: both above)
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

        # Reversal detection: True Price crosses VWAP or EMAVWAP against the position
        # CALL: true_price < VWAP or true_price < EMAVWAP (stock dropping = bad for calls)
        # PUT:  true_price > VWAP or true_price > EMAVWAP (stock rising = bad for puts)
        reversal = False
        if true_price is not None and vwap is not None:
            if not np.isnan(true_price) and not np.isnan(vwap):
                if self.is_put:
                    reversal = true_price > vwap
                else:
                    reversal = true_price < vwap
        if not reversal and true_price is not None and emavwap is not None:
            if not np.isnan(true_price) and not np.isnan(emavwap):
                if self.is_put:
                    reversal = true_price > emavwap
                else:
                    reversal = true_price < emavwap

        # Adverse signal: EMA crosses VWAP against the position
        # CALL: EMA < VWAP (bearish for calls)
        # PUT:  EMA > VWAP (bullish for stock = bearish for puts)
        bearish_signal = False
        if ema is not None and vwap is not None:
            if not np.isnan(ema) and not np.isnan(vwap):
                if self.is_put:
                    bearish_signal = ema > vwap
                else:
                    bearish_signal = ema < vwap

        # Downtrend detection: True Price AND EMA both below vwap_ema_avg
        # CALL: both below vwap_ema_avg (downtrend = bad for calls)
        # PUT:  both above vwap_ema_avg (uptrend = bad for puts)
        downtrend = False
        if true_price is not None and ema is not None and vwap_ema_avg is not None:
            if not np.isnan(true_price) and not np.isnan(ema) and not np.isnan(vwap_ema_avg):
                if self.is_put:
                    downtrend = true_price > vwap_ema_avg and ema > vwap_ema_avg
                else:
                    downtrend = true_price < vwap_ema_avg and ema < vwap_ema_avg

        return {
            'stop_loss': self.stop_loss_price,
            'mode': self.mode,
            'triggered': triggered,
            'highest_price': self.highest_price_since_entry,
            'breakeven_threshold': self.breakeven_threshold,
            'trailing_trigger': self.trailing_trigger,
            'reversal': reversal,
            'bearish_signal': bearish_signal,
            'downtrend': downtrend
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


def CheckDueDiligence(DD_rsi, DD_option_type, DD_rsi_overbought=85, DD_rsi_oversold=15):
    """
    Due Diligence check - validates indicators before entry.

    Called after signal is developed to check historicals and indicators
    for reasons to buy or not buy.

    Args:
        DD_rsi: RSI value at signal entry time
        DD_option_type: Option type ('CALL', 'CALLS', 'C', 'PUT', 'PUTS', 'P')
        DD_rsi_overbought: RSI threshold for overbought (default: 85)
        DD_rsi_oversold: RSI threshold for oversold (default: 15)

    Returns:
        dict with:
            - DD_passed: True if all checks pass, False if trade should be skipped
            - DD_reason: Exit reason string if rejected (None if passed)
            - DD_details: Description of why the check failed
    """
    DD_passed = True
    DD_reason = None
    DD_details = None

    # RSI Check
    if not np.isnan(DD_rsi):
        # CALL: If RSI is above overbought threshold, don't buy
        if DD_option_type in ('CALL', 'CALLS', 'C'):
            if DD_rsi > DD_rsi_overbought:
                DD_passed = False
                DD_reason = 'OverBought'
                DD_details = f"RSI {DD_rsi:.1f} > {DD_rsi_overbought} (overbought)"

        # PUT: If RSI is below oversold threshold, don't buy
        elif DD_option_type in ('PUT', 'PUTS', 'P'):
            if DD_rsi < DD_rsi_oversold:
                DD_passed = False
                DD_reason = 'OverSold'
                DD_details = f"RSI {DD_rsi:.1f} < {DD_rsi_oversold} (oversold)"

    return {
        'DD_passed': DD_passed,
        'DD_reason': DD_reason,
        'DD_details': DD_details,
    }


def _calc_fulfillment(value, zero_ref, target, overflow_cap):
    """
    Calculate fulfillment percentage for a single indicator.

    Linear scale from 0% (at zero_ref) to 100% (at target), continuing
    past 100% as overflow credit, capped at overflow_cap.

    The formula is direction-agnostic: it works whether target > zero_ref
    (indicator needs to rise, e.g. PUT RSI) or target < zero_ref
    (indicator needs to fall, e.g. CALL RSI).

    Args:
        value: Current indicator value
        zero_ref: Value where fulfillment = 0% (starting point, no progress)
        target: Value where fulfillment = 100% (condition fully met)
        overflow_cap: Maximum fulfillment % (e.g. 130)

    Returns:
        Fulfillment percentage (0 to overflow_cap), clamped at floor of 0.
    """
    target_distance = target - zero_ref
    if target_distance == 0:
        return overflow_cap if value == target else 0.0

    progress = value - zero_ref
    fulfillment = (progress / target_distance) * 100.0

    return max(0.0, min(fulfillment, overflow_cap))


def score_reentry_confidence(rsi, ewo, ewo_avg, minutes_elapsed, scoring_config, option_type):
    """
    Fulfillment-based adaptive reentry scoring.

    Each indicator earns a fulfillment percentage:
    - 0%   = no progress from starting reference
    - 100% = target condition fully met
    - >100% = overflow credit (compensates for lagging indicators)

    Weighted average of all fulfillments must reach the confidence threshold
    (default 100%) AND each individual component must meet the safety floor
    (default 20%) to trigger entry.

    Args:
        rsi: Raw RSI value (instantaneous, not averaged)
        ewo: Raw EWO value (instantaneous)
        ewo_avg: EWO 15-minute average
        minutes_elapsed: Minutes since DD rejection
        scoring_config: Config dict from due_diligence.reentry_scoring
        option_type: 'CALL', 'CALLS', 'C', 'PUT', 'PUTS', 'P'

    Returns:
        dict with:
            - enter: True if reentry conditions are met
            - confidence: Weighted average fulfillment %
            - fulfillments: Dict of per-indicator fulfillment %
            - safety_floor_passed: True if all components >= floor
    """
    is_put = option_type.upper() in ('PUT', 'PUTS', 'P')
    direction_config = scoring_config.get('put' if is_put else 'call', {})

    confidence_threshold = scoring_config.get('confidence_threshold', 100)
    safety_floor = scoring_config.get('safety_floor', 20)

    # Get per-indicator configs with defaults
    rsi_cfg = direction_config.get('rsi', {})
    ewo_cfg = direction_config.get('ewo', {})
    ewo_avg_cfg = direction_config.get('ewo_avg', {})
    time_cfg = direction_config.get('time_minutes', {})

    # Calculate fulfillment for each indicator
    rsi_f = _calc_fulfillment(
        rsi,
        rsi_cfg.get('zero_ref', 85 if not is_put else 15),
        rsi_cfg.get('target', 30 if not is_put else 70),
        rsi_cfg.get('overflow_cap', 130),
    )

    ewo_f = _calc_fulfillment(
        ewo,
        ewo_cfg.get('zero_ref', 0.5 if not is_put else -0.5),
        ewo_cfg.get('target', 0),
        ewo_cfg.get('overflow_cap', 130),
    )

    ewo_avg_f = _calc_fulfillment(
        ewo_avg,
        ewo_avg_cfg.get('zero_ref', 0.5 if not is_put else -0.5),
        ewo_avg_cfg.get('target', 0),
        ewo_avg_cfg.get('overflow_cap', 120),
    )

    time_f = _calc_fulfillment(
        minutes_elapsed,
        time_cfg.get('zero_ref', 0),
        time_cfg.get('target', 15),
        time_cfg.get('overflow_cap', 115),
    )

    # Get weights
    w_rsi = rsi_cfg.get('weight', 1.5)
    w_ewo = ewo_cfg.get('weight', 1.0)
    w_ewo_avg = ewo_avg_cfg.get('weight', 0.5)
    w_time = time_cfg.get('weight', 1.0)

    total_weight = w_rsi + w_ewo + w_ewo_avg + w_time

    # Weighted average
    confidence = (w_rsi * rsi_f + w_ewo * ewo_f + w_ewo_avg * ewo_avg_f + w_time * time_f) / total_weight

    # Safety floor check — no single indicator can be strongly adverse
    fulfillments = {
        'rsi': rsi_f,
        'ewo': ewo_f,
        'ewo_avg': ewo_avg_f,
        'time': time_f,
    }
    safety_floor_passed = all(f >= safety_floor for f in fulfillments.values())

    enter = confidence >= confidence_threshold and safety_floor_passed

    return {
        'enter': enter,
        'confidence': confidence,
        'fulfillments': fulfillments,
        'safety_floor_passed': safety_floor_passed,
    }


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['StopLoss', 'check_stop_loss', 'CheckDueDiligence', 'score_reentry_confidence']
