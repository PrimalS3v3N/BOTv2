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


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['StopLoss', 'check_stop_loss', 'TakeProfitMilestones', 'MilestoneTracker',
           'MomentumPeak', 'MomentumPeakDetector', 'AIExitSignal', 'AIExitSignalDetector']


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


# =============================================================================
# INTERNAL - AI Exit Signal Strategy
# =============================================================================

class AIExitSignal:
    """
    Factory for creating AIExitSignalDetector instances.

    Wraps the local AI model (AIModel.py) to provide LLM-based exit signals.
    The model evaluates multi-timeframe market outlook and recommends hold/sell.

    Config: BACKTEST_CONFIG['ai_exit_signal']
        {
            'enabled': True,
            'model_path': '/path/to/model.gguf',
            'eval_interval': 5,           # Evaluate every N bars
            'min_bars_before_eval': 5,    # Wait N bars before first eval
            'exit_on_sell': True,         # Actually trigger exit on 'sell'
            'log_inferences': True,       # Save data for self-training
            ...
        }
    """

    def __init__(self, config=None):
        ai_config = config or Config.get_setting('backtest', 'ai_exit_signal', {})
        self.enabled = ai_config.get('enabled', False)
        self.config = ai_config

        self._model = None
        self._logger = None

    def load_model(self):
        """Load the AI model into GPU memory. Call once before backtesting."""
        if not self.enabled:
            return

        import AIModel

        model_path = self.config.get('model_path', '')
        self._model = AIModel.LocalAIModel(
            model_path=model_path,
            n_gpu_layers=self.config.get('n_gpu_layers', -1),
            n_ctx=self.config.get('n_ctx', 2048),
            temperature=self.config.get('temperature', 0.1),
            max_tokens=self.config.get('max_tokens', 256),
            seed=self.config.get('seed', 42),
        )
        self._model.load()

        if self.config.get('log_inferences', True):
            log_dir = self.config.get('log_dir', 'ai_training_data')
            self._logger = AIModel.AIAnalysisLogger(log_dir=log_dir)

    def unload_model(self):
        """Free model from GPU memory. Call after backtesting."""
        if self._model is not None:
            self._model.unload()
            self._model = None
        if self._logger is not None:
            self._logger.flush_remaining()

    def create_detector(self, ticker, option_type, strike):
        """
        Create a new AIExitSignalDetector for a position.

        Args:
            ticker: Stock ticker symbol
            option_type: 'CALL' or 'PUT'
            strike: Option strike price

        Returns:
            AIExitSignalDetector instance, or None if disabled.
        """
        if not self.enabled or self._model is None:
            return None
        return AIExitSignalDetector(
            self.config, self._model, self._logger,
            ticker, option_type, strike
        )

    @property
    def logger(self):
        return self._logger


class AIExitSignalDetector:
    """
    Tracks AI exit signal state for a single position.

    Accumulates bar data into a DataSnapshot, runs AI inference at
    configured intervals, and caches the last signal between evaluations.

    Signal fields (available per bar via current_signal):
        outlook_1m:  'bullish', 'bearish', or 'sideways'
        outlook_5m:  'bullish', 'bearish', or 'sideways'
        outlook_30m: 'bullish', 'bearish', or 'sideways'
        outlook_1h:  'bullish', 'bearish', or 'sideways'
        action:      'hold' or 'sell'
        reason:      One-sentence explanation
    """

    def __init__(self, config, model, logger, ticker, option_type, strike):
        import AIModel

        self.eval_interval = config.get('eval_interval', 5)
        self.min_bars_before_eval = config.get('min_bars_before_eval', 5)
        self.exit_on_sell = config.get('exit_on_sell', True)

        self._model = model
        self._logger = logger
        self._snapshot = AIModel.DataSnapshot(max_history=60)
        self._parser = AIModel.AISignalParser()

        self.ticker = ticker
        self.option_type = option_type
        self.strike = strike
        self.trade_label = f"{ticker}:{strike}:{option_type}"

        self._bars_since_eval = 0
        self._total_bars = 0

        # Last AI signal (persists between evaluations)
        self._current_signal = {
            'outlook_1m': None,
            'outlook_5m': None,
            'outlook_30m': None,
            'outlook_1h': None,
            'action': 'hold',
            'reason': None,
            'valid': False,
        }

    def update(self, bar_data, pnl_pct, minutes_held, option_price, timestamp):
        """
        Process a new bar and optionally run AI inference.

        Args:
            bar_data: dict with current bar's indicator values
                (stock_price, true_price, volume, vwap, ema_30, ewo, rsi,
                 supertrend_direction, market_bias, ichimoku_*, etc.)
            pnl_pct: Current position P&L percentage
            minutes_held: Minutes since entry
            option_price: Current option price
            timestamp: Bar timestamp

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self._snapshot.add_bar(bar_data)
        self._total_bars += 1
        self._bars_since_eval += 1

        # Wait for minimum bars before first evaluation
        if self._total_bars < self.min_bars_before_eval:
            return False, None

        # Only evaluate at configured intervals
        if self._bars_since_eval < self.eval_interval:
            # Return cached signal's exit decision
            if self.exit_on_sell and self._current_signal.get('action') == 'sell':
                return True, 'AI Exit Signal'
            return False, None

        # Time to run inference
        self._bars_since_eval = 0

        data_block = self._snapshot.build_prompt_data(
            ticker=self.ticker,
            option_type=self.option_type,
            strike=self.strike,
            pnl_pct=pnl_pct,
            minutes_held=minutes_held,
        )

        if data_block is None:
            return False, None

        import AIModel

        user_prompt = AIModel.USER_PROMPT_TEMPLATE.format(
            data_block=data_block,
            option_type=self.option_type,
            pnl_pct=f"{pnl_pct:.1f}" if not np.isnan(pnl_pct) else "N/A",
        )

        try:
            raw_response = self._model.inference(AIModel.SYSTEM_PROMPT, user_prompt)
            signal = self._parser.parse(raw_response)
        except Exception:
            signal = self._parser.parse(None)

        self._current_signal = signal

        # Log inference for self-training
        if self._logger is not None:
            self._logger.log_inference(
                trade_label=self.trade_label,
                timestamp=timestamp,
                data_block=data_block,
                ai_signal=signal,
                pnl_pct=pnl_pct,
                option_price=option_price,
            )

        # Check exit
        if self.exit_on_sell and signal.get('action') == 'sell':
            return True, 'AI Exit Signal'

        return False, None

    @property
    def current_signal(self):
        """Current AI signal dict (last evaluation result, cached between evals)."""
        return self._current_signal
