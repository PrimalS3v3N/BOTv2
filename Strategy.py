"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""

import numpy as np


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['MomentumPeak', 'MomentumPeakDetector', 'StatsBookExit', 'StatsBookDetector',
           'AIExitSignal', 'AIExitSignalDetector']


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
# INTERNAL - StatsBook Exit Strategy
# =============================================================================

class StatsBookExit:
    """
    Factory for creating StatsBookDetector instances.

    Uses StatsBook historical statistics to create exit/hold signals
    based on where current price action falls within the stock's
    historical bounds:

    - Below Min: Noise floor — movement too small to act on
    - Min to Median: Normal trading zone — HOLD
    - Median to Max: Extended zone — prepare to sell, tighten stops
    - At/above Max: Historical extreme — EXIT (selling zone)

    Evaluates EWO momentum and H-L range against Median.Max bounds
    (typical peaks, more actionable than absolute Max outliers).

    Config: BACKTEST_CONFIG['statsbook_exit']
        {
            'enabled': True,
            'timeframe': '5m',             # StatsBook timeframe to compare against
            'ewo_max_exit': True,          # Exit when EWO >= Median.Max(EWO)
            'hl_max_exit': True,           # Exit when rolling H-L >= Median.Max(H-L)
            'min_profit_pct': 10,          # Minimum option profit % to consider exit
            'min_hold_bars': 5,            # Minimum bars held before StatsBook exit
            'rolling_window': 5,           # Bars for rolling H-L range calculation
        }
    """

    def __init__(self, config=None):
        sb_config = config or Config.get_setting('backtest', 'statsbook_exit', {})
        self.enabled = sb_config.get('enabled', False)
        self.config = sb_config

    def create_detector(self, statsbook_df):
        """Create a new StatsBookDetector for a position. Returns None if disabled or no data."""
        if not self.enabled or statsbook_df is None or statsbook_df.empty:
            return None
        return StatsBookDetector(self.config, statsbook_df)


class StatsBookDetector:
    """
    Evaluates current price action against StatsBook historical bounds.

    Uses Median.Max / Median / Median.Min as the reference bounds
    (more robust than absolute Max/Min which can be one-off outliers).

    Zones:
    - EWO >= Median.Max(EWO): Momentum at typical peak → EXIT
    - EWO around Median(EWO): Normal momentum → HOLD
    - EWO <= Median.Min(EWO): Weak momentum, below noise floor

    - Rolling H-L >= Median.Max(H-L): Range at typical extreme → EXIT
    - Rolling H-L around Median(H-L): Normal range → HOLD
    - Rolling H-L <= Median.Min(H-L): Tight range, no signal
    """

    def __init__(self, config, statsbook_df):
        self.min_profit_pct = config.get('min_profit_pct', 10)
        self.min_hold_bars = config.get('min_hold_bars', 5)
        self.ewo_max_exit = config.get('ewo_max_exit', True)
        self.hl_max_exit = config.get('hl_max_exit', True)
        self.rolling_window = config.get('rolling_window', 5)
        tf = config.get('timeframe', '5m')

        # Extract EWO bounds from StatsBook
        self.ewo_max = self._get_bound(statsbook_df, 'Median.Max(EWO)', tf)
        self.ewo_median = self._get_bound(statsbook_df, 'Median(EWO)', tf)
        self.ewo_min = self._get_bound(statsbook_df, 'Median.Min(EWO)', tf)

        # Extract H-L bounds from StatsBook
        self.hl_max = self._get_bound(statsbook_df, 'Median.Max(H-L)', tf)
        self.hl_median = self._get_bound(statsbook_df, 'Median(H-L)', tf)
        self.hl_min = self._get_bound(statsbook_df, 'Median.Min(H-L)', tf)

        # Rolling window state
        self.bar_count = 0
        self.high_window = []
        self.low_window = []

    def _get_bound(self, df, metric, tf):
        """Safely extract a bound value from the StatsBook DataFrame."""
        try:
            val = df.loc[metric, tf]
            return float(val) if not np.isnan(float(val)) else np.nan
        except (KeyError, TypeError, ValueError):
            return np.nan

    def update(self, pnl_pct, ewo, stock_high, stock_low):
        """
        Check for exit/hold signal based on StatsBook bounds.

        Args:
            pnl_pct: Current P&L percentage
            ewo: Current EWO value
            stock_high: Current bar high price
            stock_low: Current bar low price

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.bar_count += 1

        # Track rolling window for H-L range calculation
        if not np.isnan(stock_high) and not np.isnan(stock_low) and stock_low > 0:
            self.high_window.append(stock_high)
            self.low_window.append(stock_low)
            if len(self.high_window) > self.rolling_window:
                self.high_window.pop(0)
                self.low_window.pop(0)

        # Need minimum bars before evaluating
        if self.bar_count < self.min_hold_bars:
            return False, None

        # Need minimum profit to consider exit
        if np.isnan(pnl_pct) or pnl_pct < self.min_profit_pct:
            return False, None

        # Check EWO against Median.Max bound (selling zone)
        if self.ewo_max_exit and not np.isnan(ewo) and not np.isnan(self.ewo_max):
            if ewo >= self.ewo_max:
                return True, 'StatsBook - EWO Max'

        # Check rolling H-L range against Median.Max bound (range exhaustion)
        if self.hl_max_exit and len(self.high_window) >= self.rolling_window:
            rolling_hl = max(self.high_window) - min(self.low_window)
            if not np.isnan(self.hl_max) and rolling_hl >= self.hl_max:
                return True, 'StatsBook - Range Max'

        return False, None


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
        self._optimal_logger = None

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

        log_dir = self.config.get('log_dir', 'ai_training_data')

        if self.config.get('log_inferences', True):
            self._logger = AIModel.AIAnalysisLogger(log_dir=log_dir)

        # Always create the optimal exit logger (works even without the AI model running)
        self._optimal_logger = AIModel.OptimalExitLogger(log_dir=log_dir)

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

    @property
    def optimal_logger(self):
        return self._optimal_logger


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
                (stock_price, true_price, volume, vwap, ema_21, ewo, rsi,
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
