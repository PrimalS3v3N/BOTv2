"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""

import math
import numpy as np

import Config

# Export for use by other modules
__all__ = ['MomentumPeak', 'MomentumPeakDetector', 'StatsBookExit', 'StatsBookDetector',
           'AIExitSignal', 'AIExitSignalDetector', 'OptionsExit', 'OptionsExitDetector']


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


# =============================================================================
# INTERNAL - Options Exit System (Primary TP / SL)
# =============================================================================

class OptionsExit:
    """
    Factory for creating OptionsExitDetector instances.

    Primary exit system that manages take-profit and stop-loss for options
    contracts.  Differentiates between CALLs and PUTs:
      - CALL profits when stock rises  → stock drop = bad
      - PUT  profits when stock drops   → stock rise = bad

    Three layers:
      1. Hard stop loss (initial_sl_pct below entry option price)
      2. Trailing stop loss (engages at trail_activation_pct, scales with profit)
      3. Entry favorability assessment + reversal detection

    Config: BACKTEST_CONFIG['options_exit']
    """

    def __init__(self, config=None):
        oe_config = config or Config.get_setting('backtest', 'options_exit', {})
        self.enabled = oe_config.get('enabled', False)
        self.config = oe_config

    def create_detector(self, entry_option_price, option_type):
        """
        Create a new OptionsExitDetector for a position.

        Args:
            entry_option_price: Option price at entry
            option_type: 'CALL' or 'PUT'

        Returns:
            OptionsExitDetector instance, or None if disabled.
        """
        if not self.enabled:
            return None
        return OptionsExitDetector(self.config, entry_option_price, option_type)


class OptionsExitDetector:
    """
    Per-position detector that tracks trailing SL, hard SL, entry favorability,
    and reversal detection.

    Trailing SL uses a continuous logarithmic scaling function so the stop
    ratchets smoothly with profit instead of jumping at fixed milestones:

        trail_sl_pct = base_floor + scale * ln(1 + profit_pct / norm)

    For high-risk entries an addon is layered on top:

        addon = addon_base + addon_scale * ln(1 + profit_pct / addon_norm)

    The trailing SL price is:

        sl_price = entry_price * (1 + trail_sl_pct / 100)

    It only moves up (for profit protection), never down.
    """

    def __init__(self, config, entry_option_price, option_type):
        self.entry_option_price = entry_option_price
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

        # Hard SL
        self.initial_sl_pct = config.get('initial_sl_pct', 20)
        self.hard_sl_price = entry_option_price * (1 - self.initial_sl_pct / 100)

        # Trailing SL parameters
        self.trail_activation_pct = config.get('trail_activation_pct', 10)
        self.trail_base_floor_pct = config.get('trail_base_floor_pct', 5)
        self.trail_early_floor_minutes = config.get('trail_early_floor_minutes', 5)
        self.trail_scale = config.get('trail_scale', 25.0)
        self.trail_norm = config.get('trail_norm', 30.0)

        # High-risk addon
        self.risk_addon_base = config.get('risk_addon_base', 2.0)
        self.risk_addon_scale = config.get('risk_addon_scale', 5.0)
        self.risk_addon_norm = config.get('risk_addon_norm', 30.0)

        # Entry favorability / RiskOutlook
        self.confirmation_window = config.get('confirmation_window_bars', 10)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)

        # EMA reversal
        self.ema_reversal_periods = config.get('ema_reversal_periods', [10, 21, 50])
        self.ema_reversal_sensitivity = config.get('ema_reversal_sensitivity', 2)

        # ATR-SL
        self.atr_sl_enabled = config.get('atr_sl_enabled', True)

        # MACD
        self.macd_enabled = config.get('macd_enabled', True)

        # State
        self.trailing_sl_price = None       # Current trailing SL price (None until activated)
        self.trailing_active = False
        self.highest_profit_pct = 0.0       # Watermark: best profit seen so far
        self.bar_count = 0
        self.is_high_risk = False

        # Entry favorability assessment (computed once)
        self.favorability = None             # 'LOW' / 'MEDIUM' / 'HIGH' risk
        self.favorability_reasons = None
        self._favorability_assessed = False

        # Confirmation state
        self.confirmed = 'Pending'           # 'Pending' / 'Confirmed' / 'Denied'
        self._confirmation_bars = []         # Track option prices during confirmation window

        # Trend assessment
        self.trend_30m = None                # 'Uptrend' / 'Downtrend' / 'Sideways'
        self.ema_reversal = False

    # ------------------------------------------------------------------
    # Trailing SL math
    # ------------------------------------------------------------------

    def _calculate_trail_sl_pct(self, profit_pct):
        """
        Continuous trailing SL as a percentage *above entry* to lock in.

        Returns the % of entry price that the SL should sit at.
        E.g. if entry = $2.00 and this returns 35, SL = $2.00 * 1.35 = $2.70.
        """
        if profit_pct < self.trail_activation_pct:
            return 0.0  # Not yet active

        # First N minutes: breakeven floor (0%) to give the trade room
        if self.bar_count <= self.trail_early_floor_minutes:
            base = 0.0
        else:
            base = self.trail_base_floor_pct
        scaled = self.trail_scale * math.log(1 + profit_pct / self.trail_norm)
        sl_pct = base + scaled

        # High-risk addon
        if self.is_high_risk:
            addon = self.risk_addon_base + self.risk_addon_scale * math.log(
                1 + profit_pct / self.risk_addon_norm
            )
            sl_pct += addon

        # SL can never exceed current profit (leave at least 2% buffer)
        sl_pct = min(sl_pct, profit_pct - 2.0)

        return max(0.0, sl_pct)

    # ------------------------------------------------------------------
    # RiskOutlook — entry favorability assessment
    # ------------------------------------------------------------------

    def RiskOutlook(self, rsi, rsi_avg,
                    ema_values, stock_price, atr_sl_value,
                    macd_histogram, roc_30m, roc_day,
                    supertrend_direction, ewo, ewo_avg):
        """
        Assess whether the entry is in a favorable region.

        Uses two time horizons for trend context:
          - Primary: past 30 minutes (roc_30m) — immediate momentum
          - Secondary: since market open (roc_day) — broader session context

        Evaluates current indicators against contract type to score risk.

        Args:
            rsi: Current RSI value
            rsi_avg: Current RSI 10-min average
            ema_values: dict mapping period -> EMA value (e.g. {10: 450.2, 21: 449.8, ...})
            stock_price: Current stock price
            atr_sl_value: Current ATR trailing stoploss value
            macd_histogram: Current MACD histogram value
            roc_30m: 30-minute Rate of Change (primary trend)
            roc_day: Since-open Rate of Change (secondary / session trend)
            supertrend_direction: 1 (uptrend) or -1 (downtrend)
            ewo: Current EWO value
            ewo_avg: Current EWO 15-min average

        Returns:
            (risk_label, reasons_string): risk_label is 'LOW'/'MEDIUM'/'HIGH'
        """
        if self._favorability_assessed:
            return self.favorability, self.favorability_reasons

        self._favorability_assessed = True
        reasons = []
        risk_score = 0  # Higher = more risk

        # --- 1. Primary: 30-minute trend direction via ROC ---
        if not np.isnan(roc_30m):
            if self.is_call:
                if roc_30m > 0.3:
                    reasons.append(f'ROC30(+{roc_30m:.2f}=top)')
                    risk_score += 2
                    self.trend_30m = 'Uptrend'
                elif roc_30m < -0.1:
                    reasons.append(f'ROC30({roc_30m:.2f}=downtrend)')
                    risk_score += 3
                    self.trend_30m = 'Downtrend'
                else:
                    self.trend_30m = 'Sideways'
            else:  # PUT
                if roc_30m < -0.3:
                    reasons.append(f'ROC30({roc_30m:.2f}=bottom)')
                    risk_score += 2
                    self.trend_30m = 'Downtrend'
                elif roc_30m > 0.1:
                    reasons.append(f'ROC30(+{roc_30m:.2f}=uptrend)')
                    risk_score += 3
                    self.trend_30m = 'Uptrend'
                else:
                    self.trend_30m = 'Sideways'

        # --- 1b. Secondary: since-open session trend ---
        if not np.isnan(roc_day):
            if self.is_call:
                if roc_day > 0.5:
                    reasons.append(f'ROCday(+{roc_day:.2f}=extended)')
                    risk_score += 1
                elif roc_day < -0.3:
                    reasons.append(f'ROCday({roc_day:.2f}=bearish_session)')
                    risk_score += 2
            else:  # PUT
                if roc_day < -0.5:
                    reasons.append(f'ROCday({roc_day:.2f}=extended)')
                    risk_score += 1
                elif roc_day > 0.3:
                    reasons.append(f'ROCday(+{roc_day:.2f}=bullish_session)')
                    risk_score += 2

        # --- 2. RSI overbought/oversold ---
        combined_rsi = rsi
        if not np.isnan(rsi) and not np.isnan(rsi_avg):
            combined_rsi = (rsi + rsi_avg) / 2

        if not np.isnan(combined_rsi):
            if self.is_call and combined_rsi >= self.rsi_overbought:
                reasons.append(f'RSI({combined_rsi:.0f}=overbought)')
                risk_score += 3
            elif not self.is_call and combined_rsi <= self.rsi_oversold:
                reasons.append(f'RSI({combined_rsi:.0f}=oversold)')
                risk_score += 3
            elif self.is_call and combined_rsi <= self.rsi_oversold:
                # Oversold = good for calls (potential bounce)
                risk_score -= 1
            elif not self.is_call and combined_rsi >= self.rsi_overbought:
                # Overbought = good for puts (potential drop)
                risk_score -= 1

        # --- 3. EMA positioning (price vs EMAs) ---
        emas_breached = 0
        for period in self.ema_reversal_periods:
            ema_val = ema_values.get(period, np.nan)
            if np.isnan(ema_val):
                continue
            if self.is_call and stock_price < ema_val:
                emas_breached += 1
            elif not self.is_call and stock_price > ema_val:
                emas_breached += 1

        if emas_breached > 0:
            # Smaller EMAs breached = more concerning (recent reversal)
            risk_per_ema = 1 if emas_breached < self.ema_reversal_sensitivity else 2
            risk_score += emas_breached * risk_per_ema
            breached_label = f'{emas_breached}EMAs_against'
            reasons.append(breached_label)

        # --- 4. ATR-SL favorability ---
        if self.atr_sl_enabled and not np.isnan(atr_sl_value):
            if self.is_call and stock_price < atr_sl_value:
                reasons.append('Below_ATR-SL')
                risk_score += 2
            elif not self.is_call and stock_price > atr_sl_value:
                reasons.append('Above_ATR-SL')
                risk_score += 2

        # --- 5. Supertrend direction ---
        if not np.isnan(supertrend_direction):
            if self.is_call and supertrend_direction == -1:
                reasons.append('ST_bearish')
                risk_score += 2
            elif not self.is_call and supertrend_direction == 1:
                reasons.append('ST_bullish')
                risk_score += 2

        # --- 6. MACD histogram alignment ---
        if self.macd_enabled and not np.isnan(macd_histogram):
            if self.is_call and macd_histogram < 0:
                reasons.append(f'MACD_neg({macd_histogram:.3f})')
                risk_score += 1
            elif not self.is_call and macd_histogram > 0:
                reasons.append(f'MACD_pos({macd_histogram:.3f})')
                risk_score += 1

        # --- 7. EWO momentum alignment ---
        if not np.isnan(ewo):
            if self.is_call and ewo < 0:
                reasons.append(f'EWO_neg({ewo:.3f})')
                risk_score += 1
            elif not self.is_call and ewo > 0:
                reasons.append(f'EWO_pos({ewo:.3f})')
                risk_score += 1

        # --- Score to label ---
        if risk_score >= 6:
            self.favorability = 'HIGH'
        elif risk_score >= 3:
            self.favorability = 'MEDIUM'
        else:
            self.favorability = 'LOW'

        self.favorability_reasons = '|'.join(reasons) if reasons else None
        self.is_high_risk = self.favorability == 'HIGH'

        return self.favorability, self.favorability_reasons

    # ------------------------------------------------------------------
    # Per-bar update
    # ------------------------------------------------------------------

    def update(self, option_price, stock_price, ema_values=None):
        """
        Per-bar update: check hard SL, update trailing SL, check EMA reversal.

        Args:
            option_price: Current estimated option price
            stock_price: Current stock price
            ema_values: dict mapping period -> EMA value (for reversal detection)

        Returns:
            (should_exit, exit_reason, state_dict)
            state_dict contains per-bar state for the databook.
        """
        self.bar_count += 1
        ema_values = ema_values or {}

        profit_pct = ((option_price - self.entry_option_price) / self.entry_option_price) * 100

        # Track confirmation window
        if self.bar_count <= self.confirmation_window:
            self._confirmation_bars.append(option_price)
            if self.bar_count == self.confirmation_window:
                self._evaluate_confirmation()

        # --- EMA reversal check ---
        emas_against = 0
        for period in self.ema_reversal_periods:
            ema_val = ema_values.get(period, np.nan)
            if np.isnan(ema_val):
                continue
            if self.is_call and stock_price < ema_val:
                emas_against += 1
            elif not self.is_call and stock_price > ema_val:
                emas_against += 1
        self.ema_reversal = emas_against >= self.ema_reversal_sensitivity

        # --- Hard stop loss ---
        if option_price <= self.hard_sl_price:
            state = self._build_state(option_price, profit_pct)
            return True, 'Hard-SL', state

        # --- Trailing stop loss ---
        if profit_pct > self.highest_profit_pct:
            self.highest_profit_pct = profit_pct

        # Calculate where trailing SL should be based on watermark profit
        trail_sl_pct = self._calculate_trail_sl_pct(self.highest_profit_pct)

        if trail_sl_pct > 0:
            self.trailing_active = True
            new_trailing_sl = self.entry_option_price * (1 + trail_sl_pct / 100)

            # Ratchet: only move SL up, never down
            if self.trailing_sl_price is None or new_trailing_sl > self.trailing_sl_price:
                self.trailing_sl_price = new_trailing_sl

            # Check if price hit trailing SL
            if option_price <= self.trailing_sl_price:
                state = self._build_state(option_price, profit_pct)
                return True, 'Trail-SL', state

        state = self._build_state(option_price, profit_pct)
        return False, None, state

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def _evaluate_confirmation(self):
        """
        After the confirmation window, determine if the trade direction
        is confirmed or denied based on option price trajectory.
        """
        if len(self._confirmation_bars) < 2:
            self.confirmed = 'Pending'
            return

        # Compare last price to first price in window
        first = self._confirmation_bars[0]
        last = self._confirmation_bars[-1]
        change_pct = ((last - first) / first) * 100

        # Count positive bars
        positive_bars = sum(
            1 for i in range(1, len(self._confirmation_bars))
            if self._confirmation_bars[i] > self._confirmation_bars[i - 1]
        )
        positive_ratio = positive_bars / (len(self._confirmation_bars) - 1)

        if change_pct > 0 and positive_ratio >= 0.5:
            self.confirmed = 'Confirmed'
        elif change_pct < -3 or positive_ratio < 0.3:
            self.confirmed = 'Denied'
        else:
            self.confirmed = 'Pending'

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------

    def _build_state(self, option_price, profit_pct):
        """Build state dict for databook recording."""
        return {
            'sl_trailing': self.trailing_sl_price if self.trailing_sl_price else np.nan,
            'sl_hard': self.hard_sl_price,
            'tp_risk_outlook': self.favorability,
            'tp_risk_reasons': self.favorability_reasons,
            'tp_trend_30m': self.trend_30m,
            'sl_ema_reversal': self.ema_reversal,
            'tp_confirmed': self.confirmed,
        }
