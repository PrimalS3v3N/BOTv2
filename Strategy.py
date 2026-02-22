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
__all__ = ['DeferredEntry', 'DeferredEntryFilter',
           'MomentumPeak', 'MomentumPeakDetector', 'StatsBookExit', 'StatsBookDetector',
           'AIExitSignal', 'AIExitSignalDetector', 'OptionsExit', 'OptionsExitDetector',
           'VolumeClimaxExit', 'VolumeClimaxDetector',
           'TimeStop', 'TimeStopDetector',
           'VWAPCrossExit', 'VWAPCrossDetector',
           'SupertrendFlipExit', 'SupertrendFlipDetector',
           'MarketTrend', 'MarketTrendDetector',
           'PeakPeakExit', 'PeakPeakDetector']


# =============================================================================
#                           ENTRY STRATEGIES
# =============================================================================

# =============================================================================
# INTERNAL - Deferred Entry Filter
# =============================================================================

class DeferredEntry:
    """
    Factory for creating DeferredEntryFilter instances.

    When entry RiskOutlook is HIGH, the system defers the actual entry
    until an exhaustion confirmation signal fires — timing the entry
    closer to the price extreme (peak for PUTs, trough for CALLs).

    Three independent exhaustion filters run in parallel. The first one
    to trigger causes entry:

    1. Stochastic Overbought/Oversold Crossover
       - Wait for Stoch %K to enter extreme zone (>90 for PUTs, <10 for CALLs)
       - Then enter when %K crosses back through %D (reversal confirmed)

    2. Volume Climax
       - Wait for a volume bar >= volume_multiplier * rolling average
       - Enter on the next bar (exhaustion spike followed by reversal)

    3. Indicator Saturation (Contrarian)
       - Count how many sub-indicators agree with the adverse direction
       - When saturation_threshold+ are aligned, the extreme is likely near
       - Enter when at least one flips back (consensus breaking)

    If none fire within max_defer_bars, the original entry executes
    (timeout fallback to avoid missing the trade entirely).

    Config: BACKTEST_CONFIG['deferred_entry']
    """

    def __init__(self, config=None):
        de_config = config or Config.get_setting('backtest', 'deferred_entry', {})
        self.enabled = de_config.get('enabled', True)
        self.config = de_config

    def create_filter(self, option_type):
        """Create a new DeferredEntryFilter for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return DeferredEntryFilter(self.config, option_type)


class DeferredEntryFilter:
    """
    Scans bar-by-bar for exhaustion confirmation before allowing entry.

    For PUTs: waits for the underlying to reach an overbought extreme
    and show signs of reversal (ideal put entry = underlying peak).

    For CALLs: waits for the underlying to reach an oversold extreme
    and show signs of reversal (ideal call entry = underlying trough).

    Call update() each bar during the deferral window. When it returns
    True, the entry should execute at that bar's price.
    """

    def __init__(self, config, option_type):
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']
        self.is_put = not self.is_call

        # Stochastic crossover filter
        self.stoch_enabled = config.get('stoch_enabled', True)
        self.stoch_extreme_threshold = config.get('stoch_extreme_threshold', 90)
        self.stoch_extreme_threshold_low = config.get('stoch_extreme_threshold_low', 10)
        self.stoch_reached_extreme = False

        # Volume climax filter
        self.volume_enabled = config.get('volume_enabled', True)
        self.volume_lookback = config.get('volume_lookback', 10)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.volume_history = []
        self.volume_climax_fired = False

        # Indicator saturation filter
        self.saturation_enabled = config.get('saturation_enabled', True)
        self.saturation_threshold = config.get('saturation_threshold', 10)
        self.peak_saturation_count = 0
        self.saturation_reached = False

        # Timeout (config in minutes, supports legacy 'max_defer_bars' key)
        self.max_defer_minutes = config.get('max_defer_minutes', config.get('max_defer_bars', 10))
        self.bar_count = 0  # kept for display/logging
        self._entry_time = None

        # Track which filter triggered
        self.trigger_reason = None

    def update(self, stoch_k, stoch_d, volume, rsi,
               supertrend_direction, macd_histogram, roc,
               ema_values, stock_price, atr_sl_value, ewo,
               timestamp=None):
        """
        Evaluate whether exhaustion confirmation has occurred.

        Args:
            stoch_k: Stochastic %K value
            stoch_d: Stochastic %D value
            volume: Current bar volume
            rsi: Current RSI value
            supertrend_direction: Supertrend direction (+1/-1)
            macd_histogram: MACD histogram value
            roc: Rate of Change (30-bar)
            ema_values: dict of {period: value} for EMAs
            stock_price: Current stock close price
            atr_sl_value: ATR-based stop loss level
            ewo: Elliott Wave Oscillator value
            timestamp: Current timestamp for time-based tracking

        Returns:
            bool: True if entry should execute now
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        # --- Timeout: enter regardless after max_defer_minutes ---
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self.bar_count  # fallback: 1 bar ≈ 1 minute
        if minutes_elapsed >= self.max_defer_minutes:
            self.trigger_reason = 'Deferred-Timeout'
            return True

        triggered = False

        # --- Filter 1: Stochastic Overbought/Oversold Crossover ---
        if self.stoch_enabled and not np.isnan(stoch_k) and not np.isnan(stoch_d):
            if self.is_put:
                # For PUTs: wait for stoch to go overbought, then cross down
                if stoch_k >= self.stoch_extreme_threshold:
                    self.stoch_reached_extreme = True
                if self.stoch_reached_extreme and stoch_k < stoch_d:
                    self.trigger_reason = 'Deferred-StochCrossover'
                    triggered = True
            else:
                # For CALLs: wait for stoch to go oversold, then cross up
                if stoch_k <= self.stoch_extreme_threshold_low:
                    self.stoch_reached_extreme = True
                if self.stoch_reached_extreme and stoch_k > stoch_d:
                    self.trigger_reason = 'Deferred-StochCrossover'
                    triggered = True

        # --- Filter 2: Volume Climax ---
        if self.volume_enabled and not np.isnan(volume):
            self.volume_history.append(volume)
            if len(self.volume_history) > self.volume_lookback:
                self.volume_history.pop(0)

            if len(self.volume_history) >= 2:
                avg_volume = sum(self.volume_history[:-1]) / len(self.volume_history[:-1])
                if avg_volume > 0 and volume >= self.volume_multiplier * avg_volume:
                    self.volume_climax_fired = True

                # Enter on the bar AFTER the climax
                if self.volume_climax_fired and volume < self.volume_multiplier * avg_volume:
                    if not triggered:
                        self.trigger_reason = 'Deferred-VolumeClimax'
                    triggered = True

        # --- Filter 3: Indicator Saturation (Contrarian) ---
        if self.saturation_enabled:
            adverse_count = self._count_adverse_indicators(
                rsi, supertrend_direction, macd_histogram, roc,
                ema_values, stock_price, atr_sl_value, ewo
            )

            if adverse_count > self.peak_saturation_count:
                self.peak_saturation_count = adverse_count

            if self.peak_saturation_count >= self.saturation_threshold:
                self.saturation_reached = True

            # Enter when consensus starts breaking (at least 2 fewer than peak)
            if self.saturation_reached and adverse_count <= self.peak_saturation_count - 2:
                if not triggered:
                    self.trigger_reason = 'Deferred-Saturation'
                triggered = True

        return triggered

    def _count_adverse_indicators(self, rsi, supertrend_direction,
                                   macd_histogram, roc,
                                   ema_values, stock_price,
                                   atr_sl_value, ewo):
        """
        Count how many indicators are aligned AGAINST the position direction.

        For PUTs, "adverse" means indicators are bullish (price going up).
        For CALLs, "adverse" means indicators are bearish (price going down).

        Returns:
            int: count of adverse indicators (0-12 range)
        """
        count = 0

        # RSI direction
        if not np.isnan(rsi):
            if self.is_put and rsi > 60:
                count += 1
            elif self.is_call and rsi < 40:
                count += 1

        # Supertrend direction
        if not np.isnan(supertrend_direction):
            if self.is_put and supertrend_direction > 0:
                count += 1
            elif self.is_call and supertrend_direction < 0:
                count += 1

        # MACD histogram
        if not np.isnan(macd_histogram):
            if self.is_put and macd_histogram > 0:
                count += 1
            elif self.is_call and macd_histogram < 0:
                count += 1

        # ROC direction
        if not np.isnan(roc):
            if self.is_put and roc > 0:
                count += 1
            elif self.is_call and roc < 0:
                count += 1

        # EWO direction
        if not np.isnan(ewo):
            if self.is_put and ewo > 0:
                count += 1
            elif self.is_call and ewo < 0:
                count += 1

        # EMA positioning (price vs each EMA)
        for period, ema_val in (ema_values or {}).items():
            if np.isnan(ema_val):
                continue
            if self.is_put and stock_price > ema_val:
                count += 1
            elif self.is_call and stock_price < ema_val:
                count += 1

        # ATR-SL positioning
        if not np.isnan(atr_sl_value):
            if self.is_put and stock_price > atr_sl_value:
                count += 1
            elif self.is_call and stock_price < atr_sl_value:
                count += 1

        return count


# =============================================================================
#                           EXIT STRATEGIES
# =============================================================================

# =============================================================================
# INTERNAL - Momentum Peak Detection Strategy
# =============================================================================

class MomentumPeak:
    """
    Factory for creating MomentumPeakDetector instances.

    Detects momentum exhaustion peaks using RSI overbought/oversold reversal,
    EWO decline/incline, Stochastic crossovers, and RSI-below/above-average
    confirmation. Works for both CALLs and PUTs.

    CALLs: RSI overbought → dropping, EWO declining, Stochastic bearish crossover
    PUTs:  RSI oversold → bouncing, EWO increasing, Stochastic bullish crossover

    Config: BACKTEST_CONFIG['momentum_peak']
        {
            'enabled': True,
            'min_profit_pct': 15,          # Minimum option profit %
            'rsi_overbought': 80,          # RSI overbought threshold (CALLs)
            'rsi_oversold': 20,            # RSI oversold threshold (PUTs)
            'rsi_lookback': 5,             # Bars to check for recent extreme
            'rsi_drop_threshold': 10,      # Min RSI change from extreme
            'rsi_recovery_level': 30,      # RSI must exit extreme zone past this
            'ewo_declining_bars': 3,       # Consecutive EWO bars in adverse direction
            'require_rsi_below_avg': True, # RSI vs RSI_10min_avg confirmation
            'stoch_overbought': 80,        # Stochastic overbought zone
            'stoch_oversold': 20,          # Stochastic oversold zone
            'use_stochastic': True,        # Enable stochastic crossover confirmation
            'spread_contraction_bars': 3,  # Option price must decline N bars
        }
    """

    def __init__(self, config=None):
        mp_config = config or Config.get_setting('backtest', 'momentum_peak', {})
        self.enabled = mp_config.get('enabled', False)
        self.config = mp_config

    def create_detector(self, option_type=None):
        """Create a new MomentumPeakDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return MomentumPeakDetector(self.config, option_type)


class MomentumPeakDetector:
    """
    Tracks momentum state for a single position and detects peaks.

    For CALLs (stock topped out → sell call):
    1. Option profit >= min_profit_pct
    2. RSI reached overbought (>80) within lookback window
    3. RSI dropped by >= rsi_drop_threshold from peak
    4. RSI has dropped below (100 - rsi_recovery_level) to confirm exit from overbought
    5. EWO declining for N consecutive bars
    6. RSI crossed below RSI_10min_avg (optional)
    7. Stochastic %K crossed below %D in overbought zone (optional)
    8. Option price declining for N consecutive bars (spread contraction)

    For PUTs (stock bottomed out → sell put):
    1. Option profit >= min_profit_pct
    2. RSI reached oversold (<20) within lookback window
    3. RSI bounced up by >= rsi_drop_threshold from trough
    4. RSI has risen above rsi_recovery_level to confirm exit from oversold
    5. EWO increasing for N consecutive bars
    6. RSI crossed above RSI_10min_avg (optional)
    7. Stochastic %K crossed above %D in oversold zone (optional)
    8. Option price declining for N consecutive bars (spread contraction)
    """

    def __init__(self, config, option_type=None):
        self.min_profit_pct = config.get('min_profit_pct', 15)
        self.rsi_overbought = config.get('rsi_overbought', 80)
        self.rsi_oversold = config.get('rsi_oversold', 20)
        self.rsi_lookback = config.get('rsi_lookback', 5)
        self.rsi_drop_threshold = config.get('rsi_drop_threshold', 10)
        self.rsi_recovery_level = config.get('rsi_recovery_level', 30)
        self.ewo_declining_bars = config.get('ewo_declining_bars', 3)
        self.require_rsi_below_avg = config.get('require_rsi_below_avg', True)
        self.stoch_overbought = config.get('stoch_overbought', 80)
        self.stoch_oversold = config.get('stoch_oversold', 20)
        self.use_stochastic = config.get('use_stochastic', True)
        self.spread_contraction_bars = config.get('spread_contraction_bars', 3)

        # Determine position direction
        ot = (option_type or '').upper()
        self.is_call = ot in ('CALL', 'CALLS', 'C')

        self.rsi_history = []
        self.ewo_history = []
        self.stoch_k_history = []
        self.stoch_d_history = []
        self.spread_history = []

    def update(self, pnl_pct, rsi, rsi_avg, ewo, stoch_k=np.nan, stoch_d=np.nan, option_price=np.nan):
        """
        Check for momentum peak exit signal.

        Args:
            pnl_pct: Current P&L percentage
            rsi: Current RSI value
            rsi_avg: Current RSI 10-min average
            ewo: Current EWO value
            stoch_k: Current Stochastic %K value
            stoch_d: Current Stochastic %D value
            option_price: Current option price for spread contraction check

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.rsi_history.append(rsi)
        self.ewo_history.append(ewo)
        self.stoch_k_history.append(stoch_k)
        self.stoch_d_history.append(stoch_d)
        self.spread_history.append(option_price)

        # Need at least 2 bars of history
        if len(self.rsi_history) < 2 or len(self.ewo_history) < 2:
            return False, None

        # Skip if critical values are NaN
        if np.isnan(pnl_pct) or np.isnan(rsi) or np.isnan(ewo):
            return False, None

        # 1. Minimum profit threshold
        if pnl_pct < self.min_profit_pct:
            return False, None

        if self.is_call:
            return self._check_call_peak(rsi, rsi_avg, ewo, stoch_k, stoch_d)
        else:
            return self._check_put_peak(rsi, rsi_avg, ewo, stoch_k, stoch_d)

    def _check_call_peak(self, rsi, rsi_avg, ewo, stoch_k, stoch_d):
        """CALL peak: stock topped out, RSI overbought→dropping, EWO declining."""
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

        # 4. RSI has actually exited the overbought zone (not just a minor dip within it)
        if rsi > (100 - self.rsi_recovery_level):
            return False, None

        # 5. EWO is declining for N consecutive bars
        if not self._ewo_trending(declining=True):
            return False, None

        # 6. RSI below its 10-min average (optional confirmation)
        if self.require_rsi_below_avg and not np.isnan(rsi_avg):
            if rsi >= rsi_avg:
                return False, None

        # 7. Stochastic bearish crossover in overbought zone (optional)
        if self.use_stochastic and not self._stoch_crossover_bearish():
            return False, None

        # 8. Option price declining for N consecutive bars (spread contraction)
        if not self._spread_contracting():
            return False, None

        return True, 'Momentum Peak'

    def _check_put_peak(self, rsi, rsi_avg, ewo, stoch_k, stoch_d):
        """PUT peak: stock bottomed out, RSI oversold→bouncing, EWO increasing."""
        # 2. RSI was oversold recently (within lookback window)
        lookback = min(self.rsi_lookback, len(self.rsi_history))
        recent_rsi = self.rsi_history[-lookback:]
        valid_recent = [r for r in recent_rsi if not np.isnan(r)]
        if not valid_recent:
            return False, None
        rsi_trough = min(valid_recent)

        if rsi_trough > self.rsi_oversold:
            return False, None

        # 3. RSI has bounced up significantly from trough
        rsi_bounce = rsi - rsi_trough
        if rsi_bounce < self.rsi_drop_threshold:
            return False, None

        # 4. RSI has actually exited the oversold zone (not just a minor bounce within it)
        if rsi < self.rsi_recovery_level:
            return False, None

        # 5. EWO is increasing for N consecutive bars (bullish reversal = bad for PUT)
        if not self._ewo_trending(declining=False):
            return False, None

        # 6. RSI above its 10-min average (optional confirmation — inverse of CALL)
        if self.require_rsi_below_avg and not np.isnan(rsi_avg):
            if rsi <= rsi_avg:
                return False, None

        # 7. Stochastic bullish crossover in oversold zone (optional)
        if self.use_stochastic and not self._stoch_crossover_bullish():
            return False, None

        # 8. Option price declining for N consecutive bars (spread contraction)
        if not self._spread_contracting():
            return False, None

        return True, 'Momentum Peak'

    def _ewo_trending(self, declining=True):
        """Check if EWO has been trending in the specified direction for N bars."""
        ewo_check_len = self.ewo_declining_bars + 1
        if len(self.ewo_history) < ewo_check_len:
            return False

        recent_ewo = self.ewo_history[-ewo_check_len:]
        for j in range(len(recent_ewo) - 1):
            if np.isnan(recent_ewo[j]) or np.isnan(recent_ewo[j + 1]):
                return False
            if declining:
                if recent_ewo[j + 1] >= recent_ewo[j]:
                    return False
            else:
                if recent_ewo[j + 1] <= recent_ewo[j]:
                    return False
        return True

    def _stoch_crossover_bearish(self):
        """Check if %K crossed below %D recently while in overbought zone."""
        if len(self.stoch_k_history) < 2 or len(self.stoch_d_history) < 2:
            return False
        prev_k = self.stoch_k_history[-2]
        curr_k = self.stoch_k_history[-1]
        prev_d = self.stoch_d_history[-2]
        curr_d = self.stoch_d_history[-1]
        if np.isnan(prev_k) or np.isnan(curr_k) or np.isnan(prev_d) or np.isnan(curr_d):
            return False
        # %K was above %D and now crossed below, while in overbought zone
        return prev_k >= prev_d and curr_k < curr_d and prev_k >= self.stoch_overbought

    def _stoch_crossover_bullish(self):
        """Check if %K crossed above %D recently while in oversold zone."""
        if len(self.stoch_k_history) < 2 or len(self.stoch_d_history) < 2:
            return False
        prev_k = self.stoch_k_history[-2]
        curr_k = self.stoch_k_history[-1]
        prev_d = self.stoch_d_history[-2]
        curr_d = self.stoch_d_history[-1]
        if np.isnan(prev_k) or np.isnan(curr_k) or np.isnan(prev_d) or np.isnan(curr_d):
            return False
        # %K was below %D and now crossed above, while in oversold zone
        return prev_k <= prev_d and curr_k > curr_d and prev_k <= self.stoch_oversold

    def _spread_contracting(self):
        """Check if option price has been declining for N consecutive bars."""
        n = self.spread_contraction_bars
        if n <= 0:
            return True  # disabled
        if len(self.spread_history) < n + 1:
            return False
        recent = self.spread_history[-(n + 1):]
        for i in range(len(recent) - 1):
            if np.isnan(recent[i]) or np.isnan(recent[i + 1]):
                return False
            if recent[i + 1] >= recent[i]:  # not contracting
                return False
        return True


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
        self.min_hold_minutes = config.get('min_hold_minutes', config.get('min_hold_bars', 5))
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
        self._entry_time = None
        self.high_window = []
        self.low_window = []

    def _get_bound(self, df, metric, tf):
        """Safely extract a bound value from the StatsBook DataFrame."""
        try:
            val = df.loc[metric, tf]
            return float(val) if not np.isnan(float(val)) else np.nan
        except (KeyError, TypeError, ValueError):
            return np.nan

    def update(self, pnl_pct, ewo, stock_high, stock_low, timestamp=None):
        """
        Check for exit/hold signal based on StatsBook bounds.

        Args:
            pnl_pct: Current P&L percentage
            ewo: Current EWO value
            stock_high: Current bar high price
            stock_low: Current bar low price
            timestamp: Current timestamp for time-based tracking

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        # Track rolling window for H-L range calculation
        if not np.isnan(stock_high) and not np.isnan(stock_low) and stock_low > 0:
            self.high_window.append(stock_high)
            self.low_window.append(stock_low)
            if len(self.high_window) > self.rolling_window:
                self.high_window.pop(0)
                self.low_window.pop(0)

        # Need minimum time before evaluating
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self.bar_count
        if minutes_elapsed < self.min_hold_minutes:
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
        self._entry_time = None
        self._last_eval_time = None
        self.eval_interval_minutes = config.get('eval_interval_minutes', config.get('eval_interval', 5))
        self.min_minutes_before_eval = config.get('min_minutes_before_eval',
                                                   config.get('min_bars_before_eval', 5))

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
                (stock_price, volume, vwap, ema_21, ewo, rsi,
                 supertrend_direction, ticker_trend, ichimoku_*, etc.)
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
        if self._entry_time is None and timestamp:
            self._entry_time = timestamp

        # Wait for minimum time before first evaluation
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self._total_bars
        if minutes_elapsed < self.min_minutes_before_eval:
            return False, None

        # Only evaluate at configured intervals (time-based)
        if timestamp and self._last_eval_time:
            minutes_since_eval = (timestamp - self._last_eval_time).total_seconds() / 60
        else:
            minutes_since_eval = self._bars_since_eval
        if minutes_since_eval < self.eval_interval_minutes:
            # Return cached signal's exit decision
            if self.exit_on_sell and self._current_signal.get('action') == 'sell':
                return True, 'AI Exit Signal'
            return False, None

        # Time to run inference
        self._bars_since_eval = 0
        self._last_eval_time = timestamp

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

    def create_detector(self, entry_option_price, option_type,
                         entry_stock_price=None, entry_delta=None,
                         statsbook=None):
        """
        Create a new OptionsExitDetector for a position.

        Args:
            entry_option_price: Option price at entry
            option_type: 'CALL' or 'PUT'
            entry_stock_price: Stock price at entry (for adaptive buffer)
            entry_delta: Option delta at entry (for adaptive buffer)
            statsbook: Statsbook DataFrame for this ticker (for adaptive buffer)

        Returns:
            OptionsExitDetector instance, or None if disabled.
        """
        if not self.enabled:
            return None
        return OptionsExitDetector(self.config, entry_option_price, option_type,
                                   entry_stock_price=entry_stock_price,
                                   entry_delta=entry_delta,
                                   statsbook=statsbook)


class OptionsExitDetector:
    """
    Per-position detector that tracks trailing TP, hard SL, entry favorability,
    and reversal detection.

    Trailing TP uses a continuous logarithmic scaling function so the stop
    ratchets smoothly with profit instead of jumping at fixed milestones:

        trail_tp_pct = base_floor + scale * ln(1 + profit_pct / norm)

    For high-risk entries an addon is layered on top:

        addon = addon_base + addon_scale * ln(1 + profit_pct / addon_norm)

    The trailing TP price is:

        tp_price = entry_price * (1 + trail_tp_pct / 100)

    It only moves up (for profit protection), never down.
    """

    def __init__(self, config, entry_option_price, option_type,
                 entry_stock_price=None, entry_delta=None, statsbook=None):
        self.entry_option_price = entry_option_price
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

        # Hard SL
        self.hard_sl_enabled = config.get('hard_sl_enabled', True)
        self.initial_sl_pct = config.get('initial_sl_pct', 20)
        self.hard_sl_price = entry_option_price * (1 - self.initial_sl_pct / 100)
        self.hard_sl_tighten_on_peak = config.get('hard_sl_tighten_on_peak', True)

        # Trailing SL parameters
        self.trail_tp_enabled = config.get('trail_tp_enabled', True)
        self.trail_activation_pct = config.get('trail_activation_pct', 10)
        # Cheap options have amplified % swings — require more profit before trailing
        cheap_threshold = config.get('trail_cheap_option_threshold', 0.40)
        if cheap_threshold and entry_option_price < cheap_threshold:
            self.trail_activation_pct *= 2
        self.trail_base_floor_pct = config.get('trail_base_floor_pct', 5)
        self.trail_early_floor_minutes = config.get('trail_early_floor_minutes', 5)
        self.trail_scale = config.get('trail_scale', 20.0)
        self.trail_norm = config.get('trail_norm', 40.0)

        # Adaptive buffer parameters
        self.trail_buffer_adaptive = config.get('trail_buffer_adaptive', True)
        self.trail_buffer_bars = config.get('trail_buffer_bars', 1.5)
        self.trail_buffer_min_pct = config.get('trail_buffer_min_pct', 5.0)
        self.trail_buffer_max_pct = config.get('trail_buffer_max_pct', 25.0)
        self.trail_buffer_decay_norm = config.get('trail_buffer_decay_norm', 50.0)

        # Compute adaptive buffer base from statsbook + delta
        self._adaptive_buffer_base = self._compute_adaptive_buffer_base(
            entry_stock_price, entry_delta, statsbook
        )

        # High-risk addon
        self.risk_addon_base = config.get('risk_addon_base', 2.0)
        self.risk_addon_scale = config.get('risk_addon_scale', 5.0)
        self.risk_addon_norm = config.get('risk_addon_norm', 30.0)

        # Entry favorability / RiskOutlook (config in minutes)
        self.confirmation_window_minutes = config.get('confirmation_window_minutes',
                                                       config.get('confirmation_window_bars', 10))
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)

        # EMA reversal
        self.ema_reversal_periods = config.get('ema_reversal_periods', [10, 21, 50])
        self.ema_reversal_sensitivity = config.get('ema_reversal_sensitivity', 2)

        # ATR-SL
        self.atr_sl_enabled = config.get('atr_sl_enabled', True)

        # ATR-SL Trend Gate: suppress Trail-TP while trend is intact
        self.atr_sl_trend_gate_enabled = config.get('atr_sl_trend_gate_enabled', False)
        self.atr_sl_trend_sideways_bars = config.get('atr_sl_trend_sideways_bars', 5)
        self._recent_highs = []    # Rolling window of stock highs for sideways detection
        self._recent_lows = []     # Rolling window of stock lows for sideways detection
        self.atr_sl_trend = None   # Current ATR-SL trend: 'Uptrend' / 'Downtrend' / 'Sideways'
        self.atr_sl_trend_gate_active = False  # True = Trail-TP suppressed by trend

        # MACD
        self.macd_enabled = config.get('macd_enabled', True)

        # Velocity deceleration exit (config in minutes)
        self.velocity_exit_enabled = config.get('velocity_exit_enabled', True)
        self.velocity_window = config.get('velocity_window', 5)
        self.velocity_decel_ratio = config.get('velocity_decel_ratio', 0.30)
        self.velocity_confirm_minutes = config.get('velocity_confirm_minutes',
                                                    config.get('velocity_confirm_bars', 2))
        self.velocity_min_profit_pct = config.get('velocity_min_profit_pct', 25)
        self.velocity_min_move = config.get('velocity_min_move', 0.0)

        # Auto-calibrate velocity_min_move from statsbook if set to 0
        if self.velocity_min_move <= 0 and statsbook is not None:
            try:
                median_hl = float(statsbook.loc['Median(H-L)', '5m'])
                if not np.isnan(median_hl) and median_hl > 0:
                    # Expect a "run" to average at least 1x median bar range per bar
                    self.velocity_min_move = median_hl
            except (KeyError, TypeError, ValueError):
                pass
        if self.velocity_min_move <= 0:
            self.velocity_min_move = 0.05  # Fallback: $0.05/bar minimum

        # State
        self.trailing_tp_price = None       # Current trailing TP price (None until activated)
        self.trailing_active = False
        self.highest_profit_pct = 0.0       # Watermark: best profit seen so far
        self.bar_count = 0                  # Legacy counter (kept for logging/display)
        self._entry_time = None             # Time-based tracking
        self.is_high_risk = False
        self._stock_prices = []             # Stock close history for velocity
        self._decel_count = 0               # Consecutive seconds of deceleration
        self._decel_start_time = None       # When deceleration started

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
    # Adaptive buffer computation
    # ------------------------------------------------------------------

    def _compute_adaptive_buffer_base(self, entry_stock_price, entry_delta, statsbook):
        """
        Compute the base adaptive buffer from Statsbook volatility and entry delta.

        Uses Median.Max(H-L) from the 5m timeframe to measure the typical peak
        bar range for this stock, then converts to option-percentage terms via
        delta.  This gives the expected per-bar option noise as a % of entry price.

        Returns:
            float: base buffer in option-% terms, or None if data unavailable.
        """
        if not self.trail_buffer_adaptive:
            return None
        if entry_stock_price is None or entry_delta is None or statsbook is None:
            return None
        try:
            median_max_hl = float(statsbook.loc['Median.Max(H-L)', '5m'])
        except (KeyError, TypeError, ValueError):
            return None
        if np.isnan(median_max_hl) or median_max_hl <= 0:
            return None

        # Per-bar option noise: how much the option moves (in % of entry price)
        # for a typical max bar range in the stock
        option_noise_pct = (abs(entry_delta) * median_max_hl
                            / self.entry_option_price * 100)

        # Scale by bar multiplier (tolerate N bars of max noise)
        base_buffer = option_noise_pct * self.trail_buffer_bars

        return max(self.trail_buffer_min_pct, min(base_buffer, self.trail_buffer_max_pct))

    def _get_buffer(self, profit_pct):
        """
        Get the trailing TP buffer for the current profit level.

        If adaptive buffer is available, it decays with profit so the trade
        has room to breathe at low profits and tightens at high profits.
        Falls back to trail_buffer_min_pct if no statsbook data.
        """
        if self._adaptive_buffer_base is not None:
            # Decay: buffer shrinks as profit grows
            decay_factor = 1 + profit_pct / self.trail_buffer_decay_norm
            return self._adaptive_buffer_base / decay_factor
        # Fallback: use minimum buffer
        return self.trail_buffer_min_pct

    # ------------------------------------------------------------------
    # ATR-SL trend classification
    # ------------------------------------------------------------------

    def _classify_atr_sl_trend(self, stock_price, stock_high, stock_low, atr_sl_value):
        """
        Classify the current ATR-SL trend state.

        For PUTS (inverse logic for CALLS):
          - Downtrend: stock_price < ATR-SL (good for puts, hold)
          - Uptrend:   stock_price lows > ATR-SL (bad for puts, sell signal)
          - Sideways:  ATR-SL is between the last N-bar high and low

        Returns:
            str: 'Uptrend', 'Downtrend', or 'Sideways'
        """
        if np.isnan(atr_sl_value) or np.isnan(stock_price):
            return self.atr_sl_trend  # Keep previous state

        # Track rolling highs/lows for sideways detection
        if not np.isnan(stock_high):
            self._recent_highs.append(stock_high)
        if not np.isnan(stock_low):
            self._recent_lows.append(stock_low)

        # Trim to window size
        window = self.atr_sl_trend_sideways_bars
        if len(self._recent_highs) > window:
            self._recent_highs = self._recent_highs[-window:]
        if len(self._recent_lows) > window:
            self._recent_lows = self._recent_lows[-window:]

        # Sideways: ATR-SL is between the recent N-bar high and low
        if self._recent_highs and self._recent_lows:
            recent_high = max(self._recent_highs)
            recent_low = min(self._recent_lows)
            if recent_low <= atr_sl_value <= recent_high:
                return 'Sideways'

        # Directional classification
        if stock_price > atr_sl_value:
            return 'Uptrend'    # Stock above ATR-SL = bullish
        else:
            return 'Downtrend'  # Stock below ATR-SL = bearish

    def _is_trend_favorable(self):
        """
        Check if the ATR-SL trend is favorable for holding the position.

        CALLS: Uptrend is favorable (stock above ATR-SL → hold)
        PUTS:  Downtrend is favorable (stock below ATR-SL → hold)
        Sideways: Not favorable (no clear trend → allow exit)

        Returns:
            bool: True if trend supports holding, False if exit is OK.
        """
        if self.atr_sl_trend is None:
            return False  # No data yet, don't gate

        if self.is_call:
            return self.atr_sl_trend == 'Uptrend'
        else:
            return self.atr_sl_trend == 'Downtrend'

    # ------------------------------------------------------------------
    # Velocity deceleration detection
    # ------------------------------------------------------------------

    def _check_velocity_exit(self, stock_price, profit_pct, ema_values, timestamp=None):
        """
        Detect stock price velocity deceleration for proactive peak exit.

        Tracks price changes (velocity).  When the stock was running strongly
        and velocity drops to a fraction of its recent average, it signals
        momentum exhaustion — exit before the crash.

        Returns:
            bool: True if velocity deceleration exit should fire.
        """
        if not self.velocity_exit_enabled:
            return False

        self._stock_prices.append(stock_price)

        # Need at least window + 1 bars to compute velocities
        if len(self._stock_prices) < self.velocity_window + 2:
            return False

        # Minimum profit gate
        if profit_pct < self.velocity_min_profit_pct:
            self._decel_count = 0
            self._decel_start_time = None
            return False

        # Compute velocities: close[i] - close[i-1] for recent bars
        prices = self._stock_prices
        window = self.velocity_window

        # Direction-aware: for calls, positive velocity = good; for puts, negative = good
        if self.is_call:
            velocities = [prices[-i] - prices[-i - 1] for i in range(1, window + 2)]
        else:
            velocities = [prices[-i - 1] - prices[-i] for i in range(1, window + 2)]
        velocities.reverse()  # Oldest to newest

        # Current velocity = last element; average = mean of previous N
        current_velocity = velocities[-1]
        avg_velocity = sum(velocities[:-1]) / len(velocities[:-1])

        # Stock must have been "running" (avg velocity above minimum move)
        if avg_velocity < self.velocity_min_move:
            self._decel_count = 0
            self._decel_start_time = None
            return False

        # Deceleration = still moving in the right direction, just much slower.
        # Negative velocity (actual reversal) is NOT deceleration — let trail-TP
        # handle outright reversals.  This targets the "stall before the crash".
        if current_velocity < 0:
            self._decel_count = 0
            self._decel_start_time = None
            return False

        # Check for deceleration: current velocity < ratio × average
        if current_velocity < self.velocity_decel_ratio * avg_velocity:
            self._decel_count += 1
            if self._decel_start_time is None and timestamp:
                self._decel_start_time = timestamp
        else:
            self._decel_count = 0
            self._decel_start_time = None
            return False

        # Confirmation: deceleration must persist for N minutes
        # In intra-minute mode, _decel_count tracks consecutive ticks of deceleration.
        # Convert to time-based: check if deceleration started >= N minutes ago.
        if timestamp and self._decel_start_time:
            decel_minutes = (timestamp - self._decel_start_time).total_seconds() / 60
        else:
            decel_minutes = self._decel_count  # fallback: 1 count ≈ 1 minute
        if decel_minutes < self.velocity_confirm_minutes:
            return False

        # Extended check: price should be above short EMAs (still in "run" territory)
        if ema_values:
            for period in [10, 21]:
                ema_val = ema_values.get(period, np.nan)
                if not np.isnan(ema_val):
                    if self.is_call and stock_price < ema_val:
                        return False  # Already reversed below EMA — let trail-TP handle
                    elif not self.is_call and stock_price > ema_val:
                        return False

        return True

    # ------------------------------------------------------------------
    # Trailing SL math
    # ------------------------------------------------------------------

    def _calculate_trail_tp_pct(self, profit_pct):
        """
        Continuous trailing TP as a percentage *above entry* to lock in.

        Returns the % of entry price that the SL should sit at.
        E.g. if entry = $2.00 and this returns 35, SL = $2.00 * 1.35 = $2.70.
        """
        if profit_pct < self.trail_activation_pct:
            return 0.0  # Not yet active

        # First N minutes: breakeven floor (0%) to give the trade room
        if self._minutes_elapsed <= self.trail_early_floor_minutes:
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

        # SL can never exceed current profit minus adaptive buffer
        buffer = self._get_buffer(profit_pct)
        sl_pct = min(sl_pct, profit_pct - buffer)

        return max(0.0, sl_pct)

    # ------------------------------------------------------------------
    # RiskOutlook — entry favorability assessment
    # ------------------------------------------------------------------

    def RiskOutlook(self, rsi, rsi_avg,
                    ema_values, stock_price, atr_sl_value,
                    macd_histogram, roc_30m, roc_day,
                    supertrend_direction, ewo, ewo_avg,
                    stoch_k=np.nan, stoch_d=np.nan):
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

        # --- 8. Stochastic overbought/oversold at entry ---
        if not np.isnan(stoch_k):
            if self.is_call and stoch_k >= 80:
                reasons.append(f'Stoch_OB({stoch_k:.0f})')
                risk_score += 2
            elif not self.is_call and stoch_k <= 20:
                reasons.append(f'Stoch_OS({stoch_k:.0f})')
                risk_score += 2
            elif self.is_call and stoch_k <= 20:
                risk_score -= 1  # Oversold = good for calls
            elif not self.is_call and stoch_k >= 80:
                risk_score -= 1  # Overbought = good for puts

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
    # Per-tick update (called 60x per minute during extrapolated simulation)
    # ------------------------------------------------------------------

    def update(self, option_price, stock_price, ema_values=None, timestamp=None,
               atr_sl_value=np.nan, stock_high=np.nan, stock_low=np.nan):
        """
        Per-tick update: check hard SL, update trailing TP, check EMA reversal.

        Called 60x per minute during extrapolated backtesting, once per minute in live.

        Args:
            option_price: Current estimated option price
            stock_price: Current stock price
            ema_values: dict mapping period -> EMA value (for reversal detection)
            timestamp: Current timestamp for time-based tracking
            atr_sl_value: Current ATR-SL value (for trend gate)
            stock_high: Current bar high (for sideways detection)
            stock_low: Current bar low (for sideways detection)

        Returns:
            (should_exit, exit_reason, state_dict)
            state_dict contains per-tick state for the databook.
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp
        ema_values = ema_values or {}

        # Compute minutes elapsed (time-based instead of bar-count)
        if timestamp and self._entry_time:
            self._minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            self._minutes_elapsed = self.bar_count  # fallback

        profit_pct = ((option_price - self.entry_option_price) / self.entry_option_price) * 100

        # Track confirmation window (time-based)
        if self._minutes_elapsed <= self.confirmation_window_minutes:
            self._confirmation_bars.append(option_price)
        elif self.confirmed == 'Pending' and len(self._confirmation_bars) >= 2:
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

        # --- ATR-SL trend classification ---
        if self.atr_sl_trend_gate_enabled and not np.isnan(atr_sl_value):
            self.atr_sl_trend = self._classify_atr_sl_trend(
                stock_price, stock_high, stock_low, atr_sl_value)
            self.atr_sl_trend_gate_active = self._is_trend_favorable()

        # --- Update profit watermark (needed before hard SL adjustment) ---
        if profit_pct > self.highest_profit_pct:
            self.highest_profit_pct = profit_pct

        # --- Hard stop loss (dynamic tightening before trailing TP activates) ---
        # If the option price hasn't yet hit the trailing activation threshold,
        # tighten the hard SL based on the peak gain seen so far.
        # Peak gain scales up to just under trail_activation_pct (9.99%).
        # HIGH risk entries double the peak gain effect for faster tightening.
        # E.g. buy at 100, peak at 105 → SL% shrinks 20→15%, hard SL 80→85
        #      HIGH risk same scenario → SL% shrinks 20→10%, hard SL 80→90
        if self.hard_sl_tighten_on_peak and not self.trailing_active and self.highest_profit_pct > 0:
            # Cap peak gain just under trailing activation to avoid overlap
            peak_gain = min(self.highest_profit_pct, self.trail_activation_pct - 0.01)

            # HIGH risk: double the peak gain effect
            if self.is_high_risk:
                peak_gain *= 2

            adjusted_sl_pct = self.initial_sl_pct - peak_gain
            adjusted_sl_pct = max(adjusted_sl_pct, 0.0)  # Never go above entry
            new_hard_sl = self.entry_option_price * (1 - adjusted_sl_pct / 100)
            # Ratchet: only tighten (move up), never loosen
            if new_hard_sl > self.hard_sl_price:
                self.hard_sl_price = new_hard_sl

        if self.hard_sl_enabled and option_price <= self.hard_sl_price:
            state = self._build_state(option_price, profit_pct)
            return True, 'Hard-SL', state

        # --- Trailing stop loss ---

        # Calculate where trailing TP should be based on watermark profit
        trail_tp_pct = self._calculate_trail_tp_pct(self.highest_profit_pct)

        if trail_tp_pct > 0:
            self.trailing_active = True
            new_trailing_tp = self.entry_option_price * (1 + trail_tp_pct / 100)

            # Ratchet: only move SL up, never down
            if self.trailing_tp_price is None or new_trailing_tp > self.trailing_tp_price:
                self.trailing_tp_price = new_trailing_tp

            # Check if price hit trailing TP
            # Trend gate: suppress Trail-TP exit while ATR-SL trend is favorable
            if self.trail_tp_enabled and option_price <= self.trailing_tp_price:
                if self.atr_sl_trend_gate_active:
                    # Trend is intact — hold the position, don't exit yet.
                    # The trailing TP price continues to ratchet up, so when
                    # the trend eventually breaks, the exit will capture the
                    # higher trailing level.
                    pass
                else:
                    state = self._build_state(option_price, profit_pct)
                    return True, 'Trail-TP', state

        # --- Velocity deceleration exit (proactive peak detection) ---
        if self._check_velocity_exit(stock_price, profit_pct, ema_values, timestamp):
            state = self._build_state(option_price, profit_pct)
            return True, 'Velocity-Peak', state

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
            'tp_trailing': self.trailing_tp_price if self.trailing_tp_price else np.nan,
            'sl_hard': self.hard_sl_price,
            'tp_risk_outlook': self.favorability,
            'tp_risk_reasons': self.favorability_reasons,
            'tp_trend_30m': self.trend_30m,
            'sl_ema_reversal': self.ema_reversal,
            'tp_confirmed': self.confirmed,
            'atr_sl_trend': self.atr_sl_trend,
            'atr_sl_trend_gate': self.atr_sl_trend_gate_active,
        }


# =============================================================================
# INTERNAL - Volume Climax Exit Strategy
# =============================================================================

class VolumeClimaxExit:
    """
    Factory for creating VolumeClimaxDetector instances.

    Detects volume exhaustion events: a sudden volume spike (Nx above
    rolling average) combined with a price reversal against the position.
    High-volume reversals are among the most reliable intraday exhaustion
    signals — institutions unloading or short-covering creates a volume
    climax that typically precedes a directional shift.

    Optional: Stochastic confirmation strengthens the signal by requiring
    the stochastic oscillator to be in the extreme zone matching the reversal.

    Config: BACKTEST_CONFIG['volume_climax_exit']
        {
            'enabled': True,
            'volume_lookback': 20,         # Bars for rolling avg volume
            'volume_multiplier': 3.0,      # Current bar volume must be >= Nx avg
            'min_profit_pct': 10,          # Minimum option profit %
            'min_hold_bars': 10,           # Minimum bars before checking
            'use_stochastic': False,       # Require stochastic extreme zone confirmation
            'stoch_overbought': 75,        # Stochastic overbought zone for CALLs
            'stoch_oversold': 25,          # Stochastic oversold zone for PUTs
        }
    """

    def __init__(self, config=None):
        vc_config = config or Config.get_setting('backtest', 'volume_climax_exit', {})
        self.enabled = vc_config.get('enabled', False)
        self.config = vc_config

    def create_detector(self, option_type):
        """Create a new VolumeClimaxDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return VolumeClimaxDetector(self.config, option_type)


class VolumeClimaxDetector:
    """
    Tracks volume state for a single position and detects climax reversals.

    Conditions (ALL must be met to trigger exit):
    1. Option profit >= min_profit_pct
    2. Position held >= min_hold_bars
    3. Current bar volume >= volume_multiplier * rolling avg volume
    4. Price reversal against position direction on the same bar:
       - CALLs: stock close < stock open (bearish bar)
       - PUTs: stock close > stock open (bullish bar)
    5. (Optional) Stochastic in extreme zone confirming reversal:
       - CALLs: Stochastic %K >= stoch_overbought (exhaustion at top)
       - PUTs: Stochastic %K <= stoch_oversold (exhaustion at bottom)
    """

    def __init__(self, config, option_type):
        self.volume_lookback = config.get('volume_lookback', 20)
        self.volume_multiplier = config.get('volume_multiplier', 3.0)
        self.min_profit_pct = config.get('min_profit_pct', 10)
        self.min_hold_minutes = config.get('min_hold_minutes', config.get('min_hold_bars', 10))
        self.use_stochastic = config.get('use_stochastic', False)
        self.stoch_overbought = config.get('stoch_overbought', 75)
        self.stoch_oversold = config.get('stoch_oversold', 25)
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

        self.volume_history = []
        self.bar_count = 0
        self._entry_time = None

    def update(self, pnl_pct, volume, stock_close, stock_open, stoch_k=np.nan, timestamp=None):
        """
        Check for volume climax exit signal.

        Args:
            pnl_pct: Current P&L percentage
            volume: Current bar volume
            stock_close: Current bar close price
            stock_open: Current bar open price
            stoch_k: Current Stochastic %K value (optional)
            timestamp: Current timestamp for time-based tracking

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        # Track volume history (append before check so current bar is included in future avg)
        if not np.isnan(volume) and volume > 0:
            self.volume_history.append(volume)

        # Need minimum time and volume history
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self.bar_count
        if minutes_elapsed < self.min_hold_minutes:
            return False, None

        if len(self.volume_history) < self.volume_lookback:
            return False, None

        # Skip if critical values are NaN
        if np.isnan(pnl_pct) or np.isnan(volume) or np.isnan(stock_close) or np.isnan(stock_open):
            return False, None

        # 1. Minimum profit threshold
        if pnl_pct < self.min_profit_pct:
            return False, None

        # 2. Volume spike: current bar vs rolling average (exclude current bar from avg)
        avg_volume = sum(self.volume_history[-self.volume_lookback - 1:-1]) / self.volume_lookback
        if avg_volume <= 0:
            return False, None

        if volume < self.volume_multiplier * avg_volume:
            return False, None

        # 3. Price reversal against position direction
        price_reversal = False
        if self.is_call and stock_close < stock_open:
            price_reversal = True
        elif not self.is_call and stock_close > stock_open:
            price_reversal = True

        if not price_reversal:
            return False, None

        # 4. Stochastic extreme zone confirmation (optional)
        if self.use_stochastic and not np.isnan(stoch_k):
            if self.is_call and stoch_k < self.stoch_overbought:
                return False, None
            elif not self.is_call and stoch_k > self.stoch_oversold:
                return False, None

        return True, 'Volume Climax'


# =============================================================================
# INTERNAL - Time Stop Strategy
# =============================================================================

class TimeStop:
    """
    Factory for creating TimeStopDetector instances.

    Exits stale positions that haven't moved meaningfully within a time
    window. Options lose value every minute via theta decay, so holding a
    position that isn't moving costs real money. Freeing capital from
    stagnant trades allows redeployment into better opportunities.

    Config: BACKTEST_CONFIG['time_stop']
        {
            'enabled': True,
            'max_minutes': 90,             # Exit if held longer than N minutes
            'min_profit_pct': 5,           # ... and profit is below this %
        }
    """

    def __init__(self, config=None):
        ts_config = config or Config.get_setting('backtest', 'time_stop', {})
        self.enabled = ts_config.get('enabled', False)
        self.config = ts_config

    def create_detector(self):
        """Create a new TimeStopDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return TimeStopDetector(self.config)


class TimeStopDetector:
    """
    Monitors elapsed time and profit for a single position.

    Exit condition: minutes_held >= max_minutes AND pnl_pct < min_profit_pct.

    A position that has moved well beyond min_profit_pct is clearly
    working and should be managed by trailing TP, not time-stopped.
    """

    def __init__(self, config):
        self.max_minutes = config.get('max_minutes', 90)
        self.min_profit_pct = config.get('min_profit_pct', 5)

    def update(self, pnl_pct, minutes_held):
        """
        Check for time stop exit signal.

        Args:
            pnl_pct: Current P&L percentage
            minutes_held: Minutes since position entry

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        if np.isnan(pnl_pct) or np.isnan(minutes_held):
            return False, None

        if minutes_held >= self.max_minutes and pnl_pct < self.min_profit_pct:
            return True, 'Time-Stop'

        return False, None


# =============================================================================
# INTERNAL - VWAP Cross Exit Strategy
# =============================================================================

class VWAPCrossExit:
    """
    Factory for creating VWAPCrossDetector instances.

    VWAP (Volume Weighted Average Price) is the benchmark used by
    institutional traders. Price crossing VWAP against the position
    direction signals that institutional flow has shifted:
      - CALLs: price drops below VWAP = sellers in control
      - PUTs:  price rises above VWAP = buyers in control

    Only triggers on a confirmed cross (was above, now below) to avoid
    false signals when price is oscillating right at VWAP.

    Config: BACKTEST_CONFIG['vwap_cross_exit']
        {
            'enabled': True,
            'min_profit_pct': 5,           # Minimum option profit %
            'min_hold_bars': 10,           # Minimum bars before checking
            'confirm_bars': 2,             # Bars price must stay on wrong side
        }
    """

    def __init__(self, config=None):
        vc_config = config or Config.get_setting('backtest', 'vwap_cross_exit', {})
        self.enabled = vc_config.get('enabled', False)
        self.config = vc_config

    def create_detector(self, option_type):
        """Create a new VWAPCrossDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return VWAPCrossDetector(self.config, option_type)


class VWAPCrossDetector:
    """
    Tracks price position relative to VWAP and detects confirmed crosses.

    Uses a confirmation window to avoid whipsaws: price must stay on the
    adverse side of VWAP for N consecutive bars before triggering exit.

    Conditions (ALL must be met):
    1. Option profit >= min_profit_pct
    2. Position held >= min_hold_bars
    3. Price was on favorable side of VWAP (established position)
    4. Price crossed to adverse side and stayed for confirm_bars
    """

    def __init__(self, config, option_type):
        self.min_profit_pct = config.get('min_profit_pct', 5)
        self.min_hold_minutes = config.get('min_hold_minutes', config.get('min_hold_bars', 10))
        self.confirm_minutes = config.get('confirm_minutes', config.get('confirm_bars', 2))
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

        self.bar_count = 0
        self._entry_time = None
        self.was_favorable = False       # Price has been on favorable side at least once
        self.adverse_bar_count = 0       # Consecutive bars/ticks on adverse side
        self._adverse_start_time = None  # When adverse streak started

    def update(self, pnl_pct, stock_price, vwap, timestamp=None):
        """
        Check for VWAP cross exit signal.

        Args:
            pnl_pct: Current P&L percentage
            stock_price: Current stock price (close)
            vwap: Current VWAP value
            timestamp: Current timestamp for time-based tracking

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        if np.isnan(pnl_pct) or np.isnan(stock_price) or np.isnan(vwap):
            return False, None

        # Determine which side of VWAP price is on
        if self.is_call:
            on_favorable_side = stock_price >= vwap
        else:
            on_favorable_side = stock_price <= vwap

        # Track if price has ever been on the favorable side
        if on_favorable_side:
            self.was_favorable = True
            self.adverse_bar_count = 0
            self._adverse_start_time = None
            return False, None

        # Price is on adverse side — count consecutive bars
        if self.was_favorable:
            self.adverse_bar_count += 1
            if self._adverse_start_time is None and timestamp:
                self._adverse_start_time = timestamp

        # Need minimum time held
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self.bar_count
        if minutes_elapsed < self.min_hold_minutes:
            return False, None

        # Need minimum profit
        if pnl_pct < self.min_profit_pct:
            return False, None

        # Check confirmation: must be on adverse side for N consecutive minutes
        if timestamp and self._adverse_start_time:
            adverse_minutes = (timestamp - self._adverse_start_time).total_seconds() / 60
        else:
            adverse_minutes = self.adverse_bar_count
        if self.was_favorable and adverse_minutes >= self.confirm_minutes:
            return True, 'VWAP Cross'

        return False, None


# =============================================================================
# INTERNAL - Supertrend Flip Exit Strategy
# =============================================================================

class SupertrendFlipExit:
    """
    Factory for creating SupertrendFlipDetector instances.

    The Supertrend indicator produces a binary direction signal (1 = bullish,
    -1 = bearish) based on ATR-derived support/resistance bands. When the
    direction flips from favorable to adverse mid-trade, the underlying
    trend has mechanically reversed — a strong exit signal.

    Unlike oscillator-based strategies (RSI, EWO) that detect exhaustion,
    Supertrend flips confirm that price has already broken through a
    volatility-adjusted support/resistance level, making false signals
    less frequent.

    Config: BACKTEST_CONFIG['supertrend_flip_exit']
        {
            'enabled': True,
            'min_profit_pct': 5,           # Minimum option profit %
            'min_hold_bars': 5,            # Minimum bars before checking
            'confirm_bars': 1,             # Bars adverse direction must persist
        }
    """

    def __init__(self, config=None):
        st_config = config or Config.get_setting('backtest', 'supertrend_flip_exit', {})
        self.enabled = st_config.get('enabled', False)
        self.config = st_config

    def create_detector(self, option_type):
        """Create a new SupertrendFlipDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return SupertrendFlipDetector(self.config, option_type)


class SupertrendFlipDetector:
    """
    Tracks Supertrend direction changes for a single position.

    Favorable direction:
      - CALLs want direction = 1 (bullish / uptrend)
      - PUTs  want direction = -1 (bearish / downtrend)

    Conditions (ALL must be met to trigger exit):
    1. Position held >= min_hold_bars
    2. Option profit >= min_profit_pct
    3. Direction was favorable at some point (established trend)
    4. Direction flipped to adverse and stayed for confirm_bars
    """

    def __init__(self, config, option_type):
        self.min_profit_pct = config.get('min_profit_pct', 5)
        self.min_hold_minutes = config.get('min_hold_minutes', config.get('min_hold_bars', 5))
        self.confirm_minutes = config.get('confirm_minutes', config.get('confirm_bars', 1))
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

        # Favorable direction for this position type
        self.favorable_direction = 1.0 if self.is_call else -1.0

        self.bar_count = 0
        self._entry_time = None
        self.was_favorable = False       # Direction was favorable at some point
        self.adverse_bar_count = 0       # Consecutive bars in adverse direction
        self._adverse_start_time = None  # When adverse streak started

    def update(self, pnl_pct, supertrend_direction, timestamp=None):
        """
        Check for Supertrend flip exit signal.

        Args:
            pnl_pct: Current P&L percentage
            supertrend_direction: Current Supertrend direction (1.0 or -1.0)
            timestamp: Current timestamp for time-based tracking

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        if np.isnan(pnl_pct) or np.isnan(supertrend_direction):
            return False, None

        # Track if direction is currently favorable
        is_favorable = supertrend_direction == self.favorable_direction

        if is_favorable:
            self.was_favorable = True
            self.adverse_bar_count = 0
            self._adverse_start_time = None
            return False, None

        # Direction is adverse — count consecutive bars
        if self.was_favorable:
            self.adverse_bar_count += 1
            if self._adverse_start_time is None and timestamp:
                self._adverse_start_time = timestamp

        # Need minimum time held
        if timestamp and self._entry_time:
            minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
        else:
            minutes_elapsed = self.bar_count
        if minutes_elapsed < self.min_hold_minutes:
            return False, None

        # Need minimum profit
        if pnl_pct < self.min_profit_pct:
            return False, None

        # Check confirmation: must be adverse for N consecutive minutes
        if timestamp and self._adverse_start_time:
            adverse_minutes = (timestamp - self._adverse_start_time).total_seconds() / 60
        else:
            adverse_minutes = self.adverse_bar_count
        if self.was_favorable and adverse_minutes >= self.confirm_minutes:
            return True, 'Supertrend Flip'

        return False, None


# =============================================================================
# INTERNAL - MarketTrend Exit Strategy
# =============================================================================

class MarketTrend:
    """
    Factory for creating MarketTrendDetector instances.

    Evaluates whether the ticker's VWAP-based trend (-1/0/+1) supports
    holding the current position, and whether SPY's trend aligns.

    PUTs:
      - Bearish region (-1) = favorable → hold
      - Sideways (0) = warning → possible reversal
      - Bullish (+1) = reversal confirmed → sell signal
      - Entry in bullish region = high-risk, need reversal or exit

    CALLs (inverse):
      - Bullish region (+1) = favorable → hold
      - Sideways (0) = warning → possible reversal
      - Bearish (-1) = reversal confirmed → sell signal
      - Entry in bearish region = high-risk, need reversal or exit

    SPY divergence: if ticker & SPY were trending together and SPY turns,
    evaluate profitability and consider exit (stocks trail SPY movement).

    Config: BACKTEST_CONFIG['market_trend']
        {
            'enabled': True,
            'exit_enabled': False,          # False = blue X flag only
            'high_risk_grace_bars': 10,     # Bars to wait for reversal on bad entry
            'spy_diverge_profit_pct': 5,    # Min profit for SPY-divergence exit
        }
    """

    def __init__(self, config=None):
        mt_config = config or Config.get_setting('backtest', 'market_trend', {})
        self.enabled = mt_config.get('enabled', False)
        self.exit_enabled = mt_config.get('exit_enabled', False)
        self.config = mt_config

    def create_detector(self, option_type):
        """Create a new MarketTrendDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return MarketTrendDetector(self.config, option_type)


class MarketTrendDetector:
    """
    Per-position detector that tracks ticker trend and SPY alignment.

    State machine:
      - 'Hold':     Trend is favorable for the contract type
      - 'Warning':  Trend shifted to sideways (possible reversal)
      - 'Sell':     Trend reversed against position → exit signal
      - 'HighRisk': Entered against the trend, waiting for reversal

    SPY divergence is tracked separately: if both were trending together
    and SPY turns first, the assumption is the ticker will follow.
    """

    def __init__(self, config, option_type):
        self.is_call = option_type.upper() in ['CALL', 'CALLS', 'C']
        self.exit_enabled = config.get('exit_enabled', False)
        self.high_risk_grace_minutes = config.get('high_risk_grace_minutes',
                                                   config.get('high_risk_grace_bars', 10))
        self.spy_diverge_profit_pct = config.get('spy_diverge_profit_pct', 5)

        # State
        self.bar_count = 0
        self._entry_time = None
        self.entry_ticker_trend = None   # Ticker trend at entry
        self.entry_spy_trend = None      # SPY trend at entry
        self.prev_spy_trend = None       # Previous bar's SPY trend
        self.state = None                # Hold / Warning / Sell / HighRisk
        self.high_risk = False           # Entered against the trend

    def _favorable_trend(self):
        """Return the trend value that is favorable for this contract type."""
        return 1 if self.is_call else -1

    def _adverse_trend(self):
        """Return the trend value that is adverse for this contract type."""
        return -1 if self.is_call else 1

    def update(self, ticker_trend, spy_trend, pnl_pct, timestamp=None):
        """
        Evaluate MarketTrend exit signal.

        Args:
            ticker_trend: Current ticker trend (-1, 0, +1)
            spy_trend: Current SPY trend (-1, 0, +1)
            pnl_pct: Current P&L percentage
            timestamp: Current timestamp for time-based tracking

        Returns:
            (should_exit, exit_reason, state): Tuple.
                should_exit: True if position should close (respects exit_enabled).
                exit_reason: String reason or None.
                state: Current detector state string (Hold/Warning/Sell/HighRisk/SPY-Warning).
        """
        self.bar_count += 1
        if timestamp and self._entry_time is None:
            self._entry_time = timestamp

        if np.isnan(ticker_trend) or np.isnan(spy_trend):
            return False, None, self.state

        favorable = self._favorable_trend()
        adverse = self._adverse_trend()

        # Record entry conditions on first bar
        if self.entry_ticker_trend is None:
            self.entry_ticker_trend = ticker_trend
            self.prev_spy_trend = spy_trend
            self.entry_spy_trend = spy_trend

            # Determine if this is a high-risk entry (entered against the trend)
            # PUT entered in bullish, or CALL entered in bearish
            if ticker_trend == adverse:
                self.high_risk = True
                self.state = 'HighRisk'
            elif ticker_trend == favorable:
                self.state = 'Hold'
            else:
                self.state = 'Hold'  # Sideways entry is neutral

            return False, None, self.state

        # --- High-risk entry logic ---
        # Entered against the trend: need a reversal to favorable or exit
        if self.high_risk:
            if ticker_trend == favorable:
                # Reversed to favorable — no longer high risk, transition to Hold
                self.high_risk = False
                self.state = 'Hold'
                self.prev_spy_trend = spy_trend
                return False, None, self.state

            if ticker_trend == 0:
                # Moved to sideways — progress toward reversal, still risky
                self.state = 'HighRisk'
                self.prev_spy_trend = spy_trend
                return False, None, self.state

            # Still in adverse trend
            if timestamp and self._entry_time:
                minutes_elapsed = (timestamp - self._entry_time).total_seconds() / 60
            else:
                minutes_elapsed = self.bar_count
            if minutes_elapsed > self.high_risk_grace_minutes:
                # Grace period expired, no reversal → sell
                self.state = 'Sell'
                self.prev_spy_trend = spy_trend
                if self.exit_enabled:
                    return True, 'MarketTrend-HighRisk', self.state
                return False, None, self.state

            self.state = 'HighRisk'
            self.prev_spy_trend = spy_trend
            return False, None, self.state

        # --- Normal entry logic (entered in favorable or sideways) ---

        # Check ticker trend state
        if ticker_trend == favorable:
            self.state = 'Hold'
        elif ticker_trend == 0:
            # Sideways → warning (possible reversal)
            self.state = 'Warning'
        elif ticker_trend == adverse:
            # Full reversal confirmed → sell signal
            self.state = 'Sell'
            self.prev_spy_trend = spy_trend
            if self.exit_enabled:
                return True, 'MarketTrend-Reversal', self.state
            return False, None, self.state

        # --- SPY divergence logic ---
        # If ticker and SPY were both trending in our favorable direction
        # and SPY now turns, the ticker is likely to follow.
        spy_turned = (
            self.prev_spy_trend == favorable
            and spy_trend != favorable
        )

        if spy_turned and ticker_trend == favorable and not np.isnan(pnl_pct):
            # SPY shifted but ticker hasn't yet — evaluate exit if profitable
            if pnl_pct >= self.spy_diverge_profit_pct:
                self.state = 'SPY-Warning'
                self.prev_spy_trend = spy_trend
                if self.exit_enabled:
                    return True, 'MarketTrend-SPY', self.state
                return False, None, self.state

        self.prev_spy_trend = spy_trend
        return False, None, self.state


# =============================================================================
# INTERNAL - Peak-Peak Exit Strategy
# =============================================================================

class PeakPeakExit:
    """
    Factory for creating PeakPeakDetector instances.

    Captures a specific missed opportunity pattern: after the option reaches
    a peak price, if at least min_hold_minutes pass and the option price
    returns to that peak (or higher), the detector **engages**.

    Once engaged, the detector tracks the price upward — letting it run
    as long as it keeps increasing. When the price starts decreasing from
    the engaged high, exit.

    This improves on a simple double-top exit by allowing continuation
    moves to play out while still capturing the reversal.

    Config: BACKTEST_CONFIG['peak_peak_exit']
        {
            'enabled': True,
            'min_hold_minutes': 10,        # Minimum minutes between peak and re-touch
            'min_profit_pct': 5,           # Minimum option profit % to consider exit
        }
    """

    def __init__(self, config=None):
        pp_config = config or Config.get_setting('backtest', 'peak_peak_exit', {})
        self.enabled = pp_config.get('enabled', False)
        self.config = pp_config

    def create_detector(self):
        """Create a new PeakPeakDetector for a position. Returns None if disabled."""
        if not self.enabled:
            return None
        return PeakPeakDetector(self.config)


class PeakPeakDetector:
    """
    Tracks option price peaks and detects return-to-peak after a time delay.

    Logic:
    1. Track the maximum option price seen since entry (the "peak").
    2. Once price drops below the peak, the peak is "confirmed" (real).
    3. If at least min_hold_minutes have elapsed since the confirmed peak,
       and the option price returns to the peak level (or higher), the
       detector becomes **engaged**.
    4. While engaged, continue tracking the price upward (new running high).
       As long as price keeps increasing, keep following it.
    5. When price starts decreasing from the engaged high, exit.

    This catches the "second touch" of a peak and lets the price run
    higher if momentum continues, only exiting when it turns back down.

    Config: BACKTEST_CONFIG['peak_peak_exit']
        {
            'enabled': True,
            'min_hold_minutes': 10,        # Minimum minutes between peak and re-touch
            'min_profit_pct': 5,           # Minimum option profit % to consider exit
        }
    """

    def __init__(self, config):
        self.min_hold_minutes = config.get('min_hold_minutes', 10)
        self.min_profit_pct = config.get('min_profit_pct', 5)

        # Peak tracking
        self.peak_price = None          # Highest option price seen
        self.peak_time = None           # Timestamp when peak was set
        self.peak_confirmed = False     # True once price has moved below peak (peak is "real")
        self.entry_price = None         # Set on first update

        # Engaged state: activated when price re-touches the confirmed peak
        self.engaged = False            # True once peak is re-touched
        self.engaged_high = None        # Highest price seen while engaged

    def update(self, option_price, pnl_pct, timestamp):
        """
        Check for Peak-Peak exit signal.

        Args:
            option_price: Current option contract price
            pnl_pct: Current P&L percentage
            timestamp: Current timestamp

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        if np.isnan(option_price) or np.isnan(pnl_pct) or timestamp is None:
            return False, None

        # Initialize on first call
        if self.entry_price is None:
            self.entry_price = option_price
            self.peak_price = option_price
            self.peak_time = timestamp
            return False, None

        # --- Engaged mode: price already re-touched the peak ---
        if self.engaged:
            if option_price > self.engaged_high:
                # Price still increasing — keep tracking
                self.engaged_high = option_price
                return False, None
            else:
                # Price started decreasing from engaged high — exit
                return True, 'Peak-Peak'

        # --- Pre-engaged: tracking peaks and waiting for re-touch ---

        # Update peak if we have a new high (before confirmation)
        if option_price > self.peak_price:
            if self.peak_confirmed and pnl_pct >= self.min_profit_pct:
                # Price returned to or exceeded the confirmed peak —
                # check if enough time has passed since that peak
                minutes_since_peak = (timestamp - self.peak_time).total_seconds() / 60
                if minutes_since_peak >= self.min_hold_minutes:
                    # Engage: start tracking the upward run
                    self.engaged = True
                    self.engaged_high = option_price
                    return False, None

            # Update peak regardless (track the true high)
            self.peak_price = option_price
            self.peak_time = timestamp
            self.peak_confirmed = False
            return False, None

        # Price is below peak — confirm that a peak has formed
        if not self.peak_confirmed and self.peak_price > self.entry_price:
            self.peak_confirmed = True

        # Check if price is returning to the confirmed peak level (at or above)
        if self.peak_confirmed and self.peak_time is not None:
            minutes_since_peak = (timestamp - self.peak_time).total_seconds() / 60
            if minutes_since_peak >= self.min_hold_minutes:
                if option_price >= self.peak_price and pnl_pct >= self.min_profit_pct:
                    # Engage: start tracking the upward run
                    self.engaged = True
                    self.engaged_high = option_price
                    return False, None

        return False, None
