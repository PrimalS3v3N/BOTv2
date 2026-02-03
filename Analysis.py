"""
Analysis.py - Technical Analysis and Indicators Module

FUNCTIONALITY:
This module provides technical analysis calculations for trading decisions.
All functions accept optional parameters that override Config.ANALYSIS_CONFIG defaults.

SUPPORTED MODULES:
- Main.py: Real-time indicator calculations for live trading
- Test.py: Backtesting indicator calculations and options pricing
- Strategy.py: Technical indicators for entry/exit decisions
- Dashboard.py: Indicator data for visualization

MAINTENANCE INSTRUCTIONS:
=========================
When adding new features:
1. Add function to appropriate section below based on which module uses it
2. Update the section header comment with new function name
3. If function uses Config values, document which Config keys it reads
4. Add function to calculate_all_indicators() if it should be auto-calculated

When removing features:
1. Remove function and update section header
2. Check if function is used in calculate_all_indicators() and remove there too
3. Remove any Config keys that are no longer used

Section Organization:
- CORE INDICATORS: RSI, MACD, Moving Averages - used by all modules
- VOLUME INDICATORS: VWAP, VPOC, EWO - volume-based analysis
- VOLATILITY INDICATORS: ATR, Bollinger Bands, Stochastic - volatility measures
- TREND INDICATORS: SuperTrend, Support/Resistance, Trend ID - trend analysis
- MOMENTUM INDICATORS: Momentum, ROC - price momentum
- AGGREGATE FUNCTIONS: calculate_all_indicators, get_signal_strength
- OPTIONS PRICING: Black-Scholes, Greeks - used by Test.py for backtesting
"""

####### REQUIRED IMPORTS
import pandas as Panda
import numpy as np
import datetime as dt
from datetime import timedelta
import time
import math

# Import centralized config
import Config


# =============================================================================
# CORE INDICATORS (Main.py, Test.py, Strategy.py)
# =============================================================================
# Functions: sma, ema, rsi, macd
# Config keys: sma_period, ema_fast_period, rsi_period, macd_fast/slow/signal


def sma(prices, period=None):
    # SCOPE: Calculate Simple Moving Average
    # Returns: Series with SMA values
    # Default period from Config.ANALYSIS_CONFIG['sma_period']

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('sma_period', 20)

    if len(prices) < period:
        return None

    sma_values = prices.rolling(window=period).mean()
    return sma_values


def ema(prices, period=None):
    # SCOPE: Calculate Exponential Moving Average
    # Returns: Series with EMA values
    # Default period from Config.ANALYSIS_CONFIG['ema_fast_period']

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('ema_fast_period', 12)

    if len(prices) < period:
        return None

    ema_values = prices.ewm(span=period, adjust=False).mean()
    return ema_values


def rsi(prices, period=None):
    # SCOPE: Calculate Relative Strength Index
    # Returns: Series with RSI values (0-100)
    # Default period from Config.ANALYSIS_CONFIG['rsi_period']

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('rsi_period', 14)

    if len(prices) < period + 1:
        return None

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(prices, fast=None, slow=None, signal=None):
    # SCOPE: Calculate MACD (Moving Average Convergence Divergence)
    # Returns: Dictionary with {macd_line, signal_line, histogram}
    # Default settings from Config.ANALYSIS_CONFIG

    if fast is None:
        fast = Config.ANALYSIS_CONFIG.get('macd_fast', 12)
    if slow is None:
        slow = Config.ANALYSIS_CONFIG.get('macd_slow', 26)
    if signal is None:
        signal = Config.ANALYSIS_CONFIG.get('macd_signal', 9)

    if len(prices) < slow:
        return None

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

    return result


# =============================================================================
# VOLUME INDICATORS (Main.py, Test.py, Strategy.py)
# =============================================================================
# Functions: vwap, vpoc, ewo
# Config keys: vpoc_bins, ewo_fast, ewo_slow


def vwap(high, low, close, volume):
    # SCOPE: Calculate Volume Weighted Average Price
    # Returns: Series with VWAP values

    if len(high) < 1 or len(low) < 1 or len(close) < 1 or len(volume) < 1:
        return None

    typical_price = (high + low + close) / 3
    vwap_values = (typical_price * volume).cumsum() / volume.cumsum()

    return vwap_values


def vpoc(close, volume, bins=None):
    # SCOPE: Calculate Volume Point of Control (most traded price)
    # Returns: Float price with highest volume
    # Default bins from Config.ANALYSIS_CONFIG['vpoc_bins']

    if bins is None:
        bins = Config.ANALYSIS_CONFIG.get('vpoc_bins', 10)

    if len(close) < 1 or len(volume) < 1:
        return None

    min_price = close.min()
    max_price = close.max()

    # Create bins for price levels
    price_bins = np.linspace(min_price, max_price, bins)

    # Sum volume for each price bin
    bin_volumes = np.zeros(bins)
    for i, price in enumerate(close):
        bin_idx = min(int((price - min_price) / (max_price - min_price) * (bins - 1)), bins - 1)
        bin_volumes[bin_idx] += volume.iloc[i] if hasattr(volume, 'iloc') else volume[i]

    # Find bin with highest volume
    vpoc_bin = np.argmax(bin_volumes)
    vpoc_price = price_bins[vpoc_bin]

    return vpoc_price


def ewo(close, fast_period=None, slow_period=None):
    # SCOPE: Calculate Elliott Wave Oscillator (EWO)
    # Returns: Series with EWO values
    # Default periods from Config.ANALYSIS_CONFIG

    if fast_period is None:
        fast_period = Config.ANALYSIS_CONFIG.get('ewo_fast', 5)
    if slow_period is None:
        slow_period = Config.ANALYSIS_CONFIG.get('ewo_slow', 35)

    if len(close) < slow_period:
        return None

    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    ewo_values = ema_fast - ema_slow

    return ewo_values


# =============================================================================
# VOLATILITY INDICATORS (Main.py, Test.py, Strategy.py)
# =============================================================================
# Functions: atr, bollinger_bands, stochastic
# Config keys: atr_period, bb_period, bb_std_dev, stoch_period, stoch_k_smooth


def bollinger_bands(prices, period=None, std_dev=None):
    # SCOPE: Calculate Bollinger Bands
    # Returns: Dictionary with {upper, middle, lower}
    # Default settings from Config.ANALYSIS_CONFIG

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('bb_period', 20)
    if std_dev is None:
        std_dev = Config.ANALYSIS_CONFIG.get('bb_std_dev', 2)

    if len(prices) < period:
        return None

    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    result = {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

    return result


def atr(high, low, close, period=None):
    # SCOPE: Calculate Average True Range for volatility
    # Returns: Series with ATR values
    # Default period from Config.ANALYSIS_CONFIG['atr_period']

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('atr_period', 14)

    if len(high) < period or len(low) < period or len(close) < period:
        return None

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = Panda.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.rolling(window=period).mean()

    return atr_values


def stochastic(high, low, close, period=None):
    # SCOPE: Calculate Stochastic Oscillator
    # Returns: Dictionary with {%K, %D}
    # Default period from Config.ANALYSIS_CONFIG['stoch_period']

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('stoch_period', 14)

    if len(high) < period or len(low) < period or len(close) < period:
        return None

    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    k_smooth = Config.ANALYSIS_CONFIG.get('stoch_k_smooth', 3)
    d_percent = k_percent.rolling(window=k_smooth).mean()

    result = {
        'k': k_percent,
        'd': d_percent
    }

    return result


# =============================================================================
# TREND INDICATORS (Main.py, Test.py, Strategy.py)
# =============================================================================
# Functions: supertrend, calculate_support_resistance, identify_trend
# Config keys: supertrend_period, supertrend_multiplier, support_resistance_lookback,
#              trend_fast_period, trend_slow_period


def supertrend(high, low, close, period=None, multiplier=None):
    # SCOPE: Calculate SuperTrend indicator
    # Returns: Dictionary with {supertrend, direction, upper_band, lower_band}
    # direction: 1 for uptrend (green), -1 for downtrend (red)
    # Default settings from Config.ANALYSIS_CONFIG

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('supertrend_period', 10)
    if multiplier is None:
        multiplier = Config.ANALYSIS_CONFIG.get('supertrend_multiplier', 3.0)

    if len(high) < period or len(low) < period or len(close) < period:
        return None

    # Calculate ATR
    atr_values = atr(high, low, close, period)

    if atr_values is None:
        return None

    # Calculate basic bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr_values)
    lower_band = hl_avg - (multiplier * atr_values)

    # Initialize final bands series
    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    supertrend_values = Panda.Series(index=close.index, dtype=float)
    direction = Panda.Series(index=close.index, dtype=float)

    # Calculate SuperTrend
    for i in range(period, len(close)):
        # Final Upper Band
        if i > period:
            if upper_band.iloc[i] < final_upper_band.iloc[i-1] or close.iloc[i-1] > final_upper_band.iloc[i-1]:
                final_upper_band.iloc[i] = upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]

            # Final Lower Band
            if lower_band.iloc[i] > final_lower_band.iloc[i-1] or close.iloc[i-1] < final_lower_band.iloc[i-1]:
                final_lower_band.iloc[i] = lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]

        # SuperTrend direction
        if i > period:
            if supertrend_values.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] <= final_upper_band.iloc[i]:
                supertrend_values.iloc[i] = final_upper_band.iloc[i]
                direction.iloc[i] = -1  # Downtrend
            elif supertrend_values.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] > final_upper_band.iloc[i]:
                supertrend_values.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = 1  # Uptrend
            elif supertrend_values.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] >= final_lower_band.iloc[i]:
                supertrend_values.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = 1  # Uptrend
            elif supertrend_values.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] < final_lower_band.iloc[i]:
                supertrend_values.iloc[i] = final_upper_band.iloc[i]
                direction.iloc[i] = -1  # Downtrend
            else:
                supertrend_values.iloc[i] = supertrend_values.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        else:
            # Initial value
            if close.iloc[i] <= final_upper_band.iloc[i]:
                supertrend_values.iloc[i] = final_upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend_values.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = 1

    result = {
        'supertrend': supertrend_values,
        'direction': direction,
        'upper_band': final_upper_band,
        'lower_band': final_lower_band
    }

    return result


def calculate_support_resistance(close, lookback=None):
    # SCOPE: Calculate support and resistance levels
    # Returns: Dictionary with {support, resistance, pivot}
    # Default lookback from Config.ANALYSIS_CONFIG

    if lookback is None:
        lookback = Config.ANALYSIS_CONFIG.get('support_resistance_lookback', 20)

    if len(close) < lookback:
        return None

    recent_close = close.tail(lookback)
    high = close.max()
    low = close.min()

    resistance = high
    support = low
    pivot = (high + low + recent_close.iloc[-1]) / 3

    result = {
        'support': support,
        'resistance': resistance,
        'pivot': pivot
    }

    return result


def identify_trend(close, fast_period=None, slow_period=None):
    # SCOPE: Identify current trend using moving averages
    # Returns: String {UPTREND, DOWNTREND, SIDEWAYS}
    # Default periods from Config.ANALYSIS_CONFIG

    if fast_period is None:
        fast_period = Config.ANALYSIS_CONFIG.get('trend_fast_period', 10)
    if slow_period is None:
        slow_period = Config.ANALYSIS_CONFIG.get('trend_slow_period', 20)

    if len(close) < slow_period:
        return None

    fast_ma = close.ewm(span=fast_period, adjust=False).mean().iloc[-1]
    slow_ma = close.ewm(span=slow_period, adjust=False).mean().iloc[-1]
    current_price = close.iloc[-1]

    if fast_ma > slow_ma > current_price:
        return 'UPTREND'
    elif fast_ma < slow_ma < current_price:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'


# =============================================================================
# MOMENTUM INDICATORS (Main.py, Test.py, Strategy.py)
# =============================================================================
# Functions: calculate_momentum, calculate_roc
# Config keys: momentum_period, roc_period


def calculate_momentum(close, period=None):
    # SCOPE: Calculate momentum (price change over period)
    # Returns: Series with momentum values
    # Default period from Config.ANALYSIS_CONFIG

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('momentum_period', 10)

    if len(close) < period:
        return None

    momentum = close - close.shift(period)
    return momentum


def calculate_roc(close, period=None):
    # SCOPE: Calculate Rate of Change
    # Returns: Series with ROC percentages
    # Default period from Config.ANALYSIS_CONFIG

    if period is None:
        period = Config.ANALYSIS_CONFIG.get('roc_period', 10)

    if len(close) < period:
        return None

    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc


# =============================================================================
# BAR ANALYSIS (Main.py, Strategy.py)
# =============================================================================
# Functions: analyze_bar
# Config keys: None


def analyze_bar(high, low, close, volume, prev_close=None):
    # SCOPE: Analyze single bar characteristics
    # Returns: Dictionary with bar properties

    if prev_close is None:
        prev_close = close

    bar_analysis = {
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'range': high - low,
        'body': abs(close - prev_close),
        'close_location': (close - low) / (high - low) if high != low else 0.5,
        'is_bullish': close > prev_close,
        'is_bearish': close < prev_close,
        'volume_above_average': False  # Set based on comparison
    }

    return bar_analysis


# =============================================================================
# AGGREGATE FUNCTIONS (Main.py, Test.py, Strategy.py, Dashboard.py)
# =============================================================================
# Functions: calculate_all_indicators, get_signal_strength
# Config keys: min_bars_required (and all indicator-specific keys)


def calculate_all_indicators(ohlcv_df):
    # SCOPE: Calculate all technical indicators for a dataset
    # ohlcv_df: DataFrame with columns [open, high, low, close, volume]
    # Returns: DataFrame with all indicator columns
    # Uses default parameters from Config.ANALYSIS_CONFIG

    min_bars = Config.ANALYSIS_CONFIG.get('min_bars_required', 20)

    if ohlcv_df is None or len(ohlcv_df) < min_bars:
        return None

    result_df = ohlcv_df.copy()

    # Get config settings
    cfg = Config.ANALYSIS_CONFIG

    # Moving Averages (use Config defaults)
    result_df['sma_20'] = sma(ohlcv_df['close'])  # Uses Config default
    result_df['ema_12'] = ema(ohlcv_df['close'])  # Uses Config default
    result_df['ema_20'] = ohlcv_df['close'].ewm(span=20, adjust=False).mean()
    result_df['ema_30'] = ohlcv_df['close'].ewm(span=30, adjust=False).mean()

    # Momentum Indicators (use Config defaults)
    result_df['rsi_14'] = rsi(ohlcv_df['close'])  # Uses Config default
    result_df['momentum'] = calculate_momentum(ohlcv_df['close'])  # Uses Config default

    # MACD (use Config defaults)
    macd_result = macd(ohlcv_df['close'])  # Uses Config defaults
    if macd_result:
        result_df['macd'] = macd_result['macd']
        result_df['macd_signal'] = macd_result['signal']
        result_df['macd_histogram'] = macd_result['histogram']

    # Volatility Indicators (use Config defaults)
    result_df['atr_14'] = atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
    bb_result = bollinger_bands(ohlcv_df['close'])  # Uses Config defaults
    if bb_result:
        result_df['bb_upper'] = bb_result['upper']
        result_df['bb_middle'] = bb_result['middle']
        result_df['bb_lower'] = bb_result['lower']

    # Volume Indicators
    result_df['vwap'] = vwap(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], ohlcv_df['volume'])
    result_df['ewo'] = ewo(ohlcv_df['close'])  # Uses Config defaults

    # Stochastic (use Config defaults)
    stoch_result = stochastic(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
    if stoch_result:
        result_df['stoch_k'] = stoch_result['k']
        result_df['stoch_d'] = stoch_result['d']

    # SuperTrend (use Config defaults)
    supertrend_result = supertrend(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
    if supertrend_result:
        result_df['supertrend'] = supertrend_result['supertrend']
        result_df['supertrend_direction'] = supertrend_result['direction']
        result_df['supertrend_upper'] = supertrend_result['upper_band']
        result_df['supertrend_lower'] = supertrend_result['lower_band']

    return result_df


def get_signal_strength(indicators_df):
    # SCOPE: Summarize indicator signals into strength score
    # Returns: Integer -100 to 100 (negative=sell, positive=buy)

    if indicators_df is None or len(indicators_df) == 0:
        return 0

    score = 0
    last_row = indicators_df.iloc[-1]

    # RSI signals (0-100 scale)
    rsi_val = last_row.get('rsi_14', 50)
    if rsi_val < 30:
        score += 20  # Oversold = buy signal
    elif rsi_val > 70:
        score -= 20  # Overbought = sell signal

    # MACD signals
    if not np.isnan(last_row.get('macd_histogram', 0)):
        macd_hist = last_row['macd_histogram']
        if macd_hist > 0:
            score += 15
        else:
            score -= 15

    # Price vs Moving Average
    if not np.isnan(last_row.get('sma_20', 0)) and last_row.get('close', 0) > last_row.get('sma_20', 0):
        score += 10
    else:
        score -= 10

    # Momentum
    if not np.isnan(last_row.get('momentum', 0)) and last_row['momentum'] > 0:
        score += 10
    else:
        score -= 10

    # Clamp score
    return max(-100, min(100, score))


# =============================================================================
# OPTIONS PRICING ESTIMATOR (Test.py)
# =============================================================================
# Black-Scholes model for options pricing with Greeks calculations
# Used by Test.py for backtesting option trades
#
# Functions: black_scholes_price, calculate_greeks, estimate_option_price,
#            estimate_option_price_simple, calculate_moneyness, implied_volatility_estimate
# Config keys: options.risk_free_rate, options.default_volatility, options.min_option_price


def _norm_cdf(x):
    # SCOPE: Standard normal cumulative distribution function
    # Uses error function approximation (no scipy dependency)
    # Returns: Probability P(Z <= x) for standard normal Z

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x):
    # SCOPE: Standard normal probability density function
    # Returns: PDF value at x for standard normal distribution

    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def black_scholes_price(stock_price, strike, time_to_expiry, risk_free_rate=None,
                        volatility=None, option_type='CALL'):
    # SCOPE: Calculate theoretical option price using Black-Scholes model
    # Returns: Dictionary with {price, d1, d2} or None if invalid inputs
    #
    # Args:
    #   stock_price: Current underlying stock price (S)
    #   strike: Option strike price (K)
    #   time_to_expiry: Time to expiration in years (T)
    #   risk_free_rate: Annual risk-free rate (default 0.05 = 5%)
    #   volatility: Annual implied volatility (default 0.30 = 30%)
    #   option_type: 'CALL' or 'PUT'

    # Get defaults from config or use built-in defaults
    options_cfg = Config.ANALYSIS_CONFIG.get('options', {})
    if risk_free_rate is None:
        risk_free_rate = options_cfg.get('risk_free_rate', 0.05)
    if volatility is None:
        volatility = options_cfg.get('default_volatility', 0.30)

    # Validate inputs
    if stock_price <= 0 or strike <= 0 or time_to_expiry < 0 or volatility <= 0:
        return None

    # Handle expiration (T=0)
    if time_to_expiry == 0:
        if option_type == 'CALL':
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)
        return {'price': intrinsic, 'd1': None, 'd2': None}

    # Calculate d1 and d2
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(stock_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    # Calculate option price
    if option_type == 'CALL':
        price = stock_price * _norm_cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2)
    else:  # PUT
        price = strike * np.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2) - stock_price * _norm_cdf(-d1)

    return {
        'price': max(0.01, price),
        'd1': d1,
        'd2': d2
    }


def calculate_greeks(stock_price, strike, time_to_expiry, risk_free_rate=None,
                     volatility=None, option_type='CALL'):
    # SCOPE: Calculate option Greeks (delta, gamma, theta, vega, rho)
    # Returns: Dictionary with all Greeks or None if invalid inputs
    #
    # Greeks:
    #   delta: Price change per $1 stock move
    #   gamma: Delta change per $1 stock move
    #   theta: Daily time decay (negative for long positions)
    #   vega: Price change per 1% volatility change
    #   rho: Price change per 1% interest rate change

    # Get defaults from config or use built-in defaults
    options_cfg = Config.ANALYSIS_CONFIG.get('options', {})
    if risk_free_rate is None:
        risk_free_rate = options_cfg.get('risk_free_rate', 0.05)
    if volatility is None:
        volatility = options_cfg.get('default_volatility', 0.30)

    # Validate inputs
    if stock_price <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return None

    # Calculate d1 and d2
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(stock_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    # Common terms
    pdf_d1 = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    exp_rt = np.exp(-risk_free_rate * time_to_expiry)

    # Delta
    if option_type == 'CALL':
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1

    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (stock_price * volatility * sqrt_t)

    # Theta (annualized, divide by 365 for daily)
    theta_term1 = -(stock_price * pdf_d1 * volatility) / (2 * sqrt_t)
    if option_type == 'CALL':
        theta = theta_term1 - risk_free_rate * strike * exp_rt * cdf_d2
    else:
        theta = theta_term1 + risk_free_rate * strike * exp_rt * _norm_cdf(-d2)
    theta_daily = theta / 365  # Convert to daily theta

    # Vega (per 1% vol change, so divide by 100)
    vega = stock_price * sqrt_t * pdf_d1 / 100

    # Rho (per 1% rate change, so divide by 100)
    if option_type == 'CALL':
        rho = strike * time_to_expiry * exp_rt * cdf_d2 / 100
    else:
        rho = -strike * time_to_expiry * exp_rt * _norm_cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'theta_daily': theta_daily,
        'vega': vega,
        'rho': rho,
        'd1': d1,
        'd2': d2
    }


def estimate_option_price(stock_price, strike, option_type, days_to_expiry,
                          entry_price=None, entry_stock_price=None,
                          volatility=None, risk_free_rate=None):
    # SCOPE: Estimate option price for backtesting
    # Returns: Dictionary with {price, delta, pnl_estimate} or estimated price float
    #
    # If entry_price and entry_stock_price provided, uses delta-based tracking
    # Otherwise calculates theoretical Black-Scholes price
    #
    # Args:
    #   stock_price: Current underlying price
    #   strike: Option strike price
    #   option_type: 'CALL' or 'PUT'
    #   days_to_expiry: Days until expiration
    #   entry_price: Original option entry price (for tracking)
    #   entry_stock_price: Stock price at entry (for delta tracking)
    #   volatility: Implied volatility (default 0.30)
    #   risk_free_rate: Risk-free rate (default 0.05)

    # Get defaults from config
    options_cfg = Config.ANALYSIS_CONFIG.get('options', {})
    if volatility is None:
        volatility = options_cfg.get('default_volatility', 0.30)
    if risk_free_rate is None:
        risk_free_rate = options_cfg.get('risk_free_rate', 0.05)

    # Convert days to years
    time_to_expiry = max(0, days_to_expiry) / 365

    # Calculate Greeks for current price
    greeks = calculate_greeks(
        stock_price=stock_price,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type
    )

    # Calculate theoretical price
    bs_result = black_scholes_price(
        stock_price=stock_price,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type
    )

    theoretical_price = bs_result['price'] if bs_result else 0.01
    delta = greeks['delta'] if greeks else 0.5

    # If tracking from entry, use delta approximation for more accurate P&L
    if entry_price is not None and entry_stock_price is not None:
        # Calculate stock price change
        stock_change = stock_price - entry_stock_price

        # Entry Greeks (for more accurate delta at entry)
        entry_days = days_to_expiry + 1  # Approximate, assume 1 day passed
        entry_greeks = calculate_greeks(
            stock_price=entry_stock_price,
            strike=strike,
            time_to_expiry=entry_days / 365,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type
        )
        entry_delta = entry_greeks['delta'] if entry_greeks else delta

        # Use average delta for price change approximation
        avg_delta = (entry_delta + delta) / 2

        # Calculate option price change using delta
        # Delta gives us $ change in option per $1 change in stock
        option_change = avg_delta * stock_change

        # Also account for theta decay (time value loss)
        theta_daily = greeks['theta_daily'] if greeks else 0
        days_held = 1  # Approximate
        theta_impact = theta_daily * days_held

        # Estimated price = entry + delta impact + theta decay
        estimated_price = entry_price + option_change + theta_impact

        # Floor at intrinsic value
        if option_type == 'CALL':
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)

        estimated_price = max(intrinsic, estimated_price, 0.01)

        return {
            'price': estimated_price,
            'theoretical_price': theoretical_price,
            'delta': delta,
            'gamma': greeks['gamma'] if greeks else 0,
            'theta_daily': theta_daily,
            'vega': greeks['vega'] if greeks else 0,
            'stock_change': stock_change,
            'option_change': option_change,
            'intrinsic': intrinsic
        }

    # Simple case: just return theoretical price
    return {
        'price': theoretical_price,
        'theoretical_price': theoretical_price,
        'delta': delta,
        'gamma': greeks['gamma'] if greeks else 0,
        'theta_daily': greeks['theta_daily'] if greeks else 0,
        'vega': greeks['vega'] if greeks else 0,
        'intrinsic': max(0, stock_price - strike) if option_type == 'CALL' else max(0, strike - stock_price)
    }


def estimate_option_price_simple(stock_price, strike, option_type, days_to_expiry,
                                  entry_price=None, volatility=None):
    # SCOPE: Simple option price estimator (returns float only)
    # Convenience wrapper for Test.py compatibility
    # Returns: Estimated option price as float

    result = estimate_option_price(
        stock_price=stock_price,
        strike=strike,
        option_type=option_type,
        days_to_expiry=days_to_expiry,
        entry_price=entry_price,
        volatility=volatility
    )

    if isinstance(result, dict):
        return result.get('price', 0.01)
    return max(0.01, result)


def calculate_moneyness(stock_price, strike, option_type):
    # SCOPE: Calculate option moneyness
    # Returns: Dictionary with {moneyness, status, pct_otm_itm}
    #
    # Moneyness = S/K for calls, K/S for puts
    # Status: 'ITM', 'ATM', 'OTM'

    if option_type == 'CALL':
        moneyness = stock_price / strike
        pct = (stock_price - strike) / strike * 100
    else:  # PUT
        moneyness = strike / stock_price
        pct = (strike - stock_price) / strike * 100

    # Determine status (within 2% of strike = ATM)
    if abs(pct) <= 2:
        status = 'ATM'
    elif pct > 0:
        status = 'ITM'
    else:
        status = 'OTM'

    return {
        'moneyness': moneyness,
        'status': status,
        'pct_from_strike': pct
    }


def implied_volatility_estimate(option_price, stock_price, strike, days_to_expiry,
                                 option_type='CALL', risk_free_rate=None, max_iterations=100):
    # SCOPE: Estimate implied volatility from option price using Newton-Raphson
    # Returns: Estimated IV as decimal (e.g., 0.30 for 30%)
    #
    # Uses bisection method for stability

    options_cfg = Config.ANALYSIS_CONFIG.get('options', {})
    if risk_free_rate is None:
        risk_free_rate = options_cfg.get('risk_free_rate', 0.05)

    time_to_expiry = max(0.001, days_to_expiry / 365)

    # Starting bounds for volatility
    vol_low = 0.01
    vol_high = 3.0
    tolerance = 0.0001

    for _ in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2

        bs_result = black_scholes_price(
            stock_price=stock_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=vol_mid,
            option_type=option_type
        )

        if bs_result is None:
            return None

        price_diff = bs_result['price'] - option_price

        if abs(price_diff) < tolerance:
            return vol_mid

        if price_diff > 0:
            vol_high = vol_mid
        else:
            vol_low = vol_mid

    # Return best estimate after max iterations
    return (vol_low + vol_high) / 2
