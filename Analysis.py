"""
Analysis.py - Technical Analysis

Module Goal: Process all math here (RSI, MACD, VPOC, VWAP, EWO, etc.)

================================================================================
INTERNAL - Indicator Calculations
================================================================================
"""

import pandas as pd
import numpy as np


def EMA(df, column='close', period=30):
    """
    Calculate Exponential Moving Average.

    Args:
        df: DataFrame with price data
        column: Column name to calculate EMA on (default: 'close')
        period: EMA period in bars (default: 30 for 30-minute EMA on 1-min data)

    Returns:
        Series with EMA values
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    return df[column].ewm(span=period, adjust=False).mean()


def VWAP(df, price_col='close', volume_col='volume'):
    """
    Calculate Volume Weighted Average Price.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Resets at market open each day for intraday calculation.

    Args:
        df: DataFrame with price and volume data
        price_col: Column name for price (default: 'close')
        volume_col: Column name for volume (default: 'volume')

    Returns:
        Series with VWAP values
    """
    if price_col not in df.columns or volume_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    df = df.copy()

    # Calculate typical price (high + low + close) / 3 if available, else use close
    if 'high' in df.columns and 'low' in df.columns:
        typical_price = (df['high'] + df['low'] + df[price_col]) / 3
    else:
        typical_price = df[price_col]

    # Handle zero or missing volume
    volume = df[volume_col].fillna(0).replace(0, np.nan)

    # Price * Volume
    pv = typical_price * volume

    # Group by date for intraday VWAP (resets each day)
    if hasattr(df.index, 'date'):
        # DatetimeIndex - group by date
        date_groups = df.index.date
        cumulative_pv = pv.groupby(date_groups).cumsum()
        cumulative_vol = volume.groupby(date_groups).cumsum()
    else:
        # Fallback: continuous VWAP
        cumulative_pv = pv.cumsum()
        cumulative_vol = volume.cumsum()

    # Calculate VWAP
    vwap = cumulative_pv / cumulative_vol

    return vwap


def EWO(df, column='close', fast_period=5, slow_period=35):
    """
    Calculate Elliott Wave Oscillator.

    EWO = EMA(fast) - EMA(slow)

    The EWO helps identify wave patterns and momentum:
    - Positive EWO: Bullish momentum
    - Negative EWO: Bearish momentum
    - Zero crossings: Potential trend changes

    Args:
        df: DataFrame with price data
        column: Column name to calculate EWO on (default: 'close')
        fast_period: Fast EMA period (default: 5)
        slow_period: Slow EMA period (default: 35)

    Returns:
        Series with EWO values
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

    return ema_fast - ema_slow


def true_price(stock_price, stock_high, stock_low):
    """Calculate True Price as the average of high, low, and stock price at each interval."""
    if pd.notna(stock_high) and pd.notna(stock_low):
        return (stock_high + stock_low + stock_price) / 3
    return stock_price


def RSI(df, column='close', period=14):
    """
    Calculate Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period

    RSI interpretation:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI = 50: Neutral momentum

    Args:
        df: DataFrame with price data
        column: Column name to calculate RSI on (default: 'close')
        period: RSI lookback period (default: 14)

    Returns:
        Series with RSI values (0-100 scale)
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    # Calculate price changes
    delta = df[column].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate exponential moving average of gains and losses
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_losses is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    rsi = rsi.fillna(50)  # Neutral when no data

    return rsi


def Supertrend(df, atr_period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.

    Supertrend uses ATR (Average True Range) to create dynamic support/resistance
    bands around the price. The indicator flips between upper and lower bands
    based on trend direction.

    - When price closes above the upper band: Uptrend (bullish)
    - When price closes below the lower band: Downtrend (bearish)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        atr_period: Period for ATR calculation (default: 10)
        multiplier: ATR multiplier for band width (default: 3.0)

    Returns:
        tuple: (supertrend Series, direction Series)
            - supertrend: the Supertrend line values
            - direction: 1 for uptrend (bullish), -1 for downtrend (bearish)
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range and ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=atr_period, adjust=False).mean()

    # HL2 (midpoint)
    hl2 = (high + low) / 2

    # Basic bands
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Initialize final bands and direction
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        # Final upper band: use previous final upper if current basic upper is lower
        # and previous close was above previous final upper
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Final lower band: use previous final lower if current basic lower is higher
        # and previous close was below previous final lower
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    # Determine direction and supertrend line
    # Start with initial direction based on first valid close vs bands
    direction.iloc[0] = 1  # Assume uptrend initially

    for i in range(1, len(df)):
        prev_dir = direction.iloc[i - 1]

        if prev_dir == 1:  # Was uptrend
            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1  # Flip to downtrend
            else:
                direction.iloc[i] = 1
        else:  # Was downtrend
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1  # Flip to uptrend
            else:
                direction.iloc[i] = -1

        # Supertrend line follows lower band in uptrend, upper band in downtrend
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = final_lower.iloc[i]
        else:
            supertrend.iloc[i] = final_upper.iloc[i]

    # Set first value
    supertrend.iloc[0] = final_lower.iloc[0] if direction.iloc[0] == 1 else final_upper.iloc[0]

    return supertrend, direction


def IchimokuCloud(df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    Calculate Ichimoku Cloud indicator.

    The Ichimoku Cloud (Ichimoku Kinko Hyo) is a comprehensive indicator that defines
    support/resistance, trend direction, momentum, and trading signals.

    Components:
    - Tenkan-sen (Conversion Line): Midpoint of highest high and lowest low over tenkan_period
    - Kijun-sen (Base Line): Midpoint of highest high and lowest low over kijun_period
    - Senkou Span A (Leading Span A): Average of Tenkan and Kijun, displaced forward
    - Senkou Span B (Leading Span B): Midpoint of highest high and lowest low over senkou_b_period, displaced forward
    - Chikou Span (Lagging Span): Current close displaced backward (not included - only useful visually)

    Cloud interpretation:
    - Price above cloud: Bullish trend
    - Price below cloud: Bearish trend
    - Price inside cloud: Consolidation/no trend
    - Span A above Span B: Bullish cloud (green)
    - Span A below Span B: Bearish cloud (red)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        tenkan_period: Period for Tenkan-sen / Conversion Line (default: 9)
        kijun_period: Period for Kijun-sen / Base Line (default: 26)
        senkou_b_period: Period for Senkou Span B (default: 52)
        displacement: Forward displacement for Senkou spans (default: 26)

    Returns:
        tuple: (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b)
            - tenkan_sen: Conversion Line values
            - kijun_sen: Base Line values
            - senkou_span_a: Leading Span A (displaced forward by displacement periods)
            - senkou_span_b: Leading Span B (displaced forward by displacement periods)
    """
    empty = pd.Series(index=df.index, dtype=float)
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    high = df['high']
    low = df['low']

    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 over tenkan_period
    tenkan_sen = (high.rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  low.rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2

    # Kijun-sen (Base Line): (highest high + lowest low) / 2 over kijun_period
    kijun_sen = (high.rolling(window=kijun_period, min_periods=kijun_period).max() +
                 low.rolling(window=kijun_period, min_periods=kijun_period).min()) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, displaced forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 over senkou_b_period, displaced forward
    senkou_span_b = ((high.rolling(window=senkou_b_period, min_periods=senkou_b_period).max() +
                      low.rolling(window=senkou_b_period, min_periods=senkou_b_period).min()) / 2).shift(displacement)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


def MACD(df, column='close', fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Interpretation:
    - Histogram > 0: Bullish momentum
    - Histogram < 0: Bearish momentum
    - Histogram crossing zero: Momentum shift

    Args:
        df: DataFrame with price data
        column: Column name to calculate MACD on (default: 'close')
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    empty = pd.Series(index=df.index, dtype=float)
    if column not in df.columns:
        return empty.copy(), empty.copy(), empty.copy()

    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def ROC(df, column='close', period=30):
    """
    Calculate Rate of Change (price momentum).

    ROC = (current - N bars ago) / N bars ago * 100

    Positive ROC = upward momentum, Negative = downward momentum.
    Magnitude indicates trend strength.

    Args:
        df: DataFrame with price data
        column: Column name to calculate ROC on (default: 'close')
        period: Lookback period (default: 30 for 30-min trend on 1-min data)

    Returns:
        Series with ROC values (percentage)
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    shifted = df[column].shift(period)
    roc = ((df[column] - shifted) / shifted) * 100

    return roc


def ATR_SL(df, atr_period=5, hhv_period=10, multiplier=2.5):
    """
    ATR Trailing Stoploss indicator.

    Translated from Pine Script (ceyhun, MPL 2.0).

    Creates a trailing stop loss line based on ATR:
    1. Compute base_level = high - (multiplier * ATR)
    2. Take the highest base_level over hhv_period bars
    3. For the first 15 bars, use close as the stop level

    When close > ATR-SL: uptrend (price above stop)
    When close < ATR-SL: downtrend (price below stop)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        atr_period: ATR calculation period (default: 5)
        hhv_period: Highest high value lookback period (default: 10)
        multiplier: ATR multiplier (default: 2.5)

    Returns:
        Series with ATR Trailing Stoploss values
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(index=df.index, dtype=float)

    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR using Wilder's smoothing (RMA): alpha = 1/period
    atr = true_range.ewm(alpha=1.0 / atr_period, adjust=False).mean()

    # Base value: high - multiplier * ATR
    base = high - multiplier * atr

    # Highest base value over hhv_period bars (Prev in Pine Script)
    hhv = base.rolling(window=hhv_period, min_periods=1).max()

    # Build trailing stop: first 15 bars use close, then use hhv
    # When close > hhv and close > previous close, update to current hhv
    # Otherwise hold the previous value (ratcheting behavior)
    ts = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i < 15:
            # First 15 bars: use close
            ts.iloc[i] = close.iloc[i]
        else:
            prev_ts = ts.iloc[i - 1]
            if close.iloc[i] > hhv.iloc[i] and close.iloc[i] > close.iloc[i - 1]:
                ts.iloc[i] = hhv.iloc[i]
            else:
                ts.iloc[i] = max(prev_ts, hhv.iloc[i]) if close.iloc[i] > prev_ts else hhv.iloc[i]

    return ts


def add_indicators(df, ema_periods=None, ewo_fast=5, ewo_slow=35, ewo_avg_period=15,
                   rsi_period=14, rsi_avg_period=10,
                   supertrend_period=10, supertrend_multiplier=3.0,
                   ichimoku_tenkan=9, ichimoku_kijun=26, ichimoku_senkou_b=52, ichimoku_displacement=26,
                   atr_sl_period=5, atr_sl_hhv=10, atr_sl_multiplier=2.5,
                   macd_fast=12, macd_slow=26, macd_signal=9,
                   roc_period=30):
    """
    Add all standard indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data (must have 'close', 'volume' columns)
        ema_periods: List of EMA periods (default: [10, 21, 50, 100, 200])
        ewo_fast: Fast period for EWO (default: 5)
        ewo_slow: Slow period for EWO (default: 35)
        ewo_avg_period: Period for EWO rolling average (default: 15 for 15-min avg on 1-min data)
        rsi_period: Period for RSI calculation (default: 14)
        rsi_avg_period: Period for RSI rolling average (default: 10 for 10-min avg on 1-min data)
        supertrend_period: ATR period for Supertrend (default: 10)
        supertrend_multiplier: ATR multiplier for Supertrend bands (default: 3.0)
        ichimoku_tenkan: Tenkan-sen period (default: 9)
        ichimoku_kijun: Kijun-sen period (default: 26)
        ichimoku_senkou_b: Senkou Span B period (default: 52)
        ichimoku_displacement: Forward displacement for cloud spans (default: 26)
        atr_sl_period: ATR period for ATR-SL indicator (default: 5)
        atr_sl_hhv: HHV lookback period for ATR-SL (default: 10)
        atr_sl_multiplier: ATR multiplier for ATR-SL (default: 2.5)
        macd_fast: MACD fast EMA period (default: 12)
        macd_slow: MACD slow EMA period (default: 26)
        macd_signal: MACD signal line EMA period (default: 9)
        roc_period: Rate of Change lookback period (default: 30)

    Returns:
        DataFrame with added indicator columns:
        - ema_10, ema_21, ema_50, ema_100, ema_200: EMAs at various periods
        - vwap: Volume Weighted Average Price
        - vwap_ema_avg: (VWAP + EMA_21 + High) / 3
        - emavwap: (EMA_21 + VWAP) / 2
        - ewo: Elliott Wave Oscillator
        - ewo_15min_avg: 15-minute rolling average of EWO
        - rsi: Relative Strength Index (0-100 scale)
        - rsi_10min_avg: 10-minute rolling average of RSI
        - supertrend: Supertrend line value
        - supertrend_direction: 1 (uptrend/bullish) or -1 (downtrend/bearish)
        - ichimoku_tenkan: Tenkan-sen (Conversion Line)
        - ichimoku_kijun: Kijun-sen (Base Line)
        - ichimoku_senkou_a: Senkou Span A (Leading Span A)
        - ichimoku_senkou_b: Senkou Span B (Leading Span B)
        - atr_sl: ATR Trailing Stoploss
        - macd_line: MACD line (fast EMA - slow EMA)
        - macd_signal: MACD signal line
        - macd_histogram: MACD histogram (line - signal)
        - roc: Rate of Change (price momentum %)
    """
    if ema_periods is None:
        ema_periods = [10, 21, 50, 100, 200]

    df = df.copy()

    # Calculate true price column: (high + low + close) / 3
    # All indicators use true price instead of raw stock close price
    if 'high' in df.columns and 'low' in df.columns:
        df['_true_price'] = (df['high'] + df['low'] + df['close']) / 3
    else:
        df['_true_price'] = df['close']

    # Add EMAs at multiple periods (based on true price)
    for period in ema_periods:
        df[f'ema_{period}'] = EMA(df, column='_true_price', period=period)

    # Add VWAP (based on true price)
    df['vwap'] = VWAP(df, price_col='_true_price', volume_col='volume')

    # Add EWO (based on true price)
    df['ewo'] = EWO(df, column='_true_price', fast_period=ewo_fast, slow_period=ewo_slow)

    # Add EWO 15-minute rolling average (simple moving average over ewo_avg_period bars)
    df['ewo_15min_avg'] = df['ewo'].rolling(window=ewo_avg_period, min_periods=1).mean()

    # Add RSI (based on true price)
    df['rsi'] = RSI(df, column='_true_price', period=rsi_period)

    # Add RSI 10-minute rolling average (simple moving average over rsi_avg_period bars)
    df['rsi_10min_avg'] = df['rsi'].rolling(window=rsi_avg_period, min_periods=1).mean()

    # Add VWAP-EMA-High Average: (VWAP + EMA_21 + High) / 3
    ema_ref = df.get('ema_21', df.get(f'ema_{ema_periods[1]}', df[f'ema_{ema_periods[0]}']))
    df['vwap_ema_avg'] = (df['vwap'] + ema_ref + df['high']) / 3

    # Add EMAVWAP: (EMA_21 + VWAP) / 2
    df['emavwap'] = (ema_ref + df['vwap']) / 2

    # Add Supertrend
    df['supertrend'], df['supertrend_direction'] = Supertrend(
        df, atr_period=supertrend_period, multiplier=supertrend_multiplier
    )

    # Add Ichimoku Cloud
    df['ichimoku_tenkan'], df['ichimoku_kijun'], df['ichimoku_senkou_a'], df['ichimoku_senkou_b'] = IchimokuCloud(
        df, tenkan_period=ichimoku_tenkan, kijun_period=ichimoku_kijun,
        senkou_b_period=ichimoku_senkou_b, displacement=ichimoku_displacement
    )

    # Add ATR Trailing Stoploss
    df['atr_sl'] = ATR_SL(df, atr_period=atr_sl_period, hhv_period=atr_sl_hhv, multiplier=atr_sl_multiplier)

    # Add MACD (based on true price)
    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(
        df, column='_true_price', fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
    )

    # Add Rate of Change / Price Momentum (based on true price)
    df['roc'] = ROC(df, column='_true_price', period=roc_period)

    # Clean up temporary column
    df = df.drop(columns=['_true_price'])

    return df


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Strategy.py, Test.py
"""

import Config
from scipy.stats import norm
import math


# =============================================================================
# INTERNAL - Black-Scholes Options Pricing
# =============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a CALL option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        float: Call option price
    """
    if T <= 0:
        return max(0, S - K)  # At expiration, intrinsic value only

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(0.01, call_price)


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a PUT option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        float: Put option price
    """
    if T <= 0:
        return max(0, K - S)  # At expiration, intrinsic value only

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.01, put_price)


def black_scholes_price(S, K, T, r, sigma, option_type='CALL'):
    """
    Calculate Black-Scholes price for an option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'CALL' or 'PUT'

    Returns:
        float: Option price
    """
    if option_type.upper() in ['CALL', 'CALLS', 'C']:
        return black_scholes_call(S, K, T, r, sigma)
    else:
        return black_scholes_put(S, K, T, r, sigma)


def calculate_greeks(S, K, T, r, sigma, option_type='CALL'):
    """
    Calculate option Greeks using Black-Scholes model.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'CALL' or 'PUT'

    Returns:
        dict: Dictionary containing Delta, Gamma, Theta, Vega, Rho
    """
    is_call = option_type.upper() in ['CALL', 'CALLS', 'C']

    if T <= 0:
        # At expiration, delta is 1 or -1 if ITM, else 0
        if is_call:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Delta
    if is_call:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1  # Negative for puts

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    # Theta (per day, divide by 365)
    if is_call:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (per 1% change in volatility)
    vega = S * math.sqrt(T) * norm.pdf(d1) / 100

    # Rho (per 1% change in interest rate)
    if is_call:
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def estimate_option_price_bs(stock_price, strike, option_type, days_to_expiry,
                              entry_price=None, entry_stock_price=None,
                              entry_days_to_expiry=None,
                              volatility=None, risk_free_rate=None):
    """
    Estimate option price using Black-Scholes model with offset calibration.

    When entry data is provided, calibrates BS to the actual market price:
        offset = entry_price - BS(entry_stock_price, T_entry)
        price  = BS(current_stock_price, T_current) + offset

    The offset is computed once using T_entry (the time to expiry at entry)
    so that theta decay in the BS(current) term is reflected naturally.
    days_to_expiry should be fractional (intraday precision) so 0DTE
    options retain time value instead of collapsing to intrinsic-only.

    Args:
        stock_price: Current stock price
        strike: Option strike price
        option_type: 'CALL' or 'PUT'
        days_to_expiry: Fractional days until expiration
        entry_price: Original entry price (optional, for offset calibration)
        entry_stock_price: Stock price at entry (optional, for offset calibration)
        entry_days_to_expiry: Days to expiry at entry time (optional, for fixed offset)
        volatility: Implied volatility (default from config)
        risk_free_rate: Risk-free rate (default from config)

    Returns:
        float: Estimated option price
    """
    # Get defaults from config
    options_config = Config.get_config('analysis').get('options', {})
    if volatility is None:
        volatility = options_config.get('default_volatility', 0.30)
    if risk_free_rate is None:
        risk_free_rate = options_config.get('risk_free_rate', 0.05)
    min_price = options_config.get('min_option_price', 0.01)

    # Convert days to years
    T = max(0, days_to_expiry) / 365

    # Calculate theoretical Black-Scholes price
    theoretical_price = black_scholes_price(stock_price, strike, T, risk_free_rate, volatility, option_type)

    # If we have entry data, calibrate BS to actual market price using offset
    # offset = entry_price - BS_theoretical(entry_stock, T_entry) anchors the model
    # so that at entry the simulated price matches the signal cost exactly,
    # and subsequent bars move by full BS dynamics (delta + gamma + theta)
    if entry_price and entry_stock_price and entry_price > 0:
        # Use entry-time T for offset so theta decay isn't canceled out
        T_entry = max(0, entry_days_to_expiry) / 365 if entry_days_to_expiry is not None else T
        entry_theoretical = black_scholes_price(
            entry_stock_price, strike, T_entry, risk_free_rate, volatility, option_type
        )
        offset = entry_price - entry_theoretical
        calibrated_price = theoretical_price + offset

        return max(min_price, calibrated_price)

    return max(min_price, theoretical_price)


# Export functions for use by other modules
__all__ = ['EMA', 'VWAP', 'EWO', 'true_price', 'RSI', 'MACD', 'ROC',
           'Supertrend', 'IchimokuCloud', 'ATR_SL',
           'add_indicators', 'black_scholes_call', 'black_scholes_put', 'black_scholes_price',
           'calculate_greeks', 'estimate_option_price_bs']
