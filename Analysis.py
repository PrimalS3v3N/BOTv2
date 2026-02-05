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


def add_indicators(df, ema_period=30, ewo_fast=5, ewo_slow=35, ewo_avg_period=15, rsi_period=14):
    """
    Add all standard indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data (must have 'close', 'volume' columns)
        ema_period: Period for EMA calculation (default: 30)
        ewo_fast: Fast period for EWO (default: 5)
        ewo_slow: Slow period for EWO (default: 35)
        ewo_avg_period: Period for EWO rolling average (default: 15 for 15-min avg on 1-min data)
        rsi_period: Period for RSI calculation (default: 14)

    Returns:
        DataFrame with added indicator columns:
        - ema_30: 30-period EMA
        - vwap: Volume Weighted Average Price
        - ewo: Elliott Wave Oscillator
        - ewo_15min_avg: 15-minute rolling average of EWO
        - rsi: Relative Strength Index (0-100 scale)
    """
    df = df.copy()

    # Add EMA
    df['ema_30'] = EMA(df, column='close', period=ema_period)

    # Add VWAP
    df['vwap'] = VWAP(df, price_col='close', volume_col='volume')

    # Add EWO
    df['ewo'] = EWO(df, column='close', fast_period=ewo_fast, slow_period=ewo_slow)

    # Add EWO 15-minute rolling average (simple moving average over ewo_avg_period bars)
    df['ewo_15min_avg'] = df['ewo'].rolling(window=ewo_avg_period, min_periods=1).mean()

    # Add RSI
    df['rsi'] = RSI(df, column='close', period=rsi_period)

    return df


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Strategy.py, Test.py
"""

import Config

# Export functions for use by other modules
__all__ = ['EMA', 'VWAP', 'EWO', 'RSI', 'add_indicators']
