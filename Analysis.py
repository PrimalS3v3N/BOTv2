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


def add_indicators(df, ema_period=30, ewo_fast=5, ewo_slow=35, ewo_avg_period=15, rsi_period=14, rsi_avg_period=10):
    """
    Add all standard indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data (must have 'close', 'volume' columns)
        ema_period: Period for EMA calculation (default: 30)
        ewo_fast: Fast period for EWO (default: 5)
        ewo_slow: Slow period for EWO (default: 35)
        ewo_avg_period: Period for EWO rolling average (default: 15 for 15-min avg on 1-min data)
        rsi_period: Period for RSI calculation (default: 14)
        rsi_avg_period: Period for RSI rolling average (default: 10 for 10-min avg on 1-min data)

    Returns:
        DataFrame with added indicator columns:
        - ema_20: 20-period EMA
        - ema_30: 30-period EMA
        - vwap: Volume Weighted Average Price
        - vwap_ema_avg: (VWAP + EMA) / 2
        - ewo: Elliott Wave Oscillator
        - ewo_15min_avg: 15-minute rolling average of EWO
        - rsi: Relative Strength Index (0-100 scale)
        - rsi_10min_avg: 10-minute rolling average of RSI
    """
    df = df.copy()

    # Calculate true price column: (high + low + close) / 3
    # All indicators use true price instead of raw stock close price
    if 'high' in df.columns and 'low' in df.columns:
        df['_true_price'] = (df['high'] + df['low'] + df['close']) / 3
    else:
        df['_true_price'] = df['close']

    # Add EMA (based on true price)
    df['ema_20'] = EMA(df, column='_true_price', period=20)
    df['ema_30'] = EMA(df, column='_true_price', period=ema_period)

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

    # Add VWAP-EMA Average: (VWAP + EMA) / 2
    df['vwap_ema_avg'] = (df['vwap'] + df['ema_30']) / 2

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
                              volatility=None, risk_free_rate=None):
    """
    Estimate option price using Black-Scholes model with offset calibration.

    When entry data is provided, calibrates BS to the actual market price:
        offset = entry_price - BS(entry_stock_price)
        price  = BS(current_stock_price) + offset

    At entry the simulated price matches the signal cost exactly. Subsequent
    bars move by full BS dynamics (delta + gamma) instead of a linear delta
    approximation, keeping the stop-loss buffer consistent with entry pricing.

    Args:
        stock_price: Current stock price
        strike: Option strike price
        option_type: 'CALL' or 'PUT'
        days_to_expiry: Days until expiration
        entry_price: Original entry price (optional, for offset calibration)
        entry_stock_price: Stock price at entry (optional, for offset calibration)
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
    # offset = entry_price - BS_theoretical(entry_stock) anchors the model
    # so that at entry the simulated price matches the signal cost exactly,
    # and subsequent bars move by full BS dynamics (delta + gamma)
    if entry_price and entry_stock_price and entry_price > 0:
        entry_theoretical = black_scholes_price(
            entry_stock_price, strike, T, risk_free_rate, volatility, option_type
        )
        offset = entry_price - entry_theoretical
        calibrated_price = theoretical_price + offset

        return max(min_price, calibrated_price)

    return max(min_price, theoretical_price)


# Export functions for use by other modules
__all__ = ['EMA', 'VWAP', 'EWO', 'true_price', 'RSI', 'add_indicators',
           'black_scholes_call', 'black_scholes_put', 'black_scholes_price',
           'calculate_greeks', 'estimate_option_price_bs']
