"""
Analysis.py - Technical Analysis

Module Goal: Process all math here (EMA, VWAP, options pricing)

================================================================================
INTERNAL - Indicator Calculations
================================================================================
"""

import pandas as pd
import numpy as np


# Default EMA periods (days)
EMA_PERIODS = [10, 21, 50, 100, 200]


def EMA(df, column='close', period=30):
    """
    Calculate Exponential Moving Average.

    Args:
        df: DataFrame with price data
        column: Column name to calculate EMA on (default: 'close')
        period: EMA period in bars (default: 30)

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


def true_price(stock_price, stock_high, stock_low):
    """Calculate True Price as the average of high, low, and stock price at each interval."""
    if pd.notna(stock_high) and pd.notna(stock_low):
        return (stock_high + stock_low + stock_price) / 3
    return stock_price


def add_indicators(df, ema_periods=None):
    """
    Add all standard indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data (must have 'close', 'volume' columns)
        ema_periods: List of EMA periods to calculate (default: [10, 21, 50, 100, 200])

    Returns:
        DataFrame with added indicator columns:
        - ema_10, ema_21, ema_50, ema_100, ema_200: EMAs at each period
        - vwap: Volume Weighted Average Price
    """
    if ema_periods is None:
        ema_periods = EMA_PERIODS

    df = df.copy()

    # Calculate true price column: (high + low + close) / 3
    if 'high' in df.columns and 'low' in df.columns:
        df['_true_price'] = (df['high'] + df['low'] + df['close']) / 3
    else:
        df['_true_price'] = df['close']

    # Add EMAs at each period (based on true price)
    for period in ema_periods:
        df[f'ema_{period}'] = EMA(df, column='_true_price', period=period)

    # Add VWAP (based on true price)
    df['vwap'] = VWAP(df, price_col='_true_price', volume_col='volume')

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
__all__ = ['EMA', 'VWAP', 'true_price', 'add_indicators', 'EMA_PERIODS',
           'black_scholes_call', 'black_scholes_put', 'black_scholes_price',
           'calculate_greeks', 'estimate_option_price_bs']
