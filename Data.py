"""
Data.py - Market Data Fetching and Account Status Module

FUNCTIONALITY:
This module handles all market data retrieval and account information:
- Real-time stock quotes and pricing
- Historical OHLCV data for technical analysis
- Account balance and buying power queries
- Open positions and pending orders tracking
- Market hours checking

SUPPORTED MODULES:
- Main.py: Real-time market data for live trading decisions
- Test.py: (Not directly used - uses historical CSV data)
- Analysis.py: Provides data for indicator calculations
- Dashboard.py: Market data for visualization

MAINTENANCE INSTRUCTIONS:
=========================
When adding new features:
1. Add function to appropriate section below
2. Update the section header comment with new function name
3. If function uses Config values, document which Config keys it reads
4. Consider caching for frequently accessed data

When removing features:
1. Remove function and update section header
2. Check for dependencies in Main.py, Analysis.py
3. Remove any Config keys that are no longer used

Section Organization:
- CACHE MANAGEMENT: Internal caching for market data
- QUOTE FUNCTIONS: Real-time price quotes
- HISTORICAL DATA: OHLCV historical data retrieval
- ACCOUNT FUNCTIONS: Balance, positions, orders
- MARKET STATUS: Market hours checking
- DATA AGGREGATION: Combined data snapshots and updates
"""

####### REQUIRED IMPORTS
import pandas as Panda
import datetime as dt
from datetime import timedelta
import numpy as np
import time

# Import centralized config
import Config

# =============================================================================
# MODULE CONFIGURATION
# =============================================================================
# Settings loaded from Config.DATA_CONFIG:
# - cache_ttl: Cache time-to-live in seconds
# - max_quote_history: Maximum quotes to keep in memory
# - purge_old_quotes_minutes: Remove quotes older than this

####### INTERNAL DATA CACHE
_market_data_cache = {}
_last_fetch_time = {}
_cache_ttl = Config.DATA_CONFIG.get('cache_ttl', 60)  # From Config


def initialize_market_data():
    # SCOPE: Create empty DataFrames for market data storage
    # Returns: Dictionary with empty data structures

    data = {
        'quotes': Panda.DataFrame(),
        'ohlcv': Panda.DataFrame(),
        'account': Panda.DataFrame(),
        'positions': Panda.DataFrame(),
        'orders': Panda.DataFrame(),
        'timestamp': dt.datetime.now()
    }

    return data


def get_quote(ticker, webull=None):
    # SCOPE: Fetch latest quote for a ticker
    # Returns: Dictionary with price data {price, bid, ask, volume}

    if webull is None:
        print("get_quote(): Webull not provided")
        return {}

    try:
        quote_data = webull.get_quotes(ticker)

        if quote_data and len(quote_data) > 0:
            quote = {
                'ticker': ticker,
                'price': float(quote_data.get('S.Last Trade Price', 0)),
                'bid': float(quote_data.get('S.Bid Price', 0)),
                'ask': float(quote_data.get('S.Ask Price', 0)),
                'volume': float(quote_data.get('S.Volume', 0)),
                'timestamp': dt.datetime.now()
            }
            return quote
        else:
            print(f"get_quote(): No data for {ticker}")
            return {}

    except Exception as e:
        print(f"get_quote() Exception for {ticker}: {e}")
        return {}


def get_historical_data(ticker, timeframe='1min', bars=100, webull=None):
    # SCOPE: Fetch historical OHLCV data for a ticker
    # timeframe: '1min', '5min', '15min', '1hour', '1day'
    # Returns: DataFrame with Open, High, Low, Close, Volume

    if webull is None:
        print("get_historical_data(): Webull not provided")
        return Panda.DataFrame()

    try:
        # This will need to be adapted based on your Webull implementation
        # For now, return a properly structured empty DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        hist_data = Panda.DataFrame(columns=columns)

        # Cache implementation for faster subsequent calls
        cache_key = f"{ticker}_{timeframe}"
        now = time.time()

        if cache_key in _market_data_cache:
            if now - _last_fetch_time.get(cache_key, 0) < _cache_ttl:
                return _market_data_cache[cache_key]

        # Fetch fresh data
        # hist_data = webull.get_historicals(ticker, interval=timeframe, span=get_span(bars, timeframe))

        _market_data_cache[cache_key] = hist_data
        _last_fetch_time[cache_key] = now

        return hist_data

    except Exception as e:
        print(f"get_historical_data() Exception for {ticker}: {e}")
        return Panda.DataFrame()


def get_account_balance(webull=None):
    # SCOPE: Fetch current account balance and buying power
    # Returns: Dictionary with {balance, buying_power, portfolio_value}

    if webull is None:
        print("get_account_balance(): Webull not provided")
        return {}

    try:
        account_info = webull.get_account()

        balance_data = {
            'cash': float(account_info.get('cash', 0)),
            'buying_power': float(account_info.get('buying_power', 0)),
            'portfolio_value': float(account_info.get('equity', 0)),
            'timestamp': dt.datetime.now()
        }

        return balance_data

    except Exception as e:
        print(f"get_account_balance() Exception: {e}")
        return {}


def get_open_positions(webull=None):
    # SCOPE: Fetch all open positions in account
    # Returns: DataFrame with {ticker, shares, avg_price, current_price, gain_loss}

    if webull is None:
        print("get_open_positions(): Webull not provided")
        return Panda.DataFrame()

    try:
        positions_data = webull.get_positions()

        if positions_data and len(positions_data) > 0:
            positions_df = Panda.DataFrame()

            for i, position in enumerate(positions_data):
                positions_df.loc[i, 'ticker'] = position.get('symbol', '')
                positions_df.loc[i, 'shares'] = float(position.get('quantity', 0))
                positions_df.loc[i, 'avg_price'] = float(position.get('average_price', 0))
                positions_df.loc[i, 'current_price'] = float(position.get('current_price', 0))
                positions_df.loc[i, 'gain_loss'] = float(position.get('unrealized_pl', 0))
                positions_df.loc[i, 'gain_loss_pct'] = float(position.get('unrealized_plpc', 0))

            return positions_df

        else:
            return Panda.DataFrame()

    except Exception as e:
        print(f"get_open_positions() Exception: {e}")
        return Panda.DataFrame()


def get_open_orders(webull=None):
    # SCOPE: Fetch all pending orders
    # Returns: DataFrame with {ticker, order_type, quantity, price, status}

    if webull is None:
        print("get_open_orders(): Webull not provided")
        return Panda.DataFrame()

    try:
        orders_data = webull.get_orders()

        if orders_data and len(orders_data) > 0:
            orders_df = Panda.DataFrame()

            for i, order in enumerate(orders_data):
                if order.get('state') == 'queued' or order.get('state') == 'confirmed':
                    orders_df.loc[i, 'ticker'] = order.get('symbol', '')
                    orders_df.loc[i, 'side'] = order.get('side', '')
                    orders_df.loc[i, 'quantity'] = float(order.get('quantity', 0))
                    orders_df.loc[i, 'price'] = float(order.get('price', 0))
                    orders_df.loc[i, 'status'] = order.get('state', '')
                    orders_df.loc[i, 'created_at'] = order.get('created_at', '')

            return orders_df

        else:
            return Panda.DataFrame()

    except Exception as e:
        print(f"get_open_orders() Exception: {e}")
        return Panda.DataFrame()


def get_market_hours():
    # SCOPE: Check if market is currently open
    # Market hours from Config.MARKET_CONFIG
    # Returns: Boolean

    now = dt.datetime.now()
    weekday = now.weekday()

    # Get market config
    market_config = Config.get_config('market')
    trading_days = market_config.get('trading_days', [0, 1, 2, 3, 4])

    # Market closed on non-trading days
    if weekday not in trading_days:
        return False

    # Market hours from Config (decimal format: 9.5 = 9:30 AM)
    market_start = market_config.get('market_hours_start', 9.5)
    market_end = market_config.get('market_hours_end', 16.0)

    # Convert decimal hours to time
    start_hour = int(market_start)
    start_minute = int((market_start - start_hour) * 60)
    end_hour = int(market_end)
    end_minute = int((market_end - end_hour) * 60)

    market_open = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    market_close = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

    return market_open <= now <= market_close


def build_market_snapshot(ticker, webull=None):
    # SCOPE: Create a complete market snapshot for a ticker
    # Returns: Dictionary with all relevant market data

    snapshot = {
        'ticker': ticker,
        'quote': get_quote(ticker, webull),
        'historical': get_historical_data(ticker, timeframe='1min', bars=100, webull=webull),
        'timestamp': dt.datetime.now()
    }

    return snapshot


def update_market_data(market_data, ticker, webull=None):
    # SCOPE: Update market data dictionary with latest information
    # Returns: Updated market_data dictionary

    try:
        # Update quotes
        quote = get_quote(ticker, webull)
        if quote:
            market_data['quotes'] = Panda.concat([
                market_data['quotes'],
                Panda.DataFrame([quote])
            ], ignore_index=True)

        # Update account info
        account = get_account_balance(webull)
        market_data['account'] = account

        # Update positions
        market_data['positions'] = get_open_positions(webull)

        # Update orders
        market_data['orders'] = get_open_orders(webull)

        market_data['timestamp'] = dt.datetime.now()

        return market_data

    except Exception as e:
        print(f"update_market_data() Exception: {e}")
        return market_data


def get_latest_price(ticker, market_data):
    # SCOPE: Get most recent price from market data
    # Returns: Float price or None

    try:
        if len(market_data['quotes']) > 0:
            latest_quote = market_data['quotes'].iloc[-1]
            if latest_quote['ticker'] == ticker:
                return latest_quote['price']
    except Exception as e:
        print(f"get_latest_price() Exception: {e}")

    return None


def clear_old_quotes(market_data, max_age_minutes=None):
    # SCOPE: Remove quotes older than specified age to manage memory
    # Default from Config.DATA_CONFIG['purge_old_quotes_minutes']
    # Returns: Cleaned market_data dictionary

    if max_age_minutes is None:
        max_age_minutes = Config.DATA_CONFIG.get('purge_old_quotes_minutes', 60)

    try:
        if len(market_data['quotes']) > 0:
            cutoff_time = dt.datetime.now() - timedelta(minutes=max_age_minutes)
            market_data['quotes'] = market_data['quotes'][
                market_data['quotes']['timestamp'] > cutoff_time
            ]
    except Exception as e:
        print(f"clear_old_quotes() Exception: {e}")

    return market_data
