"""
Test.py - Backtesting Framework for Discord Trading Signals

Module Goal: Run backtests on historical Discord signals.

Functionality:
1. Fetch Discord messages from signal channels
2. Parse signals using Signal.py
3. Get historical stock data aligned with signal timestamps
4. Track positions through market hours
5. Generate DataFrames: Signals, Positions, Tracking matrices

Usage:
    from Test import Backtest
    bt = Backtest(lookback_days=5)
    results = bt.run()
    bt.summary()

================================================================================
INTERNAL - Backtesting Logic
================================================================================
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, date
from zoneinfo import ZoneInfo
import time
import random
import requests
import base64
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Install with: pip install yfinance")

# Timezone for market hours
EASTERN = ZoneInfo('America/New_York')


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Signal.py
"""

import Config
import Signal


# =============================================================================
# INTERNAL - Discord Fetcher
# =============================================================================

class DiscordFetcher:
    """
    Fetches messages from Discord signal channels.

    Implements browser spoofing to make requests appear identical to
    regular browser traffic accessing Discord web client.
    """

    def __init__(self, token=None, channel_id=None):
        discord_config = Config.get_config('discord')
        self.token = token or discord_config.get('token')
        self.channel_id = channel_id or discord_config.get('channel_id')
        self.api_version = discord_config.get('api_version', 'v10')
        self.base_url = discord_config.get('base_url', 'https://discord.com/api')
        self.message_limit = discord_config.get('test_message_limit', 100)

        # Browser spoofing configuration
        self.spoof_enabled = discord_config.get('spoof_browser', True)
        self.browser_config = discord_config.get('browser_config', {})
        self.super_properties = discord_config.get('x_super_properties', {})

        # Rate limiting configuration - human-like timing
        rate_config = discord_config.get('rate_limit', {})
        self.min_delay = rate_config.get('min_delay', 2.0)
        self.max_delay = rate_config.get('max_delay', 5.0)
        self.batch_size = rate_config.get('batch_size', 50)
        self.long_pause_chance = rate_config.get('long_pause_chance', 0.15)
        self.long_pause_min = rate_config.get('long_pause_min', 8.0)
        self.long_pause_max = rate_config.get('long_pause_max', 15.0)

        # Create persistent session for connection reuse and cookie handling
        self.session = requests.Session()
        self._configure_session()

    def _build_super_properties(self):
        """
        Build and encode X-Super-Properties header.
        This header contains Discord client fingerprint information.
        """
        if not self.super_properties:
            return None

        # Convert to JSON and base64 encode
        props_json = json.dumps(self.super_properties, separators=(',', ':'))
        return base64.b64encode(props_json.encode()).decode()

    def _build_browser_headers(self):
        """
        Build headers that mimic a real browser accessing Discord.
        Returns headers dict with proper formatting for HTTP.
        """
        if not self.spoof_enabled or not self.browser_config:
            return {'Authorization': self.token}

        bc = self.browser_config

        headers = {
            # Authentication
            'Authorization': self.token,

            # Standard browser headers
            'User-Agent': bc.get('user_agent', ''),
            'Accept': bc.get('accept', '*/*'),
            'Accept-Language': bc.get('accept_language', 'en-US,en;q=0.9'),
            'Accept-Encoding': bc.get('accept_encoding', 'gzip, deflate, br'),
            'Connection': bc.get('connection', 'keep-alive'),
            'Cache-Control': bc.get('cache_control', 'no-cache'),
            'Pragma': bc.get('pragma', 'no-cache'),

            # Security fetch headers (modern Chrome/Firefox)
            'Sec-CH-UA': bc.get('sec_ch_ua', ''),
            'Sec-CH-UA-Mobile': bc.get('sec_ch_ua_mobile', '?0'),
            'Sec-CH-UA-Platform': bc.get('sec_ch_ua_platform', ''),
            'Sec-Fetch-Dest': bc.get('sec_fetch_dest', 'empty'),
            'Sec-Fetch-Mode': bc.get('sec_fetch_mode', 'cors'),
            'Sec-Fetch-Site': bc.get('sec_fetch_site', 'same-origin'),

            # Discord-specific headers
            'X-Discord-Locale': bc.get('x_discord_locale', 'en-US'),
            'X-Discord-Timezone': bc.get('x_discord_timezone', 'America/New_York'),
            'X-Debug-Options': bc.get('x_debug_options', 'bugReporterEnabled'),

            # Origin/Referer
            'Origin': bc.get('origin', 'https://discord.com'),
            'Referer': bc.get('referer', 'https://discord.com/channels/@me'),
        }

        # Add X-Super-Properties (Discord client fingerprint)
        super_props = self._build_super_properties()
        if super_props:
            headers['X-Super-Properties'] = super_props

        # Remove empty headers
        headers = {k: v for k, v in headers.items() if v}

        return headers

    def _configure_session(self):
        """
        Configure the requests session with browser-like settings.
        Sets up persistent connections, cookies, and default headers.
        """
        # Set default headers on session
        self.session.headers.update(self._build_browser_headers())

        # Configure session for browser-like behavior
        # Retry adapter for connection reliability
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def update_referer(self, channel_id=None):
        """Update referer header to match current channel context."""
        if channel_id:
            self.session.headers['Referer'] = f'https://discord.com/channels/@me/{channel_id}'

    def fetch_messages(self, limit=100, before_id=None):
        """
        Fetch messages from Discord channel.
        Uses browser spoofing to appear as regular web traffic.
        """
        url = f"{self.base_url}/{self.api_version}/channels/{self.channel_id}/messages"
        params = {'limit': min(limit, 100)}

        if before_id:
            params['before'] = before_id

        # Update referer to match current channel
        self.update_referer(self.channel_id)

        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Discord API error: {response.status_code}")
                if response.status_code == 401:
                    print("  -> Token may be invalid or expired")
                elif response.status_code == 403:
                    print("  -> Access forbidden - check channel permissions")
                elif response.status_code == 429:
                    print("  -> Rate limited - waiting before retry")
                    retry_after = response.headers.get('Retry-After', 5)
                    time.sleep(float(retry_after))
                return []
        except requests.RequestException as e:
            print(f"Discord request failed: {e}")
            return []

    def close(self):
        """Close the session and release resources."""
        if self.session:
            self.session.close()

    def _human_delay(self):
        """
        Generate human-like delay between requests.
        Randomized timing with occasional longer pauses to simulate
        a real user scrolling and reading messages.
        """
        # Check if we should take a longer "reading" pause
        if random.random() < self.long_pause_chance:
            delay = random.uniform(self.long_pause_min, self.long_pause_max)
            print(f"    (pausing {delay:.1f}s...)")
        else:
            # Normal randomized delay
            delay = random.uniform(self.min_delay, self.max_delay)

        time.sleep(delay)

    def fetch_messages_for_days(self, days=5):
        """
        Fetch all messages for the specified number of days.
        Uses human-like timing to avoid detection.
        """
        cutoff_date = dt.datetime.now(EASTERN) - timedelta(days=days)
        all_messages = []
        before_id = None
        request_count = 0

        print(f"Fetching Discord messages for last {days} days...")
        print(f"  Using {self.batch_size} msgs/request with {self.min_delay}-{self.max_delay}s delays")

        while True:
            # Use configured batch size instead of max 100
            messages = self.fetch_messages(limit=self.batch_size, before_id=before_id)
            request_count += 1

            if not messages:
                break

            for msg in messages:
                try:
                    timestamp_str = msg.get('timestamp', '')
                    timestamp = pd.to_datetime(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=EASTERN)
                    else:
                        timestamp = timestamp.astimezone(EASTERN)

                    if timestamp < cutoff_date:
                        break

                    all_messages.append({
                        'id': msg.get('id'),
                        'timestamp': timestamp,
                        'content': msg.get('content', ''),
                        'author': msg.get('author', {}).get('username', 'unknown'),
                        'author_id': msg.get('author', {}).get('id')
                    })
                except Exception:
                    continue

            if messages:
                last_timestamp = pd.to_datetime(messages[-1].get('timestamp', ''))
                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=EASTERN)
                else:
                    last_timestamp = last_timestamp.astimezone(EASTERN)

                if last_timestamp < cutoff_date:
                    break

                before_id = messages[-1].get('id')
            else:
                break

            # Human-like delay between requests
            self._human_delay()

        if all_messages:
            df = pd.DataFrame(all_messages)
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"  Fetched {len(df)} messages")
            return df
        else:
            return pd.DataFrame(columns=['id', 'timestamp', 'content', 'author', 'author_id'])


# =============================================================================
# INTERNAL - Signal Parser
# =============================================================================

class SignalParser:
    """Parses Discord messages into trading signals using Signal.py."""

    def __init__(self):
        self.alert_marker = Config.DISCORD_CONFIG.get('alert_marker')

    def parse_message(self, content):
        """Parse a single message into a signal."""
        if not content or self.alert_marker not in content:
            return None

        order = Signal.BuildOrder(content)

        if order is None:
            return None

        signal = {
            'ticker': order.get('Ticker'),
            'strike': float(order.get('Strike')) if order.get('Strike') else None,
            'option_type': order.get('Option'),
            'expiration': self._parse_expiration(order.get('Expiration')),
            'cost': float(order.get('Cost')) if order.get('Cost') else None,
            'raw_message': content
        }

        if not all([signal['ticker'], signal['strike'], signal['option_type']]):
            return None

        return signal

    def _parse_expiration(self, exp_str):
        """Convert expiration string to date object."""
        if not exp_str:
            return None

        try:
            parts = exp_str.split()
            if len(parts) >= 3:
                month = int(parts[0])
                day = int(parts[1])
                year = int(parts[2])
                if year < 100:
                    year += 2000
                return date(year, month, day)
        except (ValueError, IndexError):
            pass

        return None

    def parse_all_messages(self, messages_df):
        """Parse all messages in a DataFrame."""
        signals = []

        for _, row in messages_df.iterrows():
            signal = self.parse_message(row.get('content', ''))

            if signal:
                signal['signal_time'] = row.get('timestamp')
                signal['message_id'] = row.get('id')
                signals.append(signal)

        if signals:
            df = pd.DataFrame(signals)
            df = df.sort_values('signal_time').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame(columns=[
                'ticker', 'strike', 'option_type', 'expiration',
                'cost', 'signal_time', 'message_id', 'raw_message'
            ])


# =============================================================================
# INTERNAL - Historical Data Fetcher
# =============================================================================

class HistoricalDataFetcher:
    """Fetches historical stock data using yfinance."""

    def __init__(self):
        self._cache = {}
        self.pre_signal_minutes = Config.DATA_CONFIG.get('pre_signal_minutes', 60)

    def fetch_stock_data(self, ticker, start_date, end_date, interval='1m'):
        """Fetch historical stock data."""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}_{interval}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            stock = yf.Ticker(ticker)

            df = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval=interval,
                prepost=True
            )

            if df.empty:
                return None

            df.columns = df.columns.str.lower()

            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC').tz_convert(EASTERN)
            else:
                df.index = df.index.tz_convert(EASTERN)

            df = df[(df.index.time >= dt.time(9, 0)) & (df.index.time <= dt.time(16, 0))]

            self._cache[cache_key] = df

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def fetch_data_around_signal(self, ticker, signal_time, hours_after=8):
        """Fetch data around a signal time."""
        start_time = signal_time - timedelta(minutes=self.pre_signal_minutes)
        end_time = signal_time + timedelta(hours=hours_after)

        market_close = signal_time.replace(hour=16, minute=0, second=0, microsecond=0)
        if end_time < market_close:
            end_time = market_close

        return self.fetch_stock_data(ticker, start_time, end_time, interval='1m')

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


# =============================================================================
# INTERNAL - Position Class
# =============================================================================

class Position:
    """Represents a trading position with state tracking."""

    def __init__(self, signal, entry_price, entry_time, contracts=1):
        # Signal data
        self.ticker = signal.get('ticker')
        self.strike = signal.get('strike')
        self.option_type = signal.get('option_type')
        self.expiration = signal.get('expiration')
        self.signal_cost = signal.get('cost')
        self.signal_time = signal.get('signal_time')
        self.raw_message = signal.get('raw_message', '')

        # Entry data
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.contracts = contracts

        # Tracking state
        self.current_price = entry_price
        self.highest_price = entry_price
        self.lowest_price = entry_price

        # Exit data
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.is_closed = False

    def update(self, timestamp, price, stock_price=None):
        """Update position with new price data."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.last_update = timestamp
        self.last_stock_price = stock_price

    def close(self, exit_price, exit_time, exit_reason):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.is_closed = True

    def get_pnl(self, price=None):
        """Get P&L in dollars per contract."""
        price = price or self.current_price
        return (price - self.entry_price) * 100 * self.contracts

    def get_pnl_pct(self, price=None):
        """Get P&L as percentage."""
        price = price or self.current_price
        if self.entry_price > 0:
            return ((price - self.entry_price) / self.entry_price) * 100
        return 0

    def get_minutes_held(self, current_time=None):
        """Get minutes held."""
        current_time = current_time or self.last_update or dt.datetime.now(EASTERN)
        if self.entry_time and isinstance(current_time, dt.datetime):
            return (current_time - self.entry_time).total_seconds() / 60
        return 0

    def get_trade_label(self):
        """Get formatted trade label."""
        return f"{self.ticker}:{self.strike}:{self.option_type}"

    def to_dict(self):
        """Convert position to dictionary."""
        return {
            'ticker': self.ticker,
            'strike': self.strike,
            'option_type': self.option_type,
            'expiration': self.expiration,
            'signal_time': self.signal_time,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_reason': self.exit_reason,
            'contracts': self.contracts,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'pnl': self.get_pnl(self.exit_price) if self.exit_price else None,
            'pnl_pct': self.get_pnl_pct(self.exit_price) if self.exit_price else None,
            'minutes_held': self.get_minutes_held(self.exit_time) if self.exit_time else None,
        }


# =============================================================================
# INTERNAL - Tracking Matrix
# =============================================================================

class TrackingMatrix:
    """High-resolution tracking data for a single trade."""

    def __init__(self, position):
        self.position = position
        self.trade_label = position.get_trade_label()
        self.records = []

    def add_record(self, timestamp, stock_price, option_price, volume, holding=True):
        """Add a tracking record."""
        pnl_pct = self.position.get_pnl_pct(option_price) if holding else np.nan

        record = {
            'timestamp': timestamp,
            'stock_price': stock_price,
            'option_price': option_price,
            'volume': volume,
            'holding': holding,
            'entry_price': self.position.entry_price if holding else np.nan,
            'pnl': self.position.get_pnl(option_price) if holding else np.nan,
            'pnl_pct': pnl_pct,
            'highest_price': self.position.highest_price if holding else np.nan,
            'lowest_price': self.position.lowest_price if holding else np.nan,
            'minutes_held': self.position.get_minutes_held(timestamp) if holding else np.nan,
        }

        self.records.append(record)

    def to_dataframe(self):
        """Convert to DataFrame."""
        if self.records:
            df = pd.DataFrame(self.records)
            df['trade_label'] = self.trade_label
            return df
        return pd.DataFrame()

    def get_summary(self):
        """Get tracking summary statistics."""
        if not self.records:
            return {}

        df = self.to_dataframe()

        return {
            'trade_label': self.trade_label,
            'total_bars': len(df),
            'duration_minutes': df['minutes_held'].max() if len(df) > 0 else 0,
            'max_pnl_pct': df['pnl_pct'].max() if len(df) > 0 else 0,
            'min_pnl_pct': df['pnl_pct'].min() if len(df) > 0 else 0,
        }


# =============================================================================
# INTERNAL - Option Price Estimation
# =============================================================================

def estimate_option_price(stock_price, strike, option_type, days_to_expiry,
                          entry_price=None, entry_stock_price=None, volatility=0.3):
    """
    Estimate option price based on stock price movement.
    Simple delta-based model for backtesting.
    """
    # Calculate intrinsic value
    if option_type == 'CALL':
        intrinsic = max(0, stock_price - strike)
    else:
        intrinsic = max(0, strike - stock_price)

    # Time value estimation
    time_factor = max(0, days_to_expiry) / 365
    time_value = stock_price * volatility * np.sqrt(time_factor) * 0.4

    theoretical_price = intrinsic + time_value

    # If we have entry data, use delta approximation
    if entry_price and entry_stock_price and entry_price > 0:
        # Estimate delta based on moneyness
        moneyness = stock_price / strike
        if option_type == 'CALL':
            delta = 0.5 + 0.4 * (moneyness - 1) if moneyness > 0.95 else 0.3
        else:
            delta = -0.5 + 0.4 * (1 - moneyness) if moneyness < 1.05 else -0.3

        delta = max(-0.9, min(0.9, delta))

        # Calculate price change based on stock movement
        stock_change = stock_price - entry_stock_price
        price_change = abs(delta) * stock_change
        estimated = entry_price + price_change

        return max(0.01, estimated)

    return max(0.01, theoretical_price)


# =============================================================================
# EXTERNAL - Backtest Class (Main Interface)
# =============================================================================

class Backtest:
    """
    Main backtesting engine.

    Orchestrates:
    - Discord message fetching
    - Signal parsing
    - Historical data retrieval
    - Position simulation
    - Results compilation
    """

    def __init__(self, lookback_days=None, config=None):
        # Load backtest config
        self.config = Config.get_config('backtest').copy()
        if config:
            self.config.update(config)

        self.lookback_days = lookback_days or self.config.get('lookback_days', 5)
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.default_contracts = self.config.get('default_contracts', 1)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)
        self.commission_per_contract = self.config.get('commission_per_contract', 0.65)

        # Initialize components
        self.discord_fetcher = DiscordFetcher()
        self.signal_parser = SignalParser()
        self.data_fetcher = HistoricalDataFetcher()

        # Results storage
        self.signals_df = None
        self.positions = []
        self.tracking_matrices = {}
        self.results = None
        self._has_run = False

    def run(self):
        """Execute the backtest."""
        print(f"\n{'='*60}")
        print("BACKTEST EXECUTION")
        print(f"{'='*60}")
        print(f"Lookback: {self.lookback_days} days")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")

        # Step 1: Fetch Discord messages
        messages_df = self.discord_fetcher.fetch_messages_for_days(self.lookback_days)

        if messages_df.empty:
            print("No messages found")
            return self._empty_results()

        # Step 2: Parse signals
        print("\nParsing signals...")
        self.signals_df = self.signal_parser.parse_all_messages(messages_df)

        if self.signals_df.empty:
            print("No valid signals found")
            return self._empty_results()

        print(f"  Found {len(self.signals_df)} signals")

        # Step 3: Process each signal
        print("\nProcessing signals...")
        self.positions = []
        self.tracking_matrices = {}

        for idx, signal in self.signals_df.iterrows():
            print(f"\n  [{idx+1}/{len(self.signals_df)}] {signal['ticker']} "
                  f"${signal['strike']} {signal['option_type']}")

            position, matrix = self._process_signal(signal)

            if position:
                self.positions.append(position)
                self.tracking_matrices[position.get_trade_label()] = matrix

                pnl = position.get_pnl(position.exit_price) if position.exit_price else 0
                print(f"    Exit: {position.exit_reason} | P&L: ${pnl:+.2f}")

        # Step 4: Compile results
        print(f"\n{'='*60}")
        print("COMPILING RESULTS")
        print(f"{'='*60}")

        self.results = self._compile_results()
        self._has_run = True

        return self.results

    def _process_signal(self, signal):
        """Process a single signal through the backtest."""
        ticker = signal['ticker']
        signal_time = signal['signal_time']

        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=EASTERN)

        # Fetch historical data
        stock_data = self.data_fetcher.fetch_data_around_signal(
            ticker, signal_time, hours_after=8
        )

        if stock_data is None or stock_data.empty:
            print(f"    No data available for {ticker}")
            return None, None

        # Find entry point
        entry_idx = stock_data.index.searchsorted(signal_time)

        if entry_idx >= len(stock_data):
            print(f"    No data after signal time")
            return None, None

        # Get entry bar
        entry_bar = stock_data.iloc[entry_idx]
        entry_time = stock_data.index[entry_idx]
        entry_stock_price = entry_bar['close']

        # Get entry option price
        days_to_expiry = (signal['expiration'] - entry_time.date()).days if signal['expiration'] else 30

        if signal.get('cost'):
            entry_option_price = signal.get('cost')
        else:
            entry_option_price = estimate_option_price(
                stock_price=entry_stock_price,
                strike=signal['strike'],
                option_type=signal['option_type'],
                days_to_expiry=days_to_expiry
            )

        entry_option_price *= (1 + self.slippage_pct)

        # Create position
        position = Position(
            signal=signal,
            entry_price=entry_option_price,
            entry_time=entry_time,
            contracts=self.default_contracts
        )

        # Create tracking matrix
        matrix = TrackingMatrix(position)

        # Simulate through all bars
        self._simulate_position(position, matrix, stock_data, signal, entry_idx, entry_stock_price)

        return position, matrix

    def _simulate_position(self, position, matrix, stock_data, signal, entry_idx, entry_stock_price):
        """Simulate position through historical data."""
        days_to_expiry = (signal['expiration'] - position.entry_time.date()).days if signal['expiration'] else 30

        for i, (timestamp, bar) in enumerate(stock_data.iterrows()):
            stock_price = bar['close']
            volume = bar.get('volume', 0)

            current_days_to_expiry = max(0, days_to_expiry - (timestamp.date() - position.entry_time.date()).days)

            # Estimate option price
            option_price = estimate_option_price(
                stock_price=stock_price,
                strike=position.strike,
                option_type=position.option_type,
                days_to_expiry=current_days_to_expiry,
                entry_price=position.entry_price,
                entry_stock_price=entry_stock_price
            )

            # Determine holding status
            is_pre_entry = i < entry_idx
            is_post_exit = position.is_closed
            holding = not is_pre_entry and not is_post_exit

            if holding:
                position.update(timestamp, option_price, stock_price)

                # Record tracking data
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=True
                )

                # Exit at market close
                if timestamp.time() >= dt.time(15, 55) and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'market_close')
            else:
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=False
                )

        # Close at end of data if still open
        if not position.is_closed:
            exit_price = position.current_price * (1 - self.slippage_pct)
            position.close(exit_price, stock_data.index[-1], 'market_close')

    def _compile_results(self):
        """Compile backtest results into DataFrames."""
        signals_df = self.signals_df.copy() if self.signals_df is not None else pd.DataFrame()

        if self.positions:
            positions_data = [p.to_dict() for p in self.positions]
            positions_df = pd.DataFrame(positions_data)
        else:
            positions_df = pd.DataFrame()

        tracking_matrices = {}
        for label, matrix in self.tracking_matrices.items():
            tracking_matrices[label] = matrix.to_dataframe()

        summary = self._calculate_summary(positions_df)

        return {
            'Signals': signals_df,
            'Positions': positions_df,
            'Tracking_matrices': tracking_matrices,
            'Summary': summary
        }

    def _calculate_summary(self, positions_df):
        """Calculate summary statistics."""
        if positions_df.empty:
            return {}

        total_trades = len(positions_df)
        closed_trades = positions_df[positions_df['exit_price'].notna()]

        if closed_trades.empty:
            return {'total_trades': total_trades, 'closed_trades': 0}

        winners = closed_trades[closed_trades['pnl'] > 0]
        losers = closed_trades[closed_trades['pnl'] <= 0]

        total_pnl = closed_trades['pnl'].sum()
        commission_total = total_trades * self.commission_per_contract * self.default_contracts * 2
        net_pnl = total_pnl - commission_total

        return {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0,
            'total_pnl': total_pnl,
            'net_pnl': net_pnl,
            'commission_total': commission_total,
            'average_pnl': closed_trades['pnl'].mean() if len(closed_trades) > 0 else 0,
            'best_trade': closed_trades['pnl'].max() if len(closed_trades) > 0 else 0,
            'worst_trade': closed_trades['pnl'].min() if len(closed_trades) > 0 else 0,
            'average_minutes_held': closed_trades['minutes_held'].mean() if len(closed_trades) > 0 else 0,
            'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
            'exit_reasons': closed_trades['exit_reason'].value_counts().to_dict() if len(closed_trades) > 0 else {}
        }

    def _empty_results(self):
        """Return empty results structure."""
        return {
            'Signals': pd.DataFrame(),
            'Positions': pd.DataFrame(),
            'Tracking_matrices': {},
            'Summary': {}
        }

    def summary(self):
        """Print backtest summary."""
        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return

        summary = self.results.get('Summary', {})

        print(f"\n{'='*60}")
        print("BACKTEST SUMMARY")
        print(f"{'='*60}")

        if not summary:
            print("No trades executed")
            return

        print(f"\nTRADE STATISTICS:")
        print(f"  Total Signals: {len(self.signals_df) if self.signals_df is not None else 0}")
        print(f"  Total Trades: {summary.get('total_trades', 0)}")
        print(f"  Closed Trades: {summary.get('closed_trades', 0)}")
        print(f"  Winners: {summary.get('winners', 0)}")
        print(f"  Losers: {summary.get('losers', 0)}")
        print(f"  Win Rate: {summary.get('win_rate', 0):.1f}%")

        print(f"\nPROFITABILITY:")
        print(f"  Total P&L: ${summary.get('total_pnl', 0):+,.2f}")
        print(f"  Commissions: ${summary.get('commission_total', 0):,.2f}")
        print(f"  Net P&L: ${summary.get('net_pnl', 0):+,.2f}")
        print(f"  Average P&L: ${summary.get('average_pnl', 0):+,.2f}")
        print(f"  Best Trade: ${summary.get('best_trade', 0):+,.2f}")
        print(f"  Worst Trade: ${summary.get('worst_trade', 0):+,.2f}")
        print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")

        print(f"\nTIMING:")
        print(f"  Avg Hold Time: {summary.get('average_minutes_held', 0):.1f} minutes")

        print(f"\nEXIT REASONS:")
        for reason, count in summary.get('exit_reasons', {}).items():
            print(f"  {reason}: {count}")

        print(f"{'='*60}\n")

    def get_tracking_matrices(self):
        """Get all trade tracking matrices."""
        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return {}

        return self.results.get('Tracking_matrices', {})

    def get_positions_df(self):
        """Get positions DataFrame."""
        if not self._has_run:
            return pd.DataFrame()
        return self.results.get('Positions', pd.DataFrame())

    def get_signals_df(self):
        """Get signals DataFrame."""
        if not self._has_run:
            return pd.DataFrame()
        return self.results.get('Signals', pd.DataFrame())

    def BT_Save(self, filepath='BT_DATA.pkl'):
        """
        Save backtest results to a pickle file for Dashboard visualization.

        Args:
            filepath: Path to save the pickle file (default: 'BT_DATA.pkl')
        """
        import pickle

        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return False

        try:
            # Format data for Dashboard.py which expects 'matrices' and 'exit_signals' keys
            dashboard_data = {
                'matrices': self.results.get('Tracking_matrices', {}),
                'exit_signals': {},  # Placeholder for future exit signal tracking
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dashboard_data, f)
            print(f"Backtest results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving backtest results: {e}")
            return False


# =============================================================================
# EXTERNAL - Quick Test Function
# =============================================================================

def quick_test(days=1):
    """Run a quick backtest."""
    bt = Backtest(lookback_days=days)
    bt.run()
    bt.summary()
    return bt


if __name__ == '__main__':
    bt = quick_test(days=1)
