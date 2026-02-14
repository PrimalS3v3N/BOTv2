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


def adjust_lookback_for_weekend(lookback_days):
    """
    Adjust lookback_days when running on weekends so the window
    always reaches back to trading days.

    On Saturday (weekday 5): add 1 day to skip Saturday
    On Sunday   (weekday 6): add 2 days to skip Saturday + Sunday

    Examples on Saturday:
        lookback_days=1 -> 2  (reaches Friday)
        lookback_days=2 -> 3  (reaches Thursday)
    Examples on Sunday:
        lookback_days=1 -> 3  (reaches Friday)
        lookback_days=2 -> 4  (reaches Thursday)
    """
    today = dt.datetime.now(EASTERN).weekday()  # Mon=0 ... Sun=6
    if today == 5:  # Saturday
        adjusted = lookback_days + 1
        print(f"  Weekend adjustment: Saturday detected, "
              f"lookback_days {lookback_days} -> {adjusted}")
        return adjusted
    elif today == 6:  # Sunday
        adjusted = lookback_days + 2
        print(f"  Weekend adjustment: Sunday detected, "
              f"lookback_days {lookback_days} -> {adjusted}")
        return adjusted
    return lookback_days


# Module-level DataFrame for variable explorer visibility
discord_messages_df = pd.DataFrame()


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Signal.py, Analysis.py, Strategy.py
"""

import Config
import Signal
import Analysis
import Strategy


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

        Args:
            days: Number of calendar days to look back.
                  days=1 means today only (from midnight)
                  days=2 means today and yesterday
        """
        # Calculate cutoff as midnight of (days-1) days ago
        # days=1 -> midnight today, days=2 -> midnight yesterday
        now = dt.datetime.now(EASTERN)
        cutoff_date = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        all_messages = []
        before_id = None
        request_count = 0

        print(f"Fetching Discord messages for last {days} day(s)...")
        print(f"  Cutoff: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
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
            # Reorder columns per Config source of truth
            col_order = Config.DATAFRAME_COLUMNS['discord_messages']
            df = df[[c for c in col_order if c in df.columns]]
            print(f"  Fetched {len(df)} messages")
            return df
        else:
            return pd.DataFrame(columns=Config.DATAFRAME_COLUMNS['discord_messages'])


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
            # Reorder columns per Config source of truth
            col_order = Config.DATAFRAME_COLUMNS['signals']
            df = df[[c for c in col_order if c in df.columns]]
            return df
        else:
            return pd.DataFrame(columns=Config.DATAFRAME_COLUMNS['signals'])


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

        # Max profit tracking (entry to end of market day)
        self.max_price_to_eod = entry_price  # Track max price from entry to end of day
        self.max_stop_loss_price = entry_price  # Track lowest price during holding (worst stop loss point)

        # Take profit milestone tracking
        self.highest_milestone_pct = None  # Highest milestone reached during holding

    def update(self, timestamp, price, stock_price=None):
        """Update position with new price data during holding period."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.last_update = timestamp
        self.last_stock_price = stock_price
        # Track lowest price during holding (worst stop loss point)
        self.max_stop_loss_price = min(self.max_stop_loss_price, price)

    def update_eod_price(self, price):
        """Update max price tracking from entry to end of market day (called even after exit)."""
        self.max_price_to_eod = max(self.max_price_to_eod, price)

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
        # Calculate Profit[min] - P&L at the worst stop loss point (lowest price during holding)
        profit_min = self.get_pnl(self.max_stop_loss_price) if self.max_stop_loss_price else None

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
            'max_price_to_eod': self.max_price_to_eod,
            'max_stop_loss_price': self.max_stop_loss_price,
            'profit_min': profit_min,
            'highest_milestone_pct': self.highest_milestone_pct,
        }


# =============================================================================
# INTERNAL - Databook
# =============================================================================

class Databook:
    """High-resolution tracking data for a single trade."""

    def __init__(self, position):
        self.position = position
        self.trade_label = position.get_trade_label()
        self.records = []
        self.day_high = -np.inf
        self.day_low = np.inf

    def add_record(self, timestamp, stock_price, option_price, volume, holding=True,
                   stop_loss=np.nan, stop_loss_mode=None, vwap=np.nan, ema_30=np.nan,
                   vwap_ema_avg=np.nan, emavwap=np.nan, stock_high=np.nan, stock_low=np.nan,
                   ewo=np.nan, ewo_15min_avg=np.nan,
                   rsi=np.nan, rsi_10min_avg=np.nan,
                   supertrend=np.nan, supertrend_direction=np.nan,
                   ichimoku_tenkan=np.nan, ichimoku_kijun=np.nan,
                   ichimoku_senkou_a=np.nan, ichimoku_senkou_b=np.nan,
                   milestone_pct=np.nan, trailing_stop_price=np.nan,
                   risk=None, risk_reasons=None, risk_trend=None,
                   spy_price=np.nan, spy_gauge=None,
                   ai_outlook_1m=None, ai_outlook_5m=None,
                   ai_outlook_30m=None, ai_outlook_1h=None,
                   ai_action=None, ai_reason=None,
                   exit_sig_tp=None, exit_sig_sb=None, exit_sig_mp=None,
                   exit_sig_ai=None, exit_sig_reversal=None, exit_sig_downtrend=None,
                   exit_sig_sl=None, exit_sig_closure_peak=None):
        """Add a tracking record."""
        pnl_pct = self.position.get_pnl_pct(option_price) if holding else np.nan

        # Calculate True Price via Analysis module
        tp = Analysis.true_price(stock_price, stock_high, stock_low)

        # Track running day high/low for sideways band calculation
        if not np.isnan(stock_high):
            self.day_high = max(self.day_high, stock_high)
        if not np.isnan(stock_low) and stock_low > 0:
            self.day_low = min(self.day_low, stock_low)

        # Market bias: +1 (bullish), 0 (sideways), -1 (bearish)
        # Sideways zone = VWAP +/- configurable % of today's high-low range
        bias_band_pct = Config.BACKTEST_CONFIG.get('bias_sideways_band', 0.05)
        if not np.isnan(vwap) and vwap > 0 and self.day_high > self.day_low:
            day_range = self.day_high - self.day_low
            sideways_band = bias_band_pct * day_range
            if stock_price >= vwap + sideways_band:
                market_bias = 1
            elif stock_price <= vwap - sideways_band:
                market_bias = -1
            else:
                market_bias = 0
        else:
            market_bias = np.nan

        # Unpack SPY gauge dict into individual columns
        spy_gauge = spy_gauge or {}

        record = {
            'timestamp': timestamp,
            'stock_price': stock_price,
            'stock_high': stock_high,
            'stock_low': stock_low,
            'true_price': tp,
            'option_price': option_price,
            'volume': volume,
            'holding': holding,
            'entry_price': self.position.entry_price if holding else np.nan,
            'pnl': self.position.get_pnl(option_price) if holding else np.nan,
            'pnl_pct': pnl_pct,
            'highest_price': self.position.highest_price if holding else np.nan,
            'lowest_price': self.position.lowest_price if holding else np.nan,
            'minutes_held': self.position.get_minutes_held(timestamp) if holding else np.nan,
            # Stop loss tracking
            'stop_loss': stop_loss,
            'stop_loss_mode': stop_loss_mode,
            # Take profit milestone tracking
            'milestone_pct': milestone_pct,
            'trailing_stop_price': trailing_stop_price,
            # Risk assessment
            'risk': risk,
            'risk_reasons': risk_reasons,
            'risk_trend': risk_trend,
            # Market assessment
            'market_bias': market_bias,
            # SPY gauge
            'spy_price': spy_price,
            'spy_since_open': spy_gauge.get('since_open'),
            'spy_1m': spy_gauge.get('1m'),
            'spy_5m': spy_gauge.get('5m'),
            'spy_15m': spy_gauge.get('15m'),
            'spy_30m': spy_gauge.get('30m'),
            'spy_1h': spy_gauge.get('1h'),
            # Technical indicators
            'vwap': vwap,
            'ema_30': ema_30,
            'vwap_ema_avg': vwap_ema_avg,
            'emavwap': emavwap,
            'ewo': ewo,
            'ewo_15min_avg': ewo_15min_avg,
            'rsi': rsi,
            'rsi_10min_avg': rsi_10min_avg,
            'supertrend': supertrend,
            'supertrend_direction': supertrend_direction,
            'ichimoku_tenkan': ichimoku_tenkan,
            'ichimoku_kijun': ichimoku_kijun,
            'ichimoku_senkou_a': ichimoku_senkou_a,
            'ichimoku_senkou_b': ichimoku_senkou_b,
            # AI exit signal tracking
            'ai_outlook_1m': ai_outlook_1m,
            'ai_outlook_5m': ai_outlook_5m,
            'ai_outlook_30m': ai_outlook_30m,
            'ai_outlook_1h': ai_outlook_1h,
            'ai_action': ai_action,
            'ai_reason': ai_reason,
            # Exit signal flags (per-bar: which signals would fire)
            'exit_sig_tp': exit_sig_tp,
            'exit_sig_sb': exit_sig_sb,
            'exit_sig_mp': exit_sig_mp,
            'exit_sig_ai': exit_sig_ai,
            'exit_sig_reversal': exit_sig_reversal,
            'exit_sig_downtrend': exit_sig_downtrend,
            'exit_sig_sl': exit_sig_sl,
            'exit_sig_closure_peak': exit_sig_closure_peak,
        }

        self.records.append(record)

    def to_dataframe(self):
        """Convert to DataFrame."""
        if self.records:
            df = pd.DataFrame(self.records)
            df['trade_label'] = self.trade_label

            # Add metadata columns from position
            df['ticker'] = self.position.ticker
            df['strike'] = self.position.strike
            df['option_type'] = self.position.option_type
            df['expiration'] = self.position.expiration
            df['contracts'] = self.position.contracts
            df['entry_time'] = self.position.entry_time
            df['exit_time'] = self.position.exit_time
            df['exit_reason'] = self.position.exit_reason

            # Reorder columns per Config source of truth
            col_order = Config.DATAFRAME_COLUMNS['databook'] + Config.DATAFRAME_COLUMNS['databook_metadata']
            df = df[[c for c in col_order if c in df.columns]]

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
# INTERNAL - StatsBook
# =============================================================================

class StatsBook:
    """
    Historical statistics matrix built when a signal is received.
    Pulls stock history at multiple timeframes (5m, 1h, 1d) to determine
    where we are in the trading span.

    Structure:
        Index (rows): Metric names (Max(H-L), Median(C-O), etc.)
        Columns: '5m', '1h', '1d' (timeframe intervals)
    """

    RANGE = 15  # Tuning variable for nlargest/nsmallest

    TIMEFRAMES = {
        '5m': {'interval': '5m', 'period': '5d'},
        '1h': {'interval': '1h', 'period': '3mo'},
        '1d': {'interval': '1d', 'period': '1y'},
    }

    METRICS = [
        'Max(H-L)', 'Median.Max(H-L)', 'Median(H-L)', 'Min(H-L)', 'Median.Min(H-L)',
        'Max(C-O)', 'Median.Max(C-O)', 'Median(C-O)', 'Min(C-O)', 'Median.Min(C-O)',
        'Max(EWO)', 'Median.Max(EWO)', 'Median(EWO)', 'Min(EWO)', 'Median.Min(EWO)',
        'Max(Vol)', 'Max(Vol)x', 'Median.Max(Vol)', 'Median(Vol)', 'Min(Vol)', 'Median.Min(Vol)',
    ]

    def __init__(self):
        self._cache = {}

    def _fetch_data(self, ticker, interval, period):
        """Fetch and prepare historical data from yfinance."""
        if not YFINANCE_AVAILABLE:
            return None

        cache_key = f"{ticker}_{interval}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(interval=interval, period=period)
            if df.empty:
                return None
            df.columns = df.columns.str.lower()
            df['delta_co'] = round(df['close'] - df['open'], 3)
            df['delta_hl'] = round(df['high'] - df['low'], 3)
            df['volume'] = df['volume'].astype('float64')
            # EWO: EMA(5) - EMA(35) on close price
            ema_fast = df['close'].ewm(span=5, adjust=False).mean()
            ema_slow = df['close'].ewm(span=35, adjust=False).mean()
            df['ewo'] = ema_fast - ema_slow
            self._cache[cache_key] = df
            return df
        except Exception as e:
            print(f"  StatsBook: Error fetching {ticker} ({interval}/{period}): {e}")
            return None

    def _compute_stats(self, df):
        """Compute all stats for a single timeframe DataFrame."""
        R = self.RANGE
        stats = {}

        # H-L Stats
        stats['Max(H-L)'] = round(float(np.max(df['delta_hl'])), 3)
        stats['Median.Max(H-L)'] = round(float(np.median(df['delta_hl'].nlargest(R))), 3)
        stats['Median(H-L)'] = round(float(np.median(df['delta_hl'])), 3)
        stats['Min(H-L)'] = round(float(np.min(df['delta_hl'])), 3)
        stats['Median.Min(H-L)'] = round(float(np.median(df['delta_hl'].nsmallest(R))), 3)

        # C-O Stats
        stats['Max(C-O)'] = round(float(np.max(df['delta_co'])), 3)
        stats['Median.Max(C-O)'] = round(float(np.median(df['delta_co'].nlargest(R))), 3)
        stats['Median(C-O)'] = round(float(np.median(df['delta_co'])), 3)
        stats['Min(C-O)'] = round(float(np.min(df['delta_co'])), 3)
        stats['Median.Min(C-O)'] = round(float(np.median(df['delta_co'].nsmallest(R))), 3)

        # EWO Stats
        stats['Max(EWO)'] = round(float(np.max(df['ewo'])), 3)
        stats['Median.Max(EWO)'] = round(float(np.median(df['ewo'].nlargest(R))), 3)
        stats['Median(EWO)'] = round(float(np.median(df['ewo'])), 3)
        stats['Min(EWO)'] = round(float(np.min(df['ewo'])), 3)
        stats['Median.Min(EWO)'] = round(float(np.median(df['ewo'].nsmallest(R))), 3)

        # Volume Stats
        max_vol = float(np.max(df['volume']))
        med_vol = float(np.median(df['volume']))
        vol_ratio = round(max_vol / med_vol, 2) if med_vol > 0 else 0
        stats['Max(Vol)'] = int(max_vol)
        stats['Max(Vol)x'] = vol_ratio
        stats['Median.Max(Vol)'] = int(np.median(df['volume'].nlargest(R)))
        stats['Median(Vol)'] = int(np.median(df['volume']))
        stats['Min(Vol)'] = int(np.min(df['volume']))
        stats['Median.Min(Vol)'] = int(np.median(df['volume'].nsmallest(R)))

        return stats

    def build(self, ticker):
        """
        Build the StatsBook matrix for a given ticker.

        Returns:
            pd.DataFrame with rows as metric names (Stats) and
            columns as timeframes (5m, 1h, 1d).
        """
        data = {}

        for label, params in self.TIMEFRAMES.items():
            df = self._fetch_data(ticker, params['interval'], params['period'])
            if df is not None and not df.empty:
                data[label] = self._compute_stats(df)
            else:
                data[label] = {m: np.nan for m in self.METRICS}

        statsbook = pd.DataFrame(data, index=self.METRICS)
        statsbook.index.name = 'Stats'

        return statsbook

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


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

        self.lookback_days = adjust_lookback_for_weekend(
            lookback_days or self.config.get('lookback_days', 5)
        )
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.default_contracts = self.config.get('default_contracts', 1)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)
        self.commission_per_contract = self.config.get('commission_per_contract', 0.65)

        # Initialize components
        self.discord_fetcher = DiscordFetcher()
        self.signal_parser = SignalParser()
        self.data_fetcher = HistoricalDataFetcher()
        self.statsbook_builder = StatsBook()

        # Initialize AI exit signal strategy (loads model into GPU if enabled)
        self.ai_strategy = Strategy.AIExitSignal(self.config.get('ai_exit_signal', {}))
        if self.ai_strategy.enabled:
            print("[AI] Loading local AI model for exit signals...")
            self.ai_strategy.load_model()
            print("[AI] Model loaded successfully.")
        else:
            # Always create the optimal exit logger even without AI model
            # so it collects hindsight-based training data every backtest
            import AIModel
            ai_config = self.config.get('ai_exit_signal', {})
            log_dir = ai_config.get('log_dir', 'ai_training_data')
            self.ai_strategy._optimal_logger = AIModel.OptimalExitLogger(log_dir=log_dir)

        # SPY gauge data cache
        self._spy_data = None

        # Results storage
        self.messages_df = None
        self.signals_df = None
        self.positions = []
        self.databooks = {}
        self.statsbooks = {}
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
        global discord_messages_df
        messages_df = self.discord_fetcher.fetch_messages_for_days(self.lookback_days)

        # Save to module-level variable for variable explorer visibility
        discord_messages_df = messages_df.copy() if not messages_df.empty else pd.DataFrame()
        self.messages_df = discord_messages_df

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
        self.databooks = {}
        self.statsbooks = {}

        for idx, signal in self.signals_df.iterrows():
            print(f"\n  [{idx+1}/{len(self.signals_df)}] {signal['ticker']} "
                  f"${signal['strike']} {signal['option_type']}")

            # Build StatsBook for ticker (once per ticker, before processing)
            ticker = signal['ticker']
            if ticker not in self.statsbooks:
                print(f"    Building StatsBook for {ticker}...")
                self.statsbooks[ticker] = self.statsbook_builder.build(ticker)

            position, matrix = self._process_signal(signal)

            if position:
                self.positions.append(position)
                self.databooks[position.get_trade_label()] = matrix

                pnl = position.get_pnl(position.exit_price) if position.exit_price else 0
                print(f"    Exit: {position.exit_reason} | P&L: ${pnl:+.2f}")

        # Step 4: Compile results
        print(f"\n{'='*60}")
        print("COMPILING RESULTS")
        print(f"{'='*60}")

        self.results = self._compile_results()
        self._has_run = True

        # Unload AI model from GPU memory
        if self.ai_strategy.enabled:
            print("[AI] Unloading AI model...")
            self.ai_strategy.unload_model()

        # Display summary in terminal
        self.summary()

        return self.results

    def _fetch_spy_data(self, signal_time):
        """Fetch SPY intraday data for the signal's trading day."""
        spy_config = self.config.get('spy_gauge', {})
        if not spy_config.get('enabled', False):
            return None

        spy_ticker = spy_config.get('ticker', 'SPY')

        # Cache key by date
        signal_date = signal_time.date()
        if self._spy_data is not None and hasattr(self, '_spy_date') and self._spy_date == signal_date:
            return self._spy_data

        try:
            start = signal_time.replace(hour=9, minute=0, second=0, microsecond=0)
            end = signal_time.replace(hour=16, minute=30, second=0, microsecond=0)
            df = self.data_fetcher.fetch_stock_data(spy_ticker, start, end, interval='1m')
            if df is not None and not df.empty:
                self._spy_data = df
                self._spy_date = signal_date
                return df
        except Exception as e:
            print(f"    SPY data fetch failed: {e}")
        return None

    def _calculate_spy_gauge(self, spy_data, timestamp):
        """
        Calculate SPY gauge at a given timestamp.

        For each timeframe, compare current SPY price to average price over that lookback.
        Returns dict: {'since_open': 'Bullish'/'Bearish', '1m': ..., '5m': ..., etc.}
        Also returns current spy_price.
        """
        if spy_data is None or spy_data.empty:
            return np.nan, {}

        # Get SPY data up to current timestamp
        available = spy_data[spy_data.index <= timestamp]
        if available.empty:
            return np.nan, {}

        spy_price = available['close'].iloc[-1]
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)

        spy_config = self.config.get('spy_gauge', {})
        timeframes = spy_config.get('timeframes', {})
        gauge = {}

        for label, minutes in timeframes.items():
            if minutes == 0:
                # Since open
                lookback_start = market_open
            else:
                lookback_start = timestamp - timedelta(minutes=minutes)

            window = available[(available.index >= lookback_start) & (available.index <= timestamp)]
            if window.empty or len(window) < 1:
                gauge[label] = None
                continue

            avg_price = window['close'].mean()
            gauge[label] = 'Bullish' if spy_price >= avg_price else 'Bearish'

        return spy_price, gauge

    def _assess_risk(self, rsi, rsi_avg, ewo_avg, statsbook, timestamp, signal_time):
        """
        Assess risk at entry time.

        Conditions (any TRUE = HIGH risk):
        1. (RSI + RSI_avg) / 2 > 80
        2. EWO_avg > Median.Max(EWO) from StatsBook (1m:5m = 5m value / 5)
        3. Purchase during first 15 minutes of market open (9:30-9:45 EST)

        Returns: (risk_level, reasons_list)
        """
        risk_config = self.config.get('risk_assessment', {})
        if not risk_config.get('enabled', False):
            return None, None

        reasons = []

        # Condition 1: RSI overbought
        rsi_threshold = risk_config.get('rsi_overbought_threshold', 80)
        if not np.isnan(rsi) and not np.isnan(rsi_avg):
            combined_rsi = (rsi + rsi_avg) / 2
            if combined_rsi > rsi_threshold:
                reasons.append(f'RSI({combined_rsi:.0f}>{rsi_threshold})')

        # Condition 2: EWO overbought vs Median.Max(EWO) from StatsBook
        if risk_config.get('ewo_overbought_enabled', True) and not np.isnan(ewo_avg):
            if statsbook is not None and not statsbook.empty:
                try:
                    # Get Median.Max(EWO) from 5m timeframe, normalize to 1m by dividing by 5
                    ewo_max_5m = float(statsbook.loc['Median.Max(EWO)', '5m'])
                    ewo_max_1m = ewo_max_5m / 5
                    if not np.isnan(ewo_max_1m) and ewo_avg > ewo_max_1m:
                        reasons.append(f'EWO({ewo_avg:.3f}>{ewo_max_1m:.3f})')
                except (KeyError, TypeError, ValueError):
                    pass

        # Condition 3: First 15 minutes of market open
        open_window = risk_config.get('market_open_window_minutes', 15)
        market_open = signal_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_open_end = market_open + timedelta(minutes=open_window)
        if market_open <= timestamp <= market_open_end:
            reasons.append(f'Open({open_window}min)')

        if reasons:
            return 'HIGH', '|'.join(reasons)
        return 'NORMAL', None

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

        # Get entry option price (use intraday precision for 0DTE)
        if signal['expiration']:
            expiry_dt = dt.datetime.combine(
                signal['expiration'], dt.time(16, 0), tzinfo=EASTERN
            )
            days_to_expiry = max(0, (expiry_dt - entry_time).total_seconds() / 86400)
        else:
            days_to_expiry = 30

        if signal.get('cost'):
            entry_option_price = signal.get('cost')
        else:
            entry_option_price = Analysis.estimate_option_price_bs(
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
        matrix = Databook(position)

        # Simulate through all bars
        self._simulate_position(position, matrix, stock_data, signal, entry_idx, entry_stock_price)

        return position, matrix

    def _simulate_position(self, position, matrix, stock_data, signal, entry_idx, entry_stock_price):
        """Simulate position through historical data."""
        # Calculate expiry as a precise datetime (market close at 16:00 ET)
        # so that 0DTE and intraday options get fractional T instead of T=0
        if signal['expiration']:
            expiry_dt = dt.datetime.combine(
                signal['expiration'], dt.time(16, 0), tzinfo=EASTERN
            )
        else:
            expiry_dt = position.entry_time + dt.timedelta(days=30)

        # Fractional days from entry to expiry (used once for offset calibration)
        entry_days_to_expiry = max(0, (expiry_dt - position.entry_time).total_seconds() / 86400)

        # Get indicator settings from config
        indicator_config = self.config.get('indicators', {})
        ema_period = indicator_config.get('ema_period', 30)

        # Get supertrend settings from config
        supertrend_period = indicator_config.get('supertrend_period', 10)
        supertrend_multiplier = indicator_config.get('supertrend_multiplier', 3.0)

        # Get ichimoku settings from config
        ichimoku_tenkan = indicator_config.get('ichimoku_tenkan', 9)
        ichimoku_kijun = indicator_config.get('ichimoku_kijun', 26)
        ichimoku_senkou_b = indicator_config.get('ichimoku_senkou_b', 52)
        ichimoku_displacement = indicator_config.get('ichimoku_displacement', 26)

        # Add technical indicators to stock data
        stock_data = Analysis.add_indicators(stock_data, ema_period=ema_period,
                                             supertrend_period=supertrend_period,
                                             supertrend_multiplier=supertrend_multiplier,
                                             ichimoku_tenkan=ichimoku_tenkan,
                                             ichimoku_kijun=ichimoku_kijun,
                                             ichimoku_senkou_b=ichimoku_senkou_b,
                                             ichimoku_displacement=ichimoku_displacement)

        # Get stop loss settings from config
        SL_config = self.config.get('stop_loss', {})
        SL_enabled = SL_config.get('enabled', True)
        SL_pct = SL_config.get('stop_loss_pct', 0.30)
        SL_trailing_trigger_pct = SL_config.get('trailing_trigger_pct', 0.50)
        SL_trailing_stop_pct = SL_config.get('trailing_stop_pct', 0.30)
        SL_breakeven_min_minutes = SL_config.get('breakeven_min_minutes', 30)
        SL_reversal_exit_enabled = SL_config.get('reversal_exit_enabled', True)
        SL_downtrend_exit_enabled = SL_config.get('downtrend_exit_enabled', True)

        # Get Take Profit - Milestones settings from config (start with normal)
        tp_config = self.config.get('take_profit_milestones', {})
        tp_strategy = Strategy.TakeProfitMilestones(tp_config)
        tp_tracker = tp_strategy.create_tracker(position.entry_price)

        # Get Momentum Peak settings from config
        mp_strategy = Strategy.MomentumPeak(self.config.get('momentum_peak', {}))
        mp_detector = mp_strategy.create_detector()

        # Get StatsBook Exit settings from config
        sb_strategy = Strategy.StatsBookExit(self.config.get('statsbook_exit', {}))
        sb_detector = sb_strategy.create_detector(self.statsbooks.get(position.ticker))

        # Get AI Exit Signal detector for this position
        ai_detector = self.ai_strategy.create_detector(
            ticker=position.ticker,
            option_type=position.option_type,
            strike=position.strike,
        )

        # Get Closure - Peak settings from config
        CP_config = self.config.get('closure_peak', {})
        CP_enabled = CP_config.get('enabled', True)
        CP_rsi_call = CP_config.get('rsi_call_threshold', 87)
        CP_rsi_put = CP_config.get('rsi_put_threshold', 13)
        CP_minutes = CP_config.get('minutes_before_close', 30)
        # Calculate closure window start time (e.g., 15:30 for 30 mins before 16:00)
        CP_start_hour = 15
        CP_start_minute = 60 - CP_minutes
        CP_start_time = dt.time(CP_start_hour, CP_start_minute)

        # Risk assessment state
        risk_level = None
        risk_reasons = None
        risk_trend = None
        risk_assessed = False
        risk_config = self.config.get('risk_assessment', {})
        risk_enabled = risk_config.get('enabled', False)
        risk_downtrend_bars = risk_config.get('downtrend_monitor_bars', 3)
        risk_downtrend_drop_pct = risk_config.get('downtrend_drop_pct', 10)
        risk_downtrend_reason = risk_config.get('downtrend_exit_reason', 'SL-DT')
        risk_negative_bar_count = 0  # Consecutive negative bars after entry for HIGH risk
        risk_tp_switched = False     # Whether TP/SL have been switched for this trade

        # Fetch SPY data for this signal's trading day
        spy_data = self._fetch_spy_data(signal['signal_time'])

        # Initialize stop loss manager
        SL_manager = Strategy.StopLoss(
            entry_price=position.entry_price,
            stop_loss_pct=SL_pct,
            trailing_trigger_pct=SL_trailing_trigger_pct,
            trailing_stop_pct=SL_trailing_stop_pct,
            breakeven_min_minutes=SL_breakeven_min_minutes,
            option_type=position.option_type
        )

        max_vwap_ema_avg = np.nan  # Running max of (VWAP+EMA+High)/3

        for i, (timestamp, bar) in enumerate(stock_data.iterrows()):
            stock_price = bar['close']
            stock_high = bar.get('high', stock_price)
            stock_low = bar.get('low', stock_price)
            volume = bar.get('volume', 0)

            # Get indicator values for this bar
            vwap = bar.get('vwap', np.nan)
            ema_30 = bar.get('ema_30', np.nan)
            current_vwap_ema_avg = bar.get('vwap_ema_avg', np.nan)

            # Track running max of (VWAP+EMA+High)/3
            if not np.isnan(current_vwap_ema_avg):
                if np.isnan(max_vwap_ema_avg):
                    max_vwap_ema_avg = current_vwap_ema_avg
                else:
                    max_vwap_ema_avg = max(max_vwap_ema_avg, current_vwap_ema_avg)
            vwap_ema_avg = max_vwap_ema_avg
            emavwap = bar.get('emavwap', np.nan)
            ewo = bar.get('ewo', np.nan)
            ewo_15min_avg = bar.get('ewo_15min_avg', np.nan)
            rsi = bar.get('rsi', np.nan)
            rsi_10min_avg = bar.get('rsi_10min_avg', np.nan)
            st_value = bar.get('supertrend', np.nan)
            st_direction = bar.get('supertrend_direction', np.nan)
            ichi_tenkan = bar.get('ichimoku_tenkan', np.nan)
            ichi_kijun = bar.get('ichimoku_kijun', np.nan)
            ichi_senkou_a = bar.get('ichimoku_senkou_a', np.nan)
            ichi_senkou_b = bar.get('ichimoku_senkou_b', np.nan)

            current_days_to_expiry = max(0, (expiry_dt - timestamp).total_seconds() / 86400)

            # Estimate option price
            option_price = Analysis.estimate_option_price_bs(
                stock_price=stock_price,
                strike=position.strike,
                option_type=position.option_type,
                days_to_expiry=current_days_to_expiry,
                entry_price=position.entry_price,
                entry_stock_price=entry_stock_price,
                entry_days_to_expiry=entry_days_to_expiry
            )

            # Determine holding status
            is_pre_entry = i < entry_idx
            is_post_exit = position.is_closed
            holding = not is_pre_entry and not is_post_exit

            # Track max price from entry to end of market day (even after exit)
            if not is_pre_entry:
                position.update_eod_price(option_price)

            # Calculate SPY gauge for current bar
            spy_price, spy_gauge_data = self._calculate_spy_gauge(spy_data, timestamp)

            if holding:
                position.update(timestamp, option_price, stock_price)

                # --- Risk Assessment (once at entry) ---
                if not risk_assessed and risk_enabled:
                    risk_assessed = True
                    risk_level, risk_reasons = self._assess_risk(
                        rsi, rsi_10min_avg, ewo_15min_avg,
                        self.statsbooks.get(position.ticker),
                        timestamp, signal['signal_time']
                    )
                    if risk_level == 'HIGH':
                        print(f"    RISK: HIGH [{risk_reasons}]")

                # --- Risk trend monitoring (only for HIGH risk trades) ---
                if risk_level == 'HIGH' and not position.is_closed:
                    pnl_pct_now = position.get_pnl_pct(option_price)

                    # Determine trend: Uptrend if option above entry, Downtrend if below
                    if option_price >= position.entry_price:
                        risk_trend = 'Uptrend'
                    else:
                        risk_trend = 'Downtrend'

                    # Switch TP/SL based on risk trend (once per trend direction change)
                    if not risk_tp_switched:
                        risk_tp_switched = True
                        if risk_trend == 'Uptrend':
                            # Switch to TIGHT take profit milestones
                            tight_milestones = tp_config.get('milestones_tight')
                            if tight_milestones:
                                tp_tracker = Strategy.MilestoneTracker(
                                    sorted(tight_milestones, key=lambda m: m['gain_pct']),
                                    position.entry_price
                                )
                                print(f"    RISK: Uptrend -> TP switched to TIGHT")
                        elif risk_trend == 'Downtrend':
                            # Switch to TIGHT stop loss
                            sl_tight = self.config.get('stop_loss_tight', {})
                            if sl_tight:
                                SL_manager = Strategy.StopLoss(
                                    entry_price=position.entry_price,
                                    stop_loss_pct=sl_tight.get('stop_loss_pct', 0.15),
                                    trailing_trigger_pct=sl_tight.get('trailing_trigger_pct', 0.15),
                                    trailing_stop_pct=sl_tight.get('trailing_stop_pct', 0.15),
                                    breakeven_min_minutes=sl_tight.get('breakeven_min_minutes', 15),
                                    option_type=position.option_type
                                )
                                print(f"    RISK: Downtrend -> SL switched to TIGHT")

                    # Track consecutive negative bars for risk downtrend exit
                    if risk_trend == 'Downtrend':
                        if pnl_pct_now < 0:
                            risk_negative_bar_count += 1
                        else:
                            risk_negative_bar_count = 0

                        # Exit: 3 consecutive negative bars OR 10% drop below entry
                        if not position.is_closed:
                            if risk_negative_bar_count >= risk_downtrend_bars:
                                exit_price = option_price * (1 - self.slippage_pct)
                                position.close(exit_price, timestamp, risk_downtrend_reason)
                            elif pnl_pct_now <= -risk_downtrend_drop_pct:
                                exit_price = option_price * (1 - self.slippage_pct)
                                position.close(exit_price, timestamp, risk_downtrend_reason)

                # Update stop loss and check if triggered
                SL_price = np.nan
                SL_mode = None
                SL_triggered = False
                SL_reversal = False
                SL_downtrend = False

                if SL_enabled:
                    minutes_held = position.get_minutes_held(timestamp)
                    true_price = Analysis.true_price(stock_price, stock_high, stock_low)
                    SL_result = SL_manager.update(
                        option_price, minutes_held=minutes_held,
                        true_price=true_price, vwap=vwap, ema=ema_30,
                        emavwap=emavwap, vwap_ema_avg=vwap_ema_avg
                    )
                    SL_price = SL_result['stop_loss']
                    SL_mode = SL_result['mode']
                    SL_triggered = SL_result['triggered']
                    SL_reversal = SL_result['reversal']
                    SL_downtrend = SL_result['downtrend']

                # Update take profit milestone tracker
                tp_exit = False
                tp_reason = None
                cur_milestone_pct = np.nan
                cur_trailing_price = np.nan
                if tp_tracker:
                    tp_exit, tp_reason = tp_tracker.update(option_price)
                    if tp_tracker.current_milestone_pct is not None:
                        cur_milestone_pct = tp_tracker.current_milestone_pct
                        cur_trailing_price = tp_tracker.trailing_exit_price

                # Update momentum peak detector
                mp_exit = False
                mp_reason = None
                if mp_detector and not position.is_closed:
                    mp_pnl = position.get_pnl_pct(option_price)
                    mp_exit, mp_reason = mp_detector.update(mp_pnl, rsi, rsi_10min_avg, ewo)

                # Update StatsBook detector
                sb_exit = False
                sb_reason = None
                if sb_detector and not position.is_closed:
                    sb_pnl = position.get_pnl_pct(option_price)
                    sb_exit, sb_reason = sb_detector.update(sb_pnl, ewo, stock_high, stock_low)

                # Update AI exit signal detector
                ai_exit = False
                ai_reason = None
                ai_signal_data = {}
                if ai_detector and not position.is_closed:
                    ai_bar_data = {
                        'stock_price': stock_price,
                        'stock_high': stock_high,
                        'stock_low': stock_low,
                        'true_price': Analysis.true_price(stock_price, stock_high, stock_low),
                        'volume': volume,
                        'option_price': option_price,
                        'pnl_pct': position.get_pnl_pct(option_price),
                        'vwap': vwap,
                        'ema_30': ema_30,
                        'ewo': ewo,
                        'ewo_15min_avg': ewo_15min_avg,
                        'rsi': rsi,
                        'rsi_10min_avg': rsi_10min_avg,
                        'supertrend_direction': st_direction,
                        'market_bias': matrix.records[-1]['market_bias'] if matrix.records else np.nan,
                        'ichimoku_tenkan': ichi_tenkan,
                        'ichimoku_kijun': ichi_kijun,
                        'ichimoku_senkou_a': ichi_senkou_a,
                        'ichimoku_senkou_b': ichi_senkou_b,
                    }
                    ai_exit, ai_reason = ai_detector.update(
                        bar_data=ai_bar_data,
                        pnl_pct=position.get_pnl_pct(option_price),
                        minutes_held=position.get_minutes_held(timestamp),
                        option_price=option_price,
                        timestamp=timestamp,
                    )
                    ai_signal_data = ai_detector.current_signal

                # Compute Closure-Peak signal flag (last 30 min RSI-based)
                cp_signal = False
                if CP_enabled and not np.isnan(rsi_10min_avg) and timestamp.time() >= CP_start_time:
                    if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= CP_rsi_call:
                        cp_signal = True
                    elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= CP_rsi_put:
                        cp_signal = True

                # Record tracking data with stop loss, indicators, risk, SPY, AI signals, and exit signal flags
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=True,
                    stop_loss=SL_price,
                    stop_loss_mode=SL_mode,
                    vwap=vwap,
                    ema_30=ema_30,
                    vwap_ema_avg=vwap_ema_avg,
                    emavwap=emavwap,
                    stock_high=stock_high,
                    stock_low=stock_low,
                    ewo=ewo,
                    ewo_15min_avg=ewo_15min_avg,
                    rsi=rsi,
                    rsi_10min_avg=rsi_10min_avg,
                    supertrend=st_value,
                    supertrend_direction=st_direction,
                    ichimoku_tenkan=ichi_tenkan,
                    ichimoku_kijun=ichi_kijun,
                    ichimoku_senkou_a=ichi_senkou_a,
                    ichimoku_senkou_b=ichi_senkou_b,
                    milestone_pct=cur_milestone_pct,
                    trailing_stop_price=cur_trailing_price,
                    risk=risk_level,
                    risk_reasons=risk_reasons,
                    risk_trend=risk_trend,
                    spy_price=spy_price,
                    spy_gauge=spy_gauge_data,
                    ai_outlook_1m=ai_signal_data.get('outlook_1m'),
                    ai_outlook_5m=ai_signal_data.get('outlook_5m'),
                    ai_outlook_30m=ai_signal_data.get('outlook_30m'),
                    ai_outlook_1h=ai_signal_data.get('outlook_1h'),
                    ai_action=ai_signal_data.get('action'),
                    ai_reason=ai_signal_data.get('reason'),
                    # Exit signal flags: which signals would fire this bar
                    exit_sig_tp=tp_exit,
                    exit_sig_sb=sb_exit,
                    exit_sig_mp=mp_exit,
                    exit_sig_ai=ai_exit,
                    exit_sig_reversal=SL_enabled and SL_reversal_exit_enabled and SL_reversal,
                    exit_sig_downtrend=SL_enabled and SL_downtrend_exit_enabled and SL_downtrend,
                    exit_sig_sl=SL_enabled and SL_triggered,
                    exit_sig_closure_peak=cp_signal,
                )

                # Take Profit - Milestones: trailing stop triggered
                if tp_exit and not position.is_closed:
                    # Fill at the trailing stop price, not the current bar price.
                    # The stop order triggers at the trailing level; using option_price
                    # would fill at whatever the bar gapped down to, understating profits.
                    exit_price = tp_tracker.trailing_exit_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, tp_reason)

                # StatsBook Exit: EWO or H-L range at historical extreme
                elif sb_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, sb_reason)

                # Momentum Peak: RSI overbought reversal + EWO decline
                elif mp_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, mp_reason)

                # AI Exit Signal: local LLM recommends selling
                elif ai_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'AI Exit Signal')

                # Check for reversal exit (True Price < VWAP)
                elif SL_enabled and SL_reversal_exit_enabled and SL_reversal and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    exit_reason = self._format_exit_reason('stop_loss_reversal')
                    position.close(exit_price, timestamp, exit_reason)

                # Check for downtrend exit (True Price & EMA < vwap_ema_avg)
                elif SL_enabled and SL_downtrend_exit_enabled and SL_downtrend and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    exit_reason = self._format_exit_reason('stop_loss_downtrend')
                    position.close(exit_price, timestamp, exit_reason)

                # Check for stop loss exit (only if stop loss is enabled)
                elif SL_enabled and SL_triggered and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    exit_reason = self._format_exit_reason(f'stop_loss_{SL_mode}')
                    position.close(exit_price, timestamp, exit_reason)

                # Closure - Peak: Avg RSI (10min) based exit in last 30 minutes of trading day
                elif CP_enabled and not position.is_closed and not np.isnan(rsi_10min_avg) and timestamp.time() >= CP_start_time:
                    if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= CP_rsi_call:
                        exit_price = option_price * (1 - self.slippage_pct)
                        position.close(exit_price, timestamp, 'Closure - Peak')
                    elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= CP_rsi_put:
                        exit_price = option_price * (1 - self.slippage_pct)
                        position.close(exit_price, timestamp, 'Closure - Peak')

                # Exit at market close
                elif timestamp.time() >= dt.time(15, 55) and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'Closure-Market')
            else:
                # Record tracking data for non-holding periods (pre-entry or post-exit)
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=False,
                    stop_loss=np.nan,
                    stop_loss_mode=None,
                    vwap=vwap,
                    ema_30=ema_30,
                    vwap_ema_avg=vwap_ema_avg,
                    emavwap=emavwap,
                    stock_high=stock_high,
                    stock_low=stock_low,
                    ewo=ewo,
                    ewo_15min_avg=ewo_15min_avg,
                    rsi=rsi,
                    rsi_10min_avg=rsi_10min_avg,
                    supertrend=st_value,
                    supertrend_direction=st_direction,
                    ichimoku_tenkan=ichi_tenkan,
                    ichimoku_kijun=ichi_kijun,
                    ichimoku_senkou_a=ichi_senkou_a,
                    ichimoku_senkou_b=ichi_senkou_b,
                    spy_price=spy_price,
                    spy_gauge=spy_gauge_data,
                )

        # Close at end of data if still open
        if not position.is_closed:
            exit_price = position.current_price * (1 - self.slippage_pct)
            position.close(exit_price, stock_data.index[-1], 'Closure-Market')

        # Record highest milestone reached during holding
        if tp_tracker and tp_tracker.current_milestone_pct is not None:
            position.highest_milestone_pct = tp_tracker.current_milestone_pct

        # Finalize AI inference log with trade outcome
        if ai_detector and self.ai_strategy.logger is not None:
            final_pnl = position.get_pnl_pct(position.exit_price) if position.exit_price else np.nan
            self.ai_strategy.logger.finalize_trade(
                trade_label=ai_detector.trade_label,
                exit_reason=position.exit_reason or 'unknown',
                final_pnl_pct=final_pnl,
                exit_price=position.exit_price or np.nan,
            )

        # Log optimal exit data (runs every backtest, even without AI model)
        if self.ai_strategy.optimal_logger is not None and matrix.records:
            final_pnl_pct = position.get_pnl_pct(position.exit_price) if position.exit_price else np.nan
            self.ai_strategy.optimal_logger.log_trade(
                databook_records=matrix.records,
                position_info={
                    'trade_label': position.get_trade_label(),
                    'ticker': position.ticker,
                    'option_type': position.option_type,
                    'strike': position.strike,
                    'expiration': position.expiration,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'exit_reason': position.exit_reason,
                    'pnl_pct': float(final_pnl_pct) if not np.isnan(final_pnl_pct) else None,
                },
            )

    def _format_exit_reason(self, reason):
        """
        Format exit reason to user-friendly display string.

        Mappings:
        - stop_loss_initial -> "SL - Initial"
        - stop_loss_breakeven -> "SL - Breakeven"
        - stop_loss_trailing -> "SL - Trailing"
        - stop_loss_reversal -> "SL - Reversal"
        - stop_loss_downtrend -> "SL - DownTrend"
        - closure_peak -> "Closure - Peak"
        - market_close -> "Closure-Market"
        """
        if reason is None:
            return 'Unknown'

        # Stop loss exits
        if reason == 'stop_loss_initial':
            return 'SL - Initial'
        elif reason == 'stop_loss_breakeven':
            return 'SL - Breakeven'
        elif reason == 'stop_loss_trailing':
            return 'SL - Trailing'
        elif reason == 'stop_loss_reversal':
            return 'SL - Reversal'
        elif reason == 'stop_loss_downtrend':
            return 'SL - DownTrend'

        # Closure - Peak
        elif reason == 'closure_peak':
            return 'Closure - Peak'

        # Closure-Market
        elif reason == 'market_close':
            return 'Closure-Market'

        # Return original if no mapping found
        return reason

    def _compile_results(self):
        """Compile backtest results into DataFrames."""
        signals_df = self.signals_df.copy() if self.signals_df is not None else pd.DataFrame()

        if self.positions:
            positions_data = [p.to_dict() for p in self.positions]
            positions_df = pd.DataFrame(positions_data)
            # Reorder columns per Config source of truth
            col_order = Config.DATAFRAME_COLUMNS['positions']
            positions_df = positions_df[[c for c in col_order if c in positions_df.columns]]
        else:
            positions_df = pd.DataFrame()

        databooks = {}
        for label, databook in self.databooks.items():
            databooks[label] = databook.to_dataframe()

        summary = self._calculate_summary(positions_df)

        return {
            'Signals': signals_df,
            'Positions': positions_df,
            'Databooks': databooks,
            'StatsBooks': dict(self.statsbooks),
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

        # Calculate capital metrics
        # Capital per position = entry_price * 100 (option multiplier) * contracts
        positions_df['position_cost'] = positions_df['entry_price'] * 100 * positions_df['contracts']
        total_capital_utilized = positions_df['position_cost'].sum()

        # Calculate max capital held at any one time (considering overlapping positions)
        max_capital_held = self._calculate_max_capital_held(positions_df)

        # Calculate Capitalized P&L: (Total P&L + Capital Utilized) / Capital Utilized
        capitalized_pnl = (total_pnl + total_capital_utilized) / total_capital_utilized if total_capital_utilized > 0 else 0

        # Calculate Profit[min] - sum of P&L at worst stop loss point for all trades
        total_profit_min = closed_trades['profit_min'].sum() if 'profit_min' in closed_trades.columns else 0

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
            'exit_reasons': closed_trades['exit_reason'].value_counts().to_dict() if len(closed_trades) > 0 else {},
            'initial_capital': self.initial_capital,
            'total_capital_utilized': total_capital_utilized,
            'max_capital_held': max_capital_held,
            'capitalized_pnl': capitalized_pnl,
            'profit_min': total_profit_min
        }

    def _calculate_max_capital_held(self, positions_df):
        """
        Calculate maximum capital held at any one time.

        Tracks overlapping positions to find peak capital usage.
        """
        if positions_df.empty:
            return 0.0

        # Create events list: (+cost at entry, -cost at exit)
        events = []
        for _, row in positions_df.iterrows():
            cost = row['entry_price'] * 100 * row['contracts']
            entry_time = row['entry_time']
            exit_time = row['exit_time']

            if pd.notna(entry_time):
                events.append((entry_time, cost))  # Add capital at entry
            if pd.notna(exit_time):
                events.append((exit_time, -cost))  # Release capital at exit

        if not events:
            return 0.0

        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Track running capital and find maximum
        running_capital = 0.0
        max_capital = 0.0

        for _, amount in events:
            running_capital += amount
            max_capital = max(max_capital, running_capital)

        return max_capital

    def _empty_results(self):
        """Return empty results structure."""
        return {
            'Signals': pd.DataFrame(),
            'Positions': pd.DataFrame(),
            'Databooks': {},
            'StatsBooks': {},
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

        print(f"\nCAPITAL:")
        print(f"  Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
        print(f"  Capital Utilized: ${summary.get('total_capital_utilized', 0):,.2f}")
        print(f"  Max Capital Held: ${summary.get('max_capital_held', 0):,.2f}")
        print(f"  Capitalized P&L: {summary.get('capitalized_pnl', 0):.2%}")

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
        print(f"  Best Trade: ${summary.get('best_trade', 0):+,.2f}")
        print(f"  Worst Trade: ${summary.get('worst_trade', 0):+,.2f}")
        print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")

        print(f"\nTIMING:")
        print(f"  Avg Hold Time: {summary.get('average_minutes_held', 0):.1f} minutes")

        print(f"\nSTATISTICS:")
        print(f"  Profit[min]: ${summary.get('profit_min', 0):+,.2f}")

        print(f"\nEXIT REASONS:")
        for reason, count in summary.get('exit_reasons', {}).items():
            print(f"  {reason}: {count}")

        print(f"{'='*60}\n")

    def get_databooks(self):
        """Get all trade databooks."""
        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return {}

        return self.results.get('Databooks', {})

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

    def BT_Save(self, filepath=None):
        """
        Save backtest results to a pickle file for Dashboard visualization.

        Args:
            filepath: Path to save the pickle file. If None, saves to BT_DATA.pkl
                     in the same directory as Dashboard.py
        """
        import pickle
        import os

        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return False

        # If no filepath specified, save to same directory as Dashboard.py
        if filepath is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, 'BT_DATA.pkl')

        try:
            # Format data for Dashboard.py which expects 'matrices' key
            dashboard_data = {
                'matrices': self.results.get('Databooks', {}),
                'statsbooks': self.results.get('StatsBooks', {}),
            }
            with open(filepath, 'wb') as f:
                pickle.dump(dashboard_data, f)
            print(f"Backtest results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving backtest results: {e}")
            return False


def test_options_pricing():
    """
    Test the Black-Scholes options pricing for both CALL and PUT options.

    This function demonstrates and validates the pricing model with
    various scenarios for both calls and puts.
    """
    print("\n" + "=" * 60)
    print("OPTIONS PRICING TEST - Black-Scholes Model")
    print("=" * 60)

    # Test parameters
    stock_price = 100.0
    strike = 100.0  # ATM option
    days_to_expiry = 30
    volatility = 0.30
    risk_free_rate = 0.05

    # Convert days to years for Black-Scholes
    T = days_to_expiry / 365

    print(f"\nTest Parameters:")
    print(f"  Stock Price: ${stock_price:.2f}")
    print(f"  Strike: ${strike:.2f}")
    print(f"  Days to Expiry: {days_to_expiry}")
    print(f"  Volatility: {volatility * 100:.0f}%")
    print(f"  Risk-Free Rate: {risk_free_rate * 100:.1f}%")

    # Test 1: ATM Call vs Put pricing
    print("\n" + "-" * 40)
    print("Test 1: ATM Options (Stock = Strike)")
    print("-" * 40)

    call_price = Analysis.black_scholes_call(stock_price, strike, T, risk_free_rate, volatility)
    put_price = Analysis.black_scholes_put(stock_price, strike, T, risk_free_rate, volatility)

    print(f"  CALL Price: ${call_price:.4f}")
    print(f"  PUT Price:  ${put_price:.4f}")

    # Verify put-call parity: C - P = S - K*e^(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = stock_price - strike * np.exp(-risk_free_rate * T)
    print(f"\n  Put-Call Parity Check:")
    print(f"    C - P = ${parity_lhs:.4f}")
    print(f"    S - Ke^(-rT) = ${parity_rhs:.4f}")
    print(f"    Parity holds: {abs(parity_lhs - parity_rhs) < 0.01}")

    # Test 2: ITM and OTM options
    print("\n" + "-" * 40)
    print("Test 2: ITM vs OTM Options")
    print("-" * 40)

    itm_call_strike = 95  # ITM call (stock > strike)
    otm_call_strike = 105  # OTM call (stock < strike)

    itm_call = Analysis.black_scholes_call(stock_price, itm_call_strike, T, risk_free_rate, volatility)
    otm_call = Analysis.black_scholes_call(stock_price, otm_call_strike, T, risk_free_rate, volatility)

    itm_put = Analysis.black_scholes_put(stock_price, otm_call_strike, T, risk_free_rate, volatility)  # ITM put (stock < strike)
    otm_put = Analysis.black_scholes_put(stock_price, itm_call_strike, T, risk_free_rate, volatility)  # OTM put (stock > strike)

    print(f"  ITM CALL (K=95):  ${itm_call:.4f}")
    print(f"  OTM CALL (K=105): ${otm_call:.4f}")
    print(f"  ITM PUT (K=105):  ${itm_put:.4f}")
    print(f"  OTM PUT (K=95):   ${otm_put:.4f}")

    # Test 3: Greeks for puts
    print("\n" + "-" * 40)
    print("Test 3: Greeks Comparison (CALL vs PUT)")
    print("-" * 40)

    call_greeks = Analysis.calculate_greeks(stock_price, strike, T, risk_free_rate, volatility, 'CALL')
    put_greeks = Analysis.calculate_greeks(stock_price, strike, T, risk_free_rate, volatility, 'PUT')

    print(f"\n  CALL Greeks:")
    print(f"    Delta: {call_greeks['delta']:.4f}")
    print(f"    Gamma: {call_greeks['gamma']:.4f}")
    print(f"    Theta: ${call_greeks['theta']:.4f}/day")
    print(f"    Vega:  ${call_greeks['vega']:.4f}/1% vol")
    print(f"    Rho:   ${call_greeks['rho']:.4f}/1% rate")

    print(f"\n  PUT Greeks:")
    print(f"    Delta: {put_greeks['delta']:.4f}")
    print(f"    Gamma: {put_greeks['gamma']:.4f}")
    print(f"    Theta: ${put_greeks['theta']:.4f}/day")
    print(f"    Vega:  ${put_greeks['vega']:.4f}/1% vol")
    print(f"    Rho:   ${put_greeks['rho']:.4f}/1% rate")

    # Verify delta relationship: Delta_call - Delta_put = 1
    delta_diff = call_greeks['delta'] - put_greeks['delta']
    print(f"\n  Delta relationship (Call - Put should = 1): {delta_diff:.4f}")

    # Test 4: Price movement simulation for PUTS
    print("\n" + "-" * 40)
    print("Test 4: PUT Price Movement Simulation")
    print("-" * 40)

    entry_stock = 100.0
    entry_put_price = Analysis.black_scholes_put(entry_stock, strike, T, risk_free_rate, volatility)
    print(f"  Entry: Stock=${entry_stock:.2f}, PUT=${entry_put_price:.4f}")

    # Simulate stock drop (good for puts)
    for stock_move in [-10, -5, 0, 5, 10]:
        new_stock = entry_stock + stock_move
        new_put = Analysis.estimate_option_price_bs(new_stock, strike, 'PUT', days_to_expiry,
                                        entry_put_price, entry_stock, volatility)
        pnl = (new_put - entry_put_price) / entry_put_price * 100
        direction = "↑" if pnl > 0 else "↓" if pnl < 0 else "→"
        print(f"  Stock ${new_stock:>6.2f} ({stock_move:+.0f}): PUT=${new_put:.4f} ({direction} {pnl:+.1f}%)")

    # Test 5: Expiration scenarios
    print("\n" + "-" * 40)
    print("Test 5: At Expiration (T=0)")
    print("-" * 40)

    for test_stock in [90, 95, 100, 105, 110]:
        call_exp = Analysis.black_scholes_call(test_stock, strike, 0, risk_free_rate, volatility)
        put_exp = Analysis.black_scholes_put(test_stock, strike, 0, risk_free_rate, volatility)
        print(f"  Stock=${test_stock}: CALL=${call_exp:.2f}, PUT=${put_exp:.2f}")

    print("\n" + "=" * 60)
    print("OPTIONS PRICING TEST COMPLETE")
    print("=" * 60 + "\n")

    return True


def sync_from_github():
    """
    Sync matching files from GitHub to the local folder.
    Returns the number of files that were updated (0 = already up to date).
    """
    try:
        from Updater import AutoUpdate
        syncer = AutoUpdate(
            github_url="https://github.com/PrimalS3v3N/BOTv2",
            onedrive_path=os.path.dirname(os.path.abspath(__file__)),
        )
        return syncer.sync_via_git() or 0
    except Exception as e:
        print(f"  Sync skipped: {e}")
        return 0


if __name__ == '__main__':
    import sys

    # Check for test flag
    if len(sys.argv) > 1 and sys.argv[1] == '--test-pricing':
        test_options_pricing()
    else:
        # Skip sync check if we already restarted after an update
        if '--post-update' not in sys.argv:
            updated = sync_from_github()
            if updated > 0:
                print(f"\n  {updated} file(s) updated from GitHub — restarting with new code...\n")
                os.execv(sys.executable, [sys.executable] + sys.argv + ['--post-update'])
        bt = Backtest(lookback_days=1)
        bt.run()
        bt.BT_Save()
