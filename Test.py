"""
Test.py - Testing Framework for Discord Trading Signals

Module Goal: Backtest, live test, and replay trading signals.

Three Testing Modes:
1. Backtest   - Historical simulation using yfinance data + Discord signals
2. LiveTest   - Real-time data collection via Robinhood webscraping + live simulation
3. LiveRerun  - Replay collected live data through simulation (or fallback to Backtest)

All three modes share SimulationEngine for identical exit strategy logic.

Data Output (signal-based, not trade-based):
- DataBook:    Full bar-by-bar stock + option + indicator data per signal
- DataSummary: 2-minute aggregated summary for dashboard display
- DataStats:   Per-signal trade statistics

Usage:
    # Backtesting
    from Test import Backtest
    bt = Backtest(lookback_days=5)
    results = bt.run()
    bt.summary()

    # Live Testing
    from Test import LiveTest
    lt = LiveTest()
    lt.start()   # Runs until market close
    lt.summary()

    # Live Rerun (with collected data)
    from Test import LiveRerun
    lr = LiveRerun(live_data_path='live_data/LT_DATA.pkl')
    lr.run()
    lr.summary()

    # Live Rerun (no collected data - falls back to Backtest)
    lr = LiveRerun()
    lr.run()

================================================================================
INTERNAL - Testing Logic (Backtest / LiveTest / LiveRerun)
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

    def update(self, timestamp, price, stock_price=None):
        """Update position with new price data during holding period."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.last_update = timestamp
        self.last_stock_price = stock_price

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
                   vwap=np.nan,
                   ema_10=np.nan, ema_21=np.nan, ema_50=np.nan, ema_100=np.nan, ema_200=np.nan,
                   vwap_ema_avg=np.nan, emavwap=np.nan, stock_high=np.nan, stock_low=np.nan,
                   ewo=np.nan, ewo_15min_avg=np.nan,
                   rsi=np.nan, rsi_10min_avg=np.nan,
                   supertrend=np.nan, supertrend_direction=np.nan,
                   ichimoku_tenkan=np.nan, ichimoku_kijun=np.nan,
                   ichimoku_senkou_a=np.nan, ichimoku_senkou_b=np.nan,
                   atr_sl=np.nan,
                   macd_line=np.nan, macd_signal_line=np.nan, macd_histogram=np.nan,
                   roc=np.nan,
                   risk=None, risk_reasons=None, risk_trend=None,
                   spy_price=np.nan, spy_gauge=None, ticker_gauge=None,
                   ai_outlook_1m=None, ai_outlook_5m=None,
                   ai_outlook_30m=None, ai_outlook_1h=None,
                   ai_action=None, ai_reason=None,
                   oe_state=None,
                   exit_sig_sb=None, exit_sig_mp=None,
                   exit_sig_ai=None,
                   exit_sig_closure_peak=None,
                   exit_sig_oe=None):
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
        ticker_gauge = ticker_gauge or {}

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
            # Ticker gauge
            'ticker_since_open': ticker_gauge.get('since_open'),
            'ticker_1m': ticker_gauge.get('1m'),
            'ticker_5m': ticker_gauge.get('5m'),
            'ticker_15m': ticker_gauge.get('15m'),
            'ticker_30m': ticker_gauge.get('30m'),
            'ticker_1h': ticker_gauge.get('1h'),
            # Technical indicators
            'vwap': vwap,
            'ema_10': ema_10,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'ema_100': ema_100,
            'ema_200': ema_200,
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
            'atr_sl': atr_sl,
            # MACD indicator columns
            'macd_line': macd_line,
            'macd_signal': macd_signal_line,
            'macd_histogram': macd_histogram,
            # Price momentum (ROC)
            'roc': roc,
            # AI exit signal tracking
            'ai_outlook_1m': ai_outlook_1m,
            'ai_outlook_5m': ai_outlook_5m,
            'ai_outlook_30m': ai_outlook_30m,
            'ai_outlook_1h': ai_outlook_1h,
            'ai_action': ai_action,
            'ai_reason': ai_reason,
            # Options Exit System columns (SL = stop loss, TP = take profit)
            'sl_trailing': (oe_state or {}).get('sl_trailing', np.nan),
            'sl_hard': (oe_state or {}).get('sl_hard', np.nan),
            'tp_risk_outlook': (oe_state or {}).get('tp_risk_outlook'),
            'tp_risk_reasons': (oe_state or {}).get('tp_risk_reasons'),
            'tp_trend_30m': (oe_state or {}).get('tp_trend_30m'),
            'sl_ema_reversal': (oe_state or {}).get('sl_ema_reversal'),
            'tp_confirmed': (oe_state or {}).get('tp_confirmed'),
            # Exit signal flags (per-bar: which signals would fire)
            'exit_sig_sb': exit_sig_sb,
            'exit_sig_mp': exit_sig_mp,
            'exit_sig_ai': exit_sig_ai,
            'exit_sig_closure_peak': exit_sig_closure_peak,
            'exit_sig_oe': exit_sig_oe,
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
# INTERNAL - DataSummary (2-minute aggregation of DataBook)
# =============================================================================

class DataSummary:
    """
    Aggregated summary of DataBook data on a 2-minute basis.

    Reduces compute for dashboard display of large live datasets.
    Each row covers a 2-minute window with OHLC stock/option prices,
    averaged indicators, and exit signal counts.

    Signal-based: keyed by signal_id, each signal gets its own summary.
    """

    def __init__(self, interval_minutes=2):
        self.interval_minutes = interval_minutes
        self.summaries = {}  # signal_id -> list of summary records

    def update_from_databook(self, signal_id, databook_df):
        """
        Generate summary from a databook DataFrame using vectorized pandas aggregation.

        Args:
            signal_id: Unique signal identifier
            databook_df: DataFrame from Databook.to_dataframe()
        """
        if databook_df is None or databook_df.empty:
            return

        if 'timestamp' not in databook_df.columns:
            return

        df = databook_df
        timestamps = pd.to_datetime(df['timestamp'])

        # Group by intervals using floor
        intervals = timestamps.dt.floor(f'{self.interval_minutes}min')

        # Pre-check which columns exist (once, not per group)
        has_stock = 'stock_price' in df.columns
        has_option = 'option_price' in df.columns
        has_volume = 'volume' in df.columns
        has_pnl = 'pnl_pct' in df.columns
        has_rsi = 'rsi' in df.columns
        has_ewo = 'ewo' in df.columns
        has_vwap = 'vwap' in df.columns
        has_bias = 'market_bias' in df.columns

        exit_sig_cols = [c for c in ['exit_sig_sb', 'exit_sig_mp', 'exit_sig_ai',
                                      'exit_sig_closure_peak', 'exit_sig_oe'] if c in df.columns]

        records = []
        for interval_start, idx_group in df.groupby(intervals).groups.items():
            group = df.loc[idx_group]
            interval_end = interval_start + timedelta(minutes=self.interval_minutes)

            # Count exit signals fired in this window
            exit_signals = []
            for sig_col in exit_sig_cols:
                if group[sig_col].any():
                    exit_signals.append(sig_col.replace('exit_sig_', ''))

            record = {
                'timestamp_start': interval_start,
                'timestamp_end': interval_end,
                'stock_open': group['stock_price'].iloc[0] if has_stock else np.nan,
                'stock_high': group['stock_price'].max() if has_stock else np.nan,
                'stock_low': group['stock_price'].min() if has_stock else np.nan,
                'stock_close': group['stock_price'].iloc[-1] if has_stock else np.nan,
                'option_open': group['option_price'].iloc[0] if has_option else np.nan,
                'option_high': group['option_price'].max() if has_option else np.nan,
                'option_low': group['option_price'].min() if has_option else np.nan,
                'option_close': group['option_price'].iloc[-1] if has_option else np.nan,
                'volume_sum': group['volume'].sum() if has_volume else 0,
                'bar_count': len(group),
                'pnl_pct_start': group['pnl_pct'].iloc[0] if has_pnl else np.nan,
                'pnl_pct_end': group['pnl_pct'].iloc[-1] if has_pnl else np.nan,
                'pnl_pct_max': group['pnl_pct'].max() if has_pnl else np.nan,
                'pnl_pct_min': group['pnl_pct'].min() if has_pnl else np.nan,
                'rsi_avg': group['rsi'].mean() if has_rsi else np.nan,
                'ewo_avg': group['ewo'].mean() if has_ewo else np.nan,
                'vwap_avg': group['vwap'].mean() if has_vwap else np.nan,
                'market_bias_mode': group['market_bias'].mode().iloc[0] if has_bias and not group['market_bias'].mode().empty else np.nan,
                'exit_signals_fired': '|'.join(exit_signals) if exit_signals else None,
                'signal_id': signal_id,
            }
            records.append(record)

        self.summaries[signal_id] = records

    def to_dataframe(self, signal_id=None):
        """
        Get summary as DataFrame.

        Args:
            signal_id: If provided, return summary for specific signal.
                      If None, return all summaries combined.
        """
        if signal_id and signal_id in self.summaries:
            records = self.summaries[signal_id]
        else:
            records = []
            for sig_records in self.summaries.values():
                records.extend(sig_records)

        if records:
            df = pd.DataFrame(records)
            col_order = Config.DATAFRAME_COLUMNS.get('datasummary', [])
            if col_order:
                df = df[[c for c in col_order if c in df.columns]]
            return df
        return pd.DataFrame()


# =============================================================================
# INTERNAL - DataStats (Per-signal trade statistics)
# =============================================================================

class DataStats:
    """
    Per-signal trade statistics.

    Tracks entry/exit data, P&L, risk, and data collection metadata
    for each signal. Signal-based (not trade-based) so future chained
    trades can be grouped under the same signal.
    """

    def __init__(self):
        self.stats = {}  # signal_id -> stats dict

    def record(self, signal_id, position, data_source='backtest', bars_recorded=0):
        """
        Record statistics for a signal.

        Args:
            signal_id: Unique signal identifier
            position: Position object (closed)
            data_source: 'backtest', 'live', or 'livererun'
            bars_recorded: Number of bars in the databook
        """
        if position is None:
            return

        stat = {
            'signal_id': signal_id,
            'ticker': position.ticker,
            'strike': position.strike,
            'option_type': position.option_type,
            'expiration': position.expiration,
            'entry_time': position.entry_time,
            'entry_price': position.entry_price,
            'entry_stock_price': getattr(position, 'last_stock_price', None),
            'exit_time': position.exit_time,
            'exit_price': position.exit_price,
            'exit_reason': position.exit_reason,
            'pnl': position.get_pnl(position.exit_price) if position.exit_price else None,
            'pnl_pct': position.get_pnl_pct(position.exit_price) if position.exit_price else None,
            'minutes_held': position.get_minutes_held(position.exit_time) if position.exit_time else None,
            'max_pnl_pct': None,  # Set from databook later
            'min_pnl_pct': None,
            'max_option_price': position.highest_price,
            'min_option_price': position.lowest_price,
            'risk_level': None,
            'risk_reasons': None,
            'bars_recorded': bars_recorded,
            'data_source': data_source,
        }

        self.stats[signal_id] = stat

    def update_from_databook(self, signal_id, databook_df):
        """Update stats with data from the databook DataFrame."""
        if signal_id not in self.stats or databook_df is None or databook_df.empty:
            return

        if 'pnl_pct' in databook_df.columns:
            holding_df = databook_df[databook_df.get('holding', True) == True]
            if not holding_df.empty:
                self.stats[signal_id]['max_pnl_pct'] = holding_df['pnl_pct'].max()
                self.stats[signal_id]['min_pnl_pct'] = holding_df['pnl_pct'].min()

        if 'risk' in databook_df.columns:
            risk_vals = databook_df['risk'].dropna()
            if not risk_vals.empty:
                self.stats[signal_id]['risk_level'] = risk_vals.iloc[0]

        if 'risk_reasons' in databook_df.columns:
            reason_vals = databook_df['risk_reasons'].dropna()
            if not reason_vals.empty:
                self.stats[signal_id]['risk_reasons'] = reason_vals.iloc[0]

    def to_dataframe(self):
        """Get all stats as DataFrame."""
        if self.stats:
            df = pd.DataFrame(list(self.stats.values()))
            col_order = Config.DATAFRAME_COLUMNS.get('datastats', [])
            if col_order:
                df = df[[c for c in col_order if c in df.columns]]
            return df
        return pd.DataFrame()


# =============================================================================
# INTERNAL - Simulation Engine (Shared between Backtest/LiveTest/LiveRerun)
# =============================================================================

class SimulationEngine:
    """
    Core simulation logic shared across Backtest, LiveTest, and LiveRerun.

    Handles:
    - Position creation from signal + stock data
    - Bar-by-bar simulation with all exit strategies
    - Indicator calculation and databook recording
    - SPY gauge and risk assessment

    All three testing modes (Backtest, LiveTest, LiveRerun) create an
    instance of this engine and call its methods with their respective
    data sources.
    """

    def __init__(self, config=None, ai_strategy=None):
        self.config = config or Config.get_config('backtest').copy()
        self.slippage_pct = self.config.get('slippage_pct', 0.001)
        self.default_contracts = self.config.get('default_contracts', 1)
        self.commission_per_contract = self.config.get('commission_per_contract', 0.65)
        self.ai_strategy = ai_strategy

    def calculate_spy_gauge(self, spy_data, timestamp):
        """
        Calculate SPY gauge at a given timestamp.

        For each timeframe, compare current SPY price to average price over that lookback.
        Returns: (spy_price, gauge_dict)
        """
        if spy_data is None or spy_data.empty:
            return np.nan, {}

        available = spy_data[spy_data.index <= timestamp]
        if available.empty:
            return np.nan, {}

        spy_price = available['close'].iloc[-1]
        gauge = self._compute_gauge_dict(available, spy_price, timestamp)

        return spy_price, gauge

    def calculate_ticker_gauge(self, stock_data, timestamp):
        """
        Calculate ticker gauge at a given timestamp.
        Same logic as SPY gauge but applied to the main ticker's stock data.
        Returns: gauge_dict
        """
        available = stock_data[stock_data.index <= timestamp]
        if available.empty:
            return {}

        current_price = available['close'].iloc[-1]
        return self._compute_gauge_dict(available, current_price, timestamp)

    def _compute_gauge_dict(self, available, current_price, timestamp):
        """Shared gauge computation for SPY and ticker gauges."""
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)

        spy_config = self.config.get('spy_gauge', {})
        timeframes = spy_config.get('timeframes', {})
        gauge = {}

        # Use numpy arrays for fast slicing
        close_vals = available['close'].values
        idx_vals = available.index

        for label, minutes in timeframes.items():
            if minutes == 0:
                lookback_start = market_open
            else:
                lookback_start = timestamp - timedelta(minutes=minutes)

            # Use searchsorted for O(log n) window lookup instead of O(n) boolean mask
            start_pos = idx_vals.searchsorted(lookback_start)
            if start_pos >= len(close_vals):
                gauge[label] = None
                continue

            window_vals = close_vals[start_pos:]
            if len(window_vals) == 0:
                gauge[label] = None
                continue

            avg_price = window_vals.mean()
            gauge[label] = 'Bullish' if current_price >= avg_price else 'Bearish'

        return gauge

    def precompute_gauges(self, data, timestamps):
        """
        Pre-compute gauge values for all timestamps at once.

        Much faster than calling calculate_*_gauge per-bar since we avoid
        repeated DataFrame slicing. Uses cumulative sum for O(n) computation.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex
            timestamps: Array of timestamps to compute gauges for

        Returns:
            list of (price, gauge_dict) tuples, one per timestamp
        """
        if data is None or data.empty:
            return [(np.nan, {})] * len(timestamps)

        spy_config = self.config.get('spy_gauge', {})
        timeframes = spy_config.get('timeframes', {})

        close_vals = data['close'].values
        data_index = data.index
        n_data = len(close_vals)

        # Pre-compute cumulative sum for fast windowed averages
        cumsum = np.concatenate([[0], np.cumsum(close_vals)])

        results = []
        for ts in timestamps:
            # Find position of current timestamp
            pos = data_index.searchsorted(ts, side='right')
            if pos == 0:
                results.append((np.nan, {}))
                continue

            price = float(close_vals[pos - 1])
            market_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
            gauge = {}

            for label, minutes in timeframes.items():
                if minutes == 0:
                    lookback_start = market_open
                else:
                    lookback_start = ts - timedelta(minutes=minutes)

                start_pos = data_index.searchsorted(lookback_start)
                end_pos = pos  # exclusive

                if start_pos >= end_pos:
                    gauge[label] = None
                    continue

                # Use cumsum for O(1) average calculation
                window_sum = cumsum[end_pos] - cumsum[start_pos]
                window_len = end_pos - start_pos
                avg_price = window_sum / window_len
                gauge[label] = 'Bullish' if price >= avg_price else 'Bearish'

            results.append((price, gauge))

        return results

    def assess_risk(self, rsi, rsi_avg, ewo_avg, statsbook, timestamp, signal_time):
        """
        Assess risk at entry time.

        Conditions (any TRUE = HIGH risk):
        1. (RSI + RSI_avg) / 2 > 80
        2. EWO_avg > Median.Max(EWO) from StatsBook (1m = 5m value / 5)
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

    def create_position(self, signal, entry_stock_price, entry_time, entry_option_price=None):
        """
        Create a Position from a signal.

        Args:
            signal: Signal dict/series with ticker, strike, option_type, expiration, cost
            entry_stock_price: Stock price at entry
            entry_time: Timestamp of entry
            entry_option_price: Option price at entry (if None, uses BS model or signal cost)

        Returns:
            (Position, entry_option_price)
        """
        if entry_option_price is None:
            if signal.get('expiration'):
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

        position = Position(
            signal=signal,
            entry_price=entry_option_price,
            entry_time=entry_time,
            contracts=self.default_contracts
        )

        return position, entry_option_price

    def simulate_position(self, position, matrix, stock_data, signal, entry_idx, entry_stock_price,
                          statsbooks=None, spy_data=None):
        """
        Simulate position through bar data (shared logic).

        This is the core simulation loop used by Backtest, LiveTest, and LiveRerun.
        It processes each bar from entry to end, applying all exit strategies.

        Args:
            position: Position object
            matrix: Databook object for recording
            stock_data: DataFrame with OHLCV + indicator columns
            signal: Signal dict/series
            entry_idx: Index of entry bar in stock_data
            entry_stock_price: Stock price at entry
            statsbooks: Dict of {ticker: statsbook_df} for StatsBook exits
            spy_data: DataFrame of SPY intraday data for gauge
        """
        statsbooks = statsbooks or {}

        # Calculate expiry as a precise datetime
        if signal['expiration']:
            expiry_dt = dt.datetime.combine(
                signal['expiration'], dt.time(16, 0), tzinfo=EASTERN
            )
        else:
            expiry_dt = position.entry_time + dt.timedelta(days=30)

        entry_days_to_expiry = max(0, (expiry_dt - position.entry_time).total_seconds() / 86400)

        # Get indicator settings from config
        indicator_config = self.config.get('indicators', {})
        ema_periods = indicator_config.get('ema_periods', [10, 21, 50, 100, 200])

        # Get supertrend settings
        supertrend_period = indicator_config.get('supertrend_period', 10)
        supertrend_multiplier = indicator_config.get('supertrend_multiplier', 3.0)

        # Get ichimoku settings
        ichimoku_tenkan = indicator_config.get('ichimoku_tenkan', 9)
        ichimoku_kijun = indicator_config.get('ichimoku_kijun', 26)
        ichimoku_senkou_b = indicator_config.get('ichimoku_senkou_b', 52)
        ichimoku_displacement = indicator_config.get('ichimoku_displacement', 26)

        # Get ATR-SL settings
        atr_sl_period = indicator_config.get('atr_sl_period', 5)
        atr_sl_hhv = indicator_config.get('atr_sl_hhv', 10)
        atr_sl_multiplier = indicator_config.get('atr_sl_multiplier', 2.5)

        # Get MACD and ROC settings
        oe_config = self.config.get('options_exit', {})
        macd_fast = oe_config.get('macd_fast', 12)
        macd_slow = oe_config.get('macd_slow', 26)
        macd_signal_period = oe_config.get('macd_signal', 9)
        roc_period = oe_config.get('roc_period', 30)

        # Add technical indicators to stock data
        stock_data = Analysis.add_indicators(stock_data, ema_periods=ema_periods,
                                             supertrend_period=supertrend_period,
                                             supertrend_multiplier=supertrend_multiplier,
                                             ichimoku_tenkan=ichimoku_tenkan,
                                             ichimoku_kijun=ichimoku_kijun,
                                             ichimoku_senkou_b=ichimoku_senkou_b,
                                             ichimoku_displacement=ichimoku_displacement,
                                             atr_sl_period=atr_sl_period,
                                             atr_sl_hhv=atr_sl_hhv,
                                             atr_sl_multiplier=atr_sl_multiplier,
                                             macd_fast=macd_fast,
                                             macd_slow=macd_slow,
                                             macd_signal=macd_signal_period,
                                             roc_period=roc_period)

        # Pre-compute SPY and ticker gauges for all bars at once (O(n) vs O(n^2))
        _all_timestamps = stock_data.index
        _spy_gauges = self.precompute_gauges(spy_data, _all_timestamps)
        _ticker_gauges = self.precompute_gauges(stock_data, _all_timestamps)

        # Create strategy detectors
        mp_strategy = Strategy.MomentumPeak(self.config.get('momentum_peak', {}))
        mp_detector = mp_strategy.create_detector()

        sb_strategy = Strategy.StatsBookExit(self.config.get('statsbook_exit', {}))
        sb_detector = sb_strategy.create_detector(statsbooks.get(position.ticker))

        ai_detector = None
        if self.ai_strategy:
            ai_detector = self.ai_strategy.create_detector(
                ticker=position.ticker,
                option_type=position.option_type,
                strike=position.strike,
            )

        oe_strategy = Strategy.OptionsExit(self.config.get('options_exit', {}))
        oe_detector = oe_strategy.create_detector(position.entry_price, position.option_type)
        oe_favorability_assessed = False

        # Closure - Peak settings
        CP_config = self.config.get('closure_peak', {})
        CP_enabled = CP_config.get('enabled', True)
        CP_rsi_call = CP_config.get('rsi_call_threshold', 87)
        CP_rsi_put = CP_config.get('rsi_put_threshold', 13)
        CP_minutes = CP_config.get('minutes_before_close', 30)
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
        risk_downtrend_delay_min = risk_config.get('downtrend_delay_minutes', 5)
        risk_downtrend_bars = risk_config.get('downtrend_monitor_bars', 3)
        risk_downtrend_drop_pct = risk_config.get('downtrend_drop_pct', 10)
        risk_downtrend_reason = risk_config.get('downtrend_exit_reason', 'DownTrend-SL')
        risk_negative_bar_count = 0

        max_vwap_ema_avg = np.nan

        # Pre-extract column arrays for fast indexed access (avoids iterrows overhead)
        _col_close = stock_data['close'].values
        _col_high = stock_data['high'].values if 'high' in stock_data.columns else _col_close
        _col_low = stock_data['low'].values if 'low' in stock_data.columns else _col_close
        _col_volume = stock_data['volume'].values if 'volume' in stock_data.columns else np.zeros(len(stock_data))
        _col_vwap = stock_data['vwap'].values if 'vwap' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ema_10 = stock_data['ema_10'].values if 'ema_10' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ema_21 = stock_data['ema_21'].values if 'ema_21' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ema_50 = stock_data['ema_50'].values if 'ema_50' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ema_100 = stock_data['ema_100'].values if 'ema_100' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ema_200 = stock_data['ema_200'].values if 'ema_200' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_vwap_ema_avg = stock_data['vwap_ema_avg'].values if 'vwap_ema_avg' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_emavwap = stock_data['emavwap'].values if 'emavwap' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ewo = stock_data['ewo'].values if 'ewo' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ewo_15min_avg = stock_data['ewo_15min_avg'].values if 'ewo_15min_avg' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_rsi = stock_data['rsi'].values if 'rsi' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_rsi_10min_avg = stock_data['rsi_10min_avg'].values if 'rsi_10min_avg' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_supertrend = stock_data['supertrend'].values if 'supertrend' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_supertrend_dir = stock_data['supertrend_direction'].values if 'supertrend_direction' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ichi_tenkan = stock_data['ichimoku_tenkan'].values if 'ichimoku_tenkan' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ichi_kijun = stock_data['ichimoku_kijun'].values if 'ichimoku_kijun' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ichi_senkou_a = stock_data['ichimoku_senkou_a'].values if 'ichimoku_senkou_a' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_ichi_senkou_b = stock_data['ichimoku_senkou_b'].values if 'ichimoku_senkou_b' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_atr_sl = stock_data['atr_sl'].values if 'atr_sl' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_macd_line = stock_data['macd_line'].values if 'macd_line' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_macd_signal = stock_data['macd_signal'].values if 'macd_signal' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_macd_hist = stock_data['macd_histogram'].values if 'macd_histogram' in stock_data.columns else np.full(len(stock_data), np.nan)
        _col_roc = stock_data['roc'].values if 'roc' in stock_data.columns else np.full(len(stock_data), np.nan)
        _timestamps = stock_data.index

        for i in range(len(stock_data)):
            timestamp = _timestamps[i]
            stock_price = float(_col_close[i])
            stock_high = float(_col_high[i])
            stock_low = float(_col_low[i])
            volume = float(_col_volume[i])

            # Get indicator values (direct array access, no dict lookup)
            vwap = float(_col_vwap[i])
            ema_10 = float(_col_ema_10[i])
            ema_21 = float(_col_ema_21[i])
            ema_50 = float(_col_ema_50[i])
            ema_100 = float(_col_ema_100[i])
            ema_200 = float(_col_ema_200[i])
            current_vwap_ema_avg = float(_col_vwap_ema_avg[i])

            if not np.isnan(current_vwap_ema_avg):
                if np.isnan(max_vwap_ema_avg):
                    max_vwap_ema_avg = current_vwap_ema_avg
                else:
                    max_vwap_ema_avg = max(max_vwap_ema_avg, current_vwap_ema_avg)
            vwap_ema_avg = max_vwap_ema_avg
            emavwap = float(_col_emavwap[i])
            ewo = float(_col_ewo[i])
            ewo_15min_avg = float(_col_ewo_15min_avg[i])
            rsi = float(_col_rsi[i])
            rsi_10min_avg = float(_col_rsi_10min_avg[i])
            st_value = float(_col_supertrend[i])
            st_direction = float(_col_supertrend_dir[i])
            ichi_tenkan = float(_col_ichi_tenkan[i])
            ichi_kijun = float(_col_ichi_kijun[i])
            ichi_senkou_a = float(_col_ichi_senkou_a[i])
            ichi_senkou_b = float(_col_ichi_senkou_b[i])
            atr_sl_value = float(_col_atr_sl[i])
            macd_line_val = float(_col_macd_line[i])
            macd_signal_val = float(_col_macd_signal[i])
            macd_histogram_val = float(_col_macd_hist[i])
            roc_val = float(_col_roc[i])

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

            # Track max price from entry to end of market day
            if not is_pre_entry:
                position.update_eod_price(option_price)

            # Use pre-computed gauges (O(1) lookup instead of O(n) per-bar)
            spy_price, spy_gauge_data = _spy_gauges[i]
            _, ticker_gauge_data = _ticker_gauges[i]

            if holding:
                position.update(timestamp, option_price, stock_price)

                # --- Risk Assessment (once at entry) ---
                if not risk_assessed and risk_enabled:
                    risk_assessed = True
                    risk_level, risk_reasons = self.assess_risk(
                        rsi, rsi_10min_avg, ewo_15min_avg,
                        statsbooks.get(position.ticker),
                        timestamp, signal['signal_time']
                    )
                    if risk_level == 'HIGH':
                        print(f"    RISK: HIGH [{risk_reasons}]")

                # --- Options Exit: RiskOutlook (once at entry) ---
                oe_state = {}
                oe_exit = False
                oe_reason = None
                if oe_detector and not oe_favorability_assessed:
                    oe_favorability_assessed = True
                    ema_vals = {10: ema_10, 21: ema_21, 50: ema_50, 100: ema_100, 200: ema_200}
                    open_price = stock_data.iloc[0]['close']
                    roc_day = ((stock_price - open_price) / open_price) * 100 if open_price > 0 else np.nan
                    fav_level, fav_reasons = oe_detector.RiskOutlook(
                        rsi=rsi,
                        rsi_avg=rsi_10min_avg,
                        ema_values=ema_vals,
                        stock_price=stock_price,
                        atr_sl_value=atr_sl_value,
                        macd_histogram=macd_histogram_val,
                        roc_30m=roc_val,
                        roc_day=roc_day,
                        supertrend_direction=st_direction,
                        ewo=ewo,
                        ewo_avg=ewo_15min_avg,
                    )
                    if risk_level == 'HIGH':
                        oe_detector.is_high_risk = True
                    if fav_level == 'HIGH':
                        print(f"    RISK OUTLOOK: HIGH [{fav_reasons}]")

                # --- Options Exit: Per-bar update ---
                if oe_detector and not position.is_closed:
                    ema_vals = {10: ema_10, 21: ema_21, 50: ema_50, 100: ema_100, 200: ema_200}
                    oe_exit, oe_reason, oe_state = oe_detector.update(
                        option_price=option_price,
                        stock_price=stock_price,
                        ema_values=ema_vals,
                    )

                # --- Risk trend monitoring ---
                if risk_level == 'HIGH' and not position.is_closed and position.get_minutes_held(timestamp) >= risk_downtrend_delay_min:
                    pnl_pct_now = position.get_pnl_pct(option_price)

                    if option_price >= position.entry_price:
                        risk_trend = 'Uptrend'
                    else:
                        risk_trend = 'Downtrend'

                    if risk_trend == 'Downtrend':
                        if pnl_pct_now < 0:
                            risk_negative_bar_count += 1
                        else:
                            risk_negative_bar_count = 0

                        if not position.is_closed:
                            if risk_negative_bar_count >= risk_downtrend_bars:
                                exit_price = option_price * (1 - self.slippage_pct)
                                position.close(exit_price, timestamp, risk_downtrend_reason)
                            elif pnl_pct_now <= -risk_downtrend_drop_pct:
                                exit_price = option_price * (1 - self.slippage_pct)
                                position.close(exit_price, timestamp, risk_downtrend_reason)

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
                        'ema_21': ema_21,
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

                # Compute Closure-Peak signal flag
                cp_signal = False
                if CP_enabled and not np.isnan(rsi_10min_avg) and timestamp.time() >= CP_start_time:
                    if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= CP_rsi_call:
                        cp_signal = True
                    elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= CP_rsi_put:
                        cp_signal = True

                # Record tracking data
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=True,
                    vwap=vwap,
                    ema_10=ema_10, ema_21=ema_21, ema_50=ema_50, ema_100=ema_100, ema_200=ema_200,
                    vwap_ema_avg=vwap_ema_avg, emavwap=emavwap,
                    stock_high=stock_high, stock_low=stock_low,
                    ewo=ewo, ewo_15min_avg=ewo_15min_avg,
                    rsi=rsi, rsi_10min_avg=rsi_10min_avg,
                    supertrend=st_value, supertrend_direction=st_direction,
                    ichimoku_tenkan=ichi_tenkan, ichimoku_kijun=ichi_kijun,
                    ichimoku_senkou_a=ichi_senkou_a, ichimoku_senkou_b=ichi_senkou_b,
                    atr_sl=atr_sl_value,
                    macd_line=macd_line_val, macd_signal_line=macd_signal_val,
                    macd_histogram=macd_histogram_val,
                    roc=roc_val,
                    risk=risk_level, risk_reasons=risk_reasons, risk_trend=risk_trend,
                    spy_price=spy_price, spy_gauge=spy_gauge_data, ticker_gauge=ticker_gauge_data,
                    ai_outlook_1m=ai_signal_data.get('outlook_1m'),
                    ai_outlook_5m=ai_signal_data.get('outlook_5m'),
                    ai_outlook_30m=ai_signal_data.get('outlook_30m'),
                    ai_outlook_1h=ai_signal_data.get('outlook_1h'),
                    ai_action=ai_signal_data.get('action'),
                    ai_reason=ai_signal_data.get('reason'),
                    oe_state=oe_state,
                    exit_sig_sb=sb_exit, exit_sig_mp=mp_exit,
                    exit_sig_ai=ai_exit, exit_sig_closure_peak=cp_signal, exit_sig_oe=oe_exit,
                )

                # === EXIT PRIORITY CHAIN ===
                if oe_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, oe_reason)
                elif sb_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, sb_reason)
                elif mp_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, mp_reason)
                elif ai_exit and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'AI Exit Signal')
                elif CP_enabled and not position.is_closed and not np.isnan(rsi_10min_avg) and timestamp.time() >= CP_start_time:
                    if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= CP_rsi_call:
                        exit_price = option_price * (1 - self.slippage_pct)
                        position.close(exit_price, timestamp, 'Closure - Peak')
                    elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= CP_rsi_put:
                        exit_price = option_price * (1 - self.slippage_pct)
                        position.close(exit_price, timestamp, 'Closure - Peak')
                elif timestamp.time() >= dt.time(15, 55) and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'Closure-Market')
            else:
                # Record tracking data for non-holding periods
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    holding=False,
                    vwap=vwap,
                    ema_10=ema_10, ema_21=ema_21, ema_50=ema_50, ema_100=ema_100, ema_200=ema_200,
                    vwap_ema_avg=vwap_ema_avg, emavwap=emavwap,
                    stock_high=stock_high, stock_low=stock_low,
                    ewo=ewo, ewo_15min_avg=ewo_15min_avg,
                    rsi=rsi, rsi_10min_avg=rsi_10min_avg,
                    supertrend=st_value, supertrend_direction=st_direction,
                    ichimoku_tenkan=ichi_tenkan, ichimoku_kijun=ichi_kijun,
                    ichimoku_senkou_a=ichi_senkou_a, ichimoku_senkou_b=ichi_senkou_b,
                    atr_sl=atr_sl_value,
                    macd_line=macd_line_val, macd_signal_line=macd_signal_val,
                    macd_histogram=macd_histogram_val,
                    roc=roc_val,
                    spy_price=spy_price, spy_gauge=spy_gauge_data, ticker_gauge=ticker_gauge_data,
                )

        # Close at end of data if still open
        if not position.is_closed:
            exit_price = position.current_price * (1 - self.slippage_pct)
            position.close(exit_price, stock_data.index[-1], 'Closure-Market')

        # Finalize AI logs
        if ai_detector and self.ai_strategy and self.ai_strategy.logger is not None:
            final_pnl = position.get_pnl_pct(position.exit_price) if position.exit_price else np.nan
            self.ai_strategy.logger.finalize_trade(
                trade_label=ai_detector.trade_label,
                exit_reason=position.exit_reason or 'unknown',
                final_pnl_pct=final_pnl,
                exit_price=position.exit_price or np.nan,
            )

        # Log optimal exit data
        if self.ai_strategy and self.ai_strategy.optimal_logger is not None and matrix.records:
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

    def compile_results(self, signals_df, positions, databooks, statsbooks):
        """
        Compile results into the standard output format.

        Returns:
            dict with keys: Signals, Positions, Databooks, StatsBooks, Summary
        """
        signals_copy = signals_df.copy() if signals_df is not None else pd.DataFrame()

        if positions:
            positions_data = [p.to_dict() for p in positions]
            positions_df = pd.DataFrame(positions_data)
            col_order = Config.DATAFRAME_COLUMNS['positions']
            positions_df = positions_df[[c for c in col_order if c in positions_df.columns]]
        else:
            positions_df = pd.DataFrame()

        databook_dfs = {}
        for label, databook in databooks.items():
            databook_dfs[label] = databook.to_dataframe()

        summary = self._calculate_summary(positions_df)

        return {
            'Signals': signals_copy,
            'Positions': positions_df,
            'Databooks': databook_dfs,
            'StatsBooks': dict(statsbooks),
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

        positions_df = positions_df.copy()
        positions_df['position_cost'] = positions_df['entry_price'] * 100 * positions_df['contracts']
        total_capital_utilized = positions_df['position_cost'].sum()

        max_capital_held = self._calculate_max_capital_held(positions_df)

        capitalized_pnl = (total_pnl + total_capital_utilized) / total_capital_utilized if total_capital_utilized > 0 else 0
        total_profit_min = closed_trades['profit_min'].sum() if 'profit_min' in closed_trades.columns else 0

        return {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0,
            'total_pnl': total_pnl,
            'net_pnl': total_pnl - commission_total,
            'commission_total': commission_total,
            'average_pnl': closed_trades['pnl'].mean() if len(closed_trades) > 0 else 0,
            'best_trade': closed_trades['pnl'].max() if len(closed_trades) > 0 else 0,
            'worst_trade': closed_trades['pnl'].min() if len(closed_trades) > 0 else 0,
            'average_minutes_held': closed_trades['minutes_held'].mean() if len(closed_trades) > 0 else 0,
            'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
            'exit_reasons': closed_trades['exit_reason'].value_counts().to_dict() if len(closed_trades) > 0 else {},
            'initial_capital': self.config.get('initial_capital', 10000.0),
            'total_capital_utilized': total_capital_utilized,
            'max_capital_held': max_capital_held,
            'capitalized_pnl': capitalized_pnl,
            'profit_min': total_profit_min,
        }

    def _calculate_max_capital_held(self, positions_df):
        """Calculate maximum capital held at any one time."""
        if positions_df.empty:
            return 0.0

        # Vectorized: compute costs and extract times
        costs = (positions_df['entry_price'] * 100 * positions_df['contracts']).values
        entry_times = positions_df['entry_time'].values
        exit_times = positions_df['exit_time'].values

        events = []
        for i in range(len(costs)):
            if pd.notna(entry_times[i]):
                events.append((entry_times[i], costs[i]))
            if pd.notna(exit_times[i]):
                events.append((exit_times[i], -costs[i]))

        if not events:
            return 0.0

        events.sort(key=lambda x: x[0])

        running_capital = 0.0
        max_capital = 0.0

        for _, amount in events:
            running_capital += amount
            if running_capital > max_capital:
                max_capital = running_capital

        return max_capital


# =============================================================================
# EXTERNAL - Backtest Class (Main Interface)
# =============================================================================

class Backtest:
    """
    Main backtesting engine.

    Orchestrates:
    - Discord message fetching
    - Signal parsing
    - Historical data retrieval (yfinance)
    - Position simulation via SimulationEngine
    - Results compilation

    Uses SimulationEngine for shared simulation logic with LiveTest/LiveRerun.
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

        # Shared simulation engine
        self.engine = SimulationEngine(config=self.config, ai_strategy=self.ai_strategy)

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

    def _process_signal(self, signal):
        """Process a single signal through the backtest using SimulationEngine."""
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

        # Create position via engine
        position, entry_option_price = self.engine.create_position(
            signal=signal,
            entry_stock_price=entry_stock_price,
            entry_time=entry_time
        )

        # Create tracking matrix
        matrix = Databook(position)

        # Fetch SPY data for this signal's trading day
        spy_data = self._fetch_spy_data(signal['signal_time'])

        # Simulate through all bars using shared engine
        self.engine.simulate_position(
            position=position,
            matrix=matrix,
            stock_data=stock_data,
            signal=signal,
            entry_idx=entry_idx,
            entry_stock_price=entry_stock_price,
            statsbooks=self.statsbooks,
            spy_data=spy_data,
        )

        return position, matrix

    def _compile_results(self):
        """Compile backtest results via SimulationEngine."""
        return self.engine.compile_results(self.signals_df, self.positions, self.databooks, self.statsbooks)

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


# =============================================================================
# EXTERNAL - LiveTest Class (Live Data Collection + Real-time Simulation)
# =============================================================================

class LiveTest:
    """
    Live testing engine - collects real-time data via Robinhood webscraping
    and applies the same exit strategies as Backtest.

    Shares SimulationEngine with Backtest for identical simulation logic.

    Architecture:
    1. Discord signals arrive (same as Backtest)
    2. RobinhoodScraper collects live stock + option prices each cycle
    3. Data is accumulated into DataFrames (1-min bars)
    4. SimulationEngine processes bars with all exit strategies
    5. DataBook/DataSummary/DataStats are updated each cycle

    Output: Three signal-based DataFrames:
    - DataBook:    Full tick-level stock + option data per signal
    - DataSummary: 2-minute aggregated summary for dashboard
    - DataStats:   Per-signal trade statistics

    Usage:
        lt = LiveTest()
        lt.start()
        # Runs continuously until market close or manual stop
        lt.stop()
        lt.summary()
        lt.save()
    """

    def __init__(self, config=None):
        self.config = config or Config.get_config('backtest').copy()
        live_config = Config.get_config('live')
        self.cycle_interval = live_config.get('cycle_interval_seconds', 60)
        self.data_dir = live_config.get('data_dir', 'live_data')
        self.summary_interval = live_config.get('summary_interval_minutes', 2)
        self.auto_save_interval = live_config.get('auto_save_interval', 5)

        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)
        self.default_contracts = self.config.get('default_contracts', 1)

        # Initialize AI strategy (same as Backtest)
        import AIModel
        ai_config = self.config.get('ai_exit_signal', {})
        self.ai_strategy = AIModel.AIExitSignal(ai_config)
        if self.ai_strategy.enabled:
            self.ai_strategy.load_model()
        else:
            log_dir = ai_config.get('log_dir', 'ai_training_data')
            self.ai_strategy._optimal_logger = AIModel.OptimalExitLogger(log_dir=log_dir)

        # Shared simulation engine
        self.engine = SimulationEngine(config=self.config, ai_strategy=self.ai_strategy)

        # Data fetchers
        from Data import RobinhoodScraper, LiveDataFetcher
        self.scraper = RobinhoodScraper()
        self.live_fetcher = LiveDataFetcher(scraper=self.scraper)

        # Discord + Signal parsing (same as Backtest)
        self.discord_fetcher = DiscordFetcher()
        self.signal_parser = SignalParser()
        self.data_fetcher = HistoricalDataFetcher()  # For SPY data

        # StatsBook builder
        self.statsbook_builder = StatsBook()

        # Results storage (signal-based)
        self.signals = {}          # signal_id -> signal dict
        self.positions = {}        # signal_id -> Position
        self.databooks = {}        # signal_id -> Databook
        self.statsbooks = {}       # ticker -> statsbook DataFrame

        # Live data accumulation (per ticker, deduplicated)
        self._stock_bars = {}      # ticker -> list of bar dicts
        self._option_bars = {}     # signal_id -> list of bar dicts
        self._stock_df_cache = {}  # ticker -> (len_at_build, DataFrame) cached stock DFs

        # Per-signal incremental simulation state (avoids O(n^2) full re-simulation)
        self._signal_detectors = {}  # signal_id -> dict of detectors and state
        self._signal_last_bar = {}   # signal_id -> last processed bar index

        # Three output dataframes
        self.data_book = DataSummary(interval_minutes=1)    # Full 1-min data (uses DataSummary container but 1-min interval)
        self.data_summary = DataSummary(interval_minutes=self.summary_interval)
        self.data_stats = DataStats()

        # Control
        self._running = False
        self._cycle_count = 0
        self._start_time = None
        self._spy_data = None

    def start(self):
        """
        Start live testing session.

        1. Authenticate with Robinhood
        2. Fetch initial Discord signals
        3. Begin collection cycle loop
        """
        import os
        os.makedirs(self.data_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("LIVE TEST - Starting")
        print(f"{'='*60}")
        print(f"  Cycle interval: {self.cycle_interval}s")
        print(f"  Data directory: {self.data_dir}")

        # Authenticate
        self.live_fetcher.start()
        self._running = True
        self._start_time = dt.datetime.now(EASTERN)

        # Fetch and parse initial signals
        self._fetch_signals()

        if not self.signals:
            print("  No signals found. Waiting for signals...")

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\n  Interrupted by user")
        finally:
            self.stop()

    def _fetch_signals(self):
        """Fetch Discord signals (same logic as Backtest step 1+2)."""
        print("  Fetching Discord signals...")
        messages = self.discord_fetcher.fetch_messages_for_days(days=1)

        if messages is None or messages.empty:
            print("  No Discord messages found")
            return

        signals_df = self.signal_parser.parse_all_messages(messages)
        if signals_df is None or signals_df.empty:
            print("  No valid signals parsed")
            return

        # Register each signal
        for _, signal in signals_df.iterrows():
            signal_dict = signal.to_dict()
            signal_id = f"{signal_dict['ticker']}:{signal_dict['strike']}:{signal_dict['option_type']}:{signal_dict.get('message_id', '')}"
            signal_dict['signal_id'] = signal_id

            if signal_id not in self.signals:
                self.signals[signal_id] = signal_dict

                # Build StatsBook for ticker if not already done
                ticker = signal_dict['ticker']
                if ticker not in self.statsbooks:
                    print(f"  Building StatsBook for {ticker}...")
                    self.statsbooks[ticker] = self.statsbook_builder.build(ticker)

                # Add to live fetcher for data collection
                self.live_fetcher.add_signal(signal_dict)

                # Initialize position and databook
                print(f"  Registered signal: {signal_id}")

        print(f"  {len(self.signals)} signals active")

    def _run_loop(self):
        """Main collection + simulation cycle loop."""
        market_close = dt.time(16, 0)

        while self._running:
            now = dt.datetime.now(EASTERN)

            # Check market hours (stop after 4:00 PM ET)
            if now.time() >= market_close:
                print(f"\n  Market closed at {now.strftime('%H:%M:%S')}")
                break

            self._cycle_count += 1
            cycle_start = time.time()

            # Collect live data
            cycle_data = self.live_fetcher.collect_cycle()

            # Process collected data
            self._process_cycle(cycle_data)

            # Auto-save periodically
            if self._cycle_count % self.auto_save_interval == 0:
                self._auto_save()

            # Print cycle status
            cycle_ms = (time.time() - cycle_start) * 1000
            active_count = sum(1 for p in self.positions.values() if not p.is_closed)
            print(f"  Cycle {self._cycle_count}: {cycle_ms:.0f}ms | "
                  f"Active: {active_count} | "
                  f"Data: {cycle_data.get('cycle_time_ms', 0):.0f}ms | "
                  f"{now.strftime('%H:%M:%S')}")

            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.cycle_interval - elapsed)
            if sleep_time > 0 and self._running:
                time.sleep(sleep_time)

    def _process_cycle(self, cycle_data):
        """
        Process one data collection cycle (incremental).

        Instead of re-running full simulation from bar 0, we:
        1. Accumulate new bar data
        2. Rebuild indicators on the full stock DataFrame (required for rolling indicators)
        3. Only process the NEW bar through the per-bar simulation logic
        4. Update DataSummary incrementally (only new interval if completed)

        This reduces per-cycle cost from O(n) to O(1) for the simulation loop.
        """
        stock_quotes = cycle_data.get('stock_quotes', {})
        option_quotes = cycle_data.get('option_quotes', {})
        timestamp = cycle_data.get('timestamp', dt.datetime.now(EASTERN))

        for signal_id, signal in self.signals.items():
            ticker = signal['ticker']

            # Get stock data (deduplicated)
            stock_quote = stock_quotes.get(ticker)
            option_quote = option_quotes.get(signal_id)

            if not stock_quote:
                continue

            stock_price = stock_quote['price']
            stock_high = stock_quote.get('high', stock_price)
            stock_low = stock_quote.get('low', stock_price)
            volume = stock_quote.get('volume', 0)

            # Accumulate stock bar
            if ticker not in self._stock_bars:
                self._stock_bars[ticker] = []
            self._stock_bars[ticker].append({
                'timestamp': timestamp,
                'open': stock_quote.get('open', stock_price),
                'high': stock_high,
                'low': stock_low,
                'close': stock_price,
                'volume': volume,
            })

            # Get option price (live from RH, or fallback to BS)
            if option_quote:
                option_price = option_quote.get('mark_price', 0)
            else:
                # Fallback to Black-Scholes estimation
                if signal.get('expiration'):
                    expiry_dt = dt.datetime.combine(
                        signal['expiration'], dt.time(16, 0), tzinfo=EASTERN
                    )
                    days_to_expiry = max(0, (expiry_dt - timestamp).total_seconds() / 86400)
                else:
                    days_to_expiry = 30

                option_price = Analysis.estimate_option_price_bs(
                    stock_price=stock_price,
                    strike=signal['strike'],
                    option_type=signal['option_type'],
                    days_to_expiry=days_to_expiry
                )

            # Accumulate option bar
            if signal_id not in self._option_bars:
                self._option_bars[signal_id] = []
            self._option_bars[signal_id].append({
                'timestamp': timestamp,
                'mark_price': option_price,
                'bid': option_quote.get('bid', 0) if option_quote else 0,
                'ask': option_quote.get('ask', 0) if option_quote else 0,
                'iv': option_quote.get('implied_volatility', 0) if option_quote else 0,
                'delta': option_quote.get('delta', 0) if option_quote else 0,
                'volume': option_quote.get('volume', 0) if option_quote else 0,
            })

            # Create position on first data point (if not already created)
            if signal_id not in self.positions:
                entry_price = signal.get('cost', option_price)
                if entry_price and entry_price > 0:
                    position, _ = self.engine.create_position(
                        signal=signal,
                        entry_stock_price=stock_price,
                        entry_time=timestamp,
                        entry_option_price=entry_price
                    )
                    self.positions[signal_id] = position
                    self.databooks[signal_id] = Databook(position)
                    self._signal_last_bar[signal_id] = -1  # No bars processed yet
                    print(f"    Position opened: {signal_id} @ ${entry_price:.2f}")

            # Update position if open
            position = self.positions.get(signal_id)
            if position and not position.is_closed:
                # Build stock DataFrame with indicators (cached, only rebuilt when new bars added)
                stock_df = self._build_stock_dataframe(ticker)
                if stock_df is not None and len(stock_df) >= 2:
                    matrix = self.databooks[signal_id]

                    # Initialize detectors if first time
                    if signal_id not in self._signal_detectors:
                        self._init_signal_detectors(signal_id, signal, position, stock_df)

                    # Process ONLY the new bar (incremental)
                    last_bar = self._signal_last_bar.get(signal_id, -1)
                    new_bar_idx = len(stock_df) - 1

                    if new_bar_idx > last_bar:
                        # Fetch SPY data (cached)
                        spy_data = self._fetch_spy_data(timestamp)

                        # Process the new bar through incremental simulation
                        self._process_new_bar(
                            signal_id=signal_id,
                            signal=signal,
                            position=position,
                            matrix=matrix,
                            stock_df=stock_df,
                            bar_idx=new_bar_idx,
                            option_price=option_price,
                            spy_data=spy_data,
                        )
                        self._signal_last_bar[signal_id] = new_bar_idx

                    # Update DataStats (lightweight, just updates the record)
                    self.data_stats.record(signal_id, position, data_source='live',
                                          bars_recorded=len(matrix.records))

                    if position.is_closed:
                        # Final DataSummary update on close
                        db_df = matrix.to_dataframe()
                        self.data_summary.update_from_databook(signal_id, db_df)
                        self.data_stats.update_from_databook(signal_id, db_df)
                        print(f"    Position closed: {signal_id} | "
                              f"Reason: {position.exit_reason} | "
                              f"PnL: {position.get_pnl_pct(position.exit_price):+.1f}%")

        # Periodic DataSummary rebuild (every summary_interval cycles, not every cycle)
        if self._cycle_count % self.summary_interval == 0:
            for sig_id, db in self.databooks.items():
                db_df = db.to_dataframe()
                self.data_summary.update_from_databook(sig_id, db_df)

    def _init_signal_detectors(self, signal_id, signal, position, stock_df):
        """Initialize per-signal strategy detectors (once per signal)."""
        config = self.config
        mp_strategy = Strategy.MomentumPeak(config.get('momentum_peak', {}))
        mp_detector = mp_strategy.create_detector()

        sb_strategy = Strategy.StatsBookExit(config.get('statsbook_exit', {}))
        sb_detector = sb_strategy.create_detector(self.statsbooks.get(position.ticker))

        ai_detector = None
        if self.ai_strategy and self.ai_strategy.enabled:
            ai_detector = self.ai_strategy.create_detector(
                ticker=position.ticker,
                option_type=position.option_type,
                strike=position.strike,
            )

        oe_strategy = Strategy.OptionsExit(config.get('options_exit', {}))
        oe_detector = oe_strategy.create_detector(position.entry_price, position.option_type)

        # Risk assessment config
        risk_config = config.get('risk_assessment', {})
        CP_config = config.get('closure_peak', {})

        self._signal_detectors[signal_id] = {
            'mp_detector': mp_detector,
            'sb_detector': sb_detector,
            'ai_detector': ai_detector,
            'oe_detector': oe_detector,
            'oe_favorability_assessed': False,
            'risk_assessed': False,
            'risk_level': None,
            'risk_reasons': None,
            'risk_trend': None,
            'risk_negative_bar_count': 0,
            'max_vwap_ema_avg': np.nan,
            # Config caches
            'risk_config': risk_config,
            'CP_config': CP_config,
            'CP_enabled': CP_config.get('enabled', True),
            'CP_rsi_call': CP_config.get('rsi_call_threshold', 87),
            'CP_rsi_put': CP_config.get('rsi_put_threshold', 13),
            'CP_start_time': dt.time(15, 60 - CP_config.get('minutes_before_close', 30)),
            # Expiry info
            'expiry_dt': (dt.datetime.combine(signal['expiration'], dt.time(16, 0), tzinfo=EASTERN)
                          if signal.get('expiration') else position.entry_time + dt.timedelta(days=30)),
            'entry_days_to_expiry': None,  # Set below
        }
        state = self._signal_detectors[signal_id]
        state['entry_days_to_expiry'] = max(0, (state['expiry_dt'] - position.entry_time).total_seconds() / 86400)

    def _process_new_bar(self, signal_id, signal, position, matrix, stock_df, bar_idx,
                         option_price, spy_data):
        """
        Process a single new bar incrementally (avoids full re-simulation).

        This mirrors the logic in SimulationEngine.simulate_position() but processes
        only one bar, maintaining state in self._signal_detectors[signal_id].
        """
        state = self._signal_detectors[signal_id]
        config = self.config

        timestamp = stock_df.index[bar_idx]
        bar = stock_df.iloc[bar_idx]

        stock_price = float(bar['close'])
        stock_high = float(bar.get('high', stock_price))
        stock_low = float(bar.get('low', stock_price))
        volume = float(bar.get('volume', 0))

        # Get indicator values
        vwap = float(bar.get('vwap', np.nan))
        ema_10 = float(bar.get('ema_10', np.nan))
        ema_21 = float(bar.get('ema_21', np.nan))
        ema_50 = float(bar.get('ema_50', np.nan))
        ema_100 = float(bar.get('ema_100', np.nan))
        ema_200 = float(bar.get('ema_200', np.nan))
        current_vwap_ema_avg = float(bar.get('vwap_ema_avg', np.nan))

        if not np.isnan(current_vwap_ema_avg):
            if np.isnan(state['max_vwap_ema_avg']):
                state['max_vwap_ema_avg'] = current_vwap_ema_avg
            else:
                state['max_vwap_ema_avg'] = max(state['max_vwap_ema_avg'], current_vwap_ema_avg)
        vwap_ema_avg = state['max_vwap_ema_avg']
        emavwap = float(bar.get('emavwap', np.nan))
        ewo = float(bar.get('ewo', np.nan))
        ewo_15min_avg = float(bar.get('ewo_15min_avg', np.nan))
        rsi = float(bar.get('rsi', np.nan))
        rsi_10min_avg = float(bar.get('rsi_10min_avg', np.nan))
        st_value = float(bar.get('supertrend', np.nan))
        st_direction = float(bar.get('supertrend_direction', np.nan))
        ichi_tenkan = float(bar.get('ichimoku_tenkan', np.nan))
        ichi_kijun = float(bar.get('ichimoku_kijun', np.nan))
        ichi_senkou_a = float(bar.get('ichimoku_senkou_a', np.nan))
        ichi_senkou_b = float(bar.get('ichimoku_senkou_b', np.nan))
        atr_sl_value = float(bar.get('atr_sl', np.nan))
        macd_line_val = float(bar.get('macd_line', np.nan))
        macd_signal_val = float(bar.get('macd_signal', np.nan))
        macd_histogram_val = float(bar.get('macd_histogram', np.nan))
        roc_val = float(bar.get('roc', np.nan))

        current_days_to_expiry = max(0, (state['expiry_dt'] - timestamp).total_seconds() / 86400)

        # Use live option price directly (no BS estimation needed)
        # option_price is passed in from the cycle data

        # Track max price from entry to end of market day
        position.update_eod_price(option_price)
        position.update(timestamp, option_price, stock_price)

        # SPY gauge (use engine's method for single bar)
        spy_price, spy_gauge_data = self.engine.calculate_spy_gauge(spy_data, timestamp)
        ticker_gauge_data = self.engine.calculate_ticker_gauge(stock_df, timestamp)

        # --- Risk Assessment (once at entry) ---
        risk_level = state['risk_level']
        risk_reasons = state['risk_reasons']
        risk_trend = state['risk_trend']

        if not state['risk_assessed'] and state['risk_config'].get('enabled', False):
            state['risk_assessed'] = True
            risk_level, risk_reasons = self.engine.assess_risk(
                rsi, rsi_10min_avg, ewo_15min_avg,
                self.statsbooks.get(position.ticker),
                timestamp, signal['signal_time']
            )
            state['risk_level'] = risk_level
            state['risk_reasons'] = risk_reasons
            if risk_level == 'HIGH':
                print(f"    RISK: HIGH [{risk_reasons}]")

        # --- Options Exit: RiskOutlook (once at entry) ---
        oe_state = {}
        oe_exit = False
        oe_reason = None
        oe_detector = state['oe_detector']

        if oe_detector and not state['oe_favorability_assessed']:
            state['oe_favorability_assessed'] = True
            ema_vals = {10: ema_10, 21: ema_21, 50: ema_50, 100: ema_100, 200: ema_200}
            open_price = stock_df.iloc[0]['close']
            roc_day = ((stock_price - open_price) / open_price) * 100 if open_price > 0 else np.nan
            fav_level, fav_reasons = oe_detector.RiskOutlook(
                rsi=rsi, rsi_avg=rsi_10min_avg, ema_values=ema_vals,
                stock_price=stock_price, atr_sl_value=atr_sl_value,
                macd_histogram=macd_histogram_val, roc_30m=roc_val,
                roc_day=roc_day, supertrend_direction=st_direction,
                ewo=ewo, ewo_avg=ewo_15min_avg,
            )
            if risk_level == 'HIGH':
                oe_detector.is_high_risk = True

        # --- Options Exit: Per-bar update ---
        if oe_detector and not position.is_closed:
            ema_vals = {10: ema_10, 21: ema_21, 50: ema_50, 100: ema_100, 200: ema_200}
            oe_exit, oe_reason, oe_state = oe_detector.update(
                option_price=option_price, stock_price=stock_price, ema_values=ema_vals,
            )

        # --- Risk trend monitoring ---
        risk_config = state['risk_config']
        if risk_level == 'HIGH' and not position.is_closed:
            delay_min = risk_config.get('downtrend_delay_minutes', 5)
            if position.get_minutes_held(timestamp) >= delay_min:
                pnl_pct_now = position.get_pnl_pct(option_price)
                risk_trend = 'Uptrend' if option_price >= position.entry_price else 'Downtrend'
                state['risk_trend'] = risk_trend

                if risk_trend == 'Downtrend':
                    if pnl_pct_now < 0:
                        state['risk_negative_bar_count'] += 1
                    else:
                        state['risk_negative_bar_count'] = 0

                    monitor_bars = risk_config.get('downtrend_monitor_bars', 3)
                    drop_pct = risk_config.get('downtrend_drop_pct', 10)
                    reason = risk_config.get('downtrend_exit_reason', 'DownTrend-SL')

                    if not position.is_closed:
                        if state['risk_negative_bar_count'] >= monitor_bars:
                            exit_price = option_price * (1 - self.slippage_pct)
                            position.close(exit_price, timestamp, reason)
                        elif pnl_pct_now <= -drop_pct:
                            exit_price = option_price * (1 - self.slippage_pct)
                            position.close(exit_price, timestamp, reason)

        # Update strategy detectors
        mp_exit, mp_reason = False, None
        mp_detector = state['mp_detector']
        if mp_detector and not position.is_closed:
            mp_pnl = position.get_pnl_pct(option_price)
            mp_exit, mp_reason = mp_detector.update(mp_pnl, rsi, rsi_10min_avg, ewo)

        sb_exit, sb_reason = False, None
        sb_detector = state['sb_detector']
        if sb_detector and not position.is_closed:
            sb_pnl = position.get_pnl_pct(option_price)
            sb_exit, sb_reason = sb_detector.update(sb_pnl, ewo, stock_high, stock_low)

        ai_exit, ai_reason = False, None
        ai_signal_data = {}
        ai_detector = state['ai_detector']
        if ai_detector and not position.is_closed:
            ai_bar_data = {
                'stock_price': stock_price, 'stock_high': stock_high, 'stock_low': stock_low,
                'true_price': Analysis.true_price(stock_price, stock_high, stock_low),
                'volume': volume, 'option_price': option_price,
                'pnl_pct': position.get_pnl_pct(option_price),
                'vwap': vwap, 'ema_21': ema_21, 'ewo': ewo, 'ewo_15min_avg': ewo_15min_avg,
                'rsi': rsi, 'rsi_10min_avg': rsi_10min_avg,
                'supertrend_direction': st_direction,
                'market_bias': matrix.records[-1]['market_bias'] if matrix.records else np.nan,
                'ichimoku_tenkan': ichi_tenkan, 'ichimoku_kijun': ichi_kijun,
                'ichimoku_senkou_a': ichi_senkou_a, 'ichimoku_senkou_b': ichi_senkou_b,
            }
            ai_exit, ai_reason = ai_detector.update(
                bar_data=ai_bar_data, pnl_pct=position.get_pnl_pct(option_price),
                minutes_held=position.get_minutes_held(timestamp),
                option_price=option_price, timestamp=timestamp,
            )
            ai_signal_data = ai_detector.current_signal

        # Closure-Peak signal
        CP_enabled = state['CP_enabled']
        cp_signal = False
        if CP_enabled and not np.isnan(rsi_10min_avg) and timestamp.time() >= state['CP_start_time']:
            if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= state['CP_rsi_call']:
                cp_signal = True
            elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= state['CP_rsi_put']:
                cp_signal = True

        # Record tracking data
        matrix.add_record(
            timestamp=timestamp, stock_price=stock_price, option_price=option_price,
            volume=volume, holding=True, vwap=vwap,
            ema_10=ema_10, ema_21=ema_21, ema_50=ema_50, ema_100=ema_100, ema_200=ema_200,
            vwap_ema_avg=vwap_ema_avg, emavwap=emavwap,
            stock_high=stock_high, stock_low=stock_low,
            ewo=ewo, ewo_15min_avg=ewo_15min_avg, rsi=rsi, rsi_10min_avg=rsi_10min_avg,
            supertrend=st_value, supertrend_direction=st_direction,
            ichimoku_tenkan=ichi_tenkan, ichimoku_kijun=ichi_kijun,
            ichimoku_senkou_a=ichi_senkou_a, ichimoku_senkou_b=ichi_senkou_b,
            atr_sl=atr_sl_value,
            macd_line=macd_line_val, macd_signal_line=macd_signal_val,
            macd_histogram=macd_histogram_val, roc=roc_val,
            risk=risk_level, risk_reasons=risk_reasons, risk_trend=risk_trend,
            spy_price=spy_price, spy_gauge=spy_gauge_data, ticker_gauge=ticker_gauge_data,
            ai_outlook_1m=ai_signal_data.get('outlook_1m'),
            ai_outlook_5m=ai_signal_data.get('outlook_5m'),
            ai_outlook_30m=ai_signal_data.get('outlook_30m'),
            ai_outlook_1h=ai_signal_data.get('outlook_1h'),
            ai_action=ai_signal_data.get('action'),
            ai_reason=ai_signal_data.get('reason'),
            oe_state=oe_state,
            exit_sig_sb=sb_exit, exit_sig_mp=mp_exit,
            exit_sig_ai=ai_exit, exit_sig_closure_peak=cp_signal, exit_sig_oe=oe_exit,
        )

        # === EXIT PRIORITY CHAIN ===
        slippage = self.slippage_pct
        if oe_exit and not position.is_closed:
            position.close(option_price * (1 - slippage), timestamp, oe_reason)
        elif sb_exit and not position.is_closed:
            position.close(option_price * (1 - slippage), timestamp, sb_reason)
        elif mp_exit and not position.is_closed:
            position.close(option_price * (1 - slippage), timestamp, mp_reason)
        elif ai_exit and not position.is_closed:
            position.close(option_price * (1 - slippage), timestamp, 'AI Exit Signal')
        elif CP_enabled and not position.is_closed and not np.isnan(rsi_10min_avg) and timestamp.time() >= state['CP_start_time']:
            if position.option_type.upper() in ['CALL', 'CALLS', 'C'] and rsi_10min_avg >= state['CP_rsi_call']:
                position.close(option_price * (1 - slippage), timestamp, 'Closure - Peak')
            elif position.option_type.upper() in ['PUT', 'PUTS', 'P'] and rsi_10min_avg <= state['CP_rsi_put']:
                position.close(option_price * (1 - slippage), timestamp, 'Closure - Peak')
        elif timestamp.time() >= dt.time(15, 55) and not position.is_closed:
            position.close(option_price * (1 - slippage), timestamp, 'Closure-Market')

    def _build_stock_dataframe(self, ticker):
        """
        Convert accumulated stock bars into a DataFrame with indicators.

        Uses a cache to avoid rebuilding when no new bars have been added.
        Indicators must be recalculated on the full DataFrame because they
        use rolling windows, but the DataFrame construction is cached.
        """
        bars = self._stock_bars.get(ticker, [])
        if not bars:
            return None

        # Check cache: only rebuild if new bars added
        cache = self._stock_df_cache.get(ticker)
        if cache and cache[0] == len(bars):
            return cache[1]  # Return cached DataFrame with indicators

        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()

        # Add indicators (must recalculate on full DF for rolling windows)
        indicator_config = self.config.get('indicators', {})
        oe_config = self.config.get('options_exit', {})
        df = Analysis.add_indicators(
            df,
            ema_periods=indicator_config.get('ema_periods', [10, 21, 50, 100, 200]),
            supertrend_period=indicator_config.get('supertrend_period', 10),
            supertrend_multiplier=indicator_config.get('supertrend_multiplier', 3.0),
            ichimoku_tenkan=indicator_config.get('ichimoku_tenkan', 9),
            ichimoku_kijun=indicator_config.get('ichimoku_kijun', 26),
            ichimoku_senkou_b=indicator_config.get('ichimoku_senkou_b', 52),
            ichimoku_displacement=indicator_config.get('ichimoku_displacement', 26),
            atr_sl_period=indicator_config.get('atr_sl_period', 5),
            atr_sl_hhv=indicator_config.get('atr_sl_hhv', 10),
            atr_sl_multiplier=indicator_config.get('atr_sl_multiplier', 2.5),
            macd_fast=oe_config.get('macd_fast', 12),
            macd_slow=oe_config.get('macd_slow', 26),
            macd_signal=oe_config.get('macd_signal', 9),
            roc_period=oe_config.get('roc_period', 30),
        )

        # Cache with bar count as key
        self._stock_df_cache[ticker] = (len(bars), df)
        return df

    def _fetch_spy_data(self, signal_time):
        """Fetch SPY intraday data (cached)."""
        if self._spy_data is not None:
            return self._spy_data

        try:
            signal_date = signal_time.date() if hasattr(signal_time, 'date') else signal_time
            start_date = signal_date
            end_date = signal_date + timedelta(days=1)
            df = self.data_fetcher.fetch_stock_data('SPY', start_date, end_date, interval='1m')
            if df is not None and not df.empty:
                self._spy_data = df
                return df
        except Exception:
            pass
        return None

    def stop(self):
        """Stop live testing and cleanup."""
        self._running = False
        print(f"\n{'='*60}")
        print("LIVE TEST - Stopped")
        print(f"{'='*60}")
        print(f"  Cycles completed: {self._cycle_count}")
        print(f"  Signals tracked: {len(self.signals)}")

        # Final save
        self.save()

        # Cleanup
        self.live_fetcher.close()
        self.discord_fetcher.close()

    def _auto_save(self):
        """Auto-save live data periodically."""
        try:
            self.save(quiet=True)
        except Exception as e:
            print(f"  Auto-save failed: {e}")

    def save(self, filepath=None, quiet=False):
        """
        Save live test data to pickle.

        Saves: DataBooks (per signal), DataSummary, DataStats,
               positions, raw stock/option bars.
        """
        import pickle
        import os

        if filepath is None:
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, 'LT_DATA.pkl')

        try:
            # Convert databooks to DataFrames
            databook_dfs = {}
            for sig_id, db in self.databooks.items():
                databook_dfs[sig_id] = db.to_dataframe()

            save_data = {
                'mode': 'live',
                'matrices': databook_dfs,                              # DataBook (per signal)
                'data_summary': self.data_summary.to_dataframe(),      # DataSummary
                'data_stats': self.data_stats.to_dataframe(),          # DataStats
                'statsbooks': dict(self.statsbooks),
                'signals': self.signals,
                'stock_bars': self._stock_bars,
                'option_bars': self._option_bars,
                'start_time': self._start_time,
                'cycles': self._cycle_count,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            if not quiet:
                print(f"  Live test data saved to {filepath}")
            return True
        except Exception as e:
            if not quiet:
                print(f"  Error saving live test data: {e}")
            return False

    def summary(self):
        """Print live test summary."""
        stats_df = self.data_stats.to_dataframe()

        print(f"\n{'='*60}")
        print("LIVE TEST SUMMARY")
        print(f"{'='*60}")

        print(f"\n  Signals: {len(self.signals)}")
        closed = sum(1 for p in self.positions.values() if p.is_closed)
        active = sum(1 for p in self.positions.values() if not p.is_closed)
        print(f"  Closed positions: {closed}")
        print(f"  Active positions: {active}")
        print(f"  Cycles: {self._cycle_count}")

        if not stats_df.empty and 'pnl_pct' in stats_df.columns:
            completed = stats_df[stats_df['pnl_pct'].notna()]
            if not completed.empty:
                winners = completed[completed['pnl_pct'] > 0]
                print(f"\n  Win rate: {len(winners)/len(completed)*100:.1f}%")
                print(f"  Avg PnL: {completed['pnl_pct'].mean():+.1f}%")
                print(f"  Best: {completed['pnl_pct'].max():+.1f}%")
                print(f"  Worst: {completed['pnl_pct'].min():+.1f}%")

        print(f"{'='*60}\n")

    def get_databooks(self):
        """Get all signal databooks as DataFrames."""
        return {sig_id: db.to_dataframe() for sig_id, db in self.databooks.items()}

    def get_data_summary(self):
        """Get DataSummary DataFrame."""
        return self.data_summary.to_dataframe()

    def get_data_stats(self):
        """Get DataStats DataFrame."""
        return self.data_stats.to_dataframe()


# =============================================================================
# EXTERNAL - LiveRerun Class (Replay collected live data through simulation)
# =============================================================================

class LiveRerun:
    """
    Re-run simulation on previously collected live data OR use standard
    backtesting data sources.

    Modes:
    1. Live data available: Load from LT_DATA.pkl and replay through SimulationEngine
    2. No live data: Fall back to standard Backtest (yfinance data + Discord signals)

    This allows re-testing exit strategies on real collected data, comparing
    live vs backtest results, or re-running with different config.

    Shares SimulationEngine with Backtest and LiveTest.

    Usage:
        # Mode 1: Replay live collected data
        lr = LiveRerun(live_data_path='live_data/LT_DATA.pkl')
        lr.run()

        # Mode 2: No live data, use Backtest path
        lr = LiveRerun()
        lr.run()  # Equivalent to Backtest.run()
    """

    def __init__(self, live_data_path=None, lookback_days=None, config=None):
        self.config = config or Config.get_config('backtest').copy()
        self.live_data_path = live_data_path
        self.lookback_days = lookback_days or self.config.get('lookback_days', 5)

        self.initial_capital = self.config.get('initial_capital', 10000.0)

        # Initialize AI strategy
        import AIModel
        ai_config = self.config.get('ai_exit_signal', {})
        self.ai_strategy = AIModel.AIExitSignal(ai_config)
        if self.ai_strategy.enabled:
            self.ai_strategy.load_model()
        else:
            log_dir = ai_config.get('log_dir', 'ai_training_data')
            self.ai_strategy._optimal_logger = AIModel.OptimalExitLogger(log_dir=log_dir)

        # Shared engine
        self.engine = SimulationEngine(config=self.config, ai_strategy=self.ai_strategy)

        # Data source detection
        self._has_live_data = False
        self._live_data = None

        if live_data_path:
            self._has_live_data = self._load_live_data(live_data_path)

        # Results storage
        self.positions = []
        self.databooks = {}
        self.statsbooks = {}
        self.signals_df = None
        self.results = None
        self._has_run = False

        # Three output dataframes
        self.data_summary = DataSummary(interval_minutes=Config.get_config('live').get('summary_interval_minutes', 2))
        self.data_stats = DataStats()

    def _load_live_data(self, path):
        """Load previously collected live data."""
        import pickle
        try:
            with open(path, 'rb') as f:
                self._live_data = pickle.load(f)
            print(f"Loaded live data from {path}")
            print(f"  Mode: {self._live_data.get('mode', 'unknown')}")
            print(f"  Signals: {len(self._live_data.get('signals', {}))}")
            print(f"  Cycles: {self._live_data.get('cycles', 0)}")
            return True
        except FileNotFoundError:
            print(f"Live data not found at {path}")
            return False
        except Exception as e:
            print(f"Error loading live data: {e}")
            return False

    def run(self):
        """
        Execute the rerun.

        If live data is available, replay it through SimulationEngine.
        Otherwise, fall back to standard Backtest.
        """
        if self._has_live_data:
            return self._run_live_rerun()
        else:
            return self._run_backtest_fallback()

    def _run_live_rerun(self):
        """Replay collected live data through SimulationEngine."""
        print(f"\n{'='*60}")
        print("LIVE RERUN - Replaying collected data")
        print(f"{'='*60}")

        signals = self._live_data.get('signals', {})
        stock_bars = self._live_data.get('stock_bars', {})
        statsbooks_data = self._live_data.get('statsbooks', {})

        self.statsbooks = statsbooks_data

        # Reconstruct SPY data from saved stock bars
        spy_data = None
        if 'SPY' in stock_bars:
            spy_df = pd.DataFrame(stock_bars['SPY'])
            spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'])
            spy_df = spy_df.set_index('timestamp').sort_index()
            spy_data = spy_df

        print(f"  Signals to replay: {len(signals)}")

        for signal_id, signal in signals.items():
            ticker = signal['ticker']
            print(f"\n  Processing: {signal_id}")

            # Build stock DataFrame from collected bars
            if ticker not in stock_bars or not stock_bars[ticker]:
                print(f"    No stock data for {ticker}")
                continue

            stock_df = pd.DataFrame(stock_bars[ticker])
            stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])
            stock_df = stock_df.set_index('timestamp').sort_index()

            if stock_df.empty:
                print(f"    Empty stock data for {ticker}")
                continue

            # Create position
            entry_stock_price = stock_df.iloc[0]['close']
            entry_time = stock_df.index[0]

            # Use recorded option price if available
            option_bars = self._live_data.get('option_bars', {}).get(signal_id, [])
            entry_option_price = None
            if option_bars:
                entry_option_price = option_bars[0].get('mark_price')

            position, _ = self.engine.create_position(
                signal=signal,
                entry_stock_price=entry_stock_price,
                entry_time=entry_time,
                entry_option_price=entry_option_price
            )

            matrix = Databook(position)

            # Run simulation on the collected data
            self.engine.simulate_position(
                position=position,
                matrix=matrix,
                stock_data=stock_df,
                signal=signal,
                entry_idx=0,
                entry_stock_price=entry_stock_price,
                statsbooks=self.statsbooks,
                spy_data=spy_data,
            )

            # Store results
            self.positions.append(position)
            trade_label = position.get_trade_label()
            self.databooks[trade_label] = matrix

            # Update DataSummary and DataStats
            db_df = matrix.to_dataframe()
            self.data_summary.update_from_databook(signal_id, db_df)
            self.data_stats.record(signal_id, position, data_source='livererun', bars_recorded=len(db_df))
            self.data_stats.update_from_databook(signal_id, db_df)

            # Print result
            if position.is_closed:
                pnl = position.get_pnl_pct(position.exit_price)
                print(f"    Exit: {position.exit_reason} | PnL: {pnl:+.1f}%")

        # Compile results
        self.results = self.engine.compile_results(
            pd.DataFrame(list(signals.values())),
            self.positions, self.databooks, self.statsbooks
        )
        self._has_run = True

        self.summary()
        return self.results

    def _run_backtest_fallback(self):
        """Fall back to standard Backtest when no live data is available."""
        print(f"\n{'='*60}")
        print("LIVE RERUN - No live data found, using Backtest mode")
        print(f"{'='*60}")

        bt = Backtest(lookback_days=self.lookback_days, config=self.config)
        self.results = bt.run()
        self._has_run = True

        # Copy results to match LiveRerun interface
        self.positions = bt.positions
        self.databooks = bt.databooks
        self.statsbooks = bt.statsbooks
        self.signals_df = bt.signals_df

        # Generate DataSummary and DataStats from backtest results
        for label, db in self.databooks.items():
            db_df = db.to_dataframe() if hasattr(db, 'to_dataframe') else db
            self.data_summary.update_from_databook(label, db_df)

        for pos in self.positions:
            sig_id = pos.get_trade_label()
            self.data_stats.record(sig_id, pos, data_source='backtest')
            if sig_id in self.results.get('Databooks', {}):
                self.data_stats.update_from_databook(sig_id, self.results['Databooks'][sig_id])

        return self.results

    def summary(self):
        """Print rerun summary."""
        if not self._has_run:
            print("Rerun has not been executed yet. Call run() first.")
            return

        summary = self.results.get('Summary', {})

        print(f"\n{'='*60}")
        print("LIVE RERUN SUMMARY")
        print(f"{'='*60}")

        if not summary:
            print("  No trades executed")
            return

        data_source = "Live Data" if self._has_live_data else "Backtest (yfinance)"
        print(f"\n  Data Source: {data_source}")

        print(f"\n  TRADES:")
        print(f"    Total: {summary.get('total_trades', 0)}")
        print(f"    Winners: {summary.get('winners', 0)}")
        print(f"    Losers: {summary.get('losers', 0)}")
        print(f"    Win Rate: {summary.get('win_rate', 0):.1f}%")

        print(f"\n  P&L:")
        print(f"    Total: ${summary.get('total_pnl', 0):+,.2f}")
        print(f"    Net: ${summary.get('net_pnl', 0):+,.2f}")
        print(f"    Best: ${summary.get('best_trade', 0):+,.2f}")
        print(f"    Worst: ${summary.get('worst_trade', 0):+,.2f}")

        print(f"\n  EXIT REASONS:")
        for reason, count in summary.get('exit_reasons', {}).items():
            print(f"    {reason}: {count}")

        print(f"{'='*60}\n")

    def save(self, filepath=None):
        """Save rerun results to pickle."""
        import pickle
        import os

        if not self._has_run:
            print("Rerun has not been executed yet.")
            return False

        if filepath is None:
            live_config = Config.get_config('live')
            data_dir = live_config.get('data_dir', 'live_data')
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, 'LR_DATA.pkl')

        try:
            save_data = {
                'mode': 'livererun',
                'matrices': self.results.get('Databooks', {}),
                'data_summary': self.data_summary.to_dataframe(),
                'data_stats': self.data_stats.to_dataframe(),
                'statsbooks': self.results.get('StatsBooks', {}),
                'summary': self.results.get('Summary', {}),
                'has_live_data': self._has_live_data,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"  Rerun results saved to {filepath}")
            return True
        except Exception as e:
            print(f"  Error saving rerun results: {e}")
            return False

    def get_databooks(self):
        """Get all trade databooks."""
        if not self._has_run:
            return {}
        return self.results.get('Databooks', {})

    def get_data_summary(self):
        """Get DataSummary DataFrame."""
        return self.data_summary.to_dataframe()

    def get_data_stats(self):
        """Get DataStats DataFrame."""
        return self.data_stats.to_dataframe()


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
