"""
Test.py - Backtesting Framework for Discord Trading Signals

FUNCTIONALITY:
1. Gets X days of Discord messages from signal channels
2. Parses tickers, strike prices, contracts (calls/puts), and expirations
3. Gets historical options/stock data aligned with signal timestamps
4. Plays out data with high-resolution tracking (<5min candles)
5. Identifies exit opportunities via Strategy.py's ExitDecisionTree
6. Generates detailed trade tracking matrices with stock/option/volume data
7. Provides DataFrames: Signals, Positions, Tracking_matrices of trades

EXIT STRATEGY INTEGRATION:
All exit logic is delegated to Strategy.py's ExitDecisionTree, including:
- Stop Loss, Profit Target (with VWAP conditional), Trailing Stop
- Time Stop, Expiration, Discord Signal
- End of Day Exit (day trading), Max Hold Days (swing trading)
- Technical Indicators: RSI, MACD, VWAP, VPOC, SuperTrend

Usage:
    from Test import Backtest
    bt = Backtest(lookback_days=5)
    results = bt.run()
    bt.summary()

    # Get detailed tracking matrices for each trade
    matrices = bt.get_trade_Data_BT()
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, date
from zoneinfo import ZoneInfo
import time
import requests
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Install with: pip install yfinance")

# Local imports
import Config
import Signal
import Analysis
import Strategy

# Timezone for market hours
EASTERN = ZoneInfo('America/New_York')


# =============================================================================
# DISCORD FETCHER
# =============================================================================

class DiscordFetcher:
    """
    Fetches messages from Discord signal channels.

    Uses Discord API v10 to retrieve historical messages
    from the configured signal channel.
    """

    def __init__(self, token=None, channel_id=None):
        """
        Initialize Discord fetcher.

        Args:
            token: Discord bot token (defaults to Config)
            channel_id: Channel ID to fetch from (defaults to Config)
        """
        discord_config = Config.get_config('discord')
        self.token = token or discord_config.get('token')
        self.channel_id = channel_id or discord_config.get('channel_id')
        self.api_version = discord_config.get('api_version', 'v10')
        self.base_url = discord_config.get('base_url', 'https://discord.com/api')
        self.message_limit = discord_config.get('test_message_limit', 100)

    def fetch_messages(self, limit=100, before_id=None):
        """
        Fetch messages from Discord channel.

        Args:
            limit: Number of messages to fetch (max 100 per request)
            before_id: Fetch messages before this ID (for pagination)

        Returns:
            List of message dictionaries
        """
        url = f"{self.base_url}/{self.api_version}/channels/{self.channel_id}/messages"
        headers = {'Authorization': self.token}
        params = {'limit': min(limit, 100)}

        if before_id:
            params['before'] = before_id

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Discord API error: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Discord request failed: {e}")
            return []

    def fetch_messages_for_days(self, days=5):
        """
        Fetch all messages for the specified number of days.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with columns: id, timestamp, content, author
        """
        cutoff_date = dt.datetime.now(EASTERN) - timedelta(days=days)
        all_messages = []
        before_id = None

        print(f"Fetching Discord messages for last {days} days...")

        while True:
            messages = self.fetch_messages(limit=100, before_id=before_id)

            if not messages:
                break

            for msg in messages:
                try:
                    # Parse timestamp
                    timestamp_str = msg.get('timestamp', '')
                    timestamp = pd.to_datetime(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=EASTERN)
                    else:
                        timestamp = timestamp.astimezone(EASTERN)

                    # Check if within date range
                    if timestamp < cutoff_date:
                        # Found messages older than cutoff
                        break

                    all_messages.append({
                        'id': msg.get('id'),
                        'timestamp': timestamp,
                        'content': msg.get('content', ''),
                        'author': msg.get('author', {}).get('username', 'unknown'),
                        'author_id': msg.get('author', {}).get('id')
                    })
                except Exception as e:
                    continue

            # Check if we've gone past cutoff
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

            # Rate limiting
            time.sleep(0.5)

        if all_messages:
            df = pd.DataFrame(all_messages)
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"  Fetched {len(df)} messages")
            return df
        else:
            return pd.DataFrame(columns=['id', 'timestamp', 'content', 'author', 'author_id'])


# =============================================================================
# SIGNAL PARSER
# =============================================================================

class SignalParser:
    """
    Parses Discord messages into trading signals.

    Extracts: ticker, strike, option_type, expiration, cost
    Uses Signal.BuildOrder for parsing logic.
    """

    def __init__(self):
        """Initialize signal parser."""
        self.alert_marker = Config.DISCORD_CONFIG.get('alert_marker')

    def parse_message(self, content):
        """
        Parse a single message into a signal.

        Args:
            content: Raw message content string

        Returns:
            Dictionary with signal data or None if not a valid signal
        """
        if not content or self.alert_marker not in content:
            return None

        # Use Signal.BuildOrder for parsing
        order = Signal.BuildOrder(content)

        if order is None:
            return None

        # Convert to signal format
        signal = {
            'ticker': order.get('Ticker'),
            'strike': float(order.get('Strike')) if order.get('Strike') else None,
            'option_type': order.get('Option'),  # CALL or PUT
            'expiration': self._parse_expiration(order.get('Expiration')),
            'cost': float(order.get('Cost')) if order.get('Cost') else None,
            'raw_message': content
        }

        # Validate required fields
        if not all([signal['ticker'], signal['strike'], signal['option_type']]):
            return None

        return signal

    def _parse_expiration(self, exp_str):
        """
        Convert expiration string to date object.

        Args:
            exp_str: Expiration string in "MM DD YY" format

        Returns:
            date object or None
        """
        if not exp_str:
            return None

        try:
            parts = exp_str.split()
            if len(parts) >= 3:
                month = int(parts[0])
                day = int(parts[1])
                year = int(parts[2])
                # Handle 2-digit year
                if year < 100:
                    year += 2000
                return date(year, month, day)
        except (ValueError, IndexError):
            pass

        return None

    def parse_all_messages(self, messages_df):
        """
        Parse all messages in a DataFrame.

        Args:
            messages_df: DataFrame with 'content' and 'timestamp' columns

        Returns:
            DataFrame with parsed signals
        """
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
# HISTORICAL DATA FETCHER
# =============================================================================

class HistoricalDataFetcher:
    """
    Fetches historical stock and options data.

    Uses yfinance for stock data with 1-minute candles.
    Includes data caching to avoid redundant API calls.
    """

    def __init__(self):
        """Initialize data fetcher with cache."""
        self._cache = {}
        self.pre_signal_minutes = Config.DATA_CONFIG.get('pre_signal_minutes', 60)

    def fetch_stock_data(self, ticker, start_date, end_date, interval='1m'):
        """
        Fetch historical stock data.

        Args:
            ticker: Stock symbol
            start_date: Start datetime
            end_date: End datetime
            interval: Candle interval (default '1m')

        Returns:
            DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        # Create cache key
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}_{interval}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # yfinance requires specific date format
            stock = yf.Ticker(ticker)

            # For 1m data, yfinance allows max 7 days at a time
            df = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval=interval,
                prepost=True  # Include pre/post market
            )

            if df.empty:
                return None

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Ensure timezone-aware index
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC').tz_convert(EASTERN)
            else:
                df.index = df.index.tz_convert(EASTERN)

            # Filter to regular market hours (9:00 AM - 4:00 PM ET)
            df = df[(df.index.time >= dt.time(9, 0)) & (df.index.time <= dt.time(16, 0))]

            # Cache the result
            self._cache[cache_key] = df

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def fetch_data_around_signal(self, ticker, signal_time, hours_after=8):
        """
        Fetch data around a signal time.

        Args:
            ticker: Stock symbol
            signal_time: Signal timestamp
            hours_after: Hours of data to fetch after signal

        Returns:
            DataFrame with OHLCV data
        """
        # Start from pre-signal time for indicator warmup
        start_time = signal_time - timedelta(minutes=self.pre_signal_minutes)
        end_time = signal_time + timedelta(hours=hours_after)

        # Extend to cover market close if needed
        market_close = signal_time.replace(hour=16, minute=0, second=0, microsecond=0)
        if end_time < market_close:
            end_time = market_close

        return self.fetch_stock_data(
            ticker,
            start_time,
            end_time,
            interval='1m'
        )

    def estimate_option_price(self, stock_price, strike, option_type,
                               days_to_expiry, entry_price=None, volatility=0.3):
        """
        Estimate option price based on stock price movement.

        Uses simplified model: intrinsic value + time value estimation.
        If entry_price provided, scales proportionally to stock movement.

        Args:
            stock_price: Current underlying stock price
            strike: Option strike price
            option_type: 'CALL' or 'PUT'
            days_to_expiry: Days until expiration
            entry_price: Original option entry price (for scaling)
            volatility: Implied volatility estimate

        Returns:
            Estimated option price
        """
        # Calculate intrinsic value
        if option_type == 'CALL':
            intrinsic = max(0, stock_price - strike)
        else:  # PUT
            intrinsic = max(0, strike - stock_price)

        # Time value estimation (simplified)
        time_factor = max(0, days_to_expiry) / 365
        time_value = stock_price * volatility * np.sqrt(time_factor) * 0.4

        theoretical_price = intrinsic + time_value

        # If we have an entry price, use delta approximation
        if entry_price and entry_price > 0:
            # Assume delta of ~0.5 for ATM options
            delta = 0.5 if abs(stock_price - strike) / strike < 0.05 else (
                0.7 if (option_type == 'CALL' and stock_price > strike) or
                       (option_type == 'PUT' and stock_price < strike) else 0.3
            )
            return max(0.01, theoretical_price)

        return max(0.01, theoretical_price)

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


# =============================================================================
# POSITION CLASS
# =============================================================================

class Position:
    """
    Represents a trading position with full state tracking.

    Tracks:
    - Entry details (price, time, signal data)
    - Current state (price, high, low)
    - All exit signals that triggered during the trade
    - Final exit details
    """

    def __init__(self, signal, entry_price, entry_time, contracts=1):
        """
        Initialize a new position.

        Args:
            signal: Signal dictionary from SignalParser
            entry_price: Entry option price
            entry_time: Entry timestamp
            contracts: Number of contracts
        """
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
        self.stop_loss = None
        self.profit_target = None

        # Dynamic stop loss tracking (values managed by Strategy.DynamicStopLoss)
        self.stop_loss_mode = 'initial'  # 'initial', 'breakeven', 'trailing'

        # Exit data
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.is_closed = False

        # All exit signals recorded during the trade
        self.exit_signals = []

    def update(self, timestamp, price, stock_price=None):
        """
        Update position with new price data.

        Args:
            timestamp: Current timestamp
            price: Current option price
            stock_price: Current underlying stock price
        """
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.last_update = timestamp
        self.last_stock_price = stock_price

    def record_exit_signal(self, timestamp, signal_type, price, reason,
                           stock_price=None, pnl_pct=None):
        """
        Record an exit signal that triggered.

        Args:
            timestamp: When the signal triggered
            signal_type: Type of exit (stop_loss, profit_target, etc.)
            price: Price at trigger
            reason: Reason string
            stock_price: Underlying stock price
            pnl_pct: P&L percentage at trigger
        """
        self.exit_signals.append({
            'timestamp': timestamp,
            'signal_type': signal_type,
            'price': price,
            'stock_price': stock_price,
            'reason': reason,
            'pnl_pct': pnl_pct or self.get_pnl_pct(price),
            'minutes_held': self.get_minutes_held(timestamp)
        })

    def close(self, exit_price, exit_time, exit_reason):
        """
        Close the position.

        Args:
            exit_price: Exit option price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
        """
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

    def get_exit_signals_by_type(self):
        """Group exit signals by type."""
        by_type = {}
        for signal in self.exit_signals:
            sig_type = signal['signal_type']
            if sig_type not in by_type:
                by_type[sig_type] = []
            by_type[sig_type].append(signal)
        return by_type

    def get_first_exit_signal(self):
        """Get the first exit signal that triggered."""
        if self.exit_signals:
            return min(self.exit_signals, key=lambda x: x['timestamp'])
        return None

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
            'stop_loss': self.stop_loss,
            'stop_loss_mode': self.stop_loss_mode,
            'pnl': self.get_pnl(self.exit_price) if self.exit_price else None,
            'pnl_pct': self.get_pnl_pct(self.exit_price) if self.exit_price else None,
            'minutes_held': self.get_minutes_held(self.exit_time) if self.exit_time else None,
            'total_exit_signals': len(self.exit_signals)
        }


# =============================================================================
# TRACKING MATRIX
# =============================================================================

class TrackingMatrix:
    """
    High-resolution tracking data for a single trade.

    Records tick-by-tick:
    - Stock and option prices
    - Volume data
    - Technical indicators
    - Exit signal triggers
    - Stop loss tracking (price, mode)
    - Conditional trailing state
    """

    def __init__(self, position, profit_target_pct=1.0):
        """
        Initialize tracking matrix for a position.

        Args:
            position: Position object
            profit_target_pct: Profit target percentage for conditional trailing detection
        """
        self.position = position
        self.trade_label = position.get_trade_label()
        self.profit_target_pct = profit_target_pct
        self.records = []

    def add_record(self, timestamp, stock_price, option_price, volume,
                   analysis_data, exit_evaluation, holding=True):
        """
        Add a tracking record.

        Args:
            timestamp: Current timestamp
            stock_price: Stock price
            option_price: Option price
            volume: Current volume
            analysis_data: Technical indicators DataFrame
            exit_evaluation: Result from ExitDecisionTree.evaluate_all()
            holding: Whether position is currently held (default True)
        """
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
            # Stop loss tracking
            'stop_loss': self.position.stop_loss if holding else np.nan,
            'stop_loss_mode': self.position.stop_loss_mode if holding else ''
        }

        # Add technical indicators
        vwap = np.nan
        ema_20 = np.nan
        ema_30 = np.nan
        if analysis_data is not None and len(analysis_data) > 0:
            try:
                latest = analysis_data.iloc[-1]
                vwap = latest.get('vwap', np.nan)
                ema_20 = latest.get('ema_20', np.nan)
                ema_30 = latest.get('ema_30', np.nan)
                record['vwap'] = vwap
                record['ema_20'] = ema_20
                record['ema_30'] = ema_30
                record['rsi'] = latest.get('rsi_14', np.nan)
                record['macd_histogram'] = latest.get('macd_histogram', np.nan)
                record['supertrend_direction'] = latest.get('supertrend_direction', np.nan)
            except (IndexError, AttributeError):
                pass

        # Add exit evaluation results
        if exit_evaluation:
            record['would_exit'] = exit_evaluation.get('would_exit', False)
            record['would_exit_on'] = exit_evaluation.get('would_exit_on')

            # Add individual strategy triggers
            for key, value in exit_evaluation.items():
                if key.endswith('_triggered'):
                    record[key] = value

        # =================================================================
        # SL_Cx: Stop Loss Condition Tracking
        # Each condition is True when that specific sell condition is met
        # Only calculated when holding a position
        # =================================================================
        if holding:
            would_exit = exit_evaluation.get('would_exit', False) if exit_evaluation else False

            # C1: Conditional trailing - at profit target but VWAP says hold
            # True when profit target reached but not exiting (letting profits run)
            # Note: pnl_pct is in whole percentage (e.g., 42.0 for 42%)
            # profit_target_pct is decimal (e.g., 1.0 for 100%), so multiply by 100
            record['SL_C1'] = (
                pnl_pct >= self.profit_target_pct * 100 and not would_exit
            )

            # C2: EMA/VWAP bearish divergence sell signal
            # True when EMA_30 > VWAP AND stock price < EMA_30 (bearish condition)
            c2_condition = False
            if not np.isnan(ema_30) and not np.isnan(vwap):
                c2_condition = (ema_30 > vwap) and (stock_price < ema_30)
            record['SL_C2'] = c2_condition
        else:
            record['SL_C1'] = False
            record['SL_C2'] = False

        self.records.append(record)

    def update_last_record_exit_price(self, exit_price):
        """
        Update the last record with the corrected exit price.
        Called when exit uses a different price than the market price (e.g., stop loss fill).
        """
        if self.records:
            self.records[-1]['option_price'] = exit_price
            self.records[-1]['pnl'] = self.position.get_pnl(exit_price)
            self.records[-1]['pnl_pct'] = self.position.get_pnl_pct(exit_price)

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
            'exit_signals_triggered': df['would_exit'].sum() if 'would_exit' in df.columns else 0
        }


# =============================================================================
# BACKTEST CLASS
# =============================================================================

class Backtest:
    """
    Main backtesting engine.

    Orchestrates:
    - Discord message fetching
    - Signal parsing
    - Historical data retrieval
    - Position simulation with exit strategy evaluation
    - Results compilation and reporting
    """

    def __init__(self, lookback_days=None, config=None):
        """
        Initialize backtest.

        Args:
            lookback_days: Days of history to test (default from Config)
            config: Optional config overrides
        """
        # Load backtest config
        self.config = Config.get_config('backtest').copy()
        if config:
            self.config.update(config)

        self.lookback_days = lookback_days or self.config.get('lookback_days', 5)
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.position_size_pct = self.config.get('position_size_pct', 0.02)
        self.default_contracts = self.config.get('default_contracts', 1)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)
        self.commission_per_contract = self.config.get('commission_per_contract', 0.65)
        self.instrument_type = self.config.get('instrument_type', 'option')

        # Initialize components
        self.discord_fetcher = DiscordFetcher()
        self.signal_parser = SignalParser()
        self.data_fetcher = HistoricalDataFetcher()

        # Build exit strategy config
        self._build_exit_config()

        # Results storage
        self.signals_df = None
        self.positions = []
        self.Data_BT = {}
        self.results = None
        self._has_run = False

    def _build_exit_config(self):
        """Build exit strategy configuration for ExitDecisionTree."""
        exit_strategies = self.config.get('exit_strategies', {})
        indicator_settings = self.config.get('indicator_settings', {})
        swing_trade = self.config.get('swing_trade', {})

        self.exit_config = {
            # Risk parameters
            'stop_loss_pct': self.config.get('stop_loss_pct', 0.35),
            'stop_loss_warning_pct': self.config.get('stop_loss_warning_pct', 0.15),
            'profit_target_pct': self.config.get('profit_target_pct', 1.00),
            'trailing_stop_pct': self.config.get('trailing_stop_pct', 0.20),
            'time_stop_minutes': self.config.get('time_stop_minutes', 60),

            # Exit strategy toggles
            'exit_strategies': exit_strategies,

            # Indicator settings
            'indicator_settings': indicator_settings,

            # Swing trade config
            'swing_trade': swing_trade
        }

        # Initialize exit decision tree
        self.exit_tree = Strategy.ExitDecisionTree(
            config=self.exit_config,
            instrument_type=self.instrument_type
        )

    def run(self):
        """
        Execute the backtest.

        Returns:
            Dictionary with results DataFrames
        """
        print(f"\n{'='*60}")
        print("BACKTEST EXECUTION")
        print(f"{'='*60}")
        print(f"Lookback: {self.lookback_days} days")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Instrument: {self.instrument_type}")
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
        self.Data_BT = {}

        for idx, signal in self.signals_df.iterrows():
            print(f"\n  [{idx+1}/{len(self.signals_df)}] {signal['ticker']} "
                  f"${signal['strike']} {signal['option_type']}")

            position, matrix = self._process_signal(signal)

            if position:
                self.positions.append(position)
                self.Data_BT[position.get_trade_label()] = matrix

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
        """
        Process a single signal through the backtest.

        Args:
            signal: Signal dictionary

        Returns:
            Tuple of (Position, TrackingMatrix)
        """
        ticker = signal['ticker']
        signal_time = signal['signal_time']

        # Ensure timezone-aware
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=EASTERN)

        # Fetch historical data
        stock_data = self.data_fetcher.fetch_data_around_signal(
            ticker, signal_time, hours_after=8
        )

        if stock_data is None or stock_data.empty:
            print(f"    No data available for {ticker}")
            return None, None

        # Find entry point (first bar after signal time)
        entry_idx = stock_data.index.searchsorted(signal_time)

        if entry_idx >= len(stock_data):
            print(f"    No data after signal time")
            return None, None

        # Get entry bar
        entry_bar = stock_data.iloc[entry_idx]
        entry_time = stock_data.index[entry_idx]
        entry_stock_price = entry_bar['close']

        # Estimate entry option price using Black-Scholes from Analysis.py
        days_to_expiry = (signal['expiration'] - entry_time.date()).days if signal['expiration'] else 30

        if signal.get('cost'):
            entry_option_price = signal.get('cost')
        else:
            entry_result = Analysis.estimate_option_price(
                stock_price=entry_stock_price,
                strike=signal['strike'],
                option_type=signal['option_type'],
                days_to_expiry=days_to_expiry
            )
            entry_option_price = entry_result['price'] if isinstance(entry_result, dict) else entry_result

        # Apply slippage
        entry_option_price *= (1 + self.slippage_pct)

        # Create position
        position = Position(
            signal=signal,
            entry_price=entry_option_price,
            entry_time=entry_time,
            contracts=self.default_contracts
        )

        # Create DynamicStopLoss manager with config settings
        dynamic_sl_config = {
            'initial_stop_loss_pct': self.exit_config.get('stop_loss_pct', 0.35),
            'breakeven_threshold': self.exit_config.get('breakeven_threshold', 0.35),
            'trailing_threshold': self.exit_config.get('trailing_threshold', 0.55),
            'trailing_stop_pct': self.exit_config.get('dynamic_trailing_pct', 0.35)
        }
        dynamic_stop_loss = Strategy.DynamicStopLoss(dynamic_sl_config)

        # Set initial stop loss and profit target prices from entry price
        profit_target_pct = self.exit_config.get('profit_target_pct', 1.00)
        position.stop_loss = dynamic_stop_loss.get_initial_stop_loss(entry_option_price)
        position.profit_target = entry_option_price * (1 + profit_target_pct)

        # Create tracking matrix with profit target for conditional trailing detection
        matrix = TrackingMatrix(position, profit_target_pct=profit_target_pct)

        # Simulate through all bars (including pre-entry and post-exit for full market hours view)
        self._simulate_position(position, matrix, stock_data, signal, dynamic_stop_loss, entry_idx)

        return position, matrix

    def _simulate_position(self, position, matrix, stock_data, signal, dynamic_stop_loss, entry_idx=0):
        """
        Simulate position through historical data.

        Iterates through ALL bars (9am-4pm) to record full market hours data.
        Pre-entry and post-exit bars are recorded with holding=False.

        Args:
            position: Position object
            matrix: TrackingMatrix for recording
            stock_data: DataFrame with stock OHLCV (full market hours)
            signal: Original signal dictionary
            dynamic_stop_loss: Strategy.DynamicStopLoss instance for stop loss management
            entry_idx: Index of entry bar in stock_data
        """
        days_to_expiry = (signal['expiration'] - position.entry_time.date()).days if signal['expiration'] else 30

        # Calculate indicators for full data
        analysis_data = Analysis.calculate_all_indicators(stock_data)
        entry_stock_price = stock_data.iloc[entry_idx]['close']

        for i, (timestamp, bar) in enumerate(stock_data.iterrows()):
            stock_price = bar['close']
            volume = bar.get('volume', 0)

            # Update days to expiry
            current_days_to_expiry = max(0, days_to_expiry - (timestamp.date() - position.entry_time.date()).days)

            # Estimate current option price using Black-Scholes from Analysis.py
            option_result = Analysis.estimate_option_price(
                stock_price=stock_price,
                strike=position.strike,
                option_type=position.option_type,
                days_to_expiry=current_days_to_expiry,
                entry_price=position.entry_price,
                entry_stock_price=entry_stock_price
            )
            option_price = option_result['price'] if isinstance(option_result, dict) else option_result

            # Get analysis slice up to current bar
            analysis_slice = analysis_data.loc[:timestamp] if analysis_data is not None else None

            # Determine if we're holding at this timestamp
            is_pre_entry = i < entry_idx
            is_post_exit = position.is_closed
            holding = not is_pre_entry and not is_post_exit

            if holding:
                # Update position state (only while holding)
                position.update(timestamp, option_price, stock_price)

                # Update dynamic stop loss using Strategy.DynamicStopLoss
                sl_result = dynamic_stop_loss.calculate(
                    entry_price=position.entry_price,
                    current_price=option_price,
                    highest_price=position.highest_price,
                    current_stop_loss=position.stop_loss,
                    current_mode=position.stop_loss_mode
                )
                position.stop_loss = sl_result['stop_loss']
                position.stop_loss_mode = sl_result['mode']

                # Build context for exit evaluation
                context = {
                    'position': position,
                    'current_price': option_price,
                    'stock_price': stock_price,
                    'current_time': timestamp,
                    'analysis_data': analysis_slice,
                    'discord_exit_signal': False
                }

                # Evaluate all exit strategies
                exit_evaluation = self.exit_tree.evaluate_all(context)

                # Record to tracking matrix (holding=True)
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    analysis_data=analysis_slice,
                    exit_evaluation=exit_evaluation,
                    holding=True
                )

                # Handle exit if triggered
                if exit_evaluation.get('would_exit', False):
                    exit_type = exit_evaluation.get('would_exit_on')

                    # Calculate exit price - fill at market price with slippage
                    exit_price = option_price * (1 - self.slippage_pct)

                    # Record exit signal with corrected price
                    position.record_exit_signal(
                        timestamp=timestamp,
                        signal_type=exit_type,
                        price=exit_price,
                        reason=exit_evaluation.get('would_exit_reason', ''),
                        stock_price=stock_price,
                        pnl_pct=position.get_pnl_pct(exit_price)
                    )

                    # Execute exit on first signal
                    if not position.is_closed:
                        position.close(exit_price, timestamp, exit_type)

                # Check for end of day exit
                if timestamp.time() >= dt.time(15, 55) and not position.is_closed:
                    exit_price = option_price * (1 - self.slippage_pct)
                    position.close(exit_price, timestamp, 'market_close')
                    position.record_exit_signal(
                        timestamp=timestamp,
                        signal_type='market_close',
                        price=exit_price,
                        reason='Market closing - forced exit',
                        stock_price=stock_price
                    )
            else:
                # Record pre-entry or post-exit bar (holding=False)
                matrix.add_record(
                    timestamp=timestamp,
                    stock_price=stock_price,
                    option_price=option_price,
                    volume=volume,
                    analysis_data=analysis_slice,
                    exit_evaluation=None,
                    holding=False
                )

        # If still open at end of data, close at market_close
        if not position.is_closed:
            exit_price = position.current_price * (1 - self.slippage_pct)
            position.close(exit_price, stock_data.index[-1], 'market_close')

    def _compile_results(self):
        """
        Compile backtest results into DataFrames.

        Returns:
            Dictionary with Signals, Positions, and Tracking_matrices
        """
        # Signals DataFrame
        signals_df = self.signals_df.copy() if self.signals_df is not None else pd.DataFrame()

        # Positions DataFrame
        if self.positions:
            positions_data = [p.to_dict() for p in self.positions]
            positions_df = pd.DataFrame(positions_data)
        else:
            positions_df = pd.DataFrame()

        # Data_BT dict of DataFrames
        Data_BT = {}
        for label, matrix in self.Data_BT.items():
            Data_BT[label] = matrix.to_dataframe()

        # Summary statistics
        summary = self._calculate_summary(positions_df)

        return {
            'Signals': signals_df,
            'Positions': positions_df,
            'Data_BT': Data_BT,
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

    def get_trade_Data_BT(self):
        """
        Get all trade Data_BT matrices.

        Returns:
            Dictionary mapping trade labels to DataFrames
        """
        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return {}

        return self.results.get('Data_BT', {})

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

    def export_Data_BT(self, output_dir='Data_BT'):
        """
        Export Data_BT matrices to CSV files.

        Args:
            output_dir: Directory to save files
        """
        import os

        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return

        matrices = self.get_trade_Data_BT()

        if not matrices:
            print("No Data_BT matrices to export")
            return

        os.makedirs(output_dir, exist_ok=True)

        for label, df in matrices.items():
            # Clean label for filename
            filename = label.replace(':', '_').replace(' ', '_')
            filepath = os.path.join(output_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
            print(f"Exported: {filepath}")

        print(f"\nExported {len(matrices)} Data_BT matrices to {output_dir}/")

    def BT_Save(self, filepath='backtest_data_matrices.pkl'):
        """
        Save backtest data to pickle file for Dashboard.py.

        Args:
            filepath: Path to save pickle file (default: backtest_data_matrices.pkl)

        The saved file contains:
        - matrices: Dict of DataFrames with tracking data for each trade
        - exit_signals: Dict of exit signal lists for each trade
        """
        import pickle
        import os

        if not self._has_run:
            print("Backtest has not been run yet. Call run() first.")
            return

        # Build matrices dict with enhanced data for Dashboard
        matrices = {}
        exit_signals = {}

        for position in self.positions:
            trade_label = position.get_trade_label()
            matrix = self.Data_BT.get(trade_label)

            if matrix is None:
                continue

            # Get base DataFrame from tracking matrix
            df = matrix.to_dataframe()

            if df.empty:
                continue

            # Add position details that Dashboard expects
            df['ticker'] = position.ticker
            df['strike'] = position.strike
            df['option_type'] = position.option_type
            df['contracts'] = position.contracts
            df['entry_time'] = position.entry_time
            df['exit_time'] = position.exit_time
            df['exit_reason'] = position.exit_reason
            df['expiration'] = position.expiration

            # Build exit_signals_at_time column from would_exit_on
            if 'would_exit_on' in df.columns:
                df['exit_signals_at_time'] = df['would_exit_on'].fillna('')

            # Store using trade_label as key
            matrices[trade_label] = df

            # Store exit signals list for this position
            exit_signals[trade_label] = position.exit_signals

        # Save to pickle
        data = {
            'matrices': matrices,
            'exit_signals': exit_signals
        }

        # Handle relative/absolute paths
        if not os.path.isabs(filepath):
            # Save in same directory as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved backtest data to {filepath}")
        print(f"  {len(matrices)} trades saved")


# =============================================================================
# QUICK TEST FUNCTIONS
# =============================================================================

def quick_test(days=1):
    """
    Run a quick backtest with minimal settings.

    Args:
        days: Number of days to test

    Returns:
        Backtest instance
    """
    bt = Backtest(lookback_days=days)
    bt.run()
    bt.summary()
    return bt


if __name__ == '__main__':
    # Run quick test when executed directly
    bt = quick_test(days=1)
