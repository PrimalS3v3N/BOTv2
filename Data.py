"""
Data.py - Market Data Sourcing

Module Goal: Source data, return DataFrames.

Functionality:
1. RobinhoodScraper: Internal webscraping library for live stock/option data
   - Authenticates via Robinhood's OAuth2 API (credentials from settings.csv)
   - Fetches real-time stock quotes and option chain pricing
   - Uses browser spoofing to mimic regular web traffic
   - Thread-safe for parallel data collection
2. LiveDataFetcher: Coordinates multi-threaded data collection across tickers
   - Deduplicates stock data when multiple signals share the same underlying
   - Collects both stock OHLCV and option pricing per cycle

Security: No third-party Robinhood packages. All API interaction is done
via standard requests library with browser-like headers.

Usage:
    from Data import RobinhoodScraper, LiveDataFetcher
    scraper = RobinhoodScraper()
    scraper.authenticate()
    quote = scraper.get_stock_quote('AAPL')
    option = scraper.get_option_quote('AAPL', 150.0, 'CALL', '2025-01-17')

================================================================================
INTERNAL - Robinhood Webscraping & Live Data Collection
================================================================================
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
from zoneinfo import ZoneInfo
import time
import uuid
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import Config

EASTERN = ZoneInfo('America/New_York')


# =============================================================================
# INTERNAL - Robinhood Scraper (Internal Webscraping Library)
# =============================================================================

class RobinhoodScraper:
    """
    Internal webscraping library for Robinhood data.

    Uses Robinhood's REST API directly with browser-like headers.
    No third-party Robinhood packages - all interaction is via
    standard requests with proper authentication.

    Thread-safe: Uses a lock around session state modifications.
    The session itself (requests.Session) handles connection pooling.

    Authentication flow:
    1. POST to /oauth2/token/ with username/password/device_token
    2. Handle MFA if required (TOTP from settings.csv)
    3. Store access_token for subsequent requests
    4. Auto-refresh when token expires
    """

    def __init__(self, config=None):
        rh_config = config or Config.get_config('robinhood')
        self.username = rh_config.get('username', '')
        self.password = rh_config.get('password', '')
        self.mfa_secret = rh_config.get('mfa_secret', '')
        self.client_id = rh_config.get('client_id', '')
        self.api_base = rh_config.get('api_base', 'https://api.robinhood.com')
        self.login_url = rh_config.get('login_url', 'https://api.robinhood.com/oauth2/token/')
        self.quotes_url = rh_config.get('quotes_url', 'https://api.robinhood.com/quotes/')
        self.options_url = rh_config.get('options_url', 'https://api.robinhood.com/options/marketdata/')
        self.instruments_url = rh_config.get('instruments_url', 'https://api.robinhood.com/instruments/')
        self.options_instruments_url = rh_config.get('options_instruments_url', 'https://api.robinhood.com/options/instruments/')
        self.options_chains_url = rh_config.get('options_chains_url', 'https://api.robinhood.com/options/chains/')
        self.request_timeout = rh_config.get('request_timeout', 10)
        self.rate_limit_delay = rh_config.get('rate_limit_delay', 0.5)
        self.max_retries = rh_config.get('max_retries', 3)
        self.browser_config = rh_config.get('browser_config', {})

        # Generate device token (persistent per session)
        self.device_token = rh_config.get('device_token') or str(uuid.uuid4())

        # Auth state
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None
        self._account_url = None

        # Thread safety
        self._lock = threading.Lock()

        # Instrument cache: ticker -> instrument_url
        self._instrument_cache = {}
        # Option instrument cache: (ticker, strike, type, expiry) -> instrument_id
        self._option_instrument_cache = {}

        # Session setup
        self.session = requests.Session()
        self._configure_session()

    def _configure_session(self):
        """Configure requests session with browser-like headers and retry logic."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Set browser headers
        bc = self.browser_config
        self.session.headers.update({
            'User-Agent': bc.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
            'Accept': bc.get('accept', 'application/json'),
            'Accept-Language': bc.get('accept_language', 'en-US,en;q=0.9'),
            'Accept-Encoding': bc.get('accept_encoding', 'gzip, deflate, br'),
            'Origin': bc.get('origin', 'https://robinhood.com'),
            'Referer': bc.get('referer', 'https://robinhood.com/'),
            'Sec-CH-UA': bc.get('sec_ch_ua', ''),
            'Sec-CH-UA-Mobile': bc.get('sec_ch_ua_mobile', '?0'),
            'Sec-CH-UA-Platform': bc.get('sec_ch_ua_platform', '"Windows"'),
            'Sec-Fetch-Dest': bc.get('sec_fetch_dest', 'empty'),
            'Sec-Fetch-Mode': bc.get('sec_fetch_mode', 'cors'),
            'Sec-Fetch-Site': bc.get('sec_fetch_site', 'same-site'),
        })

    def _generate_mfa_code(self):
        """Generate TOTP MFA code from the stored secret."""
        if not self.mfa_secret:
            return None
        try:
            import hmac
            import hashlib
            import struct
            import base64

            # TOTP: RFC 6238
            secret_bytes = base64.b32decode(self.mfa_secret.upper().replace(' ', ''), casefold=True)
            counter = int(time.time()) // 30
            counter_bytes = struct.pack('>Q', counter)
            hmac_hash = hmac.new(secret_bytes, counter_bytes, hashlib.sha1).digest()
            offset = hmac_hash[-1] & 0x0F
            code = struct.unpack('>I', hmac_hash[offset:offset + 4])[0] & 0x7FFFFFFF
            return str(code % 1000000).zfill(6)
        except Exception as e:
            print(f"  MFA code generation failed: {e}")
            return None

    def authenticate(self):
        """
        Authenticate with Robinhood via OAuth2.

        Returns:
            bool: True if authentication succeeded.
        """
        if not self.username or not self.password:
            print("RobinhoodScraper: No credentials configured (check settings.csv rows 2-4, col B)")
            return False

        payload = {
            'grant_type': 'password',
            'scope': 'internal',
            'client_id': self.client_id,
            'expires_in': 86400,
            'username': self.username,
            'password': self.password,
            'device_token': self.device_token,
        }

        # Add MFA if available
        mfa_code = self._generate_mfa_code()
        if mfa_code:
            payload['mfa_code'] = mfa_code

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.login_url,
                    data=payload,
                    timeout=self.request_timeout,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )

                if response.status_code == 200:
                    data = response.json()
                    with self._lock:
                        self._access_token = data.get('access_token')
                        self._refresh_token = data.get('refresh_token')
                        self._token_expiry = time.time() + data.get('expires_in', 86400)
                        self.session.headers['Authorization'] = f'Bearer {self._access_token}'
                    print("RobinhoodScraper: Authenticated successfully")
                    return True

                elif response.status_code == 400:
                    data = response.json()
                    # MFA challenge
                    if 'mfa_required' in str(data) or data.get('mfa_type'):
                        if mfa_code:
                            print(f"  MFA rejected (attempt {attempt + 1})")
                        else:
                            print("  MFA required but no mfa_secret in settings.csv row 4")
                            return False
                    else:
                        print(f"  Auth error: {data.get('detail', data)}")

                elif response.status_code == 429:
                    wait = float(response.headers.get('Retry-After', 5))
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)

                else:
                    print(f"  Auth failed: HTTP {response.status_code}")

            except requests.RequestException as e:
                print(f"  Auth request failed: {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        print("RobinhoodScraper: Authentication failed after retries")
        return False

    def _refresh_auth(self):
        """Refresh authentication token if expired."""
        with self._lock:
            if self._token_expiry and time.time() < self._token_expiry - 300:
                return True  # Token still valid (with 5min buffer)
            if not self._refresh_token:
                return self.authenticate()

        try:
            payload = {
                'grant_type': 'refresh_token',
                'scope': 'internal',
                'client_id': self.client_id,
                'refresh_token': self._refresh_token,
                'device_token': self.device_token,
            }
            response = self.session.post(
                self.login_url,
                data=payload,
                timeout=self.request_timeout,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            if response.status_code == 200:
                data = response.json()
                with self._lock:
                    self._access_token = data.get('access_token')
                    self._refresh_token = data.get('refresh_token')
                    self._token_expiry = time.time() + data.get('expires_in', 86400)
                    self.session.headers['Authorization'] = f'Bearer {self._access_token}'
                return True
        except Exception:
            pass

        return self.authenticate()

    @property
    def is_authenticated(self):
        """Check if we have a valid access token."""
        with self._lock:
            return self._access_token is not None

    def _api_get(self, url, params=None, skip_rate_limit=False):
        """
        Make authenticated GET request.

        Args:
            url: API endpoint URL
            params: Query parameters
            skip_rate_limit: If True, skip the rate limit delay (used for parallel requests)
        """
        if not self.is_authenticated:
            self._refresh_auth()

        if not skip_rate_limit:
            time.sleep(self.rate_limit_delay)

        try:
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            if response.status_code == 401:
                # Token expired, try refresh
                if self._refresh_auth():
                    response = self.session.get(url, params=params, timeout=self.request_timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - back off and retry once
                wait = float(response.headers.get('Retry-After', 2))
                time.sleep(wait)
                response = self.session.get(url, params=params, timeout=self.request_timeout)
                if response.status_code == 200:
                    return response.json()
            return None
        except requests.RequestException:
            return None

    # -----------------------------------------------------------------
    # Stock Quotes
    # -----------------------------------------------------------------

    def get_stock_quote(self, ticker):
        """
        Get real-time stock quote for a ticker.

        Returns:
            dict with keys: ticker, price, high, low, open, close, volume,
                           bid, ask, bid_size, ask_size, timestamp
            None if request fails.
        """
        url = f"{self.quotes_url}{ticker.upper()}/"
        data = self._api_get(url)
        if not data:
            return None

        try:
            return {
                'ticker': ticker.upper(),
                'price': float(data.get('last_trade_price', 0)),
                'high': float(data.get('high_today') or data.get('last_trade_price', 0)),
                'low': float(data.get('low_today') or data.get('last_trade_price', 0)),
                'open': float(data.get('open_today') or data.get('last_trade_price', 0)),
                'close': float(data.get('previous_close', 0)),
                'volume': int(float(data.get('volume') or 0)),
                'bid': float(data.get('bid_price', 0)),
                'ask': float(data.get('ask_price', 0)),
                'bid_size': int(float(data.get('bid_size', 0))),
                'ask_size': int(float(data.get('ask_size', 0))),
                'timestamp': dt.datetime.now(EASTERN),
            }
        except (TypeError, ValueError):
            return None

    def get_stock_quotes_batch(self, tickers):
        """
        Get real-time stock quotes for multiple tickers in one request.

        Args:
            tickers: list of ticker symbols

        Returns:
            dict mapping ticker -> quote_dict
        """
        if not tickers:
            return {}

        symbols = ','.join(t.upper() for t in tickers)
        url = f"{self.quotes_url}?symbols={symbols}"
        data = self._api_get(url, skip_rate_limit=True)

        if not data or 'results' not in data:
            return {}

        quotes = {}
        now = dt.datetime.now(EASTERN)
        for item in data['results']:
            if not item:
                continue
            try:
                symbol = item.get('symbol', '').upper()
                quotes[symbol] = {
                    'ticker': symbol,
                    'price': float(item.get('last_trade_price', 0)),
                    'high': float(item.get('high_today') or item.get('last_trade_price', 0)),
                    'low': float(item.get('low_today') or item.get('last_trade_price', 0)),
                    'open': float(item.get('open_today') or item.get('last_trade_price', 0)),
                    'close': float(item.get('previous_close', 0)),
                    'volume': int(float(item.get('volume') or 0)),
                    'bid': float(item.get('bid_price', 0)),
                    'ask': float(item.get('ask_price', 0)),
                    'bid_size': int(float(item.get('bid_size', 0))),
                    'ask_size': int(float(item.get('ask_size', 0))),
                    'timestamp': now,
                }
            except (TypeError, ValueError):
                continue

        return quotes

    # -----------------------------------------------------------------
    # Option Quotes
    # -----------------------------------------------------------------

    def _get_instrument_url(self, ticker):
        """Get Robinhood instrument URL for a ticker (cached)."""
        if ticker in self._instrument_cache:
            return self._instrument_cache[ticker]

        url = f"{self.instruments_url}?symbol={ticker.upper()}"
        data = self._api_get(url)
        if data and 'results' in data and data['results']:
            instrument_url = data['results'][0].get('url', '')
            instrument_id = data['results'][0].get('id', '')
            self._instrument_cache[ticker] = (instrument_url, instrument_id)
            return instrument_url, instrument_id
        return None, None

    def _find_option_instrument(self, ticker, strike, option_type, expiration):
        """
        Find the specific option instrument ID on Robinhood.

        Args:
            ticker: Stock ticker
            strike: Strike price (float)
            option_type: 'CALL' or 'PUT'
            expiration: date object or 'YYYY-MM-DD' string

        Returns:
            option instrument URL or None
        """
        cache_key = (ticker, strike, option_type, str(expiration))
        if cache_key in self._option_instrument_cache:
            return self._option_instrument_cache[cache_key]

        # Get chain ID for this ticker
        instrument_url, instrument_id = self._get_instrument_url(ticker)
        if not instrument_id:
            return None

        # Find chain
        chain_url = f"{self.options_chains_url}?equity_instrument_ids={instrument_id}"
        chain_data = self._api_get(chain_url)
        if not chain_data or 'results' not in chain_data or not chain_data['results']:
            return None

        chain_id = chain_data['results'][0].get('id')
        if not chain_id:
            return None

        # Format expiration
        if isinstance(expiration, dt.date):
            exp_str = expiration.strftime('%Y-%m-%d')
        else:
            exp_str = str(expiration)

        # Find specific option instrument
        opt_type = 'call' if option_type.upper() in ['CALL', 'CALLS', 'C'] else 'put'
        params = {
            'chain_id': chain_id,
            'expiration_dates': exp_str,
            'strike_price': f"{float(strike):.4f}",
            'type': opt_type,
            'state': 'active',
        }

        opt_data = self._api_get(self.options_instruments_url, params=params)
        if opt_data and 'results' in opt_data and opt_data['results']:
            opt_url = opt_data['results'][0].get('url', '')
            opt_id = opt_data['results'][0].get('id', '')
            self._option_instrument_cache[cache_key] = opt_id
            return opt_id

        return None

    def get_option_quote(self, ticker, strike, option_type, expiration):
        """
        Get real-time option quote.

        Args:
            ticker: Stock ticker
            strike: Strike price (float)
            option_type: 'CALL' or 'PUT'
            expiration: date object or 'YYYY-MM-DD' string

        Returns:
            dict with keys: ticker, strike, option_type, expiration,
                           mark_price, bid, ask, volume, open_interest,
                           implied_volatility, delta, gamma, theta, vega,
                           timestamp
            None if not found.
        """
        opt_id = self._find_option_instrument(ticker, strike, option_type, expiration)
        if not opt_id:
            return None

        url = f"{self.options_url}{opt_id}/"
        data = self._api_get(url)
        if not data:
            return None

        try:
            bid = float(data.get('bid_price', 0) or 0)
            ask = float(data.get('ask_price', 0) or 0)
            mark = float(data.get('mark_price', 0) or 0)
            if mark == 0 and bid > 0 and ask > 0:
                mark = (bid + ask) / 2

            return {
                'ticker': ticker.upper(),
                'strike': float(strike),
                'option_type': option_type.upper(),
                'expiration': str(expiration),
                'mark_price': mark,
                'bid': bid,
                'ask': ask,
                'volume': int(float(data.get('volume', 0) or 0)),
                'open_interest': int(float(data.get('open_interest', 0) or 0)),
                'implied_volatility': float(data.get('implied_volatility', 0) or 0),
                'delta': float(data.get('delta', 0) or 0),
                'gamma': float(data.get('gamma', 0) or 0),
                'theta': float(data.get('theta', 0) or 0),
                'vega': float(data.get('vega', 0) or 0),
                'high_price': float(data.get('high_price', 0) or 0),
                'low_price': float(data.get('low_price', 0) or 0),
                'timestamp': dt.datetime.now(EASTERN),
            }
        except (TypeError, ValueError):
            return None

    def get_option_quotes_batch(self, option_specs):
        """
        Get option quotes for multiple contracts in parallel.

        Args:
            option_specs: list of dicts with keys: ticker, strike, option_type, expiration

        Returns:
            list of option quote dicts (None entries for failed lookups)
        """
        if not option_specs:
            return []

        # For single spec, avoid thread overhead
        if len(option_specs) == 1:
            return [self.get_option_quote(
                option_specs[0]['ticker'], option_specs[0]['strike'],
                option_specs[0]['option_type'], option_specs[0]['expiration']
            )]

        # Parallel fetch for multiple specs
        results = [None] * len(option_specs)
        with ThreadPoolExecutor(max_workers=min(len(option_specs), 5)) as executor:
            future_to_idx = {}
            for idx, spec in enumerate(option_specs):
                future = executor.submit(
                    self.get_option_quote,
                    spec['ticker'], spec['strike'],
                    spec['option_type'], spec['expiration']
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None

        return results

    def close(self):
        """Close the session and release resources."""
        if self.session:
            self.session.close()


# =============================================================================
# INTERNAL - Live Data Fetcher (Multi-threaded Collection)
# =============================================================================

class LiveDataFetcher:
    """
    Coordinates live data collection across multiple tickers/options.

    Deduplicates stock data when multiple signals share the same underlying
    ticker. Uses ThreadPoolExecutor for parallel webscraping to minimize
    cycle time.

    Usage:
        fetcher = LiveDataFetcher()
        fetcher.start()
        # Add signals to monitor
        fetcher.add_signal(signal_dict)
        # Each cycle collects stock + option data
        data = fetcher.collect_cycle()
        fetcher.stop()
    """

    def __init__(self, scraper=None, config=None):
        live_config = config or Config.get_config('live')
        self.thread_pool_size = live_config.get('thread_pool_size', 5)
        self.deduplicate_stock = live_config.get('deduplicate_stock_data', True)
        self.rate_limit_delay = Config.ROBINHOOD_CONFIG.get('rate_limit_delay', 0.5)

        self.scraper = scraper or RobinhoodScraper()
        self._executor = None
        self._active_signals = []  # List of signal dicts being monitored
        self._lock = threading.Lock()

    def start(self):
        """Initialize the thread pool and authenticate."""
        self._executor = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        if not self.scraper.is_authenticated:
            if not self.scraper.authenticate():
                print("LiveDataFetcher: Authentication failed, falling back to BS pricing")

    def stop(self):
        """Shutdown the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def add_signal(self, signal):
        """
        Add a signal to monitor.

        Args:
            signal: dict with keys: ticker, strike, option_type, expiration, signal_id
        """
        with self._lock:
            self._active_signals.append(signal)

    def remove_signal(self, signal_id):
        """Remove a signal from monitoring."""
        with self._lock:
            self._active_signals = [s for s in self._active_signals if s.get('signal_id') != signal_id]

    def get_unique_tickers(self):
        """Get deduplicated list of tickers being monitored."""
        with self._lock:
            return list(set(s['ticker'] for s in self._active_signals))

    def collect_cycle(self):
        """
        Execute one data collection cycle.

        Stock quotes use batch API (single request for all tickers).
        Option quotes are fetched in parallel via thread pool.
        No rate limit delays on parallel requests - handled by 429 retry.

        Returns:
            dict: {
                'stock_quotes': {ticker: quote_dict, ...},
                'option_quotes': {signal_id: quote_dict, ...},
                'timestamp': datetime,
                'cycle_time_ms': float,
            }
        """
        start_time = time.time()
        result = {
            'stock_quotes': {},
            'option_quotes': {},
            'timestamp': dt.datetime.now(EASTERN),
            'cycle_time_ms': 0,
        }

        with self._lock:
            signals = list(self._active_signals)

        if not signals:
            return result

        if not self._executor:
            self.start()

        if not self.scraper.is_authenticated:
            result['cycle_time_ms'] = (time.time() - start_time) * 1000
            return result

        # Deduplicate tickers for stock data
        unique_tickers = list(set(s['ticker'] for s in signals))

        futures = {}

        # Stock quotes: always batch (single API call, no per-ticker overhead)
        future = self._executor.submit(self.scraper.get_stock_quotes_batch, unique_tickers)
        futures[future] = ('stock_batch', None)

        # Option quotes: parallel fetch via thread pool
        for signal in signals:
            future = self._executor.submit(
                self.scraper.get_option_quote,
                signal['ticker'], signal['strike'],
                signal['option_type'], signal['expiration']
            )
            futures[future] = ('option', signal.get('signal_id'))

        # Collect results as they complete
        for future in as_completed(futures):
            data_type, key = futures[future]
            try:
                data = future.result()
                if data_type == 'stock_batch' and isinstance(data, dict):
                    result['stock_quotes'].update(data)
                elif data_type == 'option' and data:
                    result['option_quotes'][key] = data
            except Exception:
                continue

        result['cycle_time_ms'] = (time.time() - start_time) * 1000
        return result

    def close(self):
        """Cleanup resources."""
        self.stop()
        if self.scraper:
            self.scraper.close()
