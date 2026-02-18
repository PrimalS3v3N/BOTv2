"""
Config.py - Centralized Configuration

Module Goal: Store all configurable variables, organized by module.

Usage:
    import Config
    discord_settings = Config.get_config('discord')
    token = Config.get_setting('discord', 'token')
"""

import json
import os
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --- Initialization helper (must run before DISCORD_CONFIG) ---
def _load_discord_token_from_excel():
    """Load Discord token from Setting.xlsx (Row 1, Column B)."""
    try:
        import pandas as pd
        config_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(config_dir, 'Setting.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, header=None)
            token = df.iloc[0, 1]
            if pd.notna(token):
                return str(token).strip()
    except Exception as e:
        print(f"Warning: Could not load Discord token from Setting.xlsx: {e}")
    return ''

_DISCORD_TOKEN_FROM_EXCEL = _load_discord_token_from_excel()


# =============================================================================
# SIGNAL MODULE (Signal.py, Test.py)
# =============================================================================

DISCORD_CONFIG = {
    'token': _DISCORD_TOKEN_FROM_EXCEL or os.getenv('DISCORD_TOKEN', ''),  # Auth token from Setting.xlsx or env
    'channel_id': os.getenv('DISCORD_CHANNEL_ID', '748401380288364575'),   # Signal channel ID
    'api_version': 'v10',                          # Discord API version
    'base_url': 'https://discord.com/api',         # Discord API base URL
    'send_timeout': 0.35,                          # Send timeout (seconds)
    'receive_timeout': 0.35,                       # Receive timeout (seconds)
    'message_limit': 20,                           # Messages per request (normal mode)
    'test_message_limit': 100,                     # Messages per request (test mode)
    'alert_marker': '<a:RedAlert:759583962237763595>',  # Discord alert emoji ID
    'rate_limit': {
        'min_delay': .5,                          # Min seconds between requests
        'max_delay': 1.5,                          # Max seconds between requests
        'batch_size': 75,                          # Messages per request
        'long_pause_chance': 0.15,                 # Chance of longer pause (15%)
        'long_pause_min': 1.0,                     # Long pause min (seconds)
        'long_pause_max': 2.0,                    # Long pause max (seconds)
    },
    'spoof_browser': True,                         # Enable browser spoofing
    'browser_config': {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0',
        'accept': '*/*',
        'accept_language': 'en-US,en;q=0.9',
        'accept_encoding': 'gzip, deflate, br',
        'connection': 'keep-alive',
        'cache_control': 'no-cache',
        'pragma': 'no-cache',
        'sec_ch_ua': '"Not_A Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"',
        'sec_ch_ua_mobile': '?0',
        'sec_ch_ua_platform': '"Windows"',
        'sec_fetch_dest': 'empty',
        'sec_fetch_mode': 'cors',
        'sec_fetch_site': 'same-origin',
        'x_discord_locale': 'en-US',
        'x_discord_timezone': 'America/New_York',
        'x_debug_options': 'bugReporterEnabled',
        'origin': 'https://discord.com',
        'referer': 'https://discord.com/channels/@me',
    },
    'x_super_properties': {
        'os': 'Windows',
        'browser': 'Edge',
        'device': '',
        'system_locale': 'en-US',
        'browser_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0',
        'browser_version': '144.0.0.0',
        'os_version': '10',
        'referrer': '',
        'referring_domain': '',
        'referrer_current': '',
        'referring_domain_current': '',
        'release_channel': 'stable',
        'client_build_number': 254573,
        'client_event_source': None,
        'design_id': 0,
    },
}


# =============================================================================
# DATA MODULE (Data.py, Test.py)
# =============================================================================

# Index symbols that are valid option underlyings but not tradeable stocks on Robinhood.
# Robinhood's /quotes/ endpoint returns 404 for these, so we skip API validation.
INDEX_SYMBOLS = {'SPX', 'NDX', 'RUT', 'DJX', 'VIX', 'XSP', 'OEX'}

DATA_CONFIG = {
    'cache_ttl': 60,                               # Cache TTL (seconds)
    'max_quote_history': 1000,                     # Max quotes in memory
    'purge_old_quotes_minutes': 60,                # Purge quotes older than (minutes)
    'timeframes': ['1min', '5min', '15min'],       # Available timeframes
    'bars_to_fetch': 100,                          # Initial bars to fetch
    'pre_signal_minutes': 60,                      # Pre-signal data window (minutes)
}


# =============================================================================
# ROBINHOOD SCRAPER (Data.py - Internal Webscraping)
# =============================================================================

def _load_robinhood_credentials():
    """Load Robinhood credentials from settings.csv (rows 2,3,4+ col B)."""
    try:
        import pandas as pd
        config_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(config_dir, 'settings.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None)
            creds = {}
            if len(df) > 1 and pd.notna(df.iloc[1, 1]):
                creds['username'] = str(df.iloc[1, 1]).strip()
            if len(df) > 2 and pd.notna(df.iloc[2, 1]):
                creds['password'] = str(df.iloc[2, 1]).strip()
            if len(df) > 3 and pd.notna(df.iloc[3, 1]):
                creds['mfa_secret'] = str(df.iloc[3, 1]).strip()
            return creds
    except Exception as e:
        print(f"Warning: Could not load Robinhood credentials from settings.csv: {e}")
    return {}

_RH_CREDENTIALS = _load_robinhood_credentials()

ROBINHOOD_CONFIG = {
    'username': _RH_CREDENTIALS.get('username', ''),
    'password': _RH_CREDENTIALS.get('password', ''),
    'mfa_secret': _RH_CREDENTIALS.get('mfa_secret', ''),
    'base_url': 'https://robinhood.com',
    'api_base': 'https://api.robinhood.com',
    'login_url': 'https://api.robinhood.com/oauth2/token/',
    'quotes_url': 'https://api.robinhood.com/quotes/',
    'options_url': 'https://api.robinhood.com/options/marketdata/',
    'instruments_url': 'https://api.robinhood.com/instruments/',
    'options_instruments_url': 'https://api.robinhood.com/options/instruments/',
    'options_chains_url': 'https://api.robinhood.com/options/chains/',
    'client_id': 'c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS',  # Robinhood public client ID
    'device_token': None,              # Generated at runtime (UUID4)
    'session_timeout': 86400,          # Session valid for 24 hours
    'max_retries': 3,                  # Max login retries
    'request_timeout': 10,            # HTTP request timeout (seconds)
    'rate_limit_delay': 0.5,          # Delay between requests (seconds)
    'browser_config': {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'accept': 'application/json',
        'accept_language': 'en-US,en;q=0.9',
        'accept_encoding': 'gzip, deflate, br',
        'origin': 'https://robinhood.com',
        'referer': 'https://robinhood.com/',
        'sec_ch_ua': '"Chromium";v="144", "Google Chrome";v="144"',
        'sec_ch_ua_mobile': '?0',
        'sec_ch_ua_platform': '"Windows"',
        'sec_fetch_dest': 'empty',
        'sec_fetch_mode': 'cors',
        'sec_fetch_site': 'same-site',
    },
}


# =============================================================================
# LIVE TESTING MODULE (Test.py)
# =============================================================================

LIVE_CONFIG = {
    'cycle_interval_seconds': 60,      # Data collection cycle interval (1 minute)
    'max_concurrent_tickers': 10,      # Max tickers to scrape in parallel
    'thread_pool_size': 5,             # Thread pool for parallel webscraping
    'data_dir': 'live_data',           # Directory for live data storage
    'summary_interval_minutes': 2,     # DataSummary aggregation interval
    'auto_save_interval': 5,           # Auto-save data every N minutes

    # Stock data deduplication: track unique tickers across signals
    # to avoid pulling stock data twice for same underlying
    'deduplicate_stock_data': True,

    # Data retention
    'max_databook_rows': 50000,        # Max rows per signal databook
    'compress_on_save': True,          # Compress pickle files on save
}


# =============================================================================
# ANALYSIS MODULE (Analysis.py)
# =============================================================================

ANALYSIS_CONFIG = {
    # Only settings actively consumed by Analysis.py live here.
    # Indicator periods (RSI, EMA, EWO, etc.) are sourced from
    # BACKTEST_CONFIG['indicators'] — see below.
    'options': {
        'risk_free_rate': 0.05,                    # Annual risk-free rate (5%)
        'default_volatility': 0.30,                # Default implied volatility (30%)
        'min_option_price': 0.10,                  # Min option price floor
    },
}


# =============================================================================
# BACKTEST MODULE (Test.py)
# =============================================================================

BACKTEST_CONFIG = {
    'lookback_days': 5,                            # Days of history to backtest
    'initial_capital': 10000.0,                    # Starting capital
    'default_contracts': 1,                        # Default option contracts
    'slippage_pct': 0.001,                         # Slippage per fill (0.1%)
    'commission_per_contract': 0.65,               # Per-contract commission

    # Technical indicators for backtest
    # NOTE — RSI thresholds used across the system:
    #   indicators.rsi_overbought/oversold  = 70/30  (stock-level, shared by options_exit)
    #   momentum_peak.rsi_overbought        = 80     (options peak detection)
    #   risk_assessment.rsi_overbought_threshold = 80 (options entry risk flag)
    #   closure_peak.rsi_call/put_threshold = 85/15  (end-of-day exit)
    'indicators': {
        'ema_periods': [10, 21, 50, 100, 200],     # EMA periods (bars)
        'vwap_enabled': True,                      # Calculate VWAP
        'rsi_period': 14,                          # RSI calculation period
        'rsi_overbought': 70,                      # RSI overbought threshold
        'rsi_oversold': 30,                        # RSI oversold threshold
        'supertrend_period': 10,                   # Supertrend ATR period (bars)
        'supertrend_multiplier': 3.0,              # Supertrend ATR multiplier
        'ichimoku_tenkan': 9,                      # Ichimoku Tenkan-sen period (Conversion Line)
        'ichimoku_kijun': 26,                      # Ichimoku Kijun-sen period (Base Line)
        'ichimoku_senkou_b': 52,                   # Ichimoku Senkou Span B period
        'ichimoku_displacement': 26,               # Ichimoku cloud forward displacement
        'atr_sl_period': 5,                        # ATR-SL: ATR calculation period
        'atr_sl_hhv': 10,                          # ATR-SL: HHV lookback period
        'atr_sl_multiplier': 2.5,                  # ATR-SL: ATR multiplier
    },

    # Closure - Peak: Avg RSI (10min) based exit in last 30 minutes of trading day
    'closure_peak': {
        'enabled': True,
        'rsi_call_threshold': 85,              # Sell CALL contracts when Avg RSI (10min) >= this
        'rsi_put_threshold': 15,               # Sell PUT contracts when Avg RSI (10min) <= this
        'minutes_before_close': 30,            # Activate in last N minutes (15:30+)
    },

    # Momentum Peak: Detect momentum exhaustion peaks for early exit
    # Combines RSI overbought reversal + EWO decline + RSI < avg confirmation
    # Designed to exit 1-2 bars after a peak, before the bulk of the reversal
    'momentum_peak': {
        'enabled': True,
        'min_profit_pct': 15,              # Only consider when option profit >= this %
        'rsi_overbought': 80,              # RSI must have reached this recently
        'rsi_lookback': 5,                 # Bars back to check for overbought RSI
        'rsi_drop_threshold': 10,          # RSI must drop by >= this from recent peak
        'ewo_declining_bars': 1,           # EWO must decline for N consecutive bars
        'require_rsi_below_avg': True,     # Require RSI < RSI_10min_avg
    },

    # StatsBook Exit: Exit based on historical statistical bounds
    # Uses StatsBook data to identify when price action reaches historical extremes
    # Min = noise floor, Median = normal zone, Max = selling zone
    'statsbook_exit': {
        'enabled': True,
        'timeframe': '5m',                 # StatsBook timeframe to compare against
        'ewo_max_exit': True,              # Exit when EWO >= Median.Max(EWO)
        'hl_max_exit': False,              # Exit when rolling H-L >= Median.Max(H-L)
        'min_profit_pct': 10,              # Minimum option profit % to consider exit
        'min_hold_bars': 5,                # Minimum bars held before StatsBook exit
        'rolling_window': 5,               # Bars for rolling H-L range calculation
    },

    # Volume Climax Exit: Detect exhaustion via volume spike + price reversal
    # A sudden volume spike (Nx above rolling average) combined with a price
    # reversal bar signals institutional exhaustion. High-volume reversals are
    # among the most reliable intraday signals for directional shifts.
    'volume_climax_exit': {
        'enabled': True,
        'volume_lookback': 20,             # Bars for rolling avg volume calculation
        'volume_multiplier': 3.0,          # Volume must be >= Nx rolling avg to qualify
        'min_profit_pct': 10,              # Minimum option profit % to consider exit
        'min_hold_bars': 10,               # Minimum bars held before checking
    },

    # Time Stop: Exit stale positions that haven't moved meaningfully
    # Options lose value every minute via theta decay. Holding a position
    # that isn't moving costs real money. Frees capital for redeployment.
    'time_stop': {
        'enabled': True,
        'max_minutes': 90,                 # Exit if held longer than N minutes
        'min_profit_pct': 5,               # ... and profit is below this %
    },

    # VWAP Cross Exit: Exit when price crosses VWAP against position direction
    # VWAP is the institutional benchmark. Price crossing to the adverse side
    # signals that institutional flow has shifted against the position.
    # Requires confirmation (N bars on wrong side) to avoid whipsaws.
    'vwap_cross_exit': {
        'enabled': True,
        'min_profit_pct': 5,               # Minimum option profit % to consider exit
        'min_hold_bars': 10,               # Minimum bars held before checking
        'confirm_bars': 2,                 # Bars price must stay on adverse side of VWAP
    },

    # Supertrend Flip Exit: Exit when Supertrend direction flips against position
    # Supertrend uses ATR-based bands to define trend direction. A flip from
    # favorable to adverse means price broke through a volatility-adjusted
    # support/resistance level — a mechanical trend reversal confirmation.
    'supertrend_flip_exit': {
        'enabled': True,
        'min_profit_pct': 5,               # Minimum option profit % to consider exit
        'min_hold_bars': 5,                # Minimum bars held before checking
        'confirm_bars': 1,                 # Bars adverse direction must persist (1 = immediate)
    },

    # AI Exit Signal: Local LLM-based exit signal generation
    # Runs a quantized model via llama-cpp-python on GPU during backtesting.
    # The model analyzes multi-timeframe technical data and recommends hold/sell.
    'ai_exit_signal': {
        'enabled': False,                              # Disabled by default (requires model file)
        'model_path': r'C:\Users\Vadim\OneDrive\Software\Meta-Llama-3-8B-Instruct.Q5_K_M.gguf',  # Absolute path to GGUF model file
        'n_gpu_layers': -1,                            # GPU layers to offload (-1 = all)
        'n_ctx': 2048,                                 # Context window (tokens)
        'temperature': 0.1,                            # Low = deterministic (good for backtesting)
        'max_tokens': 256,                             # Max response tokens
        'seed': 42,                                    # Fixed seed for reproducible backtests
        'eval_interval': 5,                            # Evaluate every N bars (1 = every bar, 5 = every 5 min)
        'min_bars_before_eval': 5,                     # Minimum bars held before first AI evaluation
        'exit_on_sell': True,                          # Actually exit when AI says 'sell'
        'log_inferences': True,                        # Save inference data for self-training
        'log_dir': 'ai_training_data',                 # Directory for training data logs
    },

    # Risk Assessment: Detect overbought entry conditions
    # Evaluates at signal time whether we are buying into an overbought zone.
    # If any condition is TRUE, RISK = HIGH.
    'risk_assessment': {
        'enabled': True,
        # Condition 1: RSI overbought — (RSI + RSI_avg) / 2 > threshold
        'rsi_overbought_threshold': 80,
        # Condition 2: EWO overbought — EWO_avg > Median.Max(EWO) from StatsBook (1m)
        'ewo_overbought_enabled': True,
        # Condition 3: First 15 minutes of market open (9:30 - 9:45 EST)
        'market_open_window_minutes': 15,
        # Post-purchase monitoring for HIGH risk trades
        'downtrend_delay_minutes': 5,      # Wait N minutes after entry before monitoring
        'downtrend_monitor_bars': 3,       # If next N bars are all negative, sell
        'downtrend_drop_pct': 10,          # OR if option drops X% below entry, sell
        'downtrend_exit_reason': 'DownTrend-SL',  # Exit reason label for risk downtrend
    },

    # SPY Market Gauge: Use SPY as overall market health indicator
    # Compares current SPY price against average price over lookback windows.
    # If current > avg → Bullish, else → Bearish for that timeframe.
    'spy_gauge': {
        'enabled': True,
        'ticker': 'SPY',
        'timeframes': {
            'since_open': 0,     # 0 = from market open (9:30)
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
        },
    },

    # Bias sensitivity: Controls the sideways band width for market bias calculation.
    # Smaller = more sensitive (more bull/bear signals), Larger = wider band (more sideways).
    # Default was 0.10, reduced to 0.05 for higher sensitivity.
    'bias_sideways_band': 0.05,

    # ==========================================================================
    # Options Exit System (Primary TP/SL)
    # ==========================================================================
    # Core exit system: take profit via trailing stop loss, hard stop loss on
    # option contract price.  Differentiates CALLs vs PUTs — a rising stock is
    # good for calls / bad for puts, and vice-versa.
    #
    # Flow:
    #   1. At entry → set initial hard SL at -20 % of option price
    #   2. Assess entry favorability (30-min lookback, EMA positioning, ATR-SL)
    #   3. Once profit >= trail_activation_pct → engage trailing SL
    #   4. Trailing SL scales continuously with profit margin
    #   5. Monitor for reversals against our contract type
    'options_exit': {
        'enabled': True,

        # --- Initial Stop Loss ---
        'initial_sl_pct': 20,              # Hard SL: exit if option drops X% below entry (adjustable)

        # --- Trailing Stop Loss ---
        'trail_activation_pct': 10,        # Engage trailing SL when profit margin >= X%
        'trail_base_floor_pct': 5,         # At activation: lock in X% above entry as floor
        'trail_early_floor_minutes': 5,    # First N minutes: SL floor = 0% (breakeven) instead of base

        # Continuous trailing SL scaling parameters (logarithmic curve)
        # trail_sl = base + scale * ln(1 + profit / norm)
        # At 10% profit  → SL ~5%   (just above entry)
        # At 50% profit  → SL ~35%
        # At 100% profit → SL ~70%
        # At 200% profit → SL ~115%
        'trail_scale': 25.0,               # Controls slope of the trailing curve
        'trail_norm': 30.0,                # Normalisation factor (higher = slower ramp)

        # High-risk addon: extra % locked in when entry is flagged HIGH risk
        # addon = risk_addon_base + risk_addon_scale * ln(1 + profit / risk_addon_norm)
        # At 10% profit → +2%, at 100% profit → +15%
        'risk_addon_base': 2.0,
        'risk_addon_scale': 5.0,
        'risk_addon_norm': 30.0,

        # --- RiskOutlook (Entry Favorability Assessment) ---
        # Two time horizons:
        #   Primary: past 30 minutes (ROC over roc_period bars)
        #   Secondary: since market open (computed dynamically from first bar of day)
        # Confirmation window: first N minutes after purchase to confirm direction
        'confirmation_window_bars': 10,    # Observe 5-10 bars post-entry

        # RSI thresholds for overbought/oversold (stock-level)
        'rsi_overbought': 70,              # Stock RSI above = overbought zone
        'rsi_oversold': 30,                # Stock RSI below = oversold zone

        # --- EMA Reversal Detection ---
        # Price below short EMAs = higher reversal risk for calls (lower for puts).
        # Sensitivity: how many EMAs must be breached to flag a reversal.
        'ema_reversal_periods': [10, 21, 50],     # EMAs to check for reversal
        'ema_reversal_sensitivity': 2,             # N of those EMAs breached = reversal (adjustable)

        # --- ATR-SL Favorability ---
        # If stock price is below ATR-SL at entry → unfavorable for calls, favorable for puts
        'atr_sl_enabled': True,

        # --- MACD Confirmation ---
        # Use MACD histogram direction to confirm trend alignment with contract type
        'macd_enabled': True,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,

        # --- Price Momentum (ROC) ---
        # Rate of change over lookback to quantify 30-min trend strength
        'roc_period': 30,                  # 30-bar ROC for trend strength
    },
}


# =============================================================================
# DATAFRAME COLUMN DEFINITIONS (Source of Truth)
# =============================================================================
# All DataFrame column ordering is defined here.
# Modules reference these lists for consistent column order.

DATAFRAME_COLUMNS = {
    # Discord messages fetched by DiscordFetcher (Test.py)
    'discord_messages': [
        'id', 'timestamp', 'content', 'author', 'author_id',
    ],

    # Parsed trading signals from SignalParser (Test.py, Signal.py)
    'signals': [
        'ticker', 'strike', 'option_type', 'expiration',
        'cost', 'signal_time', 'message_id', 'raw_message',
    ],

    # Closed position results from Backtest (Test.py)
    'positions': [
        'ticker', 'strike', 'option_type', 'expiration', 'signal_time',
        'entry_price', 'entry_time', 'exit_price', 'exit_time', 'exit_reason',
        'contracts', 'highest_price', 'lowest_price',
        'pnl', 'pnl_pct', 'minutes_held',
        'max_price_to_eod',
    ],

    # Per-bar tracking data from Databook (Test.py)
    'databook': [
        'timestamp', 'stock_price', 'stock_high', 'stock_low', 'true_price',
        'option_price', 'volume', 'holding', 'entry_price',
        'pnl', 'pnl_pct', 'highest_price', 'lowest_price', 'minutes_held',
        'risk', 'risk_reasons', 'risk_trend',
        'market_bias',
        'spy_price', 'spy_since_open', 'spy_1m', 'spy_5m', 'spy_15m', 'spy_30m', 'spy_1h',
        'ticker_since_open', 'ticker_1m', 'ticker_5m', 'ticker_15m', 'ticker_30m', 'ticker_1h',
        'vwap', 'ema_10', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
        'vwap_ema_avg', 'emavwap', 'ewo', 'ewo_15min_avg', 'rsi', 'rsi_10min_avg',
        'supertrend', 'supertrend_direction',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
        'atr_sl',
        # MACD indicator columns
        'macd_line', 'macd_signal', 'macd_histogram',
        # Price momentum (ROC)
        'roc',
        'ai_outlook_1m', 'ai_outlook_5m', 'ai_outlook_30m', 'ai_outlook_1h',
        'ai_action', 'ai_reason',
        # Options Exit System columns (SL = stop loss, TP = take profit)
        'sl_trailing',             # Current trailing SL price level
        'sl_hard',                 # Hard stop loss price level
        'tp_risk_outlook',         # RiskOutlook score: LOW / MEDIUM / HIGH
        'tp_risk_reasons',         # Pipe-separated reasons
        'tp_trend_30m',            # 30-min trend: Uptrend / Downtrend / Sideways
        'sl_ema_reversal',         # EMA reversal flag (True/False)
        'tp_confirmed',            # Post-entry confirmation: Confirmed / Denied / Pending
        # Exit signal flags (per-bar boolean: which signals would fire)
        'exit_sig_sb', 'exit_sig_mp', 'exit_sig_ai',
        'exit_sig_closure_peak',
        'exit_sig_oe',             # Options Exit system SL/TP triggered
        'exit_sig_vc',             # Volume Climax exit triggered
        'exit_sig_ts',             # Time Stop exit triggered
        'exit_sig_vwap',           # VWAP Cross exit triggered
        'exit_sig_st',             # Supertrend Flip exit triggered
    ],

    # Metadata columns appended to databook (Test.py)
    'databook_metadata': [
        'trade_label', 'ticker', 'strike', 'option_type', 'expiration',
        'contracts', 'entry_time', 'exit_time', 'exit_reason',
    ],

    # DataSummary: 2-minute aggregated summary of DataBook for dashboard display
    'datasummary': [
        'timestamp_start', 'timestamp_end',
        'stock_open', 'stock_high', 'stock_low', 'stock_close',
        'option_open', 'option_high', 'option_low', 'option_close',
        'volume_sum', 'bar_count',
        'pnl_pct_start', 'pnl_pct_end', 'pnl_pct_max', 'pnl_pct_min',
        'rsi_avg', 'ewo_avg', 'vwap_avg',
        'market_bias_mode',
        'exit_signals_fired',
        'signal_id',
    ],

    # DataStats: per-signal trade statistics
    'datastats': [
        'signal_id', 'ticker', 'strike', 'option_type', 'expiration',
        'entry_time', 'entry_price', 'entry_stock_price',
        'exit_time', 'exit_price', 'exit_reason',
        'pnl', 'pnl_pct', 'minutes_held',
        'max_pnl_pct', 'min_pnl_pct', 'max_option_price', 'min_option_price',
        'risk_level', 'risk_reasons',
        'bars_recorded', 'data_source',
    ],

    # Dashboard databook display columns (Dashboard.py)
    # Mirrors 'databook' columns so the dashboard table reflects the full databook.
    'dashboard_databook': [
        'timestamp', 'stock_price', 'stock_high', 'stock_low', 'true_price',
        'option_price', 'volume', 'holding', 'entry_price',
        'pnl', 'pnl_pct', 'highest_price', 'lowest_price', 'minutes_held',
        'risk', 'risk_reasons', 'risk_trend',
        'market_bias',
        'spy_price', 'spy_since_open', 'spy_1m', 'spy_5m', 'spy_15m', 'spy_30m', 'spy_1h',
        'ticker_since_open', 'ticker_1m', 'ticker_5m', 'ticker_15m', 'ticker_30m', 'ticker_1h',
        'vwap', 'ema_10', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
        'vwap_ema_avg', 'emavwap', 'ewo', 'ewo_15min_avg', 'rsi', 'rsi_10min_avg',
        'supertrend', 'supertrend_direction',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
        'atr_sl',
        # MACD indicator columns
        'macd_line', 'macd_signal', 'macd_histogram',
        # Price momentum (ROC)
        'roc',
        'ai_outlook_1m', 'ai_outlook_5m', 'ai_outlook_30m', 'ai_outlook_1h',
        'ai_action', 'ai_reason',
        # Options Exit System columns (SL = stop loss, TP = take profit)
        'sl_trailing',
        'sl_hard',
        'tp_risk_outlook',
        'tp_risk_reasons',
        'tp_trend_30m',
        'sl_ema_reversal',
        'tp_confirmed',
        # Exit signal flags (per-bar boolean: which signals would fire)
        'exit_sig_sb', 'exit_sig_mp', 'exit_sig_ai',
        'exit_sig_closure_peak',
        'exit_sig_oe',
    ],
}

# Resolve alias: dashboard_databook uses the same columns as databook
DATAFRAME_COLUMNS['dashboard_databook'] = DATAFRAME_COLUMNS['databook']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config(section):
    """Get configuration section by name."""
    configs = {
        'discord': DISCORD_CONFIG,
        'data': DATA_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'robinhood': ROBINHOOD_CONFIG,
        'live': LIVE_CONFIG,
        'ai': BACKTEST_CONFIG.get('ai_exit_signal', {}),
    }
    return configs.get(section.lower(), {})


def get_setting(section, key, default=None):
    """Get individual configuration setting."""
    config = get_config(section)
    return config.get(key, default)


def load_config_json(config_path='config.json'):
    """Load configuration overrides from JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
            return {}
    return {}


# Load JSON overrides at import time
CONFIG_JSON = load_config_json('config.json')


def validate_config():
    """Validate active configuration settings. Returns (errors, warnings)."""
    errors = []
    warnings = []

    if not DISCORD_CONFIG.get('token'):
        warnings.append("DISCORD_TOKEN not set (required for live trading)")
    if not DISCORD_CONFIG.get('channel_id'):
        warnings.append("DISCORD_CHANNEL_ID not set (required for live trading)")

    return errors, warnings


def validate_and_report():
    """Validate settings and print report. Returns True if valid."""
    errors, warnings = validate_config()

    if warnings:
        print("\n" + "=" * 60)
        print("CONFIGURATION WARNINGS:")
        print("=" * 60)
        for warning in warnings:
            print(f"  ! {warning}")
        print("=" * 60 + "\n")

    if errors:
        print("\n" + "!" * 60)
        print("CONFIGURATION ERRORS:")
        print("!" * 60)
        for error in errors:
            print(f"  X {error}")
        print("!" * 60 + "\n")
        return False

    if not warnings:
        print("\n[OK] Configuration validated successfully\n")
    else:
        print("[OK] Configuration valid (with warnings above)\n")

    return True
