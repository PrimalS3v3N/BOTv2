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
# ANALYSIS MODULE (Analysis.py)
# =============================================================================

ANALYSIS_CONFIG = {
    'sma_period': 20,                              # SMA period
    'ema_fast_period': 12,                         # EMA fast period
    'ema_slow_period': 26,                         # EMA slow period
    'rsi_period': 14,                              # RSI lookback period
    'rsi_overbought': 70,                          # RSI overbought (stock default; use 80 for options)
    'rsi_oversold': 30,                            # RSI oversold (stock default; use 20 for options)
    'ewo_fast': 5,                                 # EWO fast EMA period
    'ewo_slow': 35,                                # EWO slow EMA period
    'support_resistance_lookback': 20,             # S/R lookback bars
    'trend_fast_period': 10,                       # Trend fast EMA
    'trend_slow_period': 20,                       # Trend slow EMA
    'momentum_period': 10,                         # Momentum period
    'roc_period': 10,                              # Rate of change period
    'min_bars_required': 20,                       # Min data bars required
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

    # Stop Loss - three phases: initial -> breakeven -> trailing
    # Normal mode (default)
    'stop_loss': {
        'enabled': True,
        'stop_loss_pct': 0.27,                     # Max loss from entry (27%)
        'stop_loss_warning_pct': 0.15,             # Warning threshold (15%)
        'profit_target_pct': 1.00,                 # Profit target (100%)
        'trailing_stop_pct': 0.27,                 # Trailing below highest (27%)
        'time_stop_minutes': 60,                   # Time stop (minutes)
        'breakeven_threshold_pct': None,           # None = auto-calculate as entry/(1-stop_loss_pct)
        'breakeven_min_minutes': 30,               # Min hold before breakeven transition
        'trailing_trigger_pct': 0.30,              # Start trailing at profit (30%)
        'reversal_exit_enabled': True,             # Exit when True Price < VWAP (reversal)
        'downtrend_exit_enabled': True,            # Exit when True Price & EMA < vwap_ema_avg (downtrend)
    },
    # Tight stop loss — used when RISK=HIGH + Downtrend (cut losses faster)
    'stop_loss_tight': {
        'stop_loss_pct': 0.15,                     # Max loss from entry (15%)
        'stop_loss_warning_pct': 0.08,             # Warning threshold (8%)
        'trailing_stop_pct': 0.15,                 # Trailing below highest (15%)
        'breakeven_min_minutes': 15,               # Min hold before breakeven (15 min)
        'trailing_trigger_pct': 0.15,              # Start trailing at profit (15%)
    },
    # Loose stop loss — wider tolerance for strong trends
    'stop_loss_loose': {
        'stop_loss_pct': 0.40,                     # Max loss from entry (40%)
        'stop_loss_warning_pct': 0.20,             # Warning threshold (20%)
        'trailing_stop_pct': 0.35,                 # Trailing below highest (35%)
        'breakeven_min_minutes': 45,               # Min hold before breakeven (45 min)
        'trailing_trigger_pct': 0.50,              # Start trailing at profit (50%)
    },

    # Technical indicators for backtest
    'indicators': {
        'ema_period': 25,                          # EMA period (bars)
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
        'hl_max_exit': True,               # Exit when rolling H-L >= Median.Max(H-L)
        'min_profit_pct': 10,              # Minimum option profit % to consider exit
        'min_hold_bars': 5,                # Minimum bars held before StatsBook exit
        'rolling_window': 5,               # Bars for rolling H-L range calculation
    },

    # AI Exit Signal: Local LLM-based exit signal generation
    # Runs a quantized model via llama-cpp-python on GPU during backtesting.
    # The model analyzes multi-timeframe technical data and recommends hold/sell.
    'ai_exit_signal': {
        'enabled': True,                               # Disabled by default (requires model file)
        'model_path': 'models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf',  # Absolute path to GGUF model file
        'n_gpu_layers': -1,                            # GPU layers to offload (-1 = all)
        'n_ctx': 4096,                                 # Context window (tokens)
        'temperature': 0.1,                            # Low = deterministic (good for backtesting)
        'max_tokens': 512,                             # Max response tokens
        'seed': 42,                                    # Fixed seed for reproducible backtests
        'eval_interval': 5,                            # Evaluate every N bars (1 = every bar, 5 = every 5 min)
        'min_bars_before_eval': 5,                     # Minimum bars held before first AI evaluation
        'exit_on_sell': True,                          # Actually exit when AI says 'sell'
        'log_inferences': True,                        # Save inference data for self-training
        'log_dir': 'ai_training_data',                 # Directory for training data logs
    },

    # Take Profit - Milestones: Ratcheting trailing stops at profit levels
    # Once a milestone gain% is reached, trailing stop is set at trailing_pct% above entry.
    # Milestones only ratchet upward — once reached, trailing stop never drops below that level.
    # Exit triggers when price falls to or below the trailing stop price.
    'take_profit_milestones': {
        'enabled': True,
        # Normal milestones (default)
        'milestones': [
            {'gain_pct': 10,  'trailing_pct': 0},     # Breakeven at entry cost
            {'gain_pct': 20,  'trailing_pct': 5},     # 5% above entry
            {'gain_pct': 30,  'trailing_pct': 10},    # 10% above entry
            {'gain_pct': 40,  'trailing_pct': 20},    # 20% above entry
            {'gain_pct': 50,  'trailing_pct': 30},    # 30% above entry
            {'gain_pct': 75,  'trailing_pct': 50},    # 50% above entry
            {'gain_pct': 100, 'trailing_pct': 75},    # 75% above entry
            {'gain_pct': 125, 'trailing_pct': 95},    # 95% above entry
            {'gain_pct': 150, 'trailing_pct': 115},   # 115% above entry
            {'gain_pct': 200, 'trailing_pct': 150},   # 150% above entry
            {'gain_pct': 300, 'trailing_pct': 225},   # 225% above entry
            {'gain_pct': 400, 'trailing_pct': 300},   # 300% above entry
            {'gain_pct': 500, 'trailing_pct': 375},   # 375% above entry
        ],
        # Tight milestones — used when RISK=HIGH + Uptrend (lock profits faster)
        'milestones_tight': [
            {'gain_pct': 5,   'trailing_pct': 0},     # Breakeven at entry cost
            {'gain_pct': 10,  'trailing_pct': 3},     # 3% above entry
            {'gain_pct': 15,  'trailing_pct': 7},     # 7% above entry
            {'gain_pct': 20,  'trailing_pct': 12},    # 12% above entry
            {'gain_pct': 30,  'trailing_pct': 20},    # 20% above entry
            {'gain_pct': 40,  'trailing_pct': 28},    # 28% above entry
            {'gain_pct': 50,  'trailing_pct': 38},    # 38% above entry
            {'gain_pct': 75,  'trailing_pct': 60},    # 60% above entry
            {'gain_pct': 100, 'trailing_pct': 85},    # 85% above entry
            {'gain_pct': 150, 'trailing_pct': 130},   # 130% above entry
            {'gain_pct': 200, 'trailing_pct': 175},   # 175% above entry
            {'gain_pct': 300, 'trailing_pct': 265},   # 265% above entry
        ],
        # Loose milestones — wider trailing for strong trends
        'milestones_loose': [
            {'gain_pct': 15,  'trailing_pct': 0},     # Breakeven at entry cost
            {'gain_pct': 30,  'trailing_pct': 5},     # 5% above entry
            {'gain_pct': 50,  'trailing_pct': 15},    # 15% above entry
            {'gain_pct': 75,  'trailing_pct': 35},    # 35% above entry
            {'gain_pct': 100, 'trailing_pct': 60},    # 60% above entry
            {'gain_pct': 150, 'trailing_pct': 100},   # 100% above entry
            {'gain_pct': 200, 'trailing_pct': 140},   # 140% above entry
            {'gain_pct': 300, 'trailing_pct': 210},   # 210% above entry
            {'gain_pct': 500, 'trailing_pct': 350},   # 350% above entry
        ],
    },

    # Risk Assessment: Detect overbought entry conditions
    # Evaluates at signal time whether we are buying into an overbought zone.
    # If any condition is TRUE, RISK = HIGH.
    'risk_assessment': {
        'enabled': True,
        # Condition 1: RSI overbought — (RSI + RSI_avg) / 2 > threshold
        'rsi_overbought_threshold': 80,
        # Condition 2: EWO overbought — EWO_avg > Median.Max(EWO) from StatsBook (1m:5m)
        'ewo_overbought_enabled': True,
        # Condition 3: First 15 minutes of market open (9:30 - 9:45 EST)
        'market_open_window_minutes': 15,
        # Post-purchase monitoring for HIGH risk trades
        'downtrend_monitor_bars': 3,       # If next N bars are all negative, sell
        'downtrend_drop_pct': 10,          # OR if option drops X% below entry, sell
        'downtrend_exit_reason': 'SL-DT',  # Exit reason label for risk downtrend
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
        'max_price_to_eod', 'max_stop_loss_price', 'profit_min',
        'highest_milestone_pct',
    ],

    # Per-bar tracking data from Databook (Test.py)
    'databook': [
        'timestamp', 'stock_price', 'stock_high', 'stock_low', 'true_price', 'atr',
        'option_price', 'volume', 'holding', 'entry_price',
        'pnl', 'pnl_pct', 'highest_price', 'lowest_price', 'minutes_held',
        'stop_loss', 'stop_loss_mode',
        'milestone_pct', 'trailing_stop_price',
        'risk', 'risk_reasons', 'risk_trend',
        'market_bias',
        'spy_price', 'spy_since_open', 'spy_1m', 'spy_5m', 'spy_15m', 'spy_30m', 'spy_1h',
        'vwap', 'ema_30', 'vwap_ema_avg', 'emavwap', 'ewo', 'ewo_15min_avg', 'rsi', 'rsi_10min_avg',
        'supertrend', 'supertrend_direction',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
        'ai_outlook_1m', 'ai_outlook_5m', 'ai_outlook_30m', 'ai_outlook_1h',
        'ai_action', 'ai_reason',
    ],

    # Metadata columns appended to databook (Test.py)
    'databook_metadata': [
        'trade_label', 'ticker', 'strike', 'option_type', 'expiration',
        'contracts', 'entry_time', 'exit_time', 'exit_reason',
    ],

    # Dashboard databook display columns (Dashboard.py)
    # Mirrors 'databook' columns so the dashboard table reflects the full databook.
    'dashboard_databook': [
        'timestamp', 'stock_price', 'stock_high', 'stock_low', 'true_price', 'atr',
        'option_price', 'volume', 'holding', 'entry_price',
        'pnl', 'pnl_pct', 'highest_price', 'lowest_price', 'minutes_held',
        'stop_loss', 'stop_loss_mode',
        'milestone_pct', 'trailing_stop_price',
        'risk', 'risk_reasons', 'risk_trend',
        'market_bias',
        'spy_price', 'spy_since_open', 'spy_1m', 'spy_5m', 'spy_15m', 'spy_30m', 'spy_1h',
        'vwap', 'ema_30', 'vwap_ema_avg', 'emavwap', 'ewo', 'ewo_15min_avg', 'rsi', 'rsi_10min_avg',
        'supertrend', 'supertrend_direction',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
        'ai_outlook_1m', 'ai_outlook_5m', 'ai_outlook_30m', 'ai_outlook_1h',
        'ai_action', 'ai_reason',
    ],
}


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

# Apply contract limits from config.json if present
# if 'contract_limits' in CONFIG_JSON:
#     CONTRACT_LIMITS_CONFIG.update(CONFIG_JSON['contract_limits'])


def validate_config():
    """Validate active configuration settings. Returns (errors, warnings)."""
    errors = []
    warnings = []

    if not DISCORD_CONFIG.get('token'):
        warnings.append("DISCORD_TOKEN not set (required for live trading)")
    if not DISCORD_CONFIG.get('channel_id'):
        warnings.append("DISCORD_CHANNEL_ID not set (required for live trading)")

    if ANALYSIS_CONFIG['min_bars_required'] < 1:
        errors.append("min_bars_required must be at least 1")

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
