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
        'min_delay': 2.0,                          # Min seconds between requests
        'max_delay': 5.0,                          # Max seconds between requests
        'batch_size': 50,                          # Messages per request
        'long_pause_chance': 0.15,                 # Chance of longer pause (15%)
        'long_pause_min': 8.0,                     # Long pause min (seconds)
        'long_pause_max': 15.0,                    # Long pause max (seconds)
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
        'min_option_price': 0.01,                  # Min option price floor
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
    'stop_loss': {
        'enabled': True,
        'stop_loss_pct': 0.25,                     # Max loss from entry (25%)
        'stop_loss_warning_pct': 0.15,             # Warning threshold (15%)
        'profit_target_pct': 1.00,                 # Profit target (100%)
        'trailing_stop_pct': 0.20,                 # Trailing below highest (20%)
        'time_stop_minutes': 60,                   # Time stop (minutes)
        'breakeven_threshold_pct': None,           # None = auto-calculate as entry/(1-stop_loss_pct)
        'breakeven_min_minutes': 30,               # Min hold before breakeven transition
        'trailing_trigger_pct': 0.50,              # Start trailing at profit (50%)
    },

    # Technical indicators for backtest
    'indicators': {
        'ema_period': 25,                          # EMA period (bars)
        'vwap_enabled': True,                      # Calculate VWAP
        'rsi_period': 14,                          # RSI calculation period
        'rsi_overbought': 70,                      # RSI overbought threshold
        'rsi_oversold': 30,                        # RSI oversold threshold
    },

    # --- Unused backtest settings (commented out for review) ---
    # 'position_size_pct': 0.02,                   # 2% per trade
    # 'max_positions': 5,                          # Max concurrent positions
    # 'commission_per_trade': 1.0,                 # $1 per trade
    # 'candle_interval': '1m',                     # Candle interval
    # 'use_simulated_fills': True,                 # Use simulated fills
    # 'instrument_type': 'option',                 # Instrument type
    # 'exit_strategies': {                         # Exit strategy toggles
    #     'use_stop_loss': True,
    #     'use_sl_ema': False,
    #     'use_profit_target': True,
    #     'use_trailing_stop': False,
    #     'use_peak_exit': False,
    #     'use_time_stop': False,
    #     'use_expiration_stop': False,
    #     'use_discord_signal': False,
    #     'use_end_of_day': False,
    #     'use_max_hold_days': False,
    #     'use_rsi': False,
    #     'use_macd': False,
    #     'use_vwap': False,
    #     'use_vpoc': False,
    #     'use_supertrend': False,
    # },
    # 'indicator_settings': {                      # Options-specific indicator overrides
    #     'rsi_period': 14,
    #     'rsi_overbought': 80,
    #     'rsi_oversold': 20,
    #     'macd_fast': 12,
    #     'macd_slow': 26,
    #     'macd_signal': 9,
    #     'vwap_threshold_pct': 0.02,
    #     'vpoc_threshold_pct': 0.03,
    #     'supertrend_period': 10,
    #     'supertrend_multiplier': 3.0,
    # },
    # 'swing_trade': {                             # Swing trading settings
    #     'enabled': False,
    #     'allow_overnight_holds': False,
    #     'max_hold_days': 5,
    #     'use_end_of_day_exit': False,
    #     'eod_exit_time': '15:45',
    # },
    # 'auto_export_matrices': False,               # Auto-export matrices after backtest
    # 'export_matrices_dir': 'Data_BT',            # Export directory
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
    ],

    # Per-bar tracking data from TrackingMatrix (Test.py)
    'tracking_matrix': [
        'timestamp', 'stock_price', 'stock_high', 'stock_low', 'true_price', 'atr',
        'option_price', 'volume', 'holding', 'entry_price',
        'pnl', 'pnl_pct', 'highest_price', 'lowest_price', 'minutes_held',
        'stop_loss', 'stop_loss_mode',
        'vwap', 'ema_30', 'ewo', 'ewo_15min_avg', 'rsi',
    ],

    # Metadata columns appended to tracking matrix (Test.py)
    'tracking_matrix_metadata': [
        'trade_label', 'ticker', 'strike', 'option_type', 'expiration',
        'contracts', 'entry_time', 'exit_time', 'exit_reason',
    ],

    # Dashboard matrix display columns (Dashboard.py)
    'dashboard_matrix': [
        'timestamp', 'holding', 'stock_price', 'stock_high', 'stock_low',
        'true_price', 'option_price', 'pnl_pct',
        'stop_loss', 'stop_loss_mode', 'sl_cushion',
        'vwap', 'ema_20', 'ema_30', 'ewo', 'ewo_15min_avg', 'rsi',
        'SL_C1', 'SL_C2',
    ],
}


# =============================================================================
# UNUSED MODULES - Commented out for review
# =============================================================================
# Settings defined in Alpha.MD but not yet used by any active script.
# Uncomment when their corresponding modules are implemented.

# TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'paper' or 'live'

# --- Credentials (Webull.py) ---
# CREDENTIALS_CONFIG = {
#     'email': os.getenv('WEBULL_EMAIL', ''),
#     'password': os.getenv('WEBULL_PASSWORD', ''),
#     'device_id': os.getenv('WEBULL_DEVICE_ID', ''),
#     'trading_pin': os.getenv('WEBULL_TRADING_PIN', ''),
# }

# --- Market Hours (Data.py, Main.py) ---
# MARKET_CONFIG = {
#     'market_hours_start': 9.5,                   # 9:30 AM ET
#     'market_hours_end': 16.0,                    # 4:00 PM ET
#     'trading_days': [0, 1, 2, 3, 4],             # Mon-Fri
#     'enable_premarket': False,
#     'premarket_start': 4.0,                      # 4:00 AM ET
#     'enable_afterhours': False,
#     'afterhours_end': 20.0,                      # 8:00 PM ET
#     'timezone': 'America/New_York',
# }

# --- Paper Trading (PaperTrading.py) ---
# PAPER_CONFIG = {
#     'initial_balance': 10000.00,
#     'state_file': 'paper_trading_state.json',
#     'simulated_slippage_pct': 0.001,             # 0.1% slippage
#     'simulated_fill_delay_ms': 100,              # Fill delay (ms)
#     'simulated_spread_pct': 0.0001,              # 0.01% bid/ask spread
#     'use_real_prices': True,
# }

# --- Risk Management (Strategy.py, Main.py) ---
# RISK_CONFIG = {
#     'max_position_size_pct': 0.02,               # Max 2% per trade
#     'max_daily_loss_pct': 0.05,                  # Stop at 5% daily loss
#     'max_concurrent_positions': 5,
#     'instrument_type': 'stock',
#     'stock_defaults': {
#         'stop_loss_pct': 0.02,
#         'profit_target_pct': 0.05,
#         'trailing_stop_pct': 0.05,
#         'time_stop_minutes': None,
#         'use_trailing_stop': True,
#         'use_profit_target': True,
#         'use_time_stop': False,
#         'use_technical_stop': True,
#         'use_expiration_stop': False,
#     },
#     'option_defaults': {
#         'profit_target_pct': 1.00,
#         'use_profit_target': True,
#         'use_technical_stop': True,
#         'use_expiration_stop': True,
#         'stop_loss': {
#             'enabled': True,
#             'stop_loss_pct': 0.35,
#             'stop_loss_warning_pct': 0.15,
#             'trailing_stop_pct': 0.20,
#             'time_stop_minutes': 60,
#             'breakeven_threshold_pct': None,
#             'breakeven_min_minutes': 30,
#             'trailing_trigger_pct': 0.50,
#         },
#     },
#     'atr_multiplier': 2.0,
#     'reward_ratio': 2.0,
# }

# --- Contract Limits (Signal.py, Strategy.py) ---
# CONTRACT_LIMITS_CONFIG = {
#     'max_contract_price': 10.00,                 # Max price per contract ($10)
#     'max_contract_capital': 1000.00,             # Max total capital per order ($1000)
#     'enforce_limits': True,
# }

# --- Entry Conditions (Strategy.py, Main.py) ---
# ENTRY_CONFIG = {
#     'min_entry_confidence': 30,                  # Confidence threshold (-100 to 100)
#     'require_volume_confirmation': True,
#     'volume_multiplier': 1.2,                    # Volume must be 1.2x average
#     'require_trend_confirmation': True,
#     'allow_countertrend_entries': False,
#     'max_entry_price_range_pct': 0.02,           # Entry within 2% of signal price
# }

# --- Exit Conditions (Strategy.py, Test.py) ---
# EXIT_CONFIG = {
#     'use_stop_loss': True,
#     'use_profit_target': True,
#     'use_trailing_stop': True,
#     'use_time_stop': False,
#     'use_technical_stop': True,
#     'use_expiration_stop': False,
#     'use_discord_signal': True,
#     'exit_on_opposite_signal': True,
#     'use_rsi_exit': True,
#     'use_macd_exit': True,
#     'use_vwap_exit': True,
#     'use_vpoc_exit': True,
#     'use_supertrend_exit': True,
# }

# --- Order Execution (Orders.py, Main.py) ---
# ORDER_CONFIG = {
#     'order_type': 'market',                      # 'market' or 'limit'
#     'limit_price_offset_pct': 0.01,              # 1% offset for limit orders
#     'check_order_status_interval': 5,            # Check every 5 seconds
#     'order_timeout_seconds': 300,                # Cancel after 5 minutes
#     'allow_partial_fills': True,
# }

# --- System/Logging (Main.py, Manager.py) ---
# SYSTEM_CONFIG = {
#     'enable_performance_tracking': True,
#     'log_level': 'INFO',                         # DEBUG, INFO, WARNING, ERROR
#     'max_log_file_size_mb': 100,
#     'keep_old_logs_days': 30,
#     'enable_position_logging': True,
#     'enable_trade_logging': True,
#     'position_log_cleanup_days': 30,
#     'cycle_sleep_ms': 1000,                      # Sleep between cycles
#     'min_cycle_time_ms': 500,                    # Min cycle duration
# }

# --- Alerts/Notifications (Main.py, Manager.py) ---
# ALERT_CONFIG = {
#     'enable_position_alerts': True,
#     'enable_loss_alerts': True,
#     'enable_profit_alerts': True,
#     'alert_webhook_url': os.getenv('ALERT_WEBHOOK_URL', ''),
#     'large_loss_threshold': 100,                 # Alert if loss > $100
#     'large_profit_threshold': 500,               # Alert if profit > $500
# }


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
    }
    return configs.get(section.lower(), {})


def get_setting(section, key, default=None):
    """Get individual configuration setting."""
    config = get_config(section)
    return config.get(key, default)


# def get_risk_defaults(instrument_type='stock'):
#     """Get risk management defaults for instrument type."""
#     if instrument_type == 'option':
#         return RISK_CONFIG['option_defaults'].copy()
#     return RISK_CONFIG['stock_defaults'].copy()


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


# def print_config_summary():
#     """Print a summary of current configuration to console."""
#     print("\n" + "=" * 60)
#     print("TRADING BOT CONFIGURATION SUMMARY")
#     print("=" * 60)
#     print(f"\nDISCORD:")
#     print(f"  Token Set: {bool(DISCORD_CONFIG.get('token'))}")
#     print(f"  Channel: {DISCORD_CONFIG.get('channel_id')}")
#     print(f"\nANALYSIS:")
#     print(f"  Min Bars Required: {ANALYSIS_CONFIG['min_bars_required']}")
#     print(f"  RSI Period: {ANALYSIS_CONFIG['rsi_period']}")
#     print(f"\nBACKTEST:")
#     print(f"  Lookback Days: {BACKTEST_CONFIG['lookback_days']}")
#     print(f"  Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,.2f}")
#     print("=" * 60 + "\n")
