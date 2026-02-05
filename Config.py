"""
Config.py - Centralized Configuration Management for Trading Bot

This module contains ALL configuration settings for the trading bot,
organized by their corresponding module for easy identification and modification.

Settings are grouped into sections:
    - TRADING MODE: Live vs Paper trading
    - CREDENTIALS: API keys and authentication (use environment variables!)
    - DISCORD: Discord bot settings (Discord.py, Signal.py)
    - MARKET: Market hours and trading days (Data.py, Main.py)
    - PAPER TRADING: Paper trading simulation (PaperTrading.py)
    - RISK MANAGEMENT: Position sizing, stops, targets (Strategy.py)
    - ANALYSIS: Technical indicator parameters (Analysis.py)
    - ENTRY: Trade entry conditions (Strategy.py)
    - EXIT: Trade exit conditions (Strategy.py)
    - DATA: Market data and caching (Data.py)
    - ORDERS: Order execution settings (Orders.py)
    - BACKTEST: Backtesting parameters (Test.py)
    - SYSTEM: Logging and performance (Main.py, Manager.py)
    - ALERTS: Notification settings

Usage:
    import Config

    # Get entire section
    discord_settings = Config.get_config('discord')

    # Get individual setting
    token = Config.get_setting('discord', 'token')

    # Direct access
    is_paper = Config.TRADING_MODE == 'paper'
"""

####### REQUIRED IMPORTS
import json
import os
from datetime import datetime

# Load environment variables from .env file (if exists)
# This allows storing credentials in a .env file instead of exporting them
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass


def _load_discord_token_from_excel():
    """
    Load Discord token from Setting.xlsx file (Row 1, Column B).
    This keeps sensitive credentials local and out of version control.

    Returns:
        str: Discord token or empty string if not found
    """
    try:
        import pandas as pd
        # Look for Setting.xlsx in the same directory as Config.py
        config_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(config_dir, 'Setting.xlsx')

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, header=None)
            # Row 1 = index 0, Column B = index 1
            token = df.iloc[0, 1]
            if pd.notna(token):
                return str(token).strip()
    except Exception as e:
        print(f"Warning: Could not load Discord token from Setting.xlsx: {e}")

    return ''


# Load Discord token from Excel file
_DISCORD_TOKEN_FROM_EXCEL = _load_discord_token_from_excel()


# =============================================================================
# TRADING MODE CONFIGURATION
# =============================================================================
# Controls whether the bot trades with real money or simulated paper trading

TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'paper' or 'live'


# =============================================================================
# CREDENTIALS CONFIGURATION
# =============================================================================
# SECURITY: Always use environment variables for sensitive credentials!
# Never commit actual credentials to version control.
#
# Used by: Webull.py

CREDENTIALS_CONFIG = {
    'email': os.getenv('WEBULL_EMAIL', ''),
    'password': os.getenv('WEBULL_PASSWORD', ''),
    'device_id': os.getenv('WEBULL_DEVICE_ID', ''),
    'trading_pin': os.getenv('WEBULL_TRADING_PIN', '')
}


# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================
# Settings for Discord bot integration and message fetching
#
# Used by: Discord.py, Signal.py, Test.py

DISCORD_CONFIG = {
    # Authentication - loads from Setting.xlsx (Row 1, Col B) or environment variable
    'token': _DISCORD_TOKEN_FROM_EXCEL or os.getenv('DISCORD_TOKEN', ''),
    'channel_id': os.getenv('DISCORD_CHANNEL_ID', '748401380288364575'),

    # API settings
    'api_version': 'v10',
    'base_url': 'https://discord.com/api',

    # Timeout settings (seconds)
    'send_timeout': 0.35,
    'receive_timeout': 0.35,

    # Message fetching
    'message_limit': 20,           # Messages per request (normal mode)
    'test_message_limit': 100,     # Messages per request (test mode)

    # Signal parsing
    'alert_marker': '<a:RedAlert:759583962237763595>',  # Discord alert emoji ID

    # Rate limiting - human-like request timing to avoid detection
    'rate_limit': {
        'min_delay': 2.0,           # Minimum seconds between requests
        'max_delay': 5.0,           # Maximum seconds between requests
        'batch_size': 50,           # Messages per request (lower = more natural)
        'long_pause_chance': 0.15,  # 15% chance of longer pause (simulates reading)
        'long_pause_min': 8.0,      # Long pause minimum seconds
        'long_pause_max': 15.0,     # Long pause maximum seconds
    },

    # Browser spoofing settings - makes requests appear as regular browser traffic
    'spoof_browser': True,         # Enable browser spoofing
    'browser_config': {
        # User-Agent string (Edge on Windows)
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0',

        # Standard browser headers
        'accept': '*/*',
        'accept_language': 'en-US,en;q=0.9',
        'accept_encoding': 'gzip, deflate, br',
        'connection': 'keep-alive',
        'cache_control': 'no-cache',
        'pragma': 'no-cache',

        # Security headers (Sec-Fetch-* for modern browsers)
        'sec_ch_ua': '"Not_A Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"',
        'sec_ch_ua_mobile': '?0',
        'sec_ch_ua_platform': '"Windows"',
        'sec_fetch_dest': 'empty',
        'sec_fetch_mode': 'cors',
        'sec_fetch_site': 'same-origin',

        # Discord-specific headers
        'x_discord_locale': 'en-US',
        'x_discord_timezone': 'America/New_York',
        'x_debug_options': 'bugReporterEnabled',

        # Referer/Origin
        'origin': 'https://discord.com',
        'referer': 'https://discord.com/channels/@me',
    },

    # X-Super-Properties (Discord client fingerprint - base64 encoded)
    # This identifies the client as a Discord web browser client
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
        'design_id': 0
    },
}


# =============================================================================
# MARKET CONFIGURATION
# =============================================================================
# Market hours and trading day settings
#
# Used by: Data.py, Main.py, Strategy.py

MARKET_CONFIG = {
    # Market hours (decimal format: 9.5 = 9:30 AM)
    'market_hours_start': 9.5,     # 9:30 AM ET
    'market_hours_end': 16.0,      # 4:00 PM ET

    # Trading days (0=Monday, 4=Friday)
    'trading_days': [0, 1, 2, 3, 4],

    # Extended hours
    'enable_premarket': False,
    'premarket_start': 4.0,        # 4:00 AM ET
    'enable_afterhours': False,
    'afterhours_end': 20.0,        # 8:00 PM ET

    # Timezone
    'timezone': 'America/New_York'
}


# =============================================================================
# PAPER TRADING CONFIGURATION
# =============================================================================
# Settings for simulated paper trading
#
# Used by: PaperTrading.py

PAPER_CONFIG = {
    # Account settings
    'initial_balance': 10000.00,
    'state_file': 'paper_trading_state.json',

    # Simulation settings
    'simulated_slippage_pct': 0.001,   # 0.1% slippage on fills
    'simulated_fill_delay_ms': 100,     # Order fill delay (ms)
    'simulated_spread_pct': 0.0001,     # 0.01% bid/ask spread

    # Data settings
    'use_real_prices': True             # Use real market prices when available
}


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================
# Position sizing and risk control settings
# NOTE: Different defaults for stocks vs options are handled in Strategy.py
#
# Used by: Strategy.py, Main.py

RISK_CONFIG = {
    # Position sizing
    'max_position_size_pct': 0.02,     # Max 2% of account per trade
    'max_daily_loss_pct': 0.05,        # Stop trading if daily loss > 5%
    'max_concurrent_positions': 5,     # Max open positions

    # Default instrument type ('stock' or 'option')
    'instrument_type': 'stock',

    # Stock-specific defaults (2% stop, 5% target)
    'stock_defaults': {
        'stop_loss_pct': 0.02,
        'profit_target_pct': 0.05,
        'trailing_stop_pct': 0.05,
        'time_stop_minutes': None,     # No time stop for stocks
        'use_trailing_stop': True,
        'use_profit_target': True,
        'use_time_stop': False,
        'use_technical_stop': True,
        'use_expiration_stop': False,
    },

    # Option-specific defaults (35% stop, 100% target)
    'option_defaults': {
        'profit_target_pct': 1.00,
        'use_profit_target': True,
        'use_technical_stop': True,
        'use_expiration_stop': True,
        # Dynamic stop loss configuration (unified settings)
        'dynamic_stop_loss': {
            'enabled': True,
            'stop_loss_pct': 0.35,              # 35% max loss from entry
            'stop_loss_warning_pct': 0.15,      # 15% warning threshold before stop
            'trailing_stop_pct': 0.20,          # 20% trailing below highest price
            'time_stop_minutes': 60,            # 1 hour time stop
            'breakeven_threshold_pct': None,    # None = auto-calculate
            'breakeven_min_minutes': 30,        # Min minutes before breakeven transition
            'trailing_trigger_pct': 0.50,       # Start trailing at 50% profit
        },
    },

    # ATR-based stop calculation
    'atr_multiplier': 2.0,
    'reward_ratio': 2.0,               # Risk:Reward ratio (2:1)
}


# =============================================================================
# CONTRACT LIMITS CONFIGURATION
# =============================================================================
# Prevents accidentally placing oversized option orders
# (e.g., typo in Discord signal: $90 instead of $0.90)
#
# Used by: Signal.py, Strategy.py

CONTRACT_LIMITS_CONFIG = {
    'max_contract_price': 10.00,       # Max price per contract ($10)
    'max_contract_capital': 1000.00,   # Max total capital per order ($1000)
    'enforce_limits': True             # Enable/disable limit checks
}


# =============================================================================
# TECHNICAL ANALYSIS CONFIGURATION
# =============================================================================
# Default parameters for all technical indicators
#
# Used by: Analysis.py, Test.py, Strategy.py

ANALYSIS_CONFIG = {
    # Moving Averages
    'sma_period': 20,
    'ema_fast_period': 12,
    'ema_slow_period': 26,

    # RSI
    'rsi_period': 14,
    'rsi_overbought': 70,              # Stock default (80 for options)
    'rsi_oversold': 30,                # Stock default (20 for options)

    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # Bollinger Bands
    'bb_period': 20,
    'bb_std_dev': 2,

    # ATR
    'atr_period': 14,

    # Stochastic
    'stoch_period': 14,
    'stoch_k_smooth': 3,

    # SuperTrend
    'supertrend_period': 10,
    'supertrend_multiplier': 3.0,

    # Elliott Wave Oscillator
    'ewo_fast': 5,
    'ewo_slow': 35,

    # Relative Strength Index (RSI)
    'rsi_period': 14,                     # RSI calculation period
    'rsi_overbought': 70,                 # RSI overbought threshold (>70 = overbought)
    'rsi_oversold': 30,                   # RSI oversold threshold (<30 = oversold)

    # Volume Analysis
    'vwap_lookback': 30,
    'vpoc_bins': 10,
    'vwap_threshold_pct': 0.02,        # 2% threshold for VWAP exits
    'vpoc_threshold_pct': 0.03,        # 3% threshold for VPOC exits

    # Support/Resistance
    'support_resistance_lookback': 20,

    # Trend
    'trend_fast_period': 10,
    'trend_slow_period': 20,

    # Momentum
    'momentum_period': 10,
    'roc_period': 10,

    # Minimum data requirements
    'min_bars_required': 20,

    # Options pricing parameters (used by Analysis.py options estimator)
    'options': {
        'risk_free_rate': 0.05,        # 5% annual risk-free rate
        'default_volatility': 0.30,    # 30% default implied volatility
        'min_option_price': 0.01,      # Minimum option price floor
    }
}


# =============================================================================
# ENTRY CONFIGURATION
# =============================================================================
# Trade entry signal validation settings
#
# Used by: Strategy.py, Main.py

ENTRY_CONFIG = {
    # Confidence threshold (-100 to 100)
    'min_entry_confidence': 30,

    # Volume confirmation
    'require_volume_confirmation': True,
    'volume_multiplier': 1.2,          # Volume must be 1.2x average

    # Trend confirmation
    'require_trend_confirmation': True,
    'allow_countertrend_entries': False,

    # Price range
    'max_entry_price_range_pct': 0.02  # Entry within 2% of signal price
}


# =============================================================================
# EXIT CONFIGURATION
# =============================================================================
# Exit strategy settings and toggles
#
# Used by: Strategy.py, Test.py

EXIT_CONFIG = {
    # Exit strategy toggles
    # Note: Stop loss settings are managed via dynamic_stop_loss in RISK_CONFIG/BACKTEST_CONFIG
    'use_stop_loss': True,
    'use_profit_target': True,
    'use_trailing_stop': True,
    'use_time_stop': False,            # Disabled for stocks by default
    'use_technical_stop': True,
    'use_expiration_stop': False,      # Not applicable for stocks
    'use_discord_signal': True,        # Exit on Discord exit signal
    'exit_on_opposite_signal': True,

    # Technical indicator exit toggles
    'use_rsi_exit': True,
    'use_macd_exit': True,
    'use_vwap_exit': True,
    'use_vpoc_exit': True,
    'use_supertrend_exit': True,
}


# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Market data fetching and caching settings
#
# Used by: Data.py

DATA_CONFIG = {
    # Cache settings
    'cache_ttl': 60,                   # Cache TTL in seconds
    'max_quote_history': 1000,         # Max quotes in memory
    'purge_old_quotes_minutes': 60,    # Remove quotes older than this

    # Data fetching
    'timeframes': ['1min', '5min', '15min'],
    'bars_to_fetch': 100,              # Initial bars to fetch

    # Pre-signal data window (for analysis before signal)
    'pre_signal_minutes': 60           # 1 hour of data before signal
}


# =============================================================================
# ORDER CONFIGURATION
# =============================================================================
# Order execution settings
#
# Used by: Orders.py, Main.py

ORDER_CONFIG = {
    # Order type
    'order_type': 'market',            # 'market' or 'limit'
    'limit_price_offset_pct': 0.01,    # 1% offset for limit orders

    # Order monitoring
    'check_order_status_interval': 5,  # Check every 5 seconds
    'order_timeout_seconds': 300,      # Cancel after 5 minutes

    # Partial fills
    'allow_partial_fills': True
}


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================
# Settings for backtesting in Test.py
#
# Used by: Test.py

BACKTEST_CONFIG = {
    # Time range
    'lookback_days': 5,

    # Capital
    'initial_capital': 10000.0,
    'position_size_pct': 0.02,         # 2% per trade
    'max_positions': 5,
    'default_contracts': 1,            # Default option contracts

    # Costs
    'slippage_pct': 0.001,             # 0.1% slippage
    'commission_per_trade': 1.0,       # $1 per trade
    'commission_per_contract': 0.65,   # Per contract for options

    # Data settings
    'candle_interval': '1m',           # 1-minute candles
    'use_simulated_fills': True,

    # Instrument type (determines risk defaults)
    'instrument_type': 'option',

    # Dynamic Stop Loss Settings
    # Unified stop loss configuration with three phases:
    # Phase 1 (INITIAL): Fixed stop at entry * (1 - stop_loss_pct)
    # Phase 2 (BREAKEVEN): Move stop to entry when price >= breakeven_threshold
    #                      Only after breakeven_min_minutes has passed
    # Phase 3 (TRAILING): Trail at trailing_stop_pct below highest price since entry
    'dynamic_stop_loss': {
        'enabled': True,
        'stop_loss_pct': 0.35,              # 35% max loss from entry
        'stop_loss_warning_pct': 0.15,      # 15% warning threshold before stop
        'profit_target_pct': 1.00,          # 100% profit target
        'trailing_stop_pct': 0.20,          # 20% trailing below highest price
        'time_stop_minutes': 60,            # 1 hour time stop
        'breakeven_threshold_pct': None,    # None = auto-calculate as entry/(1-stop_loss_pct)
        'breakeven_min_minutes': 30,        # Minimum minutes before allowing breakeven transition
        'trailing_trigger_pct': 0.50,       # Start trailing at 50% profit
    },

    # Tiered Profit Exit Settings
    # Dynamic profit target system based on profit percentage
    # Only active when stock price > EMA 30
    # Tiers define when profit targets are set and what they are
    'tiered_profit_exit': {
        'enabled': True,
        # Note: Uses dynamic_stop_loss.stop_loss_pct for trailing calculation

        # Contract tiers (for future multi-contract support)
        # Tier 1: 1 contract - sell all at exit signal
        # Tier 2: 3 contracts - sell 1 at each profit tier
        # Tier 3: 5+ contracts - sell progressively at profit tiers
        'tier_1_contracts': 1,
        'tier_2_contracts': 3,
        'tier_3_contracts': 5,

        # Profit tiers: {profit_pct_threshold: target_above_entry_pct}
        # When profit reaches threshold, set target to % above entry
        'profit_tiers': {
            35: 0.10,   # 35% profit -> target = 10% above entry
            50: 0.20,   # 50% profit -> target = 20% above entry
            75: 0.35,   # 75% profit -> target = 35% above entry
            100: 0.50,  # 100% profit -> target = 50% above entry, start trailing
            125: None,  # 125% profit -> trailing = (1-stop_loss) * max_price
            200: None,  # 200% profit -> immediate sell
        },

        # Trailing mode settings (activates at 100%+ profit)
        'trailing_start_pct': 100,          # Start trailing at 100% profit
        'trailing_floor_pct': 0.50,         # Minimum target = 50% above entry
        'immediate_sell_pct': 200,          # Sell immediately at 200% profit
    },

    # TEST Exit Strategy - Profit Trailing Stop
    # Captures profits by trailing the maximum profit percentage
    # Exits when profit drops from peak by a threshold amount
    'test_peak_exit': {
        'enabled': True,                    # Enable TEST profit trailing

        # Minimum Profit to Activate
        # Only start trailing after this profit is reached (0.50 = 50%)
        'min_profit_pct': 0.50,

        # Pullback Threshold
        # Exit when profit drops this much from peak (0.15 = 15%)
        # Example: Peak at 126%, exit when drops to 111% (126 - 15 = 111)
        'pullback_pct': 0.15,

        # RSI Overbought Level
        # RSI value considered overbought (default: 70)
        'rsi_overbought': 70,

        # RSI Pullback Threshold
        # Use tighter pullback when RSI was overbought (0.10 = 10%)
        # This allows faster exits when momentum is exhausted
        'rsi_pullback_pct': 0.10,
    },

    # Technical Indicators for Backtest
    'indicators': {
        'ema_period': 30,                   # 30-bar EMA (30 minutes on 1m data)
        'vwap_enabled': True,               # Calculate VWAP
        'rsi_period': 14,                   # RSI calculation period
        'rsi_overbought': 70,               # RSI overbought threshold
        'rsi_oversold': 30,                 # RSI oversold threshold
    },

    # Exit strategy toggles for backtest
    # Only Stop Loss and Profit Target are enabled by default
    'exit_strategies': {
        # Stop Loss conditions (Priority 1-1.5)
        'use_stop_loss': True,
        'use_sl_ema': False,            # SL_EMA: EMA30 > VWAP AND price < EMA30 (C2)
        # Profit taking (Priority 2-3)
        'use_profit_target': True,
        'use_trailing_stop': False,
        # Experimental exits
        'use_test_peak_exit': False,    # TEST: EWO-based peak detection (experimental)
        # Time-based exits (Priority 4-8)
        'use_time_stop': False,
        'use_expiration_stop': False,
        'use_discord_signal': False,
        'use_end_of_day': False,        # Day trading EOD exit
        'use_max_hold_days': False,     # Swing trading max hold
        # Technical indicator exits (Priority 10-14)
        'use_rsi': False,
        'use_macd': False,
        'use_vwap': False,
        'use_vpoc': False,
        'use_supertrend': False
    },

    # Indicator settings for backtest
    'indicator_settings': {
        'rsi_period': 14,
        'rsi_overbought': 80,          # Higher threshold for options
        'rsi_oversold': 20,            # Lower threshold for options
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'vwap_threshold_pct': 0.02,
        'vpoc_threshold_pct': 0.03,
        'supertrend_period': 10,
        'supertrend_multiplier': 3.0
    },

    # Swing trading settings
    'swing_trade': {
        'enabled': False,
        'allow_overnight_holds': False,
        'max_hold_days': 5,
        'use_end_of_day_exit': False,
        'eod_exit_time': '15:45'
    },

    # Export settings
    'auto_export_matrices': False,
    'export_matrices_dir': 'Data_BT'
}


# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
# Logging, performance tracking, and main loop settings
#
# Used by: Main.py, Manager.py

SYSTEM_CONFIG = {
    # Performance tracking
    'enable_performance_tracking': True,

    # Logging
    'log_level': 'INFO',               # DEBUG, INFO, WARNING, ERROR
    'max_log_file_size_mb': 100,
    'keep_old_logs_days': 30,
    'enable_position_logging': True,
    'enable_trade_logging': True,
    'position_log_cleanup_days': 30,

    # Main loop timing
    'cycle_sleep_ms': 1000,            # Sleep between cycles
    'min_cycle_time_ms': 500           # Minimum cycle duration
}


# =============================================================================
# ALERT CONFIGURATION
# =============================================================================
# Notification and alert settings
#
# Used by: Main.py, Manager.py

ALERT_CONFIG = {
    # Alert toggles
    'enable_position_alerts': True,
    'enable_loss_alerts': True,
    'enable_profit_alerts': True,

    # Webhook
    'alert_webhook_url': os.getenv('ALERT_WEBHOOK_URL', ''),

    # Thresholds
    'large_loss_threshold': 100,       # Alert if loss > $100
    'large_profit_threshold': 500      # Alert if profit > $500
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config(section):
    """
    Get configuration section by name.

    Args:
        section: Section name (e.g., 'discord', 'risk', 'analysis')

    Returns:
        Dictionary with configuration parameters
    """
    configs = {
        'credentials': CREDENTIALS_CONFIG,
        'discord': DISCORD_CONFIG,
        'market': MARKET_CONFIG,
        'paper': PAPER_CONFIG,
        'risk': RISK_CONFIG,
        'contract_limits': CONTRACT_LIMITS_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'entry': ENTRY_CONFIG,
        'exit': EXIT_CONFIG,
        'data': DATA_CONFIG,
        'order': ORDER_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'system': SYSTEM_CONFIG,
        'alert': ALERT_CONFIG
    }

    return configs.get(section.lower(), {})


def get_setting(section, key, default=None):
    """
    Get individual configuration setting.

    Args:
        section: Section name
        key: Setting key
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    config = get_config(section)
    return config.get(key, default)


def get_risk_defaults(instrument_type='stock'):
    """
    Get risk management defaults for instrument type.

    Args:
        instrument_type: 'stock' or 'option'

    Returns:
        Dictionary with risk parameters
    """
    if instrument_type == 'option':
        return RISK_CONFIG['option_defaults'].copy()
    return RISK_CONFIG['stock_defaults'].copy()


def load_config_json(config_path='config.json'):
    """
    Load configuration overrides from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Dictionary with configuration overrides
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
            return {}
    return {}


# Load any config.json overrides
CONFIG_JSON = load_config_json('config.json')

# Apply contract limits from config.json if present
if 'contract_limits' in CONFIG_JSON:
    CONTRACT_LIMITS_CONFIG.update(CONFIG_JSON['contract_limits'])


def validate_config():
    """
    Validate all configuration settings.

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []

    # Validate trading mode
    if TRADING_MODE not in ['paper', 'live']:
        errors.append("TRADING_MODE must be 'paper' or 'live'")

    # Validate credentials (warnings only)
    if not DISCORD_CONFIG.get('token'):
        warnings.append("DISCORD_TOKEN not set (required for live trading)")
    if not DISCORD_CONFIG.get('channel_id'):
        warnings.append("DISCORD_CHANNEL_ID not set (required for live trading)")

    if TRADING_MODE == 'live' and not CREDENTIALS_CONFIG.get('email'):
        errors.append("Webull credentials required for live trading")
    elif not CREDENTIALS_CONFIG.get('email'):
        warnings.append("Webull credentials not configured")

    # Validate risk settings
    if not 0 < RISK_CONFIG['max_position_size_pct'] <= 1:
        errors.append("max_position_size_pct must be between 0 and 1")
    if RISK_CONFIG['max_concurrent_positions'] < 1:
        errors.append("max_concurrent_positions must be at least 1")

    # Validate analysis settings
    if ANALYSIS_CONFIG['min_bars_required'] < 1:
        errors.append("min_bars_required must be at least 1")

    # Validate entry settings
    if not -100 <= ENTRY_CONFIG['min_entry_confidence'] <= 100:
        errors.append("min_entry_confidence must be between -100 and 100")

    # Validate contract limits
    if CONTRACT_LIMITS_CONFIG.get('max_contract_price', 0) <= 0:
        errors.append("max_contract_price must be greater than 0")
    if CONTRACT_LIMITS_CONFIG.get('max_contract_capital', 0) <= 0:
        errors.append("max_contract_capital must be greater than 0")

    return errors, warnings


def validate_and_report():
    """
    Validate all settings and report any issues.

    Returns:
        Boolean (True if valid, False if errors)
    """
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
        print("CONFIGURATION ERRORS DETECTED:")
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


def print_config_summary():
    """Print a summary of current configuration to console."""
    print("\n" + "=" * 60)
    print("TRADING BOT CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"\nTRADING MODE: {TRADING_MODE.upper()}")
    if TRADING_MODE == 'paper':
        print(f"  Initial Balance: ${PAPER_CONFIG['initial_balance']:,.2f}")
        print(f"  State File: {PAPER_CONFIG['state_file']}")

    print("\nMARKET SETTINGS:")
    print(f"  Trading Hours: {MARKET_CONFIG['market_hours_start']:.1f} - {MARKET_CONFIG['market_hours_end']:.1f}")
    print(f"  Pre-market: {MARKET_CONFIG['enable_premarket']}")
    print(f"  After-hours: {MARKET_CONFIG['enable_afterhours']}")

    print("\nRISK MANAGEMENT:")
    print(f"  Max Position Size: {RISK_CONFIG['max_position_size_pct']*100:.1f}%")
    print(f"  Max Daily Loss: {RISK_CONFIG['max_daily_loss_pct']*100:.1f}%")
    print(f"  Max Concurrent Positions: {RISK_CONFIG['max_concurrent_positions']}")

    print("\nTECHNICAL ANALYSIS:")
    print(f"  Min Bars Required: {ANALYSIS_CONFIG['min_bars_required']}")
    print(f"  RSI Period: {ANALYSIS_CONFIG['rsi_period']}")
    print(f"  RSI Levels: {ANALYSIS_CONFIG['rsi_oversold']} - {ANALYSIS_CONFIG['rsi_overbought']}")

    print("\nENTRY SETTINGS:")
    print(f"  Min Confidence: {ENTRY_CONFIG['min_entry_confidence']}")
    print(f"  Volume Confirmation: {ENTRY_CONFIG['require_volume_confirmation']}")

    print("\nCONTRACT LIMITS:")
    print(f"  Max Contract Price: ${CONTRACT_LIMITS_CONFIG['max_contract_price']:.2f}")
    print(f"  Max Contract Capital: ${CONTRACT_LIMITS_CONFIG['max_contract_capital']:.2f}")
    print(f"  Enforce Limits: {CONTRACT_LIMITS_CONFIG['enforce_limits']}")

    print("\nSYSTEM:")
    print(f"  Cycle Sleep: {SYSTEM_CONFIG['cycle_sleep_ms']}ms")
    print(f"  Log Level: {SYSTEM_CONFIG['log_level']}")

    print("\n" + "=" * 60 + "\n")
