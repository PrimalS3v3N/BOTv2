"""
Config.py - Centralized Configuration

Module Goal: Store all configurable variables for the trading bot system.

================================================================================
INTERNAL - Configuration Storage
================================================================================
"""

# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================
DISCORD_CONFIG = {
    'token': '',                    # Discord bot token
    'channel_id': '',               # Signal channel ID
    'api_version': 'v10',           # Discord API version
    'base_url': 'https://discord.com/api',
    'alert_marker': '<a:RedAlert:759583962237763595>',
    'test_message_limit': 100,
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_CONFIG = {
    'pre_signal_minutes': 60,       # Minutes of data before signal for warmup
}

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================
BACKTEST_CONFIG = {
    'lookback_days': 5,
    'initial_capital': 10000.0,
    'position_size_pct': 0.02,
    'default_contracts': 1,
    'slippage_pct': 0.001,
    'commission_per_contract': 0.65,
    'instrument_type': 'option',
}


"""
================================================================================
EXTERNAL - Module Interface Functions
================================================================================
Modules: Signal.py, Test.py, Data.py, Analysis.py, Strategy.py, Orders.py
"""

def get_config(section):
    """
    Get configuration section by name.

    Args:
        section: Config section name ('discord', 'data', 'backtest')

    Returns:
        Dictionary of configuration values
    """
    configs = {
        'discord': DISCORD_CONFIG,
        'data': DATA_CONFIG,
        'backtest': BACKTEST_CONFIG,
    }
    return configs.get(section, {})
