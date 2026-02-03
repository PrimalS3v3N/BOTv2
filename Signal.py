"""
Signal.py - Discord Trading Signal Parser

Module Goal: Parse Discord trading alert messages into structured order data.
Extracts ticker, strike price, option type, expiration, and contract cost.

================================================================================
INTERNAL - Signal Parsing Logic
================================================================================
"""

import re
import datetime as dt
from datetime import timedelta, date
import pandas as pd
import robin_stocks.robinhood as RH


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Test.py
"""

import Config

# Discord alert marker from Config
ALERT_MARKER = Config.DISCORD_CONFIG.get('alert_marker', '<a:RedAlert:759583962237763595>')


# =============================================================================
# INTERNAL - Utility Functions
# =============================================================================

def is_number(n):
    """Check if value is a valid number."""
    try:
        if n is None:
            return False
        num = float(n)
        return num == num  # checks for nan
    except (ValueError, TypeError):
        return False


def validate_ticker(ticker):
    """Validate ticker by checking if it exists via RH quotes."""
    try:
        ticker_clean = re.findall(r'[A-Za-z]+', ticker)
        if not ticker_clean:
            return None
        ticker_sym = ticker_clean[0].upper()
        quote = RH.get_quotes(ticker_sym)
        if quote and quote[0] is not None:
            return ticker_sym
    except Exception:
        pass
    return None


# =============================================================================
# INTERNAL - Expiration Parsing
# =============================================================================

def parse_expiration(tokens, start_idx):
    """Parse expiration date from tokens. Returns (expiration_str, success)."""
    today = dt.datetime.now()
    weekday = int(today.strftime("%u"))  # 1=Mon, 7=Sun

    for i in range(start_idx, len(tokens)):
        word = tokens[i].upper()

        # Skip common prefixes
        if word in ("EXPIRATION", "EXP", "CALL", "PUT", "CALLS", "PUTS"):
            continue

        # Today/0DTE
        if word in ("TODAY", "0DTE"):
            return today.strftime("%m %d %y"), True

        # Tomorrow/1DTE
        if word in ("TOMORROW", "1DTE"):
            exp = today + timedelta(days=1)
            return exp.strftime("%m %d %y"), True

        # This week/Friday
        if word in ("THIS", "WEEK", "FRIDAY"):
            days_to_friday = (5 - weekday) % 7
            if days_to_friday == 0 and word != "THIS":
                days_to_friday = 0  # Already Friday
            exp = date.today() + timedelta(days=max(0, days_to_friday))
            return exp.strftime("%m %d %y"), True

        # Next week/Friday
        if word == "NEXT":
            days_to_next_friday = (5 - weekday) + 7
            if days_to_next_friday > 7:
                days_to_next_friday = days_to_next_friday % 7 + 7
            exp = date.today() + timedelta(days=days_to_next_friday)
            return exp.strftime("%m %d %y"), True

        # Date format: 10/5, 10-5, etc.
        parts = re.findall(r'\d+', tokens[i])
        if len(parts) >= 2:
            month, day = int(parts[0]), int(parts[1])
            if 1 <= month <= 12 and 1 <= day <= 31:
                year = today.year if month >= today.month else today.year + 1
                return f"{month:02d} {day:02d} {str(year)[2:]}", True

    return None, False


# =============================================================================
# EXTERNAL - Order Building (Main Interface)
# =============================================================================
# Used by: Test.py

def BuildOrder(message):
    """
    Parse a Discord trading signal message.

    Args:
        message: Raw message string from Discord

    Returns:
        dict with keys: Ticker, Strike, Option, Expiration, Cost
        or None if not a valid signal
    """
    if not message or ALERT_MARKER not in message:
        return None

    # Split message, handle newlines
    first_line = message.split("\n")[0]
    tokens = first_line.split()

    if not tokens or tokens[0] != ALERT_MARKER:
        return None

    if len(tokens) < 2:
        return None

    order = {
        "Ticker": None,
        "Strike": None,
        "Option": None,
        "Expiration": None,
        "Cost": None
    }

    # 1. TICKER - next word after alert marker
    ticker = validate_ticker(tokens[1])
    if not ticker:
        return None
    order["Ticker"] = ticker

    idx = 2  # Start parsing after ticker

    # 2. STRIKE PRICE
    for i in range(idx, len(tokens)):
        parts = re.findall(r'[\d.]+', tokens[i])
        if parts and is_number(parts[0]):
            strike = parts[0]
            if len(parts) > 1:
                strike = f"{parts[0]}.{parts[1]}"
            order["Strike"] = strike
            idx = i + 1
            break

    if not order["Strike"]:
        return None

    # 3. OPTION TYPE (CALL/PUT)
    for i in range(idx, len(tokens)):
        word = tokens[i].upper()
        word_clean = re.findall(r'[A-Za-z]+', word)
        if word_clean:
            w = word_clean[0]
            if w in ("CALL", "CALLS", "C"):
                order["Option"] = "CALL"
                idx = i + 1
                break
            elif w in ("PUT", "PUTS", "P"):
                order["Option"] = "PUT"
                idx = i + 1
                break

    if not order["Option"]:
        return None

    # 4. EXPIRATION
    exp, found = parse_expiration(tokens, idx)
    if found:
        order["Expiration"] = exp
        idx += 1

    # 5. CONTRACT PRICE/COST
    for i in range(idx, len(tokens)):
        token = tokens[i]
        if token.startswith('$'):
            token = token[1:]
        token = token.rstrip(',.;:!?')
        if is_number(token) or (token.startswith('.') and is_number(token)):
            if '.' not in token:
                price_value = float(token) / 100.0
                order["Cost"] = str(price_value)
            else:
                order["Cost"] = token
            break

    return order


def BuildOrderFromLog(log_df):
    """
    Build order from a Discord log DataFrame.

    Args:
        log_df: DataFrame with 'Message' column

    Returns:
        dict order or None
    """
    if log_df is None or len(log_df) == 0:
        return None

    last_msg = log_df.iloc[len(log_df) - 1]['Message']
    return BuildOrder(last_msg)
