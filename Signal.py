"""
Signal.py - Discord Trading Signal Parser

Module Goal: Parse Discord trading alert messages into structured order data.
Extracts ticker, strike price, option type, expiration, and contract cost.

================================================================================
INTERNAL - Signal Parsing Logic
================================================================================
"""

import os
import re
import json
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
    """Validate ticker by checking if it exists via RH quotes or is a known index."""
    try:
        # Match tickers with optional dot-suffix (e.g. BRK.B, BRK.A)
        ticker_match = re.match(r'[A-Za-z]+(?:\.[A-Za-z]+)?', ticker)
        if not ticker_match:
            return None
        ticker_sym = ticker_match.group(0).upper()
        # Index symbols (SPX, NDX, etc.) are valid option underlyings but
        # not tradeable stocks — Robinhood's /quotes/ endpoint returns 404.
        if ticker_sym in Config.INDEX_SYMBOLS:
            return ticker_sym
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


# =============================================================================
# INTERNAL - AI Message Classifier
# =============================================================================

# Cache for AI classification results to avoid re-classifying identical messages
_ai_classify_cache = {}


def _classify_with_ai(message):
    """
    Use an AI model to classify whether a Discord reply indicates an exit or trim.

    Supports two providers (configured in BACKTEST_CONFIG.discord_exit_signals.ai):
      - 'anthropic' : Uses Claude (claude-haiku via Anthropic SDK)
      - 'openai'    : Uses GPT (gpt-4o-mini via OpenAI SDK)

    Returns:
        dict {'type': 'exit'|'trim', 'keyword': 'ai'} or None
    """
    exit_config = Config.BACKTEST_CONFIG.get('discord_exit_signals', {})
    ai_config = exit_config.get('ai', {})
    provider = ai_config.get('provider', 'anthropic')

    # Check cache first
    cache_key = message.strip().lower()
    if cache_key in _ai_classify_cache:
        return _ai_classify_cache[cache_key]

    prompt = (
        "You are classifying a Discord message that was posted as a reply to a "
        "stock/options trading alert. Determine if the reply indicates the trader "
        "is exiting (selling) or trimming (partially selling) their position.\n\n"
        "Respond with ONLY one word:\n"
        "- EXIT  — if the message indicates a full sell/close of the position\n"
        "- TRIM  — if the message indicates a partial sell (trimming, scaling out)\n"
        "- NONE  — if the message does NOT indicate any selling action\n\n"
        f"Message: \"{message}\""
    )

    result = None
    try:
        if provider == 'anthropic':
            result = _classify_anthropic(prompt, ai_config)
        elif provider == 'openai':
            result = _classify_openai(prompt, ai_config)
    except Exception as e:
        print(f"    AI classification failed ({provider}): {e}")

    _ai_classify_cache[cache_key] = result
    return result


def _classify_anthropic(prompt, ai_config):
    """Classify using Anthropic Claude API."""
    import anthropic

    api_key = ai_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY', '')
    model = ai_config.get('model', 'claude-haiku-4-5-20251001')

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip().upper()
    return _parse_ai_answer(answer)


def _classify_openai(prompt, ai_config):
    """Classify using OpenAI API."""
    import openai

    api_key = ai_config.get('api_key') or os.getenv('OPENAI_API_KEY', '')
    model = ai_config.get('model', 'gpt-4o-mini')

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content.strip().upper()
    return _parse_ai_answer(answer)


def _parse_ai_answer(answer):
    """Convert AI model's one-word answer to result dict."""
    if 'EXIT' in answer:
        return {'type': 'exit', 'keyword': 'ai'}
    elif 'TRIM' in answer:
        return {'type': 'trim', 'keyword': 'ai'}
    return None


# =============================================================================
# EXTERNAL - Exit Signal Parsing
# =============================================================================
# Used by: Test.py

def ParseExitSignal(message):
    """
    Parse a Discord reply message for exit signal keywords.

    Two-stage approach:
      1. Fast keyword matching (no API calls, instant)
      2. AI classification fallback (optional, if keywords miss)

    Args:
        message: Raw reply message string from Discord

    Returns:
        dict with keys: type ('exit', 'trim'), keyword (matched word or 'ai')
        or None if no exit signal detected
    """
    if not message:
        return None

    exit_config = Config.BACKTEST_CONFIG.get('discord_exit_signals', {})
    if not exit_config.get('enabled', True):
        return None

    exit_keywords = exit_config.get('exit_keywords', [])
    trim_keywords = exit_config.get('trim_keywords', [])
    negation_words = exit_config.get('negation_words', [])
    scan_max = exit_config.get('scan_max_words', 0)

    # Normalize: lowercase, collapse whitespace
    text = message.lower().strip()
    words = text.split()

    if scan_max > 0:
        words = words[:scan_max]
        text = ' '.join(words)

    # Check for negation — if any negation word appears before or near an
    # exit keyword, skip this message.  Simple heuristic: if ANY negation
    # word is present anywhere in the scanned text, treat it as not an exit.
    for neg in negation_words:
        if neg.lower() in words:
            return None

    # Stage 1: Fast keyword matching
    for kw in exit_keywords:
        if kw.lower() in text:
            return {'type': 'exit', 'keyword': kw}

    for kw in trim_keywords:
        if kw.lower() in text:
            return {'type': 'trim', 'keyword': kw}

    # Stage 2: AI classification fallback (if enabled)
    ai_config = exit_config.get('ai', {})
    if ai_config.get('enabled', False):
        ai_result = _classify_with_ai(message)
        if ai_result:
            return ai_result

    return None


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
