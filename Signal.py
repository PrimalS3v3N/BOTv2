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
import base64
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

# Base prompts (custom_instructions support commented out for now — see _build_system_prompt)
_AI_SYSTEM_PROMPT_BASE = (
    "You classify Discord messages that reply to stock/options trading alerts. "
    "Determine if the reply indicates the trader is exiting or trimming their position. "
    "Respond with ONLY one word: EXIT, TRIM, or NONE."
)

_AI_USER_TEMPLATE = (
    "EXIT = full sell/close of position\n"
    "TRIM = partial sell (trimming, scaling out, took some off)\n"
    "NONE = not a sell action\n\n"
    "Message: \"{message}\""
)


def _build_system_prompt(ai_config):
    """Build system prompt, appending custom_instructions if configured."""
    prompt = _AI_SYSTEM_PROMPT_BASE
    # Uncomment to enable custom_instructions from Config:
    # custom = ai_config.get('custom_instructions', '').strip()
    # if custom:
    #     prompt += f"\n\nAdditional instructions:\n{custom}"
    return prompt


def _classify_with_ai(message):
    """
    Use an AI model to classify whether a Discord reply indicates an exit or trim.

    Supports three providers (configured in BACKTEST_CONFIG.discord_exit_signals.ai):
      - 'ollama'    : Local model via Ollama HTTP API (~10-50ms, no API key needed)
      - 'anthropic' : Claude Haiku via Anthropic SDK (~200-500ms)
      - 'openai'    : GPT-4o-mini via OpenAI SDK (~200-500ms)

    Returns:
        dict {'type': 'exit'|'trim', 'keyword': 'ai'} or None
    """
    exit_config = Config.BACKTEST_CONFIG.get('discord_exit_signals', {})
    ai_config = exit_config.get('ai', {})
    provider = ai_config.get('provider', 'ollama')

    # Check cache first
    cache_key = message.strip().lower()
    if cache_key in _ai_classify_cache:
        return _ai_classify_cache[cache_key]

    user_prompt = _AI_USER_TEMPLATE.format(message=message)

    result = None
    try:
        if provider == 'ollama':
            result = _classify_ollama(user_prompt, ai_config)
        elif provider == 'anthropic':
            result = _classify_anthropic(user_prompt, ai_config)
        elif provider == 'openai':
            result = _classify_openai(user_prompt, ai_config)
    except Exception as e:
        print(f"    AI classification failed ({provider}): {e}")

    _ai_classify_cache[cache_key] = result
    return result


def _classify_ollama(user_prompt, ai_config):
    """
    Classify using a local Ollama model.

    Ollama runs locally — no API key, no network latency, no cost.
    Requires: `ollama pull <model>` once to download the model.

    Recommended models (sorted by speed):
      - gemma3:1b     (~10-20ms, 1B params, very fast)
      - phi4-mini     (~20-40ms, 3.8B params, good accuracy)
      - llama3.2:3b   (~20-40ms, 3B params, solid balance)
      - gemma3:4b     (~30-50ms, 4B params, best accuracy at this tier)
    """
    import requests

    base_url = ai_config.get('ollama_url', 'http://localhost:11434')
    model = ai_config.get('model', 'gemma3:1b')
    timeout = ai_config.get('timeout', 10)

    system_prompt = _build_system_prompt(ai_config)

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": 10,       # Only need 1 word back
                "temperature": 0.0,      # Deterministic classification
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()
    answer = response.json().get('message', {}).get('content', '').strip().upper()
    return _parse_ai_answer(answer)


def _classify_anthropic(user_prompt, ai_config):
    """Classify using Anthropic Claude API."""
    import anthropic

    api_key = ai_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY', '')
    model = ai_config.get('model', 'claude-haiku-4-5-20251001')

    system_prompt = _build_system_prompt(ai_config)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=10,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = response.content[0].text.strip().upper()
    return _parse_ai_answer(answer)


def _classify_openai(user_prompt, ai_config):
    """Classify using OpenAI API."""
    import openai

    api_key = ai_config.get('api_key') or os.getenv('OPENAI_API_KEY', '')
    model = ai_config.get('model', 'gpt-4o-mini')

    system_prompt = _build_system_prompt(ai_config)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
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
# INTERNAL - Vision (Image) Classifier
# =============================================================================

# Prompt for image-based classification
_AI_VISION_PROMPT = (
    "This image was posted as a reply to a stock/options trading alert on Discord. "
    "It may show a P&L screenshot, a brokerage confirmation, a chart, or other trading content.\n\n"
    "Determine if this image indicates the trader has:\n"
    "- EXIT — fully sold/closed their position (e.g. P&L screenshot showing closed trade, "
    "'sold' confirmation, zero position)\n"
    "- TRIM — partially sold (e.g. reduced shares/contracts, partial fill)\n"
    "- NONE — not a sell action (e.g. chart analysis, watchlist, meme, unrelated)\n\n"
    "Respond with ONLY one word: EXIT, TRIM, or NONE."
)

# Cache for downloaded images (URL -> base64 bytes)
_image_download_cache = {}


def _download_image_as_base64(url, timeout=10):
    """Download an image from URL and return as base64 string."""
    if url in _image_download_cache:
        return _image_download_cache[url]

    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get('content-type', 'image/png')
    b64 = base64.b64encode(response.content).decode('utf-8')

    result = {'base64': b64, 'media_type': content_type}
    _image_download_cache[url] = result
    return result


def _classify_vision_with_ai(image_urls, text_content=None):
    """
    Classify image attachments using a vision-capable AI model.

    Downloads the first image, sends it to the model with text context
    (if any), and returns EXIT/TRIM/NONE classification.

    Args:
        image_urls: List of image URLs from Discord attachments
        text_content: Optional text content accompanying the image

    Returns:
        dict {'type': 'exit'|'trim', 'keyword': 'ai-vision'} or None
    """
    if not image_urls:
        return None

    exit_config = Config.BACKTEST_CONFIG.get('discord_exit_signals', {})
    ai_config = exit_config.get('ai', {})
    provider = ai_config.get('provider', 'ollama')
    timeout = ai_config.get('timeout', 30)

    # Cache key: first image URL + text
    cache_key = f"vision:{image_urls[0]}:{(text_content or '').strip().lower()}"
    if cache_key in _ai_classify_cache:
        return _ai_classify_cache[cache_key]

    # Download the first image
    try:
        img_data = _download_image_as_base64(image_urls[0], timeout=timeout)
    except Exception as e:
        print(f"    Image download failed: {e}")
        return None

    # Build prompt with optional text context
    prompt = _AI_VISION_PROMPT
    # Uncomment to enable custom_instructions from Config:
    # custom = ai_config.get('custom_instructions', '').strip()
    # if custom:
    #     prompt += f"\n\nAdditional instructions:\n{custom}"
    if text_content and text_content.strip():
        prompt += f"\n\nAccompanying text: \"{text_content.strip()}\""

    result = None
    try:
        if provider == 'ollama':
            result = _classify_vision_ollama(prompt, img_data, ai_config)
        elif provider == 'anthropic':
            result = _classify_vision_anthropic(prompt, img_data, ai_config)
        elif provider == 'openai':
            result = _classify_vision_openai(prompt, img_data, ai_config)
    except Exception as e:
        print(f"    Vision classification failed ({provider}): {e}")

    _ai_classify_cache[cache_key] = result
    return result


def _classify_vision_ollama(prompt, img_data, ai_config):
    """Classify image using Ollama vision model (gemma3, llava, etc.)."""
    import requests

    base_url = ai_config.get('ollama_url', 'http://localhost:11434')
    model = ai_config.get('vision_model') or ai_config.get('model', 'gemma3:4b')
    timeout = ai_config.get('timeout', 30)

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [img_data['base64']]},
            ],
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.0,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()
    answer = response.json().get('message', {}).get('content', '').strip().upper()
    result = _parse_ai_answer(answer)
    if result:
        result['keyword'] = 'ai-vision'
    return result


def _classify_vision_anthropic(prompt, img_data, ai_config):
    """Classify image using Anthropic Claude vision API."""
    import anthropic

    api_key = ai_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY', '')
    model = ai_config.get('vision_model') or ai_config.get('model', 'claude-haiku-4-5-20251001')

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_data['media_type'],
                        "data": img_data['base64'],
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    answer = response.content[0].text.strip().upper()
    result = _parse_ai_answer(answer)
    if result:
        result['keyword'] = 'ai-vision'
    return result


def _classify_vision_openai(prompt, img_data, ai_config):
    """Classify image using OpenAI vision API."""
    import openai

    api_key = ai_config.get('api_key') or os.getenv('OPENAI_API_KEY', '')
    model = ai_config.get('vision_model') or ai_config.get('model', 'gpt-4o-mini')

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_data['media_type']};base64,{img_data['base64']}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    answer = response.choices[0].message.content.strip().upper()
    result = _parse_ai_answer(answer)
    if result:
        result['keyword'] = 'ai-vision'
    return result


# =============================================================================
# EXTERNAL - Exit Signal Parsing
# =============================================================================
# Used by: Test.py

def ParseExitSignal(message, image_urls=None):
    """
    Parse a Discord reply message for exit signal keywords.

    Three-stage approach:
      1. Fast keyword matching (no API calls, instant)
      2. AI text classification fallback (if keywords miss)
      3. AI vision classification (if message has image attachments)

    Args:
        message:    Raw reply message string from Discord
        image_urls: List of image attachment URLs (or None)

    Returns:
        dict with keys: type ('exit', 'trim'), keyword (matched word, 'ai', or 'ai-vision')
        or None if no exit signal detected
    """
    exit_config = Config.BACKTEST_CONFIG.get('discord_exit_signals', {})
    if not exit_config.get('enabled', True):
        return None

    has_text = bool(message and message.strip())
    has_images = bool(image_urls)

    if not has_text and not has_images:
        return None

    # Stage 1: Fast keyword matching (text only)
    if has_text:
        exit_keywords = exit_config.get('exit_keywords', [])
        trim_keywords = exit_config.get('trim_keywords', [])
        negation_words = exit_config.get('negation_words', [])
        scan_max = exit_config.get('scan_max_words', 0)

        text = message.lower().strip()
        words = text.split()

        if scan_max > 0:
            words = words[:scan_max]
            text = ' '.join(words)

        # Check for negation
        negated = False
        for neg in negation_words:
            if neg.lower() in words:
                negated = True
                break

        if not negated:
            for kw in exit_keywords:
                if kw.lower() in text:
                    return {'type': 'exit', 'keyword': kw}

            for kw in trim_keywords:
                if kw.lower() in text:
                    return {'type': 'trim', 'keyword': kw}

    # Stage 2 & 3: AI classification (if enabled)
    ai_config = exit_config.get('ai', {})
    if ai_config.get('enabled', False):
        # Stage 2: AI text classification (text-only messages)
        if has_text and not has_images:
            ai_result = _classify_with_ai(message)
            if ai_result:
                return ai_result

        # Stage 3: AI vision classification (messages with images)
        if has_images:
            vision_result = _classify_vision_with_ai(
                image_urls, text_content=message
            )
            if vision_result:
                return vision_result

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
