"""
AIModel.py - Local AI Model for Exit Signal Generation

Module Goal: Run a local LLM (via llama-cpp-python) to provide AI-driven exit
signals during backtesting. The model analyzes multi-timeframe technical data
and returns structured opinions on market outlook and hold/sell recommendations.

Hardware: Designed for NVIDIA 3080Ti (12GB) / 4090 (24GB) GPUs.
Model: Any GGUF-format model compatible with llama-cpp-python.
       Recommended: Mistral-7B-Instruct Q4_K_M (~4.4GB VRAM)

================================================================================
INTERNAL - AI Inference Engine
================================================================================
"""

import json
import os
import datetime as dt
import hashlib
import uuid
import numpy as np

import Config


# Generate a unique run ID per backtest session (set once at import time,
# refreshed each time a logger is instantiated via Backtest.__init__).
_RUN_ID = None


def _new_run_id():
    """Generate a short unique run ID for this backtest session."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# INTERNAL - Data Snapshot Builder
# =============================================================================

class DataSnapshot:
    """
    Builds a structured data snapshot from the simulation loop state
    for feeding to the local AI model.

    Aggregates 1-minute bar data into multi-timeframe views:
    - 1min:  Current bar
    - 5min:  Last 5 bars aggregated
    - 30min: Last 30 bars aggregated
    - 1hr:   Last 60 bars aggregated
    """

    def __init__(self, max_history=60):
        """
        Args:
            max_history: Number of bars to retain for multi-timeframe analysis.
                         60 bars = 1 hour of 1-min data.
        """
        self.max_history = max_history
        self.bars = []

    def add_bar(self, bar_data):
        """
        Record a bar snapshot from the simulation loop.

        Args:
            bar_data: dict with keys matching the databook record format:
                stock_price, stock_high, stock_low, true_price, volume,
                option_price, pnl_pct, vwap, ema_30, ewo, ewo_15min_avg,
                rsi, rsi_10min_avg, supertrend_direction, market_bias,
                ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a, ichimoku_senkou_b
        """
        self.bars.append(bar_data)
        if len(self.bars) > self.max_history:
            self.bars = self.bars[-self.max_history:]

    def _safe(self, val, fmt='.2f'):
        """Format a numeric value, returning 'N/A' for NaN."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f'{val:{fmt}}'

    def _price_change_pct(self, bars):
        """Percentage change from first to last bar in a slice."""
        if len(bars) < 2:
            return np.nan
        first = bars[0].get('stock_price', np.nan)
        last = bars[-1].get('stock_price', np.nan)
        if np.isnan(first) or np.isnan(last) or first == 0:
            return np.nan
        return ((last - first) / first) * 100

    def _rsi_trend(self, bars):
        """Summarize RSI trend: rising, falling, or flat."""
        vals = [b.get('rsi', np.nan) for b in bars if not np.isnan(b.get('rsi', np.nan))]
        if len(vals) < 2:
            return 'N/A'
        diff = vals[-1] - vals[0]
        if diff > 2:
            return f'rising ({vals[0]:.0f}->{vals[-1]:.0f})'
        elif diff < -2:
            return f'falling ({vals[0]:.0f}->{vals[-1]:.0f})'
        return f'flat (~{vals[-1]:.0f})'

    def _ewo_trend(self, bars):
        """Summarize EWO trend."""
        vals = [b.get('ewo', np.nan) for b in bars if not np.isnan(b.get('ewo', np.nan))]
        if len(vals) < 2:
            return 'N/A'
        diff = vals[-1] - vals[0]
        if diff > 0.05:
            return f'rising ({vals[-1]:.3f})'
        elif diff < -0.05:
            return f'falling ({vals[-1]:.3f})'
        return f'flat ({vals[-1]:.3f})'

    def _avg_volume(self, bars):
        """Average volume across bars."""
        vals = [b.get('volume', 0) for b in bars if b.get('volume', 0) > 0]
        if not vals:
            return 'N/A'
        avg = sum(vals) / len(vals)
        if avg >= 1_000_000:
            return f'{avg/1_000_000:.1f}M'
        elif avg >= 1_000:
            return f'{avg/1_000:.0f}K'
        return f'{avg:.0f}'

    def _supertrend_summary(self, bar):
        """Supertrend direction as text."""
        d = bar.get('supertrend_direction', np.nan)
        if isinstance(d, float) and np.isnan(d):
            return 'N/A'
        return 'BULLISH' if d == 1 else 'BEARISH'

    def _ichimoku_summary(self, bar):
        """Summarize Ichimoku cloud position."""
        price = bar.get('true_price', np.nan)
        span_a = bar.get('ichimoku_senkou_a', np.nan)
        span_b = bar.get('ichimoku_senkou_b', np.nan)
        tenkan = bar.get('ichimoku_tenkan', np.nan)
        kijun = bar.get('ichimoku_kijun', np.nan)

        parts = []
        if not any(np.isnan(v) for v in [price, span_a, span_b] if isinstance(v, float)):
            cloud_top = max(span_a, span_b)
            cloud_bot = min(span_a, span_b)
            if price > cloud_top:
                parts.append('Price ABOVE cloud')
            elif price < cloud_bot:
                parts.append('Price BELOW cloud')
            else:
                parts.append('Price INSIDE cloud')

        if not any(np.isnan(v) for v in [tenkan, kijun] if isinstance(v, float)):
            if tenkan > kijun:
                parts.append('Tenkan > Kijun (bullish)')
            elif tenkan < kijun:
                parts.append('Tenkan < Kijun (bearish)')
            else:
                parts.append('Tenkan = Kijun (neutral)')

        return ' | '.join(parts) if parts else 'N/A'

    def build_prompt_data(self, ticker, option_type, strike, pnl_pct, minutes_held):
        """
        Build a structured text block of current market data for the AI prompt.

        Returns:
            str: Formatted data block, or None if insufficient data.
        """
        if not self.bars:
            return None

        current = self.bars[-1]
        bars_5m = self.bars[-5:] if len(self.bars) >= 5 else self.bars
        bars_30m = self.bars[-30:] if len(self.bars) >= 30 else self.bars
        bars_1h = self.bars[-60:] if len(self.bars) >= 60 else self.bars

        data_block = (
            f"POSITION: {ticker} {option_type} ${strike} | P&L: {self._safe(pnl_pct)}% | Held: {minutes_held}min\n"
            f"\n"
            f"1-MINUTE (current bar):\n"
            f"  Price: ${self._safe(current.get('stock_price'))} | True Price: ${self._safe(current.get('true_price'))}\n"
            f"  RSI: {self._safe(current.get('rsi'), '.1f')} | EWO: {self._safe(current.get('ewo'), '.3f')}\n"
            f"  VWAP: ${self._safe(current.get('vwap'))} | EMA: ${self._safe(current.get('ema_30'))}\n"
            f"  Supertrend: {self._supertrend_summary(current)} | Bias: {self._safe(current.get('market_bias'), '.0f')}\n"
            f"\n"
            f"5-MINUTE VIEW ({len(bars_5m)} bars):\n"
            f"  Price Change: {self._safe(self._price_change_pct(bars_5m))}%\n"
            f"  RSI Trend: {self._rsi_trend(bars_5m)}\n"
            f"  EWO Trend: {self._ewo_trend(bars_5m)}\n"
            f"  Avg Volume: {self._avg_volume(bars_5m)}\n"
            f"\n"
            f"30-MINUTE VIEW ({len(bars_30m)} bars):\n"
            f"  Price Change: {self._safe(self._price_change_pct(bars_30m))}%\n"
            f"  RSI Trend: {self._rsi_trend(bars_30m)}\n"
            f"  EWO Trend: {self._ewo_trend(bars_30m)}\n"
            f"  Avg Volume: {self._avg_volume(bars_30m)}\n"
            f"\n"
            f"1-HOUR VIEW ({len(bars_1h)} bars):\n"
            f"  Price Change: {self._safe(self._price_change_pct(bars_1h))}%\n"
            f"  RSI Trend: {self._rsi_trend(bars_1h)}\n"
            f"  EWO Trend: {self._ewo_trend(bars_1h)}\n"
            f"  Avg Volume: {self._avg_volume(bars_1h)}\n"
            f"\n"
            f"ICHIMOKU: {self._ichimoku_summary(current)}\n"
        )

        return data_block


# =============================================================================
# INTERNAL - Prompt Templates
# =============================================================================

# System prompt: sets the role and constraints for the AI model.
SYSTEM_PROMPT = (
    "You are a professional intraday options trader AI assistant. "
    "You analyze technical indicators on 1-minute stock data and provide "
    "structured trading opinions. You are decisive, concise, and always "
    "respond in the exact JSON format requested. Never add commentary "
    "outside the JSON."
)

# User prompt template: filled with DataSnapshot output.
# The model must respond with the exact JSON schema below.
USER_PROMPT_TEMPLATE = (
    "Analyze this intraday options position and provide your assessment.\n"
    "\n"
    "{data_block}"
    "\n"
    "INSTRUCTIONS:\n"
    "1. Assess the market outlook at each timeframe (1min, 5min, 30min, 1hr).\n"
    "   - 'bullish' = price likely to rise (good for CALL, bad for PUT)\n"
    "   - 'bearish' = price likely to fall (good for PUT, bad for CALL)\n"
    "   - 'sideways' = no clear direction, range-bound\n"
    "2. Given the option type ({option_type}) and current P&L ({pnl_pct}%), "
    "recommend 'hold' or 'sell'.\n"
    "   - Consider: Does the multi-timeframe outlook favor the position direction?\n"
    "   - Consider: Is momentum fading? Is the position in profit that should be protected?\n"
    "   - Consider: Are key indicators (RSI, EWO, Supertrend, Ichimoku) confirming or diverging?\n"
    "3. Provide a one-sentence reason for your recommendation.\n"
    "\n"
    'Respond with ONLY this JSON (no markdown, no extra text):\n'
    '{{"outlook_1m":"bullish|bearish|sideways",'
    '"outlook_5m":"bullish|bearish|sideways",'
    '"outlook_30m":"bullish|bearish|sideways",'
    '"outlook_1h":"bullish|bearish|sideways",'
    '"action":"hold|sell",'
    '"reason":"one sentence explanation"}}'
)


# =============================================================================
# INTERNAL - Response Parser
# =============================================================================

class AISignalParser:
    """Parse and validate the structured JSON response from the AI model."""

    VALID_OUTLOOKS = {'bullish', 'bearish', 'sideways'}
    VALID_ACTIONS = {'hold', 'sell'}

    @staticmethod
    def parse(raw_response):
        """
        Parse the AI model's raw text response into a structured signal dict.

        Args:
            raw_response: Raw string output from the model.

        Returns:
            dict with keys:
                outlook_1m, outlook_5m, outlook_30m, outlook_1h: str
                action: str ('hold' or 'sell')
                reason: str
                valid: bool (True if parsing succeeded and all fields valid)
            Returns a default 'hold' signal if parsing fails.
        """
        default = {
            'outlook_1m': 'sideways',
            'outlook_5m': 'sideways',
            'outlook_30m': 'sideways',
            'outlook_1h': 'sideways',
            'action': 'hold',
            'reason': 'parse_error',
            'valid': False,
        }

        if not raw_response or not isinstance(raw_response, str):
            return default

        # Try to extract JSON from the response (handle markdown wrapping)
        text = raw_response.strip()
        if '```' in text:
            # Strip markdown code blocks
            lines = text.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_block = not in_block
                    continue
                if in_block or (not in_block and line.strip().startswith('{')):
                    json_lines.append(line)
            text = '\n'.join(json_lines)

        # Find the JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return default

        json_str = text[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return default

        # Validate and normalize fields
        result = {}
        for key in ['outlook_1m', 'outlook_5m', 'outlook_30m', 'outlook_1h']:
            val = str(data.get(key, '')).lower().strip()
            result[key] = val if val in AISignalParser.VALID_OUTLOOKS else 'sideways'

        action = str(data.get('action', '')).lower().strip()
        result['action'] = action if action in AISignalParser.VALID_ACTIONS else 'hold'

        result['reason'] = str(data.get('reason', 'no reason provided'))[:200]
        result['valid'] = True

        return result


# =============================================================================
# INTERNAL - Local AI Model Wrapper
# =============================================================================

class LocalAIModel:
    """
    Wrapper around llama-cpp-python for running local GGUF models.

    Loads a quantized LLM into GPU memory and runs structured inference
    for exit signal generation during backtesting.

    Recommended models (GGUF format):
        - 3080Ti (12GB): Mistral-7B-Instruct-v0.3-Q4_K_M.gguf (~4.4GB)
        - 4090  (24GB):  Llama-3-8B-Instruct-Q5_K_M.gguf (~5.7GB)
                         or Mistral-7B-Instruct-Q8_0.gguf (~7.7GB)

    Usage:
        model = LocalAIModel('/path/to/model.gguf')
        model.load()
        response = model.inference(system_prompt, user_prompt)
        model.unload()
    """

    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=2048, temperature=0.1,
                 max_tokens=256, seed=42):
        """
        Args:
            model_path: Absolute path to a GGUF model file.
            n_gpu_layers: GPU layers to offload (-1 = all layers to GPU).
            n_ctx: Context window size in tokens.
            temperature: Sampling temperature (low = more deterministic).
            max_tokens: Max tokens to generate per inference.
            seed: Random seed for reproducibility in backtesting.
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._llm = None

    def load(self):
        """Load the model into GPU memory. Call once before backtesting."""
        if self._llm is not None:
            return  # Already loaded

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for AI exit signals.\n"
                "Install with GPU support:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python\n"
                "See: https://github.com/abetlen/llama-cpp-python#installation"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download a GGUF model and set the path in Config.BACKTEST_CONFIG['ai_exit_signal']['model_path'].\n"
                "Recommended: Mistral-7B-Instruct-v0.3-Q4_K_M.gguf from https://huggingface.co/TheBloke"
            )

        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            seed=self.seed,
            verbose=False,
        )

    def unload(self):
        """Free the model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None

    def inference(self, system_prompt, user_prompt):
        """
        Run a single inference call.

        Args:
            system_prompt: System/role prompt.
            user_prompt: User prompt with data and instructions.

        Returns:
            str: Raw text response from the model.
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract text from the response
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return ''

    @property
    def is_loaded(self):
        return self._llm is not None


# =============================================================================
# INTERNAL - Analysis Logger (Self-Training Data Collection)
# =============================================================================

class AIAnalysisLogger:
    """
    Logs AI inference results alongside position data for future self-training.

    Each inference is saved as a JSONL record containing:
    - Input data snapshot (what the AI saw)
    - AI prediction (outlook + action + reason)
    - Position metadata (ticker, option_type, strike, pnl at time of prediction)
    - Outcome fields (filled in after position closes)

    The log file accumulates across backtests and can be used to:
    1. Fine-tune the model with LoRA/QLoRA on correct predictions
    2. Build preference datasets (correct vs incorrect predictions)
    3. Analyze accuracy by timeframe, ticker, and market condition

    File format: JSONL (one JSON object per line)
    Default path: ./ai_training_data/inference_log.jsonl
    """

    def __init__(self, log_dir='ai_training_data'):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, 'inference_log.jsonl')
        self._pending_records = {}  # trade_label -> list of records awaiting outcome
        self.run_id = _new_run_id()
        self._ensure_dir()
        self._existing_trade_keys = self._load_existing_keys()

    def _ensure_dir(self):
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_existing_keys(self):
        """Load trade keys already in the log to prevent duplicates across reruns."""
        keys = set()
        if not os.path.exists(self.log_path):
            return keys
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        trade_key = record.get('trade_key')
                        if trade_key:
                            keys.add(trade_key)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return keys

    @staticmethod
    def _make_trade_key(trade_label, timestamp):
        """
        Create a deterministic dedup key from trade label + timestamp.
        Same trade on same data always produces the same key.
        """
        raw = f"{trade_label}|{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def log_inference(self, trade_label, timestamp, data_block, ai_signal, pnl_pct, option_price):
        """
        Log a single AI inference.

        Args:
            trade_label: Position identifier (e.g., 'SPY:450:CALL')
            timestamp: Bar timestamp (datetime)
            data_block: The formatted data string sent to the AI
            ai_signal: Parsed AI response dict
            pnl_pct: Current P&L percentage at time of inference
            option_price: Current option price at time of inference
        """
        ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
        trade_key = self._make_trade_key(trade_label, ts_str)

        record = {
            'trade_key': trade_key,
            'run_id': self.run_id,
            'trade_label': trade_label,
            'timestamp': ts_str,
            'input_data': data_block,
            'prediction': {
                'outlook_1m': ai_signal.get('outlook_1m'),
                'outlook_5m': ai_signal.get('outlook_5m'),
                'outlook_30m': ai_signal.get('outlook_30m'),
                'outlook_1h': ai_signal.get('outlook_1h'),
                'action': ai_signal.get('action'),
                'reason': ai_signal.get('reason'),
                'valid': ai_signal.get('valid'),
            },
            'context': {
                'pnl_pct': float(pnl_pct) if not np.isnan(pnl_pct) else None,
                'option_price': float(option_price) if not np.isnan(option_price) else None,
            },
            'outcome': None,  # Filled in by finalize_trade()
        }

        if trade_label not in self._pending_records:
            self._pending_records[trade_label] = []
        self._pending_records[trade_label].append(record)

    def finalize_trade(self, trade_label, exit_reason, final_pnl_pct, exit_price):
        """
        Attach outcome data to all pending records for a completed trade,
        then flush them to disk.

        Args:
            trade_label: Position identifier
            exit_reason: How the position was closed
            final_pnl_pct: Final P&L percentage
            exit_price: Exit option price
        """
        records = self._pending_records.pop(trade_label, [])
        if not records:
            return

        outcome = {
            'exit_reason': exit_reason,
            'final_pnl_pct': float(final_pnl_pct) if final_pnl_pct is not None and not np.isnan(final_pnl_pct) else None,
            'exit_price': float(exit_price) if exit_price is not None and not np.isnan(exit_price) else None,
        }

        # For each inference, also record how much P&L changed after the prediction
        for i, record in enumerate(records):
            record['outcome'] = outcome
            pred_pnl = record['context']['pnl_pct']
            if pred_pnl is not None and outcome['final_pnl_pct'] is not None:
                record['outcome']['pnl_change_after'] = outcome['final_pnl_pct'] - pred_pnl

            # Was the AI's action correct?
            # "sell" was correct if P&L decreased after the prediction
            # "hold" was correct if P&L increased after the prediction
            if record['outcome'].get('pnl_change_after') is not None:
                action = record['prediction']['action']
                pnl_change = record['outcome']['pnl_change_after']
                if action == 'sell':
                    record['outcome']['action_correct'] = pnl_change < 0
                elif action == 'hold':
                    record['outcome']['action_correct'] = pnl_change >= 0

        # Append records to JSONL file, skipping duplicates from prior runs
        new_records = [r for r in records if r.get('trade_key') not in self._existing_trade_keys]
        if new_records:
            with open(self.log_path, 'a') as f:
                for record in new_records:
                    self._existing_trade_keys.add(record['trade_key'])
                    f.write(json.dumps(record) + '\n')

    def flush_remaining(self):
        """Flush any trades that never got finalized (e.g., still open at end of backtest)."""
        for trade_label in list(self._pending_records.keys()):
            self.finalize_trade(trade_label, 'unfinalised', np.nan, np.nan)


# =============================================================================
# INTERNAL - Optimal Exit Logger (Hindsight-Based Self-Training)
# =============================================================================

class OptimalExitLogger:
    """
    After each trade closes, walks the databook to find the bar where the option
    price peaked during holding. Logs the full indicator snapshot at that optimal
    exit point, plus context windows before/after the peak.

    This creates high-quality supervised training data:
    - Input: indicator conditions at the optimal sell moment
    - Label: "sell" (this WAS the best time)
    - Negative examples: bars before/after where "hold" was correct

    The model learns the statistical fingerprint of price peaks rather than
    relying on noisy after-the-fact binary correctness signals.

    Output: JSONL file at ./ai_training_data/optimal_exits.jsonl
    Each record contains:
        - trade metadata (ticker, option_type, strike, entry/exit info)
        - optimal_bar: full indicator snapshot at the peak
        - context_before: N bars of indicators leading into the peak
        - context_after: N bars of indicators after the peak (the decline)
        - actual_exit: indicator snapshot at the actual exit bar
        - efficiency: how close the actual exit was to optimal (%)
    """

    # Indicator columns to capture from each databook bar
    INDICATOR_KEYS = [
        'stock_price', 'true_price', 'option_price', 'volume',
        'pnl_pct', 'stop_loss', 'stop_loss_mode',
        'market_bias',
        'vwap', 'ema_30', 'vwap_ema_avg', 'emavwap',
        'ewo', 'ewo_15min_avg', 'rsi', 'rsi_10min_avg',
        'supertrend', 'supertrend_direction',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
        'milestone_pct', 'trailing_stop_price',
    ]

    def __init__(self, log_dir='ai_training_data', context_bars=5):
        """
        Args:
            log_dir: Directory for training data output.
            context_bars: Number of bars before/after peak to capture as context.
        """
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, 'optimal_exits.jsonl')
        self.context_bars = context_bars
        self.run_id = _new_run_id()
        os.makedirs(log_dir, exist_ok=True)
        self._existing_trade_keys = self._load_existing_keys()

    def _load_existing_keys(self):
        """Load trade keys already in the log to prevent duplicates across reruns."""
        keys = set()
        if not os.path.exists(self.log_path):
            return keys
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        trade_key = record.get('trade_key')
                        if trade_key:
                            keys.add(trade_key)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return keys

    @staticmethod
    def _make_trade_key(trade_label, entry_timestamp):
        """
        Create a deterministic dedup key from trade label + entry time.
        Same trade on same data always produces the same key, regardless
        of how many times the backtest is rerun.
        """
        raw = f"{trade_label}|{entry_timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _extract_bar(self, row):
        """Extract indicator values from a databook row (dict or Series) into a clean dict."""
        bar = {}
        for key in self.INDICATOR_KEYS:
            val = row.get(key, None)
            if val is None:
                bar[key] = None
            elif isinstance(val, float) and np.isnan(val):
                bar[key] = None
            elif isinstance(val, (np.integer, np.floating)):
                bar[key] = float(val)
            else:
                bar[key] = val
        # Always include timestamp
        ts = row.get('timestamp', None)
        bar['timestamp'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
        bar['minutes_held'] = float(row.get('minutes_held', 0)) if row.get('minutes_held') is not None and not (isinstance(row.get('minutes_held'), float) and np.isnan(row.get('minutes_held'))) else None
        return bar

    def log_trade(self, databook_records, position_info):
        """
        Analyze a completed trade's databook and log optimal exit data.

        Args:
            databook_records: list of dicts (Databook.records) from the trade.
            position_info: dict with keys:
                trade_label, ticker, option_type, strike, expiration,
                entry_price, entry_time, exit_price, exit_time, exit_reason,
                pnl_pct (final)
        """
        # Filter to holding bars only
        holding_bars = [r for r in databook_records if r.get('holding', False)]
        if len(holding_bars) < 3:
            return  # Not enough data to analyze

        # Dedup: build key from trade_label + entry timestamp of first holding bar
        trade_label = position_info.get('trade_label', '')
        entry_ts = holding_bars[0].get('timestamp', '')
        entry_ts_str = entry_ts.isoformat() if hasattr(entry_ts, 'isoformat') else str(entry_ts)
        trade_key = self._make_trade_key(trade_label, entry_ts_str)
        if trade_key in self._existing_trade_keys:
            return  # Already logged this exact trade from a prior run

        # Find the bar with the highest option price (optimal exit)
        peak_idx = 0
        peak_price = -np.inf
        for i, bar in enumerate(holding_bars):
            opt_price = bar.get('option_price', -np.inf)
            if not np.isnan(opt_price) and opt_price > peak_price:
                peak_price = opt_price
                peak_idx = i

        peak_bar = holding_bars[peak_idx]
        entry_price = position_info.get('entry_price', 0)
        exit_price = position_info.get('exit_price', 0)

        # Calculate exit efficiency: how much of the peak profit was captured
        # Efficiency = (actual_exit - entry) / (peak - entry) * 100
        peak_profit = peak_price - entry_price if entry_price else 0
        actual_profit = exit_price - entry_price if entry_price else 0
        if peak_profit > 0:
            efficiency = (actual_profit / peak_profit) * 100
        elif peak_profit == 0:
            efficiency = 100.0  # Exited at peak
        else:
            efficiency = 0.0  # Peak was below entry (bad trade)

        # Context windows around the peak
        ctx_start = max(0, peak_idx - self.context_bars)
        ctx_end = min(len(holding_bars), peak_idx + self.context_bars + 1)
        context_before = [self._extract_bar(holding_bars[i]) for i in range(ctx_start, peak_idx)]
        context_after = [self._extract_bar(holding_bars[i]) for i in range(peak_idx + 1, ctx_end)]

        # Actual exit bar (last holding bar)
        actual_exit_bar = self._extract_bar(holding_bars[-1])

        # Build the training record
        record = {
            'trade_key': trade_key,
            'run_id': self.run_id,
            'trade': {
                'trade_label': position_info.get('trade_label'),
                'ticker': position_info.get('ticker'),
                'option_type': position_info.get('option_type'),
                'strike': position_info.get('strike'),
                'expiration': str(position_info.get('expiration')) if position_info.get('expiration') else None,
                'entry_price': float(entry_price) if entry_price else None,
                'exit_price': float(exit_price) if exit_price else None,
                'exit_reason': position_info.get('exit_reason'),
                'final_pnl_pct': position_info.get('pnl_pct'),
                'total_holding_bars': len(holding_bars),
            },
            'optimal_exit': {
                'bar_index': peak_idx,
                'bars_from_entry': peak_idx,
                'bars_before_actual_exit': len(holding_bars) - 1 - peak_idx,
                'peak_option_price': float(peak_price),
                'peak_pnl_pct': float(((peak_price - entry_price) / entry_price) * 100) if entry_price else None,
                'indicators': self._extract_bar(peak_bar),
            },
            'context_before_peak': context_before,
            'context_after_peak': context_after,
            'actual_exit': {
                'indicators': actual_exit_bar,
            },
            'efficiency': round(efficiency, 2),
        }

        self._existing_trade_keys.add(trade_key)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


# =============================================================================
# EXTERNAL - Model Downloader
# =============================================================================

# Pre-configured models. Each entry has the HuggingFace repo, filename, and VRAM usage.
AVAILABLE_MODELS = {
    # 3080Ti (12GB) — leaves ~7GB headroom for other apps
    'mistral-7b-q4': {
        'repo_id': 'bartowski/Mistral-7B-Instruct-v0.3-GGUF',
        'filename': 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
        'vram_gb': 4.4,
        'description': 'Mistral 7B Instruct Q4_K_M — best balance of speed and quality for 3080Ti',
    },
    'llama3.1-8b-q4': {
        'repo_id': 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
        'filename': 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        'vram_gb': 4.9,
        'description': 'Llama 3.1 8B Instruct Q4_K_M — strong reasoning, slightly larger',
    },
    # 4090 (24GB) — can run higher quant for better quality
    'mistral-7b-q8': {
        'repo_id': 'bartowski/Mistral-7B-Instruct-v0.3-GGUF',
        'filename': 'Mistral-7B-Instruct-v0.3-Q8_0.gguf',
        'vram_gb': 7.7,
        'description': 'Mistral 7B Instruct Q8 — higher quality, needs 4090',
    },
    'llama3.1-8b-q8': {
        'repo_id': 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
        'filename': 'Meta-Llama-3.1-8B-Instruct-Q8_0.gguf',
        'vram_gb': 8.5,
        'description': 'Llama 3.1 8B Instruct Q8 — best quality for 4090',
    },
}


def download_model(model_key='mistral-7b-q4', models_dir=None):
    """
    Download a GGUF model from HuggingFace to the local models directory.

    Run this from Spyder's console:
        >>> import AIModel
        >>> AIModel.download_model()                    # Default: Mistral 7B Q4
        >>> AIModel.download_model('llama3.1-8b-q4')   # Llama 3.1 8B Q4

    Available model keys:
        'mistral-7b-q4'   — Mistral 7B Q4_K_M  (4.4 GB, recommended for 3080Ti)
        'llama3.1-8b-q4'  — Llama 3.1 8B Q4_K_M (4.9 GB, strong reasoning)
        'mistral-7b-q8'   — Mistral 7B Q8_0     (7.7 GB, for 4090)
        'llama3.1-8b-q8'  — Llama 3.1 8B Q8_0   (8.5 GB, for 4090)

    Args:
        model_key: Key from AVAILABLE_MODELS dict.
        models_dir: Where to save. Defaults to ./models/ in the project root.

    Returns:
        str: Absolute path to the downloaded model file (use this for Config model_path).
    """
    if model_key not in AVAILABLE_MODELS:
        print(f"Unknown model key: '{model_key}'")
        print(f"Available: {list(AVAILABLE_MODELS.keys())}")
        return None

    model_info = AVAILABLE_MODELS[model_key]

    # Default to project_root/models/
    if models_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    dest_path = os.path.join(models_dir, model_info['filename'])

    # Skip if already downloaded
    if os.path.exists(dest_path):
        size_gb = os.path.getsize(dest_path) / (1024 ** 3)
        print(f"[OK] Model already exists: {dest_path} ({size_gb:.1f} GB)")
        print(f"     Set Config model_path to: {dest_path}")
        return dest_path

    # Install huggingface_hub if not present
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'huggingface_hub'])
        from huggingface_hub import hf_hub_download

    print(f"Downloading: {model_info['description']}")
    print(f"  Repo:   {model_info['repo_id']}")
    print(f"  File:   {model_info['filename']}")
    print(f"  Size:   ~{model_info['vram_gb']} GB")
    print(f"  Dest:   {dest_path}")
    print()
    print("This may take several minutes depending on your connection...")
    print()

    # Download to HuggingFace cache, then symlink/copy to our models dir
    downloaded_path = hf_hub_download(
        repo_id=model_info['repo_id'],
        filename=model_info['filename'],
        local_dir=models_dir,
        local_dir_use_symlinks=False,  # Actual file copy, not symlink
    )

    # hf_hub_download may place it in a subfolder; move to expected location
    if os.path.exists(downloaded_path) and downloaded_path != dest_path:
        import shutil
        shutil.move(downloaded_path, dest_path)

    size_gb = os.path.getsize(dest_path) / (1024 ** 3)
    print()
    print(f"[OK] Download complete: {dest_path} ({size_gb:.1f} GB)")
    print()
    print(f"Next step — set your Config.py model_path:")
    print(f"    'model_path': r'{dest_path}',")
    print(f"    'enabled': True,")

    return dest_path


def list_models():
    """Print available models for download. Run from Spyder console."""
    print("\nAvailable GGUF Models for Download")
    print("=" * 65)
    for key, info in AVAILABLE_MODELS.items():
        print(f"\n  '{key}'")
        print(f"    {info['description']}")
        print(f"    VRAM: ~{info['vram_gb']} GB | File: {info['filename']}")
    print()
    print("Download with:  import AIModel; AIModel.download_model('model-key')")
    print()


# =============================================================================
# EXTERNAL - Module Interface
# =============================================================================

__all__ = [
    'DataSnapshot',
    'LocalAIModel',
    'AISignalParser',
    'AIAnalysisLogger',
    'OptimalExitLogger',
    'SYSTEM_PROMPT',
    'USER_PROMPT_TEMPLATE',
    'AVAILABLE_MODELS',
    'download_model',
    'list_models',
]
