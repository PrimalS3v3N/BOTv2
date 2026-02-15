"""
Strategy.py - Trading Strategy Logic

Module Goal: Logic for strategies. Process inputs from Data & Analysis,
output decision logic for execution.

================================================================================
INTERNAL - Strategy Decision Logic
================================================================================
"""

import numpy as np


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: Config.py, Data.py, Analysis.py, Orders.py, Test.py
"""

import Config

# Export for use by other modules
__all__ = ['AIExitSignal', 'AIExitSignalDetector']


# =============================================================================
# INTERNAL - AI Exit Signal Strategy
# =============================================================================

class AIExitSignal:
    """
    Factory for creating AIExitSignalDetector instances.

    Wraps the local AI model (AIModel.py) to provide LLM-based exit signals.
    The model evaluates multi-timeframe market outlook and recommends hold/sell.

    Config: BACKTEST_CONFIG['ai_exit_signal']
        {
            'enabled': True,
            'model_path': '/path/to/model.gguf',
            'eval_interval': 5,           # Evaluate every N bars
            'min_bars_before_eval': 5,    # Wait N bars before first eval
            'exit_on_sell': True,         # Actually trigger exit on 'sell'
            'log_inferences': True,       # Save data for self-training
            ...
        }
    """

    def __init__(self, config=None):
        ai_config = config or Config.get_setting('backtest', 'ai_exit_signal', {})
        self.enabled = ai_config.get('enabled', False)
        self.config = ai_config

        self._model = None
        self._logger = None
        self._optimal_logger = None

    def load_model(self):
        """Load the AI model into GPU memory. Call once before backtesting."""
        if not self.enabled:
            return

        import AIModel

        model_path = self.config.get('model_path', '')
        self._model = AIModel.LocalAIModel(
            model_path=model_path,
            n_gpu_layers=self.config.get('n_gpu_layers', -1),
            n_ctx=self.config.get('n_ctx', 2048),
            temperature=self.config.get('temperature', 0.1),
            max_tokens=self.config.get('max_tokens', 256),
            seed=self.config.get('seed', 42),
        )
        self._model.load()

        log_dir = self.config.get('log_dir', 'ai_training_data')

        if self.config.get('log_inferences', True):
            self._logger = AIModel.AIAnalysisLogger(log_dir=log_dir)

        # Always create the optimal exit logger (works even without the AI model running)
        self._optimal_logger = AIModel.OptimalExitLogger(log_dir=log_dir)

    def unload_model(self):
        """Free model from GPU memory. Call after backtesting."""
        if self._model is not None:
            self._model.unload()
            self._model = None
        if self._logger is not None:
            self._logger.flush_remaining()

    def create_detector(self, ticker, option_type, strike):
        """
        Create a new AIExitSignalDetector for a position.

        Args:
            ticker: Stock ticker symbol
            option_type: 'CALL' or 'PUT'
            strike: Option strike price

        Returns:
            AIExitSignalDetector instance, or None if disabled.
        """
        if not self.enabled or self._model is None:
            return None
        return AIExitSignalDetector(
            self.config, self._model, self._logger,
            ticker, option_type, strike
        )

    @property
    def logger(self):
        return self._logger

    @property
    def optimal_logger(self):
        return self._optimal_logger


class AIExitSignalDetector:
    """
    Tracks AI exit signal state for a single position.

    Accumulates bar data into a DataSnapshot, runs AI inference at
    configured intervals, and caches the last signal between evaluations.

    Signal fields (available per bar via current_signal):
        outlook_1m:  'bullish', 'bearish', or 'sideways'
        outlook_5m:  'bullish', 'bearish', or 'sideways'
        outlook_30m: 'bullish', 'bearish', or 'sideways'
        outlook_1h:  'bullish', 'bearish', or 'sideways'
        action:      'hold' or 'sell'
        reason:      One-sentence explanation
    """

    def __init__(self, config, model, logger, ticker, option_type, strike):
        import AIModel

        self.eval_interval = config.get('eval_interval', 5)
        self.min_bars_before_eval = config.get('min_bars_before_eval', 5)
        self.exit_on_sell = config.get('exit_on_sell', True)

        self._model = model
        self._logger = logger
        self._snapshot = AIModel.DataSnapshot(max_history=60)
        self._parser = AIModel.AISignalParser()

        self.ticker = ticker
        self.option_type = option_type
        self.strike = strike
        self.trade_label = f"{ticker}:{strike}:{option_type}"

        self._bars_since_eval = 0
        self._total_bars = 0

        # Last AI signal (persists between evaluations)
        self._current_signal = {
            'outlook_1m': None,
            'outlook_5m': None,
            'outlook_30m': None,
            'outlook_1h': None,
            'action': 'hold',
            'reason': None,
            'valid': False,
        }

    def update(self, bar_data, pnl_pct, minutes_held, option_price, timestamp):
        """
        Process a new bar and optionally run AI inference.

        Args:
            bar_data: dict with current bar's indicator values
                (stock_price, true_price, volume, vwap, ema values, market_bias, etc.)
            pnl_pct: Current position P&L percentage
            minutes_held: Minutes since entry
            option_price: Current option price
            timestamp: Bar timestamp

        Returns:
            (should_exit, exit_reason): Tuple. exit_reason is None if no exit.
        """
        self._snapshot.add_bar(bar_data)
        self._total_bars += 1
        self._bars_since_eval += 1

        # Wait for minimum bars before first evaluation
        if self._total_bars < self.min_bars_before_eval:
            return False, None

        # Only evaluate at configured intervals
        if self._bars_since_eval < self.eval_interval:
            # Return cached signal's exit decision
            if self.exit_on_sell and self._current_signal.get('action') == 'sell':
                return True, 'AI Exit Signal'
            return False, None

        # Time to run inference
        self._bars_since_eval = 0

        data_block = self._snapshot.build_prompt_data(
            ticker=self.ticker,
            option_type=self.option_type,
            strike=self.strike,
            pnl_pct=pnl_pct,
            minutes_held=minutes_held,
        )

        if data_block is None:
            return False, None

        import AIModel

        user_prompt = AIModel.USER_PROMPT_TEMPLATE.format(
            data_block=data_block,
            option_type=self.option_type,
            pnl_pct=f"{pnl_pct:.1f}" if not np.isnan(pnl_pct) else "N/A",
        )

        try:
            raw_response = self._model.inference(AIModel.SYSTEM_PROMPT, user_prompt)
            signal = self._parser.parse(raw_response)
        except Exception:
            signal = self._parser.parse(None)

        self._current_signal = signal

        # Log inference for self-training
        if self._logger is not None:
            self._logger.log_inference(
                trade_label=self.trade_label,
                timestamp=timestamp,
                data_block=data_block,
                ai_signal=signal,
                pnl_pct=pnl_pct,
                option_price=option_price,
            )

        # Check exit
        if self.exit_on_sell and signal.get('action') == 'sell':
            return True, 'AI Exit Signal'

        return False, None

    @property
    def current_signal(self):
        """Current AI signal dict (last evaluation result, cached between evals)."""
        return self._current_signal
