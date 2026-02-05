#!/usr/bin/env python3
"""
test_peak_strategy.py - ARCHIVED/UNUSED

This file contains an experimental TestPeakExitSimple implementation that
uses EWO (Elliott Wave Oscillator) for peak detection. However, this
implementation is NOT USED by the main backtesting pipeline.

The ACTUAL peak exit strategy used is `TestPeakExit` in Strategy.py,
which uses RSI-based overbought detection instead.

This file is kept for reference but all code is commented out to avoid
confusion about which implementation is active.

Primary Bot Workflow:
- Discord messages -> Signal.py -> Test.py -> Strategy.py -> Dashboard.py
- Peak exit strategy: Strategy.TestPeakExit (RSI-based)

================================================================================
ARCHIVED CODE BELOW - Not part of active workflow
================================================================================
"""

# class TestPeakExitSimple:
#     '''Simple implementation of TEST Peak Exit for testing without numpy dependency.'''
#
#     def __init__(self, entry_price,
#                  ewo_overbought_threshold=0.5,
#                  ewo_spread_threshold=0.1,
#                  min_profit_pct=0.35,
#                  confirmation_bars=2,
#                  velocity_lookback=3,
#                  overbought_memory=3):
#         self.entry_price = entry_price
#         self.ewo_overbought_threshold = ewo_overbought_threshold
#         self.ewo_spread_threshold = ewo_spread_threshold
#         self.min_profit_pct = min_profit_pct
#         self.confirmation_bars = confirmation_bars
#         self.velocity_lookback = velocity_lookback
#         self.overbought_memory = overbought_memory
#
#         self.mode = 'watching'
#         self.max_option_price = entry_price
#         self.max_ewo_fast = 0.0
#         self.ewo_history = []
#         self.declining_count = 0
#         self.peak_ewo_value = 0.0
#         self.peak_price = entry_price
#         self.bars_since_overbought = 999
#
#     def _calculate_profit_pct(self, current_price):
#         if self.entry_price <= 0:
#             return 0.0
#         return (current_price - self.entry_price) / self.entry_price
#
#     def _calculate_ewo_velocity(self):
#         if len(self.ewo_history) < 2:
#             return None
#         lookback = min(self.velocity_lookback, len(self.ewo_history) - 1)
#         recent = self.ewo_history[-lookback:]
#         velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
#         return sum(velocities) / len(velocities) if velocities else None
#
#     def _is_overbought(self, ewo_fast, ewo_slow):
#         ewo_spread = ewo_fast - ewo_slow
#         return (ewo_fast >= self.ewo_overbought_threshold and
#                 ewo_spread >= self.ewo_spread_threshold)
#
#     def _was_recently_overbought(self, ewo_fast, ewo_slow):
#         '''Check if currently or recently overbought.'''
#         if self._is_overbought(ewo_fast, ewo_slow):
#             self.bars_since_overbought = 0
#             return True
#         self.bars_since_overbought += 1
#         return self.bars_since_overbought <= self.overbought_memory
#
#     def update(self, current_option_price, ewo_fast, ewo_slow):
#         if current_option_price > self.max_option_price:
#             self.max_option_price = current_option_price
#         if ewo_fast > self.max_ewo_fast:
#             self.max_ewo_fast = ewo_fast
#
#         self.ewo_history.append(ewo_fast)
#         if len(self.ewo_history) > 20:
#             self.ewo_history.pop(0)
#
#         profit_pct = self._calculate_profit_pct(current_option_price)
#         ewo_spread = ewo_fast - ewo_slow
#         ewo_velocity = self._calculate_ewo_velocity()
#
#         triggered = False
#         sell_reason = None
#
#         if len(self.ewo_history) >= 2:
#             if self.ewo_history[-1] < self.ewo_history[-2]:
#                 self.declining_count += 1
#             else:
#                 self.declining_count = 0
#
#         recently_overbought = self._was_recently_overbought(ewo_fast, ewo_slow)
#
#         if self.mode == 'watching':
#             if (profit_pct >= self.min_profit_pct and
#                     recently_overbought and
#                     ewo_velocity is not None and ewo_velocity < 0):
#                 self.mode = 'peak_detected'
#                 self.peak_ewo_value = self.max_ewo_fast
#                 self.peak_price = self.max_option_price
#
#         elif self.mode == 'peak_detected':
#             if self.declining_count >= self.confirmation_bars:
#                 self.mode = 'confirming'
#             if ewo_fast > self.peak_ewo_value * 1.02:
#                 self.mode = 'watching'
#                 self.declining_count = 0
#
#         elif self.mode == 'confirming':
#             if current_option_price < self.peak_price * 0.995:
#                 self.mode = 'triggered'
#                 triggered = True
#                 sell_reason = 'test_peak_momentum_reversal'
#             if ewo_fast > self.peak_ewo_value * 1.02:
#                 self.mode = 'watching'
#                 self.declining_count = 0
#
#         elif self.mode == 'triggered':
#             triggered = True
#             sell_reason = 'test_peak_momentum_reversal'
#
#         return {
#             'triggered': triggered,
#             'sell_reason': sell_reason,
#             'mode': self.mode,
#             'max_price': self.max_option_price,
#             'ewo_velocity': ewo_velocity,
#             'ewo_spread': ewo_spread,
#             'declining_count': self.declining_count,
#             'peak_ewo_value': self.peak_ewo_value,
#             'peak_price': self.peak_price,
#             'profit_pct': profit_pct,
#             'is_overbought': self._is_overbought(ewo_fast, ewo_slow),
#             'recently_overbought': recently_overbought
#         }
#
#
# Sample data and test functions also archived - see git history for full implementation
# def parse_data(data_str): ...
# def test_run(strategy, records, entry_price, verbose=False): ...
# def run_test(): ...
# if __name__ == '__main__': run_test()
