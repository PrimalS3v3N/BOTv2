# Strategies Reference

Complete documentation of all exit strategies in `Strategy.py`.
Each strategy follows a Factory → Detector pattern: the factory reads config and creates per-position detector instances.

---

## Table of Contents

1. [Options Exit (Primary TP/SL)](#1-options-exit-primary-tpsl)
2. [Momentum Peak](#2-momentum-peak)
3. [StatsBook Exit](#3-statsbook-exit)
4. [Volume Climax Exit](#4-volume-climax-exit)
5. [Time Stop](#5-time-stop)
6. [VWAP Cross Exit](#6-vwap-cross-exit)
7. [Supertrend Flip Exit](#7-supertrend-flip-exit)
8. [Market Trend](#8-market-trend)
9. [AI Exit Signal](#9-ai-exit-signal)
10. [Closure Peak](#10-closure-peak)

---

## 1. Options Exit (Primary TP/SL)

**Classes:** `OptionsExit` → `OptionsExitDetector`

### Scope

The backbone of the entire exit system. Every position gets an OptionsExitDetector. It manages two stop-loss layers and an entry risk assessment:

**Hard Stop-Loss:**
- Set at entry: `entry_price * (1 - initial_sl_pct / 100)`. Default 20% below entry.
- Dynamically tightens as the position gains profit (before trailing SL activates). If the option peaks at +8% before trailing kicks in at +10%, the hard SL ratchets up proportionally. HIGH-risk entries double the tightening speed.
- Never loosens — ratchet only moves up.

**Trailing Stop-Loss:**
- Activates once profit hits `trail_activation_pct` (default 10%).
- Uses a continuous logarithmic curve instead of fixed milestones:
  ```
  trail_sl_pct = base_floor + scale * ln(1 + profit_pct / norm)
  ```
- The SL is expressed as a % above entry price, so the actual SL price = `entry * (1 + trail_sl_pct / 100)`.
- For HIGH-risk entries, an addon is layered on top that ratchets the SL even tighter:
  ```
  addon = addon_base + addon_scale * ln(1 + profit_pct / addon_norm)
  ```
- An **adaptive buffer** (computed from StatsBook Median.Max(H-L) and option delta) ensures the SL does not sit so tight that normal per-bar noise triggers it. The buffer decays as profit grows — loose at low profits, tight at high profits.
- Ratchet: SL only moves up, never down.

**RiskOutlook (Entry Favorability Assessment):**
Called once at entry. Evaluates 8 factors against the contract direction and assigns LOW / MEDIUM / HIGH risk:

| # | Factor | What it checks | Risk points |
|---|--------|----------------|-------------|
| 1 | ROC 30m | Is the 30-min trend with or against us? | Up to +3 |
| 1b | ROC Day | Is the session trend extended or adverse? | Up to +2 |
| 2 | RSI | Combined RSI + RSI_avg at entry | Up to +3 (or -1 if favorable) |
| 3 | EMAs | How many EMAs (10/21/50) is price on the wrong side of? | +1 or +2 per EMA |
| 4 | ATR-SL | Is price below (CALL) / above (PUT) the ATR stop? | +2 |
| 5 | Supertrend | Is Supertrend direction against us? | +2 |
| 6 | MACD | Is MACD histogram against position direction? | +1 |
| 7 | EWO | Is EWO momentum against position direction? | +1 |
| 8 | Stochastic | Is Stochastic in extreme zone against us? | +2 (or -1 if favorable) |

Score >= 6 = HIGH, >= 3 = MEDIUM, else LOW. HIGH-risk entries get tighter SL via the addon.

**Confirmation Window:**
Tracks option price over the first N bars (default 10). Classifies trade as Confirmed (price rising, >50% positive bars), Denied (price dropped >3% or <30% positive bars), or Pending.

**EMA Reversal Detection:**
Per-bar check: if stock price crosses the wrong side of N EMAs (default sensitivity=2 of [10, 21, 50]), flags ema_reversal=True.

### Strengths

- **Always-on protection** — every position has a hard floor preventing catastrophic losses.
- **Logarithmic trailing** — locks in more profit at lower levels while giving room at higher levels. No milestone gap dead zones.
- **Adaptive buffer** — stock-specific volatility calibration via StatsBook. TSLA gets more room than KO.
- **Risk-aware tightening** — HIGH-risk entries get double-speed hard SL tightening and trailing addon.
- **Entry context** — RiskOutlook gives all other strategies awareness of entry quality.

### Weaknesses

- **No directional intelligence** — trailing SL is purely price-based. A healthy pullback and a genuine reversal look the same.
- **Log curve too generous at high profits** — at +80%, the gap between watermark and SL can be wide.
- **Confirmation is one-shot** — evaluated once at bar N, never revisited.
- **RiskOutlook is entry-only** — assessed once, permanent for the position.
- **Hard SL can be too tight for high-IV options** — 20% may be within normal noise for deep OTM.

### Settings

```python
'options_exit': {
    'enabled': True,
    'initial_sl_pct': 20,
    'hard_sl_tighten_on_peak': True,
    'trail_activation_pct': 10,
    'trail_base_floor_pct': 5,
    'trail_early_floor_minutes': 5,
    'trail_scale': 20.0,
    'trail_norm': 40.0,
    'trail_buffer_adaptive': True,
    'trail_buffer_bars': 1.5,
    'trail_buffer_min_pct': 5.0,
    'trail_buffer_max_pct': 25.0,
    'trail_buffer_decay_norm': 50.0,
    'risk_addon_base': 2.0,
    'risk_addon_scale': 5.0,
    'risk_addon_norm': 30.0,
    'confirmation_window_bars': 10,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_reversal_periods': [10, 21, 50],
    'ema_reversal_sensitivity': 2,
    'atr_sl_enabled': True,
    'macd_enabled': True,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'roc_period': 30,
}
```

### Upgrades

- **Dynamic risk re-assessment**: Re-evaluate RiskOutlook periodically instead of once at entry.
- **Profit-dependent curve switching**: Steeper curve at very high profits (>60%) to lock in aggressively.
- **IV-adjusted hard SL**: Scale initial_sl_pct based on implied volatility at entry.
- **Confirmation re-evaluation**: Allow Denied trades to upgrade if they recover.
- **Multi-leg awareness**: Coordinate SL across spread legs.

---

## 2. Momentum Peak

**Classes:** `MomentumPeak` → `MomentumPeakDetector`

### Scope

Detects momentum exhaustion — the point where a stock reached an extreme (RSI overbought/oversold), started reversing, and multiple indicators confirm the peak is in. Exits near the top of a move rather than waiting for trailing SL.

**Architecture:** 3 hard gates + 6 graduated scored conditions + 2 bonus signals.

**Hard Gates (must ALL pass):**
1. **Min profit** — position must be at least +15%.
2. **RSI hit extreme** — within last 5 bars, RSI >= 80 (CALLs) or <= 20 (PUTs).
3. **RSI delta** — RSI moved >= 10 points away from the extreme.

**Scored Conditions (graduated — stronger signal = up to 1.25x weight):**

| # | Condition | Weight | Graduation |
|---|-----------|--------|------------|
| 4 | RSI zone exit | 2 | RSI@31=1.05, RSI@50=2.0, RSI@60+=2.5 |
| 5 | EWO adverse trend | 2 | 3 bars=1.5, 6 bars=2.0, 9+=2.5 |
| 6 | RSI vs 10-min avg | 1 | gap@2=0.4, gap@10=1.0, gap@20+=1.25 |
| 7 | Stochastic crossover | 1 | K/D divergence@0=0.75, @10=1.0, @25+=1.25 |
| 8 | Spread contraction | 2 | 3 bars=1.5, 6 bars=2.0, 9+=2.5 |
| 9 | Bar range contraction | 2 | 3 bars=1.5, 6 bars=2.0, 9+=2.5 |

**Bonus Confidence Signals (add to score, never gate):**

| # | Signal | Weight | How it works |
|---|--------|--------|-------------|
| 10 | RSI velocity | 1 | RSI rate of change over 3 bars. Speed 5=0.5, 15=1.0, 25+=1.25 |
| 11 | EWO acceleration | 1 | Adverse trend steepening. Ratio 1.5x=0.75, 2.0x=1.0, 3.0x+=1.25 |

**Total possible: ~14.5 pts. Default threshold: 7.**

### Strengths

- **Multi-signal confirmation** — 8+ signals agreeing is reliable. Reduces false exits.
- **Graduated scoring** — captures signal quality. RSI@50 scores higher than RSI@31.
- **Flexible threshold** — strong signals compensate for one lagging condition.
- **Velocity/acceleration bonuses** — catch fast reversals before bar-count conditions accumulate.

### Weaknesses

- **Requires RSI extreme first** — blind to reversals from moderate levels (RSI 65 to 40).
- **5-bar lookback can miss extended peaks** — slow broad peaks fall outside the window.
- **Stochastic is bar-specific** — a crossover 3 bars ago gets no credit.
- **Threshold tuning is empirical** — depends on market conditions and volatility.

### Settings

```python
'momentum_peak': {
    'enabled': True,
    'min_profit_pct': 15,
    'rsi_overbought': 80,
    'rsi_oversold': 20,
    'rsi_lookback': 5,
    'rsi_drop_threshold': 10,
    'rsi_recovery_level': 30,
    'ewo_declining_bars': 3,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    'spread_contraction_bars': 3,
    'bar_range_contraction_bars': 3,
    'score_rsi_recovery': 2,
    'score_ewo_trend': 2,
    'score_rsi_vs_avg': 1,
    'score_stochastic': 1,
    'score_spread_contraction': 2,
    'score_bar_range': 2,
    'score_rsi_velocity': 1,
    'score_ewo_acceleration': 1,
    'score_threshold': 7,
}
```

### Upgrades

- **Extend stochastic window** — partial credit for crossover within last N bars.
- **Spread/bar range magnitude** — factor in total % decline, not just bar count.
- **Dynamic RSI lookback** — RSI@90+ extends lookback (reversal takes longer).
- **Volume confirmation bonus** — declining volume = strong exhaustion signal.
- **Adaptive threshold** — lower for HIGH-risk entries, raise for LOW-risk.

---

## 3. StatsBook Exit

**Classes:** `StatsBookExit` → `StatsBookDetector`

### Scope

Uses StatsBook — pre-computed historical statistical bounds per stock — to identify when price action reached historically extreme levels. If EWO or rolling H-L range hits the stock typical peak (Median.Max), the move is statistically near its limit.

**How it works:**
1. Extracts bounds from StatsBook for 5m timeframe:
   - Median.Max(EWO) — typical peak EWO
   - Median.Max(H-L) — typical peak bar range
2. Per bar, after min_hold_bars and min_profit_pct gates:
   - EWO >= Median.Max(EWO) = exit
   - Rolling H-L >= Median.Max(H-L) = exit

Uses Median.Max (not absolute Max) because absolute max includes outliers.

### Strengths

- **Stock-specific calibration** — each stock has its own bounds.
- **Historically grounded** — bounds from actual data, not assumptions.
- **Complementary** — catches absolute levels; Momentum Peak catches directional reversals.

### Weaknesses

- **Backward-looking** — may not apply during earnings/news.
- **No directional awareness** — EWO at max could be rising or falling.
- **H-L exit disabled** — rolling range too noisy.
- **Binary** — 99% of max = zero credit, 100% = exit.

### Settings

```python
'statsbook_exit': {
    'enabled': True,
    'timeframe': '5m',
    'ewo_max_exit': True,
    'hl_max_exit': False,
    'min_profit_pct': 10,
    'min_hold_bars': 5,
    'rolling_window': 5,
}
```

### Upgrades

- **Graduated scoring** — feed percentile into Momentum Peak. EWO at 80% of max = partial credit.
- **Direction-aware** — only exit if EWO peaked AND declining.
- **Multi-timeframe** — check 5m AND 15m bounds.
- **Percentile scoring** — continuous percentile instead of binary max check.

---

## 4. Volume Climax Exit

**Classes:** `VolumeClimaxExit` → `VolumeClimaxDetector`

### Scope

Detects volume climax events — huge volume spike (>= 3x rolling avg) coinciding with a price reversal candle. Marks blow-off tops (CALLs) or capitulation bottoms (PUTs).

**Conditions (all must be met):**
1. Profit >= 10%
2. Held >= 10 bars
3. Volume >= 3x the 20-bar rolling average
4. Price reversal: CALLs = bearish candle, PUTs = bullish candle
5. (Optional, disabled) Stochastic extreme zone

### Strengths

- **Catches blow-off tops/bottoms** — high-conviction, rare events.
- **Single-bar signal** — reacts immediately.
- **Low false positive rate** — 3x volume AND reversal candle is strict.

### Weaknesses

- **Misses slow exhaustion** — only dramatic single-bar climaxes.
- **No magnitude awareness** — 3x and 10x spike treated identically.
- **Volume data quality** — block trades can fake spikes.

### Settings

```python
'volume_climax_exit': {
    'enabled': True,
    'volume_lookback': 20,
    'volume_multiplier': 3.0,
    'min_profit_pct': 10,
    'min_hold_bars': 10,
    'use_stochastic': False,
    'stoch_overbought': 75,
    'stoch_oversold': 25,
}
```

### Upgrades

- **Graduated volume scoring** — 3x/5x/10x as graduated score.
- **Candle body ratio** — score abs(close-open)/(high-low).
- **Multi-bar climax** — 2-3 bars of elevated volume with reversal.
- **Delta volume** — bid/ask volume for aggressive selling/buying detection.

---

## 5. Time Stop

**Classes:** `TimeStop` → `TimeStopDetector`

### Scope

Exits stale positions. Options lose value via theta decay, so holding stagnant trades costs real money.

**Logic:** `Exit if: minutes_held >= 90 AND pnl_pct < 5%`

Above 5% profit = working trade, managed by trailing SL.

### Strengths

- **Addresses theta decay** — the risk every other strategy ignores.
- **Dead simple** — two parameters, zero bugs.
- **Capital efficiency** — frees buying power.

### Weaknesses

- **Fixed time window** — 90 min treats all stocks equally.
- **No context** — ignores entry timing, DTE, market session.
- **No graduated urgency** — nothing at 89 min, full exit at 90.

### Settings

```python
'time_stop': {
    'enabled': True,
    'max_minutes': 90,
    'min_profit_pct': 5,
}
```

### Upgrades

- **Theta-aware** — scale max_minutes by DTE. 0DTE = shorter.
- **Graduated exits** — reduce at 60 min, close at 90.
- **Sliding profit threshold** — raise required profit as time increases.
- **Market session awareness** — morning vs midday vs power hour.

---

## 6. VWAP Cross Exit

**Classes:** `VWAPCrossExit` → `VWAPCrossDetector`

### Scope

Price crossing VWAP against position signals institutional flow shifted:
- CALLs: price drops below VWAP = sellers in control
- PUTs: price rises above VWAP = buyers in control

**Conditions:**
1. Profit >= 5%
2. Held >= 10 bars
3. Price was on favorable side of VWAP at some point
4. Crossed to adverse side, stayed for 2 consecutive bars

### Strengths

- **Institutional signal** — VWAP is the benchmark algorithms trade against.
- **Confirmation reduces whipsaws** — 2-bar filter.
- **Low false positive in trends** — only crosses when trend weakens.

### Weaknesses

- **Noisy in ranging markets** — constant VWAP crosses in chop.
- **Resets daily** — first 15-30 min unreliable.
- **No magnitude** — 1 cent below VWAP = same as $5 below.

### Settings

```python
'vwap_cross_exit': {
    'enabled': True,
    'min_profit_pct': 5,
    'min_hold_bars': 10,
    'confirm_bars': 2,
}
```

### Upgrades

- **Distance scoring** — score by distance past VWAP.
- **VWAP band tolerance** — neutral zone around VWAP.
- **Morning filter** — disable first 15-30 min.
- **VWAP slope** — factor in VWAP trend direction.

---

## 7. Supertrend Flip Exit

**Classes:** `SupertrendFlipExit` → `SupertrendFlipDetector`

### Scope

Supertrend produces binary direction (1=bullish, -1=bearish) via ATR bands. When it flips favorable to adverse mid-trade, price broke through a volatility-adjusted support/resistance level.

**Conditions:**
1. Held >= 5 bars
2. Profit >= 5%
3. Direction was favorable at some point
4. Direction flipped adverse, stayed 1 bar

### Strengths

- **Volatility-adjusted** — ATR bands adapt to market conditions.
- **Structural breaks** — detects actual S/R level breaks, not oscillator exhaustion.
- **Low noise** — does not flip on small pullbacks.

### Weaknesses

- **Lagging** — price already moved through the band before detection.
- **Whipsaws in chop** — rapid flip-flop around the band.
- **Single timeframe** — 1m flip may be noise; 15m flip is significant.

### Settings

```python
'supertrend_flip_exit': {
    'enabled': True,
    'min_profit_pct': 5,
    'min_hold_bars': 5,
    'confirm_bars': 1,
}
```

### Upgrades

- **Multi-timeframe** — check 5m and 15m Supertrend.
- **ATR distance scoring** — score by penetration depth past band.
- **Dynamic confirm_bars** — more bars for volatile stocks.
- **Feed into Momentum Peak** — as bonus signal in scoring system.

---

## 8. Market Trend

**Classes:** `MarketTrend` → `MarketTrendDetector`

### Scope

Evaluates macro trend using VWAP-based signals (-1/0/+1) for ticker and SPY. State machine:

- **Hold** — trend favorable.
- **Warning** — shifted to sideways.
- **Sell** — reversed against position.
- **HighRisk** — entered against trend. Grace period (10 bars) then Sell.
- **SPY-Warning** — SPY turned but ticker has not yet.

**exit_enabled = False (default)** — monitoring/flagging only.

### Strengths

- **Macro awareness** — only strategy considering SPY/broader market.
- **State machine** — models natural trend deterioration.
- **SPY leading indicator** — SPY often turns before individual stocks.
- **Non-destructive default** — context without forced exits.

### Weaknesses

- **Coarse** — only -1/0/+1, no trend strength.
- **SPY correlation varies** — not all stocks track SPY.
- **Exit disabled** — state machine analysis currently unused for exits.

### Settings

```python
'market_trend': {
    'enabled': True,
    'exit_enabled': False,
    'high_risk_grace_bars': 10,
    'spy_diverge_profit_pct': 5,
}
```

### Upgrades

- **Trend strength scoring** — continuous score not just -1/0/+1.
- **Dynamic grace period** — scale by how adverse the entry was.
- **Sector ETF** — check sector ETF (XLK, XLF) not just SPY.
- **Feed into other strategies** — lower Momentum Peak threshold on Warning state.

---

## 9. AI Exit Signal

**Classes:** `AIExitSignal` → `AIExitSignalDetector`

### Scope

Runs local LLM (Meta-Llama-3-8B) on GPU for multi-timeframe outlook and hold/sell decisions:
- Outlooks: 1m/5m/30m/1h (bullish/bearish/sideways)
- Action: hold or sell
- Reason: one-sentence explanation

Evaluates every 5 bars, caches between evaluations. All inferences logged for training.

**Currently disabled by default.**

### Strengths

- **Holistic analysis** — sees ALL indicators simultaneously.
- **Natural language reasoning** — human-readable explanations.
- **Self-improving** — logs enable future fine-tuning.

### Weaknesses

- **Latency** — 100-500ms per inference.
- **GPU requirement** — needs capable hardware.
- **Not fine-tuned** — general-purpose model, not trading-specific.
- **5-bar gap** — cached signals go stale.

### Settings

```python
'ai_exit_signal': {
    'enabled': False,
    'model_path': '...Meta-Llama-3-8B-Instruct.Q5_K_M.gguf',
    'n_gpu_layers': -1,
    'n_ctx': 2048,
    'temperature': 0.1,
    'max_tokens': 256,
    'seed': 42,
    'eval_interval': 5,
    'min_bars_before_eval': 5,
    'exit_on_sell': True,
    'log_inferences': True,
    'log_dir': 'ai_training_data',
}
```

### Upgrades

- **Fine-tune on outcomes** — use logged data + actual P&L for training.
- **Confidence scoring** — output 1-10 confidence; only act on high.
- **Ensemble** — weight AI sell as bonus in Momentum Peak scoring.
- **Smaller model** — 3B/1B for lower latency.

---

## 10. Closure Peak

**Config only** — handled in backtester/dashboard, not a Strategy.py class.

### Scope

End-of-day RSI-based exit. Within 30 min of close:
- CALLs: RSI >= 85 = exit
- PUTs: RSI <= 15 = exit

### Strengths

- **End-of-day risk** — power hour volatility, MOC orders.
- **Simple and targeted** — narrow activation window.
- **0DTE safety** — holding through close is catastrophic for same-day expiry.

### Weaknesses

- **Very narrow** — only fires if RSI hits 85/15 in last 30 min.
- **No profit consideration** — triggers regardless of P&L.
- **Not in Strategy.py** — no factory/detector pattern.

### Settings

```python
'closure_peak': {
    'enabled': True,
    'rsi_call_threshold': 85,
    'rsi_put_threshold': 15,
    'minutes_before_close': 30,
}
```

### Upgrades

- **Migrate to Strategy.py** — proper factory/detector class.
- **Graduated timing** — lower threshold as close approaches (30 min: 85, 15 min: 75, 5 min: 65).
- **Profit-aware** — if highly profitable, lower threshold to lock in.
- **Force exit** — close all positions within 5 min of close regardless.

---

## Strategy Interaction Map

| Strategy | Informs | Informed By |
|----------|---------|-------------|
| **Options Exit** | Sets risk label (HIGH/MED/LOW) | StatsBook (adaptive buffer) |
| **Momentum Peak** | Primary smart exit signal | RSI, EWO, Stoch, spread, bar range |
| **StatsBook Exit** | Exits at historical extremes | StatsBook data |
| **Volume Climax** | Catches blow-off events | Volume + price action |
| **Time Stop** | Frees stale capital | Time + P&L only |
| **VWAP Cross** | Institutional flow shift | VWAP price relationship |
| **Supertrend Flip** | Structural trend break | Supertrend direction |
| **Market Trend** | Macro context + SPY alignment | Ticker + SPY VWAP trends |
| **AI Exit Signal** | Holistic multi-factor analysis | All indicators |
| **Closure Peak** | End-of-day safety net | RSI + time to close |

**Exit Priority:** When multiple strategies signal simultaneously, the first to trigger wins. Options Exit (Hard-SL) has implicit priority since it checks before trailing SL, which checks before other strategies.

---

## Quick Tuning Guide

| Want to... | Adjust |
|------------|--------|
| Exit peaks earlier | Lower `momentum_peak.score_threshold` (try 6) |
| Let winners run longer | Raise `momentum_peak.score_threshold` (try 9) |
| Tighter stop losses | Lower `options_exit.initial_sl_pct` (try 15) |
| More room for volatile stocks | Raise `options_exit.trail_buffer_max_pct` |
| Cut stale trades faster | Lower `time_stop.max_minutes` (try 60) |
| Reduce false VWAP exits | Raise `vwap_cross_exit.confirm_bars` (try 3) |
| More aggressive peak detection | Raise `momentum_peak.score_rsi_velocity` (try 2) |
| Disable a condition entirely | Set its `score_*` weight to 0 |
