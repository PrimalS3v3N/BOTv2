# Alpha Trading Bot

## System Overview

A modular options trading bot that parses Discord signals and executes backtests.

## Module Architecture

```
Config.py ─────────────────────────────────────────────────────────────
    │
    ├── Signal.py (parses Discord signals)
    │
    ├── Data.py (sources market data)
    │       │
    │       └── Analysis.py (calculates indicators)
    │               │
    │               └── Strategy.py (decision logic)
    │                       │
    │                       └── Orders.py (order execution)
    │
    ├── Test.py (backtesting framework)
    │       │
    │       ├── Dashboard.py (visualization)
    │       └── AIModel.py (local LLM inference)
```

---

## Module Summaries

### Config.py
**Goal:** Store all configurable variables.

| Section | Description |
|---------|-------------|
| DISCORD_CONFIG | Discord bot token, channel ID, API settings |
| DATA_CONFIG | Data fetching parameters, cache TTL |
| ANALYSIS_CONFIG | Indicator periods, options pricing parameters |
| BACKTEST_CONFIG | Backtesting parameters, exit strategy configs, options_exit system |
| DATAFRAME_COLUMNS | Source of truth for all DataFrame column ordering |

---

### Signal.py
**Goal:** Parse Discord trading alert messages into structured order data.

**Outputs:**
- Ticker symbol
- Strike price
- Option type (CALL/PUT)
- Expiration date
- Contract cost

**Rules:**
- Do NOT modify parsing logic without testing against sample messages
- Always validate ticker via Robinhood API
- Handle multiple date formats (0DTE, tomorrow, explicit dates)

---

### Data.py
**Goal:** Source data, return DataFrames.

**Status:** Placeholder (to be implemented)

**Rules:**
- All data fetching logic goes here
- Return pandas DataFrames
- Implement caching to avoid redundant API calls
- Handle timezone conversions (use Eastern time)

---

### Analysis.py
**Goal:** Process all math here (technical indicators).

**Status:** Active

**Implemented Indicators:**

| Category | Indicator | Function | Default Period |
|----------|-----------|----------|---------------|
| Trend | EMA | `EMA(df, period)` | 10, 21, 50, 100, 200 |
| Trend | Supertrend | `Supertrend(df, atr_period, mult)` | 10 bars, 3.0x |
| Trend | Ichimoku Cloud | `IchimokuCloud(df)` | 9, 26, 52 |
| Momentum | RSI | `RSI(df, period)` | 14 |
| Momentum | EWO | `EWO(df, fast, slow)` | 5, 35 |
| Momentum | MACD | `MACD(df, fast, slow, signal)` | 12, 26, 9 |
| Momentum | ROC | `ROC(df, period)` | 30 |
| Volume | VWAP | `VWAP(df)` | Intraday reset |
| Volatility | ATR-SL | `ATR_SL(df, atr_period)` | 5 bar ATR, 10 bar HHV |
| Pricing | Black-Scholes | `estimate_option_price_bs(...)` | Calibrated to market |
| Pricing | Greeks | `calculate_greeks(...)` | Delta, Gamma, Theta, Vega, Rho |
| Utility | True Price | `true_price(close, high, low)` | (H + L + C) / 3 |

**Bundle:** `add_indicators(df)` adds all standard indicators in one call.

**Rules:**
- **ALL math and calculations go here** - no exceptions
- Pure calculation functions only
- No trading logic here
- No configuration values here (pull from Config.py)
- Accept DataFrames/values, return DataFrames/values with results

---

### Strategy.py
**Goal:** Logic for strategies. Process inputs from Data & Analysis, output decision logic.

**Status:** Active (4 exit strategies + primary TP/SL system)

**Exit Strategies:** See [Exit System Reference](#exit-system-reference) below.

**Rules:**
- **ALL trading logic and decision-making goes here** - no exceptions
- Entry logic goes here
- Exit logic goes here
- Accept analysis data, return buy/sell decisions
- No math or indicator calculations here (use Analysis.py)
- No configuration values here (pull from Config.py)
- Keep strategies modular and testable

---

### Orders.py
**Goal:** Central platform for sending, managing, and maintaining orders.

**Status:** Placeholder (to be implemented)

**Rules:**
- All broker API calls go here
- Handle order lifecycle (submit, monitor, cancel)
- Implement position tracking
- Log all order activity

---

### Test.py
**Goal:** Run backtests on historical Discord signals.

**Features:**
- Fetch Discord messages from signal channels
- Parse signals using Signal.py
- Get historical stock data (yfinance)
- Track positions through market hours
- Estimate option prices via calibrated Black-Scholes
- Generate results: Signals, Positions, Tracking matrices (Databook)
- Full indicator suite in tracking data

**Rules:**
- Exit strategies configurable via Config.py
- Use Config.py for all parameters
- Orchestration only — no inline math or trading logic

---

### Dashboard.py
**Goal:** Visualize backtest trade data from Test.py.

**Features:**
- Streamlit-based web dashboard
- Dual-axis charts (stock price vs option price)
- Entry/exit markers with annotations
- Stop loss tracking visualization (trailing TP, hard SL lines)
- Technical indicator overlays (EMAs, VWAP, Supertrend, Ichimoku)
- EWO and RSI subplots
- SPY and ticker gauge background shading
- Trade summary metrics

**Usage:**
```bash
streamlit run Dashboard.py
```

**Rules:**
- Reads data from BT_DATA.pkl (generated by Test.py)
- Display only - no trading logic here

---

### AIModel.py
**Goal:** Local LLM-based exit signal generation.

**Components:**
- `DataSnapshot` - Multi-timeframe data aggregation for prompts
- `AISignalParser` - Parse LLM response into structured signals
- `LocalAIModel` - GGUF model inference on GPU
- `AIAnalysisLogger` - Log inferences for self-training
- `OptimalExitLogger` - Hindsight analysis for training data

---

## Exit System Reference

All exit strategies live in Strategy.py. Each follows the Factory + Detector pattern:
a Factory class creates per-position Detector instances that track state bar-by-bar.

### Exit Priority Chain (evaluated in order)

| Priority | Strategy | Exit Reasons | Description |
|----------|----------|-------------|-------------|
| 1 | **Options Exit (Primary TP/SL)** | `OE-Hard-SL`, `OE-Trail-TP` | Hard stop loss + trailing take profit on option contract price |
| 2 | **Risk Downtrend** | `DownTrend-SL` | HIGH-risk trades: 3 negative bars or -10% drop |
| 3 | **StatsBook Exit** | `StatsBook - EWO Max`, `StatsBook - Range Max` | EWO or H-L range hits historical extreme |
| 4 | **Momentum Peak** | `Momentum Peak` | RSI overbought reversal + EWO decline |
| 5 | **AI Exit Signal** | `AI Exit Signal` | Local LLM recommends selling |
| 6 | **Closure-Peak** | `Closure - Peak` | RSI-based exit in last 30 minutes |
| 7 | **Closure-Market** | `Closure-Market` | Market close fallback (15:55) |

---

### 1. Options Exit — Primary TP/SL System

**Classes:** `OptionsExit` (factory) / `OptionsExitDetector` (per-position)

**Config:** `BACKTEST_CONFIG['options_exit']`

The primary exit system. Operates entirely on **option contract price** and differentiates
between CALLs and PUTs — a rising stock is good for calls but bad for puts.

#### Hard Stop Loss

On entry, a hard SL is placed at `entry_option_price * (1 - initial_sl_pct / 100)`.

Default: **20%** below entry. If the option price touches this level, exit immediately
with reason `OE-Hard-SL`. This is the maximum acceptable loss per trade.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_sl_pct` | 20 | Hard SL distance below entry (%) |

#### Trailing Take Profit

Once the option is profitable by `trail_activation_pct` (default 10%), a trailing TP
engages. It uses a **continuous logarithmic curve** rather than stepped milestones:

```
trail_tp_pct = trail_base_floor + trail_scale * ln(1 + profit_pct / trail_norm)
```

The TP price is: `entry_price * (1 + trail_tp_pct / 100)`

It **only ratchets up** (locks in more profit), never moves down.

**Example TP levels (standard risk):**

| Profit | Trailing SL locks in | Net if hit |
|--------|---------------------|------------|
| 10% | 5% above entry | +5% profit |
| 25% | ~20% | +20% |
| 50% | ~35% | +35% |
| 100% | ~70% | +70% |
| 200% | ~115% | +115% |

**High-risk addon:** When `RiskOutlook` scores the entry as HIGH risk, an extra
buffer is added:

```
addon = risk_addon_base + risk_addon_scale * ln(1 + profit_pct / risk_addon_norm)
```

This tightens the trailing TP so we lock in more profit faster on risky entries.
At 10% profit the addon is ~+2%, at 100% profit it's ~+15%.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trail_activation_pct` | 10 | Profit % required to activate trailing TP |
| `trail_base_floor_pct` | 5 | Minimum % above entry once activated |
| `trail_scale` | 25.0 | Controls slope of logarithmic curve |
| `trail_norm` | 30.0 | Normalisation factor (higher = slower ramp) |
| `risk_addon_base` | 2.0 | Base high-risk addon % |
| `risk_addon_scale` | 5.0 | Slope of high-risk addon curve |
| `risk_addon_norm` | 30.0 | Normalisation for high-risk addon |

Exit reason: `OE-Trail-TP`

#### RiskOutlook (Entry Favorability Assessment)

Called once at entry. Scores the entry environment as **LOW / MEDIUM / HIGH** risk
by evaluating 7 factors against the contract type (CALL vs PUT).

Two time horizons for trend context:
- **Primary (30 minutes):** ROC over the last 30 bars — immediate momentum
- **Secondary (since open):** ROC from first bar of day to current — session context

**Risk factors scored:**

| # | Factor | What it checks | Risk added |
|---|--------|---------------|------------|
| 1 | ROC 30-min | Buying at top (calls up a lot) or bottom (puts down a lot) | +2 to +3 |
| 1b | ROC since-open | Session extended in wrong direction | +1 to +2 |
| 2 | RSI overbought/oversold | Overbought for calls or oversold for puts = bad | +3 (bad) / -1 (good) |
| 3 | EMA positioning | Price below EMAs for calls = reversal risk | +1 to +2 per EMA |
| 4 | ATR-SL | Price below ATR trailing stop for calls = bearish | +2 |
| 5 | Supertrend | Direction opposes contract type | +2 |
| 6 | MACD histogram | Negative for calls or positive for puts | +1 |
| 7 | EWO momentum | Negative for calls or positive for puts | +1 |

**Score thresholds:** LOW (0-2), MEDIUM (3-5), HIGH (6+)

HIGH risk entries get the trailing TP addon and are eligible for risk downtrend monitoring.

#### EMA Reversal Detection

Checked every bar. Counts how many of the configured EMAs (default: 10, 21, 50) the
stock price has crossed against the contract direction. If `ema_reversal_sensitivity`
(default: 2) or more are breached, the `sl_ema_reversal` flag is set to True.

For calls: price below EMA = against. For puts: price above EMA = against.
Smaller EMAs breached = more concerning (more recent reversal).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ema_reversal_periods` | [10, 21, 50] | EMAs to check |
| `ema_reversal_sensitivity` | 2 | How many breached = reversal flag |

#### Post-Entry Confirmation

During the first N bars (default: 10), the detector tracks option price trajectory.
After the window closes, the trade is marked:
- **Confirmed** — price increased with >= 50% positive bars
- **Denied** — price dropped > 3% or < 30% positive bars
- **Pending** — inconclusive

Stored in `tp_confirmed` column.

#### Databook Columns

| Column | Prefix | Description |
|--------|--------|-------------|
| `tp_trailing` | TP | Current trailing TP price level (NaN until activated) |
| `sl_hard` | SL | Hard stop loss price level (constant) |
| `tp_risk_outlook` | TP | RiskOutlook score: LOW / MEDIUM / HIGH |
| `tp_risk_reasons` | TP | Pipe-separated reason flags |
| `tp_trend_30m` | TP | 30-min trend: Uptrend / Downtrend / Sideways |
| `sl_ema_reversal` | SL | EMA reversal flag (True/False) |
| `tp_confirmed` | TP | Post-entry confirmation: Confirmed / Denied / Pending |
| `exit_sig_oe` | — | Boolean: would OE trigger exit this bar |

---

### 2. Risk Downtrend Monitor

**Not a Strategy class** — logic lives inline in Test.py's simulation loop.

Only activates for trades where `_assess_risk()` returned HIGH at entry.
Monitors for sustained downtrend:
- **3 consecutive negative bars** (configurable via `downtrend_monitor_bars`)
- **-10% drop** below entry (configurable via `downtrend_drop_pct`)

Exit reason: `DownTrend-SL`

---

### 3. StatsBook Exit

**Classes:** `StatsBookExit` (factory) / `StatsBookDetector` (per-position)

**Config:** `BACKTEST_CONFIG['statsbook_exit']`

Uses pre-computed historical statistics (StatsBook) to identify when current price
action reaches historical extremes. Compares EWO and rolling H-L range against
Median.Max bounds from 5-day, 3-month, and 1-year history.

**Conditions (any triggers exit):**
1. EWO >= Median.Max(EWO) for the configured timeframe — momentum at typical peak
2. Rolling H-L range >= Median.Max(H-L) — range at typical extreme

**Guards:**
- Minimum profit required (`min_profit_pct`, default: 10%)
- Minimum bars held (`min_hold_bars`, default: 5)

Exit reasons: `StatsBook - EWO Max`, `StatsBook - Range Max`

---

### 4. Momentum Peak

**Classes:** `MomentumPeak` (factory) / `MomentumPeakDetector` (per-position)

**Config:** `BACKTEST_CONFIG['momentum_peak']`

Detects momentum exhaustion peaks — fires 1-2 bars after a peak, before the bulk
of the reversal.

**All conditions must be met:**
1. Option profit >= `min_profit_pct` (15%)
2. RSI reached `rsi_overbought` (80) within last `rsi_lookback` (5) bars
3. RSI dropped >= `rsi_drop_threshold` (10 points) from its recent peak
4. EWO declining for `ewo_declining_bars` (1+) consecutive bars
5. RSI crossed below its 10-min average (optional, `require_rsi_below_avg`)

Exit reason: `Momentum Peak`

---

### 5. AI Exit Signal

**Classes:** `AIExitSignal` (factory) / `AIExitSignalDetector` (per-position)

**Config:** `BACKTEST_CONFIG['ai_exit_signal']`

Runs a local quantized LLM (Meta-Llama-3-8B-Instruct GGUF) on GPU. Evaluates
multi-timeframe technical data and returns structured outlook + action.

**Signal fields:**
- `outlook_1m`, `outlook_5m`, `outlook_30m`, `outlook_1h`: bullish / bearish / sideways
- `action`: hold or sell
- `reason`: one-sentence explanation

Evaluated every `eval_interval` (5) bars after `min_bars_before_eval` (5) bars.
Between evaluations the last signal is cached.

Exit reason: `AI Exit Signal`

---

### 6. Closure-Peak

**Not a Strategy class** — logic lives inline in Test.py.

In the last 30 minutes before market close, if the 10-min average RSI is:
- >= 85 for CALLs (overbought, take profits)
- <= 15 for PUTs (oversold, take profits)

Exit reason: `Closure - Peak`

---

### 7. Closure-Market

**Not a Strategy class** — hardcoded in Test.py at 15:55.

Fallback exit: close any position still open at end of day.

Exit reason: `Closure-Market`

---

## Code Organization Rules

Every module follows this structure:

```python
"""
ModuleName.py - Brief Description

Module Goal: One sentence description.

================================================================================
INTERNAL - Core Logic
================================================================================
"""

# Internal implementation here


"""
================================================================================
EXTERNAL - Module Interface
================================================================================
Modules: List of modules this interacts with
"""

import Config

# External interface functions here
```

---

## Development Rules

1. **Keep it simple** - Code should be easy to read and reason about
2. **One module, one purpose** - Each module has a single responsibility
3. **Config-driven** - All parameters come from Config.py
4. **Internal/External sections** - Clearly separate internal logic from external interfaces
5. **Incremental development** - Build features one at a time, test before adding more

---

## Module Boundary Rules (ENFORCED)

These rules are mandatory. All code changes must respect module boundaries.

### Config.py owns ALL settings
- Every tunable parameter, threshold, toggle, and default value lives in Config.py
- Other modules read settings via `Config.get_config()` - they never define their own
- If you need a new setting, add it to the appropriate Config section first

### Analysis.py owns ALL math
- Every calculation, formula, indicator, and pricing model lives in Analysis.py
- This includes: technical indicators (EMA, VWAP, EWO, ATR, RSI, MACD, ROC, Supertrend, Ichimoku), options pricing (Black-Scholes, Greeks), and any future math
- Other modules call `Analysis.<function>()` - they never compute inline
- If you need a new calculation, add a function to Analysis.py first

### Strategy.py owns ALL trading logic
- Every entry condition, exit condition, stop loss rule, and profit target lives in Strategy.py
- Strategy reads from Analysis for indicator values and from Config for thresholds
- Other modules call `Strategy.<function>()` for decisions - they never inline trading logic
- If you need a new strategy or decision rule, add it to Strategy.py first

### Test.py is orchestration only
- Test.py wires together Config, Analysis, Strategy, Signal, and Data
- It fetches data, calls Analysis for calculations, calls Strategy for decisions, and records results
- It must NOT contain math formulas, trading logic, or hardcoded settings
- Wrapper functions around Analysis/Strategy calls are not allowed - call the source module directly

### What goes where - quick reference
| Type | Belongs in | NOT in |
|------|-----------|--------|
| Thresholds, periods, percentages | Config.py | Analysis, Strategy, Test |
| Indicators, formulas, pricing | Analysis.py | Strategy, Test, Dashboard |
| Entry/exit rules, stop loss logic | Strategy.py | Test, Analysis |
| Data fetching, backtesting loop | Test.py | Analysis, Strategy |
| Visualization, charts | Dashboard.py | everywhere else |
| Signal parsing | Signal.py | everywhere else |

---

## Current Status

| Module | Status |
|--------|--------|
| Config.py | Active |
| Signal.py | Active |
| Test.py | Active (backtesting framework) |
| Dashboard.py | Active (full visualization) |
| Data.py | Placeholder |
| Analysis.py | Active (EMA, VWAP, EWO, RSI, MACD, ROC, Supertrend, Ichimoku, ATR-SL, Black-Scholes) |
| Strategy.py | Active (MomentumPeak, StatsBookExit, AIExitSignal, OptionsExit) |
| AIModel.py | Active (local LLM inference + training data logging) |
| Orders.py | Placeholder |

---

## Next Steps

1. Build out Data.py for data sourcing
2. Add more technical indicators in Analysis.py (VPOC, Volume Profile)
3. Expand Strategy.py with entry conditions
4. Build Orders.py for live trading
