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
    │       └── Dashboard.py (visualization)
```

---

## Module Summaries

### Config.py
**Goal:** Store all configurable variables.

| Section | Description |
|---------|-------------|
| DISCORD_CONFIG | Discord bot token, channel ID, API settings |
| RISK_CONFIG | Position sizing, instrument defaults, stop loss |
| EXIT_CONFIG | Exit strategy toggles (flags only) |
| DATA_CONFIG | Data fetching parameters |
| BACKTEST_CONFIG | Backtesting parameters, stop loss settings |

**Stop Loss Configuration:**
All stop loss settings are consolidated into a `stop_loss` nested structure:
```python
'stop_loss': {
    'enabled': True,
    'stop_loss_pct': 0.35,              # Max loss from entry
    'stop_loss_warning_pct': 0.15,      # Warning threshold
    'trailing_stop_pct': 0.20,          # Trailing below highest
    'time_stop_minutes': 60,            # Time stop
    'breakeven_threshold_pct': None,    # Auto-calculate
    'breakeven_min_minutes': 30,        # Min hold before breakeven
    'trailing_trigger_pct': 0.50,       # Start trailing at profit
}
```
This structure exists in both `RISK_CONFIG.option_defaults` (live trading) and `BACKTEST_CONFIG` (backtesting) as `stop_loss`.

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
- EMA (Exponential Moving Average) - 30-period default
- VWAP (Volume Weighted Average Price) - resets daily
- EWO (Elliott Wave Oscillator) - EMA(5) - EMA(35)
- ATR (Average True Range) - single bar high-low range
- RSI (Relative Strength Index) - 14-period default
- Black-Scholes Options Pricing (calls, puts, Greeks)

**Planned Indicators:**
- MACD (Moving Average Convergence Divergence)
- VPOC (Volume Point of Control)
- SuperTrend

**Functions:**
- `EMA(df, column='close', period=30)` - Calculate EMA on any column
- `VWAP(df, price_col='close', volume_col='volume')` - Calculate intraday VWAP
- `EWO(df, column='close', fast=5, slow=35)` - Calculate Elliott Wave Oscillator
- `ATR(stock_high, stock_low)` - Calculate Average True Range for a single bar
- `RSI(df, column='close', period=14)` - Calculate Relative Strength Index
- `add_indicators(df, ema_period=30)` - Add all standard indicators to DataFrame
- `black_scholes_price(S, K, T, r, sigma, option_type)` - Option pricing
- `calculate_greeks(S, K, T, r, sigma, option_type)` - Delta, Gamma, Theta, Vega, Rho
- `estimate_option_price_bs(...)` - Black-Scholes with delta adjustment

**Rules:**
- **ALL math and calculations go here** - no exceptions
- Pure calculation functions only
- No trading logic here
- No configuration values here (pull from Config.py)
- Accept DataFrames/values, return DataFrames/values with results

---

### Strategy.py
**Goal:** Logic for strategies. Process inputs from Data & Analysis, output decision logic.

**Status:** Active

**Implemented Strategies:**
- StopLoss (3-phase exit strategy)
- TestPeakExit (profit trailing stop with RSI detection)

**StopLoss:**
Three-phase stop loss system for options contracts (configured via `stop_loss` dict):

1. **INITIAL Phase**: Fixed stop loss at `entry_price × (1 - stop_loss_pct)`
   - Default: 35% below entry (`stop_loss_pct: 0.35`)
   - Example: Entry $1.00 → Stop at $0.65

2. **BREAKEVEN Phase**: When price rises above threshold, move stop to entry price
   - Threshold = `entry_price / (1 - stop_loss_pct)` (auto-calculated if `breakeven_threshold_pct: None`)
   - Example: Entry $1.00, 35% SL → threshold = $1.54
   - **Requires minimum `breakeven_min_minutes` (30)** before transitioning

3. **TRAILING Phase**: When profit reaches `trailing_trigger_pct` (50%), trail at `trailing_stop_pct` (20%) below highest price
   - Only uses price highs AFTER entry (not before)
   - Stop only moves up, never down

4. **REVERSAL Exit**: If True Price falls below VWAP, reversal detected → exit as "SL - Reversal"
   - Enabled via `reversal_exit_enabled: True` in `stop_loss` config
   - Bearish Signal: When EMA falls below VWAP (use to buy a Put in future reversal system)

**Classes:**
- `StopLoss` - Stateful stop loss manager
- `check_stop_loss()` - Stateless function for backtesting
- `TestPeakExit` - Stateful profit trailing stop manager
- `check_test_peak_exit()` - Stateless function for backtesting

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
- Generate results: Signals, Positions, Tracking matrices
- Stop loss with 3-phase strategy
- Technical indicators (EMA_30, VWAP) in tracking data

**Exit Strategies:**
- StopLoss (3-phase: initial → breakeven → trailing)
- SL - Reversal (True Price < VWAP)
- Market close exit (default fallback)

**Tracking Matrix Columns:**
- timestamp, stock_price, stock_high, stock_low, option_price, volume
- holding, entry_price, pnl, pnl_pct
- stop_loss, stop_loss_mode (initial/breakeven/trailing)
- vwap, ema_30, ewo

**Rules:**
- Exit strategies configurable via Config.py
- Use Config.py for all parameters

---

### Dashboard.py
**Goal:** Visualize backtest trade data from Test.py.

**Features:**
- Streamlit-based web dashboard
- Dual-axis charts (stock price vs option price)
- Entry/exit markers with annotations
- Stop loss tracking visualization (shows mode changes)
- Technical indicator overlays (EMA_30, VWAP)
- Trade summary metrics
- Matrix data table with all tracking columns

**Chart Displays:**
- Stock price with error bars showing high/low (blue line, left axis)
- Option price (orange line, right axis)
- VWAP (green dashed line, left axis)
- EMA 30 (purple dotted line, left axis)
- Stop Loss (green dashed line with green shaded area below, right axis)
- Entry/Exit markers with annotations
- EWO subplot below main chart (with positive/negative fill)

**Trade Summary Metrics:**
- Entry, Exit, P&L %, Max Profit %, Min Profit %, Exit Reason, Duration

**Usage:**
```bash
streamlit run Dashboard.py
```

**Rules:**
- Reads data from BT_DATA.pkl (generated by Test.py)
- Display only - no trading logic here

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
- This includes: technical indicators (EMA, VWAP, EWO, ATR, RSI), options pricing (Black-Scholes, Greeks), and any future math
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
| Analysis.py | Active (EMA, VWAP, EWO, ATR, RSI, Black-Scholes) |
| Strategy.py | Active (StopLoss, TestPeakExit) |
| Orders.py | Placeholder |

---

## Known Bugs

### BUG-001: Reversal Exit Fires on Entry Bar (Same-Candle Buy/Sell)

**Severity:** High
**File:** `Test.py:1257`
**Observed:** 2026-02-09 11:24

**How to identify in data:** Single row with `holding=TRUE`, `pnl_pct=+0.0%`,
`stop_loss_mode=initial`. All surrounding rows are `FALSE`. The exit reason in the positions
table will show `SL - Reversal`. Check: is `True Price < VWAP` (CALL) or `True Price > VWAP`
(PUT) at the entry bar? True Price = (high + low + close) / 3.

**Problem:** The reversal exit (`SL - Reversal`) has no guard for the entry bar. It runs on
every bar including `i == entry_idx`, so if `True Price < VWAP` (CALL) or `True Price > VWAP`
(PUT) is already true at entry, the position opens and immediately closes on the same candle
with ~0% PnL.

**Code path:** `Test.py` → `_simulate_position()` loop → StopLoss.update() sets
`reversal=True` (Strategy.py:139-151) → exit check at line 1257 fires with no `i > entry_idx`
guard → position.close() on same bar.

**Example:** At 11:24, True Price $1000.95 < VWAP $1001.15 → instant reversal exit.
RSI was 32.8 (DD check passed fine), but the VWAP relationship was never checked pre-entry.

**Fix Options:**
1. Add `i > entry_idx` guard to the reversal exit check at line 1257
2. Add VWAP/price relationship check to DD pre-entry validation
3. Both (recommended)

---

### BUG-002: Downtrend Exit Fires on Entry Bar (Same-Candle Buy/Sell)

**Severity:** High
**File:** `Test.py:1263`
**Observed:** 2026-02-09 09:56

**How to identify in data:** Same pattern as BUG-001 (single `TRUE` row, `+0.0%`, `initial`)
but exit reason is `SL - DownTrend`. Check: is `True Price < vwap_ema_avg AND EMA <
vwap_ema_avg` (CALL) or the inverse (PUT) at the entry bar?

**Problem:** Same structural bug as BUG-001 but for the downtrend exit (`SL - DownTrend`).
The downtrend check also has no `i > entry_idx` guard, so if `True Price < vwap_ema_avg AND
EMA < vwap_ema_avg` is already true at entry, the position is immediately closed.

**Code path:** `Test.py` → `_simulate_position()` loop → StopLoss.update() sets
`downtrend=True` (Strategy.py:164-173) → exit check at line 1263 fires with no `i > entry_idx`
guard → position.close() on same bar.

**Example:** At 09:56, True Price $690.58 < vwap_ema_avg $690.78, AND EMA $690.50 <
vwap_ema_avg $690.78 → instant downtrend exit. RSI was 51.1 (perfectly neutral).

**Fix Options:**
1. Add `i > entry_idx` guard to the downtrend exit check at line 1263 (same fix as BUG-001)
2. Add downtrend condition check to DD pre-entry validation
3. Both (recommended)

---

### BUG-003: Overbought Reentry Conditions Too Strict (Missed Entries)

**Severity:** Medium
**File:** `Test.py:922-924`
**Observed:** 2026-02-10 09:42

**How to identify in data:** ALL rows show `holding=FALSE` for the entire day. The positions
table will show exit reason `Overbought - No Reentry`. Look for a period where raw RSI dropped
well below 30 but RSI_10min_avg stayed above 30, meaning the averages never caught up to the
actual pullback.

**Problem:** When DD rejects a CALL entry as overbought, the reentry scan requires ALL THREE
conditions simultaneously: `rsi_10min_avg <= 30 AND ewo < 0 AND ewo_15min_avg < 0`. The use
of lagging averaged indicators (10min RSI avg, 15min EWO avg) makes this triple-AND condition
nearly impossible to satisfy during V-shaped recoveries, causing missed profitable entries.

**Code path:** `Test.py` → `_process_signal()` → DD check fails (RSI > 85) → scan loop at
line 910 → checks triple-AND at line 922-924 → condition never met → falls through to
line 961 → `Overbought - No Reentry`.

**Example:** Signal at 09:30 with RSI 91.9 (correctly rejected). At 09:42, raw RSI was 24.2
(excellent CALL entry), but RSI_10min_avg was 41.0 (still > 30 threshold) and EWO_15min_avg
was 0.088 (still > 0). The triple condition was NEVER met. Stock recovered from $693.66 to
$696+ by 09:58 — missed >100% option move.

**Bar-by-bar near misses (showing why triple-AND fails):**
| Time  | RSI  | RSI_avg | EWO    | EWO_avg | RSI_avg<=30? | EWO<0? | EWO_avg<0? |
|-------|------|---------|--------|---------|--------------|--------|------------|
| 09:42 | 24.2 | 41.0    | -0.446 | 0.088   | NO (41>30)   | YES    | NO         |
| 09:43 | 31.6 | 37.4    | -0.488 | 0.050   | NO (37>30)   | YES    | NO         |
| 09:44 | 37.8 | 34.3    | -0.468 | 0.012   | NO (34>30)   | YES    | NO         |
| 09:45 | 36.1 | 32.2    | -0.464 | -0.035  | NO (32>30)   | YES    | YES        |
| 09:46 | 41.4 | 32.6    | -0.423 | -0.088  | NO (33>30)   | YES    | YES        |

RSI_avg bottomed at 32.2 (never reached <=30). By the time EWO_avg went negative, the
opportunity was gone.

**Root Cause:** The triple-AND with lagging averages is a binary gate that doesn't account
for signal strength. A raw RSI of 24.2 is a very strong pullback signal, but it gets zero
credit because the 10-minute average hasn't caught up. The system needs to weigh HOW MUCH
conditions are met, not just IF they are met.

**Design: Replace Binary Gate with Confidence Score (see below)**

---

### DESIGN: Adaptive Reentry Scoring System (BUG-003 Fix)

**Goal:** Replace the binary triple-AND gate with a weighted confidence score that lets strong
signals compensate for lagging indicators. The system should self-adjust: a deep RSI pullback
can overcome a slightly positive EWO average, and time elapsed since rejection adds confidence
that the overbought condition has cooled.

**Current system (binary):**
```
IF rsi_10min_avg <= 30 AND ewo < 0 AND ewo_15min_avg < 0 → ENTER
ELSE → SKIP
```

**Proposed system (weighted score):**
```
score = (w_rsi * rsi_score) + (w_ewo * ewo_score) + (w_ewo_avg * ewo_avg_score) + (w_time * time_score)
IF score >= confidence_threshold → ENTER
```

**Component scoring (CALL overbought reentry):**

| Component | What it measures | Score formula | Range |
|-----------|-----------------|---------------|-------|
| RSI pullback | How far RSI dropped from overbought | `(85 - raw_rsi) / 85` clamped to [0, 1] | 0-1 |
| EWO direction | Is momentum shifting? | `1.0 if ewo < 0, else max(0, 1 - ewo/0.5)` | 0-1 |
| EWO avg trend | Has avg momentum shifted? | `1.0 if ewo_avg < 0, else max(0, 1 - ewo_avg/0.5)` | 0-1 |
| Time decay | How long since rejection? | `min(minutes_elapsed / 20, 1.0)` | 0-1 |

**Default weights (configurable in Config.py):**
```python
'reentry_scoring': {
    'enabled': True,
    'weight_rsi': 0.40,          # RSI pullback depth (most important)
    'weight_ewo': 0.25,          # EWO direction
    'weight_ewo_avg': 0.15,      # EWO average (least important, lags most)
    'weight_time': 0.20,         # Time elapsed since rejection
    'confidence_threshold': 0.60, # Minimum score to enter
    'min_rsi_raw': 40,           # Safety: raw RSI must be below this regardless of score
}
```

**09:42 example with scoring:**
```
rsi_score    = (85 - 24.2) / 85           = 0.715
ewo_score    = 1.0  (ewo = -0.446 < 0)    = 1.000
ewo_avg_score = max(0, 1 - 0.088/0.5)     = 0.824
time_score   = min(12/20, 1.0)            = 0.600

score = (0.40 * 0.715) + (0.25 * 1.0) + (0.15 * 0.824) + (0.20 * 0.6)
      = 0.286 + 0.250 + 0.124 + 0.120
      = 0.780

0.780 >= 0.60 → ENTER ✓
raw RSI 24.2 < 40 → safety check passes ✓
```

The strong RSI pullback (0.715) and EWO direction (1.0) overcome the lagging EWO average.

**PUT oversold reentry (mirror logic):**
Same structure but inverted: RSI rising from oversold, EWO positive, etc.
```
rsi_score = (raw_rsi - 15) / 85  (how far RSI rose from oversold)
ewo_score = 1.0 if ewo > 0
ewo_avg_score = 1.0 if ewo_avg > 0
```

**Where this lives in the codebase:**
- Scoring logic: `Strategy.py` (new function `score_reentry_confidence()`)
- Config: `Config.py` under `due_diligence.reentry_scoring`
- Caller: `Test.py:910-926` (replace the triple-AND scan with scoring call)

**Advantages over current system:**
1. No missed V-shaped recoveries — strong pullbacks score high even with lagging averages
2. Self-adjusting — each component contributes proportionally, no single indicator can veto
3. Time decay adds urgency — the longer we wait, the more likely overbought has cooled
4. Safety rail — `min_rsi_raw` prevents entering if RSI hasn't actually pulled back
5. Fully configurable — all weights/thresholds in Config.py, easy to tune via backtesting
6. Backwards compatible — set `enabled: False` to keep the old binary logic

---

## TODO

- [ ] Build a contract reversal system looking for Bearish or Bullish Signals
  - Bearish Signal: EMA < VWAP → buy a Put
  - Bullish Signal: (TBD criteria) → buy a Call
  - Integrate with StopLoss reversal detection for automatic contract flipping

## Next Steps

1. Build out Data.py for data sourcing
2. Add more technical indicators in Analysis.py (MACD, VPOC, SuperTrend)
3. Expand Strategy.py with entry conditions
4. Build Orders.py for live trading
