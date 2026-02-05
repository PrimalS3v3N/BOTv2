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
| RISK_CONFIG | Position sizing, instrument defaults, dynamic stop loss |
| EXIT_CONFIG | Exit strategy toggles (flags only) |
| DATA_CONFIG | Data fetching parameters |
| BACKTEST_CONFIG | Backtesting parameters, dynamic stop loss settings |

**Dynamic Stop Loss Configuration:**
All stop loss settings are consolidated into a `dynamic_stop_loss` nested structure:
```python
'dynamic_stop_loss': {
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
This structure exists in both `RISK_CONFIG.option_defaults` (live trading) and `BACKTEST_CONFIG` (backtesting).

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
- Dynamic Stop Loss (3-phase exit strategy)
- Tiered Profit Exit (profit-based exit with trailing)
- TestPeakExit (profit trailing stop with RSI detection)

**Dynamic Stop Loss:**
Three-phase stop loss system for options contracts (configured via `dynamic_stop_loss` dict):

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

**Tiered Profit Exit:**
Dynamic profit target system based on profit percentage tiers.
Only active when stock price is ABOVE EMA 30.
Uses `dynamic_stop_loss.stop_loss_pct` for trailing calculations (single source of truth).

Contract Tiers (framework for multi-contract management):
- Tier 1: 1 contract - sell all at exit signal
- Tier 2: 3 contracts - sell 1 at each profit tier
- Tier 3: 5+ contracts - sell progressively at profit tiers

Profit Target Tiers (when stock > EMA 30):
| Profit % | Target Set To | Behavior |
|----------|---------------|----------|
| 35%+ | 10% above entry | Lock in 10% profit floor |
| 50%+ | 20% above entry | Lock in 20% profit floor |
| 75%+ | 35% above entry | Lock in 35% profit floor |
| 100%+ | 50% above entry | Start trailing, 50% floor |
| 125%+ | (1-SL%) × max price | Advanced trailing |
| 200%+ | Immediate sell | Take full profit |

Sell Signals:
- Option price drops below profit target
- Stock price drops below EMA 30 (bearish signal)
- 200% profit reached (immediate sell)

**Classes:**
- `DynamicStopLoss` - Stateful stop loss manager
- `check_stop_loss()` - Stateless function for backtesting
- `TieredProfitExit` - Stateful profit target manager
- `check_profit_target()` - Stateless function for backtesting
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
- Dynamic stop loss with 3-phase strategy
- Technical indicators (EMA_30, VWAP) in tracking data

**Exit Strategies:**
- Dynamic Stop Loss (3-phase: initial → breakeven → trailing)
- Tiered Profit Exit (profit-based: tier_35 → tier_50 → tier_75 → tier_100 → tier_125 → tier_200)
- Market close exit (default fallback)

**Tracking Matrix Columns:**
- timestamp, stock_price, stock_high, stock_low, option_price, volume
- holding, entry_price, pnl, pnl_pct
- stop_loss, stop_loss_mode (initial/breakeven/trailing)
- profit_target, profit_target_mode (initial/tier_35/tier_50/tier_75/tier_100/tier_125/tier_200)
- max_option_price, profit_target_active
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
- Profit Target (blue dotted line with blue shaded area above, right axis)
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
| Strategy.py | Active (Dynamic Stop Loss, Tiered Profit, TestPeakExit) |
| Orders.py | Placeholder |

---

## Next Steps

1. Build out Data.py for data sourcing
2. Add more technical indicators in Analysis.py (MACD, VPOC, SuperTrend)
3. Expand Strategy.py with entry conditions
4. Build Orders.py for live trading
