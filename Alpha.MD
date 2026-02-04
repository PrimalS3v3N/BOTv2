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
| DATA_CONFIG | Data fetching parameters |
| BACKTEST_CONFIG | Backtesting parameters |

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

**Planned Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- VPOC (Volume Point of Control)
- SuperTrend

**Functions:**
- `EMA(df, column='close', period=30)` - Calculate EMA on any column
- `VWAP(df, price_col='close', volume_col='volume')` - Calculate intraday VWAP
- `EWO(df, column='close', fast=5, slow=35)` - Calculate Elliott Wave Oscillator
- `add_indicators(df, ema_period=30)` - Add all standard indicators to DataFrame

**Rules:**
- Pure calculation functions only
- No trading logic here
- Accept DataFrames, return DataFrames with new columns

---

### Strategy.py
**Goal:** Logic for strategies. Process inputs from Data & Analysis, output decision logic.

**Status:** Active

**Implemented Strategies:**
- Dynamic Stop Loss (3-phase exit strategy)
- Tiered Profit Exit (profit-based exit with trailing)

**Dynamic Stop Loss:**
Three-phase stop loss system for options contracts:

1. **INITIAL Phase**: Fixed stop loss at entry price - (stop_loss_pct × entry price)
   - Default: 30% below entry if no stop loss specified in signal
   - Example: Entry $1.00 → Stop at $0.70

2. **BREAKEVEN Phase**: When price rises above threshold, move stop to entry price
   - Threshold = entry_price / (1 - stop_loss_pct)
   - Example: Entry $1.00, 30% SL → threshold = $1.43
   - **Requires minimum 30 minutes held** before transitioning to breakeven

3. **TRAILING Phase**: When price reaches 50% above entry, trail at 30% below highest price
   - Only uses price highs AFTER entry (not before)
   - Stop only moves up, never down

**Tiered Profit Exit:**
Dynamic profit target system based on profit percentage tiers.
Only active when stock price is ABOVE EMA 30.

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

**Rules:**
- Entry logic goes here
- Exit logic goes here
- Accept analysis data, return buy/sell decisions
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

## Current Status

| Module | Status |
|--------|--------|
| Config.py | Active |
| Signal.py | Active |
| Test.py | Active (dynamic stop loss + indicators) |
| Dashboard.py | Active (EMA_30, VWAP, Stop Loss display) |
| Data.py | Placeholder |
| Analysis.py | Active (EMA, VWAP) |
| Strategy.py | Active (Dynamic Stop Loss) |
| Orders.py | Placeholder |

---

## Next Steps

1. Add more exit strategies (profit targets, time stops)
2. Build out Data.py for data sourcing
3. Add more technical indicators in Analysis.py (RSI, MACD, VPOC)
4. Expand Strategy.py with entry conditions
5. Build Orders.py for live trading
