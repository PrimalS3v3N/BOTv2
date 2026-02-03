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
    └── Test.py (backtesting framework)
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

**Status:** Placeholder (to be implemented)

**Planned Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- VPOC (Volume Point of Control)
- VWAP (Volume Weighted Average Price)
- EWO (Elliott Wave Oscillator)
- SuperTrend
- EMA (Exponential Moving Averages)

**Rules:**
- Pure calculation functions only
- No trading logic here
- Accept DataFrames, return DataFrames with new columns

---

### Strategy.py
**Goal:** Logic for strategies. Process inputs from Data & Analysis, output decision logic.

**Status:** Placeholder (to be implemented)

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

**Rules:**
- Exit strategies will be added incrementally
- Currently exits at market close only
- Use Config.py for all parameters

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
| Test.py | Active (market close exit only) |
| Data.py | Placeholder |
| Analysis.py | Placeholder |
| Strategy.py | Placeholder |
| Orders.py | Placeholder |

---

## Next Steps

1. Implement exit strategies in Strategy.py (one at a time)
2. Build out Data.py for data sourcing
3. Implement technical indicators in Analysis.py
4. Connect Strategy.py to Test.py for backtesting
5. Build Orders.py for live trading
