"""
Dashboard.py - Trade Visualization Dashboard

Visualizes backtest trade data from Test.py.
Shows stock price vs option price with entry/exit markers.

Usage:
    streamlit run Dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import Config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'BT_DATA.pkl')

st.set_page_config(page_title="Trade Dashboard", layout="wide")

COLORS = {
    'stock': '#2962FF',
    'option': '#FF6D00',
    'entry': '#00C853',
    'exit': '#FF1744',
    'vwap': '#0D47A1',
    'supertrend': '#9C27B0',
    # EMA gradient: bright (short period) -> dark (long period)
    'ema_10': '#EA80FC',           # Brightest purple (10-period)
    'ema_21': '#CE93D8',           # Light purple (21-period)
    'ema_50': '#AB47BC',           # Medium purple (50-period)
    'ema_100': '#8E24AA',          # Dark purple (100-period)
    'ema_200': '#6A1B9A',          # Darkest purple (200-period)
    'vwap_ema_avg': '#FFEB3B',    # Yellow
    'emavwap': '#00E5FF',         # Cyan
    # ATR Trailing Stoploss
    'atr_sl': '#FF7043',          # Deep Orange
    # Trading range
    'trading_range': '#FF1744',   # Red
    # Ichimoku Cloud
    'ichimoku_tenkan': '#E91E63',   # Pink (Conversion Line)
    'ichimoku_kijun': '#3F51B5',    # Indigo (Base Line)
    'ichimoku_cloud_bull': 'rgba(0, 200, 83, 0.15)',   # Green cloud fill
    'ichimoku_cloud_bear': 'rgba(255, 23, 68, 0.15)',  # Red cloud fill
    'ichimoku_senkou_a': '#26A69A',  # Teal (Leading Span A)
    'ichimoku_senkou_b': '#EF5350',  # Red (Leading Span B)
    # Exit signal markers
    'sig_sb': '#FF9800',           # Orange (StatsBook)
    'sig_mp': '#E040FB',           # Purple (Momentum Peak)
    'sig_ai': '#00BCD4',           # Cyan (AI Exit)
    'sig_closure_peak': '#7C4DFF', # Deep Purple (Closure Peak)
    'sig_mt': '#2196F3',           # Blue (MarketTrend)
}

# Colors for per-trade option price lines (when multiple trades on one chart)
TRADE_OPTION_COLORS = ['#00C853', '#FF6D00', '#2979FF', '#AA00FF', '#FFD600']

# Exit signal definitions: column name -> (display label, color key, marker symbol)
EXIT_SIGNAL_DEFS = {
    'exit_sig_sb':           ('StatsBook',    'sig_sb',           'diamond'),
    'exit_sig_mp':           ('Mom. Peak',    'sig_mp',           'diamond'),
    'exit_sig_ai':           ('AI Exit',      'sig_ai',           'diamond'),
    'exit_sig_closure_peak': ('Closure Peak', 'sig_closure_peak', 'diamond'),
    'exit_sig_mt':           ('MktTrend',     'sig_mt',           'x'),
}

@st.cache_data
def _load_pickle(path, _mtime):
    """Load and cache pickle data. _mtime param busts cache on file change."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('matrices', {}), data.get('statsbooks', {}), data.get('signal_groups', {})


def load_data():
    """Load backtest data from pickle file with Streamlit caching."""
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        mtime = os.path.getmtime(DATA_PATH)
        matrices, statsbooks, signal_groups = _load_pickle(DATA_PATH, mtime)
        st.session_state.statsbooks = statsbooks
        st.session_state.signal_groups = signal_groups
        return matrices
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}


def parse_time(ts):
    """Parse timestamp string."""
    if pd.isna(ts):
        return None
    try:
        return pd.to_datetime(str(ts).replace(' : ', ' '))
    except:
        return pd.to_datetime(ts)


def find_entry_exit(df):
    """Find entry and exit rows using multiple detection methods."""
    opt_col = 'option_price_estimate' if 'option_price_estimate' in df.columns else 'option_price'

    if df.empty:
        return None, None, opt_col

    entry_row = None
    exit_row = None

    # Method 1: Use entry_time/exit_time columns to match timestamps
    if 'entry_time' in df.columns and 'timestamp' in df.columns:
        entry_time_val = df['entry_time'].iloc[0]
        if pd.notna(entry_time_val):
            # Match timestamp to entry_time
            mask = df['timestamp'] == entry_time_val
            if mask.any():
                entry_row = df[mask].iloc[0]

    if 'exit_time' in df.columns and 'timestamp' in df.columns:
        exit_time_val = df['exit_time'].iloc[0]
        if pd.notna(exit_time_val):
            mask = df['timestamp'] == exit_time_val
            if mask.any():
                exit_row = df[mask].iloc[0]

    # Method 2: Fallback - use minutes_held (entry = first row, exit = where position_status changes to 'sold')
    if entry_row is None:
        if 'minutes_held' in df.columns:
            min_held = df['minutes_held'].min()
            entry_row = df[df['minutes_held'] == min_held].iloc[0]
        else:
            entry_row = df.iloc[0]

    if exit_row is None:
        # Try to find exit by exit_reason being set
        if 'exit_reason' in df.columns:
            exit_mask = df['exit_reason'].notna() & (df['exit_reason'] != '')
            if exit_mask.any():
                exit_row = df[exit_mask].iloc[0]

        # Fallback: use position_status == 'sold'
        if exit_row is None and 'position_status' in df.columns:
            sold_mask = df['position_status'] == 'sold'
            if sold_mask.any():
                exit_row = df[sold_mask].iloc[0]

        # Final fallback: last row
        if exit_row is None:
            exit_row = df.iloc[-1]

    return entry_row, exit_row, opt_col


def create_trade_chart(df, trade_label, market_hours_only=False, show_ewo=True, show_rsi=True, show_supertrend=False, show_ichimoku=False, show_atr_sl=True, show_market_trend=True, trades_info=None):
    """Create dual-axis chart with stock/option prices, error bars, EWO/RSI subplot, and SPY subplot.

    Args:
        trades_info: Optional list of dicts with per-trade data for multi-trade signals.
            Each dict has: 'num' (trade number), 'entry_row', 'exit_row', 'opt_col', 'df' (trade DF).
            When None, single-trade behavior is used with markers labeled [1].
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['timestamp'].astype(str).str.replace(' : ', ' '), errors='coerce')
    df = df.dropna(subset=['time'])

    # Filter based on toggle:
    # ON (market_hours_only=True): Show full market hours (9:00 AM - 4:00 PM)
    # OFF (market_hours_only=False): Show only holding period (where holding=True)
    if market_hours_only:
        import datetime as dt
        df = df[
            (df['time'].dt.time >= dt.time(9, 0)) &
            (df['time'].dt.time <= dt.time(16, 0))
        ]
    else:
        # Filter to holding period only
        if 'holding' in df.columns:
            df = df[df['holding'] == True]

    if df.empty:
        return None

    # Build trades_info if not provided (single-trade fallback)
    if trades_info is None:
        entry_row, exit_row, opt_col = find_entry_exit(df)
        trades_info = [{'num': 1, 'entry_row': entry_row, 'exit_row': exit_row, 'opt_col': opt_col, 'df': df}]
    opt_col = trades_info[0]['opt_col']

    # Prepare per-trade data filtered to visible time window
    time_min, time_max = df['time'].min(), df['time'].max()
    for ti in trades_info:
        ti_df = ti['df'].copy()
        ti_df['time'] = pd.to_datetime(ti_df['timestamp'].astype(str).str.replace(' : ', ' '), errors='coerce')
        ti_df = ti_df.dropna(subset=['time'])
        ti_df = ti_df[(ti_df['time'] >= time_min) & (ti_df['time'] <= time_max)]
        ti['_filtered_df'] = ti_df

    # Check if EWO data is available for subplot (and toggle is on)
    has_ewo = show_ewo and 'ewo' in df.columns and df['ewo'].notna().any()

    # Check if RSI data is available (and toggle is on)
    has_rsi = show_rsi and 'rsi' in df.columns and df['rsi'].notna().any()

    # Check if MarketTrend data is available (and toggle is on)
    has_market_trend = show_market_trend and 'ticker_trend' in df.columns and df['ticker_trend'].notna().any()

    # Check if SPY data is available
    has_spy = 'spy_price' in df.columns and df['spy_price'].notna().any()

    # RSI uses secondary y-axis only when EWO is also shown (they share row 2)
    # When EWO is off, RSI uses the primary y-axis to avoid empty primary axis issues
    rsi_secondary = has_ewo

    # Determine number of subplot rows
    has_indicators = has_ewo or has_rsi or has_market_trend

    # Create subplots: main chart + indicators + SPY
    if has_indicators and has_spy:
        # 3 rows: main chart, indicators, SPY
        row2_spec = {"secondary_y": True} if has_ewo else {"secondary_y": False}
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.50, 0.25, 0.25],
            specs=[[{"secondary_y": True}], [row2_spec], [{"secondary_y": True}]]
        )
    elif has_indicators:
        # 2 rows: main chart, indicators
        row2_spec = {"secondary_y": True} if has_ewo else {"secondary_y": False}
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.60, 0.40],
            specs=[[{"secondary_y": True}], [row2_spec]]
        )
    elif has_spy:
        # 2 rows: main chart, SPY
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.60, 0.40],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Option price line (right y-axis) - single line from combined data
    # Same-signal re-entries track the same contract, so one line is correct
    if opt_col in df.columns and df[opt_col].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[opt_col],
                name='Option',
                line=dict(color=COLORS['option'], width=2),
                hovertemplate='Option: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )

    # Stock Price (left y-axis) - blue with high/low error bars
    if 'stock_price' in df.columns and df['stock_price'].notna().any():
        price_trace = go.Scatter(
            x=df['time'],
            y=df['stock_price'],
            name='Price',
            line=dict(color='#2196F3', width=2),
            hovertemplate='Price: $%{y:.2f}<extra></extra>'
        )

        # Add error bars showing high/low range relative to stock price
        if 'stock_high' in df.columns and 'stock_low' in df.columns:
            error_plus = df['stock_high'] - df['stock_price']
            error_minus = df['stock_price'] - df['stock_low']
            price_trace.error_y = dict(
                type='data',
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color='rgba(255, 255, 255, 0.8)',
                thickness=3,
                width=0
            )

        fig.add_trace(price_trace, row=1, col=1, secondary_y=False)

    # VWAP (left y-axis)
    if 'vwap' in df.columns and df['vwap'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['vwap'],
                name='VWAP',
                line=dict(color='#0D47A1', width=2),
                hovertemplate='VWAP: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )

    # EMAs (left y-axis) - purple gradient: bright (short) to dark (long)
    ema_defs = [
        ('ema_10',  'EMA 10',  COLORS['ema_10'],  1.0),
        ('ema_21',  'EMA 21',  COLORS['ema_21'],  1.0),
        ('ema_50',  'EMA 50',  COLORS['ema_50'],  1.5),
        ('ema_100', 'EMA 100', COLORS['ema_100'], 1.5),
        ('ema_200', 'EMA 200', COLORS['ema_200'], 2.0),
    ]
    for ema_col, ema_name, ema_color, ema_width in ema_defs:
        if ema_col in df.columns and df[ema_col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df[ema_col],
                    name=ema_name,
                    line=dict(color=ema_color, width=ema_width, dash='dot'),
                    hovertemplate=f'{ema_name}: $%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

    # MA (left y-axis) - yellow
    if 'vwap_ema_avg' in df.columns and df['vwap_ema_avg'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['vwap_ema_avg'],
                name='MA',
                line=dict(color='#FFEB3B', width=1.5, dash='dashdot'),
                hovertemplate='MA: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )

    # EMAVWAP (left y-axis) - cyan
    if 'emavwap' in df.columns and df['emavwap'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['emavwap'],
                name='EMAVWAP',
                line=dict(color=COLORS['emavwap'], width=1.5, dash='dash'),
                hovertemplate='EMAVWAP: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )

    # Supertrend (left y-axis) - color-coded by direction
    if show_supertrend and 'supertrend' in df.columns and df['supertrend'].notna().any():
        if 'supertrend_direction' in df.columns:
            # Use full time axis with NaN for non-matching direction so connectgaps
            # breaks lines at direction transitions instead of drawing across gaps
            st_bull = df['supertrend'].where(df['supertrend_direction'] == 1)
            st_bear = df['supertrend'].where(df['supertrend_direction'] == -1)

            if st_bull.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df['time'],
                        y=st_bull,
                        name='ST (Bull)',
                        mode='markers+lines',
                        line=dict(color='#00C853', width=2),
                        marker=dict(size=3, color='#00C853'),
                        connectgaps=False,
                        hovertemplate='Supertrend: $%{y:.2f} (Bull)<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False
                )
            if st_bear.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df['time'],
                        y=st_bear,
                        name='ST (Bear)',
                        mode='markers+lines',
                        line=dict(color='#FF1744', width=2),
                        marker=dict(size=3, color='#FF1744'),
                        connectgaps=False,
                        hovertemplate='Supertrend: $%{y:.2f} (Bear)<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False
                )
        else:
            # Fallback: single color if no direction data
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['supertrend'],
                    name='Supertrend',
                    line=dict(color=COLORS['supertrend'], width=2),
                    hovertemplate='Supertrend: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

    # ATR Trailing Stoploss (left y-axis)
    if show_atr_sl and 'atr_sl' in df.columns and df['atr_sl'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['atr_sl'],
                name='ATR-SL',
                line=dict(color=COLORS['atr_sl'], width=2, dash='dash'),
                hovertemplate='ATR-SL: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )

        # ATR-SL Trend Gate shading: color background based on trend state
        if 'atr_sl_trend' in df.columns and df['atr_sl_trend'].notna().any():
            trend_data = df[['time', 'atr_sl_trend']].dropna(subset=['atr_sl_trend']).copy()
            if not trend_data.empty:
                trend_data['group'] = (trend_data['atr_sl_trend'] != trend_data['atr_sl_trend'].shift()).cumsum()
                trend_bounds = trend_data.groupby('group').agg(
                    x0=('time', 'first'),
                    x1=('time', 'last'),
                    val=('atr_sl_trend', 'first')
                )
                trend_colors = {
                    'Uptrend': 'rgba(0, 200, 83, 0.06)',     # Faint green
                    'Downtrend': 'rgba(255, 23, 68, 0.06)',   # Faint red
                    'Sideways': 'rgba(255, 235, 59, 0.06)',   # Faint yellow
                }
                shapes = list(fig.layout.shapes or [])
                for _, row_data in trend_bounds.iterrows():
                    color = trend_colors.get(row_data['val'])
                    if color:
                        shapes.append(dict(
                            type='rect', xref='x', yref='paper',
                            x0=row_data['x0'], x1=row_data['x1'], y0=0, y1=1,
                            fillcolor=color, layer='below', line_width=0,
                        ))
                fig.update_layout(shapes=shapes)

    # Ichimoku Cloud (left y-axis) - cloud fill between Senkou spans + Tenkan/Kijun lines
    if show_ichimoku:
        has_senkou_a = 'ichimoku_senkou_a' in df.columns and df['ichimoku_senkou_a'].notna().any()
        has_senkou_b = 'ichimoku_senkou_b' in df.columns and df['ichimoku_senkou_b'].notna().any()
        has_tenkan = 'ichimoku_tenkan' in df.columns and df['ichimoku_tenkan'].notna().any()
        has_kijun = 'ichimoku_kijun' in df.columns and df['ichimoku_kijun'].notna().any()

        # Cloud fill: Senkou Span A and B with shaded region between
        if has_senkou_a and has_senkou_b:
            # Senkou Span A (upper/lower boundary of cloud)
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ichimoku_senkou_a'],
                    name='Senkou A',
                    line=dict(color=COLORS['ichimoku_senkou_a'], width=1),
                    hovertemplate='Senkou A: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

            # Senkou Span B with fill to Senkou A
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ichimoku_senkou_b'],
                    name='Senkou B',
                    line=dict(color=COLORS['ichimoku_senkou_b'], width=1),
                    fill='tonexty',
                    fillcolor=COLORS['ichimoku_cloud_bull'],
                    hovertemplate='Senkou B: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

        # Tenkan-sen (Conversion Line)
        if has_tenkan:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ichimoku_tenkan'],
                    name='Tenkan',
                    line=dict(color=COLORS['ichimoku_tenkan'], width=1.5),
                    hovertemplate='Tenkan: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

        # Kijun-sen (Base Line)
        if has_kijun:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ichimoku_kijun'],
                    name='Kijun',
                    line=dict(color=COLORS['ichimoku_kijun'], width=1.5, dash='dash'),
                    hovertemplate='Kijun: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )

    # Entry/Exit markers per trade
    for ti in trades_info:
        t_num = ti['num']
        t_opt_col = ti['opt_col']
        entry_row = ti['entry_row']
        exit_row = ti['exit_row']
        prefix = f'[{t_num}]'

        if entry_row is not None:
            entry_time = parse_time(entry_row['timestamp'])
            entry_price = entry_row[t_opt_col] if t_opt_col in entry_row else entry_row['stock_price']
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode='markers',
                    name=f'{prefix} : Entry',
                    marker=dict(symbol='triangle-down', size=18, color=COLORS['entry'], line=dict(color='white', width=2)),
                    hovertemplate=f'{prefix} ENTRY<br>$%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1, secondary_y=True
            )

        if exit_row is not None:
            exit_time = parse_time(exit_row['timestamp'])
            exit_price = exit_row[t_opt_col] if t_opt_col in exit_row else exit_row['stock_price']
            exit_reason = exit_row.get('exit_reason', 'unknown') if hasattr(exit_row, 'get') else exit_row['exit_reason'] if 'exit_reason' in exit_row.index else 'unknown'
            if pd.isna(exit_reason):
                exit_reason = 'unknown'
            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[exit_price],
                    mode='markers+text',
                    name=f'{prefix} : Exit ({exit_reason})',
                    marker=dict(symbol='triangle-up', size=18, color=COLORS['exit'], line=dict(color='white', width=2)),
                    text=[str(exit_reason)],
                    textposition='top center',
                    textfont=dict(size=10, color=COLORS['exit']),
                    hovertemplate=f'{prefix} EXIT: {exit_reason}<br>$%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1, secondary_y=True
            )

    # Exit signal markers on main chart (right y-axis, aligned with option price)
    # Compute opt_max/min across all trades for signal marker positioning
    _opt_prices = []
    for ti in trades_info:
        t_df = ti['_filtered_df']
        t_opt_col = ti['opt_col']
        if t_opt_col in t_df.columns:
            _opt_prices.append(t_df[t_opt_col].dropna())
    if _opt_prices:
        _all_opts = pd.concat(_opt_prices)
        opt_max = _all_opts.max() if not _all_opts.empty else 1
        opt_min = _all_opts.min() if not _all_opts.empty else 0
        opt_range = opt_max - opt_min if opt_max > opt_min else opt_max * 0.1
        # Stack signals vertically above the chart area
        sig_offset_step = opt_range * 0.04

        for sig_idx, (sig_col, (sig_label, sig_color_key, sig_symbol)) in enumerate(EXIT_SIGNAL_DEFS.items()):
            if sig_col in df.columns and df[sig_col].notna().any():
                sig_mask = df[sig_col] == True
                if sig_mask.any():
                    sig_y = opt_max + sig_offset_step * (sig_idx + 1)
                    fig.add_trace(
                        go.Scatter(
                            x=df.loc[sig_mask, 'time'],
                            y=[sig_y] * sig_mask.sum(),
                            mode='markers',
                            name=f'Sig: {sig_label}',
                            marker=dict(
                                symbol=sig_symbol,
                                size=8,
                                color=COLORS[sig_color_key],
                                line=dict(color='white', width=0.5),
                            ),
                            hovertemplate=f'{sig_label} signal<br>%{{x}}<extra></extra>',
                            legendgroup='exit_signals',
                            visible='legendonly',
                        ),
                        row=1, col=1, secondary_y=True
                    )

    # Gauge sentiment as background shading + hover on main chart
    # Use ticker gauge if available, otherwise fall back to SPY gauge
    ticker_gauge_cols = {
        'ticker_since_open': 'O',
        'ticker_1m': '1m',
        'ticker_5m': '5m',
        'ticker_15m': '15m',
        'ticker_30m': '30m',
        'ticker_1h': '1h',
    }
    spy_gauge_fallback_cols = {
        'spy_since_open': 'O',
        'spy_1m': '1m',
        'spy_5m': '5m',
        'spy_15m': '15m',
        'spy_30m': '30m',
        'spy_1h': '1h',
    }
    has_ticker_gauge = any(col in df.columns and df[col].notna().any() for col in ticker_gauge_cols)
    has_spy_fallback = any(col in df.columns and df[col].notna().any() for col in spy_gauge_fallback_cols)

    # Pick which gauge data to use for main chart shading
    if has_ticker_gauge:
        shading_col = 'ticker_5m'
        hover_gauge_cols = ticker_gauge_cols
    elif has_spy_fallback:
        shading_col = 'spy_5m'
        hover_gauge_cols = spy_gauge_fallback_cols
    else:
        shading_col = None
        hover_gauge_cols = {}

    if shading_col and shading_col in df.columns and df[shading_col].notna().any():
        # Show the 5m gauge as background shading for quick visual reference
        # Build vrect shapes in bulk instead of per-group add_vrect calls
        sentiment = df[['time', shading_col]].dropna(subset=[shading_col]).copy()
        if not sentiment.empty:
            sentiment['group'] = (sentiment[shading_col] != sentiment[shading_col].shift()).cumsum()
            grp_bounds = sentiment.groupby('group').agg(
                x0=('time', 'first'),
                x1=('time', 'last'),
                val=(shading_col, 'first')
            )
            shapes = []
            for _, row in grp_bounds.iterrows():
                shapes.append(dict(
                    type='rect', xref='x', yref='paper',
                    x0=row['x0'], x1=row['x1'], y0=0, y1=1,
                    fillcolor='rgba(0, 200, 83, 0.10)' if row['val'] == 'Bullish' else 'rgba(255, 23, 68, 0.10)',
                    layer='below', line_width=0,
                ))
            fig.update_layout(shapes=list(fig.layout.shapes or []) + shapes)

    if hover_gauge_cols:
        # Build hover text using vectorized string ops instead of row-by-row .apply()
        hover_parts = []
        for col, label in hover_gauge_cols.items():
            if col in df.columns:
                series = df[col]
                part = series.map({'Bullish': f'{label}[+]', 'Bearish': f'{label}[-]'}).fillna('')
                hover_parts.append(part)
        if hover_parts:
            hover_texts = hover_parts[0].str.cat(hover_parts[1:], sep=' ').str.strip()
        else:
            hover_texts = pd.Series('', index=df.index)
        if hover_texts.str.len().sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['stock_price'],
                    name='Ticker Gauge',
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts,
                    showlegend=False,
                ),
                row=1, col=1, secondary_y=False
            )

    # EWO subplot (row 2) — displayed as histogram bars
    if has_ewo:
        # Vectorized color assignment using numpy
        ewo_colors = np.where(
            df['ewo'].values >= 0,
            'rgba(0, 200, 83, 0.7)',
            'rgba(255, 23, 68, 0.7)'
        )

        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['ewo'],
                name='EWO',
                marker_color=ewo_colors,
                hovertemplate='EWO: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )

    # RSI on row 2 (secondary y-axis when sharing with EWO, primary when alone)
    if has_rsi:
        # RSI line in white
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='white', width=1.5),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1, secondary_y=rsi_secondary
        )

    # Avg(EWO) 15-min average line
    if has_ewo and 'ewo_15min_avg' in df.columns and df['ewo_15min_avg'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ewo_15min_avg'],
                name='Avg(EWO)',
                line=dict(color='#FF9800', width=2, dash='dot'),
                hovertemplate='Avg(EWO): %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )

    # Avg(RSI) 10-min average line
    if has_rsi and 'rsi_10min_avg' in df.columns and df['rsi_10min_avg'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['rsi_10min_avg'],
                name='Avg(RSI)',
                line=dict(color='#FF9800', width=2, dash='dot'),
                hovertemplate='Avg(RSI): %{y:.1f}<extra></extra>'
            ),
            row=2, col=1, secondary_y=rsi_secondary
        )

    # EWO zero line
    if has_ewo:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    # RSI reference lines
    if has_rsi:
        # Use scatter traces for reference lines since add_hline doesn't support secondary_y
        time_range = [df['time'].iloc[0], df['time'].iloc[-1]]

        # Overbought line (70)
        fig.add_trace(
            go.Scatter(
                x=time_range, y=[70, 70],
                mode='lines',
                line=dict(color='rgba(255, 82, 82, 0.7)', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ),
            row=2, col=1, secondary_y=rsi_secondary
        )

        # Oversold line (30)
        fig.add_trace(
            go.Scatter(
                x=time_range, y=[30, 30],
                mode='lines',
                line=dict(color='rgba(105, 240, 174, 0.7)', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ),
            row=2, col=1, secondary_y=rsi_secondary
        )

        # Neutral line (50)
        fig.add_trace(
            go.Scatter(
                x=time_range, y=[50, 50],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ),
            row=2, col=1, secondary_y=rsi_secondary
        )

    # MarketTrend on row 2 primary y-axis (color-coded step fill: green=bull, red=bear)
    if has_market_trend and (has_ewo or has_rsi or has_market_trend):
        trend_vals = df['ticker_trend'].copy()

        # Bullish segments (+1) with green fill from 0 to +1
        trend_bull = trend_vals.where(trend_vals >= 1)
        if trend_bull.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'], y=trend_bull,
                    name='Trend (Bull)',
                    mode='lines',
                    line=dict(color='#00C853', width=2, shape='hv'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 200, 83, 0.12)',
                    connectgaps=False,
                    hovertemplate='Trend: +1 (Bull)<extra></extra>'
                ),
                row=2, col=1
            )

        # Bearish segments (-1) with red fill from 0 to -1
        trend_bear = trend_vals.where(trend_vals <= -1)
        if trend_bear.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'], y=trend_bear,
                    name='Trend (Bear)',
                    mode='lines',
                    line=dict(color='#FF1744', width=2, shape='hv'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 23, 68, 0.12)',
                    connectgaps=False,
                    hovertemplate='Trend: -1 (Bear)<extra></extra>'
                ),
                row=2, col=1
            )

        # Reference lines at +1 and -1
        time_range = [df['time'].iloc[0], df['time'].iloc[-1]]
        fig.add_trace(
            go.Scatter(
                x=time_range, y=[1, 1],
                mode='lines',
                line=dict(color='rgba(0, 200, 83, 0.4)', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=time_range, y=[-1, -1],
                mode='lines',
                line=dict(color='rgba(255, 23, 68, 0.4)', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ),
            row=2, col=1
        )

    # SPY subplot (row 3 if indicators present, row 2 if no indicators)
    spy_row = 3 if has_indicators and has_spy else (2 if has_spy else None)
    if has_spy and spy_row:
        # SPY price line
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['spy_price'],
                name='SPY',
                line=dict(color='#FFD700', width=2),
                hovertemplate='SPY: $%{y:.2f}<extra></extra>'
            ),
            row=spy_row, col=1, secondary_y=False
        )

        # Cumulative P&L on secondary y-axis of SPY subplot
        if 'pnl' in df.columns and df['pnl'].notna().any():
            holding_mask = df['holding'] == True if 'holding' in df.columns else pd.Series(True, index=df.index)
            pnl_data = df.loc[holding_mask, ['time', 'pnl']].dropna()
            pnl_data['pnl'] = pnl_data['pnl'].round(2)
            if not pnl_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pnl_data['time'],
                        y=pnl_data['pnl'],
                        name='P&L ($)',
                        line=dict(color='#00E5FF', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 229, 255, 0.08)',
                        hovertemplate='P&L: $%{y:+.2f}<extra></extra>'
                    ),
                    row=spy_row, col=1, secondary_y=True
                )

        # SPY gauge sentiment as color-coded markers on secondary axis
        # Map gauge columns to display
        spy_gauge_cols = {
            'spy_since_open': 'O',
            'spy_1m': '1m',
            'spy_5m': '5m',
            'spy_15m': '15m',
            'spy_30m': '30m',
            'spy_1h': '1h',
        }

        # Show the 5m gauge as background shading for quick visual reference
        # Build vrect shapes in bulk for SPY subplot
        if 'spy_5m' in df.columns and df['spy_5m'].notna().any():
            sentiment = df[['time', 'spy_5m']].dropna(subset=['spy_5m']).copy()
            if not sentiment.empty:
                sentiment['group'] = (sentiment['spy_5m'] != sentiment['spy_5m'].shift()).cumsum()
                grp_bounds = sentiment.groupby('group').agg(
                    x0=('time', 'first'),
                    x1=('time', 'last'),
                    val=('spy_5m', 'first')
                )
                # SPY subplot uses xref/yref for the correct subplot axis
                x_axis = f'x{spy_row}' if spy_row > 1 else 'x'
                y_axis = f'y{(spy_row - 1) * 2 + 1}' if spy_row > 1 else 'y'
                shapes = []
                for _, row in grp_bounds.iterrows():
                    shapes.append(dict(
                        type='rect', xref=x_axis, yref='paper',
                        x0=row['x0'], x1=row['x1'], y0=0, y1=1,
                        fillcolor='rgba(0, 200, 83, 0.10)' if row['val'] == 'Bullish' else 'rgba(255, 23, 68, 0.10)',
                        layer='below', line_width=0,
                    ))
                fig.update_layout(shapes=list(fig.layout.shapes or []) + shapes)

        # Build hover text using vectorized string ops
        spy_mask = df['spy_price'].notna()
        spy_price_text = pd.Series("SPY: N/A", index=df.index)
        if spy_mask.any():
            spy_price_text[spy_mask] = 'SPY: $' + df.loc[spy_mask, 'spy_price'].round(2).astype(str)
        gauge_parts = []
        for col, label in spy_gauge_cols.items():
            if col in df.columns:
                part = df[col].map({'Bullish': f'{label}[+]', 'Bearish': f'{label}[-]'}).fillna('')
                gauge_parts.append(part)
        if gauge_parts:
            gauge_text = gauge_parts[0].str.cat(gauge_parts[1:], sep=' ').str.strip()
        else:
            gauge_text = pd.Series('', index=df.index)
        # Add SPY trend label to hover
        spy_trend_text = pd.Series('', index=df.index)
        if 'spy_trend' in df.columns:
            spy_trend_text = df['spy_trend'].map({1: 'SPY Trend: Bull', 0: 'SPY Trend: Side', -1: 'SPY Trend: Bear'}).fillna('')
        hover_texts = spy_price_text + '<br>' + spy_trend_text + '<br>' + gauge_text
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['spy_price'],
                name='SPY Gauge',
                mode='markers',
                marker=dict(size=0, opacity=0),
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts,
                showlegend=False,
            ),
            row=spy_row, col=1, secondary_y=False
        )

    # Layout
    if has_indicators and has_spy:
        chart_height = 1000
    elif has_indicators or has_spy:
        chart_height = 800
    else:
        chart_height = 500

    fig.update_layout(
        template='plotly_dark',
        height=chart_height,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.05,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
        ),
        margin=dict(l=60, r=180, t=30, b=40)
    )

    # Update axes
    fig.update_xaxes(title_text="Time", tickformat='%H:%M:%S', row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, tickformat='$.2f', row=1, col=1)
    fig.update_yaxes(title_text="Option ($)", secondary_y=True, tickformat='$.2f', row=1, col=1)

    if has_indicators:
        fig.update_xaxes(title_text="Time", tickformat='%H:%M:%S', row=2, col=1)

    if has_ewo:
        ewo_title = "EWO / Trend" if has_market_trend else "EWO"
        fig.update_yaxes(title_text=ewo_title, secondary_y=False, row=2, col=1)
    elif has_market_trend and not has_rsi:
        fig.update_yaxes(
            title_text="Trend", secondary_y=False, row=2, col=1,
            range=[-1.5, 1.5], tickvals=[-1, 0, 1], ticktext=['Bear', 'Side', 'Bull']
        )

    if has_rsi:
        fig.update_yaxes(
            title_text="RSI", secondary_y=rsi_secondary, row=2, col=1,
            range=[0, 100], tickvals=[0, 30, 50, 70, 100]
        )

    if has_spy and spy_row:
        fig.update_xaxes(title_text="Time", tickformat='%H:%M:%S', row=spy_row, col=1)
        # Set explicit y-axis range for tight fit around price data
        spy_prices = df['spy_price'].dropna()
        if not spy_prices.empty:
            spy_min = spy_prices.min()
            spy_max = spy_prices.max()
            spy_padding = (spy_max - spy_min) * 0.05 if spy_max > spy_min else 0.5
            fig.update_yaxes(title_text="SPY ($)", secondary_y=False, tickformat='$.2f',
                             range=[spy_min - spy_padding, spy_max + spy_padding],
                             autorange=False, row=spy_row, col=1)
        else:
            fig.update_yaxes(title_text="SPY ($)", secondary_y=False, tickformat='$.2f', row=spy_row, col=1)

        # P&L secondary y-axis label
        fig.update_yaxes(title_text="P&L ($)", secondary_y=True, tickformat='$+.2f', row=spy_row, col=1)

    return fig


@st.cache_data
def get_trade_summary(df):
    """Extract concise trade summary including max/min profit percentages."""
    if df.empty:
        return {
            'entry': 0, 'exit': 0, 'pnl_pct': 0, 'exit_reason': 'N/A',
            'duration': 0, 'max_price': 0, 'min_price': 0,
            'max_profit_pct': 0, 'min_profit_pct': 0, 'profit_min': 0
        }

    entry_row, exit_row, opt_col = find_entry_exit(df)

    entry_price = entry_row[opt_col] if entry_row is not None and opt_col in entry_row else 0
    exit_price = exit_row[opt_col] if exit_row is not None and opt_col in exit_row else 0

    if exit_row is not None and 'exit_reason' in exit_row.index:
        exit_reason = exit_row['exit_reason']
        if pd.isna(exit_reason):
            exit_reason = 'N/A'
    else:
        exit_reason = 'N/A'

    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    duration = df['minutes_held'].max() if 'minutes_held' in df.columns else 0

    # Compute holding_df once and reuse for all holding-period metrics
    TREND_LABELS = {1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}
    max_price = 0
    min_price = 0
    max_profit_pct = 0
    min_profit_pct = 0
    profit_min = 0
    market_trend_label = 'N/A'
    risk = 'N/A'
    risk_reasons = ''
    risk_trend = ''
    exit_signal_counts = {}

    holding_df = df[df['holding'] == True] if 'holding' in df.columns else pd.DataFrame()

    if not holding_df.empty:
        # Max/min contract prices during holding
        if opt_col in holding_df.columns and holding_df[opt_col].notna().any():
            max_price = holding_df[opt_col].max()
            min_price = holding_df[opt_col].min()
            if entry_price > 0:
                max_profit_pct = (max_price / entry_price - 1) * 100
                min_profit_pct = (min_price / entry_price - 1) * 100
                contracts = int(df['contracts'].iloc[0]) if 'contracts' in df.columns else 1
                profit_min = (min_price - entry_price) * contracts * 100

        # MarketTrend at exit
        if 'ticker_trend' in holding_df.columns:
            last_trend = holding_df['ticker_trend'].iloc[-1]
            if pd.notna(last_trend):
                market_trend_label = TREND_LABELS.get(int(last_trend), 'N/A')

        # Risk level at entry
        if 'risk' in holding_df.columns:
            first_risk = holding_df['risk'].iloc[0]
            if pd.notna(first_risk):
                risk = str(first_risk)
            if 'risk_reasons' in holding_df.columns:
                first_reasons = holding_df['risk_reasons'].iloc[0]
                if pd.notna(first_reasons) and first_reasons:
                    risk_reasons = str(first_reasons)
            if 'risk_trend' in holding_df.columns:
                last_trend = holding_df['risk_trend'].iloc[-1]
                if pd.notna(last_trend):
                    risk_trend = str(last_trend)

        # Count exit signal activations during holding period
        for sig_col, (sig_label, _, _) in EXIT_SIGNAL_DEFS.items():
            if sig_col in holding_df.columns:
                count = int((holding_df[sig_col] == True).sum())
                exit_signal_counts[sig_label] = count

    # ATR-SL trend gate tracking
    atr_sl_trend_at_exit = 'N/A'
    trend_gate_bars = 0
    if not holding_df.empty:
        if 'atr_sl_trend' in holding_df.columns and holding_df['atr_sl_trend'].notna().any():
            atr_sl_trend_at_exit = str(holding_df['atr_sl_trend'].iloc[-1])
        if 'atr_sl_trend_gate' in holding_df.columns:
            trend_gate_bars = int((holding_df['atr_sl_trend_gate'] == True).sum())

    # Trade number (for re-entry tracking)
    trade_number = int(df['trade_number'].iloc[0]) if 'trade_number' in df.columns and df['trade_number'].notna().any() else 1

    return {
        'entry': entry_price,
        'exit': exit_price,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'duration': duration,
        'max_price': max_price,
        'min_price': min_price,
        'max_profit_pct': max_profit_pct,
        'min_profit_pct': min_profit_pct,
        'profit_min': profit_min,
        'market_trend': market_trend_label,
        'risk': risk,
        'risk_reasons': risk_reasons,
        'risk_trend': risk_trend,
        'exit_signal_counts': exit_signal_counts,
        'atr_sl_trend': atr_sl_trend_at_exit,
        'trend_gate_bars': trend_gate_bars,
        'trade_number': trade_number,
    }


@st.cache_data
def get_trade_table(df):
    """Create a concise data table for the trade."""
    if df.empty:
        return pd.DataFrame({'Metric': ['No data'], 'Value': ['Empty trade']})
    entry_row, exit_row, opt_col = find_entry_exit(df)

    # Build table data
    table_data = {
        'Metric': [],
        'Value': []
    }

    # Ticker info
    if 'ticker' in df.columns:
        table_data['Metric'].append('Ticker')
        table_data['Value'].append(df['ticker'].iloc[0])

    if 'strike' in df.columns:
        table_data['Metric'].append('Strike')
        table_data['Value'].append(f"${df['strike'].iloc[0]:.2f}")

    if 'option_type' in df.columns:
        table_data['Metric'].append('Type')
        table_data['Value'].append(df['option_type'].iloc[0])

    if 'contracts' in df.columns:
        table_data['Metric'].append('Contracts')
        table_data['Value'].append(str(int(df['contracts'].iloc[0])))

    # Entry info
    if entry_row is not None:
        table_data['Metric'].append('Entry Time')
        table_data['Value'].append(str(entry_row['timestamp']))
        table_data['Metric'].append('Entry Price')
        table_data['Value'].append(f"${entry_row[opt_col]:.2f}")
        table_data['Metric'].append('Stock @ Entry')
        table_data['Value'].append(f"${entry_row['stock_price']:.2f}")

    # Exit info
    if exit_row is not None:
        table_data['Metric'].append('Exit Time')
        table_data['Value'].append(str(exit_row['timestamp']))
        table_data['Metric'].append('Exit Price')
        table_data['Value'].append(f"${exit_row[opt_col]:.2f}")
        table_data['Metric'].append('Stock @ Exit')
        table_data['Value'].append(f"${exit_row['stock_price']:.2f}")
        if 'exit_reason' in exit_row.index and pd.notna(exit_row['exit_reason']):
            table_data['Metric'].append('Exit Reason')
            table_data['Value'].append(str(exit_row['exit_reason']))

    # P&L
    if entry_row is not None and exit_row is not None:
        entry_price = entry_row[opt_col]
        exit_price = exit_row[opt_col]
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
        contracts = int(df['contracts'].iloc[0]) if 'contracts' in df.columns else 1
        pnl_dollars = pnl * contracts * 100

        table_data['Metric'].append('P&L')
        table_data['Value'].append(f"${pnl_dollars:+.2f} ({pnl_pct:+.1f}%)")

    # Duration
    if 'minutes_held' in df.columns:
        duration = df['minutes_held'].max()
        table_data['Metric'].append('Duration')
        table_data['Value'].append(f"{duration:.0f} min")

    # Max/Min contract prices after entry
    if 'holding' in df.columns and opt_col in df.columns:
        holding_df = df[df['holding'] == True]
        if not holding_df.empty and holding_df[opt_col].notna().any():
            max_price = holding_df[opt_col].max()
            min_price = holding_df[opt_col].min()
            table_data['Metric'].append('Contract High')
            table_data['Value'].append(f"${max_price:.2f}")
            table_data['Metric'].append('Contract Low')
            table_data['Value'].append(f"${min_price:.2f}")

    return pd.DataFrame(table_data)


def main():
    matrices = load_data()

    if not matrices:
        st.warning("No backtest data found. Run Test.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Trades")

        # Compute all trade summaries once and reuse
        summaries = {}
        total_profit_dollars = 0.0
        total_investment = 0.0
        potential_profit_dollars = 0.0

        # Group trades by signal_group_id for multi-trade signals
        # signal_group_map: group_id -> list of (pos_id, tdf, summary)
        signal_group_map = {}
        ungrouped = []

        for pos_id, tdf in matrices.items():
            if tdf.empty:
                continue

            summary = get_trade_summary(tdf)
            summaries[pos_id] = summary

            # Accumulate totals in the same pass
            entry_p = summary['entry']
            exit_p = summary['exit']
            max_p = summary['max_price']
            contracts = int(tdf['contracts'].iloc[0]) if 'contracts' in tdf.columns else 1
            if entry_p > 0:
                inv = entry_p * contracts * 100
                total_investment += inv
                total_profit_dollars += (exit_p - entry_p) * contracts * 100
                potential_profit_dollars += (max_p - entry_p) * contracts * 100

            # Check for signal_group_id
            group_id = None
            if 'signal_group_id' in tdf.columns and tdf['signal_group_id'].notna().any():
                group_id = tdf['signal_group_id'].iloc[0]

            if group_id:
                if group_id not in signal_group_map:
                    signal_group_map[group_id] = []
                signal_group_map[group_id].append((pos_id, tdf, summary))
            else:
                ungrouped.append((pos_id, tdf, summary))

        # Build trade_list: each entry is (label, pos_ids_list, total_pnl)
        # For grouped signals: single entry with combined P&L
        # For ungrouped: single entry per trade
        trade_list = []

        for group_id, trades in signal_group_map.items():
            # Sort trades by trade_number within group
            trades.sort(key=lambda t: int(t[1]['trade_number'].iloc[0]) if 'trade_number' in t[1].columns and t[1]['trade_number'].notna().any() else 1)
            first_tdf = trades[0][1]
            ticker = first_tdf['ticker'].iloc[0] if 'ticker' in first_tdf.columns else 'UNK'
            strike = first_tdf['strike'].iloc[0] if 'strike' in first_tdf.columns else 0
            opt_type = first_tdf['option_type'].iloc[0] if 'option_type' in first_tdf.columns else '?'
            cp = opt_type[0].upper() if opt_type else '?'

            pos_ids = [t[0] for t in trades]

            if len(trades) > 1:
                # Multi-trade signal: compute total P&L across all trades
                total_pnl_dollars = 0.0
                total_entry_cost = 0.0
                for _, tdf, s in trades:
                    contracts = int(tdf['contracts'].iloc[0]) if 'contracts' in tdf.columns else 1
                    entry_cost = s['entry'] * contracts * 100
                    total_entry_cost += entry_cost
                    total_pnl_dollars += (s['exit'] - s['entry']) * contracts * 100
                total_pnl_pct = (total_pnl_dollars / total_entry_cost * 100) if total_entry_cost > 0 else 0
                label = f"{ticker} : {strike:.0f} : {cp}"
                trade_list.append((label, pos_ids, total_pnl_pct))
            else:
                # Single trade in group
                label = f"{ticker} : {strike:.0f} : {cp}"
                pnl = trades[0][2]['pnl_pct']
                trade_list.append((label, pos_ids, pnl))

        for pos_id, tdf, summary in ungrouped:
            ticker = tdf['ticker'].iloc[0] if 'ticker' in tdf.columns else 'UNK'
            strike = tdf['strike'].iloc[0] if 'strike' in tdf.columns else 0
            opt_type = tdf['option_type'].iloc[0] if 'option_type' in tdf.columns else '?'
            cp = opt_type[0].upper() if opt_type else '?'
            label = f"{ticker} : {strike:.0f} : {cp}"
            pnl = summary['pnl_pct']
            trade_list.append((label, [pos_id], pnl))

        # Sort by P&L descending (winners first, losers last)
        trade_list.sort(key=lambda x: x[2], reverse=True)

        if not trade_list:
            st.warning("All trades have empty data.")
            return

        selected_idx = st.selectbox(
            "Select Trade",
            range(len(trade_list)),
            format_func=lambda i: f"{trade_list[i][0]} ({trade_list[i][2]:+.1f}%)"
        )

        if st.button("Reload Data"):
            st.session_state.pop('statsbooks', None)
            st.session_state.pop('signal_groups', None)
            _load_pickle.clear()
            get_trade_summary.clear()
            get_trade_table.clear()
            st.rerun()

        st.markdown("---")
        market_hours_only = st.toggle("Market Hours", value=True, help="ON: Full market hours view | OFF: Holding period only")
        show_ewo = st.toggle("Show EWO Graph", value=True)
        show_rsi = st.toggle("Show RSI Graph", value=True)
        show_supertrend = st.toggle("Show Supertrend", value=False, help="Overlay Supertrend indicator on main chart")
        show_ichimoku = st.toggle("Show Ichimoku Cloud", value=False, help="Overlay Ichimoku Cloud (Tenkan, Kijun, Senkou spans) on main chart")
        show_atr_sl = st.toggle("Show ATR-SL", value=True, help="Overlay ATR Trailing Stoploss indicator on main chart")
        show_market_trend = st.toggle("Show Market Trend", value=True, help="MarketTrend on indicator subplot: +1 Bull, 0 Side, -1 Bear")
        show_trade_summary = st.toggle("Show Trade Summary", value=False, help="Display trade summary details")
        show_stats_book = st.toggle("Show Stats Book", value=False, help="Display statistics book")
        show_data_book = st.toggle("Show Data Book", value=False, help="Display data book")

        st.markdown("---")

        total_profit_pct = (total_profit_dollars / total_investment * 100) if total_investment > 0 else 0
        potential_profit_pct = (potential_profit_dollars / total_investment * 100) if total_investment > 0 else 0

        st.markdown(f"**Total Profit:** ${total_profit_dollars:,.2f} ({total_profit_pct:+.1f}%)")
        st.markdown(f"**Potential Profit:** ${potential_profit_dollars:,.2f} ({potential_profit_pct:+.1f}%)")

        st.markdown("---")
        st.caption(f"{len(matrices)} trades loaded")

    # Display selected trade(s) — all trades consolidated on one chart
    trade_label, pos_ids, _ = trade_list[selected_idx]
    is_multi_trade = len(pos_ids) > 1

    # Build per-trade info and combined DF for consolidated chart
    trade_dfs = []
    trades_info_list = []
    for trade_idx, pos_id in enumerate(pos_ids):
        tdf = matrices[pos_id]
        trade_num = int(tdf['trade_number'].iloc[0]) if 'trade_number' in tdf.columns and tdf['trade_number'].notna().any() else trade_idx + 1
        entry_row, exit_row, t_opt_col = find_entry_exit(tdf)
        trade_dfs.append(tdf)
        trades_info_list.append({
            'num': trade_num,
            'entry_row': entry_row,
            'exit_row': exit_row,
            'opt_col': t_opt_col,
            'df': tdf,
        })

    # Build combined DF: first trade as base, extend with unique timestamps from others
    combined_df = trade_dfs[0].copy()
    if len(trade_dfs) > 1:
        any_holding_ts = set()
        for tdf in trade_dfs:
            if 'holding' in tdf.columns:
                any_holding_ts.update(tdf.loc[tdf['holding'] == True, 'timestamp'].astype(str))
        for extra_df in trade_dfs[1:]:
            existing_ts = set(combined_df['timestamp'].astype(str))
            new_rows = extra_df[~extra_df['timestamp'].astype(str).isin(existing_ts)]
            if not new_rows.empty:
                combined_df = pd.concat([combined_df, new_rows], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        if 'holding' in combined_df.columns:
            combined_df['holding'] = combined_df['timestamp'].astype(str).isin(any_holding_ts)

    # Single consolidated chart for all trades
    fig = create_trade_chart(
        combined_df, trade_label, market_hours_only, show_ewo, show_rsi,
        show_supertrend, show_ichimoku, show_atr_sl, show_market_trend,
        trades_info=trades_info_list
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not create chart for {trade_label}")

    # Trade summaries below the chart
    for trade_idx, pos_id in enumerate(pos_ids):
        df = matrices[pos_id]
        trade_num = int(df['trade_number'].iloc[0]) if 'trade_number' in df.columns and df['trade_number'].notna().any() else trade_idx + 1

        if show_trade_summary:
            if is_multi_trade:
                st.subheader(f"Trade Summary — T{trade_num}")
            else:
                st.subheader("Trade Summary")
            summary = summaries[pos_id]

            # Row 1: Entry | Exit | P&L | ATR-SL Trend | Risk | Risk Trend | Profit[min]
            row1 = st.columns(7)
            trade_label_display = f"${summary['entry']:.2f}"
            if summary.get('trade_number', 1) > 1:
                trade_label_display += f" (T{summary['trade_number']})"
            row1[0].metric("Entry", trade_label_display)
            row1[1].metric("Exit", f"${summary['exit']:.2f}")
            row1[2].metric("P&L", f"{summary['pnl_pct']:+.1f}%")
            row1[3].metric("ATR Trend", summary.get('atr_sl_trend', 'N/A'))
            row1[4].metric("Risk", summary['risk'])
            row1[5].metric("Risk Trend", summary['risk_trend'] or 'N/A')
            row1[6].metric("Profit[min]", f"${summary['profit_min']:+.2f}")

            # Row 2: Min Profit | Exit Reason | Max Profit | Risk Reasons | Trend Gate | Signal Counts
            row2 = st.columns(7)
            row2[0].metric("Min Profit", f"{summary['min_profit_pct']:+.1f}%")
            row2[1].metric("Exit Reason", summary['exit_reason'])
            row2[2].metric("Max Profit", f"{summary['max_profit_pct']:+.1f}%")
            row2[3].metric("Risk Reasons", summary['risk_reasons'] or 'N/A')

            # Show trend gate info + exit signal counts
            trend_gate_bars = summary.get('trend_gate_bars', 0)
            if trend_gate_bars > 0:
                row2[4].metric("Gate Held", f"{trend_gate_bars} bars")
            else:
                row2[4].metric("Gate Held", "N/A")

            sig_counts = summary.get('exit_signal_counts', {})
            active_sigs = {k: v for k, v in sig_counts.items() if v > 0}
            active_sig_items = list(active_sigs.items())
            for col_idx in range(2):
                if col_idx < len(active_sig_items):
                    sig_name, sig_count = active_sig_items[col_idx]
                    row2[5 + col_idx].metric(f"Sig: {sig_name}", f"{sig_count} bars")
                else:
                    if col_idx == 0 and not active_sig_items:
                        row2[5].metric("Signals", "None")
                    else:
                        row2[5 + col_idx].metric("", "")

            # Row 3: Exit signal bar counts (all signals that fired during holding)
            if active_sig_items:
                st.subheader("Exit Signals")
                # Show all exit signal counts in a dynamic row
                num_sigs = len(active_sig_items)
                sig_cols = st.columns(max(num_sigs, 1))
                for i, (sig_name, sig_count) in enumerate(active_sig_items):
                    sig_cols[i].metric(sig_name, f"{sig_count} bars")

        # Separator between trade summaries in multi-trade view
        if is_multi_trade and trade_idx < len(pos_ids) - 1:
            st.markdown("---")

    # Use first trade's df for StatsBook display (per-ticker, same across signal group)
    df = matrices[pos_ids[0]]

    # StatsBook table - only render when toggle is on
    if show_stats_book:
        st.subheader("Statsbook")
        ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None
        statsbooks = st.session_state.get('statsbooks', {})
        if ticker and ticker in statsbooks:
            sb_df = statsbooks[ticker]
            if isinstance(sb_df, pd.DataFrame) and not sb_df.empty:
                # Build display DataFrame with 1-minute normalized column
                display_df = sb_df.copy()

                # Compute 1-minute reference column from 5m data (divided by 5)
                if '5m' in display_df.columns:
                    display_df['1m'] = display_df['5m'] / 5

                # Arrange columns: 1m | 5m | 1h | 1d
                ordered_cols = []
                if '1m' in display_df.columns:
                    ordered_cols.append('1m')
                for tf in ['5m', '1h', '1d']:
                    if tf in display_df.columns:
                        ordered_cols.append(tf)
                display_df = display_df[ordered_cols]

                # Transpose: timeframes become rows, metrics become columns
                display_df = display_df.T
                display_df.index.name = 'Timeframe'
                display_df.columns.name = None
                display_df = display_df.reset_index()

                # Format numeric values using vectorized ops instead of per-cell apply
                vol_metrics = {'Max(Vol)', 'Median.Max(Vol)', 'Median(Vol)', 'Min(Vol)', 'Median.Min(Vol)'}
                ratio_metrics = {'Max(Vol)x'}
                for col in display_df.columns:
                    if col == 'Timeframe':
                        continue
                    mask = display_df[col].notna()
                    if not mask.any():
                        display_df[col] = ''
                        continue
                    if col in vol_metrics:
                        display_df.loc[mask, col] = display_df.loc[mask, col].astype(int).map('{:,}'.format)
                    elif col in ratio_metrics:
                        display_df.loc[mask, col] = display_df.loc[mask, col].round(2).astype(str)
                    else:
                        display_df.loc[mask, col] = display_df.loc[mask, col].round(3).astype(str)
                    display_df[col] = display_df[col].where(mask, '')

                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No StatsBook data available for {ticker}.")
        else:
            st.info(f"No StatsBook data available{f' for {ticker}' if ticker else ''}.")

    # Databook section - only render when toggle is on
    if not show_data_book:
        return

    st.subheader("Databook")

    # Column order from Config (source of truth)
    matrix_cols = Config.DATAFRAME_COLUMNS['dashboard_databook']

    # Show databook for each trade in the signal group
    for trade_idx, pos_id in enumerate(pos_ids):
        trade_df = matrices[pos_id]
        trade_num = int(trade_df['trade_number'].iloc[0]) if 'trade_number' in trade_df.columns and trade_df['trade_number'].notna().any() else trade_idx + 1

        if is_multi_trade:
            st.markdown(f"**T{trade_num}**")

        # Filter to columns that exist in the dataframe
        available_cols = [col for col in matrix_cols if col in trade_df.columns]

        if available_cols:
            matrix_df = trade_df[available_cols].copy()

            # Filter based on toggle:
            # ON (market_hours_only=True): Show full market hours (9:00 AM - 4:00 PM)
            # OFF (market_hours_only=False): Show only holding period (where holding=True)
            if market_hours_only and 'timestamp' in matrix_df.columns:
                import datetime as dt
                parsed_times = pd.to_datetime(matrix_df['timestamp'].astype(str).str.replace(' : ', ' '), errors='coerce')
                time_mask = (parsed_times.dt.time >= dt.time(9, 0)) & (parsed_times.dt.time <= dt.time(16, 0))
                matrix_df = matrix_df[time_mask]
            elif not market_hours_only and 'holding' in trade_df.columns:
                # Filter to holding period only
                matrix_df = matrix_df[trade_df['holding'] == True]

            # Vectorized formatting — uses numpy/pandas ops instead of per-cell lambdas

            # Format price columns as $X.XX using vectorized string concatenation
            price_cols = [c for c in ['stock_price', 'stock_high', 'stock_low', 'option_price',
                         'entry_price', 'highest_price', 'lowest_price',
                         'vwap', 'ema_10', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
                         'vwap_ema_avg', 'emavwap', 'supertrend',
                         'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
                         'atr_sl', 'spy_price'] if c in matrix_df.columns]
            for col in price_cols:
                mask = matrix_df[col].notna()
                if mask.any():
                    matrix_df.loc[mask, col] = '$' + matrix_df.loc[mask, col].round(2).astype(str)
                matrix_df[col] = matrix_df[col].where(mask, '')

            # Format volume as integer with commas
            if 'volume' in matrix_df.columns:
                mask = matrix_df['volume'].notna()
                if mask.any():
                    matrix_df.loc[mask, 'volume'] = matrix_df.loc[mask, 'volume'].astype(int).map('{:,}'.format)
                matrix_df['volume'] = matrix_df['volume'].where(mask, '')

            # Format P&L dollar amount
            if 'pnl' in matrix_df.columns:
                mask = matrix_df['pnl'].notna()
                if mask.any():
                    vals = matrix_df.loc[mask, 'pnl']
                    sign = np.where(vals >= 0, '+', '')
                    matrix_df.loc[mask, 'pnl'] = '$' + pd.Series(sign, index=vals.index) + vals.round(2).astype(str)
                matrix_df['pnl'] = matrix_df['pnl'].where(mask, '')

            if 'pnl_pct' in matrix_df.columns:
                mask = matrix_df['pnl_pct'].notna()
                if mask.any():
                    vals = matrix_df.loc[mask, 'pnl_pct']
                    sign = np.where(vals >= 0, '+', '')
                    matrix_df.loc[mask, 'pnl_pct'] = pd.Series(sign, index=vals.index) + vals.round(1).astype(str) + '%'
                matrix_df['pnl_pct'] = matrix_df['pnl_pct'].where(mask, '')

            # Format minutes held as integer
            if 'minutes_held' in matrix_df.columns:
                mask = matrix_df['minutes_held'].notna()
                if mask.any():
                    matrix_df.loc[mask, 'minutes_held'] = matrix_df.loc[mask, 'minutes_held'].astype(int).astype(str)
                matrix_df['minutes_held'] = matrix_df['minutes_held'].where(mask, '')

            if 'ticker_trend' in matrix_df.columns:
                matrix_df['ticker_trend'] = matrix_df['ticker_trend'].map({1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}).fillna('')
            if 'spy_trend' in matrix_df.columns:
                matrix_df['spy_trend'] = matrix_df['spy_trend'].map({1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}).fillna('')

            # Format decimal columns using vectorized round + astype
            decimal_fmts = [('ewo', 3), ('ewo_15min_avg', 3), ('rsi', 1), ('rsi_10min_avg', 1)]
            for col, decimals in decimal_fmts:
                if col in matrix_df.columns:
                    mask = matrix_df[col].notna()
                    if mask.any():
                        matrix_df.loc[mask, col] = matrix_df.loc[mask, col].round(decimals).astype(str)
                    matrix_df[col] = matrix_df[col].where(mask, '')

            # Format exit signal flags as Y/blank
            for sig_col in EXIT_SIGNAL_DEFS:
                if sig_col in matrix_df.columns:
                    matrix_df[sig_col] = np.where(matrix_df[sig_col] == True, 'Y', '')

            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No databook data available for trade T{trade_num}.")

        if is_multi_trade and trade_idx < len(pos_ids) - 1:
            st.markdown("---")


if __name__ == "__main__":
    main()
