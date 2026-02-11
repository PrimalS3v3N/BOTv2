"""
Dashboard.py - Trade Visualization Dashboard

Visualizes backtest trade data from Test.py.
Shows stock price vs option price with entry/exit markers.

Usage:
    streamlit run Dashboard.py
"""

import streamlit as st
import pandas as pd
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
    'ema_30': '#AB47BC',          # Purple
    'vwap_ema_avg': '#FFEB3B',    # Yellow
    'emavwap': '#00E5FF',         # Cyan
    'stop_loss_line': '#00C853',  # Green
    # Trading range
    'trading_range': '#FF1744',   # Red
    # Ichimoku Cloud
    'ichimoku_tenkan': '#E91E63',   # Pink (Conversion Line)
    'ichimoku_kijun': '#3F51B5',    # Indigo (Base Line)
    'ichimoku_cloud_bull': 'rgba(0, 200, 83, 0.15)',   # Green cloud fill
    'ichimoku_cloud_bear': 'rgba(255, 23, 68, 0.15)',  # Red cloud fill
    'ichimoku_senkou_a': '#26A69A',  # Teal (Leading Span A)
    'ichimoku_senkou_b': '#EF5350',  # Red (Leading Span B)
}

def load_data():
    """Load backtest data from pickle file, caching in session state."""
    if 'matrices' in st.session_state and st.session_state.matrices:
        return st.session_state.matrices

    if not os.path.exists(DATA_PATH):
        return {}
    try:
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        matrices = data.get('matrices', {})
        st.session_state.matrices = matrices
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


def create_trade_chart(df, trade_label, market_hours_only=False, show_ewo=True, show_rsi=True, show_supertrend=False, show_ichimoku=False, show_market_bias=True):
    """Create dual-axis chart with stock/option prices, error bars, and combined EWO/RSI subplot."""
    df = df.copy()
    df['time'] = df['timestamp'].apply(parse_time)
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

    entry_row, exit_row, opt_col = find_entry_exit(df)

    # Check if EWO data is available for subplot (and toggle is on)
    has_ewo = show_ewo and 'ewo' in df.columns and df['ewo'].notna().any()

    # Check if RSI data is available (and toggle is on)
    has_rsi = show_rsi and 'rsi' in df.columns and df['rsi'].notna().any()

    # Check if market bias data is available (and toggle is on)
    has_market_bias = show_market_bias and 'market_bias' in df.columns and df['market_bias'].notna().any()

    # RSI uses secondary y-axis only when EWO is also shown (they share row 2)
    # When EWO is off, RSI uses the primary y-axis to avoid empty primary axis issues
    rsi_secondary = has_ewo

    # Create subplots: main chart + combined indicator subplot below
    if has_ewo or has_rsi or has_market_bias:
        # 2 rows: main chart, indicators
        row2_spec = {"secondary_y": True} if has_ewo else {"secondary_y": False}
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.60, 0.40],
            specs=[[{"secondary_y": True}], [row2_spec]]
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Option price (right y-axis) - green
    if opt_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[opt_col],
                name='Option',
                line=dict(color='#00C853', width=2),
                hovertemplate='Option: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )

    # Price (left y-axis) - blue with high/low error bars
    if 'true_price' in df.columns and df['true_price'].notna().any():
        true_price_trace = go.Scatter(
            x=df['time'],
            y=df['true_price'],
            name='Price',
            line=dict(color='#2196F3', width=2),
            hovertemplate='Price: $%{y:.2f}<extra></extra>'
        )

        # Add error bars showing high/low range relative to true price
        if 'stock_high' in df.columns and 'stock_low' in df.columns:
            tp_error_plus = df['stock_high'] - df['true_price']
            tp_error_minus = df['true_price'] - df['stock_low']
            true_price_trace.error_y = dict(
                type='data',
                symmetric=False,
                array=tp_error_plus,
                arrayminus=tp_error_minus,
                color='rgba(255, 255, 255, 0.8)',
                thickness=3,
                width=0
            )

        fig.add_trace(true_price_trace, row=1, col=1, secondary_y=False)

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

    # EMA (left y-axis) - orange
    if 'ema_30' in df.columns and df['ema_30'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema_30'],
                name='EMA',
                line=dict(color='#FF9800', width=1.5, dash='dot'),
                hovertemplate='EMA: $%{y:.2f}<extra></extra>'
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

    # Stop Loss line (right y-axis, tracks stop loss price)
    if 'stop_loss' in df.columns and df['stop_loss'].notna().any():
        # Create hover text with stop_loss_mode if available
        if 'stop_loss_mode' in df.columns:
            hover_text = df.apply(
                lambda r: f"Stop Loss: ${r['stop_loss']:.2f}<br>Mode: {r['stop_loss_mode']}"
                if pd.notna(r['stop_loss']) else "", axis=1
            )
        else:
            hover_text = df['stop_loss'].apply(lambda x: f"Stop Loss: ${x:.2f}" if pd.notna(x) else "")

        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['stop_loss'],
                name='Stop Loss',
                line=dict(color=COLORS['stop_loss_line'], width=1.5, dash='dash'),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            ),
            row=1, col=1, secondary_y=True
        )

    # Ichimoku Cloud (left y-axis) - cloud fill between Senkou spans + Tenkan/Kijun lines
    if show_ichimoku:
        has_senkou_a = 'ichimoku_senkou_a' in df.columns and df['ichimoku_senkou_a'].notna().any()
        has_senkou_b = 'ichimoku_senkou_b' in df.columns and df['ichimoku_senkou_b'].notna().any()
        has_tenkan = 'ichimoku_tenkan' in df.columns and df['ichimoku_tenkan'].notna().any()
        has_kijun = 'ichimoku_kijun' in df.columns and df['ichimoku_kijun'].notna().any()

        # Cloud fill: Senkou Span A and B with shaded region between
        if has_senkou_a and has_senkou_b:
            # Determine cloud color per bar: green when A >= B (bullish), red when A < B (bearish)
            cloud_colors = [
                COLORS['ichimoku_cloud_bull'] if a >= b else COLORS['ichimoku_cloud_bear']
                for a, b in zip(
                    df['ichimoku_senkou_a'].fillna(0),
                    df['ichimoku_senkou_b'].fillna(0)
                )
            ]

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

    # Entry marker
    if entry_row is not None:
        entry_time = parse_time(entry_row['timestamp'])
        entry_price = entry_row[opt_col] if opt_col in entry_row else entry_row['stock_price']
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers',
                name='Entry',
                marker=dict(symbol='triangle-down', size=18, color=COLORS['entry'], line=dict(color='white', width=2)),
                hovertemplate='ENTRY<br>$%{y:.2f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )

    # Exit marker
    if exit_row is not None:
        exit_time = parse_time(exit_row['timestamp'])
        exit_price = exit_row[opt_col] if opt_col in exit_row else exit_row['stock_price']
        exit_reason = exit_row.get('exit_reason', 'unknown') if hasattr(exit_row, 'get') else exit_row['exit_reason'] if 'exit_reason' in exit_row.index else 'unknown'
        if pd.isna(exit_reason):
            exit_reason = 'unknown'
        fig.add_trace(
            go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode='markers+text',
                name=f'Exit ({exit_reason})',
                marker=dict(symbol='triangle-up', size=18, color=COLORS['exit'], line=dict(color='white', width=2)),
                text=[str(exit_reason)],
                textposition='top center',
                textfont=dict(size=10, color=COLORS['exit']),
                hovertemplate=f'EXIT: {exit_reason}<br>$%{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )

    # EWO subplot (row 2) — displayed as histogram bars
    if has_ewo:
        # Color each bar green (positive) or red (negative)
        ewo_colors = ['rgba(0, 200, 83, 0.7)' if v >= 0 else 'rgba(255, 23, 68, 0.7)'
                       for v in df['ewo']]

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

    # Market bias on row 2 primary y-axis (color-coded step fill: green=bull, red=bear)
    if has_market_bias and (has_ewo or has_rsi or has_market_bias):
        bias_vals = df['market_bias'].copy()

        # Bullish segments (+1) with green fill from 0 to +1
        bias_bull = bias_vals.where(bias_vals >= 1)
        if bias_bull.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'], y=bias_bull,
                    name='Bias (Bull)',
                    mode='lines',
                    line=dict(color='#00C853', width=2, shape='hv'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 200, 83, 0.12)',
                    connectgaps=False,
                    hovertemplate='Bias: +1 (Bull)<extra></extra>'
                ),
                row=2, col=1
            )

        # Bearish segments (-1) with red fill from 0 to -1
        bias_bear = bias_vals.where(bias_vals <= -1)
        if bias_bear.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'], y=bias_bear,
                    name='Bias (Bear)',
                    mode='lines',
                    line=dict(color='#FF1744', width=2, shape='hv'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 23, 68, 0.12)',
                    connectgaps=False,
                    hovertemplate='Bias: -1 (Bear)<extra></extra>'
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

    # Layout
    if has_ewo or has_rsi or has_market_bias:
        chart_height = 800
    else:
        chart_height = 500

    fig.update_layout(
        title=trade_label,
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
        margin=dict(l=60, r=180, t=60, b=40)
    )

    # Update axes
    fig.update_xaxes(title_text="Time", tickformat='%H:%M', row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, tickformat='$.2f', row=1, col=1)
    fig.update_yaxes(title_text="Option ($)", secondary_y=True, tickformat='$.2f', row=1, col=1)

    if has_ewo or has_rsi or has_market_bias:
        fig.update_xaxes(title_text="Time", tickformat='%H:%M', row=2, col=1)

    if has_ewo:
        ewo_title = "EWO / Bias" if has_market_bias else "EWO"
        fig.update_yaxes(title_text=ewo_title, secondary_y=False, row=2, col=1)
    elif has_market_bias and not has_rsi:
        fig.update_yaxes(
            title_text="Bias", secondary_y=False, row=2, col=1,
            range=[-1.5, 1.5], tickvals=[-1, 0, 1], ticktext=['Bear', 'Side', 'Bull']
        )

    if has_rsi:
        fig.update_yaxes(
            title_text="RSI", secondary_y=rsi_secondary, row=2, col=1,
            range=[0, 100], tickvals=[0, 30, 50, 70, 100]
        )

    return fig


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

    # Calculate max/min contract prices AFTER entry (only during holding period)
    max_price = 0
    min_price = 0
    max_profit_pct = 0
    min_profit_pct = 0
    profit_min = 0  # P&L at the worst stop loss point (min price during holding)
    if 'holding' in df.columns and opt_col in df.columns:
        holding_df = df[df['holding'] == True]
        if not holding_df.empty and holding_df[opt_col].notna().any():
            max_price = holding_df[opt_col].max()
            min_price = holding_df[opt_col].min()
            # Max Profit % = (max_price / entry_price - 1) * 100
            if entry_price > 0:
                max_profit_pct = (max_price / entry_price - 1) * 100
                min_profit_pct = (min_price / entry_price - 1) * 100
                # Calculate Profit[min] - P&L in dollars at the worst stop loss point
                contracts = int(df['contracts'].iloc[0]) if 'contracts' in df.columns else 1
                profit_min = (min_price - entry_price) * contracts * 100

    # Market bias at exit (last holding bar's VWAP assessment)
    # Values: +1 = Bullish, 0 = Sideways, -1 = Bearish
    BIAS_LABELS = {1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}
    market_bias = 'N/A'
    if 'market_bias' in df.columns and 'holding' in df.columns:
        holding_df = df[df['holding'] == True]
        if not holding_df.empty:
            last_bias = holding_df['market_bias'].iloc[-1]
            if pd.notna(last_bias):
                market_bias = BIAS_LABELS.get(int(last_bias), 'N/A')

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
        'market_bias': market_bias
    }


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
    st.title("Trade Dashboard")

    matrices = load_data()

    if not matrices:
        st.warning("No backtest data found. Run Test.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Trades")

        if st.button("Reload Data"):
            st.session_state.pop('matrices', None)
            st.rerun()

        trade_list = []
        for pos_id, df in matrices.items():
            if df.empty:
                continue
            # Build label as "Ticker : strike : c/p"
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNK'
            strike = df['strike'].iloc[0] if 'strike' in df.columns else 0
            opt_type = df['option_type'].iloc[0] if 'option_type' in df.columns else '?'
            # Use first letter for c/p
            cp = opt_type[0].upper() if opt_type else '?'
            label = f"{ticker} : {strike:.0f} : {cp}"

            # Calculate P&L for sorting
            summary = get_trade_summary(df)
            pnl = summary['pnl_pct']

            trade_list.append((label, pos_id, pnl))

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

        st.markdown("---")
        market_hours_only = st.toggle("Market Hours", value=True, help="ON: Full market hours view | OFF: Holding period only")
        show_ewo = st.toggle("Show EWO Graph", value=True)
        show_rsi = st.toggle("Show RSI Graph", value=True)
        show_supertrend = st.toggle("Show Supertrend", value=False, help="Overlay Supertrend indicator on main chart")
        show_ichimoku = st.toggle("Show Ichimoku Cloud", value=False, help="Overlay Ichimoku Cloud (Tenkan, Kijun, Senkou spans) on main chart")
        show_market_bias = st.toggle("Show Market Bias", value=True, help="Market bias on indicator subplot: +1 Bull, 0 Side, -1 Bear")

        st.markdown("---")

        # Calculate Total Profit and Potential Profit across all trades
        total_profit_dollars = 0.0
        total_investment = 0.0
        potential_profit_dollars = 0.0
        for _, tdf in matrices.items():
            if tdf.empty:
                continue
            s = get_trade_summary(tdf)
            entry_p = s['entry']
            exit_p = s['exit']
            max_p = s['max_price']
            contracts = int(tdf['contracts'].iloc[0]) if 'contracts' in tdf.columns else 1
            if entry_p > 0:
                inv = entry_p * contracts * 100
                total_investment += inv
                total_profit_dollars += (exit_p - entry_p) * contracts * 100
                potential_profit_dollars += (max_p - entry_p) * contracts * 100

        total_profit_pct = (total_profit_dollars / total_investment * 100) if total_investment > 0 else 0
        potential_profit_pct = (potential_profit_dollars / total_investment * 100) if total_investment > 0 else 0

        st.markdown(f"**Total Profit:** ${total_profit_dollars:,.2f} ({total_profit_pct:+.1f}%)")
        st.markdown(f"**Potential Profit:** ${potential_profit_dollars:,.2f} ({potential_profit_pct:+.1f}%)")

        st.markdown("---")
        st.caption(f"{len(matrices)} trades loaded")

    # Display selected trade
    trade_label, pos_id, _ = trade_list[selected_idx]
    df = matrices[pos_id]

    # Chart first (no summary above)
    fig = create_trade_chart(df, trade_label, market_hours_only, show_ewo, show_rsi, show_supertrend, show_ichimoku, show_market_bias)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create chart")

    # Trade Summary below chart (two rows)
    st.subheader("Trade Summary")
    summary = get_trade_summary(df)

    # Row 1: Entry | Exit | P&L | Market | Profit[min] TBD | TBD | TBD
    row1 = st.columns(7)
    row1[0].metric("Entry", f"${summary['entry']:.2f}")
    row1[1].metric("Exit", f"${summary['exit']:.2f}")
    row1[2].metric("P&L", f"{summary['pnl_pct']:+.1f}%")
    row1[3].metric("Market", summary['market_bias'])
    row1[4].metric("Profit[min] TBD", f"${summary['profit_min']:+.2f}")
    row1[5].metric("TBD", "1")
    row1[6].metric("TBD", "1")

    # Row 2: Min Profit | Exit Reason | Max Profit | TBD | TBD | TBD | TBD
    row2 = st.columns(7)
    row2[0].metric("Min Profit", f"{summary['min_profit_pct']:+.1f}%")
    row2[1].metric("Exit Reason", summary['exit_reason'])
    row2[2].metric("Max Profit", f"{summary['max_profit_pct']:+.1f}%")
    row2[3].metric("TBD", "1")
    row2[4].metric("TBD", "1")
    row2[5].metric("TBD", "1")
    row2[6].metric("TBD", "1")

    # Data table
    st.subheader("Trade Details")
    table_df = get_trade_table(df)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Databook section
    st.subheader("Databook")

    # Column order from Config (source of truth)
    matrix_cols = Config.DATAFRAME_COLUMNS['dashboard_databook']

    # Filter to columns that exist in the dataframe
    available_cols = [col for col in matrix_cols if col in df.columns]

    if available_cols:
        matrix_df = df[available_cols].copy()

        # Filter based on toggle:
        # ON (market_hours_only=True): Show full market hours (9:00 AM - 4:00 PM)
        # OFF (market_hours_only=False): Show only holding period (where holding=True)
        if market_hours_only and 'timestamp' in matrix_df.columns:
            import datetime as dt
            matrix_df['_time'] = matrix_df['timestamp'].apply(parse_time)
            matrix_df = matrix_df[
                (matrix_df['_time'].dt.time >= dt.time(9, 0)) &
                (matrix_df['_time'].dt.time <= dt.time(16, 0))
            ]
            matrix_df = matrix_df.drop(columns=['_time'])
        elif not market_hours_only and 'holding' in df.columns:
            # Filter to holding period only
            matrix_df = matrix_df[df['holding'] == True]

        # Format price columns as $X.XX
        for col in ['stock_price', 'stock_high', 'stock_low', 'true_price', 'option_price',
                     'entry_price', 'highest_price', 'lowest_price', 'stop_loss', 'trailing_stop_price',
                     'vwap', 'ema_30', 'vwap_ema_avg', 'emavwap', 'supertrend',
                     'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b']:
            if col in matrix_df.columns:
                matrix_df[col] = matrix_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

        # Format ATR
        if 'atr' in matrix_df.columns:
            matrix_df['atr'] = matrix_df['atr'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        # Format volume as integer
        if 'volume' in matrix_df.columns:
            matrix_df['volume'] = matrix_df['volume'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")

        # Format P&L dollar amount
        if 'pnl' in matrix_df.columns:
            matrix_df['pnl'] = matrix_df['pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "")

        if 'pnl_pct' in matrix_df.columns:
            matrix_df['pnl_pct'] = matrix_df['pnl_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")

        # Format minutes held as integer
        if 'minutes_held' in matrix_df.columns:
            matrix_df['minutes_held'] = matrix_df['minutes_held'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        # Format milestone percentage
        if 'milestone_pct' in matrix_df.columns:
            matrix_df['milestone_pct'] = matrix_df['milestone_pct'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

        if 'market_bias' in matrix_df.columns:
            bias_map = {1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}
            matrix_df['market_bias'] = matrix_df['market_bias'].apply(
                lambda x: bias_map.get(int(x), '') if pd.notna(x) else ""
            )

        if 'ewo' in matrix_df.columns:
            matrix_df['ewo'] = matrix_df['ewo'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        if 'ewo_15min_avg' in matrix_df.columns:
            matrix_df['ewo_15min_avg'] = matrix_df['ewo_15min_avg'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        if 'rsi' in matrix_df.columns:
            matrix_df['rsi'] = matrix_df['rsi'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

        if 'rsi_10min_avg' in matrix_df.columns:
            matrix_df['rsi_10min_avg'] = matrix_df['rsi_10min_avg'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info("No databook data available for this trade.")


if __name__ == "__main__":
    main()
