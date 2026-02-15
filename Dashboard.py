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
    # EMA colors
    'ema_10': '#FF9800',          # Orange
    'ema_21': '#AB47BC',          # Purple
    'ema_50': '#FFEB3B',          # Yellow
    'ema_100': '#00E5FF',         # Cyan
    'ema_200': '#E91E63',         # Pink
    # Trading range
    'trading_range': '#FF1744',   # Red
    # Exit signal markers
    'sig_ai': '#00BCD4',           # Cyan (AI Exit)
}

# Exit signal definitions: column name -> (display label, color key, marker symbol)
EXIT_SIGNAL_DEFS = {
    'exit_sig_ai':           ('AI Exit',      'sig_ai',           'diamond'),
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
        statsbooks = data.get('statsbooks', {})
        st.session_state.matrices = matrices
        st.session_state.statsbooks = statsbooks
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


def create_trade_chart(df, trade_label, market_hours_only=False, show_market_bias=True):
    """Create dual-axis chart with stock/option prices, error bars, market bias, and SPY subplot."""
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

    # Check if market bias data is available (and toggle is on)
    has_market_bias = show_market_bias and 'market_bias' in df.columns and df['market_bias'].notna().any()

    # Check if SPY data is available
    has_spy = 'spy_price' in df.columns and df['spy_price'].notna().any()

    # Determine number of subplot rows
    has_indicators = has_market_bias

    # Create subplots: main chart + indicators + SPY
    if has_indicators and has_spy:
        # 3 rows: main chart, indicators, SPY
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.50, 0.25, 0.25],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]]
        )
    elif has_indicators:
        # 2 rows: main chart, indicators
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.60, 0.40],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
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

    # EMA lines (left y-axis)
    ema_defs = [
        ('ema_10', 'EMA 10', COLORS['ema_10'], 1.5, 'dot'),
        ('ema_21', 'EMA 21', COLORS['ema_21'], 1.5, 'dot'),
        ('ema_50', 'EMA 50', COLORS['ema_50'], 1.5, 'dashdot'),
        ('ema_100', 'EMA 100', COLORS['ema_100'], 1.5, 'dash'),
        ('ema_200', 'EMA 200', COLORS['ema_200'], 1.5, 'dash'),
    ]
    for col_name, label, color, width, dash in ema_defs:
        if col_name in df.columns and df[col_name].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df[col_name],
                    name=label,
                    line=dict(color=color, width=width, dash=dash),
                    hovertemplate=f'{label}: $%{{y:.2f}}<extra></extra>'
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

    # Exit signal markers on main chart (right y-axis, aligned with option price)
    if opt_col in df.columns:
        opt_max = df[opt_col].max() if df[opt_col].notna().any() else 1
        opt_min = df[opt_col].min() if df[opt_col].notna().any() else 0
        opt_range = opt_max - opt_min if opt_max > opt_min else opt_max * 0.1
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

    # Market bias on row 2 (color-coded step fill: green=bull, red=bear)
    if has_market_bias:
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

        # SPY gauge sentiment as color-coded markers on secondary axis
        spy_gauge_cols = {
            'spy_since_open': 'Open',
            'spy_1m': '1m',
            'spy_5m': '5m',
            'spy_15m': '15m',
            'spy_30m': '30m',
            'spy_1h': '1h',
        }

        # Show the 5m gauge as background shading
        if 'spy_5m' in df.columns and df['spy_5m'].notna().any():
            sentiment = df[['time', 'spy_5m']].dropna(subset=['spy_5m']).copy()
            if not sentiment.empty:
                sentiment['group'] = (sentiment['spy_5m'] != sentiment['spy_5m'].shift()).cumsum()
                for _, grp in sentiment.groupby('group'):
                    color = 'rgba(0, 200, 83, 0.10)' if grp['spy_5m'].iloc[0] == 'Bullish' else 'rgba(255, 23, 68, 0.10)'
                    fig.add_vrect(
                        x0=grp['time'].iloc[0], x1=grp['time'].iloc[-1],
                        fillcolor=color, layer='below', line_width=0,
                        row=spy_row, col=1
                    )

        # Build hover text showing all gauge timeframes
        def build_spy_hover(row):
            parts = [f"SPY: ${row['spy_price']:.2f}" if pd.notna(row.get('spy_price')) else "SPY: N/A"]
            for col, label in spy_gauge_cols.items():
                val = row.get(col)
                if pd.notna(val) and val:
                    icon = '+' if val == 'Bullish' else '-'
                    parts.append(f"{label}:{icon}")
            return '<br>'.join([parts[0], ' | '.join(parts[1:])])

        hover_texts = df.apply(build_spy_hover, axis=1)
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['spy_price'],
                name='SPY Gauge',
                mode='none',
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
    fig.update_xaxes(title_text="Time", tickformat='%H:%M', row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, tickformat='$.2f', row=1, col=1)
    fig.update_yaxes(title_text="Option ($)", secondary_y=True, tickformat='$.2f', row=1, col=1)

    if has_indicators:
        fig.update_xaxes(title_text="Time", tickformat='%H:%M', row=2, col=1)
        fig.update_yaxes(
            title_text="Bias", secondary_y=False, row=2, col=1,
            range=[-1.5, 1.5], tickvals=[-1, 0, 1], ticktext=['Bear', 'Side', 'Bull']
        )

    if has_spy and spy_row:
        fig.update_xaxes(title_text="Time", tickformat='%H:%M', row=spy_row, col=1)
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

    return fig


def get_trade_summary(df):
    """Extract concise trade summary including max/min profit percentages."""
    if df.empty:
        return {
            'entry': 0, 'exit': 0, 'pnl_pct': 0, 'exit_reason': 'N/A',
            'duration': 0, 'max_price': 0, 'min_price': 0,
            'max_profit_pct': 0, 'min_profit_pct': 0,
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
    if 'holding' in df.columns and opt_col in df.columns:
        holding_df = df[df['holding'] == True]
        if not holding_df.empty and holding_df[opt_col].notna().any():
            max_price = holding_df[opt_col].max()
            min_price = holding_df[opt_col].min()
            if entry_price > 0:
                max_profit_pct = (max_price / entry_price - 1) * 100
                min_profit_pct = (min_price / entry_price - 1) * 100

    # Market bias at exit (last holding bar's VWAP assessment)
    BIAS_LABELS = {1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}
    market_bias = 'N/A'
    if 'market_bias' in df.columns and 'holding' in df.columns:
        holding_df = df[df['holding'] == True]
        if not holding_df.empty:
            last_bias = holding_df['market_bias'].iloc[-1]
            if pd.notna(last_bias):
                market_bias = BIAS_LABELS.get(int(last_bias), 'N/A')

    # Count exit signal activations during holding period
    exit_signal_counts = {}
    if 'holding' in df.columns:
        holding_df = df[df['holding'] == True]
        for sig_col, (sig_label, _, _) in EXIT_SIGNAL_DEFS.items():
            if sig_col in holding_df.columns:
                count = (holding_df[sig_col] == True).sum()
                exit_signal_counts[sig_label] = int(count)

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
        'market_bias': market_bias,
        'exit_signal_counts': exit_signal_counts,
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
    matrices = load_data()

    if not matrices:
        st.warning("No backtest data found. Run Test.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Trades")

        trade_list = []
        for pos_id, df in matrices.items():
            if df.empty:
                continue
            # Build label as "Ticker : strike : c/p"
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNK'
            strike = df['strike'].iloc[0] if 'strike' in df.columns else 0
            opt_type = df['option_type'].iloc[0] if 'option_type' in df.columns else '?'
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

        if st.button("Reload Data"):
            st.session_state.pop('matrices', None)
            st.session_state.pop('statsbooks', None)
            st.rerun()

        st.markdown("---")
        market_hours_only = st.toggle("Market Hours", value=True, help="ON: Full market hours view | OFF: Holding period only")
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

    # Chart first
    fig = create_trade_chart(df, trade_label, market_hours_only, show_market_bias)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create chart")

    # Trade Summary below chart
    st.subheader("Trade Summary")
    summary = get_trade_summary(df)

    # Row 1: Entry | Exit | P&L | Market | Min Profit | Max Profit | Exit Reason
    row1 = st.columns(7)
    row1[0].metric("Entry", f"${summary['entry']:.2f}")
    row1[1].metric("Exit", f"${summary['exit']:.2f}")
    row1[2].metric("P&L", f"{summary['pnl_pct']:+.1f}%")
    row1[3].metric("Market", summary['market_bias'])
    row1[4].metric("Min Profit", f"{summary['min_profit_pct']:+.1f}%")
    row1[5].metric("Max Profit", f"{summary['max_profit_pct']:+.1f}%")
    row1[6].metric("Exit Reason", summary['exit_reason'])

    # Show exit signal counts
    sig_counts = summary.get('exit_signal_counts', {})
    active_sigs = {k: v for k, v in sig_counts.items() if v > 0}
    if active_sigs:
        st.subheader("Exit Signals")
        num_sigs = len(active_sigs)
        sig_cols = st.columns(max(num_sigs, 1))
        for i, (sig_name, sig_count) in enumerate(active_sigs.items()):
            sig_cols[i].metric(sig_name, f"{sig_count} bars")

    # StatsBook table
    st.subheader("Statsbook")
    ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None
    statsbooks = st.session_state.get('statsbooks', {})
    if ticker and ticker in statsbooks:
        sb_df = statsbooks[ticker]
        if isinstance(sb_df, pd.DataFrame) and not sb_df.empty:
            # Build display DataFrame with 1-minute normalized columns
            display_df = sb_df.copy()

            # Compute 1-minute reference columns from raw numeric data
            norm_map = {'1m:5m': ('5m', 5), '1m:1h': ('1h', 60)}
            for norm_col, (src_col, divisor) in norm_map.items():
                if src_col in display_df.columns:
                    display_df[norm_col] = display_df[src_col] / divisor

            # Arrange columns: 5m | 1m:5m | 1h | 1m:1h | 1d
            ordered_cols = []
            for tf in ['5m', '1h', '1d']:
                if tf in display_df.columns:
                    ordered_cols.append(tf)
                norm = f"1m:{tf}"
                if norm in display_df.columns:
                    ordered_cols.append(norm)
            display_df = display_df[ordered_cols]

            # Transpose
            display_df = display_df.T
            display_df.index.name = 'Timeframe'
            display_df.columns.name = None
            display_df = display_df.reset_index()

            # Format numeric values
            vol_metrics = {'Max(Vol)', 'Median.Max(Vol)', 'Median(Vol)', 'Min(Vol)', 'Median.Min(Vol)'}
            ratio_metrics = {'Max(Vol)x'}
            for col in display_df.columns:
                if col == 'Timeframe':
                    continue
                display_df[col] = display_df[col].apply(
                    lambda x, c=col: (
                        f"{int(x):,}" if c in vol_metrics and pd.notna(x)
                        else f"{x:.2f}" if c in ratio_metrics and pd.notna(x)
                        else f"{x:.3f}" if pd.notna(x)
                        else ""
                    )
                )

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No StatsBook data available for {ticker}.")
    else:
        st.info(f"No StatsBook data available{f' for {ticker}' if ticker else ''}.")

    # Databook section
    st.subheader("Databook")

    # Column order from Config (source of truth)
    matrix_cols = Config.DATAFRAME_COLUMNS['dashboard_databook']

    # Filter to columns that exist in the dataframe
    available_cols = [col for col in matrix_cols if col in df.columns]

    if available_cols:
        matrix_df = df[available_cols].copy()

        # Filter based on toggle
        if market_hours_only and 'timestamp' in matrix_df.columns:
            import datetime as dt
            matrix_df['_time'] = matrix_df['timestamp'].apply(parse_time)
            matrix_df = matrix_df[
                (matrix_df['_time'].dt.time >= dt.time(9, 0)) &
                (matrix_df['_time'].dt.time <= dt.time(16, 0))
            ]
            matrix_df = matrix_df.drop(columns=['_time'])
        elif not market_hours_only and 'holding' in df.columns:
            matrix_df = matrix_df[df['holding'] == True]

        # Format price columns as $X.XX
        for col in ['stock_price', 'stock_high', 'stock_low', 'true_price', 'option_price',
                     'entry_price', 'highest_price', 'lowest_price',
                     'vwap', 'ema_10', 'ema_21', 'ema_50', 'ema_100', 'ema_200']:
            if col in matrix_df.columns:
                matrix_df[col] = matrix_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

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

        if 'market_bias' in matrix_df.columns:
            bias_map = {1: 'Bullish', 0: 'Sideways', -1: 'Bearish'}
            matrix_df['market_bias'] = matrix_df['market_bias'].apply(
                lambda x: bias_map.get(int(x), '') if pd.notna(x) else ""
            )

        # Format SPY price
        if 'spy_price' in matrix_df.columns:
            matrix_df['spy_price'] = matrix_df['spy_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

        # Format exit signal flags as checkmarks
        for sig_col in EXIT_SIGNAL_DEFS:
            if sig_col in matrix_df.columns:
                matrix_df[sig_col] = matrix_df[sig_col].apply(
                    lambda x: 'Y' if x is True else '' if pd.isna(x) else ''
                )

        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info("No databook data available for this trade.")


if __name__ == "__main__":
    main()
