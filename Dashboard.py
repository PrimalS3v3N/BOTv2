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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'BT_DATA.pkl')

st.set_page_config(page_title="Trade Dashboard", layout="wide")

COLORS = {
    'stock': '#2962FF',
    'option': '#FF6D00',
    'entry': '#00C853',
    'exit': '#FF1744',
    'stop_loss': '#D50000',
    'profit_target': '#00E676',
    'trailing_stop': '#FFAB00',
    'time_stop': '#7C4DFF',
    'rsi': '#E91E63',
    'macd_crossover': '#00BCD4',
    'vwap': '#8BC34A',
    'vpoc': '#FF5722',
    'supertrend': '#9C27B0',
    'expiration': '#795548',
    # Stop loss tracking
    'ema_20': '#29B6F6',          # Light Blue
    'ema_30': '#AB47BC',          # Purple
    'stop_loss_line': '#FF5252',  # Red
    'sl_c1': '#FFD54F',           # Amber (conditional trailing)
    'sl_c2': '#FF8A65',           # Deep Orange (EMA/VWAP bearish)
    'sl_ema': '#FF8A65',          # Same as sl_c2
}

EXIT_SYMBOLS = {
    'stop_loss': 'x',
    'profit_target': 'star',
    'trailing_stop': 'triangle-down',
    'time_stop': 'square',
    'rsi': 'diamond',
    'macd_crossover': 'cross',
    'vwap': 'pentagon',
    'vpoc': 'hexagon',
    'supertrend': 'star-triangle-up',
    'expiration': 'hourglass',
    'sl_ema': 'triangle-down',
    'sl_c1': 'circle',
    'sl_c2': 'triangle-down',
}


def load_data():
    """Load backtest data from pickle file, caching in session state."""
    if 'matrices' in st.session_state and st.session_state.matrices:
        return st.session_state.matrices, st.session_state.exit_signals

    if not os.path.exists(DATA_PATH):
        return {}, {}
    try:
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        matrices = data.get('matrices', {})
        exit_signals = data.get('exit_signals', {})
        st.session_state.matrices = matrices
        st.session_state.exit_signals = exit_signals
        return matrices, exit_signals
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, {}


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


def create_trade_chart(df, trade_label, show_all_exits=False, market_hours_only=False):
    """Create dual-axis chart with stock/option prices and entry/exit markers."""
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

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stock price (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['stock_price'],
            name='Stock',
            line=dict(color=COLORS['stock'], width=2),
            hovertemplate='Stock: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )

    # VWAP (left y-axis, same as stock price)
    if 'vwap' in df.columns and df['vwap'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['vwap'],
                name='VWAP',
                line=dict(color=COLORS['vwap'], width=2, dash='dash'),
                hovertemplate='VWAP: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )

    # EMA_20 (left y-axis, same as stock price)
    if 'ema_20' in df.columns and df['ema_20'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema_20'],
                name='EMA 20',
                line=dict(color=COLORS['ema_20'], width=1.5, dash='dot'),
                hovertemplate='EMA 20: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )

    # EMA_30 (left y-axis, same as stock price)
    if 'ema_30' in df.columns and df['ema_30'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema_30'],
                name='EMA 30',
                line=dict(color=COLORS['ema_30'], width=1.5, dash='dot'),
                hovertemplate='EMA 30: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )

    # Option price (right y-axis)
    if opt_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[opt_col],
                name='Option',
                line=dict(color=COLORS['option'], width=2),
                hovertemplate='Option: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=True
        )

    # Stop Loss line (right y-axis, tracks dynamic stop loss price)
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
            secondary_y=True
        )

    # SL_C1 markers: Conditional trailing active (profit target + VWAP hold)
    if 'SL_C1' in df.columns and opt_col in df.columns:
        sl_c1_df = df[df['SL_C1'] == True]
        if not sl_c1_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sl_c1_df['time'],
                    y=sl_c1_df[opt_col],
                    mode='markers',
                    name='SL_C1 (Trailing)',
                    marker=dict(symbol='circle', size=8, color=COLORS['sl_c1'], opacity=0.6),
                    hovertemplate='SL_C1: Conditional Trailing<br>$%{y:.2f}<extra></extra>'
                ),
                secondary_y=True
            )

    # SL_C2 markers: EMA/VWAP bearish condition
    if 'SL_C2' in df.columns and opt_col in df.columns:
        sl_c2_df = df[df['SL_C2'] == True]
        if not sl_c2_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sl_c2_df['time'],
                    y=sl_c2_df[opt_col],
                    mode='markers',
                    name='SL_C2 (EMA/VWAP)',
                    marker=dict(symbol='triangle-down', size=8, color=COLORS['sl_c2'], opacity=0.6),
                    hovertemplate='SL_C2: EMA30>VWAP & Price<EMA30<br>$%{y:.2f}<extra></extra>'
                ),
                secondary_y=True
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
            secondary_y=True
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
                hovertemplate=f'EXIT: {exit_reason}<br>${{y:.2f}}<extra></extra>'
            ),
            secondary_y=True
        )

    # Show all exit strategy markers (when toggle is on)
    if show_all_exits and 'exit_signals_at_time' in df.columns:
        signal_types = set()
        for s in df['exit_signals_at_time'].dropna():
            if s:
                signal_types.update([x.strip() for x in str(s).split(',') if x.strip()])

        for sig_type in signal_types:
            mask = df['exit_signals_at_time'].astype(str).str.contains(sig_type, na=False)
            sig_df = df[mask]
            if not sig_df.empty:
                color = COLORS.get(sig_type, '#888888')
                symbol = EXIT_SYMBOLS.get(sig_type, 'circle')
                fig.add_trace(
                    go.Scatter(
                        x=sig_df['time'],
                        y=sig_df[opt_col],
                        mode='markers',
                        name=sig_type.replace('_', ' ').title(),
                        marker=dict(symbol=symbol, size=10, color=color, opacity=0.7),
                        hovertemplate=f'{sig_type.replace("_", " ").title()}<br>${{y:.2f}}<extra></extra>'
                    ),
                    secondary_y=True
                )

    # Layout
    fig.update_layout(
        title=trade_label,
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=60, t=60, b=40)
    )

    fig.update_xaxes(title_text="Time", tickformat='%H:%M')
    fig.update_yaxes(title_text="Stock ($)", secondary_y=False, tickformat='$.2f')
    fig.update_yaxes(title_text="Option ($)", secondary_y=True, tickformat='$.2f')

    return fig


def get_trade_summary(df):
    """Extract concise trade summary."""
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

    return {
        'entry': entry_price,
        'exit': exit_price,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'duration': duration
    }


def get_trade_table(df):
    """Create a concise data table for the trade."""
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

    return pd.DataFrame(table_data)


def main():
    st.title("Trade Dashboard")

    matrices, _ = load_data()

    if not matrices:
        st.warning("No backtest data found. Run Test.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Trades")

        if st.button("Reload Data"):
            st.session_state.pop('matrices', None)
            st.session_state.pop('exit_signals', None)
            st.rerun()

        trade_list = []
        for pos_id, df in matrices.items():
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

        selected_idx = st.selectbox(
            "Select Trade",
            range(len(trade_list)),
            format_func=lambda i: f"{trade_list[i][0]} ({trade_list[i][2]:+.1f}%)"
        )

        st.markdown("---")
        market_hours_only = st.toggle("Market Hours (9:00-4:00)", value=True, help="ON: Full market hours view | OFF: Holding period only")
        show_all_exits = st.toggle("Show All Exit Signals", value=False)

        st.markdown("---")
        st.caption(f"{len(matrices)} trades loaded")

    # Display selected trade
    trade_label, pos_id, _ = trade_list[selected_idx]
    df = matrices[pos_id]

    # Summary metrics row
    summary = get_trade_summary(df)
    cols = st.columns(5)
    cols[0].metric("Entry", f"${summary['entry']:.2f}")
    cols[1].metric("Exit", f"${summary['exit']:.2f}")
    cols[2].metric("P&L", f"{summary['pnl_pct']:+.1f}%")
    cols[3].metric("Exit Reason", summary['exit_reason'])
    cols[4].metric("Duration", f"{summary['duration']:.0f}m")

    # Chart
    fig = create_trade_chart(df, trade_label, show_all_exits, market_hours_only)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not create chart")

    # Data table
    st.subheader("Trade Details")
    table_df = get_trade_table(df)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Matrix Data section
    st.subheader("Matrix Data")

    # Define columns to display in matrix
    matrix_cols = [
        'timestamp', 'holding', 'stock_price', 'option_price', 'pnl_pct',
        'stop_loss', 'stop_loss_mode', 'sl_cushion',
        'vwap', 'ema_20', 'ema_30',
        'SL_C1', 'SL_C2'
    ]

    # Filter to columns that exist in the dataframe
    available_cols = [col for col in matrix_cols if col in df.columns]

    if available_cols:
        matrix_df = df[available_cols].copy()

        # Calculate sl_cushion (Option_price - Stop_loss) if both columns exist
        if 'option_price' in df.columns and 'stop_loss' in df.columns:
            matrix_df['sl_cushion'] = df['option_price'] - df['stop_loss']

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

        # Format numeric columns
        for col in ['stock_price', 'option_price', 'stop_loss', 'vwap', 'ema_20', 'ema_30']:
            if col in matrix_df.columns:
                matrix_df[col] = matrix_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

        if 'pnl_pct' in matrix_df.columns:
            matrix_df['pnl_pct'] = matrix_df['pnl_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")

        if 'sl_cushion' in matrix_df.columns:
            matrix_df['sl_cushion'] = matrix_df['sl_cushion'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info("No matrix data available for this trade.")


if __name__ == "__main__":
    main()
