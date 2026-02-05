"""
DiscordMessages.py - Discord Messages DataFrame

Creates a dataframe of all Discord messages aligned in a single data matrix.
Variable 'discord_messages_df' is exposed for variable explorer visibility.

Usage:
    from DiscordMessages import discord_messages_df, fetch_messages

    # Access the dataframe directly
    discord_messages_df

    # Or fetch fresh messages
    df = fetch_messages(days=5)

================================================================================
"""

import pandas as pd
import datetime as dt
from datetime import timedelta
from zoneinfo import ZoneInfo

import Config
from Test import DiscordFetcher

EASTERN = ZoneInfo('America/New_York')

# Module-level variable for variable explorer visibility
discord_messages_df = pd.DataFrame()


def fetch_messages(days=5, channel_id=None):
    """
    Fetch Discord messages and return as aligned DataFrame.

    Args:
        days: Number of days to look back
        channel_id: Optional channel ID override

    Returns:
        DataFrame with columns: id, timestamp, content, author, author_id,
                               message_length, has_alert, date, time, hour
    """
    global discord_messages_df

    # Initialize fetcher
    fetcher = DiscordFetcher(channel_id=channel_id)

    # Fetch messages
    print(f"Fetching Discord messages for last {days} days...")
    df = fetcher.fetch_messages_for_days(days=days)

    # Close session
    fetcher.close()

    if df.empty:
        print("No messages found")
        discord_messages_df = df
        return df

    # Add derived columns for analysis
    alert_marker = Config.DISCORD_CONFIG.get('alert_marker', '')

    df['message_length'] = df['content'].str.len()
    df['has_alert'] = df['content'].str.contains(alert_marker, regex=False)
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()

    # Align messages by index (row number = message sequence)
    df = df.reset_index(drop=True)
    df.index.name = 'message_num'

    # Update module-level variable
    discord_messages_df = df

    print(f"Loaded {len(df)} messages into discord_messages_df")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Messages with alerts: {df['has_alert'].sum()}")

    return df


def get_content_matrix(df=None):
    """
    Create a content matrix with message text aligned by row.

    Returns:
        DataFrame where each row is a message, columns include
        parsed content fields.
    """
    if df is None:
        df = discord_messages_df

    if df.empty:
        print("No messages loaded. Call fetch_messages() first.")
        return pd.DataFrame()

    # Create content matrix with essential fields
    content_df = df[['timestamp', 'author', 'content', 'has_alert']].copy()
    content_df = content_df.reset_index(drop=True)

    return content_df


def get_alerts_only(df=None):
    """
    Filter to only messages containing alert markers.

    Returns:
        DataFrame with only alert messages.
    """
    if df is None:
        df = discord_messages_df

    if df.empty:
        print("No messages loaded. Call fetch_messages() first.")
        return pd.DataFrame()

    alerts_df = df[df['has_alert'] == True].copy()
    alerts_df = alerts_df.reset_index(drop=True)

    print(f"Found {len(alerts_df)} alert messages")
    return alerts_df


def summary():
    """Print summary of loaded messages."""
    if discord_messages_df.empty:
        print("No messages loaded. Call fetch_messages() first.")
        return

    df = discord_messages_df

    print("\n" + "="*60)
    print("DISCORD MESSAGES SUMMARY")
    print("="*60)
    print(f"\nTotal Messages: {len(df)}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Messages with Alerts: {df['has_alert'].sum()}")
    print(f"\nMessages by Author:")
    for author, count in df['author'].value_counts().head(10).items():
        print(f"  {author}: {count}")
    print(f"\nMessages by Day:")
    for day, count in df['day_of_week'].value_counts().items():
        print(f"  {day}: {count}")
    print(f"\nMessages by Hour:")
    hourly = df.groupby('hour').size()
    for hour, count in hourly.items():
        print(f"  {hour:02d}:00 - {count}")
    print("="*60 + "\n")


# Auto-load on import if running in interactive mode
if __name__ == '__main__':
    # Fetch last 5 days of messages when run directly
    discord_messages_df = fetch_messages(days=5)
    summary()
