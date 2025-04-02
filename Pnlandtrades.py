# Save this as pages/06_Trades_PNL_Table.py in your Streamlit app folder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="User Trades & Platform PNL Table",
    page_icon="💰",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("User Trades & Platform PNL Table (30min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
expected_points = 48  # Expected data points per pair over 24 hours (24 hours * 2 intervals per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set thresholds for highlighting
high_trade_count_threshold = 100  # Number of trades considered "high activity"
high_pnl_threshold = 1000  # Platform PNL amount considered "high" (in USD)
low_pnl_threshold = -1000  # Platform PNL amount considered "low" (in USD)

# Fetch all available pairs from DB
@st.cache_data(ttl=600, show_spinner="Fetching pairs...")
def fetch_all_pairs():
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

all_pairs = fetch_all_pairs()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all = st.checkbox("Select All Pairs", value=True)
    
    if select_all:
        selected_pairs = all_pairs
    else:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            all_pairs,
            default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to generate aligned 30-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 30-minute time blocks for past 24 hours,
    aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
    """
    # Round down to the nearest 30-minute mark
    if current_time.minute < 30:
        # Round down to XX:00
        latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Round down to XX:30
        latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(48):  # 24 hours of 30-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Fetch trades data for the past 24 hours in 30min intervals
@st.cache_data(ttl=600, show_spinner="Fetching trade counts...")
def fetch_trade_counts(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Updated query to use trade_fill_fresh with consistent time handling
    query = f"""
    SELECT
        date_trunc('hour', created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore')::INT / 30) 
        AS timestamp,
        COUNT(*) AS trade_count
    FROM public.trade_fill_fresh
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
    GROUP BY
        date_trunc('hour', created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore')::INT / 30)
    ORDER BY timestamp
    """
    
    try:
        print(f"[{pair_name}] Executing trade count query")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] Trade count query executed. DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"[{pair_name}] No trade data found.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df.index.strftime('%H:%M')
        
        return df
    except Exception as e:
        st.error(f"Error processing trade counts for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing trade counts: {e}")
        return None
    
# Fetch platform PNL data for the past 24 hours in 30min intervals
@st.cache_data(ttl=600, show_spinner="Calculating platform PNL...")
def fetch_platform_pnl(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # This query combines order PNL, fee data, funding PNL, and rebate data in 30-minute intervals
    query = f"""
    WITH time_intervals AS (
      -- Generate 30-minute intervals for the past 24 hours
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 30),
          '{end_time_utc}'::timestamp,
          INTERVAL '30 minutes'
        ) AS "UTC+8"
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" IN (0, 1, 2, 3, 4)
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    fee_data AS (
      -- Calculate user fee payments
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    funding_pnl AS (
      -- Calculate platform funding fee PNL
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" = 0
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    rebate_data AS (
      -- Calculate platform rebate payments
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
      FROM
        "public"."user_cashbooks"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "remark" = '给邀请人返佣'
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    )
    
    -- Final query: combine all data sources
    SELECT
      t."UTC+8" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp",
      COALESCE(o."platform_order_pnl", 0) +
      COALESCE(f."user_fee_payments", 0) +
      COALESCE(ff."platform_funding_pnl", 0) +
      COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
    FROM
      time_intervals t
    LEFT JOIN
      order_pnl o ON t."UTC+8" = o."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      fee_data f ON t."UTC+8" = f."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      funding_pnl ff ON t."UTC+8" = ff."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      rebate_data r ON t."UTC+8" = r."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      t."UTC+8" DESC
    """
    
    try:
        print(f"[{pair_name}] Executing platform PNL query")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] Platform PNL query executed. DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"[{pair_name}] No PNL data found.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df.index.strftime('%H:%M')
        
        return df
    except Exception as e:
        st.error(f"Error processing platform PNL for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing platform PNL: {e}")
        return None

# Combine Trade Count and Platform PNL data for visualization
def combine_data(trade_data, pnl_data):
    if trade_data is None and pnl_data is None:
        return None
    
    # Create a DataFrame with time blocks as index
    time_blocks = pd.DataFrame(index=[block[2] for block in aligned_time_blocks])
    
    # Add trade count data if available
    if trade_data is not None and not trade_data.empty:
        for time_label in time_blocks.index:
            # Find matching rows in trade_data by time_label
            matching_rows = trade_data[trade_data['time_label'] == time_label]
            if not matching_rows.empty:
                time_blocks.at[time_label, 'trade_count'] = matching_rows['trade_count'].sum()
    
    # Add PNL data if available
    if pnl_data is not None and not pnl_data.empty:
        for time_label in time_blocks.index:
            # Find matching rows in pnl_data by time_label
            matching_rows = pnl_data[pnl_data['time_label'] == time_label]
            if not matching_rows.empty:
                time_blocks.at[time_label, 'platform_pnl'] = matching_rows['platform_total_pnl'].sum()
    
    # Fill NaN values with 0
    time_blocks.fillna(0, inplace=True)
    
    return time_blocks

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate trade count and platform PNL for each pair
pair_results = {}
for i, pair_name in enumerate(selected_pairs):
    try:
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
        
        # Fetch trade count data
        trade_data = fetch_trade_counts(pair_name)
        
        # Fetch platform PNL data
        pnl_data = fetch_platform_pnl(pair_name)
        
        # Combine data
        combined_data = combine_data(trade_data, pnl_data)
        
        if combined_data is not None:
            pair_results[pair_name] = combined_data
    except Exception as e:
        st.error(f"Error processing pair {pair_name}: {e}")
        print(f"Error processing pair {pair_name} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

# Create tables for display - Trade Count Table
if pair_results:
    # Create trade count table data
    trade_count_data = {}
    for pair_name, df in pair_results.items():
        if 'trade_count' in df.columns:
            trade_count_data[pair_name] = df['trade_count']
    
    # Create DataFrame with all pairs
    trade_count_table = pd.DataFrame(trade_count_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(trade_count_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    trade_count_table = trade_count_table.reindex(ordered_times)
    
    # Round to integers - trade counts should be whole numbers
    trade_count_table = trade_count_table.round(0).astype('Int64')  # Using Int64 to handle NaN values properly
    
    def color_trade_cells(val):
        if pd.isna(val) or val == 0:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
        elif val < 10:  # Low activity
            intensity = max(0, min(255, int(255 * val / 10)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 50:  # Medium activity
            intensity = max(0, min(255, int(255 * (val - 10) / 40)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < high_trade_count_threshold:  # High activity
            intensity = max(0, min(255, int(255 * (val - 50) / 50)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Very high activity
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    styled_trade_table = trade_count_table.style.applymap(color_trade_cells)
    st.markdown("## User Trades Table (30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Low Activity</span>, <span style='color:#aaaa00'>Medium Activity</span>, <span style='color:orange'>High Activity</span>, <span style='color:red'>Very High Activity</span>", unsafe_allow_html=True)
    st.markdown("Values shown as number of trades per 30-minute period")
    st.dataframe(styled_trade_table, height=700, use_container_width=True)
    
    # Create Platform PNL table data
    pnl_data = {}
    for pair_name, df in pair_results.items():
        if 'platform_pnl' in df.columns:
            if df['platform_pnl'].abs().sum() > 0:
                pnl_data[pair_name] = df['platform_pnl']
    
    # Create DataFrame with all pairs
    pnl_table = pd.DataFrame(pnl_data)
    
    # Apply the time blocks in the proper order (most recent first)
    pnl_table = pnl_table.reindex(ordered_times)
    
    # Round to 2 decimal places for display
    pnl_table = pnl_table.round(0).astype(int)
    
    def color_pnl_cells(val):
        if pd.isna(val) or val == 0:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
        elif val < low_pnl_threshold:  # Large negative PNL (loss) - red
            return f'background-color: rgba(255, 0, 0, 0.9); color: white'
        elif val < 0:  # Small negative PNL (loss) - light red
            intensity = max(0, min(255, int(255 * abs(val) / abs(low_pnl_threshold))))
            return f'background-color: rgba(255, {100-intensity}, {100-intensity}, 0.9); color: black'
        elif val < high_pnl_threshold:  # Small positive PNL (profit) - light green
            intensity = max(0, min(255, int(255 * val / high_pnl_threshold)))
            return f'background-color: rgba({100-intensity}, 180, {100-intensity}, 0.9); color: black'
        else:  # Large positive PNL (profit) - green
            return 'background-color: rgba(0, 120, 0, 0.7); color: black'
    
    styled_pnl_table = pnl_table.style.applymap(color_pnl_cells)
    st.markdown("## Platform PNL Table (USD, 30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:red'>Loss</span>, <span style='color:#ff9999'>Small Loss</span>, <span style='color:#99ff99'>Small Profit</span>, <span style='color:green'>Large Profit</span>", unsafe_allow_html=True)
    st.markdown("Values shown in USD")
    st.dataframe(styled_pnl_table, height=700, use_container_width=True)
    
    # Create summary tables with improved legibility
    st.subheader("Summary Statistics (Last 24 Hours)")
    
    # Add a separator
    st.markdown("---")
    
    # Prepare Trades Summary data
    trades_summary = {}
    for pair_name, df in pair_results.items():
        if 'trade_count' in df.columns:
            total_trades = df['trade_count'].sum()
            max_trades = df['trade_count'].max()
            max_trades_time = df['trade_count'].idxmax() if max_trades > 0 else "N/A"
            
            trades_summary[pair_name] = {
                'Total Trades': int(total_trades),
                'Max Trades in 30min': int(max_trades),
                'Busiest Time': max_trades_time if max_trades > 0 else "N/A",
                'Avg Trades per 30min': round(df['trade_count'].mean(), 1)
            }

    # Trading Activity Summary with improved formatting
    if trades_summary:
        # Convert to DataFrame and sort
        trades_summary_df = pd.DataFrame(trades_summary).T
        trades_summary_df = trades_summary_df.sort_values(by='Total Trades', ascending=False)
        
        # Format the dataframe for better legibility
        trades_summary_df = trades_summary_df.rename(columns={
            'Total Trades': '📊 Total Trades',
            'Max Trades in 30min': '⏱️ Max Trades (30min)',
            'Busiest Time': '🕒 Busiest Time',
            'Avg Trades per 30min': '📈 Avg Trades/30min'
        })
        
        # Add a clear section header
        st.markdown("### 📊 Trading Activity Summary")
        
        # Use a larger height and make sure the table has proper spacing
        st.dataframe(
            trades_summary_df.style.format({
                '📈 Avg Trades/30min': '{:.1f}'
            }).set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=350,
            use_container_width=True
        )

    # Add spacing between tables
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Prepare PNL Summary data
    pnl_summary = {}
    for pair_name, df in pair_results.items():
        if 'platform_pnl' in df.columns:
            total_pnl = df['platform_pnl'].sum()
            max_pnl = df['platform_pnl'].max()
            min_pnl = df['platform_pnl'].min()
            max_pnl_time = df['platform_pnl'].idxmax() if abs(max_pnl) > 0 else "N/A"
            min_pnl_time = df['platform_pnl'].idxmin() if abs(min_pnl) > 0 else "N/A"
            
            pnl_summary[pair_name] = {
                'Total PNL (USD)': round(total_pnl, 2),
                'Max Profit in 30min': round(max_pnl, 2),
                'Max Profit Time': max_pnl_time if abs(max_pnl) > 0 else "N/A",
                'Max Loss in 30min': round(min_pnl, 2),
                'Max Loss Time': min_pnl_time if abs(min_pnl) > 0 else "N/A",
                'Avg PNL per 30min': round(df['platform_pnl'].mean(), 2)
            }

    # PNL Summary with improved formatting
    if pnl_summary:
        # Convert to DataFrame and sort
        pnl_summary_df = pd.DataFrame(pnl_summary).T
        pnl_summary_df = pnl_summary_df.sort_values(by='Total PNL (USD)', ascending=False)
        
        # Format the dataframe for better legibility
        pnl_summary_df = pnl_summary_df.rename(columns={
            'Total PNL (USD)': '💰 Total PNL (USD)',
            'Max Profit in 30min': '📈 Max Profit (30min)',
            'Max Profit Time': '⏱️ Max Profit Time',
            'Max Loss in 30min': '📉 Max Loss (30min)',
            'Max Loss Time': '⏱️ Max Loss Time', 
            'Avg PNL per 30min': '📊 Avg PNL/30min'
        })
        
        # Add a clear section header
        st.markdown("### 💰 Platform PNL Summary")
        
        # Style the dataframe for better legibility
        styled_pnl_df = pnl_summary_df.style.format({
            '💰 Total PNL (USD)': '${:,.2f}',
            '📈 Max Profit (30min)': '${:,.2f}',
            '📉 Max Loss (30min)': '${:,.2f}',
            '📊 Avg PNL/30min': '${:,.2f}'
        })
        
        # Apply conditional formatting
        def highlight_profits(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
            return ''
        
        styled_pnl_df = styled_pnl_df.applymap(highlight_profits, subset=['💰 Total PNL (USD)', '📈 Max Profit (30min)', '📉 Max Loss (30min)', '📊 Avg PNL/30min'])
        
        # Use a larger height and make sure the table has proper spacing
        st.dataframe(
            styled_pnl_df.set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=350,
            use_container_width=True
        )

    # Add spacing for the next section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Combined Analysis with improved formatting
    st.subheader("Combined Analysis: Trading Activity vs. Platform PNL")