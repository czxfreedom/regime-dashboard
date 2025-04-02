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
    AND taker_way IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
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
    # Extract periods with both high activity and significant PNL
    high_activity_periods = []
    for pair_name, df in pair_results.items():
        if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
            # Find time periods with both significant activity and significant PNL
            for time_label, row in df.iterrows():
                trade_count = row['trade_count']
                pnl = row['platform_pnl']
                
                # Check if this is a noteworthy period
                if (trade_count >= high_trade_count_threshold or 
                    pnl >= high_pnl_threshold or pnl <= low_pnl_threshold):
                    high_activity_periods.append({
                        'Pair': pair_name,
                        'Time': time_label,
                        'Trade Count': int(trade_count),
                        'Platform PNL (USD)': round(pnl, 2),
                        'Revenue per Trade (USD)': round(pnl / trade_count, 2) if trade_count > 0 else 0
                    })
    
    # Extract periods with both high activity and significant PNL with better formatting
    if high_activity_periods:
        # Convert to DataFrame
        high_activity_df = pd.DataFrame(high_activity_periods)
        
        # Rename columns for clarity
        high_activity_df = high_activity_df.rename(columns={
            'Pair': '🔄 Trading Pair',
            'Time': '⏰ Time Period',
            'Trade Count': '📊 Number of Trades',
            'Platform PNL (USD)': '💰 PNL (USD)',
            'Revenue per Trade (USD)': '💸 PNL/Trade (USD)'
        })
        
        # Sort by Trade Count (highest first)
        high_activity_df = high_activity_df.sort_values(by='📊 Number of Trades', ascending=False)
        
        # Style the dataframe
        styled_activity_df = high_activity_df.style.format({
            '💰 PNL (USD)': '${:,.2f}',
            '💸 PNL/Trade (USD)': '${:,.2f}'
        })
        
        # Apply conditional formatting
        def highlight_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
            return ''
        
        styled_activity_df = styled_activity_df.applymap(highlight_pnl, subset=['💰 PNL (USD)', '💸 PNL/Trade (USD)'])
        
        # Add a clear section header
        st.markdown("### 🔍 High Activity Periods")
        
        # Display the dataframe with improved styling
        st.dataframe(
            styled_activity_df.set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=350,
            use_container_width=True
        )
        
        # Create visual representation
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 trading periods by volume
            top_trading_periods = high_activity_df.head(10)
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_trading_periods.iterrows()],
                    y=top_trading_periods['📊 Number of Trades'],
                    marker_color='blue'
                )
            ])
            fig.update_layout(
                title="Top 10 Trading Periods by Volume",
                xaxis_title="Pair and Time",
                yaxis_title="Number of Trades",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 10 PNL periods
            top_pnl_periods = high_activity_df.sort_values(by='💰 PNL (USD)', ascending=False).head(10)
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_pnl_periods.iterrows()],
                    y=top_pnl_periods['💰 PNL (USD)'],
                    marker_color='green'
                )
            ])
            fig.update_layout(
                title="Top 10 Trading Periods by Platform PNL",
                xaxis_title="Pair and Time",
                yaxis_title="Platform PNL (USD)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Overall Trade Count vs. PNL correlation analysis
        st.subheader("Correlation Analysis: Trade Count vs. Platform PNL")
        
        correlation_data = []
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                # Calculate correlation between trade count and PNL
                correlation = df['trade_count'].corr(df['platform_pnl'])
                # Filter out rows with zero trades
                non_zero_trades = df[df['trade_count'] > 0]
                # Calculate average PNL per trade
                avg_pnl_per_trade = non_zero_trades['platform_pnl'].sum() / non_zero_trades['trade_count'].sum() if non_zero_trades['trade_count'].sum() > 0 else 0
                
                correlation_data.append({
                    'Pair': pair_name,
                    'Correlation': round(correlation, 3) if not pd.isna(correlation) else 0,
                    'Total Trades': int(df['trade_count'].sum()),
                    'Total PNL (USD)': round(df['platform_pnl'].sum(), 2),
                    'Avg PNL per Trade (USD)': round(avg_pnl_per_trade, 3)
                })
        
        if correlation_data:
            # Convert to DataFrame
            correlation_df = pd.DataFrame(correlation_data)
            
            # Sort by correlation (highest first)
            correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
            
            # Format the dataframe for better legibility
            correlation_df = correlation_df.rename(columns={
                'Pair': '🔄 Trading Pair',
                'Correlation': '📊 Correlation',
                'Total Trades': '📈 Total Trades',
                'Total PNL (USD)': '💰 Total PNL (USD)',
                'Avg PNL per Trade (USD)': '💸 Avg PNL/Trade (USD)'
            })
            
            # Style the dataframe
            styled_correlation_df = correlation_df.style.format({
                '📊 Correlation': '{:.3f}',
                '💰 Total PNL (USD)': '${:,.2f}',
                '💸 Avg PNL/Trade (USD)': '${:,.3f}'
            })
            
            # Apply conditional formatting
            def highlight_correlation(val):
                if isinstance(val, (int, float)):
                    if val > 0.5:
                        return 'color: green; font-weight: bold'
                    elif val < -0.5:
                        return 'color: red; font-weight: bold'
                return ''
            
            styled_correlation_df = styled_correlation_df.applymap(highlight_correlation, subset=['📊 Correlation'])
            styled_correlation_df = styled_correlation_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 Avg PNL/Trade (USD)'])
            
            # Create columns for display
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Display the correlation table with improved styling
                st.markdown("### 📊 Trade Count vs. PNL Correlation by Pair")
                st.dataframe(
                    styled_correlation_df.set_properties(**{
                        'font-size': '16px',
                        'text-align': 'center',
                        'background-color': '#f0f2f6'
                    }),
                    height=400,
                    use_container_width=True
                )
            
            with col2:
                # Create a scatter plot to visualize the correlation
                # Gather all data points for the scatter plot
                scatter_data = []
                for pair_name, df in pair_results.items():
                    if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                        for time_label, row in df.iterrows():
                            if row['trade_count'] > 0:  # Only include periods with trades
                                scatter_data.append({
                                    'Pair': pair_name,
                                    'Trade Count': int(row['trade_count']),
                                    'Platform PNL (USD)': round(row['platform_pnl'], 2)
                                })
                
                if scatter_data:
                    scatter_df = pd.DataFrame(scatter_data)
                    
                    # Create a scatter plot using Plotly
                    fig = px.scatter(
                        scatter_df, 
                        x='Trade Count', 
                        y='Platform PNL (USD)',
                        color='Pair',
                        title='Trade Count vs. Platform PNL Correlation',
                        hover_data=['Pair', 'Trade Count', 'Platform PNL (USD)'],
                        trendline="ols"  # Add trend line
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title="Number of Trades",
                        yaxis_title="Platform PNL (USD)",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Add spacing for the next section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Time-based Analysis
    st.subheader("Time-based Analysis")
    
    # Analyze trading patterns over time
    hourly_patterns = {}
    for time_block in ordered_times:
        # Extract hour from time block
        hour = int(time_block.split(':')[0])
        
        # Initialize if this hour is not yet in the dictionary
        if hour not in hourly_patterns:
            hourly_patterns[hour] = {
                'total_trades': 0,
                'total_pnl': 0,
                'count': 0
            }
        
        # Sum up trades and PNL for this hour across all pairs
        for pair_name, df in pair_results.items():
            if time_block in df.index:
                if 'trade_count' in df.columns:
                    hourly_patterns[hour]['total_trades'] += df.at[time_block, 'trade_count']
                if 'platform_pnl' in df.columns:
                    hourly_patterns[hour]['total_pnl'] += df.at[time_block, 'platform_pnl']
                hourly_patterns[hour]['count'] += 1
    
    # Convert to DataFrame for display
    hourly_patterns_df = pd.DataFrame([
        {
            'Hour (SG Time)': f"{hour:02d}:00-{hour:02d}:59",
            'Avg Trades': round(data['total_trades'] / data['count'] if data['count'] > 0 else 0, 1),
            'Avg PNL (USD)': round(data['total_pnl'] / data['count'] if data['count'] > 0 else 0, 2),
            'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'] if data['total_trades'] > 0 else 0, 3)
        }
        for hour, data in hourly_patterns.items()
    ])
    
    # Sort by hour for display
    hourly_patterns_df = hourly_patterns_df.sort_values(by='Hour (SG Time)')
    
    # Format the dataframe for better legibility
    hourly_patterns_df = hourly_patterns_df.rename(columns={
        'Hour (SG Time)': '🕒 Hour (SG Time)',
        'Avg Trades': '📊 Avg Trades',
        'Avg PNL (USD)': '💰 Avg PNL (USD)',
        'PNL per Trade (USD)': '💸 PNL/Trade (USD)'
    })
    
    # Style the dataframe
    styled_hourly_df = hourly_patterns_df.style.format({
        '📊 Avg Trades': '{:.1f}',
        '💰 Avg PNL (USD)': '${:,.2f}',
        '💸 PNL/Trade (USD)': '${:,.3f}'
    })
    
    # Apply conditional formatting
    styled_hourly_df = styled_hourly_df.applymap(highlight_pnl, subset=['💰 Avg PNL (USD)', '💸 PNL/Trade (USD)'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕒 Hourly Trading Patterns")
        st.dataframe(
            styled_hourly_df.set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=500,
            use_container_width=True
        )
    
    with col2:
        # Create charts for hourly patterns
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_patterns_df['🕒 Hour (SG Time)'],
            y=hourly_patterns_df['📊 Avg Trades'],
            name='Avg Trades',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_patterns_df['🕒 Hour (SG Time)'],
            y=hourly_patterns_df['💰 Avg PNL (USD)'],
            name='Avg PNL (USD)',
            yaxis='y2',
            mode='lines+markers',
            marker_color='green',
            line=dict(width=3)
        ))
        
        # Update layout with two y-axes
        fig.update_layout(
            title="Hourly Trading Activity and PNL (Singapore Time)",
            xaxis=dict(title="Hour"),
            yaxis=dict(
                title="Avg Number of Trades",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue")
            ),
            yaxis2=dict(
                title="Avg PNL (USD)",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add spacing for the next section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # PNL Breakdown Analysis
    st.subheader("Platform Profit Distribution")
    
    # Calculate platform total profit
    total_platform_profit = sum(df['platform_pnl'].sum() for pair, df in pair_results.items() if 'platform_pnl' in df.columns)
    
    # Calculate per-pair contribution to total profit
    profit_distribution = []
    for pair_name, df in pair_results.items():
        if 'platform_pnl' in df.columns:
            pair_pnl = df['platform_pnl'].sum()
            contribution_pct = 100 * pair_pnl / total_platform_profit if total_platform_profit != 0 else 0
            
            profit_distribution.append({
                'Pair': pair_name,
                'Total PNL (USD)': round(pair_pnl, 0),
                'Contribution (%)': round(contribution_pct, 2)
            })
    
    if profit_distribution:
        # Sort by total PNL (highest first)
        profit_distribution_df = pd.DataFrame(profit_distribution)
        profit_distribution_df = profit_distribution_df.sort_values(by='Total PNL (USD)', ascending=False)
        
        # Format the dataframe for better legibility
        profit_distribution_df = profit_distribution_df.rename(columns={
            'Pair': '🔄 Trading Pair',
            'Total PNL (USD)': '💰 Total PNL (USD)',
            'Contribution (%)': '📊 Contribution (%)'
        })
        
        # Apply styling to make numbers clearer
        styled_profit_df = profit_distribution_df.style.format({
            '💰 Total PNL (USD)': '${:,.0f}',  # Format as integers with comma for thousands
            '📊 Contribution (%)': '{:+.2f}%'  # Show with + or - sign and 2 decimal places
        })
    
        # Conditionally color the cells based on values
        def color_pnl_and_contribution(val, column):
            if column == '💰 Total PNL (USD)':
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
                return ''
            elif column == '📊 Contribution (%)':
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
                return ''
            return ''
    
        # Apply the styling function
        styled_profit_df = styled_profit_df.applymap(
            lambda x: color_pnl_and_contribution(x, '💰 Total PNL (USD)'), 
            subset=['💰 Total PNL (USD)']
         ).applymap(
            lambda x: color_pnl_and_contribution(x, '📊 Contribution (%)'), 
            subset=['📊 Contribution (%)']
        )
        
        # Display the styled dataframe
        st.markdown("### 💰 Profit Contribution by Pair")
        st.dataframe(
            styled_profit_df.set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=500,
            use_container_width=True
        )
        
        # Create a pie chart visualizing contribution
        top_pairs = profit_distribution_df.head(10)
        
        # Filter to only positive contributions for the pie chart
        positive_pairs = top_pairs[top_pairs['💰 Total PNL (USD)'] > 0]
        if not positive_pairs.empty:
            fig = px.pie(
                positive_pairs, 
                values='💰 Total PNL (USD)', 
                names='🔄 Trading Pair',
                title='Top 10 Pairs by Positive PNL Contribution',
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Add spacing for the next section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Identify Most Profitable Time Periods
    st.subheader("Most Profitable Time Periods")
    
    # Calculate profitability for each time period across all pairs
    time_period_profit = {}
    for time_block in ordered_times:
        time_period_profit[time_block] = {
            'total_pnl': 0,
            'total_trades': 0,
            'pair_breakdown': {}
        }
        
        for pair_name, df in pair_results.items():
            if time_block in df.index:
                if 'platform_pnl' in df.columns:
                    pair_pnl = df.at[time_block, 'platform_pnl']
                    time_period_profit[time_block]['total_pnl'] += pair_pnl
                    time_period_profit[time_block]['pair_breakdown'][pair_name] = pair_pnl
                
                if 'trade_count' in df.columns:
                    time_period_profit[time_block]['total_trades'] += df.at[time_block, 'trade_count']
    
    # Convert to DataFrame for display
    time_profit_df = pd.DataFrame([
        {
            'Time Period': time_block,
            'Total PNL (USD)': round(data['total_pnl'], 2),
            'Total Trades': int(data['total_trades']),
            'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'], 3) if data['total_trades'] > 0 else 0,
            'Top Contributing Pair': max(data['pair_breakdown'].items(), key=lambda x: x[1])[0] if data['pair_breakdown'] else "None"
        }
        for time_block, data in time_period_profit.items()
    ])
    
    # Format the dataframe for better legibility
    time_profit_df = time_profit_df.rename(columns={
        'Time Period': '⏰ Time Period',
        'Total PNL (USD)': '💰 Total PNL (USD)',
        'Total Trades': '📊 Total Trades',
        'PNL per Trade (USD)': '💸 PNL/Trade (USD)',
        'Top Contributing Pair': '🔄 Top Pair'
    })
    
    # Sort by total PNL (highest first)
    time_profit_df = time_profit_df.sort_values(by='💰 Total PNL (USD)', ascending=False)
    
    # Style the dataframe
    styled_time_profit_df = time_profit_df.style.format({
        '💰 Total PNL (USD)': '${:,.2f}',
        '💸 PNL/Trade (USD)': '${:,.3f}'
    })
    
    # Apply conditional formatting
    styled_time_profit_df = styled_time_profit_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 PNL/Trade (USD)'])
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        # Show top profitable time periods
        st.markdown("### 📈 Top 10 Most Profitable Time Periods")
        st.dataframe(
            styled_time_profit_df.head(10).set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=300,
            use_container_width=True
        )
    
    with col2:
        # Show bottom profitable (loss-making) time periods
        st.markdown("### 📉 Top 10 Least Profitable Time Periods")
        least_profitable = styled_time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')
        st.dataframe(
            least_profitable.set_properties(**{
                'font-size': '16px',
                'text-align': 'center',
                'background-color': '#f0f2f6'
            }),
            height=300,
            use_container_width=True
        )
    
    # Create visualization of top profitable and loss-making periods
    fig = go.Figure()
    
    # Top 10 profitable periods
    fig.add_trace(go.Bar(
        x=time_profit_df.head(10)['⏰ Time Period'],
        y=time_profit_df.head(10)['💰 Total PNL (USD)'],
        name='Top Profitable Periods',
        marker_color='green'
    ))
    
    # Bottom 10 profitable periods
    bottom_10 = time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')
    fig.add_trace(go.Bar(
        x=bottom_10['⏰ Time Period'],
        y=bottom_10['💰 Total PNL (USD)'],
        name='Least Profitable Periods',
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Most and Least Profitable Time Periods",
        xaxis_title="Time Period (SG Time)",
        yaxis_title="Total PNL (USD)",
        height=500,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation for dashboard
    with st.expander("Understanding the Trading & PNL Dashboard"):
        st.markdown("""
        ## 📊 How to Use This Dashboard
        
        This dashboard shows trading activity and platform profit/loss (PNL) across all selected trading pairs using 30-minute intervals over the past 24 hours (Singapore time).
        
        ### Main Tables
        - **User Trades Table**: Shows the number of trades completed in each 30-minute period
        - **Platform PNL Table**: Shows the platform's profit/loss in each 30-minute period
        
        ### Color Coding
        - **Trades Table**: 
          - 🟩 Green: Low activity
          - 🟨 Yellow: Medium activity
          - 🟧 Orange: High activity
          - 🟥 Red: Very high activity
        
        - **PNL Table**: 
          - 🟥 Red: Significant loss
          - 🟠 Light red: Small loss
          - 🟢 Light green: Small profit
          - 🟩 Green: Significant profit
        
        ### Key Insights
        - **Trading Activity Summary**: See which pairs have the highest trading volume
        - **Platform PNL Summary**: Identify which pairs are most profitable
        - **High Activity Periods**: Spot times of significant trading and PNL outcomes
        - **Correlation Analysis**: Understand the relationship between trade volume and profitability
        - **Time-based Analysis**: Discover trading patterns throughout the day
        - **Profit Distribution**: See which pairs contribute most to overall profit
        - **Time Period Analysis**: Identify the most and least profitable time periods
        
        ### Technical Details
        - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
        - All values are shown in USD
        - The dashboard refreshes when you click the "Refresh Data" button
        - Singapore timezone (UTC+8) is used throughout
        """
        )

else:
    st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")