import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Trading Pairs PNL Dashboard",
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
st.title("Trading Pairs PNL Dashboard")
st.subheader("Performance Analysis by Time Period (Singapore Time)")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to get Singapore midnight today, yesterday, and 7 days ago
def get_time_boundaries():
    # Current time in Singapore
    now_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)
    
    # Today's midnight in Singapore
    today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Yesterday's midnight in Singapore
    yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)
    
    # 7 days ago midnight in Singapore
    week_ago_midnight_sg = today_midnight_sg - timedelta(days=7)
    
    # All time (use a far past date, e.g., 5 years ago)
    all_time_start_sg = today_midnight_sg.replace(year=today_midnight_sg.year-5)
    
    # Convert all times back to UTC for database queries
    today_midnight_utc = today_midnight_sg.astimezone(pytz.utc)
    yesterday_midnight_utc = yesterday_midnight_sg.astimezone(pytz.utc)
    day_before_yesterday_midnight_utc = (yesterday_midnight_sg - timedelta(days=1)).astimezone(pytz.utc)
    week_ago_midnight_utc = week_ago_midnight_sg.astimezone(pytz.utc)
    all_time_start_utc = all_time_start_sg.astimezone(pytz.utc)
    now_utc = now_sg.astimezone(pytz.utc)
    
    return {
        "today": {
            "start": today_midnight_utc,
            "end": now_utc,
            "label": f"Today ({today_midnight_sg.strftime('%Y-%m-%d')})"
        },
        "yesterday": {
            "start": yesterday_midnight_utc,
            "end": today_midnight_utc,
            "label": f"Yesterday ({yesterday_midnight_sg.strftime('%Y-%m-%d')})"
        },
        "day_before_yesterday": {
            "start": day_before_yesterday_midnight_utc,
            "end": yesterday_midnight_utc,
            "label": f"Day Before ({(yesterday_midnight_sg - timedelta(days=1)).strftime('%Y-%m-%d')})"
        },
        "this_week": {
            "start": week_ago_midnight_utc,
            "end": now_utc,
            "label": f"This Week (Last 7 Days)"
        },
        "all_time": {
            "start": all_time_start_utc,
            "end": now_utc,
            "label": "All Time"
        }
    }

# Calculate time boundaries
time_boundaries = get_time_boundaries()

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

# Function to fetch PNL data for a specific time period
@st.cache_data(ttl=600)
def fetch_pnl_data(pair_name, start_time, end_time):
    """Fetch platform PNL data for a specific time period."""
    
    query = f"""
    WITH order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" IN (0, 1, 2, 3, 4)
    ),
    
    fee_data AS (
      -- Calculate user fee payments
      SELECT
        COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
    ),
    
    funding_pnl AS (
      -- Calculate platform funding fee PNL
      SELECT
        COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" = 0
    ),
    
    rebate_data AS (
      -- Calculate platform rebate payments
      SELECT
        COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
      FROM
        "public"."user_cashbooks"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "remark" = '给邀请人返佣'
    ),
    
    trade_count AS (
      -- Calculate total number of trades
      SELECT
        COUNT(*) AS "total_trades"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
    )
    
    -- Final query: combine all data sources
    SELECT
      (SELECT "platform_order_pnl" FROM order_pnl) +
      (SELECT "user_fee_payments" FROM fee_data) +
      (SELECT "platform_funding_pnl" FROM funding_pnl) +
      (SELECT "platform_rebate_payments" FROM rebate_data) AS "platform_total_pnl",
      (SELECT "total_trades" FROM trade_count) AS "total_trades"
    """
    
    try:
        print(f"[{pair_name}] Executing PNL query for period {start_time} to {end_time}")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] PNL query executed. Result: {df.iloc[0]['platform_total_pnl']}")
        
        if df.empty:
            return {"pnl": 0, "trades": 0}
        
        return {
            "pnl": float(df.iloc[0]['platform_total_pnl']),
            "trades": int(df.iloc[0]['total_trades'])
        }
    except Exception as e:
        st.error(f"Error processing PNL for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing PNL: {e}")
        return {"pnl": 0, "trades": 0}

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Gather PNL data for all pairs and time periods
results = {}
periods = ["today", "yesterday", "day_before_yesterday", "this_week", "all_time"]

for i, pair_name in enumerate(selected_pairs):
    progress_bar.progress((i) / len(selected_pairs))
    status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
    
    pair_data = {"pair_name": pair_name}
    
    for period in periods:
        start_time = time_boundaries[period]["start"]
        end_time = time_boundaries[period]["end"]
        
        # Fetch PNL data for this pair and time period
        period_data = fetch_pnl_data(pair_name, start_time, end_time)
        
        # Store the results
        pair_data[f"{period}_pnl"] = period_data["pnl"]
        pair_data[f"{period}_trades"] = period_data["trades"]
    
    results[pair_name] = pair_data

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(results)}/{len(selected_pairs)} pairs successfully")

# Create DataFrame from results
pnl_df = pd.DataFrame([results[pair] for pair in selected_pairs])

# If DataFrame is empty, show warning and stop
if pnl_df.empty:
    st.warning("No PNL data found for the selected pairs and time periods.")
    st.stop()

# Reformat the DataFrame for display
display_df = pd.DataFrame({
    'Trading Pair': pnl_df['pair_name'],
    'Today PNL (USD)': pnl_df['today_pnl'].round(2),
    'Today Trades': pnl_df['today_trades'],
    'Yesterday PNL (USD)': pnl_df['yesterday_pnl'].round(2),
    'Yesterday Trades': pnl_df['yesterday_trades'],
    'Day Before PNL (USD)': pnl_df['day_before_yesterday_pnl'].round(2),
    'Day Before Trades': pnl_df['day_before_yesterday_trades'],
    'Week PNL (USD)': pnl_df['this_week_pnl'].round(2),
    'Week Trades': pnl_df['this_week_trades'],
    'All Time PNL (USD)': pnl_df['all_time_pnl'].round(2),
    'All Time Trades': pnl_df['all_time_trades'],
})

# Function to format display DataFrame
def format_display_df(df):
    # Add derived columns
    if 'Today PNL (USD)' in df.columns and 'Today Trades' in df.columns:
        df['Today PNL/Trade'] = (df['Today PNL (USD)'] / df['Today Trades']).round(2)
        df['Today PNL/Trade'] = df['Today PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
    
    if 'Yesterday PNL (USD)' in df.columns and 'Yesterday Trades' in df.columns:
        df['Yesterday PNL/Trade'] = (df['Yesterday PNL (USD)'] / df['Yesterday Trades']).round(2)
        df['Yesterday PNL/Trade'] = df['Yesterday PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
    
    if 'Day Before PNL (USD)' in df.columns and 'Day Before Trades' in df.columns:
        df['Day Before PNL/Trade'] = (df['Day Before PNL (USD)'] / df['Day Before Trades']).round(2)
        df['Day Before PNL/Trade'] = df['Day Before PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
    
    if 'Week PNL (USD)' in df.columns and 'Week Trades' in df.columns:
        df['Week PNL/Trade'] = (df['Week PNL (USD)'] / df['Week Trades']).round(2)
        df['Week PNL/Trade'] = df['Week PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
    
    if 'All Time PNL (USD)' in df.columns and 'All Time Trades' in df.columns:
        df['All Time PNL/Trade'] = (df['All Time PNL (USD)'] / df['All Time Trades']).round(2)
        df['All Time PNL/Trade'] = df['All Time PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
    
    # Calculate daily average
    if 'All Time PNL (USD)' in df.columns:
        # Assuming All Time is 5 years (1825 days) - this is an approximation
        df['Avg Daily PNL'] = (df['All Time PNL (USD)'] / 1825).round(2)
    
    return df

# Format the display DataFrame
display_df = format_display_df(display_df)

# Sort DataFrame by Today's PNL (descending)
display_df = display_df.sort_values(by='Today PNL (USD)', ascending=False)

# Function to color cells based on value
def color_pnl_cells(val):
    """Color cells based on PNL value."""
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
    elif val < -1000:  # Large negative PNL (loss) - red
        return 'background-color: rgba(255, 0, 0, 0.9); color: white'
    elif val < 0:  # Small negative PNL (loss) - light red
        intensity = max(0, min(255, int(255 * abs(val) / 1000)))
        return f'background-color: rgba(255, {180-intensity}, {180-intensity}, 0.7); color: black'
    elif val < 1000:  # Small positive PNL (profit) - light green
        intensity = max(0, min(255, int(255 * val / 1000)))
        return f'background-color: rgba({180-intensity}, 255, {180-intensity}, 0.7); color: black'
    else:  # Large positive PNL (profit) - green
        return 'background-color: rgba(0, 200, 0, 0.8); color: black'

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Main Dashboard", "Detailed View", "Statistics & Insights"])

with tab1:
    # Main Dashboard View
    st.subheader("PNL Overview by Trading Pair")
    
    # Create a simplified display DataFrame for the main dashboard
    main_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']]
    
    # Apply styling
    styled_df = main_df.style.applymap(
        color_pnl_cells, 
        subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']
    ).format({
        'Today PNL (USD)': '${:,.2f}',
        'Yesterday PNL (USD)': '${:,.2f}',
        'Week PNL (USD)': '${:,.2f}',
        'All Time PNL (USD)': '${:,.2f}'
    })
    
    # Display the styled DataFrame
    st.dataframe(styled_df, height=600, use_container_width=True)
    
    # Create summary cards
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_today_pnl = display_df['Today PNL (USD)'].sum()
        st.metric(
            "Total Today PNL", 
            f"${total_today_pnl:,.2f}", 
            delta=f"{(total_today_pnl - display_df['Yesterday PNL (USD)'].sum()):,.2f}"
        )
    
    with col2:
        total_yesterday_pnl = display_df['Yesterday PNL (USD)'].sum()
        st.metric(
            "Total Yesterday PNL", 
            f"${total_yesterday_pnl:,.2f}"
        )
    
    with col3:
        total_week_pnl = display_df['Week PNL (USD)'].sum()
        daily_avg = total_week_pnl / 7
        st.metric(
            "Week PNL (7 days)", 
            f"${total_week_pnl:,.2f}",
            delta=f"${daily_avg:,.2f}/day"
        )
    
    with col4:
        total_all_time_pnl = display_df['All Time PNL (USD)'].sum()
        st.metric(
            "All Time PNL", 
            f"${total_all_time_pnl:,.2f}"
        )
    
    # Create a visualization of top and bottom performers today
    st.subheader("Today's Top Performers")
    
    # Filter out zero PNL pairs
    non_zero_today = display_df[display_df['Today PNL (USD)'] != 0].copy()
    
    # Get top 5 and bottom 5 performers
    top_5 = non_zero_today.nlargest(5, 'Today PNL (USD)')
    bottom_5 = non_zero_today.nsmallest(5, 'Today PNL (USD)')
    
    # Plot top and bottom performers
    fig = go.Figure()
    
    # Top performers
    fig.add_trace(go.Bar(
        x=top_5['Trading Pair'],
        y=top_5['Today PNL (USD)'],
        name='Top Performers',
        marker_color='green'
    ))
    
    # Bottom performers
    fig.add_trace(go.Bar(
        x=bottom_5['Trading Pair'],
        y=bottom_5['Today PNL (USD)'],
        name='Bottom Performers',
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Top and Bottom Performers Today",
        xaxis_title="Trading Pair",
        yaxis_title="PNL (USD)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Detailed View
    st.subheader("Detailed PNL and Trade Data")
    
    # Create a detailed display DataFrame
    detailed_df = display_df[['Trading Pair', 
                              'Today PNL (USD)', 'Today Trades', 'Today PNL/Trade',
                              'Yesterday PNL (USD)', 'Yesterday Trades', 'Yesterday PNL/Trade',
                              'Week PNL (USD)', 'Week Trades', 'Week PNL/Trade',
                              'All Time PNL (USD)', 'All Time Trades', 'All Time PNL/Trade']]
    
    # Apply styling
    styled_detailed_df = detailed_df.style.applymap(
        color_pnl_cells, 
        subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)',
                'Today PNL/Trade', 'Yesterday PNL/Trade', 'Week PNL/Trade', 'All Time PNL/Trade']
    ).format({
        'Today PNL (USD)': '${:,.2f}',
        'Yesterday PNL (USD)': '${:,.2f}',
        'Week PNL (USD)': '${:,.2f}',
        'All Time PNL (USD)': '${:,.2f}',
        'Today PNL/Trade': '${:,.2f}',
        'Yesterday PNL/Trade': '${:,.2f}',
        'Week PNL/Trade': '${:,.2f}',
        'All Time PNL/Trade': '${:,.2f}'
    })
    
    # Display the styled DataFrame
    st.dataframe(styled_detailed_df, height=600, use_container_width=True)
    
    # Show day-to-day comparison
    st.subheader("Day-to-Day PNL Comparison")
    
    day_comparison_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)']].copy()
    day_comparison_df['Day-to-Day Change'] = day_comparison_df['Today PNL (USD)'] - day_comparison_df['Yesterday PNL (USD)']
    day_comparison_df['Yesterday Change'] = day_comparison_df['Yesterday PNL (USD)'] - day_comparison_df['Day Before PNL (USD)']
    
    # Sort by day-to-day change
    day_comparison_df = day_comparison_df.sort_values(by='Day-to-Day Change', ascending=False)
    
    # Apply styling
    styled_day_comparison = day_comparison_df.style.applymap(
        color_pnl_cells, 
        subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Day-to-Day Change', 'Yesterday Change']
    ).format({
        'Today PNL (USD)': '${:,.2f}',
        'Yesterday PNL (USD)': '${:,.2f}',
        'Day Before PNL (USD)': '${:,.2f}',
        'Day-to-Day Change': '${:,.2f}',
        'Yesterday Change': '${:,.2f}'
    })
    
    # Display the styled DataFrame
    st.dataframe(styled_day_comparison, height=400, use_container_width=True)
    
    # Create a visualization for day-to-day comparison
    # Get top 10 pairs by absolute day-to-day change
    top_change = day_comparison_df.reindex(day_comparison_df['Day-to-Day Change'].abs().sort_values(ascending=False).index).head(10)
    
    fig = go.Figure()
    
    # Add bars for each day
    fig.add_trace(go.Bar(
        x=top_change['Trading Pair'],
        y=top_change['Day Before PNL (USD)'],
        name='Day Before Yesterday',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=top_change['Trading Pair'],
        y=top_change['Yesterday PNL (USD)'],
        name='Yesterday',
        marker_color='royalblue'
    ))
    
    fig.add_trace(go.Bar(
        x=top_change['Trading Pair'],
        y=top_change['Today PNL (USD)'],
        name='Today',
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Top 10 Pairs by Change - 3-Day PNL Comparison",
        xaxis_title="Trading Pair",
        yaxis_title="PNL (USD)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Statistics & Insights
    st.subheader("PNL Statistics and Insights")
    
    # Create a statistics DataFrame
    stats_df = pd.DataFrame({
        'Metric': [
            'Total Trading Pairs',
            'Profitable Pairs Today',
            'Unprofitable Pairs Today',
            'Most Profitable Pair Today',
            'Least Profitable Pair Today',
            'Highest PNL/Trade Today',
            'Average PNL/Trade (All Pairs)',
            'Total Platform PNL Today',
            'Total Platform PNL Yesterday',
            'Week-to-Date PNL',
            'Estimated Monthly PNL (based on week)',
            'Total Trades Today',
            'Total Trades Yesterday',
            'Week-to-Date Trades'
        ],
        'Value': [
            len(display_df),
            len(display_df[display_df['Today PNL (USD)'] > 0]),
            len(display_df[display_df['Today PNL (USD)'] < 0]),
            display_df.loc[display_df['Today PNL (USD)'].idxmax()]['Trading Pair'] if not display_df.empty else 'N/A',
            display_df.loc[display_df['Today PNL (USD)'].idxmin()]['Trading Pair'] if not display_df.empty else 'N/A',
            f"${display_df['Today PNL/Trade'].max():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
            f"${display_df['Today PNL/Trade'].mean():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
            f"${display_df['Today PNL (USD)'].sum():.2f}",
            f"${display_df['Yesterday PNL (USD)'].sum():.2f}",
            f"${display_df['Week PNL (USD)'].sum():.2f}",
            f"${(display_df['Week PNL (USD)'].sum() / 7 * 30):.2f}",
            f"{display_df['Today Trades'].sum():,}",
            f"{display_df['Yesterday Trades'].sum():,}",
            f"{display_df['Week Trades'].sum():,}"
        ]
    })
    
    # Display statistics
    st.dataframe(stats_df, hide_index=True, height=400, use_container_width=True)
    
    # Visualize PNL breakdown by time period
    st.subheader("PNL Breakdown by Time Period")
    
    # For Top 10 Pairs
    top_10_pairs = display_df.nlargest(10, 'Week PNL (USD)')['Trading Pair'].tolist()
    top_10_df = display_df[display_df['Trading Pair'].isin(top_10_pairs)].copy()
    
    # Prepare data for stacked bar chart
    chart_data = []
    for pair in top_10_pairs:
        pair_data = top_10_df[top_10_df['Trading Pair'] == pair].iloc[0]
        chart_data.append({
            'Trading Pair': pair,
            'Today': pair_data['Today PNL (USD)'],
            'Yesterday': pair_data['Yesterday PNL (USD)'],
            'Rest of Week': pair_data['Week PNL (USD)'] - pair_data['Today PNL (USD)'] - pair_data['Yesterday PNL (USD)']
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create the stacked bar chart
    fig = px.bar(
        chart_df,
        x='Trading Pair',
        y=['Today', 'Yesterday', 'Rest of Week'],
        title='PNL Breakdown for Top 10 Pairs',
        labels={'value': 'PNL (USD)', 'variable': 'Time Period'},
        barmode='group'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualize cumulative PNL distribution
    st.subheader("Cumulative PNL Distribution")
    
    # Prepare data for cumulative chart
    sorted_df = display_df.sort_values(by='All Time PNL (USD)', ascending=False).copy()
    sorted_df['Cumulative PNL'] = sorted_df['All Time PNL (USD)'].cumsum()
    sorted_df['Contribution %'] = (sorted_df['All Time PNL (USD)'] / sorted_df['All Time PNL (USD)'].sum() * 100)
    sorted_df['Cumulative Contribution %'] = sorted_df['Contribution %'].cumsum()
    
    # Create the cumulative chart
    fig = go.Figure()
    
    # Add the bar chart for individual contribution
    fig.add_trace(go.Bar(
        x=sorted_df['Trading Pair'].head(20),
        y=sorted_df['All Time PNL (USD)'].head(20),
        name='Individual PNL',
        marker_color='lightblue'
    ))
    
    # Add the line chart for cumulative contribution
    fig.add_trace(go.Scatter(
        x=sorted_df['Trading Pair'].head(20),
        y=sorted_df['Cumulative Contribution %'].head(20),
        name='Cumulative Contribution %',
        yaxis='y2',
        marker_color='darkblue',
        line=dict(width=3)
    ))
    
    # Update layout with two y-axes
    fig.update_layout(
        title="Top 20 Pairs - Individual and Cumulative PNL Contribution",
        xaxis=dict(title="Trading Pair"),
        yaxis=dict(
            title="Individual PNL (USD)",
            titlefont=dict(color="lightblue"),
            tickfont=dict(color="lightblue")
        ),
        yaxis2=dict(
            title="Cumulative Contribution %",
            titlefont=dict(color="darkblue"),
            tickfont=dict(color="darkblue"),
            overlaying="y",
            side="right",
            range=[0, 100]
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
    
    # Show performance trends
    st.subheader("PNL Trend Analysis")
    
    # Calculate week-over-week change
    trend_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)']].copy()
    trend_df['Daily Avg This Week'] = trend_df['Week PNL (USD)'] / 7
    trend_df['3-Day Total'] = trend_df['Today PNL (USD)'] + trend_df['Yesterday PNL (USD)'] + trend_df['Day Before PNL (USD)']
    trend_df['3-Day Daily Avg'] = trend_df['3-Day Total'] / 3
    trend_df['Performance Trend'] = (trend_df['3-Day Daily Avg'] / trend_df['Daily Avg This Week']).round(2)
    
    # Remove pairs with no activity
    trend_df = trend_df[trend_df['Week PNL (USD)'] != 0].copy()
    
    # Sort by performance trend (descending)
    trend_df = trend_df.sort_values(by='Performance Trend', ascending=False)
    
    # Apply styling
    styled_trend_df = trend_df.style.applymap(
        color_pnl_cells, 
        subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)', '3-Day Total', '3-Day Daily Avg', 'Daily Avg This Week']
    ).format({
        'Today PNL (USD)': '${:,.2f}',
        'Yesterday PNL (USD)': '${:,.2f}',
        'Day Before PNL (USD)': '${:,.2f}',
        'Week PNL (USD)': '${:,.2f}',
        '3-Day Total': '${:,.2f}',
        '3-Day Daily Avg': '${:,.2f}',
        'Daily Avg This Week': '${:,.2f}',
        'Performance Trend': '{:.2f}x'
    })
    
    # Add conditional formatting for Performance Trend
    def color_trend(val):
        if val > 1.5:
            return 'background-color: rgba(0, 200, 0, 0.8); color: black; font-weight: bold'
        elif val > 1.1:
            return 'background-color: rgba(150, 255, 150, 0.7); color: black'
        elif val < 0.5:
            return 'background-color: rgba(255, 0, 0, 0.9); color: white; font-weight: bold'
        elif val < 0.9:
            return 'background-color: rgba(255, 150, 150, 0.7); color: black'
        else:
            return 'background-color: rgba(255, 255, 200, 0.7); color: black'  # Neutral/stable
    
    styled_trend_df = styled_trend_df.applymap(color_trend, subset=['Performance Trend'])
    
    # Display the trend analysis
    st.markdown("### Recent Performance Trends (3-Day vs Weekly Average)")
    st.markdown("Performance Trend > 1: Recent performance better than weekly average")
    st.markdown("Performance Trend < 1: Recent performance worse than weekly average")
    
    st.dataframe(styled_trend_df, height=500, use_container_width=True)
    
    # Add explanatory text
    with st.expander("Understanding the PNL Dashboard"):
        st.markdown("""
        ## 📊 How to Use This PNL Dashboard
        
        This dashboard shows platform profit and loss (PNL) across all selected trading pairs over different time periods.
        
        ### Time Periods
        - **Today**: From midnight Singapore time (SGT) until now
        - **Yesterday**: Full 24 hours from midnight to midnight SGT
        - **This Week**: Last 7 days including today
        - **All Time**: Cumulative PNL since records began
        
        ### Color Coding
        - 🟩 **Green**: Profit (darker green for higher profits)
        - 🟥 **Red**: Loss (darker red for higher losses)
        - ⬜ **Grey**: No activity/zero PNL
        
        ### Key Metrics
        - **PNL (USD)**: Platform's profit/loss in USD for each time period
        - **Trades**: Number of trades executed in each time period
        - **PNL/Trade**: Average profit per trade
        
        ### Dashboard Tabs
        - **Main Dashboard**: Quick overview of PNL by time period
        - **Detailed View**: Complete breakdown including trade counts and per-trade metrics
        - **Statistics & Insights**: Trends, correlations, and deeper analysis
        
        ### Performance Trends
        - **Performance Trend > 1**: Recent performance (3-day avg) is better than the weekly average
        - **Performance Trend < 1**: Recent performance is worse than the weekly average
        
        ### Technical Details
        - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
        - All values are shown in USD
        - The dashboard refreshes when you click the "Refresh Data" button
        - Singapore timezone (UTC+8) is used throughout
        """
        )

# Add footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")