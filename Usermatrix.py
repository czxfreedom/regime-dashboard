import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="User PNL Matrix Dashboard",
    page_icon="👤",
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
st.title("User PNL Matrix Dashboard")
st.subheader("Performance Analysis by User (Singapore Time)")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to get Singapore time boundaries
def get_time_boundaries():
    # Current time in Singapore
    now_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)
    
    # Today's midnight in Singapore
    today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Yesterday's midnight in Singapore
    yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)
    
    # 7 days ago midnight in Singapore
    week_ago_midnight_sg = today_midnight_sg - timedelta(days=7)
    
    # 30 days ago midnight in Singapore
    month_ago_midnight_sg = today_midnight_sg - timedelta(days=30)
    
    # All time (use a far past date, e.g., 5 years ago)
    all_time_start_sg = today_midnight_sg.replace(year=today_midnight_sg.year-5)
    
    # Convert all times back to UTC for database queries
    today_midnight_utc = today_midnight_sg.astimezone(pytz.utc)
    yesterday_midnight_utc = yesterday_midnight_sg.astimezone(pytz.utc)
    week_ago_midnight_utc = week_ago_midnight_sg.astimezone(pytz.utc)
    month_ago_midnight_utc = month_ago_midnight_sg.astimezone(pytz.utc)
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
        "this_week": {
            "start": week_ago_midnight_utc,
            "end": now_utc,
            "label": f"This Week (Last 7 Days)"
        },
        "this_month": {
            "start": month_ago_midnight_utc,
            "end": now_utc,
            "label": f"This Month (Last 30 Days)"
        },
        "all_time": {
            "start": all_time_start_utc,
            "end": now_utc,
            "label": "All Time"
        }
    }

# Calculate time boundaries
time_boundaries = get_time_boundaries()

# Fetch all pairs from DB
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
    
@st.cache_data(ttl=600, show_spinner="Fetching users...")
def fetch_top_users(limit=100):
    """Fetch top users by trading volume."""
    # Calculate the date range internally
    now_utc = datetime.now(pytz.utc)
    month_ago_utc = now_utc - timedelta(days=30)
    
    query = f"""
    SELECT 
        "taker_account_id" as user_identifier,
        COUNT(*) as trade_count,
        SUM(ABS("deal_size" * "deal_price")) as total_volume
    FROM 
        "public"."trade_fill_fresh"
    WHERE 
        "created_at" >= '{month_ago_utc.strftime("%Y-%m-%d %H:%M:%S")}'
        AND "taker_way" IN (1, 2, 3, 4)
    GROUP BY 
        "taker_account_id"
    ORDER BY 
        total_volume DESC
    LIMIT {limit}
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No active users found in the database.")
            return []
        return df['user_identifier'].tolist()
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        # For debugging, print the full error with traceback
        import traceback
        st.error(traceback.format_exc())
        # Return some mock user IDs for testing
        return [f"user_{i}" for i in range(1, 11)]    

# Fetch top users from DB

# UI Controls
all_pairs = fetch_all_pairs()
top_users = fetch_top_users(limit=100)  # Get top 100 users by volume

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all_pairs = st.checkbox("Select All Pairs", value=False)
    
    if select_all_pairs:
        selected_pairs = all_pairs
    else:
        selected_pairs = st.multiselect(
            "Select Trading Pairs", 
            all_pairs,
            default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
        )

with col2:
    # Let user select how many users to include
    user_limit = st.slider(
        "Number of Top Users to Show", 
        min_value=5, 
        max_value=min(100, len(top_users)), 
        value=25,
        step=5
    )
    
    top_selected_users = top_users[:user_limit]

with col3:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one trading pair")
    st.stop()

if not top_selected_users:
    st.warning("No active users found for the selected period")
    st.stop()

# Function to fetch user PNL data
@st.cache_data(ttl=600)
def fetch_user_pnl_data(taker_account_id, pair_name, start_time, end_time):
    """Fetch user PNL data for a specific pair and time period."""
    
    query = f"""
    WITH 
    user_order_pnl AS (
      -- Calculate user order PNL
      SELECT
        COALESCE(SUM("taker_pnl" * "collateral_price"), 0) AS "user_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    ),
    
    user_fee_payments AS (
      -- Calculate user fee payments
      SELECT
        COALESCE(SUM(-1 * "taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
    ),
    
    user_funding_payments AS (
      -- Calculate user funding fee payments
      SELECT
        COALESCE(SUM("funding_fee" * "collateral_price"), 0) AS "user_funding_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" = 0
    ),
    
    user_trade_count AS (
      -- Calculate total number of trades
      SELECT
        COUNT(*) AS "trade_count"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    )
    
    -- Final query: combine all data sources
    SELECT
      (SELECT "user_order_pnl" FROM user_order_pnl) +
      (SELECT "user_fee_payments" FROM user_fee_payments) +
      (SELECT "user_funding_payments" FROM user_funding_payments) AS "user_total_pnl",
      (SELECT "trade_count" FROM user_trade_count) AS "trade_count"
    """
    
    # Rest of the function remains the same
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return {"pnl": 0, "trades": 0}
        
        return {
            "pnl": float(df.iloc[0]['user_total_pnl']),
            "trades": int(df.iloc[0]['trade_count'])
        }
    except Exception as e:
        st.error(f"Error processing PNL for user {taker_account_id} on {pair_name}: {e}")
        return {"pnl": 0, "trades": 0}

# Calculate user metadata
@st.cache_data(ttl=600)
@st.cache_data(ttl=600)
def fetch_user_metadata(taker_account_id):
    """Fetch additional metadata about a user."""
    
    query = f"""
    SELECT 
        MIN(created_at) as first_trade_date,
        TO_CHAR(MIN(created_at), 'YYYY-MM-DD') as first_trade_date_str,
        COUNT(*) as all_time_trades,
        SUM(ABS("deal_size" * "deal_price")) as all_time_volume
    FROM 
        "public"."trade_fill_fresh"
    WHERE 
        "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    """
    
    # Rest of the function remains the same
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return {
                "first_trade_date": "Unknown",
                "all_time_trades": 0,
                "all_time_volume": 0,
                "account_age_days": 0
            }
        
        first_trade = df.iloc[0]['first_trade_date']
        if first_trade:
            account_age = (now_utc - first_trade).days
        else:
            account_age = 0
            
        return {
            "first_trade_date": df.iloc[0]['first_trade_date_str'],
            "all_time_trades": int(df.iloc[0]['all_time_trades']),
            "all_time_volume": float(df.iloc[0]['all_time_volume']),
            "account_age_days": account_age
        }
    except Exception as e:
        st.error(f"Error fetching metadata for user {user_id}: {e}")
        return {
            "first_trade_date": "Unknown",
            "all_time_trades": 0,
            "all_time_volume": 0,
            "account_age_days": 0
        }

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Initialize data structure
results = {}
periods = ["today", "yesterday", "this_week", "this_month", "all_time"]

# Process data for all selected users and pairs
total_combinations = len(top_selected_users) * len(selected_pairs)
progress_counter = 0

for user_id in top_selected_users:
    if user_id not in results:
        results[user_id] = {
            "user_id": user_id,
            "metadata": fetch_user_metadata(user_id),
            "pairs": {}
        }
    
    for pair_name in selected_pairs:
        progress_counter += 1
        progress_percentage = progress_counter / total_combinations
        progress_bar.progress(progress_percentage)
        status_text.text(f"Processing User {user_id} - Pair {pair_name} ({progress_counter}/{total_combinations})")
        
        if pair_name not in results[user_id]["pairs"]:
            results[user_id]["pairs"][pair_name] = {}
        
        for period in periods:
            start_time = time_boundaries[period]["start"]
            end_time = time_boundaries[period]["end"]
            
            # Fetch PNL data for this user, pair, and time period
            period_data = fetch_user_pnl_data(user_id, pair_name, start_time, end_time)
            
            # Store the results
            results[user_id]["pairs"][pair_name][period] = period_data

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed data for {len(results)} users across {len(selected_pairs)} trading pairs")

# Calculate totals for each user and period
for user_id in results:
    for period in periods:
        results[user_id][f"total_{period}_pnl"] = sum(
            results[user_id]["pairs"][pair][period]["pnl"] 
            for pair in results[user_id]["pairs"]
        )
        results[user_id][f"total_{period}_trades"] = sum(
            results[user_id]["pairs"][pair][period]["trades"] 
            for pair in results[user_id]["pairs"]
        )

# Create a DataFrame for the matrix view
matrix_rows = []
for user_id, user_data in results.items():
    row = {
        'User ID': user_id,
        'Account Age (days)': user_data["metadata"]["account_age_days"],
        'First Trade': user_data["metadata"]["first_trade_date"],
        'All Time Volume': user_data["metadata"]["all_time_volume"],
        'Today PNL': user_data["total_today_pnl"],
        'Today Trades': user_data["total_today_trades"],
        'Yesterday PNL': user_data["total_yesterday_pnl"],
        'Yesterday Trades': user_data["total_yesterday_trades"],
        'Week PNL': user_data["total_this_week_pnl"],
        'Week Trades': user_data["total_this_week_trades"],
        'Month PNL': user_data["total_this_month_pnl"],
        'Month Trades': user_data["total_this_month_trades"],
        'All Time PNL': user_data["total_all_time_pnl"],
        'All Time Trades': user_data["total_all_time_trades"]
    }
    matrix_rows.append(row)

user_matrix_df = pd.DataFrame(matrix_rows)

# Create a DataFrame for per-pair analysis
pair_rows = []
for user_id, user_data in results.items():
    for pair_name, pair_data in user_data["pairs"].items():
        row = {
            'User ID': user_id,
            'Trading Pair': pair_name,
            'Today PNL': pair_data["today"]["pnl"],
            'Today Trades': pair_data["today"]["trades"],
            'Yesterday PNL': pair_data["yesterday"]["pnl"],
            'Yesterday Trades': pair_data["yesterday"]["trades"],
            'Week PNL': pair_data["this_week"]["pnl"],
            'Week Trades': pair_data["this_week"]["trades"],
            'Month PNL': pair_data["this_month"]["pnl"],
            'Month Trades': pair_data["this_month"]["trades"],
            'All Time PNL': pair_data["all_time"]["pnl"],
            'All Time Trades': pair_data["all_time"]["trades"]
        }
        pair_rows.append(row)

user_pair_df = pd.DataFrame(pair_rows)

# Calculate additional metrics
if not user_matrix_df.empty:
    user_matrix_df['Avg PNL/Trade (All Time)'] = (
        user_matrix_df['All Time PNL'] / user_matrix_df['All Time Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    user_matrix_df['Week PNL/Trade'] = (
        user_matrix_df['Week PNL'] / user_matrix_df['Week Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    user_matrix_df['Daily Avg PNL (Week)'] = user_matrix_df['Week PNL'] / 7
    user_matrix_df['Avg Daily Trades (Week)'] = user_matrix_df['Week Trades'] / 7
    
    # Sort by Today's PNL (descending)
    user_matrix_df = user_matrix_df.sort_values(by='Today PNL', ascending=False)

# Function to color cells based on PNL value
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
tab1, tab2, tab3, tab4 = st.tabs([
    "User PNL Matrix", 
    "User-Pair Analysis", 
    "Heat Map", 
    "Insights & Trends"
])

with tab1:
    # User PNL Matrix View
    st.subheader("User PNL Overview Matrix")
    
    if user_matrix_df.empty:
        st.warning("No data available for the selected users and pairs")
    else:
        # Create a simplified display DataFrame
        display_cols = [
            'User ID', 'Today PNL', 'Yesterday PNL', 'Week PNL', 
            'Month PNL', 'All Time PNL', 'Week PNL/Trade', 'All Time Trades'
        ]
        
        display_df = user_matrix_df[display_cols].copy()
        
        # Apply styling
        styled_df = display_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL', 'Yesterday PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'Week PNL/Trade']
        ).format({
            'Today PNL': '${:,.2f}',
            'Yesterday PNL': '${:,.2f}',
            'Week PNL': '${:,.2f}',
            'Month PNL': '${:,.2f}',
            'All Time PNL': '${:,.2f}',
            'Week PNL/Trade': '${:,.2f}',
            'All Time Trades': '{:,}'
        })
        
        # Display the styled DataFrame
        st.dataframe(styled_df, height=600, use_container_width=True)
        
        # Create summary cards
        st.subheader("Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_today_pnl = user_matrix_df['Today PNL'].sum()
            profitable_users_today = len(user_matrix_df[user_matrix_df['Today PNL'] > 0])
            st.metric(
                "Total Users PNL Today", 
                f"${total_today_pnl:,.2f}",
                f"{profitable_users_today}/{len(user_matrix_df)} profitable"
            )
        
        with col2:
            total_yesterday_pnl = user_matrix_df['Yesterday PNL'].sum()
            profitable_users_yesterday = len(user_matrix_df[user_matrix_df['Yesterday PNL'] > 0])
            st.metric(
                "Total Users PNL Yesterday", 
                f"${total_yesterday_pnl:,.2f}",
                f"{profitable_users_yesterday}/{len(user_matrix_df)} profitable"
            )
        
        with col3:
            total_week_pnl = user_matrix_df['Week PNL'].sum()
            daily_avg = total_week_pnl / 7
            st.metric(
                "Total Week PNL (7 days)", 
                f"${total_week_pnl:,.2f}",
                f"${daily_avg:,.2f}/day avg"
            )
        
        with col4:
            total_month_pnl = user_matrix_df['Month PNL'].sum()
            st.metric(
                "Total Month PNL (30 days)", 
                f"${total_month_pnl:,.2f}"
            )
        
        with col5:
            total_all_time_pnl = user_matrix_df['All Time PNL'].sum()
            st.metric(
                "All Time Total PNL", 
                f"${total_all_time_pnl:,.2f}"
            )
        
        # Create a visualization of top and bottom performers today
        st.subheader("Today's Top and Bottom Users by PNL")
        
        # Filter out zero PNL users
        non_zero_today = user_matrix_df[user_matrix_df['Today PNL'] != 0].copy()
        
        if not non_zero_today.empty:
            # Get top 5 and bottom 5 performers
            top_5 = non_zero_today.nlargest(5, 'Today PNL')
            bottom_5 = non_zero_today.nsmallest(5, 'Today PNL')
            
            # Plot top and bottom performers
            fig = go.Figure()
            
            # Top performers
            fig.add_trace(go.Bar(
                x=top_5['User ID'],
                y=top_5['Today PNL'],
                name='Top Performers',
                marker_color='green'
            ))
            
            # Bottom performers
            fig.add_trace(go.Bar(
                x=bottom_5['User ID'],
                y=bottom_5['Today PNL'],
                name='Bottom Performers',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Top and Bottom Users by PNL Today",
                xaxis_title="User ID",
                yaxis_title="PNL (USD)",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-zero PNL data available for today.")

with tab2:
    # User-Pair Analysis
    st.subheader("User Performance by Trading Pair")
    
    # User selector
    selected_user_id = st.selectbox(
        "Select User to Analyze", 
        user_matrix_df['User ID'].tolist()
    )
    
    # Filter for selected user
    user_pairs_df = user_pair_df[user_pair_df['User ID'] == selected_user_id].copy()
    
    if user_pairs_df.empty:
        st.warning(f"No data available for user {selected_user_id}")
    else:
        # User metadata
        user_metadata = results[selected_user_id]["metadata"]
        
        # Display user info
        user_info_cols = st.columns(4)
        with user_info_cols[0]:
            st.metric("Account Age", f"{user_metadata['account_age_days']} days")
        with user_info_cols[1]:
            st.metric("First Trade Date", user_metadata['first_trade_date'])
        with user_info_cols[2]:
            st.metric("Total Trades", f"{user_metadata['all_time_trades']:,}")
        with user_info_cols[3]:
            st.metric("Total Trading Volume", f"${user_metadata['all_time_volume']:,.2f}")
        
        # Display pair performance
        st.subheader(f"Trading Pairs Performance for User {selected_user_id}")
        
        # Sort by All Time PNL
        user_pairs_df = user_pairs_df.sort_values(by='All Time PNL', ascending=False)
        
        # Calculate additional metrics
        user_pairs_df['All Time PNL/Trade'] = (
            user_pairs_df['All Time PNL'] / user_pairs_df['All Time Trades']
        ).replace([np.inf, -np.inf, np.nan], 0)
        
        # Display columns
        display_cols = [
            'Trading Pair', 'Today PNL', 'Yesterday PNL', 'Week PNL', 
            'Month PNL', 'All Time PNL', 'All Time Trades', 'All Time PNL/Trade'
        ]
        
        # Apply styling
        styled_pairs_df = user_pairs_df[display_cols].style.applymap(
            color_pnl_cells, 
            subset=['Today PNL', 'Yesterday PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'All Time PNL/Trade']
        ).format({
            'Today PNL': '${:,.2f}',
            'Yesterday PNL': '${:,.2f}',
            'Week PNL': '${:,.2f}',
            'Month PNL': '${:,.2f}',
            'All Time PNL': '${:,.2f}',
            'All Time PNL/Trade': '${:,.2f}',
            'All Time Trades': '{:,}'
        })
        
        # Display the styled DataFrame
        st.dataframe(styled_pairs_df, height=400, use_container_width=True)
        
        # Create a pie chart of All Time PNL by pair
        st.subheader("PNL Distribution by Trading Pair")
        
        # Filter out pairs with zero PNL
        non_zero_pairs = user_pairs_df[user_pairs_df['All Time PNL'] != 0].copy()
        
        if not non_zero_pairs.empty:
            # Create a pie chart for all time PNL distribution
            fig = px.pie(
                non_zero_pairs,
                values='All Time PNL',
                names='Trading Pair',
                title=f"All Time PNL Distribution for User {selected_user_id}",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a time comparison bar chart
            st.subheader("Time Period Comparison by Trading Pair")
            
            # Select top pairs by absolute PNL
            top_pairs = non_zero_pairs.reindex(non_zero_pairs['All Time PNL'].abs().sort_values(ascending=False).index).head(10)
            
            # Reshape data for grouped bar chart
            fig = go.Figure()
            
            # Add bars for each time period
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Today PNL'],
                name='Today',
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Yesterday PNL'],
                name='Yesterday',
                marker_color='#ff7f0e'
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Week PNL'],
                name='Week',
                marker_color='#2ca02c'
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Month PNL'],
                name='Month',
                marker_color='#d62728'
            ))
            
            fig.update_layout(
                title=f"PNL Comparison Across Time Periods for User {selected_user_id}",
                xaxis_title="Trading Pair",
                yaxis_title="PNL (USD)",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-zero PNL data available for this user.")

with tab3:
    # Heat Map View
    st.subheader("User-Pair PNL Heat Map")
    
    # Time period selector
    time_period = st.selectbox(
        "Select Time Period for Heat Map",
        ["Today", "Yesterday", "Week", "Month", "All Time"],
        index=0
    )
    
    # Map selection to dataframe column
    period_map = {
        "Today": "Today PNL",
        "Yesterday": "Yesterday PNL",
        "Week": "Week PNL",
        "Month": "Month PNL",
        "All Time": "All Time PNL"
    }
    
    selected_period = period_map[time_period]
    
    # Create pivot table for heatmap
    if not user_pair_df.empty:
        pivot_df = user_pair_df.pivot_table(
            values=selected_period,
            index='User ID',
            columns='Trading Pair',
            aggfunc='sum'
        ).fillna(0)
        
        # Generate the heatmap
        if not pivot_df.empty:
            # Select top users by absolute PNL
            top_user_pnls = user_matrix_df[selected_period].abs().sort_values(ascending=False)
            top_users_for_heatmap = top_user_pnls.head(min(20, len(top_user_pnls))).index
            
            # Filter pivot table for top users
            filtered_pivot = pivot_df.loc[pivot_df.index.isin(top_users_for_heatmap)]
            
            # Create heatmap
            fig = px.imshow(
                filtered_pivot,
                labels=dict(x="Trading Pair", y="User ID", color=f"{time_period} PNL (USD)"),
                x=filtered_pivot.columns,
                y=filtered_pivot.index,
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title=f"User-Pair PNL Heat Map ({time_period})",
                height=800
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            # Add annotations (PNL values)
            for i, user_id in enumerate(filtered_pivot.index):
                for j, pair in enumerate(filtered_pivot.columns):
                    value = filtered_pivot.iloc[i, j]
                    text_color = "black" if abs(value) < 1000 else "white"
                    
                    fig.add_annotation(
                        x=pair,
                        y=user_id,
                        text=f"${value:.0f}",
                        showarrow=False,
                        font=dict(color=text_color, size=10)
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            ### Understanding the Heat Map
            
            - **Green cells**: Positive PNL (user profit)
            - **Red cells**: Negative PNL (user loss)
            - **Intensity**: Darker colors indicate larger PNL values
            - **White/Light cells**: Near-zero PNL
            
            The heat map shows the relationship between users and trading pairs, allowing you to identify:
            - Which users are most profitable on which pairs
            - Patterns of success or failure across different user segments
            - Opportunities for targeted user engagement
            """)
        else:
            st.info(f"No PNL data available for {time_period}")
    else:
        st.warning("Insufficient data to create heat map")
    
    # Create correlation heatmap
    st.subheader("Pair Correlation Analysis")
    st.markdown("This shows how trading pairs are correlated in terms of user PNL performance")
    
    if not user_pair_df.empty:
        # Create a wider pivot for correlation analysis
        corr_pivot = user_pair_df.pivot_table(
            values='All Time PNL',
            index='User ID',
            columns='Trading Pair',
            aggfunc='sum'
        ).fillna(0)
        
        if not corr_pivot.empty and corr_pivot.shape[1] > 1:
            # Calculate correlation between pairs
            correlation_matrix = corr_pivot.corr()
            
            # Create heatmap
            fig = px.imshow(
                correlation_matrix,
                labels=dict(x="Trading Pair", y="Trading Pair", color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                color_continuous_scale='RdBu_r',  # Red for negative, Blue for positive
                aspect="auto",
                title="Trading Pair PNL Correlation Matrix",
                height=800,
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            # Add annotations (correlation values)
            for i, row_pair in enumerate(correlation_matrix.index):
                for j, col_pair in enumerate(correlation_matrix.columns):
                    value = correlation_matrix.iloc[i, j]
                    text_color = "black" if abs(value) < 0.7 else "white"
                    
                    fig.add_annotation(
                        x=col_pair,
                        y=row_pair,
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(color=text_color, size=10)
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            ### Understanding the Correlation Matrix
            
            - **Blue cells (positive correlation)**: Pairs where users tend to perform similarly (both profit or both lose)
            - **Red cells (negative correlation)**: Pairs where users tend to perform inversely (profit on one, lose on the other)
            - **White/Light cells**: No strong correlation
            
            Strong positive correlations may indicate similar market dynamics or trading strategies.
            Strong negative correlations may suggest hedging opportunities or diverse market behaviors.
            """)
        else:
            st.info("Insufficient data to create correlation matrix")
    else:
        st.warning("No data available for correlation analysis")

with tab4:
    # Insights & Trends
    st.subheader("User Performance Insights")
    
    # User categorization
    if not user_matrix_df.empty:
        # Categorize users
        user_categories = {
            "consistently_profitable": [],
            "high_volume": [],
            "high_volatility": [],
            "consistently_unprofitable": [],
            "improving": [],
            "declining": []
        }
        
        for user_id, user_data in results.items():
            # High volume users (top 20% by all-time trades)
            if user_data["total_all_time_trades"] > user_matrix_df["All Time Trades"].quantile(0.8):
                user_categories["high_volume"].append(user_id)
            
            # Consistently profitable (positive PNL in today, yesterday, week)
            if (user_data["total_today_pnl"] > 0 and 
                user_data["total_yesterday_pnl"] > 0 and 
                user_data["total_this_week_pnl"] > 0):
                user_categories["consistently_profitable"].append(user_id)
            
            # Consistently unprofitable (negative PNL in today, yesterday, week)
            if (user_data["total_today_pnl"] < 0 and 
                user_data["total_yesterday_pnl"] < 0 and 
                user_data["total_this_week_pnl"] < 0):
                user_categories["consistently_unprofitable"].append(user_id)
            
            # Improving trend (today > yesterday > day before)
            if (user_data["total_today_pnl"] > user_data["total_yesterday_pnl"]):
                user_categories["improving"].append(user_id)
            
            # Declining trend (today < yesterday)
            if (user_data["total_today_pnl"] < user_data["total_yesterday_pnl"]):
                user_categories["declining"].append(user_id)
            
            # High volatility (large swings between today and yesterday)
            daily_change = abs(user_data["total_today_pnl"] - user_data["total_yesterday_pnl"])
            if daily_change > user_matrix_df["Today PNL"].std() * 2:  # More than 2 standard deviations
                user_categories["high_volatility"].append(user_id)
        
        # Display user segments
        st.subheader("User Segments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Positive Segments")
            st.markdown(f"**Consistently Profitable Users:** {len(user_categories['consistently_profitable'])}")
            st.markdown(f"**Improving Trend Users:** {len(user_categories['improving'])}")
            st.markdown(f"**High-Volume Users:** {len(user_categories['high_volume'])}")
        
        with col2:
            st.markdown("### Watch Segments")
            st.markdown(f"**Consistently Unprofitable Users:** {len(user_categories['consistently_unprofitable'])}")
            st.markdown(f"**Declining Trend Users:** {len(user_categories['declining'])}")
            st.markdown(f"**High-Volatility Users:** {len(user_categories['high_volatility'])}")
        
        # User PNL distribution
        st.subheader("User PNL Distribution Analysis")
        
        # Create histograms for PNL distribution
        pnl_periods = ["Today PNL", "Week PNL", "All Time PNL"]
        fig = go.Figure()
        
        for period in pnl_periods:
            # Filter out zero values
            non_zero_data = user_matrix_df[user_matrix_df[period] != 0][period]
            
            if not non_zero_data.empty:
                fig.add_trace(go.Histogram(
                    x=non_zero_data,
                    name=period,
                    opacity=0.7,
                    nbinsx=20
                ))
        
        fig.update_layout(
            title="PNL Distribution Comparison",
            xaxis_title="PNL (USD)",
            yaxis_title="Number of Users",
            barmode='overlay',
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
        
        # Time series analysis
        st.subheader("User Performance Over Time")
        
        # Get top 5 users by absolute PNL
        top_users_by_pnl = user_matrix_df.reindex(
            user_matrix_df['All Time PNL'].abs().sort_values(ascending=False).index
        ).head(5)['User ID'].tolist()
        
        # Create a time series chart
        fig = go.Figure()
        
        # Time periods in order
        time_points = ['Day Before Yesterday', 'Yesterday', 'Today']
        
        # For each top user, plot a line
        for user_id in top_users_by_pnl:
            user_data = results[user_id]
            
            # Extract PNL for each time period
            pnl_values = [
                user_data["total_day_before_yesterday_pnl"] if "total_day_before_yesterday_pnl" in user_data else 0,
                user_data["total_yesterday_pnl"],
                user_data["total_today_pnl"]
            ]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=pnl_values,
                mode='lines+markers',
                name=f'User {user_id}'
            ))
        
        fig.update_layout(
            title="PNL Trend for Top Users (3-Day Period)",
            xaxis_title="Time Period",
            yaxis_title="PNL (USD)",
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
        
        # Additional insights
        st.subheader("Key Insights & Recommendations")
        
        # Calculate key statistics
        total_users = len(user_matrix_df)
        profitable_today = len(user_matrix_df[user_matrix_df['Today PNL'] > 0])
        profitable_pct = (profitable_today / total_users * 100) if total_users > 0 else 0
        
        top_user_today = user_matrix_df.loc[user_matrix_df['Today PNL'].idxmax()]['User ID'] if not user_matrix_df.empty else 'N/A'
        top_user_pnl = user_matrix_df['Today PNL'].max() if not user_matrix_df.empty else 0
        
        bottom_user_today = user_matrix_df.loc[user_matrix_df['Today PNL'].idxmin()]['User ID'] if not user_matrix_df.empty else 'N/A'
        bottom_user_pnl = user_matrix_df['Today PNL'].min() if not user_matrix_df.empty else 0
        
        most_popular_pair = user_pair_df['Trading Pair'].value_counts().index[0] if not user_pair_df.empty else 'N/A'
        best_pair = user_pair_df.groupby('Trading Pair')['Today PNL'].sum().idxmax() if not user_pair_df.empty else 'N/A'
        
        # Display insights
        insights_cols = st.columns(2)
        
        with insights_cols[0]:
            st.markdown("### Today's Performance Snapshot")
            st.markdown(f"- **Total Users Analyzed:** {total_users}")
            st.markdown(f"- **Profitable Users:** {profitable_today} ({profitable_pct:.1f}%)")
            st.markdown(f"- **Top Performing User:** User {top_user_today} (${top_user_pnl:,.2f})")
            st.markdown(f"- **Bottom Performing User:** User {bottom_user_today} (${bottom_user_pnl:,.2f})")
            st.markdown(f"- **Most Popular Trading Pair:** {most_popular_pair}")
            st.markdown(f"- **Best Performing Pair:** {best_pair}")
        
        with insights_cols[1]:
            st.markdown("### Key Recommendations")
            st.markdown("- **For Profitable Users:** Consider incentive programs to increase trading volume")
            st.markdown("- **For Unprofitable Users:** Provide educational resources and risk management tools")
            st.markdown("- **For High-Volatility Users:** Offer hedging instruments and stability mechanisms")
            st.markdown("- **For Declining Users:** Targeted campaigns to re-engage and improve experience")
            st.markdown("- **For New Users:** Create onboarding tutorials for best-performing pairs")
            st.markdown("- **For High-Volume Users:** Implement tiered fee structures based on volume")
    else:
        st.warning("Insufficient data to generate insights")

# Add a section for user cohort analysis
st.markdown("---")
st.subheader("User Cohort Analysis")

# Define cohort criteria
cohort_options = [
    "Profitable Users",
    "Unprofitable Users",
    "High Volume Traders",
    "New Users (Last 30 Days)",
    "Experienced Users (90+ Days)",
    "Improving Trend Users",
    "All Users"
]

selected_cohort = st.selectbox("Select User Cohort", cohort_options)

# Apply filters based on selected cohort
if not user_matrix_df.empty:
    cohort_df = user_matrix_df.copy()
    
    if selected_cohort == "Profitable Users":
        cohort_df = cohort_df[cohort_df['Week PNL'] > 0]
    elif selected_cohort == "Unprofitable Users":
        cohort_df = cohort_df[cohort_df['Week PNL'] < 0]
    elif selected_cohort == "High Volume Traders":
        # Top 20% by trade count
        volume_threshold = cohort_df['All Time Trades'].quantile(0.8)
        cohort_df = cohort_df[cohort_df['All Time Trades'] > volume_threshold]
    elif selected_cohort == "New Users (Last 30 Days)":
        cohort_df = cohort_df[cohort_df['Account Age (days)'] <= 30]
    elif selected_cohort == "Experienced Users (90+ Days)":
        cohort_df = cohort_df[cohort_df['Account Age (days)'] > 90]
    elif selected_cohort == "Improving Trend Users":
        cohort_df = cohort_df[cohort_df['Today PNL'] > cohort_df['Yesterday PNL']]
    # "All Users" doesn't need filtering
    
    # Display cohort size
    st.markdown(f"**Cohort Size:** {len(cohort_df)} users")
    
    if not cohort_df.empty:
        # Summary statistics for the cohort
        cohort_cols = st.columns(4)
        
        with cohort_cols[0]:
            avg_today_pnl = cohort_df['Today PNL'].mean()
            st.metric("Average Today PNL", f"${avg_today_pnl:,.2f}")
        
        with cohort_cols[1]:
            avg_week_pnl = cohort_df['Week PNL'].mean()
            st.metric("Average Week PNL", f"${avg_week_pnl:,.2f}")
        
        with cohort_cols[2]:
            avg_pnl_per_trade = cohort_df['Week PNL/Trade'].mean()
            st.metric("Average PNL/Trade", f"${avg_pnl_per_trade:,.2f}")
        
        with cohort_cols[3]:
            avg_trades = cohort_df['Week Trades'].mean()
            st.metric("Average Weekly Trades", f"{avg_trades:.1f}")
        
        # Create a cohort performance visualization
        st.subheader(f"Top Trading Pairs for {selected_cohort}")
        
        # Filter relevant user pairs
        cohort_users = cohort_df['User ID'].tolist()
        cohort_pairs_df = user_pair_df[user_pair_df['User ID'].isin(cohort_users)].copy()
        
        if not cohort_pairs_df.empty:
            # Aggregate by trading pair
            pair_performance = cohort_pairs_df.groupby('Trading Pair').agg({
                'Today PNL': 'sum',
                'Week PNL': 'sum',
                'Month PNL': 'sum',
                'All Time PNL': 'sum',
                'Week Trades': 'sum'
            }).reset_index()
            
            # Sort by Week PNL
            pair_performance = pair_performance.sort_values(by='Week PNL', ascending=False)
            
            # Calculate efficiency
            pair_performance['PNL/Trade'] = (
                pair_performance['Week PNL'] / pair_performance['Week Trades']
            ).replace([np.inf, -np.inf, np.nan], 0)
            
            # Display top pairs table
            st.dataframe(
                pair_performance.head(10).style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'PNL/Trade']
                ).format({
                    'Today PNL': '${:,.2f}',
                    'Week PNL': '${:,.2f}',
                    'Month PNL': '${:,.2f}',
                    'All Time PNL': '${:,.2f}',
                    'PNL/Trade': '${:,.2f}',
                    'Week Trades': '{:,}'
                }),
                height=400,
                use_container_width=True
            )
            
            # Create bar chart of top pairs
            top_pairs = pair_performance.head(10)
            
            fig = px.bar(
                top_pairs,
                x='Trading Pair',
                y=['Today PNL', 'Week PNL', 'Month PNL'],
                title=f"PNL Performance for Top Trading Pairs ({selected_cohort})",
                barmode='group'
            )
            
            fig.update_layout(
                xaxis_title="Trading Pair",
                yaxis_title="PNL (USD)",
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
        else:
            st.info(f"No trading pair data available for this cohort")
    else:
        st.info(f"No users match the {selected_cohort} criteria")
else:
    st.warning("No user data available")

# Add footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")
