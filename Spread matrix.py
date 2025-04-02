import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(
    page_title="Exchange Spread Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS styling
st.markdown("""
<style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
    }
    .header-style {
        font-size:28px !important;
        font-weight: bold;
        color: #1E88E5;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .subheader-style {
        font-size:22px !important;
        font-weight: bold;
        color: #1976D2;
        padding: 5px 0;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .highlight-text {
        color: #1E88E5;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Configuration ---
try:
    # Try to get database config from secrets
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
    st.sidebar.success("Connected to database successfully")
except Exception as e:
    # If secrets not available, allow manual entry
    st.sidebar.error(f"Error connecting to the database: {e}")
    
    # Provide manual connection option
    st.sidebar.header("Database Connection")
    db_user = st.sidebar.text_input("Database Username")
    db_password = st.sidebar.text_input("Database Password", type="password")
    db_host = st.sidebar.text_input("Database Host")
    db_port = st.sidebar.text_input("Database Port", "5432")
    db_name = st.sidebar.text_input("Database Name")
    
    if st.sidebar.button("Connect to Database"):
        try:
            db_uri = (
                f"postgresql+psycopg2://{db_user}:{db_password}"
                f"@{db_host}:{db_port}/{db_name}"
            )
            engine = create_engine(db_uri)
            st.sidebar.success("Connected to database successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to connect: {e}")
            st.stop()
    else:
        st.error("Please connect to the database to continue")
        st.stop()

# --- Constants and Configuration ---
# Define exchanges
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "surfFuture"]
exchanges_display = {
    "binanceFuture": "Binance",
    "gateFuture": "Gate",
    "hyperliquidFuture": "Hyperliquid",
    "surfFuture": "SurfFuture"
}

# Define time parameters
interval_minutes = 10  # 10-minute intervals
singapore_timezone = pytz.timezone('Asia/Singapore')

# --- Utility Functions ---
# Function to convert time string to sortable minutes value
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Function to determine if a token is a major
def is_major(token):
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
            return True
    return False

# Function to get appropriate depth tiers based on token type
def get_depth_tiers(token):
    if is_major(token):
        return [50000, 100000, 200000, 500000]  # Majors: 50k, 100k, 200k, 500k
    else:
        return [20000, 50000, 100000, 200000]   # Altcoins: 20k, 50k, 100k, 200k

# --- Data Fetching Functions ---
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_all_tokens():
    """Fetch all available tokens from the database"""
    query = """
    SELECT DISTINCT pair_name 
    FROM oracle_exchange_fee 
    ORDER BY pair_name
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default fallback

@st.cache_data(ttl=600, show_spinner="Fetching spread data...")
def fetch_10min_spread_data(token, lookback_days=1):
    """Fetch 10-minute spread data for a specific token"""
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # Query to get the fee data for the specified token
        query = f"""
        SELECT 
            time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
            pair_name,
            source,
            fee1, fee2, fee3, fee4,
            total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name = '{token}'
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
        ORDER BY 
            timestamp ASC,
            source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
            
    except Exception as e:
        print(f"[{token}] Error fetching spread data: {e}")
        import traceback
        traceback.print_exc()
        return None

@st.cache_data(ttl=600, show_spinner="Analyzing spread data...")
def analyze_10min_spread_data(df, token):
    """Process and analyze 10-minute spread data"""
    try:
        if df is None or df.empty:
            return None, None, None, None
            
        # Process each exchange separately
        exchange_dfs = {}
        for exchange in exchanges:
            exchange_df = df[df['source'] == exchange].copy()
            if not exchange_df.empty:
                exchange_df = exchange_df.set_index('timestamp')
                exchange_dfs[exchanges_display[exchange]] = exchange_df
        
        # If no valid data found for any exchange
        if not exchange_dfs:
            return None, None, None, None
        
        # Prepare result data frames for each fee level
        result_dfs = {}
        fee_columns = ['fee1', 'fee2', 'fee3', 'fee4']
        
        for fee_col in fee_columns:
            # Create DataFrame with all timestamps across exchanges
            timestamps = []
            for exchange, ex_df in exchange_dfs.items():
                if fee_col in ex_df.columns:
                    timestamps.extend(ex_df.index)
            unique_timestamps = sorted(set(timestamps))
            
            # Create DataFrame with all timestamps and exchanges for this fee level
            result_df = pd.DataFrame(index=unique_timestamps)
            
            for exchange, ex_df in exchange_dfs.items():
                if fee_col in ex_df.columns:
                    result_df[exchange] = pd.Series(ex_df[fee_col].values, index=ex_df.index)
            
            # Add time label column
            result_df['time_label'] = result_df.index.strftime('%H:%M')
            
            # Calculate non-surf average (Binance, Gate, Hyperliquid)
            non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
            # Only include columns that exist in the DataFrame
            non_surf_columns = [col for col in non_surf_columns if col in result_df.columns]
            
            if non_surf_columns:
                # Calculate row by row to handle missing values
                non_surf_avg = []
                for idx, row in result_df.iterrows():
                    values = [row[col] for col in non_surf_columns if not pd.isna(row[col])]
                    if values:
                        non_surf_avg.append(sum(values) / len(values))
                    else:
                        non_surf_avg.append(np.nan)
                result_df['Avg (Non-Surf)'] = non_surf_avg
            
            # Store this fee level's result DataFrame
            result_dfs[fee_col] = result_df
        
        # Calculate daily averages for each exchange and fee level
        daily_avgs = {}
        for fee_col, result_df in result_dfs.items():
            daily_avgs[fee_col] = {}
            for column in result_df.columns:
                if column != 'time_label':
                    values = result_df[column].dropna().values
                    if len(values) > 0:
                        daily_avgs[fee_col][column] = sum(values) / len(values)
                    else:
                        daily_avgs[fee_col][column] = np.nan
        
        # Create summary dataframes for each exchange
        summary_dfs = {}
        for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture', 'Avg (Non-Surf)']:
            summary_df = pd.DataFrame(index=unique_timestamps)
            for fee_col in fee_columns:
                if exchange in result_dfs[fee_col].columns:
                    summary_df[fee_col] = result_dfs[fee_col][exchange]
            summary_df['time_label'] = summary_df.index.strftime('%H:%M')
            summary_dfs[exchange] = summary_df
        
        return result_dfs, daily_avgs, summary_dfs, df
            
    except Exception as e:
        print(f"[{token}] Error analyzing spread data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

@st.cache_data(ttl=600, show_spinner="Fetching daily spread averages...")
def fetch_daily_spread_averages(tokens, lookback_days=1):
    """Fetch daily spread averages for multiple tokens"""
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Create placeholders for tokens
        tokens_str = "', '".join(tokens)

        # Query to get average fee data for all selected tokens
        query = f"""
        SELECT 
            pair_name,
            source,
            AVG(fee1) as avg_fee1,
            AVG(fee2) as avg_fee2,
            AVG(fee3) as avg_fee3,
            AVG(fee4) as avg_fee4,
            AVG(total_fee) as avg_total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name IN ('{tokens_str}')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        return df
            
    except Exception as e:
        st.error(f"Error fetching daily spread averages: {e}")
        print(f"Error fetching daily spread averages: {e}")
        return None

def calculate_matrix_data(avg_data):
    """Transform the average data into matrix format for display"""
    if avg_data is None or avg_data.empty:
        return None
    
    # Create pivot tables for each fee level
    matrix_data = {}
    fee_columns = ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4']
    
    for fee_col in fee_columns:
        # Convert the long format to wide format (pivot)
        pivot_df = avg_data.pivot(index='pair_name', columns='source', values=fee_col).reset_index()
        
        # Rename columns to display names
        pivot_df = pivot_df.rename(columns=exchanges_display)
        
        # Calculate non-surf average
        non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
        non_surf_columns = [col for col in non_surf_columns if col in pivot_df.columns]
        
        if non_surf_columns:
            pivot_df['Avg (Non-Surf)'] = pivot_df[non_surf_columns].mean(axis=1)
        
        # Add a column indicating if SurfFuture is better than non-surf avg
        if 'SurfFuture' in pivot_df.columns and 'Avg (Non-Surf)' in pivot_df.columns:
            pivot_df['Surf Better'] = pivot_df['SurfFuture'] < pivot_df['Avg (Non-Surf)']
        
        # Store the pivot table for this fee level
        matrix_data[fee_col.replace('avg_', '')] = pivot_df
    
    return matrix_data

def format_with_color(val, is_better=False):
    """Format a cell with color based on its value and comparison"""
    color = 'green' if is_better else 'black'
    return f'<span style="color: {color};">{val}</span>'

# --- Main Application ---
st.markdown('<div class="header-style">Exchange Spread Analysis Dashboard</div>', unsafe_allow_html=True)

# Fetch all available tokens
all_tokens = fetch_all_tokens()

# Sidebar filters
st.sidebar.header("Filters")

# Token selection
st.sidebar.subheader("Token Selection")
select_all = st.sidebar.checkbox("Select All Tokens", value=False)

if select_all:
    selected_tokens = all_tokens
else:
    # Group tokens as majors and altcoins
    major_tokens = [t for t in all_tokens if is_major(t)]
    altcoin_tokens = [t for t in all_tokens if not is_major(t)]
    
    # Create expandable sections for majors and altcoins
    with st.sidebar.expander("Major Tokens", expanded=True):
        selected_majors = st.multiselect(
            "Select Major Tokens", 
            major_tokens,
            default=major_tokens[:3] if major_tokens else []
        )
    
    with st.sidebar.expander("Altcoin Tokens", expanded=True):
        selected_altcoins = st.multiselect(
            "Select Altcoin Tokens", 
            altcoin_tokens,
            default=altcoin_tokens[:3] if altcoin_tokens else []
        )
    
    selected_tokens = selected_majors + selected_altcoins

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Other filters
lookback_days = st.sidebar.slider("Lookback Period (Days)", min_value=1, max_value=7, value=1)

# Add a refresh button
if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

# Create tabs for different analyses
tab1, tab2 = st.tabs(["📊 Daily Average Spreads", "🔍 Detailed Spread Analysis"])

with tab1:
    st.markdown('<div class="subheader-style">Daily Average Spread Analysis (10-minute intervals)</div>', unsafe_allow_html=True)
    
    # Fetch daily spread averages for all selected tokens
    daily_avg_data = fetch_daily_spread_averages(selected_tokens, lookback_days)
    
    if daily_avg_data is not None and not daily_avg_data.empty:
        # Calculate matrix data for display
        matrix_data = calculate_matrix_data(daily_avg_data)
        
        if matrix_data:
            # Display a matrix table for each fee level
            fee_depths = {
                'fee1': "Depth Tier 1",
                'fee2': "Depth Tier 2",
                'fee3': "Depth Tier 3",
                'fee4': "Depth Tier 4"
            }
            
            for fee_key, fee_name in fee_depths.items():
                if fee_key in matrix_data:
                    st.markdown(f"### {fee_name} Spreads")
                    
                    df = matrix_data[fee_key]
                    
                    # Determine scale factor for better readability
                    scale_factor = 1
                    scale_label = ""
                    
                    # Calculate mean for scaling
                    numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better']]
                    if numeric_cols:
                        values = []
                        for col in numeric_cols:
                            values.extend(df[col].dropna().tolist())
                        
                        if values:
                            mean_fee = sum(values) / len(values)
                            
                            # Determine scale factor based on mean fee value
                            if mean_fee < 0.001:
                                scale_factor = 1000
                                scale_label = "× 1,000"
                            elif mean_fee < 0.0001:
                                scale_factor = 10000
                                scale_label = "× 10,000"
                            elif mean_fee < 0.00001:
                                scale_factor = 100000
                                scale_label = "× 100,000"
                    
                    # Apply scaling if needed
                    if scale_factor > 1:
                        for col in numeric_cols:
                            df[col] = df[col] * scale_factor
                        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                    
                    # Format and display the dataframe
                    display_df = df.copy()
                    
                    # Round values for display
                    for col in numeric_cols:
                        display_df[col] = display_df[col].round(6)
                    
                    # Sort by token name
                    display_df = display_df.sort_values(by='pair_name')
                    
                    # Define column order with SurfFuture at the end
                    desired_order = ['pair_name', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
                    ordered_columns = [col for col in desired_order if col in display_df.columns]
                    
                    # Add Surf Better column if it exists
                    if 'Surf Better' in display_df.columns:
                        ordered_columns.append('Surf Better')
                    
                    display_df = display_df[ordered_columns]
                    
                    # Rename columns for display
                    display_df = display_df.rename(columns={'pair_name': 'Token'})
                    
                    # Display the table
                    token_count = len(display_df)
                    table_height = max(100 + 35 * token_count, 300)  # Minimum height of 300px
                    st.dataframe(display_df, height=table_height, use_container_width=True)
                    
                    # Check if SurfFuture is better overall
                    if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
                        surf_values = display_df['SurfFuture'].dropna()
                        nonsurf_values = display_df['Avg (Non-Surf)'].dropna()
                        
                        if not surf_values.empty and not nonsurf_values.empty:
                            # Match indices to compare only pairs with both values
                            common_indices = surf_values.index.intersection(nonsurf_values.index)
                            if len(common_indices) > 0:
                                surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                                total_count = len(common_indices)
                                
                                if surf_better_count > 0:
                                    st.success(f"🏆 **SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_count/total_count*100:.1f}%)**")
                                
                                # Calculate averages
                                surf_avg = surf_values.mean()
                                nonsurf_avg = nonsurf_values.mean()
                                
                                if surf_avg < nonsurf_avg:
                                    st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is lower than other exchanges ({nonsurf_avg:.6f})**")
                    
                    # Add visualization - which exchange has the lowest spread for each token
                    st.markdown("#### Exchange with Lowest Spread by Token")
                    
                    # Create a copy for visualization
                    viz_df = df.copy()
                    
                    # Exclude non-comparison columns
                    exclude_cols = ['pair_name', 'Avg (Non-Surf)', 'Surf Better']
                    exchange_cols = [col for col in viz_df.columns if col not in exclude_cols]
                    
                    # Determine the best exchange for each token
                    best_exchange = []
                    for _, row in viz_df.iterrows():
                        best_ex = "None"
                        best_val = float('inf')
                        
                        for ex in exchange_cols:
                            if not pd.isna(row[ex]) and row[ex] < best_val:
                                best_val = row[ex]
                                best_ex = ex
                        
                        best_exchange.append(best_ex)
                    
                    viz_df['Best Exchange'] = best_exchange
                    
                    # Count occurrences of each exchange
                    exchange_counts = viz_df['Best Exchange'].value_counts().reset_index()
                    exchange_counts.columns = ['Exchange', 'Count']
                    
                    # Create a pie chart
                    if not exchange_counts.empty:
                        fig = px.pie(
                            exchange_counts, 
                            values='Count', 
                            names='Exchange',
                            title=f"Exchange with Lowest Spread ({fee_name})",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(height=400)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")  # Add separator between fee levels
    else:
        st.warning("No daily average spread data available for the selected tokens and time period.")

with tab2:
    st.markdown('<div class="subheader-style">Detailed Spread Analysis by Token</div>', unsafe_allow_html=True)
    
    # Select a token for detailed analysis
    selected_token = st.selectbox("Select Token for Detailed Analysis", selected_tokens)
    
    if selected_token:
        # Fetch and analyze spread data for the selected token
        raw_data = fetch_10min_spread_data(selected_token, lookback_days)
        
        if raw_data is not None and not raw_data.empty:
            result_dfs, daily_avgs, summary_dfs, _ = analyze_10min_spread_data(raw_data, selected_token)
            
            if result_dfs and daily_avgs:
                # Create subtabs for different views
                subtab1, subtab2 = st.tabs(["Exchange Comparison", "Individual Exchange Analysis"])
                
                with subtab1:
                    st.markdown(f"### {selected_token} - Exchange Comparison by Depth Tier")
                    
                    # Display tables for each fee level
                    fee_depths = {
                        'fee1': "Depth Tier 1",
                        'fee2': "Depth Tier 2",
                        'fee3': "Depth Tier 3",
                        'fee4': "Depth Tier 4"
                    }
                    
                    for fee_key, fee_name in fee_depths.items():
                        if fee_key in result_dfs:
                            st.markdown(f"#### {fee_name} Spread")
                            
                            df = result_dfs[fee_key].copy()
                            
                            # Determine scale factor for better readability
                            scale_factor = 1
                            scale_label = ""
                            
                            # Calculate mean for scaling
                            numeric_cols = [col for col in df.columns if col != 'time_label']
                            if numeric_cols:
                                values = []
                                for col in numeric_cols:
                                    values.extend(df[col].dropna().tolist())
                                
                                if values:
                                    mean_fee = sum(values) / len(values)
                                    
                                    # Determine scale factor based on mean fee value
                                    if mean_fee < 0.001:
                                        scale_factor = 1000
                                        scale_label = "× 1,000"
                                    elif mean_fee < 0.0001:
                                        scale_factor = 10000
                                        scale_label = "× 10,000"
                                    elif mean_fee < 0.00001:
                                        scale_factor = 100000
                                        scale_label = "× 100,000"
                            
                            # Apply scaling if needed
                            if scale_factor > 1:
                                for col in numeric_cols:
                                    df[col] = df[col] * scale_factor
                                st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                            
                            # Sort by time in descending order
                            df = df.sort_values(by='time_label', key=lambda x: x.map(time_to_minutes), ascending=False)
                            
                            # Define column order with SurfFuture at the end
                            desired_order = ['time_label', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
                            ordered_columns = [col for col in desired_order if col in df.columns]
                            df = df[ordered_columns]
                            
                            # Round values for display
                            for col in numeric_cols:
                                if col in df.columns:
                                    df[col] = df[col].round(6)
                            
                            # Display the table
                            st.dataframe(df, height=400, use_container_width=True)
                            
                            # Check if SurfFuture has tighter spread than average
                            if 'SurfFuture' in df.columns and 'Avg (Non-Surf)' in df.columns:
                                surf_values = df['SurfFuture'].dropna()
                                nonsurf_values = df['Avg (Non-Surf)'].dropna()
                                
                                if not surf_values.empty and not nonsurf_values.empty:
                                    # Match indices to compare only pairs with both values
                                    common_indices = surf_values.index.intersection(nonsurf_values.index)
                                    if len(common_indices) > 0:
                                        surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                                        total_count = len(common_indices)
                                        
                                        if surf_better_count > 0:
                                            st.success(f"✅ **SurfFuture has tighter spreads for {surf_better_count}/{total_count} time periods ({surf_better_count/total_count*100:.1f}%)**")
                                        
                                        # Calculate averages
                                        surf_avg = surf_values.mean()
                                        nonsurf_avg = nonsurf_values.mean()
                                        
                                        if surf_avg < nonsurf_avg:
                                            st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is lower than other exchanges ({nonsurf_avg:.6f})**")
                            
                            # Add visualization - line chart comparing exchanges over time
                            st.markdown("#### Exchange Spread Comparison Over Time")
                            
                            # Create a line chart
                            viz_df = df.copy().reset_index()
                            
                            # Sort by time in ascending order for the chart
                            viz_df = viz_df.sort_values(by='time_label', key=lambda x: x.map(time_to_minutes))
                            
                            # Create the line chart
                            fig = go.Figure()
                            
                            for col in ordered_columns:
                                if col != 'time_label':
                                    fig.add_trace(go.Scatter(
                                        x=viz_df['time_label'],
                                        y=viz_df[col],
                                        mode='lines+markers',
                                        name=col
                                    ))
                            
                            fig.update_layout(
                                title=f"{selected_token} - {fee_name} Spread Over Time {scale_label}",
                                xaxis_title="Time (Singapore)",
                                yaxis_title=f"Spread {scale_label}",
                                height=400,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("---")  # Add separator between fee levels
                    
                    # Summary of daily averages
                    st.markdown("### Daily Average Summary")
                    
                    # Create a summary table
                    summary_data = []
                    for fee_key, fee_name in fee_depths.items():
                        if fee_key in daily_avgs:
                            row_data = {'Depth Tier': fee_name}
                            for exchange, value in daily_avgs[fee_key].items():
                                row_data[exchange] = value
                            summary_data.append(row_data)
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Determine scale factor for better readability
                        scale_factor = 1
                        scale_label = ""
                        
                        numeric_cols = [col for col in summary_df.columns if col != 'Depth Tier']
                        if numeric_cols:
                            values = []
                            for col in numeric_cols:
                                values.extend(summary_df[col].dropna().tolist())
                            
                            if values:
                                mean_fee = sum(values) / len(values)
                                
                                # Determine scale factor based on mean fee value
                                if mean_fee < 0.001:
                                    scale_factor = 1000
                                    scale_label = "× 1,000"
                                elif mean_fee < 0.0001:
                                    scale_factor = 10000
                                    scale_label = "× 10,000"
                                elif mean_fee < 0.00001:
                                    scale_factor = 100000
                                    scale_label = "× 100,000"
                        
                        # Apply scaling if needed
                        if scale_factor > 1:
                            for col in numeric_cols:
                                summary_df[col] = summary_df[col] * scale_factor
                            st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                        
                        # Round values for display
                        for col in numeric_cols:
                            summary_df[col] = summary_df[col].round(6)
                        
                        # Define column order with SurfFuture at the end
                        desired_order = ['Depth Tier', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
                        ordered_columns = [col for col in desired_order if col in summary_df.columns]
                        summary_df = summary_df[ordered_columns]
                        
                        # Display the summary table
                        st.dataframe(summary_df, height=200, use_container_width=True)
                        
                        # Create a bar chart comparing exchanges across depth tiers
                        st.markdown("#### Exchange Comparison Across Depth Tiers")
                        
                        # Prepare data for grouped bar chart
                        fig = go.Figure()
                        
                        # Add bars for each exchange
                        for col in numeric_cols:
                            fig.add_trace(go.Bar(
                                x=summary_df['Depth Tier'],
                                y=summary_df[col],
                                name=col,
                                text=summary_df[col].round(6).astype(str),
                                textposition='auto'
                            ))
                        
                        fig.update_layout(
                            title=f"{selected_token} - Average Spread by Depth Tier {scale_label}",
                            xaxis_title="Depth Tier",
                            yaxis_title=f"Average Spread {scale_label}",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with subtab2:
                    st.markdown(f"### {selected_token} - Individual Exchange Analysis")
                    
                    # Select an exchange for detailed analysis
                    exchange_options = ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture', 'Avg (Non-Surf)']
                    selected_exchange = st.selectbox("Select Exchange", exchange_options)
                    
                    if selected_exchange and selected_exchange in summary_dfs:
                        st.markdown(f"#### {selected_exchange} - Spread Analysis by Depth Tier")
                        
                        df = summary_dfs[selected_exchange].copy()
                        
                        # Determine scale factor for better readability
                        scale_factor = 1
                        scale_label = ""
                        
                        # Calculate mean for scaling
                        numeric_cols = [col for col in df.columns if col != 'time_label']
                        if numeric_cols:
                            values = []
                            for col in numeric_cols:
                                values.extend(df[col].dropna().tolist())
                            
                            if values:
                                mean_fee = sum(values) / len(values)
                                
                                # Determine scale factor based on mean fee value
                                if mean_fee < 0.001:
                                    scale_factor = 1000
                                    scale_label = "× 1,000"
                                elif mean_fee < 0.0001:
                                    scale_factor = 10000
                                    scale_label = "× 10,000"
                                elif mean_fee < 0.00001:
                                    scale_factor = 100000
                                    scale_label = "× 100,000"
                        
                        # Apply scaling if needed
                        if scale_factor > 1:
                            for col in numeric_cols:
                                df[col] = df[col] * scale_factor
                            st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                        
                        # Sort by time in descending order
                        df = df.sort_values(by='time_label', key=lambda x: x.map(time_to_minutes), ascending=False)
                        
                        # Round values for display
                        for col in numeric_cols:
                            df[col] = df[col].round(6)
                        
                        # Display the table
                        st.dataframe(df, height=400, use_container_width=True)
                        
                        # Create a line chart for this exchange across depth tiers
                        st.markdown(f"#### {selected_exchange} - Spread Comparison Across Depth Tiers")
                        
                        # Prepare data for visualization
                        viz_df = df.copy().reset_index()
                        
                        # Sort by time in ascending order for the chart
                        viz_df = viz_df.sort_values(by='time_label', key=lambda x: x.map(time_to_minutes))
                        
                        # Create the line chart
                        fig = go.Figure()
                        
                        fee_names = {
                            'fee1': "Depth Tier 1",
                            'fee2': "Depth Tier 2",
                            'fee3': "Depth Tier 3",
                            'fee4': "Depth Tier 4"
                        }
                        
                        for col in numeric_cols:
                            fig.add_trace(go.Scatter(
                                x=viz_df['time_label'],
                                y=viz_df[col],
                                mode='lines+markers',
                                name=fee_names.get(col, col)
                            ))
                        
                        fig.update_layout(
                            title=f"{selected_token} - {selected_exchange} Spread by Depth Tier {scale_label}",
                            xaxis_title="Time (Singapore)",
                            yaxis_title=f"Spread {scale_label}",
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show average spread by depth tier for this exchange
                        st.markdown(f"#### {selected_exchange} - Average Spread by Depth Tier")
                        
                        # Calculate averages for each fee level
                        avg_data = []
                        for col in numeric_cols:
                            values = df[col].dropna().values
                            if len(values) > 0:
                                avg_value = sum(values) / len(values)
                                avg_data.append({
                                    'Depth Tier': fee_names.get(col, col),
                                    'Average Spread': avg_value
                                })
                        
                        if avg_data:
                            avg_df = pd.DataFrame(avg_data)
                            
                            # Round values for display
                            avg_df['Average Spread'] = avg_df['Average Spread'].round(6)
                            
                            # Create a bar chart
                            fig = px.bar(
                                avg_df,
                                x='Depth Tier',
                                y='Average Spread',
                                text='Average Spread',
                                title=f"{selected_token} - {selected_exchange} Average Spread by Depth Tier {scale_label}",
                                height=400,
                                color_discrete_sequence=['#1E88E5']
                            )
                            
                            fig.update_layout(
                                xaxis_title="Depth Tier",
                                yaxis_title=f"Average Spread {scale_label}"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No spread analysis data available for {selected_token}")
        else:
            st.warning(f"No raw spread data available for {selected_token}")

# Footer with explanatory information
with st.expander("Understanding Exchange Spreads"):
    st.markdown("""
    ### About This Dashboard
    
    This dashboard provides comprehensive analysis of trading spreads across multiple cryptocurrency exchanges. 
    
    ### Key Concepts:
    
    - **Spread**: The difference between the buy and sell price, representing the cost of trading.
    
    - **Depth Tiers**: Different price levels based on order size:
      - **Majors** (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K
      - **Altcoins**: 20K, 50K, 100K, 200K
    
    - **Fee Columns**: 
      - `fee1`, `fee2`, `fee3`, `fee4` correspond to spreads at different depth tiers
    
    - **Exchange Comparison**: The dashboard highlights when SurfFuture has tighter spreads (lower fees) than the average of other exchanges.
    
    ### Interpreting the Data:
    
    - Lower spread values indicate better pricing for traders
    - Green highlights indicate where SurfFuture outperforms other exchanges
    - The scaling factor is applied to make small decimal values more readable
    
    ### Data Source:
    
    Data is fetched from the `oracle_exchange_fee` table, with 10-minute interval data points that represent the average of previous 10 one-minute points.""")
    "# Execute the app
if __name__ == '__main__':
    pass  # The app is already running