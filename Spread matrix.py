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

# Apply custom CSS styling - more minimal design with centered numeric columns and color coding
st.markdown("""
<style>
    .header-style {
        font-size:24px !important;
        font-weight: bold;
        padding: 10px 0;
    }
    .subheader-style {
        font-size:20px !important;
        font-weight: bold;
        padding: 5px 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    /* Simplified tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 15px;
        background-color: #f5f5f5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4682B4;
        color: white;
    }
    
    /* Center numeric columns in dataframes */
    .dataframe th, .dataframe td {
        text-align: center !important;
    }
    /* First column (Token) remains left-aligned */
    .dataframe th:first-child, .dataframe td:first-child {
        text-align: left !important;
    }
    
    /* Color coding for spread values */
    .very-low-spread {
        background-color: rgba(0, 128, 0, 0.1) !important;
        color: #006600 !important;
        font-weight: bold;
    }
    .low-spread {
        background-color: rgba(0, 102, 204, 0.1) !important;
        color: #0066cc !important;
    }
    .medium-spread {
        background-color: rgba(255, 153, 0, 0.1) !important;
        color: #ff9900 !important;
    }
    .high-spread {
        background-color: rgba(204, 0, 0, 0.1) !important;
        color: #cc0000 !important;
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
lookback_days = 1  # Fixed to 1 day

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

# Function to get depth label based on token type
def get_depth_label(fee_column, token):
    if is_major(token):
        depth_map = {
            'fee1': '50K', 'fee2': '100K', 'fee3': '200K', 'fee4': '500K',
            'avg_fee1': '50K', 'avg_fee2': '100K', 'avg_fee3': '200K', 'avg_fee4': '500K'
        }
    else:
        depth_map = {
            'fee1': '20K', 'fee2': '50K', 'fee3': '100K', 'fee4': '200K',
            'avg_fee1': '20K', 'avg_fee2': '50K', 'avg_fee3': '100K', 'avg_fee4': '200K'
        }
    return depth_map.get(fee_column, fee_column)

# Replace the existing color_code_value function with this:
def color_code_value(value, thresholds=None):
    """Apply bold formatting to spread values without color backgrounds"""
    if pd.isna(value):
        return value
    
    # Always use bold formatting for better readability
    return f'<span style="font-weight: bold;">{value:.6f}</span>'
    
    if thresholds is None:
        # Default thresholds - adjust based on your data
        thresholds = [0.0005, 0.001, 0.005]
        
    if value < thresholds[0]:
        return f'<span class="very-low-spread">{value:.6f}</span>'
    elif value < thresholds[1]:
        return f'<span class="low-spread">{value:.6f}</span>'
    elif value < thresholds[2]:
        return f'<span class="medium-spread">{value:.6f}</span>'
    else:
        return f'<span class="high-spread">{value:.6f}</span>'

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
def fetch_10min_spread_data(token):
    """Fetch 10-minute spread data for a specific token (last 24 hours)"""
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # Query to get the fee data for the specified token
        # Special handling for SurfFuture: Use total_fee for all fee levels
        query = f"""
        SELECT 
            time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
            pair_name,
            source,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee1 END as fee1,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee2 END as fee2,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee3 END as fee3,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee4 END as fee4,
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

@st.cache_data(ttl=600, show_spinner="Fetching daily spread averages...")
def fetch_daily_spread_averages(tokens):
    """Fetch daily spread averages for multiple tokens (last 24 hours)"""
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
        # Special handling for SurfFuture: Use total_fee for all fee levels
        query = f"""
        SELECT 
            pair_name,
            source,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee1) END as avg_fee1,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee2) END as avg_fee2,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee3) END as avg_fee3,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee4) END as avg_fee4,
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
    fee_columns = [col for col in ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4'] 
                  if col in avg_data.columns]
    
    if not fee_columns:
        st.error("Required fee columns not found in the data. Check database schema.")
        return None
    
    for fee_col in fee_columns:
        try:
            # Convert the long format to wide format (pivot)
            pivot_df = avg_data.pivot(index='pair_name', columns='source', values=fee_col).reset_index()
            
            # Rename columns to display names
            for source in exchanges:
                if source in pivot_df.columns:
                    pivot_df = pivot_df.rename(columns={source: exchanges_display[source]})
            
            # Calculate non-surf average
            non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
            available_non_surf = [col for col in non_surf_columns if col in pivot_df.columns]
            
            if available_non_surf:
                pivot_df['Avg (Non-Surf)'] = pivot_df[available_non_surf].mean(axis=1)
            
            # Add a column indicating if SurfFuture is better than non-surf avg
            if 'SurfFuture' in pivot_df.columns and 'Avg (Non-Surf)' in pivot_df.columns:
                pivot_df['Surf Better'] = pivot_df['SurfFuture'] < pivot_df['Avg (Non-Surf)']
                
                # Calculate percentage improvement
                pivot_df['Improvement %'] = ((pivot_df['Avg (Non-Surf)'] - pivot_df['SurfFuture']) / 
                                            pivot_df['Avg (Non-Surf)'] * 100).round(2)
            
            # Store the pivot table for this fee level
            matrix_data[fee_col.replace('avg_', '')] = pivot_df
        except Exception as e:
            st.error(f"Error processing {fee_col}: {e}")
    
    return matrix_data

# --- Main Application ---
st.markdown('<div class="header-style">Exchange Spread Analysis Dashboard</div>', unsafe_allow_html=True)

# Fetch all available tokens
all_tokens = fetch_all_tokens()

# Sidebar - simplified, just refresh button
st.sidebar.header("Controls")

# Always select all tokens
selected_tokens = all_tokens

# Add a refresh button
if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["50K/20K Analysis", "100K/50K Analysis", "Spread By Size"])

# Get daily spread data (used in all tabs)
daily_avg_data = fetch_daily_spread_averages(selected_tokens)
if daily_avg_data is None or daily_avg_data.empty:
    st.warning("No spread data available for the selected time period.")
    st.stop()

# Calculate matrix data for all fee levels
matrix_data = calculate_matrix_data(daily_avg_data)
if matrix_data is None or not matrix_data:
    st.warning("Unable to process spread data. Check log for details.")
    st.stop()

# === TAB 1: 50K/20K ANALYSIS (fee1) ===
with tab1:
    st.markdown('<div class="header-style">50K/20K Spread Analysis</div>', unsafe_allow_html=True)
    
    # Display explanation of depth tiers
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definition:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K<br>
    • <b>Altcoin tokens</b>: 20K<br>
    <br>
    This tab shows daily averages of 10-minute spread data points at 50K/20K size.
    </div>
    """, unsafe_allow_html=True)
    
    if 'fee1' in matrix_data:
        df = matrix_data['fee1']
        
        # Determine scale factor for better readability
        scale_factor = 10000
        scale_label = "x10,000"
        
        # Calculate mean for scaling
        numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
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
            display_df[col] = display_df[col].round(2)
        
        # Add token type column for clarity
        display_df['Token Type'] = display_df['pair_name'].apply(
            lambda x: 'Major' if is_major(x) else 'Altcoin'
        )
        
        # Sort by token type and then by name
        display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
        
        # Define column order with SurfFuture at the end
        desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
        if 'Improvement %' in display_df.columns:
            desired_order.append('Improvement %')
        ordered_columns = [col for col in desired_order if col in display_df.columns]
        
        # Add Surf Better column if it exists (hidden, used for filtering)
        if 'Surf Better' in display_df.columns:
            ordered_columns.append('Surf Better')
        
        display_df = display_df[ordered_columns]
        
        # Rename columns for display
        display_df = display_df.rename(columns={'pair_name': 'Token'})
        
        # Apply color coding to numeric columns
        color_df = display_df.copy()
        for col in numeric_cols:
            if col in color_df.columns and col != 'Token Type':
                # Determine thresholds based on column values
                values = color_df[col].dropna().tolist()
                if values:
                    # Dynamic thresholds based on percentiles
                    q1 = np.percentile(values, 25)
                    median = np.percentile(values, 50)
                    q3 = np.percentile(values, 75)
                    thresholds = [q1, median, q3]
                    
                    color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
        
        # Special formatting for improvement percentage
        if 'Improvement %' in color_df.columns:
            color_df['Improvement %'] = color_df['Improvement %'].apply(
                lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            )
        
        # Display the table with HTML formatting
        token_count = len(color_df)
        table_height = max(100 + 35 * token_count, 300)  # Minimum height of 300px
        
        # Convert to HTML for better formatting
        html_table = color_df.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
        if 'Surf Better' in df.columns:
            st.markdown("### SurfFuture Performance Analysis")
            
            # Count tokens where SurfFuture is better
            surf_better_count = df['Surf Better'].sum()
            total_count = len(df)
            surf_worse_count = total_count - surf_better_count
            
            # Create pie chart
            fig = px.pie(
                values=[surf_better_count, surf_worse_count],
                names=['SurfFuture Better', 'Other Exchanges Better'],
                title="Proportion of Tokens Where SurfFuture Has Better Spreads",
                color_discrete_sequence=['#4CAF50', '#FFC107'],
                hole=0.4
            )
            
            # Update layout
            fig.update_layout(
                legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                margin=dict(t=60, b=60, l=20, r=20),
                height=400
            )
            
            # Display percentage text in middle
            better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
            fig.add_annotation(
                text=f"{better_percentage:.1f}%<br>Better",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary of SurfFuture performance
        if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
            surf_values = df['SurfFuture'].dropna()
            nonsurf_values = df['Avg (Non-Surf)'].dropna()
            
            if not surf_values.empty and not nonsurf_values.empty:
                # Match indices to compare only pairs with both values
                common_indices = surf_values.index.intersection(nonsurf_values.index)
                if len(common_indices) > 0:
                    surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                    total_count = len(common_indices)
                    
                    # Calculate percentages
                    surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                    
                    # Display summary box
                    st.markdown(f"""
                    <div class="info-box">
                    <b>SurfFuture Performance Summary:</b><br>
                    • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate averages
                    surf_avg = surf_values.mean()
                    nonsurf_avg = nonsurf_values.mean()
                    
                    if surf_avg < nonsurf_avg:
                        improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                        st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is {improvement:.2f}% lower than other exchanges ({nonsurf_avg:.6f})**")
                    
            # Calculate separate stats for majors and altcoins
            major_tokens_df = df[df['pair_name'].apply(is_major)]
            altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
            
            # For Major tokens
            if not major_tokens_df.empty:
                surf_major = major_tokens_df['SurfFuture'].dropna()
                nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_major.empty and not nonsurf_major.empty:
                    common_indices = surf_major.index.intersection(nonsurf_major.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_major_avg = surf_major.mean()
                            nonsurf_major_avg = nonsurf_major.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Major Tokens (50K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                            • SurfFuture average: {surf_major_avg:.6f} vs Non-Surf average: {nonsurf_major_avg:.6f}
                            </div>
                            """, unsafe_allow_html=True)
            
            # For Altcoin tokens
            if not altcoin_tokens_df.empty:
                surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                    common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_altcoin_avg = surf_altcoin.mean()
                            nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Altcoin Tokens (20K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                            • SurfFuture average: {surf_altcoin_avg:.6f} vs Non-Surf average: {nonsurf_altcoin_avg:.6f}
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Bar chart comparing exchanges
            st.markdown("### Average Spread by Exchange")
            
            # Calculate average for each exchange
            exchange_avgs = {}
            for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                if exchange in df.columns:
                    values = df[exchange].dropna()
                    if not values.empty:
                        exchange_avgs[exchange] = values.mean()
            
            if exchange_avgs:
                # Create data frame for plotting
                avg_df = pd.DataFrame({
                    'Exchange': list(exchange_avgs.keys()),
                    'Average Spread': list(exchange_avgs.values())
                })
                
                # Sort by average spread (ascending)
                avg_df = avg_df.sort_values('Average Spread')
                
                # Create bar chart
                fig = px.bar(
                    avg_df,
                    x='Exchange',
                    y='Average Spread',
                    title=f"Average Spread by Exchange ({scale_label})",
                    color='Exchange',
                    text='Average Spread',
                    color_discrete_map={
                        'SurfFuture': '#4CAF50',
                        'Binance': '#2196F3',
                        'Gate': '#FFC107',
                        'Hyperliquid': '#FF5722'
                    }
                )
                
                # Format the bars
                fig.update_traces(
                    texttemplate='%{y:.6f}',
                    textposition='outside'
                )
                
                # Format the layout
                fig.update_layout(
                    xaxis_title="Exchange",
                    yaxis_title=f"Average Spread {scale_label}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No fee1 (50K/20K) data available for analysis")

# === TAB 2: 100K/50K ANALYSIS (fee2) ===
with tab2:
    st.markdown('<div class="header-style">100K/50K Spread Analysis</div>', unsafe_allow_html=True)
    
    # Display explanation of depth tiers
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definition:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 100K<br>
    • <b>Altcoin tokens</b>: 50K<br>
    <br>
    This tab shows daily averages of 10-minute spread data points at 100K/50K size.""
    "</div>
    """, unsafe_allow_html=True)
    
    if 'fee2' in matrix_data:
        df = matrix_data['fee2']
        
        # Determine scale factor for better readability
        scale_factor = 10000
        scale_label = "x 10,000"
        
        # Calculate mean for scaling
        numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
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
            display_df[col] = display_df[col].round(2)
        
        # Add token type column for clarity
        display_df['Token Type'] = display_df['pair_name'].apply(
            lambda x: 'Major' if is_major(x) else 'Altcoin'
        )
        
        # Sort by token type and then by name
        display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
        
        # Define column order with SurfFuture at the end
        desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
        if 'Improvement %' in display_df.columns:
            desired_order.append('Improvement %')
        ordered_columns = [col for col in desired_order if col in display_df.columns]
        
        # Add Surf Better column if it exists (hidden, used for filtering)
        if 'Surf Better' in display_df.columns:
            ordered_columns.append('Surf Better')
        
        display_df = display_df[ordered_columns]
        
        # Rename columns for display
        display_df = display_df.rename(columns={'pair_name': 'Token'})
        
        # Apply color coding to numeric columns
        color_df = display_df.copy()
        for col in numeric_cols:
            if col in color_df.columns and col != 'Token Type':
                # Determine thresholds based on column values
                values = color_df[col].dropna().tolist()
                if values:
                    # Dynamic thresholds based on percentiles
                    q1 = np.percentile(values, 25)
                    median = np.percentile(values, 50)
                    q3 = np.percentile(values, 75)
                    thresholds = [q1, median, q3]
                    
                    color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
        
        # Special formatting for improvement percentage
        if 'Improvement %' in color_df.columns:
            color_df['Improvement %'] = color_df['Improvement %'].apply(
                lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            )
        
        # Display the table with HTML formatting
        token_count = len(color_df)
        table_height = max(100 + 35 * token_count, 300)  # Minimum height of 300px
        
        # Convert to HTML for better formatting
        html_table = color_df.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
        if 'Surf Better' in df.columns:
            st.markdown("### SurfFuture Performance Analysis")
            
            # Count tokens where SurfFuture is better
            surf_better_count = df['Surf Better'].sum()
            total_count = len(df)
            surf_worse_count = total_count - surf_better_count
            
            # Create pie chart
            fig = px.pie(
                values=[surf_better_count, surf_worse_count],
                names=['SurfFuture Better', 'Other Exchanges Better'],
                title="Proportion of Tokens Where SurfFuture Has Better Spreads",
                color_discrete_sequence=['#4CAF50', '#FFC107'],
                hole=0.4
            )
            
            # Update layout
            fig.update_layout(
                legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                margin=dict(t=60, b=60, l=20, r=20),
                height=400
            )
            
            # Display percentage text in middle
            better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
            fig.add_annotation(
                text=f"{better_percentage:.1f}%<br>Better",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary of SurfFuture performance
        if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
            surf_values = df['SurfFuture'].dropna()
            nonsurf_values = df['Avg (Non-Surf)'].dropna()
            
            if not surf_values.empty and not nonsurf_values.empty:
                # Match indices to compare only pairs with both values
                common_indices = surf_values.index.intersection(nonsurf_values.index)
                if len(common_indices) > 0:
                    surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                    total_count = len(common_indices)
                    
                    # Calculate percentages
                    surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                    
                    # Display summary box
                    st.markdown(f"""
                    <div class="info-box">
                    <b>SurfFuture Performance Summary:</b><br>
                    • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate averages
                    surf_avg = surf_values.mean()
                    nonsurf_avg = nonsurf_values.mean()
                    
                    if surf_avg < nonsurf_avg:
                        improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                        st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is {improvement:.2f}% lower than other exchanges ({nonsurf_avg:.6f})**")
                    
            # Calculate separate stats for majors and altcoins
            major_tokens_df = df[df['pair_name'].apply(is_major)]
            altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
            
            # For Major tokens
            if not major_tokens_df.empty:
                surf_major = major_tokens_df['SurfFuture'].dropna()
                nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_major.empty and not nonsurf_major.empty:
                    common_indices = surf_major.index.intersection(nonsurf_major.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_major_avg = surf_major.mean()
                            nonsurf_major_avg = nonsurf_major.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Major Tokens (100K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                            • SurfFuture average: {surf_major_avg:.6f} vs Non-Surf average: {nonsurf_major_avg:.6f}
                            </div>
                            """, unsafe_allow_html=True)
            
            # For Altcoin tokens
            if not altcoin_tokens_df.empty:
                surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                    common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_altcoin_avg = surf_altcoin.mean()
                            nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Altcoin Tokens (50K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                            • SurfFuture average: {surf_altcoin_avg:.6f} vs Non-Surf average: {nonsurf_altcoin_avg:.6f}
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Bar chart comparing exchanges
            st.markdown("### Average Spread by Exchange")
            
            # Calculate average for each exchange
            exchange_avgs = {}
            for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                if exchange in df.columns:
                    values = df[exchange].dropna()
                    if not values.empty:
                        exchange_avgs[exchange] = values.mean()
            
            if exchange_avgs:
                # Create data frame for plotting
                avg_df = pd.DataFrame({
                    'Exchange': list(exchange_avgs.keys()),
                    'Average Spread': list(exchange_avgs.values())
                })
                
                # Sort by average spread (ascending)
                avg_df = avg_df.sort_values('Average Spread')
                
                # Create bar chart
                fig = px.bar(
                    avg_df,
                    x='Exchange',
                    y='Average Spread',
                    title=f"Average Spread by Exchange ({scale_label})",
                    color='Exchange',
                    text='Average Spread',
                    color_discrete_map={
                        'SurfFuture': '#4CAF50',
                        'Binance': '#2196F3',
                        'Gate': '#FFC107',
                        'Hyperliquid': '#FF5722'
                    }
                )
                
                # Format the bars
                fig.update_traces(
                    texttemplate='%{y:.6f}',
                    textposition='outside'
                )
                
                # Format the layout
                fig.update_layout(
                    xaxis_title="Exchange",
                    yaxis_title=f"Average Spread {scale_label}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No fee2 (100K/50K) data available for analysis")

# === TAB 3: SPREAD BY SIZE (ALL SIZES) ===
with tab3:
    st.markdown('<div class="header-style">Spread Analysis by Size</div>', unsafe_allow_html=True)
    
    # Explanation of sizes
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definitions:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K<br>
    • <b>Altcoin tokens</b>: 20K, 50K, 100K, 200K<br>
    <br>
    Tables show daily averages of 10-minute spread data points across different size tiers.
    </div>
    """, unsafe_allow_html=True)
    
    if daily_avg_data is not None and not daily_avg_data.empty:
        # Group tokens as majors and altcoins for better organization
        major_tokens = [t for t in selected_tokens if is_major(t)]
        altcoin_tokens = [t for t in selected_tokens if not is_major(t)]
        
        # Create dictionary mapping for depths
        fee_depth_map_major = {
            'avg_fee1': '50K', 'avg_fee2': '100K', 'avg_fee3': '200K', 'avg_fee4': '500K'
        }
        fee_depth_map_altcoin = {
            'avg_fee1': '20K', 'avg_fee2': '50K', 'avg_fee3': '100K', 'avg_fee4': '200K'
        }
        
        # Check available fee columns
        available_fee_cols = [col for col in ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4'] 
                              if col in daily_avg_data.columns]
        
        if not available_fee_cols:
            st.error("Required fee columns not found in the data. Check database schema.")
        else:
            # --- Average of all exchanges (excluding SurfFuture) ---
            st.markdown("### Average Spreads Across Exchanges")
            
            try:
                # Create filtered data without surfFuture for comparison
                non_surf_data = daily_avg_data[daily_avg_data['source'].isin(['binanceFuture', 'gateFuture', 'hyperliquidFuture'])].copy()
                
                if not non_surf_data.empty:
                    # Calculate average across exchanges for each pair and depth
                    avg_all_exchanges = non_surf_data.groupby(['pair_name'])[available_fee_cols].mean().reset_index()
                    
                    # Process majors
                    if major_tokens:
                        major_df = avg_all_exchanges[avg_all_exchanges['pair_name'].isin(major_tokens)].copy()
                        if not major_df.empty:
                            # Rename columns for display
                            major_df_display = major_df.copy()
                            for col in available_fee_cols:
                                if col in fee_depth_map_major:
                                    major_df_display = major_df_display.rename(columns={col: fee_depth_map_major[col]})
                            
                            # Sort alphabetically
                            major_df_display = major_df_display.sort_values('pair_name')
                            
                            # Determine scale factor
                            depth_cols = [fee_depth_map_major[col] for col in available_fee_cols if col in fee_depth_map_major]
                            values = []
                            for col in depth_cols:
                                if col in major_df_display.columns:
                                    values.extend(major_df_display[col].dropna().tolist())
                            
                            scale_factor = 1
                            scale_label = ""
                            if values:
                                mean_fee = sum(values) / len(values)
                                if mean_fee < 0.001:
                                    scale_factor = 1,000
                                    scale_label = "× 1,000"
                                elif mean_fee < 0.0001:
                                    scale_factor = 10000
                                    scale_label = "× 10,000"
                                elif mean_fee < 0.00001:
                                    scale_factor = 100000
                                    scale_label = "× 100,000"
                            
                            # Apply scaling
                            if scale_factor > 1:
                                for col in depth_cols:
                                    if col in major_df_display.columns:
                                        major_df_display[col] = major_df_display[col] * scale_factor
                                
                                st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                            
                            # Apply color coding
                            color_df = major_df_display.copy()
                            for col in depth_cols:
                                if col in color_df.columns:
                                    # Determine thresholds based on column values
                                    col_values = color_df[col].dropna().tolist()
                                    if col_values:
                                        # Dynamic thresholds based on percentiles
                                        q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                        median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                        q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                        thresholds = [q1, median, q3]
                                        
                                        color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                            
                            # Round values for display
                            for col in depth_cols:
                                if col in major_df_display.columns:
                                    major_df_display[col] = major_df_display[col].round(2)
                            
                            # Rename pair_name column
                            color_df = color_df.rename(columns={'pair_name': 'Token'})
                            
                            st.markdown("#### Major Tokens - Average Across Exchanges")
                            
                            # Display the table with HTML formatting
                            html_table = color_df.to_html(escape=False, index=False)
                            st.markdown(html_table, unsafe_allow_html=True)
                            
                            # Create visualization - line chart showing spread by size
                            st.markdown("#### Spread vs. Size Relationship (Major Tokens)")
                            
                            # Calculate average spread for each size tier
                            size_averages = {}
                            for col in depth_cols:
                                if col in major_df_display.columns:
                                    size_averages[col] = major_df_display[col].mean()
                            
                            if size_averages:
                                # Create data frame for plotting
                                sizes = list(size_averages.keys())
                                spreads = list(size_averages.values())
                                
                                # Create line chart
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=sizes,
                                    y=spreads,
                                    mode='lines+markers',
                                    line=dict(color='#1E88E5', width=3),
                                    marker=dict(size=10, color='#1E88E5'),
                                    name='Average Spread'
                                ))
                                
                                # Format the layout
                                fig.update_layout(
                                    title=f"Relationship Between Size and Spread - Major Tokens ({scale_label})",
                                    xaxis_title="Size Tier",
                                    yaxis_title=f"Average Spread {scale_label}",
                                    height=400,
                                    xaxis=dict(
                                        tickfont=dict(size=14),
                                        tickmode='array',
                                        tickvals=sizes
                                    ),
                                    yaxis=dict(
                                        tickformat='.6f'
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Process altcoins
                    if altcoin_tokens:
                        altcoin_df = avg_all_exchanges[avg_all_exchanges['pair_name'].isin(altcoin_tokens)].copy()
                        if not altcoin_df.empty:
                            # Rename columns for display
                            altcoin_df_display = altcoin_df.copy()
                            for col in available_fee_cols:
                                if col in fee_depth_map_altcoin:
                                    altcoin_df_display = altcoin_df_display.rename(columns={col: fee_depth_map_altcoin[col]})
                            
                            # Sort alphabetically
                            altcoin_df_display = altcoin_df_display.sort_values('pair_name')
                            
                            # Determine scale factor
                            depth_cols = [fee_depth_map_altcoin[col] for col in available_fee_cols if col in fee_depth_map_altcoin]
                            values = []
                            for col in depth_cols:
                                if col in altcoin_df_display.columns:
                                    values.extend(altcoin_df_display[col].dropna().tolist())
                            
                            scale_factor = 1
                            scale_label = ""
                            if values:
                                mean_fee = sum(values) / len(values)
                                if mean_fee < 0.001:
                                    scale_factor = 1000
                                    scale_label = "× 1,000"
                                elif mean_fee < 0.0001:
                                    scale_factor = 10000
                                    scale_label = "× 10,000"
                                elif mean_fee < 0.00001:
                                    scale_factor = 100000
                                    scale_label = "× 100,000"
                            
                            # Apply scaling
                            if scale_factor > 1:
                                for col in depth_cols:
                                    if col in altcoin_df_display.columns:
                                        altcoin_df_display[col] = altcoin_df_display[col] * scale_factor
                                
                                st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                            
                            # Apply color coding
                            color_df = altcoin_df_display.copy()
                            for col in depth_cols:
                                if col in color_df.columns:
                                    # Determine thresholds based on column values
                                    col_values = color_df[col].dropna().tolist()
                                    if col_values:
                                        # Dynamic thresholds based on percentiles
                                        q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                        median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                        q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                        thresholds = [q1, median, q3]
                                        
                                        color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                            
                            # Round values for display
                            for col in depth_cols:
                                if col in altcoin_df_display.columns:
                                    altcoin_df_display[col] = altcoin_df_display[col].round(2)
                            
                            # Rename pair_name column
                            color_df = color_df.rename(columns={'pair_name': 'Token'})
                            
                            st.markdown("#### Altcoin Tokens - Average Across Exchanges")
                            
                            # Display the table with HTML formatting
                            html_table = color_df.to_html(escape=False, index=False)
                            st.markdown(html_table, unsafe_allow_html=True)
                            
                            # Create visualization - line chart showing spread by size
                            st.markdown("#### Spread vs. Size Relationship (Altcoin Tokens)")
                            
                            # Calculate average spread for each size tier
                            size_averages = {}
                            for col in depth_cols:
                                if col in altcoin_df_display.columns:
                                    size_averages[col] = altcoin_df_display[col].mean()
                            
                            if size_averages:
                                # Create data frame for plotting
                                sizes = list(size_averages.keys())
                                spreads = list(size_averages.values())
                                
                                # Create line chart
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=sizes,
                                    y=spreads,
                                    mode='lines+markers',
                                    line=dict(color='#FF9800', width=3),
                                    marker=dict(size=10, color='#FF9800'),
                                    name='Average Spread'
                                ))
                                
                                # Format the layout
                                fig.update_layout(
                                    title=f"Relationship Between Size and Spread - Altcoin Tokens ({scale_label})",
                                    xaxis_title="Size Tier",
                                    yaxis_title=f"Average Spread {scale_label}",
                                    height=400,
                                    xaxis=dict(
                                        tickfont=dict(size=14),
                                        tickmode='array',
                                        tickvals=sizes
                                    ),
                                    yaxis=dict(
                                        tickformat='.6f'
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                # --- Individual exchange analysis (excluding SurfFuture) ---
                st.markdown("### Individual Exchange Analysis")
                
                # Create tabs for each exchange (excluding SurfFuture)
                non_surf_exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture"]
                exchange_tabs = st.tabs([exchanges_display[ex] for ex in non_surf_exchanges])
                
                # Process each exchange
                for i, exchange_source in enumerate(non_surf_exchanges):
                    with exchange_tabs[i]:
                        exchange_display_name = exchanges_display[exchange_source]
                        st.markdown(f"#### {exchange_display_name} Spreads Analysis")
                        
                        # Filter data for this exchange
                        exchange_data = daily_avg_data[daily_avg_data['source'] == exchange_source].copy()
                        
                        if not exchange_data.empty:
                            # --- Process majors ---
                            if major_tokens:
                                major_ex_df = exchange_data[exchange_data['pair_name'].isin(major_tokens)].copy()
                                if not major_ex_df.empty:
                                    # Create a display DataFrame with available columns
                                    columns_to_select = ['pair_name'] + [col for col in available_fee_cols if col in major_ex_df.columns]
                                    major_ex_display = major_ex_df[columns_to_select].copy()
                                    
                                    # Rename columns for display
                                    for col in available_fee_cols:
                                        if col in fee_depth_map_major and col in major_ex_display.columns:
                                            major_ex_display = major_ex_display.rename(columns={col: fee_depth_map_major[col]})
                                    
                                    # Sort alphabetically
                                    major_ex_display = major_ex_display.sort_values('pair_name')
                                    
                                    # Determine scale factor
                                    depth_cols = [fee_depth_map_major[col] for col in available_fee_cols if col in fee_depth_map_major]
                                    values = []
                                    for col in depth_cols:
                                        if col in major_ex_display.columns:
                                            values.extend(major_ex_display[col].dropna().tolist())
                                    
                                    scale_factor = 1
                                    scale_label = ""
                                    if values:
                                        mean_fee = sum(values) / len(values)
                                        if mean_fee < 0.001:
                                            scale_factor = 1000
                                            scale_label = "× 1,000"
                                        elif mean_fee < 0.0001:
                                            scale_factor = 10000
                                            scale_label = "× 10,000"
                                        elif mean_fee < 0.00001:
                                            scale_factor = 100000
                                            scale_label = "× 100,000"
                                    
                                    # Apply scaling
                                    if scale_factor > 1:
                                        for col in depth_cols:
                                            if col in major_ex_display.columns:
                                                major_ex_display[col] = major_ex_display[col] * scale_factor
                                        
                                        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                    
                                    # Apply color coding
                                    color_df = major_ex_display.copy()
                                    for col in depth_cols:
                                        if col in color_df.columns:
                                            # Determine thresholds based on column values
                                            col_values = color_df[col].dropna().tolist()
                                            if col_values:
                                                # Dynamic thresholds based on percentiles
                                                q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                                median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                                q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                                thresholds = [q1, median, q3]
                                                
                                                color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                    
                                    # Round values for display
                                    for col in depth_cols:
                                        if col in major_ex_display.columns:
                                            major_ex_display[col] = major_ex_display[col].round(2)
                                    
                                    # Rename pair_name column
                                    color_df = color_df.rename(columns={'pair_name': 'Token'})
                                    
                                    st.markdown(f"##### Major Tokens - {exchange_display_name}")
                                    
                                    # Display the table with HTML formatting
                                    html_table = color_df.to_html(escape=False, index=False)
                                    st.markdown(html_table, unsafe_allow_html=True)
                                    
                                    # Create visualization - line chart showing spread by size
                                    st.markdown(f"##### Spread vs. Size Relationship - {exchange_display_name} (Major Tokens)")
                                    
                                    # Calculate average spread for each size tier
                                    size_averages = {}
                                    for col in depth_cols:
                                        if col in major_ex_display.columns:
                                            size_averages[col] = major_ex_display[col].mean()
                                    
                                    if size_averages:
                                        # Create data frame for plotting
                                        sizes = list(size_averages.keys())
                                        spreads = list(size_averages.values())
                                        
                                        # Create line chart
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Scatter(
                                            x=sizes,
                                            y=spreads,
                                            mode='lines+markers',
                                            line=dict(color='#1E88E5', width=3),
                                            marker=dict(size=10, color='#1E88E5'),
                                            name='Average Spread'
                                        ))
                                        
                                        # Format the layout
                                        fig.update_layout(
                                            title=f"{exchange_display_name} - Spread vs. Size (Major Tokens) {scale_label}",
                                            xaxis_title="Size Tier",
                                            yaxis_title=f"Average Spread {scale_label}",
                                            height=400,
                                            xaxis=dict(
                                                tickfont=dict(size=14),
                                                tickmode='array',
                                                tickvals=sizes
                                            ),
                                            yaxis=dict(
                                                tickformat='.6f'
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # --- Process altcoins ---
                            if altcoin_tokens:
                                altcoin_ex_df = exchange_data[exchange_data['pair_name'].isin(altcoin_tokens)].copy()
                                if not altcoin_ex_df.empty:
                                    # Create a display DataFrame with available columns
                                    columns_to_select = ['pair_name'] + [col for col in available_fee_cols if col in altcoin_ex_df.columns]
                                    altcoin_ex_display = altcoin_ex_df[columns_to_select].copy()
                                    
                                    # Rename columns for display
                                    for col in available_fee_cols:
                                        if col in fee_depth_map_altcoin and col in altcoin_ex_display.columns:
                                            altcoin_ex_display = altcoin_ex_display.rename(columns={col: fee_depth_map_altcoin[col]})
                                    
                                    # Sort alphabetically
                                    altcoin_ex_display = altcoin_ex_display.sort_values('pair_name')
                                    
                                    # Determine scale factor
                                    depth_cols = [fee_depth_map_altcoin[col] for col in available_fee_cols if col in fee_depth_map_altcoin]
                                    values = []
                                    for col in depth_cols:
                                        if col in altcoin_ex_display.columns:
                                            values.extend(altcoin_ex_display[col].dropna().tolist())
                                    
                                    scale_factor = 1
                                    scale_label = ""
                                    if values:
                                        mean_fee = sum(values) / len(values)
                                        if mean_fee < 0.001:
                                            scale_factor = 1000
                                            scale_label = "× 1,000"
                                        elif mean_fee < 0.0001:
                                            scale_factor = 10000
                                            scale_label = "× 10,000"
                                        elif mean_fee < 0.00001:
                                            scale_factor = 100000
                                            scale_label = "× 100,000"
                                    
                                    # Apply scaling
                                    if scale_factor > 1:
                                        for col in depth_cols:
                                            if col in altcoin_ex_display.columns:
                                                altcoin_ex_display[col] = altcoin_ex_display[col] * scale_factor
                                        
                                        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                    
                                    # Apply color coding
                                    color_df = altcoin_ex_display.copy()
                                    for col in depth_cols:
                                        if col in color_df.columns:
                                            # Determine thresholds based on column values
                                            col_values = color_df[col].dropna().tolist()
                                            if col_values:
                                                # Dynamic thresholds based on percentiles
                                                q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                                median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                                q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                                thresholds = [q1, median, q3]
                                                
                                                color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                    
                                    # Round values for display
                                    for col in depth_cols:
                                        if col in altcoin_ex_display.columns:
                                            altcoin_ex_display[col] = altcoin_ex_display[col].round(2)
                                    
                                    # Rename pair_name column
                                    color_df = color_df.rename(columns={'pair_name': 'Token'})
                                    
                                    st.markdown(f"##### Altcoin Tokens - {exchange_display_name}")
                                    
                                    # Display the table with HTML formatting
                                    html_table = color_df.to_html(escape=False, index=False)
                                    st.markdown(html_table, unsafe_allow_html=True)
                                    
                                    # Create visualization - line chart showing spread by size
                                    st.markdown(f"##### Spread vs. Size Relationship - {exchange_display_name} (Altcoin Tokens)")
                                    
                                    # Calculate average spread for each size tier
                                    size_averages = {}
                                    for col in depth_cols:
                                        if col in altcoin_ex_display.columns:
                                            size_averages[col] = altcoin_ex_display[col].mean()
                                    
                                    if size_averages:
                                        # Create data frame for plotting
                                        sizes = list(size_averages.keys())
                                        spreads = list(size_averages.values())
                                        
                                        # Create line chart
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Scatter(
                                            x=sizes,
                                            y=spreads,
                                            mode='lines+markers',
                                            line=dict(color='#FF9800', width=3),
                                            marker=dict(size=10, color='#FF9800'),
                                            name='Average Spread'
                                        ))
                                        
                                        # Format the layout
                                        fig.update_layout(
                                            title=f"{exchange_display_name} - Spread vs. Size (Altcoin Tokens) {scale_label}",
                                            xaxis_title="Size Tier",
                                            yaxis_title=f"Average Spread {scale_label}",
                                            height=400,
                                            xaxis=dict(
                                                tickfont=dict(size=14),
                                                tickmode='array',
                                                tickvals=sizes
                                            ),
                                            yaxis=dict(
                                                tickformat='.6f'
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data available for {exchange_display_name}")
                            
                # Cross-Exchange Comparison by Size
                st.markdown("### Cross-Exchange Comparison by Size")
                
                # Group data by token and exchange for comparison
                exchange_comparison = {}
                
                # Process major tokens
                if major_tokens and non_surf_data is not None and not non_surf_data.empty:
                    # Filter for major tokens
                    major_data = non_surf_data[non_surf_data['pair_name'].isin(major_tokens)].copy()
                    
                    if not major_data.empty:
                        # Calculate average for each exchange and fee level
                        for exchange_source in non_surf_exchanges:
                            exchange_display_name = exchanges_display[exchange_source]
                            exchange_comparison[exchange_display_name] = {}
                            
                            # Filter for this exchange
                            exchange_df = major_data[major_data['source'] == exchange_source].copy()
                            
                            if not exchange_df.empty:
                                # Calculate average for each fee level
                                for i, fee_col in enumerate(available_fee_cols):
                                    if fee_col in exchange_df.columns:
                                        values = exchange_df[fee_col].dropna()
                                        if not values.empty:
                                            # Map to correct size label
                                            size_label = fee_depth_map_major.get(fee_col, f'Size {i+1}')
                                            exchange_comparison[exchange_display_name][size_label] = values.mean()
                        
                        # Create a comparison chart if we have data
                        if exchange_comparison:
                            # Prepare data for plotting
                            comparison_data = []
                            for exchange, sizes in exchange_comparison.items():
                                for size, value in sizes.items():
                                    comparison_data.append({
                                        'Exchange': exchange,
                                        'Size': size,
                                        'Spread': value
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                
                                # Determine scale factor
                                values = comparison_df['Spread'].tolist()
                                scale_factor = 1
                                scale_label = ""
                                if values:
                                    mean_fee = sum(values) / len(values)
                                    if mean_fee < 0.001:
                                        scale_factor = 1000
                                        scale_label = "× 1,000"
                                    elif mean_fee < 0.0001:
                                        scale_factor = 10000
                                        scale_label = "× 10,000"
                                    elif mean_fee < 0.00001:
                                        scale_factor = 100000
                                        scale_label = "× 100,000"
                                
                                # Apply scaling
                                if scale_factor > 1:
                                    comparison_df['Spread'] = comparison_df['Spread'] * scale_factor
                                    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                
                                # Create grouped bar chart
                                st.markdown("#### Major Tokens - Exchange Comparison by Size")
                                
                                # Sort by size tiers for better visualization
                                size_order = ['50K', '100K', '200K', '500K']
                                comparison_df['Size'] = pd.Categorical(comparison_df['Size'], categories=size_order, ordered=True)
                                comparison_df = comparison_df.sort_values('Size')
                                
                                fig = px.bar(
                                    comparison_df,
                                    x='Size',
                                    y='Spread',
                                    color='Exchange',
                                    barmode='group',
                                    title=f"Exchange Comparison by Size - Major Tokens {scale_label}",
                                    color_discrete_map={
                                        'Binance': '#2196F3',
                                        'Gate': '#FFC107',
                                        'Hyperliquid': '#FF5722'
                                    }
                                )
                                
                                # Format the layout
                                fig.update_layout(
                                    xaxis_title="Size Tier",
                                    yaxis_title=f"Average Spread {scale_label}",
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
                
                # Process altcoin tokens
                exchange_comparison = {}
                if altcoin_tokens and non_surf_data is not None and not non_surf_data.empty:
                    # Filter for altcoin tokens
                    altcoin_data = non_surf_data[non_surf_data['pair_name'].isin(altcoin_tokens)].copy()
                    
                    if not altcoin_data.empty:
                        # Calculate average for each exchange and fee level
                        for exchange_source in non_surf_exchanges:
                            exchange_display_name = exchanges_display[exchange_source]
                            exchange_comparison[exchange_display_name] = {}
                            
                            # Filter for this exchange
                            exchange_df = altcoin_data[altcoin_data['source'] == exchange_source].copy()
                            
                            if not exchange_df.empty:
                                # Calculate average for each fee level
                                for i, fee_col in enumerate(available_fee_cols):
                                    if fee_col in exchange_df.columns:
                                        values = exchange_df[fee_col].dropna()
                                        if not values.empty:
                                            # Map to correct size label
                                            size_label = fee_depth_map_altcoin.get(fee_col, f'Size {i+1}')
                                            exchange_comparison[exchange_display_name][size_label] = values.mean()
                        
                        # Create a comparison chart if we have data
                        if exchange_comparison:
                            # Prepare data for plotting
                            comparison_data = []
                            for exchange, sizes in exchange_comparison.items():
                                for size, value in sizes.items():
                                    comparison_data.append({
                                        'Exchange': exchange,
                                        'Size': size,
                                        'Spread': value
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                
                                # Determine scale factor
                                values = comparison_df['Spread'].tolist()
                                scale_factor = 1
                                scale_label = ""
                                if values:
                                    mean_fee = sum(values) / len(values)
                                    if mean_fee < 0.001:
                                        scale_factor = 1000
                                        scale_label = "× 1,000"
                                    elif mean_fee < 0.0001:
                                        scale_factor = 10000
                                        scale_label = "× 10,000"
                                    elif mean_fee < 0.00001:
                                        scale_factor = 100000
                                        scale_label = "× 100,000"
                                
                                # Apply scaling
                                if scale_factor > 1:
                                    comparison_df['Spread'] = comparison_df['Spread'] * scale_factor
                                    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                
                                # Create grouped bar chart
                                st.markdown("#### Altcoin Tokens - Exchange Comparison by Size")
                                
                                # Sort by size tiers for better visualization
                                size_order = ['20K', '50K', '100K', '200K']
                                comparison_df['Size'] = pd.Categorical(comparison_df['Size'], categories=size_order, ordered=True)
                                comparison_df = comparison_df.sort_values('Size')
                                
                                fig = px.bar(
                                    comparison_df,
                                    x='Size',
                                    y='Spread',
                                    color='Exchange',
                                    barmode='group',
                                    title=f"Exchange Comparison by Size - Altcoin Tokens {scale_label}",
                                    color_discrete_map={
                                        'Binance': '#2196F3',
                                        'Gate': '#FFC107',
                                        'Hyperliquid': '#FF5722'
                                    }
                                )
                                
                                # Format the layout
                                fig.update_layout(
                                    xaxis_title="Size Tier",
                                    yaxis_title=f"Average Spread {scale_label}",
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
            except Exception as e:
                st.error(f"Error analyzing spread data by size: {e}")
                import traceback
                traceback.print_exc()
    else:
        st.warning("No daily average spread data available for the selected tokens and time period.")

# Footer with explanatory information
with st.expander("Understanding Exchange Spreads"):
    st.markdown("""
    ### About This Dashboard
    
    This dashboard provides comprehensive analysis of trading spreads across multiple cryptocurrency exchanges. 
    
    ### Key Concepts:
    
    - **Spread**: The difference between the buy and sell price, representing the cost of trading.
    
    - **Trading Sizes**: Different order sizes for analysis:
      - **Major tokens** (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K
      - **Altcoin tokens**: 20K, 50K, 100K, 200K
    
    - **Fee Columns**: 
      - `fee1`, `fee2`, `fee3`, `fee4` correspond to spreads at different trading sizes
    
    - **Exchange Comparison**: The dashboard highlights when SurfFuture has tighter spreads (lower fees) than the average of other exchanges.
    
    ### Interpreting the Data:
    
    - Lower spread values indicate better pricing for traders
    - Color coding highlights the relative spread values (blue/green for lower spreads, red/orange for higher spreads)
    - The "Better" percentage shows how much lower SurfFuture's spread is compared to other exchanges (positive values mean SurfFuture is better)
    - The scaling factor is applied to make small decimal values more readable
    
    ### Data Source:
    
    Data is fetched from the `oracle_exchange_fee` table, with 10-minute interval data points that represent the average of previous 10 one-minute points.
    """)

# Execute the app
if __name__ == '__main__':
    pass  # The app is already running