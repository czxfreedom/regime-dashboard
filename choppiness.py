# Save this as pages/07_Choppiness_Analysis.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import pytz
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration - optimized for speed
st.set_page_config(
    page_title="5-Minute Choppiness Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for speed
)

# Custom CSS for better performance and readability
st.markdown("""
<style>
    .block-container {padding-top: 1rem !important; padding-bottom: 1rem !important;}
    .main .block-container {max-width: 98% !important;}
    h1, h2, h3 {margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 46px; font-size: 18px;}
    div.stProgress > div > div {height: 5px !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Improved table styling */
    .dataframe {
        font-size: 16px !important;
        width: 100% !important;
    }
    
    .dataframe th {
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }
    
    .dataframe td {
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Specify the pairs to analyze (prioritized list)
PAIRS_TO_ANALYZE = [
    "PEPE/USDT", "PAXG/USDT", "DOGE/USDT", "BTC/USDT", "EOS/USDT",
    "BNB/USDT", "MERL/USDT", "FHE/USDT", "IP/USDT", "ORCA/USDT",
    "TRUMP/USDT", "LIBRA/USDT", "AI16Z/USDT", "OM/USDT", "TRX/USDT",
    "S/USDT", "PI/USDT", "JUP/USDT", "BABY/USDT", "PARTI/USDT",
    "ADA/USDT", "HYPE/USDT", "VIRTUAL/USDT", "SUI/USDT", "SATS/USDT",
    "XRP/USDT", "ORDI/USDT", "WIF/USDT", "VANA/USDT", "PENGU/USDT",
    "VINE/USDT", "GRIFFAIN/USDT", "MEW/USDT", "POPCAT/USDT", "FARTCOIN/USDT",
    "TON/USDT", "MELANIA/USDT", "SOL/USDT", "PNUT/USDT", "CAKE/USDT",
    "TST/USDT", "ETH/USDT"
]

# --- DB CONNECTION OPTIMIZATION ---
@st.cache_resource
def get_db_connection():
    """Create a cached DB connection to avoid reconnecting repeatedly"""
    try:
        conn = psycopg2.connect(
            host="aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com",
            port=5432,
            database="replication_report",
            user="public_replication",
            password="866^FKC4hllk"
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# --- APP TITLE AND DESCRIPTION ---
st.title("5-Minute Choppiness Analysis")
st.subheader("Last 6 Hours - Real-time Monitoring")

# Get the current time in Singapore timezone
sg_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(sg_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# --- UTILITY FUNCTIONS ---
def get_partition_tables(start_date, end_date):
    """Get list of partition tables that need to be queried based on date range"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # Generate dates between start and end date
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        # Create table names from dates
        table_names = [f"oracle_price_log_partition_{date}" for date in dates]
        
        # Verify which tables actually exist
        existing_tables = []
        for table in table_names:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            if cursor.fetchone()[0]:
                existing_tables.append(table)
        
        cursor.close()
        
        if not existing_tables:
            st.warning(f"No partition tables found for dates: {', '.join(dates)}")
        
        return existing_tables
        
    except Exception as e:
        st.error(f"Error getting partition tables: {e}")
        return []

def build_query_for_partition_tables(tables, pair_name, start_time, end_time):
    """Build an optimized query combining multiple partition tables"""
    if not tables:
        return ""
        
    union_parts = []
    
    for table in tables:
        # Optimized query with proper time zone handling
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp_sg, 
            final_price AS price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{pair_name}'
        """
        
        union_parts.append(query)
    
    # Join with UNION and add ORDER BY at the end
    complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp_sg"
    return complete_query

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_price_data(pair_name, hours=6):
    """Fetch price data for a given pair over the last N hours"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        # Calculate time range in Singapore timezone
        end_time = now_sg
        start_time = end_time - timedelta(hours=hours)
        
        # Convert to UTC for database query
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get tables to query
        start_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        tables = get_partition_tables(start_date, end_date)
        if not tables:
            return None
        
        # Build and execute query
        query = build_query_for_partition_tables(tables, pair_name, start_time_str, end_time_str)
        
        if not query:
            return None
            
        start_time = time.time()
        df = pd.read_sql_query(query, conn)
        query_time = time.time() - start_time
        
        if df.empty:
            return None
            
        # Rename and convert columns
        df = df.rename(columns={'timestamp_sg': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {pair_name}: {e}")
        return None

def calculate_choppiness(price_series, window=20):
    """Calculate choppiness exactly as in the DepthAnalyzer code"""
    try:
        # Calculate absolute price changes
        diff = price_series.diff().abs()
        
        # Sum of absolute changes
        sum_abs_changes = diff.rolling(window, min_periods=1).sum()
        
        # Calculate price range
        price_range = price_series.rolling(window, min_periods=1).max() - \
                      price_series.rolling(window, min_periods=1).min()
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # Calculate choppiness
        choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
        
        # Cap extreme values
        choppiness_values = np.minimum(choppiness_values, 1000)
        
        return choppiness_values
        
    except Exception as e:
        st.error(f"Error calculating choppiness: {e}")
        return None

def calculate_5min_choppiness(df):
    """Calculate choppiness for 5-minute intervals"""
    try:
        if df is None or df.empty:
            return None
            
        # Resample to 5-minute intervals
        resampled = df.resample('5min', on='timestamp').agg({'price': 'ohlc'})
        
        # Flatten the multi-index columns
        resampled.columns = ['open', 'high', 'low', 'close']
        
        # Calculate choppiness on the close prices
        resampled['choppiness'] = calculate_choppiness(resampled['close'], window=12)  # Adjusted window for 5-min data
        
        # Reset index to get timestamp as a column
        resampled = resampled.reset_index()
        
        # Add period label column for easy reference
        resampled['period'] = resampled['timestamp'].dt.strftime('%H:%M')
        
        return resampled
        
    except Exception as e:
        st.error(f"Error calculating 5-minute choppiness: {e}")
        return None

# --- PARALLEL DATA FETCHING ---
def fetch_data_for_multiple_pairs(pairs, hours=6):
    """Fetch data for multiple pairs in parallel for better performance"""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not pairs:
        return results
    
    status_text.text(f"Fetching data for {len(pairs)} pairs...")
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_pair = {executor.submit(fetch_price_data, pair, hours): pair for pair in pairs}
        
        for i, future in enumerate(as_completed(future_to_pair)):
            pair = future_to_pair[future]
            progress = (i + 1) / len(pairs)
            progress_bar.progress(progress, text=f"Processing {pair} ({i+1}/{len(pairs)})")
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[pair] = df
            except Exception as e:
                st.error(f"Error processing {pair}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def calculate_choppiness_for_all_pairs(pair_data):
    """Calculate choppiness for all pairs with data"""
    choppiness_results = {}
    
    for pair, df in pair_data.items():
        choppiness_df = calculate_5min_choppiness(df)
        if choppiness_df is not None and not choppiness_df.empty:
            choppiness_results[pair] = choppiness_df
    
    return choppiness_results

# --- UI CONTROLS ---
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=False)
    
    if select_all:
        selected_tokens = PAIRS_TO_ANALYZE
    else:
        selected_tokens = st.multiselect(
            "Select Tokens", 
            PAIRS_TO_ANALYZE,
            default=PAIRS_TO_ANALYZE[:5] if len(PAIRS_TO_ANALYZE) > 5 else PAIRS_TO_ANALYZE
        )

with col2:
    hours_to_analyze = st.selectbox(
        "Hours to Analyze",
        options=[3, 6, 12, 24],
        index=1  # Default to 6 hours
    )

with col3:
    # Add a refresh button
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# --- DATA PROCESSING ---
# Fetch data for selected pairs
pair_data = fetch_data_for_multiple_pairs(selected_tokens, hours_to_analyze)

if not pair_data:
    st.error("No data available for the selected pairs and time period.")
    st.stop()

# Calculate choppiness
choppiness_results = calculate_choppiness_for_all_pairs(pair_data)

if not choppiness_results:
    st.error("Could not calculate choppiness for any of the selected pairs.")
    st.stop()

# --- VISUALIZATIONS ---
st.subheader(f"Choppiness Analysis (5-min intervals, Last {hours_to_analyze} Hours)")

# Create tabs for different visualization options
viz_tabs = st.tabs(["Line Charts", "Heatmap", "Statistics", "Raw Data"])

# Tab 1: Line Charts
with viz_tabs[0]:
    # Create a line chart for each pair
    for pair, df in choppiness_results.items():
        fig = px.line(
            df, 
            x='timestamp', 
            y='choppiness',
            title=f"{pair} - 5-Minute Choppiness",
            labels={'timestamp': 'Time (Singapore)', 'choppiness': 'Choppiness'},
            color_discrete_sequence=['#1f77b4'],
            height=400
        )
        
        # Add moving average line
        fig.add_scatter(
            x=df['timestamp'],
            y=df['choppiness'].rolling(6).mean(),  # 30-min moving average
            mode='lines',
            name='30-min MA',
            line=dict(color='red', width=2)
        )
        
        # Improve layout
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                tickformat='%H:%M',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Heatmap
with viz_tabs[1]:
    # Prepare data for heatmap
    all_periods = []
    all_pairs = []
    all_values = []
    
    # Collect all periods first to ensure consistent x-axis
    all_timestamp_periods = set()
    for pair, df in choppiness_results.items():
        all_timestamp_periods.update(df['timestamp'].dt.floor('5min'))
    
    # Sort periods chronologically 
    all_timestamp_periods = sorted(all_timestamp_periods)
    
    # Create a mapping of timestamps to display strings
    period_mapping = {ts: ts.strftime('%H:%M') for ts in all_timestamp_periods}
    
    # Collect data for heatmap
    for pair, df in choppiness_results.items():
        for _, row in df.iterrows():
            period_key = row['timestamp'].floor('5min')
            if period_key in period_mapping:
                all_periods.append(period_mapping[period_key])
                all_pairs.append(pair)
                all_values.append(row['choppiness'])
    
    # Create a DataFrame for the heatmap
    heatmap_df = pd.DataFrame({
        'Period': all_periods,
        'Pair': all_pairs,
        'Choppiness': all_values
    })
    
    # Create the heatmap
    if not heatmap_df.empty:
        # Pivot the data for the heatmap
        pivot_df = heatmap_df.pivot_table(
            values='Choppiness', 
            index='Pair', 
            columns='Period', 
            aggfunc='mean'
        )
        
        # Sort the pivot table
        pivot_df = pivot_df.reindex(sorted(pivot_df.index))
        
        # Create the heatmap with Plotly
        fig = px.imshow(
            pivot_df,
            labels=dict(x="5-Min Period", y="Trading Pair", color="Choppiness"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='Viridis',
            height=max(500, len(pivot_df) * 30)  # Dynamic height
        )
        
        fig.update_layout(
            title="Choppiness Heatmap - 5-Minute Intervals",
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(
                title="Choppiness",
                tickfont=dict(size=12),
                title_font=dict(size=14)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for heatmap visualization.")

# Tab 3: Statistics
with viz_tabs[2]:
    # Calculate statistics for each pair
    stats_data = []
    
    for pair, df in choppiness_results.items():
        if not df.empty and 'choppiness' in df.columns:
            stats_data.append({
                'Pair': pair,
                'Mean Choppiness': np.mean(df['choppiness']),
                'Median Choppiness': np.median(df['choppiness']),
                'Min Choppiness': np.min(df['choppiness']),
                'Max Choppiness': np.max(df['choppiness']),
                'Std Dev': np.std(df['choppiness']),
                'Current': df['choppiness'].iloc[-1] if not df.empty else np.nan,
                'Volatility': np.std(df['choppiness']) / np.mean(df['choppiness']) if np.mean(df['choppiness']) > 0 else 0
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Sort by mean choppiness (descending)
        stats_df = stats_df.sort_values('Mean Choppiness', ascending=False)
        
        # Format numeric columns
        for col in stats_df.columns:
            if col != 'Pair':
                stats_df[col] = stats_df[col].round(2)
        
        # Display the dataframe
        st.dataframe(stats_df, use_container_width=True)
        
        # Create a bar chart of average choppiness
        fig = px.bar(
            stats_df, 
            x='Pair', 
            y='Mean Choppiness',
            title='Average 5-Minute Choppiness by Trading Pair',
            color='Mean Choppiness',
            color_continuous_scale='Viridis',
            height=500
        )
        
        fig.update_layout(
            xaxis_title='Trading Pair',
            yaxis_title='Average Choppiness',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a scatter plot of mean vs volatility
        fig = px.scatter(
            stats_df,
            x='Mean Choppiness',
            y='Volatility',
            title='Choppiness Mean vs Volatility',
            color='Mean Choppiness',
            size='Std Dev',
            hover_name='Pair',
            color_continuous_scale='Viridis',
            height=500
        )
        
        fig.update_layout(
            xaxis_title='Mean Choppiness',
            yaxis_title='Volatility (Std/Mean)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No statistical data available.")

# Tab 4: Raw Data
with viz_tabs[3]:
    # Select a pair to view raw data
    pair_to_view = st.selectbox("Select Pair for Raw Data", list(choppiness_results.keys()))
    
    if pair_to_view in choppiness_results:
        raw_df = choppiness_results[pair_to_view].copy()
        
        # Format columns for display
        raw_df['timestamp'] = raw_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        for col in ['open', 'high', 'low', 'close', 'choppiness']:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].round(6)
        
        # Display the dataframe
        st.dataframe(raw_df, use_container_width=True)
    else:
        st.warning("No raw data available for the selected pair.")

# --- DASHBOARD FOOTER ---
st.markdown("---")
st.markdown("""
### About This Dashboard

This dashboard shows 5-minute choppiness data for selected trading pairs over the specified time period (default: 6 hours).

**Choppiness Calculation:**
- Choppiness measures how much price oscillates within a range
- Higher values indicate more chaotic price action (more changes relative to the overall range)
- Lower values indicate more directional price movement

**Data Refresh:**
- Data is cached for 5 minutes to optimize performance
- Click the "Refresh Data" button to get the latest data
""")

# Performance optimization note
st.sidebar.markdown("### Performance Note")
st.sidebar.info(
    "This dashboard is optimized for fast loading times through:\n"
    "1. Parallel data fetching\n"
    "2. Smart data caching\n"
    "3. Optimized database queries\n"
    "4. Minimal UI elements\n\n"
    "If you're still experiencing performance issues, try reducing the number of selected pairs."
)