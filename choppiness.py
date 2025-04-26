# Save this as choppiness.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import pytz
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings('ignore')

# Performance monitoring
start_time = time.time()

# Page configuration - optimized for speed
st.set_page_config(
    page_title="Tick Choppiness",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better performance
st.markdown("""
<style>
    .block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;}
    .main .block-container {max-width: 98% !important;}
    h1, h2, h3 {margin-top: 0.25rem !important; margin-bottom: 0.25rem !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 40px; font-size: 16px;}
    div.stProgress > div > div {height: 5px !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Top trading pairs (for quick load)
TOP_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]

# All available pairs
ALL_PAIRS = [
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
            password="866^FKC4hllk",
            connect_timeout=10  # Add connection timeout
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# --- MAIN APP TITLE ---
st.title("Tick-Based Choppiness: Surf vs Rollbit")

# Get Singapore time
sg_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(sg_timezone)

# --- DATABASE UTILITY FUNCTIONS ---
def get_partition_tables(start_date, end_date):
    """Get efficiently list of partition tables"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # Generate list of possible table names
        current_date = start_date
        table_names = []
        
        while current_date <= end_date:
            table_names.append(f"oracle_price_log_partition_{current_date.strftime('%Y%m%d')}")
            current_date += timedelta(days=1)
        
        # Check which tables exist in one query (more efficient)
        table_placeholders = ', '.join(['%s'] * len(table_names))
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ({table_placeholders})
        """, tuple(table_names))
        
        existing_tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        return existing_tables
        
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return []

def build_optimized_query(tables, pair_name, start_time, end_time, exchange, max_rows_per_table=10000):
    """Build an optimized query with row limits"""
    if not tables:
        return ""
        
    source_type = 0 if exchange == 'surf' else 1
    
    # Use UNION ALL (faster than UNION as it skips duplicate checking)
    queries = []
    
    for table in tables:
        # Optimized query with LIMIT
        query = f"""
        (SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
         FROM public.{table}
         WHERE created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
           AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
           AND source_type = {source_type}
           AND pair_name = '{pair_name}'
         ORDER BY created_at
         LIMIT {max_rows_per_table})
        """
        queries.append(query)
    
    full_query = " UNION ALL ".join(queries) + " ORDER BY timestamp"
    return full_query

@st.cache_data(ttl=300)  # 5-minute cache
def fetch_price_data(pair_name, hours=3, exchange='surf', quick_mode=False):
    """Fetch price data with optimized queries and limits"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        # Set query timeout
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = 30000;")  # 30 seconds timeout
        
        # Time range calculation
        end_time = now_sg
        start_time = end_time - timedelta(hours=hours)
        
        # Convert to string format
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get table names (more efficient with dates)
        start_date = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        tables = get_partition_tables(start_date, end_date)
        if not tables:
            return None
        
        # Row limit per table (use smaller limit in quick mode)
        max_rows = 2000 if quick_mode else 10000
        
        # Build and execute optimized query
        query = build_optimized_query(tables, pair_name, start_time_str, end_time_str, exchange, max_rows)
        
        if not query:
            return None
            
        # Execute with timeout
        df = pd.read_sql_query(query, conn)
        
        # Close cursor to free resources
        cursor.close()
        
        if df.empty:
            return None
            
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna().sort_values('timestamp')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching {exchange} data for {pair_name}: {e}")
        return None

def calculate_tick_choppiness(prices):
    """Optimized choppiness calculation with vectorized operations"""
    # Ensure we have enough data
    if len(prices) < 20:
        return None
    
    # Calculate window size (smaller for quicker processing)
    window = min(20, len(prices) // 10)
    
    # Vectorized calculations
    diff = prices.diff().abs()
    sum_abs_changes = diff.rolling(window, min_periods=1).sum()
    
    # Calculate min and max in one pass
    roll = prices.rolling(window, min_periods=1)
    price_range = roll.max() - roll.min()
    
    # Avoid division by zero
    price_range = np.maximum(price_range, 1e-10)
    
    # Calculate and cap choppiness
    choppiness = 100 * sum_abs_changes / price_range
    choppiness = np.minimum(choppiness, 1000)
    
    return float(choppiness.mean())

def process_5min_blocks(df, quick_mode=False):
    """Process data into 5-minute blocks with performance optimizations"""
    if df is None or len(df) < 20:
        return None
    
    # Sample data in quick mode to process fewer points
    if quick_mode and len(df) > 1000:
        sample_size = min(1000, int(len(df) * 0.5))  # Sample at most 1000 points or 50%
        df = df.sample(n=sample_size).sort_values('timestamp')
    
    # Floor timestamps to 5-min intervals
    df['block'] = df['timestamp'].dt.floor('5min')
    
    # Group by 5-minute blocks
    result = []
    
    # Process in chunks - using a more efficient approach
    for name, group in df.groupby('block'):
        if len(group) >= 20:  # Only process if enough data points
            choppiness = calculate_tick_choppiness(group['price'])
            if choppiness is not None:
                result.append({
                    'timestamp': name,
                    'choppiness': choppiness,
                    'count': len(group)
                })
    
    if not result:
        return None
        
    return pd.DataFrame(result).sort_values('timestamp')

# --- OPTIMIZED PARALLEL PROCESSING ---
def fetch_data_parallel(pairs, hours, quick_mode=False):
    """Fetch data in parallel with optimized thread management"""
    results = {'surf': {}, 'rollbit': {}}
    status_text = st.empty()
    
    if not pairs:
        return results
    
    # Show progress without using st.progress (faster)
    status_text.text(f"Fetching data for {len(pairs)} pairs...")
    
    # Prioritize loading to get some results quickly
    processed_pairs = 0
    
    # Use a smaller thread pool to avoid overwhelming the database
    max_workers = 3 if quick_mode else 5
    
    # Process in smaller batches to show progress faster
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for exchange in ['surf', 'rollbit']:
            futures = {}
            
            for pair in pairs:
                # Submit the task
                future = executor.submit(
                    fetch_price_data, 
                    pair, 
                    hours, 
                    exchange,
                    quick_mode
                )
                futures[future] = pair
            
            # Process completed tasks
            for future in as_completed(futures):
                pair = futures[future]
                processed_pairs += 1
                
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[exchange][pair] = data
                    
                    # Update status occasionally (not every pair)
                    if processed_pairs % 3 == 0:
                        status_text.text(f"Processed {processed_pairs}/{len(pairs)*2} items...")
                except Exception as e:
                    st.error(f"Error with {pair}: {e}")
    
    status_text.empty()
    return results

def calculate_choppiness_parallel(pair_data, quick_mode=False):
    """Calculate choppiness in parallel for all pairs"""
    results = {'surf': {}, 'rollbit': {}}
    
    # Use a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        # Submit all tasks
        for exchange in ['surf', 'rollbit']:
            for pair, df in pair_data[exchange].items():
                future = executor.submit(process_5min_blocks, df, quick_mode)
                futures[future] = (exchange, pair)
        
        # Process completed tasks
        for future in as_completed(futures):
            exchange, pair = futures[future]
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    results[exchange][pair] = result_df
            except Exception as e:
                st.error(f"Error processing {exchange} {pair}: {e}")
    
    return results

# --- INITIALIZE SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# --- UI CONTROLS (OPTIMIZED FOR SPEED) ---
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Mode selection for performance
        mode = st.radio(
            "Analysis Mode",
            ["Quick (Top 5)", "Standard", "Custom Selection"],
            horizontal=True,
            help="Quick mode analyzes only top coins with optimized performance."
        )
        
        # Token selection based on mode
        if mode == "Quick (Top 5)":
            selected_tokens = TOP_PAIRS
            quick_mode = True
        elif mode == "Standard":
            selected_tokens = TOP_PAIRS + ["ADA/USDT", "BNB/USDT", "PEPE/USDT"]
            quick_mode = False
        else:  # Custom Selection
            quick_mode = False
            select_all = st.checkbox("Select All", value=False)
            if select_all:
                selected_tokens = ALL_PAIRS
            else:
                selected_tokens = st.multiselect(
                    "Select Tokens",
                    options=ALL_PAIRS,
                    default=TOP_PAIRS
                )
    
    with col2:
        hours = st.selectbox(
            "Hours to Analyze",
            options=[1, 3, 6, 12],
            index=1  # Default to 3 hours for better performance
        )
    
    with col3:
        refresh = st.button("Refresh Data", use_container_width=True)

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Display Singapore time
st.write(f"Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# --- CORE PROCESSING LOGIC ---
if refresh:
    # Start with a clean slate
    st.cache_data.clear()
    
    # Show note about quick mode if enabled
    if quick_mode:
        st.info("⚡ Quick mode enabled: Using data sampling for faster results")
    
    # Load time tracking
    load_start = time.time()
    
    # Fetch data with optimized settings
    with st.spinner("Fetching data..."):
        pair_data = fetch_data_parallel(selected_tokens, hours, quick_mode)
    
    # Check if we got any data
    if not any(pair_data[exchange] for exchange in ['surf', 'rollbit']):
        st.error("No data available. Try selecting different pairs or time range.")
        st.stop()
    
    # Calculate choppiness
    with st.spinner("Calculating choppiness..."):
        results = calculate_choppiness_parallel(pair_data, quick_mode)
    
    # Store in session state
    st.session_state.results = results
    st.session_state.last_update = now_sg
    
    # Show performance stats (only in developer mode)
    load_time = time.time() - load_start
    st.success(f"Data loaded in {load_time:.2f} seconds")

# --- VISUALIZATION ---
if st.session_state.results:
    results = st.session_state.results
    last_update = st.session_state.last_update
    
    # Display last update time
    st.info(f"Last data refresh: {last_update.strftime('%Y-%m-%d %H:%M:%S')} (SGT)")
    
    # Create streamlined tabs
    tabs = st.tabs(["Charts", "Comparison", "Raw Data"])
    
    with tabs[0]:
        # Create a selection for which pairs to display in charts
        if len(selected_tokens) > 3:
            chart_tokens = st.multiselect(
                "Select pairs to display charts for",
                options=selected_tokens,
                default=selected_tokens[:3]  # Default to first 3 for performance
            )
        else:
            chart_tokens = selected_tokens
        
        # Show each selected pair
        for pair in chart_tokens:
            surf_data = results['surf'].get(pair)
            rollbit_data = results['rollbit'].get(pair)
            
            if surf_data is not None or rollbit_data is not None:
                fig = go.Figure()
                
                # Add data traces
                if surf_data is not None:
                    fig.add_trace(go.Scatter(
                        x=surf_data['timestamp'],
                        y=surf_data['choppiness'],
                        mode='lines+markers',
                        name='Surf',
                        line=dict(color='blue', width=2)
                    ))
                
                if rollbit_data is not None:
                    fig.add_trace(go.Scatter(
                        x=rollbit_data['timestamp'],
                        y=rollbit_data['choppiness'],
                        mode='lines+markers',
                        name='Rollbit',
                        line=dict(color='red', width=2)
                    ))
                
                # Simplified layout for better performance
                fig.update_layout(
                    title=f"{pair} - Tick Choppiness",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {pair}")
    
    with tabs[1]:
        # Create comparison data
        comparison = []
        
        # Collect comparison metrics
        for pair in selected_tokens:
            surf = results['surf'].get(pair)
            rollbit = results['rollbit'].get(pair)
            
            row = {'Pair': pair}
            
            if surf is not None and not surf.empty:
                row['Surf Mean'] = surf['choppiness'].mean()
                
            if rollbit is not None and not rollbit.empty:
                row['Rollbit Mean'] = rollbit['choppiness'].mean()
                
            if 'Surf Mean' in row and 'Rollbit Mean' in row:
                row['Difference'] = row['Surf Mean'] - row['Rollbit Mean']
                row['% Diff'] = (row['Difference'] / row['Rollbit Mean'] * 100)
            
            if 'Surf Mean' in row or 'Rollbit Mean' in row:
                comparison.append(row)
        
        if comparison:
            # Create comparison dataframe
            comp_df = pd.DataFrame(comparison)
            
            # Format for display
            for col in comp_df.columns:
                if col != 'Pair' and col in comp_df.columns:
                    comp_df[col] = comp_df[col].round(2)
            
            # Sort by difference if available
            if 'Difference' in comp_df.columns:
                comp_df = comp_df.sort_values('Difference', ascending=False)
            
            # Display the table
            st.dataframe(comp_df, use_container_width=True)
            
            # Create summary chart if we have difference data
            if 'Difference' in comp_df.columns:
                fig = px.bar(
                    comp_df.head(10),  # Only show top 10 for performance
                    x='Pair', 
                    y='Difference',
                    title='Choppiness Difference (Surf - Rollbit)',
                    color='Difference',
                    color_continuous_scale='RdBu_r'
                )
                
                # Add reference line
                fig.add_shape(type="line", x0=-0.5, x1=9.5, y0=0, y1=0,
                              line=dict(color="black", width=1.5, dash="dash"))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No comparison data available")
    
    with tabs[2]:
        # Simplified raw data view
        st.write("Select exchange and pair to view raw data:")
        
        col1, col2 = st.columns(2)
        with col1:
            exchange = st.selectbox("Exchange", ["surf", "rollbit"])
        with col2:
            # Get available pairs for this exchange
            available_pairs = [p for p in selected_tokens if p in results[exchange]]
            pair = st.selectbox("Pair", available_pairs) if available_pairs else None
        
        if pair and pair in results[exchange]:
            df = results[exchange][pair].copy()
            
            # Format for display
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['choppiness'] = df['choppiness'].round(2)
            df['count'] = df['count'].astype(int)
            
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"No data available for selected pair")
else:
    # Initial message
    st.info("Click 'Refresh Data' to load choppiness data")

# --- FOOTER INFO ---
st.markdown("---")
st.markdown("""
### About This Dashboard

**Tick-Based Choppiness Calculation:**
- Analyzes raw tick data in 5-minute blocks
- Choppiness Formula: 100 * (sum of absolute changes) / (price range)
- Higher values = more oscillation, Lower values = more directional movement

**Optimization Tips:**
- Use "Quick Mode" for faster analysis
- Select fewer pairs for better performance
- Lower "Hours to Analyze" for faster loading
- Data refreshes only when requested
""")

# Show elapsed time at the bottom
total_time = time.time() - start_time
st.write(f"Page load time: {total_time:.2f} seconds")