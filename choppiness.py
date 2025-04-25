# Save this as pages/07_Tick_Choppiness_Analysis.py in your Streamlit app folder

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
    page_title="Tick-Based Choppiness Analysis",
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

# Constant for number of ticks in a 5-min period (500ms per tick * 600 = 5 min)
TICKS_PER_5MIN = 600

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
st.title("Tick-Based 5-Minute Choppiness Analysis: Surf vs Rollbit")
st.subheader("Market Microstructure Analysis")

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

def build_query_for_partition_tables(tables, pair_name, start_time, end_time, exchange):
    """Build an optimized query combining multiple partition tables"""
    if not tables:
        return ""
        
    union_parts = []
    
    for table in tables:
        # Choose source type based on exchange
        source_type = 0 if exchange == 'surf' else 1  # surf=0, rollbit=1
        
        # Optimized query with proper time zone handling
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp, 
            final_price AS price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
            AND source_type = {source_type}
            AND pair_name = '{pair_name}'
        """
        
        union_parts.append(query)
    
    # Join with UNION and add ORDER BY at the end
    complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
    return complete_query

@st.cache_data(ttl=600)  # Cache for 10 minutes, only refresh when explicitly requested
def fetch_price_data(pair_name, hours=6, exchange='surf'):
    """Fetch price data for a given pair and exchange over the last N hours"""
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
        query = build_query_for_partition_tables(tables, pair_name, start_time_str, end_time_str, exchange)
        
        if not query:
            return None
            
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return None
            
        # Convert columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching {exchange} data for {pair_name}: {e}")
        return None

def calculate_tick_choppiness(price_series, window=20):
    """
    Calculate choppiness using raw tick data, identical to the depth analyzer method.
    This measures microstructure choppiness.
    
    Args:
        price_series: Series of price ticks
        window: Window size for the rolling calculation
        
    Returns:
        Choppiness value (single float)
    """
    try:
        if len(price_series) < window:
            return None
            
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
        
        # Return the mean choppiness
        return choppiness_values.mean()
        
    except Exception as e:
        print(f"Error calculating choppiness: {e}")
        return None

def process_tick_data_into_5min_blocks(df):
    """
    Process raw tick data into 5-minute blocks and calculate choppiness for each block.
    Uses the same window approach as DepthAnalyzer on each 5-minute chunk.
    
    Args:
        df: DataFrame with timestamp and price columns
        
    Returns:
        DataFrame with 5-minute blocks and their choppiness values
    """
    if df is None or df.empty:
        return None
    
    # Round down timestamps to 5-minute intervals to create blocks
    df['block_time'] = df['timestamp'].dt.floor('5min')
    
    # Group data by 5-minute blocks
    groups = df.groupby('block_time')
    
    # Calculate choppiness for each 5-minute block
    result_data = []
    
    for block_time, group in groups:
        # Get price data for this block
        prices = group['price']
        
        # Only process if we have enough data points
        if len(prices) >= 20:  # Minimum needed for meaningful choppiness calculation
            # Calculate window size (same approach as DepthAnalyzer)
            window = min(20, len(prices) // 10)
            
            # Calculate choppiness
            choppiness = calculate_tick_choppiness(prices, window)
            
            # Store result
            result_data.append({
                'timestamp': block_time,
                'choppiness': choppiness,
                'tick_count': len(prices),
                'period': block_time.strftime('%H:%M')
            })
    
    if not result_data:
        return None
        
    # Create DataFrame from results
    result_df = pd.DataFrame(result_data)
    
    # Sort by timestamp
    result_df = result_df.sort_values('timestamp')
    
    return result_df

# --- PARALLEL DATA FETCHING ---
def fetch_data_for_multiple_pairs(pairs, hours=6):
    """Fetch data for multiple pairs and both exchanges in parallel for better performance"""
    results = {'surf': {}, 'rollbit': {}}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not pairs:
        return results
    
    status_text.text(f"Fetching data for {len(pairs)} pairs from Surf and Rollbit...")
    
    # Create tasks list - one for each pair and exchange combination
    tasks = []
    for pair in pairs:
        tasks.append(('surf', pair))
        tasks.append(('rollbit', pair))
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(fetch_price_data, pair, hours, exchange): (exchange, pair) 
                          for exchange, pair in tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            exchange, pair = future_to_task[future]
            progress = (i + 1) / len(tasks)
            progress_bar.progress(progress, text=f"Processing {exchange.upper()}: {pair} ({i+1}/{len(tasks)})")
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[exchange][pair] = df
            except Exception as e:
                st.error(f"Error processing {exchange.upper()}: {pair}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def calculate_choppiness_for_all_pairs(pair_data):
    """Calculate tick-based 5-minute choppiness for all pairs with data"""
    choppiness_results = {'surf': {}, 'rollbit': {}}
    
    for exchange in ['surf', 'rollbit']:
        for pair, df in pair_data[exchange].items():
            choppiness_df = process_tick_data_into_5min_blocks(df)
            if choppiness_df is not None and not choppiness_df.empty:
                choppiness_results[exchange][pair] = choppiness_df
    
    return choppiness_results

# --- INITIALIZE SESSION STATE FOR RESULTS ---
if 'choppiness_results' not in st.session_state:
    st.session_state.choppiness_results = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

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
    # Add a refresh button - ONLY refresh when this is clicked
    refresh_pressed = st.button("Refresh Data", use_container_width=True)

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# --- DATA PROCESSING - ONLY WHEN REFRESH IS CLICKED ---
if refresh_pressed:
    # Clear the cache to force a refresh
    st.cache_data.clear()
    
    # Fetch data for selected pairs
    pair_data = fetch_data_for_multiple_pairs(selected_tokens, hours_to_analyze)
    
    if not pair_data['surf'] and not pair_data['rollbit']:
        st.error("No data available for the selected pairs and time period.")
        st.stop()
    
    # Calculate choppiness
    choppiness_results = calculate_choppiness_for_all_pairs(pair_data)
    
    # Store in session state
    st.session_state.choppiness_results = choppiness_results
    st.session_state.last_update_time = now_sg
    
    # Force a rerun to display the updated data
    st.experimental_rerun()

# --- VISUALIZATIONS ---
if st.session_state.choppiness_results:
    choppiness_results = st.session_state.choppiness_results
    last_update_time = st.session_state.last_update_time
    
    # Show last update time
    st.info(f"Last data refresh: {last_update_time.strftime('%Y-%m-%d %H:%M:%S')} (SGT)")
    
    # Check if we have any data to display
    if not any(choppiness_results[exchange] for exchange in ['surf', 'rollbit']):
        st.error("Could not calculate choppiness for any of the selected pairs.")
        st.stop()

    # Create tabs for different visualization options
    viz_tabs = st.tabs(["Line Charts", "Heatmap", "Statistics", "Raw Data"])

    # Tab 1: Line Charts
    with viz_tabs[0]:
        # Create a line chart for each pair, comparing Surf and Rollbit
        for pair in selected_tokens:
            # Check if we have data for this pair in either exchange
            surf_data = choppiness_results['surf'].get(pair)
            rollbit_data = choppiness_results['rollbit'].get(pair)
            
            if surf_data is not None or rollbit_data is not None:
                # Create a Plotly figure
                fig = go.Figure()
                
                # Add Surf data if available
                if surf_data is not None:
                    fig.add_trace(go.Scatter(
                        x=surf_data['timestamp'],
                        y=surf_data['choppiness'],
                        mode='lines+markers',
                        name='Surf',
                        line=dict(color='blue', width=2),
                        hovertemplate='%{x}<br>Choppiness: %{y:.2f}<br>Ticks: %{text}',
                        text=surf_data['tick_count']
                    ))
                
                # Add Rollbit data if available
                if rollbit_data is not None:
                    fig.add_trace(go.Scatter(
                        x=rollbit_data['timestamp'],
                        y=rollbit_data['choppiness'],
                        mode='lines+markers',
                        name='Rollbit',
                        line=dict(color='red', width=2),
                        hovertemplate='%{x}<br>Choppiness: %{y:.2f}<br>Ticks: %{text}',
                        text=rollbit_data['tick_count']
                    ))
                
                # Add a horizontal reference line at 100 (moderate choppiness level)
                fig.add_shape(
                    type="line",
                    x0=surf_data['timestamp'].min() if surf_data is not None else rollbit_data['timestamp'].min(),
                    x1=surf_data['timestamp'].max() if surf_data is not None else rollbit_data['timestamp'].max(),
                    y0=100,
                    y1=100,
                    line=dict(
                        color="gray",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Improve layout
                fig.update_layout(
                    title=f"{pair} - 5-Minute Tick-Based Choppiness Comparison",
                    xaxis_title="Time (Singapore)",
                    yaxis_title="Choppiness",
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
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {pair}")

    # Tab 2: Heatmap
    with viz_tabs[1]:
        # Create separate tabs for Surf and Rollbit heatmaps
        heatmap_tabs = st.tabs(["Surf Heatmap", "Rollbit Heatmap"])
        
        for idx, exchange in enumerate(['surf', 'rollbit']):
            with heatmap_tabs[idx]:
                # Prepare data for heatmap
                all_periods = []
                all_pairs = []
                all_values = []
                
                # Check if we have any data for this exchange
                if not choppiness_results[exchange]:
                    st.warning(f"No data available for {exchange.upper()}")
                    continue
                
                # Collect all periods first to ensure consistent x-axis
                all_timestamp_periods = set()
                for pair, df in choppiness_results[exchange].items():
                    all_timestamp_periods.update(df['timestamp'])
                
                # Sort periods chronologically 
                all_timestamp_periods = sorted(all_timestamp_periods)
                
                # Create a mapping of timestamps to display strings
                period_mapping = {ts: ts.strftime('%H:%M') for ts in all_timestamp_periods}
                
                # Collect data for heatmap
                for pair, df in choppiness_results[exchange].items():
                    for _, row in df.iterrows():
                        period_key = row['timestamp']
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
                        title=f"{exchange.upper()} Tick-Based Choppiness Heatmap - 5-Minute Intervals",
                        margin=dict(l=10, r=10, t=40, b=10),
                        coloraxis_colorbar=dict(
                            title="Choppiness",
                            tickfont=dict(size=12),
                            title_font=dict(size=14)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for {exchange.upper()} heatmap visualization.")

    # Tab 3: Statistics
    with viz_tabs[2]:
        # Create tabs for Surf and Rollbit statistics
        stats_tabs = st.tabs(["Surf Statistics", "Rollbit Statistics", "Comparison"])
        
        # Prepare comparison data
        comparison_data = []
        
        for exchange_idx, exchange in enumerate(['surf', 'rollbit']):
            with stats_tabs[exchange_idx]:
                # Calculate statistics for each pair
                stats_data = []
                
                for pair in selected_tokens:
                    if pair in choppiness_results[exchange]:
                        df = choppiness_results[exchange][pair]
                        if not df.empty and 'choppiness' in df.columns:
                            # Store data for comparison
                            mean_chop = np.mean(df['choppiness'])
                            
                            # Add to comparison data
                            comparison_item = next((item for item in comparison_data if item['Pair'] == pair), None)
                            if comparison_item:
                                comparison_item[f'{exchange.capitalize()} Mean'] = mean_chop
                                comparison_item[f'{exchange.capitalize()} Ticks'] = df['tick_count'].mean()
                            else:
                                comparison_data.append({
                                    'Pair': pair,
                                    f'{exchange.capitalize()} Mean': mean_chop,
                                    f'{exchange.capitalize()} Ticks': df['tick_count'].mean()
                                })
                            
                            # Add to statistics
                            stats_data.append({
                                'Pair': pair,
                                'Mean Choppiness': mean_chop,
                                'Median Choppiness': np.median(df['choppiness']),
                                'Min Choppiness': np.min(df['choppiness']),
                                'Max Choppiness': np.max(df['choppiness']),
                                'Std Dev': np.std(df['choppiness']),
                                'Current': df['choppiness'].iloc[-1] if not df.empty else np.nan,
                                'Avg Ticks/5min': df['tick_count'].mean(),
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
                        title=f'{exchange.upper()} Average 5-Minute Tick-Based Choppiness',
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
                else:
                    st.warning(f"No statistical data available for {exchange.upper()}.")
        
        # Comparison tab
        with stats_tabs[2]:
            if comparison_data:
                # Create comparison DataFrame
                comp_df = pd.DataFrame(comparison_data)
                
                # Add difference column if both exchanges have data
                if 'Surf Mean' in comp_df.columns and 'Rollbit Mean' in comp_df.columns:
                    comp_df['Difference'] = comp_df['Surf Mean'] - comp_df['Rollbit Mean']
                    comp_df['Pct Difference'] = (comp_df['Difference'] / comp_df['Rollbit Mean'] * 100).round(2)
                    
                    # Sort by absolute difference
                    comp_df = comp_df.sort_values('Difference', ascending=False)
                    
                    # Format numeric columns
                    for col in comp_df.columns:
                        if col != 'Pair' and 'Ticks' not in col:
                            comp_df[col] = comp_df[col].round(2)
                        elif 'Ticks' in col:
                            comp_df[col] = comp_df[col].round(0)
                    
                    # Display the dataframe
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Create a bar chart showing the differences
                    fig = px.bar(
                        comp_df, 
                        x='Pair', 
                        y='Difference',
                        title='Choppiness Difference (Surf - Rollbit)',
                        color='Difference',
                        color_continuous_scale='RdBu_r',  # Red for negative, Blue for positive
                        height=500
                    )
                    
                    # Add a zero line
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=0,
                        x1=len(comp_df) - 0.5,
                        y1=0,
                        line=dict(
                            color="black",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    fig.update_layout(
                        xaxis_title='Trading Pair',
                        yaxis_title='Choppiness Difference',
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data to compare exchanges.")
            else:
                st.warning("No comparison data available.")

    # Tab 4: Raw Data
    with viz_tabs[3]:
        # Create tabs for Surf and Rollbit raw data
        raw_tabs = st.tabs(["Surf Raw Data", "Rollbit Raw Data"])
        
        for idx, exchange in enumerate(['surf', 'rollbit']):
            with raw_tabs[idx]:
                if not choppiness_results[exchange]:
                    st.warning(f"No raw data available for {exchange.upper()}.")
                    continue
                    
                # Select a pair to view raw data
                available_pairs = list(choppiness_results[exchange].keys())
                if not available_pairs:
                    st.warning(f"No data available for {exchange.upper()}.")
                    continue
                    
                pair_to_view = st.selectbox(f"Select Pair for {exchange.upper()} Raw Data", 
                                           available_pairs, key=f"raw_{exchange}")
                
                if pair_to_view in choppiness_results[exchange]:
                    raw_df = choppiness_results[exchange][pair_to_view].copy()
                    
                    # Format columns for display
                    raw_df['timestamp'] = raw_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    for col in ['choppiness', 'tick_count']:
                        if col in raw_df.columns:
                            if col == 'choppiness':
                                raw_df[col] = raw_df[col].round(2)
                            else:
                                raw_df[col] = raw_df[col].round(0).astype(int)
                    
                    # Display the dataframe
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.warning(f"No raw data available for {pair_to_view} on {exchange.upper()}.")

else:
    # Initial message when no data has been loaded yet
    st.info("Click the 'Refresh Data' button to load tick-based choppiness data for the selected tokens.")

# --- DASHBOARD FOOTER ---
st.markdown("---")
st.markdown("""
### About This Dashboard

This dashboard analyzes tick-by-tick data in 5-minute blocks to calculate market microstructure choppiness for Surf and Rollbit exchanges.

**Tick-Based Choppiness Calculation:**
- Analyzes approximately 600 ticks per 5-minute period (each tick is ~500ms data)
- Calculates choppiness using the formula: 100 * (sum of absolute changes) / (price range)
- Higher values indicate more oscillation relative to the overall range
- Lower values indicate more directional movement

**Key Implementation Details:**
- Uses identical calculation method as the Depth Analyzer
- Processes raw tick data without resampling
- Window size proportional to number of available ticks
- Shows true microstructure choppiness

**Data Refresh:**
- Data is only refreshed when you click the "Refresh Data" button
- No automatic refresh when switching between tabs or pages
""")

# Performance optimization note
st.sidebar.markdown("### Performance Note")
st.sidebar.info(
    "This dashboard is optimized for fast loading times through:\n"
    "1. Parallel data fetching\n"
    "2. Only refreshing when explicitly requested\n"
    "3. Smart data caching\n"
    "4. Optimized database queries\n\n"
    "If you're still experiencing performance issues, try reducing the number of selected pairs."
)