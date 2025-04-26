# Save this as choppiness.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pytz
import warnings
import time
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings('ignore')

# Performance monitoring
start_time = time.time()

# Page configuration
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

# --- DB CONNECTION ---
# 数据库配置
DB_URL = "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report"

@st.cache_resource
def get_engine():
    """创建数据库引擎"""
    try:
        return create_engine(DB_URL)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

@contextmanager
def get_session():
    """数据库会话上下文管理器"""
    engine = get_engine()
    if not engine:
        yield None
        return
        
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Database error: {e}")
        yield None
    finally:
        session.close()

# Get available pairs from the replication database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    """从数据库获取可用的交易对"""
    try:
        with get_session() as session:
            if not session:
                return TOP_PAIRS  # Fallback to top pairs
            
            query = text("SELECT pair_name FROM trade_pool_pairs WHERE status = 1")
            result = session.execute(query)
            pairs = [row[0] for row in result]
            
            # Return sorted pairs or fallback to predefined list if empty
            return sorted(pairs) if pairs else TOP_PAIRS
    
    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return TOP_PAIRS  # Fallback to top pairs

# --- APP TITLE ---
st.title("Tick-Based Choppiness: Surf vs Rollbit")

# Get Singapore time
sg_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(sg_timezone)

# --- DIRECT DATA QUERY ---
def fetch_data_simple(pair, hours=3, exchange='surf'):
    """Fetch data with a simple, direct query approach"""
    try:
        # Calculate time range
        end_time = now_sg
        start_time = end_time - timedelta(hours=hours)
        
        # Convert to UTC for database query (8 hour offset)
        end_time_utc = end_time - timedelta(hours=8)
        start_time_utc = start_time - timedelta(hours=8)
        
        end_time_str = end_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        start_time_str = start_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        
        # Use date parts to identify potential partition tables
        start_date = start_time_utc.strftime("%Y%m%d")
        end_date = end_time_utc.strftime("%Y%m%d")
        
        # Identify date range
        date_range = []
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end_date_dt = datetime.strptime(end_date, "%Y%m%d")
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime("%Y%m%d")
            date_range.append(date_str)
            current_date += timedelta(days=1)

        with get_session() as session:
            if not session:
                return None
            
            # Find available tables
            available_tables = []
            for date_str in date_range:
                table_name = f"oracle_price_log_partition_{date_str}"
                result = session.execute(
                    text("SELECT to_regclass(:table_name)"),
                    {"table_name": f"public.{table_name}"}
                )
                if result.scalar() is not None:
                    available_tables.append(table_name)
            
            if not available_tables:
                st.warning(f"No tables found for dates between {start_date} and {end_date}")
                return None
                
            # Debug info
            st.write(f"Using tables: {', '.join(available_tables)}")
            
            # Select source type based on exchange
            source_type = 0 if exchange == 'surf' else 1
            
            # Create queries for each table
            all_data = []
            for table in available_tables:
                query = text(f"""
                SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
                FROM public."{table}"
                WHERE created_at >= :start_time
                  AND created_at <= :end_time
                  AND source_type = :source_type
                  AND pair_name = :pair_name
                ORDER BY created_at
                LIMIT 10000
                """)
                
                df = pd.read_sql_query(
                    query,
                    session.connection(),
                    params={
                        "start_time": start_time_str,
                        "end_time": end_time_str,
                        "source_type": source_type,
                        "pair_name": pair
                    }
                )
                all_data.append(df)
        
        # Process data outside of database connection
        if not all_data:
            return None
            
        combined_df = pd.concat(all_data)
        
        # Clean up data
        if combined_df.empty:
            return None
            
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')
        combined_df = combined_df.dropna().sort_values('timestamp')
        
        return combined_df
        
    except Exception as e:
        st.error(f"Error fetching {exchange} data for {pair}: {e}")
        import traceback
        st.write(traceback.format_exc())
        return None

def calculate_choppiness(prices):
    """Calculate choppiness"""
    # Ensure we have enough data
    if len(prices) < 20:
        return None
    
    # Basic window size
    window = min(20, len(prices) // 10)
    
    # Calculate diff and rolling values
    diff = prices.diff().abs()
    sum_abs_changes = diff.rolling(window, min_periods=1).sum()
    
    # Calculate range in one pass
    price_max = prices.rolling(window, min_periods=1).max()
    price_min = prices.rolling(window, min_periods=1).min()
    price_range = price_max - price_min
    
    # Avoid division by zero
    epsilon = 1e-10
    choppiness = 100 * sum_abs_changes / (price_range + epsilon)
    
    # Cap extreme values
    choppiness = np.minimum(choppiness, 1000)
    
    return float(choppiness.mean())

def process_5min_blocks(df):
    """Process data into 5-minute blocks"""
    if df is None or len(df) < 20:
        return None
    
    # Floor timestamps to 5-min intervals
    df['block'] = df['timestamp'].dt.floor('5min')
    
    # Group by 5-minute blocks
    result = []
    
    for name, group in df.groupby('block'):
        if len(group) >= 20:  # Only process if enough data points
            choppiness = calculate_choppiness(group['price'])
            if choppiness is not None:
                result.append({
                    'timestamp': name,
                    'choppiness': choppiness,
                    'count': len(group)
                })
    
    if not result:
        return None
        
    return pd.DataFrame(result).sort_values('timestamp')

# --- INITIALIZE SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# --- UI CONTROLS ---
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Mode selection for performance
        mode = st.radio(
            "Analysis Mode",
            ["Quick (Top 5)", "Standard", "Custom Selection"],
            horizontal=True,
            index=0  # Default to quick mode
        )
        
        # Token selection based on mode
        if mode == "Quick (Top 5)":
            selected_tokens = TOP_PAIRS
        elif mode == "Standard":
            selected_tokens = TOP_PAIRS + ["ADA/USDT", "BNB/USDT", "PEPE/USDT"]
        else:  # Custom Selection
            select_all = st.checkbox("Select All", value=False)
            if select_all:
                selected_tokens = get_available_pairs()
            else:
                selected_tokens = st.multiselect(
                    "Select Tokens",
                    options=get_available_pairs(),
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
    st.session_state.results = None
    
    # Load time tracking
    load_start = time.time()
    
    # Use simpler approach
    results = {'surf': {}, 'rollbit': {}}
    success = False
    
    # Process each pair
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    total_tasks = len(selected_tokens) * 2  # Each pair for two exchanges
    completed = 0
    
    for pair in selected_tokens:
        for exchange in ['surf', 'rollbit']:
            progress_text.text(f"Processing {exchange} {pair}...")
            
            # Fetch data
            df = fetch_data_simple(pair, hours, exchange)
            
            if df is not None and not df.empty:
                # Calculate choppiness
                result_df = process_5min_blocks(df)
                if result_df is not None and not result_df.empty:
                    results[exchange][pair] = result_df
                    success = True
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / total_tasks)
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    if not success:
        st.error("No data available. Try selecting different pairs or time range.")
    else:
        # Store in session state
        st.session_state.results = results
        st.session_state.last_update = now_sg
        
        # Show performance stats
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

**Performance Tips:**
- Use "Quick Mode" for faster analysis
- Select fewer pairs for better performance
- Lower "Hours to Analyze" for faster loading
- Data refreshes only when requested
""")
