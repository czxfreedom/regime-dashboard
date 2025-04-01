# Save this as pages/04_Daily_Hurst_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Daily Hurst Table",
    page_icon="📊",
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
st.title("Daily Hurst Table (30min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
rolling_window = 20  # Window size for Hurst calculation
expected_points = 48  # Expected data points per pair over 24 hours
singapore_timezone = pytz.timezone('Asia/Singapore')

# Fetch all available tokens from DB
@st.cache_data(show_spinner="Fetching tokens...")
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback

all_tokens = fetch_all_tokens()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=True)
    
    if select_all:
        selected_tokens = all_tokens
    else:
        selected_tokens = st.multiselect(
            "Select Tokens", 
            all_tokens,
            default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Universal Hurst calculation function
def universal_hurst(ts):
    # ... (rest of your universal_hurst function remains the same)

# Detailed regime classification function
def detailed_regime_classification(hurst):
    if pd.isna(hurst):
        return ("UNKNOWN", 0, "Insufficient data")
    elif hurst < 0.2:
        return ("MEAN-REVERT", 3, "Strong mean-reversion")
    elif hurst < 0.3:
        return ("MEAN-REVERT", 2, "Moderate mean-reversion")
    elif hurst < 0.4:
        return ("MEAN-REVERT", 1, "Mild mean-reversion")
    elif hurst < 0.45:
        return ("NOISE", 1, "Slight mean-reversion bias")
    elif hurst <= 0.55:
        return ("NOISE", 0, "Pure random walk")
    elif hurst < 0.6:
        return ("NOISE", 1, "Slight trending bias")
    elif hurst < 0.7:
        return ("TREND", 1, "Mild trending")
    elif hurst < 0.8:
        return ("TREND", 2, "Moderate trending")
    else:
        return ("TREND", 3, "Strong trending")

# Function to convert time string to sortable minutes value
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Fetch and calculate Hurst for a token with 30min timeframe
@st.cache_data(ttl=600, show_spinner="Calculating Hurst exponents...")
def fetch_and_calculate_hurst(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Calculate start and end times, rounded to the nearest 30-minute interval
    end_time_sg = now_sg.replace(minute=now_sg.minute // 30 * 30, second=0, microsecond=0)
    start_time_sg = end_time_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = end_time_sg.astimezone(pytz.utc)

    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price, 
        pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_name = '{token}';
    """
    try:
        print(f"[{token}] Executing query: {query}")
        df = pd.read_sql(query, engine)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No data found.")
            return None

        print(f"[{token}] First few rows:\n{df.head()}")
        print(f"[{token}] DataFrame columns and types:\n{df.info()}")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            print(f"[{token}] No OHLC data after resampling.")
            return None
                
        print(f"[{token}] one_min_ohlc head:\n{one_min_ohlc.head()}")
        print(f"[{token}] one_min_ohlc info:\n{one_min_ohlc.info()}")

        # Apply universal_hurst to the 'close' prices directly
        one_min_ohlc['Hurst'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(universal_hurst)

        thirty_min_hurst = one_min_ohlc['Hurst'].resample('30min').mean().dropna()
        if thirty_min_hurst.empty:
            print(f"[{token}] No 30-min Hurst data.")
            return None
            
        # Generate expected 30-minute intervals for the last 24 hours
        expected_intervals = [end_time_sg - timedelta(minutes=30 * i) for i in range(expected_points)]
        expected_intervals.reverse()  # Reverse to start from the oldest

        # Reindex the Hurst data to match the expected intervals
        last_24h_hurst = thirty_min_hurst.reindex(expected_intervals)
        last_24h_hurst = last_24h_hurst.to_frame()
        
        # Store original datetime index for sorting
        last_24h_hurst['original_datetime'] = last_24h_hurst.index
        last_24h_hurst['time_label'] = last_24h_hurst.index.strftime('%H:%M')
        last_24h_hurst['regime_info'] = last_24h_hurst['Hurst'].apply(detailed_regime_classification)
        last_24h_hurst['regime'] = last_24h_hurst['regime_info'].apply(lambda x: x[0])
        last_24h_hurst['regime_desc'] = last_24h_hurst['regime_info'].apply(lambda x: x[2])
        print(f"[{token}] Successful Calculation")
        return last_24h_hurst
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate Hurst for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:  # Added try-except around token processing
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_hurst(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    # Get all datetimes from all tokens
    combined_datetime_df = pd.DataFrame()
    for token, df in token_results.items():
        if 'original_datetime' in df.columns:
            token_dt = df[['original_datetime', 'time_label']].copy()
            token_dt['token'] = token
            combined_datetime_df = pd.concat([combined_datetime_df, token_dt])
    
    # Group by time_label and find the latest datetime for each time slot 
    # (in case different tokens have slightly different timestamps)
    time_mapping = combined_datetime_df.groupby('time_label')['original_datetime'].max()
    
    # Now create the hurst table using time_labels
    all_times = sorted(time_mapping.index, key=time_to_minutes, reverse=True)
    
    table_data = {}
    for token, df in token_results.items():
        hurst_series = df.set_index('time_label')['Hurst']
        table_data[token] = hurst_series
    
    hurst_table = pd.DataFrame(table_data)
    # Use the sorted time labels
    hurst_table = hurst_table.reindex(all_times)
    hurst_table = hurst_table.round(2)
    
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;' # Grey for missing
        elif val < 0.4:
            intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
            return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
        elif val > 0.6:
            intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        else:
            return 'background-color: rgba(200, 200, 200, 0.5); color: black' # Lighter gray
    
    styled_table = hurst_table.style.applymap(color_cells)
    st.markdown("## Hurst Exponent Table (30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:red'>Mean Reversion</span>, <span style='color:gray'>Random Walk</span>, <span style='color:green'>Trending</span>", unsafe_allow_html=True)
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    st.subheader("Current Market Overview (Singapore Time)")
    latest_values = {}
    for token, df in token_results.items():
        if not df.empty and not df['Hurst'].isna().all():
            latest = df['Hurst'].iloc[-1]
            regime = df['regime_desc'].iloc[-1]
            latest_values[token] = (latest, regime)
    
    if latest_values:
        mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
        random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
        trending = sum(1 for v, r in latest_values.values() if v > 0.6)
        total = mean_reverting + random_walk + trending
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean-Reverting", f"{mean_reverting} ({mean_reverting/total*100:.1f}%)", delta=f"{mean_reverting/total*100:.1f}%")
        col2.metric("Random Walk", f"{random_walk} ({random_walk/total*100:.1f}%)", delta=f"{random_walk/total*100:.1f}%")
        col3.metric("Trending", f"{trending} ({trending/total*100:.1f}%)", delta=f"{trending/total*100:.1f}%")
        
        labels = ['Mean-Reverting', 'Random Walk', 'Trending']
        values = [mean_reverting, random_walk, trending]
        colors = ['rgba(255,100,100,0.8)', 'rgba(200,200,200,0.8)', 'rgba(100,255,100,0.8)'] # Slightly more opaque
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)]) # Added black borders
        fig.update_layout(
            title="Current Market Regime Distribution (Singapore Time)",
            height=400,
            font=dict(color="#000000", size=12),  # Set default font color and size
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Mean-Reverting Tokens")
            mr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v < 0.4]
            mr_tokens.sort(key=lambda x: x[1])
            if mr_tokens:
                for token, value, regime in mr_tokens:
                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Random Walk Tokens")
            rw_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if 0.4 <= v <= 0.6]
            rw_tokens.sort(key=lambda x: x[1])
            if rw_tokens:
                for token, value, regime in rw_tokens:
                    st.markdown(f"- **{token}**: <span style='color:gray'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### Trending Tokens")
            tr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v > 0.6]
            tr_tokens.sort(key=lambda x: x[1], reverse=True)
            if tr_tokens:
                for token, value, regime in tr_tokens:
                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
    else:
        st.warning("No data available for the selected tokens.")

with st.expander("Understanding the Daily Hurst Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the Hurst exponent values for all selected tokens over the last 24 hours using 30-minute bars.
    Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent