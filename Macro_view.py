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

# Universal Hurst calculation function (remains the same)
# ...

# Detailed regime classification function (remains the same)
# ...

# Fetch and calculate Hurst for a token with 30min timeframe
@st.cache_data(ttl=600, show_spinner="Calculating Hurst exponents...")
def fetch_and_calculate_hurst(token):
    end_time_utc = datetime.utcnow()
    end_time_singapore = end_time_utc.replace(tzinfo=pytz.utc).astimezone(singapore_timezone)
    start_time_singapore = end_time_singapore - timedelta(days=lookback_days)

    query = f"""
    SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_singapore.astimezone(pytz.utc)}' AND '{end_time_singapore.astimezone(pytz.utc)}'
    AND pair_name = '{token}';
    """
    try:
        df = pd.read_sql(query, engine)

        if df.empty:
            return None

        # Convert timestamps to Singapore time immediately
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(singapore_timezone)
        df = df.set_index('timestamp').sort_index()

        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            return None

        one_min_ohlc['Hurst'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(universal_hurst)

        thirty_min_hurst = one_min_ohlc['Hurst'].resample('30min').mean().dropna()
        if thirty_min_hurst.empty:
            return None

        last_24h_hurst = thirty_min_hurst.iloc[-48:]
        last_24h_hurst['time_label'] = last_24h_hurst.index.strftime('%H:%M')  # Time in Singapore time
        last_24h_hurst = last_24h_hurst.to_frame()
        last_24h_hurst['regime_info'] = last_24h_hurst['Hurst'].apply(detailed_regime_classification)
        last_24h_hurst['regime'] = last_24h_hurst['regime_info'].apply(lambda x: x[0])
        last_24h_hurst['regime_desc'] = last_24h_hurst['regime_info'].apply(lambda x: x[2])
        return last_24h_hurst
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate Hurst for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_hurst(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    all_times = set()
    for df in token_results.values():
        all_times.update(df['time_label'].tolist())
    all_times = sorted(all_times, reverse=True)  # Sort times in descending order

    # Create a complete index for the last 24 hours
    end_time_singapore = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(singapore_timezone)
    start_time_singapore = end_time_singapore - timedelta(days=lookback_days)
    all_30min_times = pd.date_range(start=start_time_singapore, end=end_time_singapore, freq='30min').strftime('%H:%M').tolist()
    all_30min_times = sorted(set(all_30min_times), reverse=True)

    table_data = {}
    for token, df in token_results.items():
        hurst_series = df.set_index('time_label')['Hurst']
        table_data[token] = hurst_series

    hurst_table = pd.DataFrame(table_data).reindex(all_30min_times)  # Use the complete index
    hurst_table = hurst_table.sort_index(ascending=False).round(2)

    # Color cells function (remains the same)
    # ...

    styled_table = hurst_table.style.applymap(color_cells)
    st.markdown("## Hurst Exponent Table (30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:red'>Mean Reversion</span>, <span style='color:gray'>Random Walk</span>, <span style='color:green'>Trending</span>", unsafe_allow_html=True)
    st.dataframe(styled_table, height=700, use_container_width=True)
    st.subheader("Current Market Overview (Singapore Time)")

    # Latest values and market overview (remains the same)
    # ...

    with st.expander("Understanding the Daily Hurst Table"):
        # Expander content (remains the same)
        # ...