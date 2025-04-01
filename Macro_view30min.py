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
    print(f"universal_hurst called with ts: {ts}")
    print(f"Type of ts: {type(ts)}")
    if isinstance(ts, (list, np.ndarray, pd.Series)) and len(ts) > 0:
        print(f"First few values of ts: {ts[:5]}")
    
    if ts is None:
        print("ts is None")
        return np.nan
    
    if isinstance(ts, pd.Series) and ts.empty:
        print("ts is empty series")
        return np.nan
        
    if not isinstance(ts, (list, np.ndarray, pd.Series)):
        print(f"ts is not a list, NumPy array, or Series. Type: {type(ts)}")
        return np.nan

    try:
        ts = np.array(ts, dtype=float)
    except Exception as e:
        print(f"ts cannot be converted to float: {e}")
        return np.nan

    if len(ts) < 10 or np.any(~np.isfinite(ts)):
        print(f"ts length < 10 or non-finite values: {ts}")
        return np.nan

    # Convert to returns - using log returns handles any scale of asset
    epsilon = 1e-10
    adjusted_ts = ts + epsilon
    log_returns = np.diff(np.log(adjusted_ts))
    
    # If all returns are exactly zero (completely flat price), return 0.5
    if np.all(log_returns == 0):
        return 0.5
    
    # Use multiple methods and average for robustness
    hurst_estimates = []
    
    # Method 1: Rescaled Range (R/S) Analysis
    try:
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        rs_values = []
        for lag in lags:
            segments = len(log_returns) // lag
            if segments < 1:
                continue
            rs_by_segment = []
            for i in range(segments):
                segment = log_returns[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:
                    continue
                mean_return = np.mean(segment)
                std_return = np.std(segment)
                if std_return == 0:
                    continue
                cumdev = np.cumsum(segment - mean_return)
                r = np.max(cumdev) - np.min(cumdev)
                s = std_return
                rs_by_segment.append(r / s)
            if rs_by_segment:
                rs_values.append((lag, np.mean(rs_by_segment)))
        if len(rs_values) >= 4:
            lags_log = np.log10([x[0] for x in rs_values])
            rs_log = np.log10([x[1] for x in rs_values])
            poly = np.polyfit(lags_log, rs_log, 1)
            h_rs = poly[0]
            hurst_estimates.append(h_rs)
    except Exception as e:
        print(f"Error in R/S calculation: {e}")
        pass
    
    # Method 2: Variance Method
    try:
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        var_values = []
        for lag in lags:
            if lag >= len(log_returns):
                continue
            lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
            if len(lagged_returns) < 2:
                continue
            var = np.var(lagged_returns)
            if var > 0:
                var_values.append((lag, var))
        if len(var_values) >= 4:
            lags_log = np.log10([x[0] for x in var_values])
            var_log = np.log10([x[1] for x in var_values])
            poly = np.polyfit(lags_log, var_log, 1)
            h_var = (poly[0] + 1) / 2
            hurst_estimates.append(h_var)
    except Exception as e:
        print(f"Error in Variance calculation: {e}")
        pass
    
    # Fallback to autocorrelation method if other methods fail
    if not hurst_estimates and len(log_returns) > 1:
        try:
            autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
            hurst_estimates.append(h_acf)
        except Exception as e:
            print(f"Error in Autocorrelation calculation: {e}")
            pass
    
    # If we have estimates, aggregate them and constrain to 0-1 range
    if hurst_estimates:
        valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
        if not valid_estimates and hurst_estimates:
            valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
        if valid_estimates:
            return np.median(valid_estimates)
    
    return 0.5

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

# Function to generate fixed 30-minute time blocks for the past 24 hours
def generate_fixed_time_blocks(current_time):
    # Round down to the nearest 30-minute mark (e.g., 3:40 PM becomes 3:30 PM)
    current_minute = current_time.minute
    current_hour = current_time.hour
    
    if current_minute < 30:
        # Round down to XX:00
        latest_block_end = current_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Round down to XX:30
        latest_block_end = current_time.replace(minute=30, second=0, microsecond=0)
    
    # Generate 48 blocks (24 hours with 30-minute blocks)
    time_blocks = []
    for i in range(48):
        block_end = latest_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}"
        time_blocks.append((block_start, block_end, block_label))
    
    return time_blocks

# Fetch and calculate Hurst for a token with fixed 30min timeframe blocks
@st.cache_data(ttl=600, show_spinner="Calculating Hurst exponents...")
def fetch_and_calculate_hurst(token, time_blocks):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Get the earliest time needed (24 hours back from now)
    earliest_time_sg = time_blocks[-1][0]
    
    # Convert to UTC for database query
    earliest_time_utc = earliest_time_sg.astimezone(pytz.utc)
    latest_time_utc = now_sg.astimezone(pytz.utc)

    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price, 
        pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{earliest_time_utc}' AND '{latest_time_utc}'
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
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            print(f"[{token}] No OHLC data after resampling.")
            return None
            
        print(f"[{token}] one_min_ohlc head:\n{one_min_ohlc.head()}")
        
        # Create a DataFrame to store Hurst values for each 30-min block
        result_df = pd.DataFrame(columns=['block_start', 'block_end', 'block_label', 'Hurst'])
        
        # Calculate Hurst for each fixed time block
        for block_start, block_end, block_label in time_blocks:
            # Extract data for this time block
            block_data = one_min_ohlc.loc[block_start:block_end]
            
            if len(block_data) >= rolling_window:  # Only compute if enough data
                # Apply universal_hurst to the 'close' prices
                hurst_value = universal_hurst(block_data['close'].values)
                
                # Add to results DataFrame
                result_df = pd.concat([result_df, pd.DataFrame({
                    'block_start': [block_start],
                    'block_end': [block_end],
                    'block_label': [block_label],
                    'Hurst': [hurst_value]
                })])
        
        if result_df.empty:
            print(f"[{token}] No Hurst values calculated for any time block.")
            return None
            
        # Add regime information
        result_df['regime_info'] = result_df['Hurst'].apply(detailed_regime_classification)
        result_df['regime'] = result_df['regime_info'].apply(lambda x: x[0])
        result_df['regime_desc'] = result_df['regime_info'].apply(lambda x: x[2])
        
        print(f"[{token}] Successful Calculation with {len(result_df)} time blocks")
        return result_df
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Get current time in Singapore timezone
current_time_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {current_time_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Generate fixed 30-minute time blocks
time_blocks = generate_fixed_time_blocks(current_time_sg)

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame(time_blocks, columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate Hurst for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:  # Added try-except around token processing
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_hurst(token, time_blocks)
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
    # Create a pivoted table with time blocks as rows and tokens as columns
    hurst_table = pd.DataFrame()
    
    # Extract block labels and use them as index
    if token_results:
        first_token = list(token_results.keys())[0]
        block_labels = token_results[first_token]['block_label'].tolist()
        hurst_table['time_block'] = block_labels
        hurst_table = hurst_table.set_index('time_block')
        
        # Fill in Hurst values for each token
        for token, df in token_results.items():
            token_hurst = dict(zip(df['block_label'], df['Hurst']))
            hurst_table[token] = hurst_table.index.map(lambda x: token_hurst.get(x, np.nan))
    
    # Round to 2 decimal places
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
    # Use the most recent time block for each token
    latest_values = {}
    for token, df in token_results.items():
        if not df.empty and not df['Hurst'].isna().all():
            latest = df['Hurst'].iloc[0]  # Most recent is at index 0
            regime = df['regime_desc'].iloc[0]
            latest_values[token] = (latest, regime)
    
    if latest_values:
        mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
        random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
        trending = sum(1 for v, r in latest_values.values() if v > 0.6)
        total = mean_reverting + random_walk + trending
        
        if total > 0:  # Avoid division by zero
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
            st.warning("No valid data found for analysis.")
    else:
        st.warning("No data available for the selected tokens.")

with st.expander("Understanding the Daily Hurst Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the Hurst exponent values for all selected tokens over the last 24 hours using fixed 30-minute blocks.
    Each row represents a specific 30-minute time period (like 3:00-3:30 PM), with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
    **Color coding:**
    - **Red** (Hurst < 0.4): The token is showing mean-reverting behavior during that time period
    - **Gray** (Hurst 0.4-0.6): The token is behaving like a random walk (no clear pattern)
    - **Green** (Hurst > 0.6): The token is showing trending behavior
    **The intensity of the color indicates the strength of the pattern:**
    - Darker red = Stronger mean-reversion
    - Darker green = Stronger trending
    **Technical details:**
    - Each Hurst value is calculated using price data strictly within each 30-minute block
    - Values are calculated using multiple methods (R/S Analysis, Variance Method, and Autocorrelation)
    - Missing values (light gray cells) indicate insufficient data for calculation
    """)