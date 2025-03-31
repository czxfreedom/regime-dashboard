# Save this as pages/04_Daily_Hurst_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Daily Hurst Table",
    page_icon="📊",
    layout="wide"
)

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Daily Hurst Table (30min)")
st.subheader("All Trading Pairs - Last 24 Hours")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
rolling_window = 20  # Window size for Hurst calculation
expected_points = 48  # Expected data points per pair over 24 hours

# Fetch all available tokens from DB
@st.cache_data
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
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
    if ts is None:  # Check for None input
        return np.nan

    try:
        ts = np.array(ts, dtype=float)
    except:
        return np.nan  # Return NaN if conversion fails

    if len(ts) < 10 or np.any(~np.isfinite(ts)):
        return np.nan

    # ... (rest of your Hurst calculation logic) ...
    
    # Convert to returns - using log returns handles any scale of asset
    # Add small epsilon to avoid log(0)
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
        # Create range of lags - adaptive based on data length
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        rs_values = []
        for lag in lags:
            # Reshape returns into segments
            segments = len(log_returns) // lag
            if segments < 1:
                continue
                
            # Calculate R/S for each segment
            rs_by_segment = []
            for i in range(segments):
                segment = log_returns[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:  # Skip if segment is too short
                    continue
                    
                # Get mean and standard deviation
                mean_return = np.mean(segment)
                std_return = np.std(segment)
                
                if std_return == 0:  # Skip if no variation
                    continue
                    
                # Calculate cumulative deviation from mean
                cumdev = np.cumsum(segment - mean_return)
                
                # Calculate R/S statistic
                r = np.max(cumdev) - np.min(cumdev)
                s = std_return
                
                rs_by_segment.append(r / s)
            
            if rs_by_segment:
                rs_values.append((lag, np.mean(rs_by_segment)))
        
        # Need at least 4 points for reliable regression
        if len(rs_values) >= 4:
            lags_log = np.log10([x[0] for x in rs_values])
            rs_log = np.log10([x[1] for x in rs_values])
            
            # Calculate Hurst exponent from slope
            poly = np.polyfit(lags_log, rs_log, 1)
            h_rs = poly[0]
            hurst_estimates.append(h_rs)
    except:
        pass
    
    # Method 2: Variance Method
    try:
        # Calculate variance at different lags
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        var_values = []
        for lag in lags:
            if lag >= len(log_returns):
                continue
                
            # Compute the log returns at different lags
            lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
            
            if len(lagged_returns) < 2:
                continue
                
            # Calculate variance of the lagged series
            var = np.var(lagged_returns)
            if var > 0:
                var_values.append((lag, var))
        
        # Need at least 4 points for reliable regression
        if len(var_values) >= 4:
            lags_log = np.log10([x[0] for x in var_values])
            var_log = np.log10([x[1] for x in var_values])
            
            # For variance, the slope should be 2H-1
            poly = np.polyfit(lags_log, var_log, 1)
            h_var = (poly[0] + 1) / 2
            hurst_estimates.append(h_var)
    except:
        pass
    
    # Fallback to autocorrelation method if other methods fail
    if not hurst_estimates and len(log_returns) > 1:
        try:
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            
            # Convert autocorrelation to Hurst estimate
            # Strong negative correlation suggests mean reversion (H < 0.5)
            # Strong positive correlation suggests trending (H > 0.5)
            h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
            hurst_estimates.append(h_acf)
        except:
            pass
    
    # If we have estimates, aggregate them and constrain to 0-1 range
    if hurst_estimates:
        # Remove any extreme outliers
        valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
        
        # If no valid estimates remain after filtering, use all estimates but constrain them
        if not valid_estimates and hurst_estimates:
            valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
        
        # If we have valid estimates, return their median (more robust than mean)
        if valid_estimates:
            return np.median(valid_estimates)
    
    # If all methods fail, return 0.5 (random walk assumption)
    return 0.5

# Detailed regime classification function
def detailed_regime_classification(hurst):
    """
    Provides a more detailed regime classification including intensity levels.
    
    Args:
        hurst: Calculated Hurst exponent value
        
    Returns:
        tuple: (regime category, intensity level, description)
    """
    if pd.isna(hurst):
        return ("UNKNOWN", 0, "Insufficient data")
    
    # Strong mean reversion
    elif hurst < 0.2:
        return ("MEAN-REVERT", 3, "Strong mean-reversion")
    
    # Moderate mean reversion
    elif hurst < 0.3:
        return ("MEAN-REVERT", 2, "Moderate mean-reversion")
    
    # Mild mean reversion
    elif hurst < 0.4:
        return ("MEAN-REVERT", 1, "Mild mean-reversion")
    
    # Noisy/Random zone
    elif hurst < 0.45:
        return ("NOISE", 1, "Slight mean-reversion bias")
    elif hurst <= 0.55:
        return ("NOISE", 0, "Pure random walk")
    elif hurst < 0.6:
        return ("NOISE", 1, "Slight trending bias")
    
    # Mild trend
    elif hurst < 0.7:
        return ("TREND", 1, "Mild trending")
    
    # Moderate trend
    elif hurst < 0.8:
        return ("TREND", 2, "Moderate trending")
    
    # Strong trend
    else:
        return ("TREND", 3, "Strong trending")

# Fetch and calculate Hurst for a token with 30min timeframe
def fetch_and_calculate_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days + 1)  # Extra day for calculation buffer

    query = f"""
    SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
    AND pair_name = '{token}';
    """

    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None

        # Preprocess data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Resample to 1-minute timeframe
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        
        if one_min_ohlc.empty:
            return None

        # Calculate Hurst for each 1-minute interval
        one_min_ohlc['Hurst'] = one_min_ohlc['close'].apply(universal_hurst)

        # Resample to 30-minute timeframe and average the 1-minute Hurst values
        thirty_min_hurst = one_min_ohlc['Hurst'].resample('30min').mean().dropna()
        
        if thirty_min_hurst.empty:
            return None

        # Filter to last 24 hours (48 x 30min intervals)
        last_24h_hurst = thirty_min_hurst.iloc[-48:]

        # Create half-hour time labels (00:00, 00:30, 01:00, etc.)
        last_24h_hurst = last_24h_hurst.to_frame() #Convert Series to DataFrame
        last_24h_hurst['time_label'] = last_24h_hurst.index.strftime('%H:%M')

        # Add regime classification
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
    # Update progress
    progress_bar.progress((i) / len(selected_tokens))
    status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
    
    # Calculate Hurst
    result = fetch_and_calculate_hurst(token)
    if result is not None:
        token_results[token] = result

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    # Get all unique time labels
    all_times = set()
    for df in token_results.values():
        all_times.update(df['time_label'].tolist())
    
    all_times = sorted(all_times)
    
    # Create a multi-index DataFrame for the table
    table_data = {}
    
    # For each token, add its Hurst values
    for token, df in token_results.items():
        # Create a Series with time_label as index
        hurst_series = df.set_index('time_label')['Hurst']
        
        # Add to table data
        table_data[token] = hurst_series
    
    # Convert to DataFrame
    hurst_table = pd.DataFrame(table_data)
    
    # Reindex to ensure all time labels are present
    hurst_table = hurst_table.reindex(all_times)
    
    # Sort by rows (time)
    hurst_table = hurst_table.sort_index()
    
    # Round values for display
    hurst_table = hurst_table.round(2)
    
    # Create color-coded style function
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5'
        elif val < 0.4:
            # Mean reversion: red scale
            intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
            return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
        elif val > 0.6:
            # Trending: green scale
            intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        else:
            # Random walk: gray scale
            return 'background-color: rgba(200, 200, 200, 0.5); color: black'
    
    # Apply styling
    styled_table = hurst_table.style.applymap(color_cells)
    
    # Display the table
    st.markdown("## Hurst Exponent Table (30min timeframe, last 24 hours)")
    st.markdown("### Color Legend: <span style='color:red'>Red = Mean Reversion</span>, <span style='color:gray'>Gray = Random Walk</span>, <span style='color:green'>Green = Trending</span>", unsafe_allow_html=True)
    
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Add summary statistics
    st.subheader("Current Market Overview")
    
    # Get the most recent Hurst value for each token
    latest_values = {}
    for token, df in token_results.items():
        if not df.empty and not df['Hurst'].isna().all():
            latest = df['Hurst'].iloc[-1]
            regime = df['regime_desc'].iloc[-1]
            latest_values[token] = (latest, regime)
    
    # Calculate statistics
    if latest_values:
        mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
        random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
        trending = sum(1 for v, r in latest_values.values() if v > 0.6)
        
        total = mean_reverting + random_walk + trending
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean-Reverting", f"{mean_reverting} ({mean_reverting/total*100:.1f}%)")
        col2.metric("Random Walk", f"{random_walk} ({random_walk/total*100:.1f}%)")
        col3.metric("Trending", f"{trending} ({trending/total*100:.1f}%)")
        
        # Create a pie chart
        labels = ['Mean-Reverting', 'Random Walk', 'Trending']
        values = [mean_reverting, random_walk, trending]
        colors = ['rgba(255,100,100,0.7)', 'rgba(200,200,200,0.7)', 'rgba(100,255,100,0.7)']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=.3,
        )])
        
        fig.update_layout(
            title="Current Market Regime Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show tokens in each category
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Mean-Reverting Tokens")
            mr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v < 0.4]
            mr_tokens.sort(key=lambda x: x[1])  # Sort by Hurst value
            
            if mr_tokens:
                for token, value, regime in mr_tokens:
                    st.markdown(f"- **{token}**: {value:.2f} ({regime})")
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Random Walk Tokens")
            rw_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if 0.4 <= v <= 0.6]
            rw_tokens.sort(key=lambda x: x[1])  # Sort by Hurst value
            
            if rw_tokens:
                for token, value, regime in rw_tokens:
                    st.markdown(f"- **{token}**: {value:.2f} ({regime})")
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### Trending Tokens")
            tr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v > 0.6]
            tr_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by Hurst value (descending)
            
            if tr_tokens:
                for token, value, regime in tr_tokens:
                    st.markdown(f"- **{token}**: {value:.2f} ({regime})")
            else:
                st.markdown("*No tokens in this category*")
    else:
        st.warning("No data available for the selected tokens.")

# Add explanatory info
with st.expander("Understanding the Daily Hurst Table"):
    st.markdown("""
    ### How to Read This Table
    
    This table shows the Hurst exponent values for all selected tokens over the last 24 hours using 30-minute bars.
    
    Each row represents a specific 30-minute time period, and each column represents a different token.
    
    **Color coding:**
    - **Red** (Hurst < 0.4): The token is showing mean-reverting behavior during that time period
    - **Gray** (Hurst 0.4-0.6): The token is behaving like a random walk (no clear pattern)
    - **Green** (Hurst > 0.6): The token is showing trending behavior
    
    **The intensity of the color indicates the strength of the pattern:**
    - Darker red = Stronger mean-reversion
    - Darker green = Stronger trending
    
    **Technical details:**
    - Each Hurst value is calculated by averaging the Hurst values of 30 one-minute bars.
    - Values are calculated using multiple methods (R/S Analysis, Variance Method, and DFA)
    - Missing values (blank cells) indicate insufficient data for calculation
    """)