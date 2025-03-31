import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Daily Hurst Table (Tick-Level)",
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
st.title("Daily Hurst Table (Tick-Level Aggregation)")
st.subheader("All Trading Pairs - Last 24 Hours")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours

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

# Universal Hurst calculation function (robust implementation)
def universal_hurst(ts):
    """
    A universal Hurst exponent calculation that works for any asset class.
    
    Args:
        ts: Time series of prices (numpy array or list)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    # Convert to numpy array and ensure floating point
    try:
        ts = np.array(ts, dtype=float)
    except:
        return np.nan  # Return NaN if conversion fails
        
    # Basic data validation
    if len(ts) < 10 or np.any(~np.isfinite(ts)):
        return np.nan
    
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

# New function to calculate Hurst from tick-level data within each 30-minute block
def calculate_tick_level_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    
    query = f"""
    SELECT 
        date_trunc('30 minutes', created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours') AS time_block,
        final_price
    FROM public.oracle_price_log
    WHERE 
        created_at BETWEEN '{start_time}' AND '{end_time}'
        AND pair_name = '{token}'
    ORDER BY created_at;
    """
    
    try:
        # Fetch all tick-level data
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None
        
        # Group by 30-minute blocks and calculate Hurst for each block
        def calculate_block_hurst(block_data):
            # If not enough data points, return NaN
            if len(block_data) < 10:
                return np.nan
            
            # Extract price data
            prices = block_data['final_price'].values
            
            # Apply universal Hurst calculation
            return universal_hurst(prices)
        
        # Group by time blocks and calculate Hurst
        hurst_results = df.groupby('time_block').apply(calculate_block_hurst)
        
        # Convert to DataFrame for easier manipulation
        hurst_df = hurst_results.reset_index()
        hurst_df.columns = ['time_block', 'Hurst']
        
        # Add regime classification
        hurst_df['regime_info'] = hurst_df['Hurst'].apply(detailed_regime_classification)
        hurst_df['regime'] = hurst_df['regime_info'].apply(lambda x: x[0])
        hurst_df['regime_desc'] = hurst_df['regime_info'].apply(lambda x: x[2])
        
        # Create time labels
        hurst_df['time_label'] = hurst_df['time_block'].dt.strftime('%H:%M')
        
        return hurst_df.set_index('time_label')[['Hurst', 'regime', 'regime_desc']]
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# (Rest of the code remains the same as in the previous artifact)