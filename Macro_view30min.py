import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta, UTC

st.set_page_config(
    page_title="Daily Hurst Table (Multi-Sample)",
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
st.title("Daily Hurst Table (Multi-Sample Analysis)")
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
        st.rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

def calculate_hurst(prices, max_samples=50):
    """
    Calculate Hurst exponent with multiple sampling strategies.
    
    Args:
        prices (array-like): Price series
        max_samples (int): Maximum number of samples to use
    
    Returns:
        float: Estimated Hurst exponent
    """
    if len(prices) < 10:
        return 0.5
    
    # Ensure we don't oversample
    sample_size = min(len(prices), max_samples)
    
    # Different sampling strategies
    hurst_estimates = []
    
    # 1. Full series log returns
    try:
        log_returns = np.diff(np.log(prices))
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        hurst_estimates.append(0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4)))
    except:
        pass
    
    # 2. Random sampling of prices
    try:
        # Take random samples without replacement
        np.random.seed(42)  # for reproducibility
        sample_indices = np.random.choice(len(prices), sample_size, replace=False)
        sample_prices = prices[sample_indices]
        
        sample_returns = np.diff(np.log(sample_prices))
        if len(sample_returns) > 0:
            sample_autocorr = np.corrcoef(sample_returns[:-1], sample_returns[1:])[0, 1]
            hurst_estimates.append(0.5 + (np.sign(sample_autocorr) * min(abs(sample_autocorr), 0.4)))
    except:
        pass
    
    # 3. Sliding window method
    try:
        window_sizes = [len(prices) // 4, len(prices) // 2]
        for window in window_sizes:
            if window > 10:
                windowed_returns = [
                    np.diff(np.log(prices[i:i+window]))
                    for i in range(0, len(prices)-window, window//2)
                ]
                
                window_correlations = []
                for returns in windowed_returns:
                    if len(returns) > 1:
                        try:
                            corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                            window_correlations.append(corr)
                        except:
                            pass
                
                if window_correlations:
                    avg_corr = np.mean(window_correlations)
                    hurst_estimates.append(0.5 + (np.sign(avg_corr) * min(abs(avg_corr), 0.4)))
    except:
        pass
    
    # Return median of estimates or default
    if hurst_estimates:
        return float(np.median(hurst_estimates))
    return 0.5

def calculate_comprehensive_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=lookback_days)
    
    query = f"""
    WITH time_blocks AS (
        SELECT 
            generate_series(
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '8 hours',
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_start,
            generate_series(
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '8 hours',
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_end
    )
    SELECT 
        time_blocks.block_start AS time_block,
        final_price,
        created_at
    FROM time_blocks
    JOIN public.oracle_price_log ON 
        pair_name = '{token}' AND
        created_at >= time_blocks.block_start AND
        created_at < time_blocks.block_end
    ORDER BY created_at;
    """
    
    try:
        # Fetch all tick-level data
        df = pd.read_sql(query, engine)
        
        if df.empty:
            st.warning(f"No data found for {token}")
            return None
        
        # Group by time blocks and calculate Hurst
        def calculate_block_hurst(block_data):
            if len(block_data) < 10:
                return np.nan
            
            prices = block_data['final_price'].values
            return calculate_hurst(prices)
        
        grouped_df = df.groupby('time_block').apply(calculate_block_hurst).reset_index()
        grouped_df.columns = ['time_block', 'Hurst']
        
        # Create time labels
        grouped_df['time_label'] = grouped_df['time_block'].dt.strftime('%H:%M')
        
        # Add regime classification
        def classify_regime(hurst):
            if pd.isna(hurst):
                return ("UNKNOWN", 0, "Insufficient data")
            elif hurst < 0.2:
                return ("MEAN-REVERT", 3, "Strong mean-reversion")
            elif hurst < 0.4:
                return ("MEAN-REVERT", 2, "Moderate mean-reversion")
            elif hurst <= 0.6:
                return ("NOISE", 0, "Random walk")
            elif hurst < 0.8:
                return ("TREND", 2, "Moderate trending")
            else:
                return ("TREND", 3, "Strong trending")
        
        grouped_df['regime_info'] = grouped_df['Hurst'].apply(classify_regime)
        grouped_df['regime'] = grouped_df['regime_info'].apply(lambda x: x[0])
        grouped_df['regime_desc'] = grouped_df['regime_info'].apply(lambda x: x[2])
        
        return grouped_df.set_index('time_label')[['Hurst', 'regime', 'regime_desc']]
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# (Rest of the code remains the same as in the previous artifact)