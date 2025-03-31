import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta, UTC

st.set_page_config(
    page_title="Daily Hurst Table",
    page_icon="📊",
    layout="wide"
)

# Database configuration
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

def calculate_hurst(prices):
    """
    Calculate Hurst-like exponent for a price series
    """
    if len(prices) < 10:
        return 0.5
    
    try:
        # Ensure unique prices to avoid calculation issues
        unique_prices = np.unique(prices)
        
        # Log returns of unique prices
        log_returns = np.diff(np.log(unique_prices))
        
        # Multiple estimation methods
        hurst_estimates = []
        
        # 1. Basic autocorrelation method
        try:
            if len(log_returns) > 1:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                hurst_estimates.append(0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4)))
        except:
            pass
        
        # 2. Variance ratio method
        try:
            # Calculate variance at different lags
            lags = [1, 2, 4]
            var_ratios = []
            for lag in lags:
                if len(log_returns) > lag:
                    var_ratio = np.var(log_returns[lag:]) / np.var(log_returns)
                    var_ratios.append(var_ratio)
            
            if var_ratios:
                hurst_var = 0.5 + np.mean(var_ratios) / 2
                hurst_estimates.append(hurst_var)
        except:
            pass
        
        # Return median estimate or default
        if hurst_estimates:
            return float(np.median(hurst_estimates))
        return 0.5
    
    except:
        return 0.5

def process_token_data(token):
    # Get current time and 24 hours ago
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=1)
    
    # Fetch data query
    query = f"""
    WITH time_blocks AS (
        SELECT 
            generate_series(
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '8 hours',
                date_trunc('day', '{start_time}'::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_start
    )
    SELECT 
        time_blocks.block_start AS time_block,
        final_price,
        created_at
    FROM time_blocks
    JOIN public.oracle_price_log ON 
        pair_name = '{token}' AND
        created_at >= time_blocks.block_start AND
        created_at < time_blocks.block_start + INTERVAL '30 minutes'
    ORDER BY created_at;
    """
    
    # Read data
    df = pd.read_sql(query, engine)
    
    # If no data, return NaN
    if df.empty:
        return None
    
    # Convert to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Group data into 5-minute windows within 30-minute block
    def calculate_block_hurst(block_data):
        # If not enough data, return NaN
        if len(block_data) < 10:
            return np.nan
        
        # Divide the block into 5-minute windows
        windows = []
        start_time = block_data['created_at'].min()
        end_time = start_time + timedelta(minutes=30)
        
        current_window_start = start_time
        current_window_end = current_window_start + timedelta(minutes=5)
        
        window_hursts = []
        
        while current_window_start < end_time:
            # Select data for this 5-minute window
            window_data = block_data[
                (block_data['created_at'] >= current_window_start) & 
                (block_data['created_at'] < current_window_end)
            ]
            
            # Calculate Hurst for this window
            if len(window_data) >= 10:
                window_hurst = calculate_hurst(window_data['final_price'].values)
                window_hursts.append(window_hurst)
            
            # Move to next window
            current_window_start = current_window_end
            current_window_end = current_window_start + timedelta(minutes=5)
        
        # Return average of window Hursts
        return np.nanmean(window_hursts) if window_hursts else np.nan
    
    # Group by 30-minute blocks and calculate multi-window Hurst
    grouped_df = df.groupby(pd.Grouper(key='created_at', freq='30min')).apply(calculate_block_hurst).reset_index()
    grouped_df.columns = ['time_block', 'Hurst']
    
    # Create time labels
    grouped_df['time_label'] = grouped_df['time_block'].dt.strftime('%H:%M')
    
    # Ensure all 48 time blocks are present
    standard_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
    
    # Reindex to ensure all time blocks are present
    result = grouped_df.set_index('time_label')['Hurst'].reindex(standard_labels)
    
    return result

# Fetch available tokens
def get_available_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log"
    return pd.read_sql(query, engine)['pair_name'].tolist()

# Streamlit app
st.title("Daily Hurst Table")
st.subheader("Market Regime Analysis")

# Generate button
if st.button("Generate Hurst Table"):
    # Get all tokens
    tokens = get_available_tokens()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results
    results = {}
    
    # Process each token
    for i, token in enumerate(tokens):
        progress_bar.progress((i+1)/len(tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(tokens)})")
        
        try:
            token_result = process_token_data(token)
            if token_result is not None:
                results[token] = token_result
        except Exception as e:
            st.error(f"Error processing {token}: {e}")
    
    # Finalize
    progress_bar.progress(1.0)
    status_text.text("Processing complete")

    # Create DataFrame
    if results:
        hurst_table = pd.DataFrame(results).round(2)
        
        # Styling
        def color_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5'
            elif val < 0.4:
                intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
                return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
            elif val > 0.6:
                intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            else:
                return 'background-color: rgba(200, 200, 200, 0.5); color: black'
        
        # Display table
        styled_table = hurst_table.style.applymap(color_cells)
        st.dataframe(styled_table, height=700, use_container_width=True)
    else:
        st.warning("No data processed")