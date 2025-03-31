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
    Simple Hurst exponent calculation
    """
    if len(prices) < 10:
        return 0.5
    
    try:
        # Log returns
        log_returns = np.diff(np.log(prices))
        
        # Autocorrelation method
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        
        # Map to Hurst-like value
        hurst = 0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4))
        
        return max(0, min(1, hurst))
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
        final_price
    FROM time_blocks
    LEFT JOIN public.oracle_price_log ON 
        pair_name = '{token}' AND
        created_at >= time_blocks.block_start AND
        created_at < time_blocks.block_start + INTERVAL '30 minutes'
    ORDER BY time_block;
    """
    
    # Read data
    df = pd.read_sql(query, engine)
    
    # Group by time blocks and calculate Hurst
    def block_hurst(block_data):
        prices = block_data['final_price'].dropna()
        return calculate_hurst(prices) if len(prices) > 0 else np.nan
    
    grouped = df.groupby('time_block').apply(block_hurst).reset_index()
    grouped.columns = ['time_block', 'Hurst']
    
    # Create time labels
    grouped['time_label'] = grouped['time_block'].dt.strftime('%H:%M')
    
    # Classify regime
    def classify_regime(hurst):
        if pd.isna(hurst):
            return "UNKNOWN"
        elif hurst < 0.4:
            return "MEAN-REVERT"
        elif hurst > 0.6:
            return "TREND"
        else:
            return "NOISE"
    
    grouped['regime'] = grouped['Hurst'].apply(classify_regime)
    
    # Ensure 48 time blocks
    standard_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
    result = grouped.set_index('time_label')['Hurst'].reindex(standard_labels)
    
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