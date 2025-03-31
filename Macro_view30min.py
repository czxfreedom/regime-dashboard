import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta, UTC

st.set_page_config(
    page_title="Daily Hurst Table (Rolling Analysis)",
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

# Hurst Calculation Function (previous implementation remains the same)
def calculate_rolling_hurst(prices, window=30, min_periods=10):
    # (Previous implementation of rolling Hurst calculation)
    if len(prices) < min_periods:
        return np.full(len(prices), 0.5)
    
    def single_hurst(sub_prices):
        if len(sub_prices) < min_periods:
            return 0.5
        
        try:
            # Log returns
            log_returns = np.diff(np.log(sub_prices))
            
            # Multiple Hurst estimation methods
            hurst_estimates = []
            
            # 1. Autocorrelation method
            try:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                hurst_estimates.append(0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4)))
            except:
                pass
            
            # 2. Variance ratio method
            try:
                # Calculate variance at different lags
                lags = [1, 2, 4, 8]
                var_ratios = []
                for lag in lags:
                    var_ratio = np.var(log_returns[lag:]) / np.var(log_returns)
                    var_ratios.append(var_ratio)
                
                # Estimate Hurst from variance ratios
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
    
    # Calculate rolling Hurst
    rolling_hurst = []
    for i in range(len(prices) - window + 1):
        sub_prices = prices[i:i+window]
        rolling_hurst.append(single_hurst(sub_prices))
    
    # Pad the beginning with initial values
    padding = [rolling_hurst[0]] * (window - 1)
    return padding + rolling_hurst

def calculate_comprehensive_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=1)
    
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
        def process_block(block_data):
            if len(block_data) < 10:
                return np.nan
            
            prices = block_data['final_price'].values
            rolling_hursts = calculate_rolling_hurst(prices)
            
            # Return average of rolling Hurst estimates
            valid_hursts = [h for h in rolling_hursts if not np.isnan(h)]
            return np.mean(valid_hursts) if valid_hursts else np.nan
        
        grouped_df = df.groupby('time_block').apply(process_block).reset_index()
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
        
        # Create standard 48-block time labels
        standard_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
        
        # Set index and reindex
        result_df = grouped_df.set_index('time_label')[['Hurst', 'regime', 'regime_desc']]
        result_df = result_df.reindex(standard_labels)
        
        return result_df
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# Fetch all tokens
@st.cache_data
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return []

# Main Streamlit app
st.title("Daily Hurst Table (Rolling Analysis)")
st.subheader("All Trading Pairs - Last 24 Hours")

# Fetch and display all tokens automatically
all_tokens = fetch_all_tokens()

# Calculate Hurst for all tokens
if st.button("Generate Hurst Table"):
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results
    token_results = {}

    # Calculate for each token
    for i, token in enumerate(all_tokens):
        # Update progress
        progress_bar.progress((i+1) / len(all_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(all_tokens)})")
        
        # Calculate Hurst
        result = calculate_comprehensive_hurst(token)
        if result is not None:
            token_results[token] = result

    # Finalize progress
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(token_results)}/{len(all_tokens)} tokens")

    # Display results if any
    if token_results:
        # Prepare table data
        table_data = {}
        for token, df in token_results.items():
            table_data[token] = df['Hurst']
        
        # Create DataFrame
        hurst_table = pd.DataFrame(table_data).round(2)
        
        # Styling function
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
        
        # Display table
        st.markdown("## Hurst Exponent Table")
        st.dataframe(styled_table, height=700, use_container_width=True)
    else:
        st.warning("No tokens could be processed.")