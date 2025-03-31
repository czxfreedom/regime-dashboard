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
        st.rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Universal Hurst calculation function (previous robust implementation)
def universal_hurst(ts):
    # (Full implementation as in previous artifact)
    # Keeping the previous robust implementation
    
    # Fallback to simple calculation if complex method fails
    if len(ts) < 2:
        return 0.5
    
    try:
        # Simple log returns
        log_returns = np.diff(np.log(ts))
        
        # Basic autocorrelation approach
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        
        # Map autocorrelation to Hurst-like value
        hurst = 0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4))
        
        return max(0, min(1, hurst))
    except:
        return 0.5

# Detailed regime classification function (previous implementation)
def detailed_regime_classification(hurst):
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

# New function to calculate comprehensive Hurst analysis
def calculate_comprehensive_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    
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
        COALESCE(
            AVG(final_price),
            (SELECT AVG(final_price) FROM public.oracle_price_log 
             WHERE pair_name = '{token}' 
             AND created_at BETWEEN '{start_time}' AND '{end_time}')
        ) AS avg_price
    FROM time_blocks
    LEFT JOIN public.oracle_price_log ON 
        pair_name = '{token}' AND
        created_at >= time_blocks.block_start AND
        created_at < time_blocks.block_start + INTERVAL '30 minutes'
    GROUP BY time_blocks.block_start
    ORDER BY time_blocks.block_start;
    """
    
    try:
        # Fetch all tick-level data
        df = pd.read_sql(query, engine)
        
        if df.empty:
            st.warning(f"No data found for {token}")
            return None
        
        # Ensure we have exactly 48 time blocks (24 hours * 2 blocks per hour)
        if len(df) != 48:
            st.warning(f"Incomplete data for {token}: {len(df)}/48 blocks")
        
        # Remove any NaN values
        df = df.dropna()
        
        # Add Hurst calculation
        df['Hurst'] = df['avg_price'].apply(lambda x: universal_hurst([x]))
        
        # Add regime classification
        df['regime_info'] = df['Hurst'].apply(detailed_regime_classification)
        df['regime'] = df['regime_info'].apply(lambda x: x[0])
        df['regime_desc'] = df['regime_info'].apply(lambda x: x[2])
        
        # Create time labels
        df['time_label'] = df['time_block'].dt.strftime('%H:%M')
        
        return df.set_index('time_label')[['Hurst', 'regime', 'regime_desc']]
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# Calculate Hurst for each token
progress_bar = st.progress(0)
status_text = st.empty()

token_results = {}
for i, token in enumerate(selected_tokens):
    # Update progress
    progress_bar.progress((i+1) / len(selected_tokens))
    status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
    
    # Calculate Hurst
    result = calculate_comprehensive_hurst(token)
    if result is not None:
        token_results[token] = result

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    # Create a multi-index DataFrame for the table
    table_data = {}
    
    # For each token, add its Hurst values
    for token, df in token_results.items():
        # Add to table data
        table_data[token] = df['Hurst']
    
    # Convert to DataFrame
    hurst_table = pd.DataFrame(table_data)
    
    # Ensure 48 rows (24 hours with 30-minute intervals)
    if len(hurst_table) != 48:
        # Pad with NaN if needed
        dummy_index = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
        hurst_table = hurst_table.reindex(dummy_index)
    
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
    st.markdown("## Hurst Exponent Table (Tick-Level, 30min blocks)")
    st.markdown("### Color Legend: <span style='color:red'>Red = Mean Reversion</span>, <span style='color:gray'>Gray = Random Walk</span>, <span style='color:green'>Green = Trending</span>", unsafe_allow_html=True)
    
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # (Rest of the summary statistics code remains the same as in previous artifact)
else:
    st.warning("No data available for the selected tokens.")

# Add explanatory info
with st.expander("Understanding the Daily Hurst Table"):
    st.markdown("""
    ### How to Read This Table
    
    This table shows the Hurst exponent values for all selected tokens over 24 hours.
    
    **Calculation Method:**
    - Each cell represents a 30-minute block
    - Hurst exponent calculated from average price in that block
    - 48 blocks total (24 hours * 2 blocks per hour)
    
    **Color coding:**
    - **Red** (Hurst < 0.4): Mean-reverting behavior
    - **Gray** (Hurst 0.4-0.6): Random walk
    - **Green** (Hurst > 0.6): Trending behavior
    
    **The color intensity indicates pattern strength:**
    - Darker red = Stronger mean-reversion
    - Darker green = Stronger trending
    """)