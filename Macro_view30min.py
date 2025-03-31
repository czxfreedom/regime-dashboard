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

# Universal Hurst calculation function (same as previous implementation)
def universal_hurst(ts):
    """
    A universal Hurst exponent calculation that works for any asset class.
    
    Args:
        ts: Time series of prices (numpy array or list)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    # (Previous implementation remains the same)
    # [The entire universal_hurst function from the previous script]
    
    return 0.5  # Placeholder to remind you to copy the full function

# Detailed regime classification function (same as previous implementation)
def detailed_regime_classification(hurst):
    """
    Provides a more detailed regime classification including intensity levels.
    
    Args:
        hurst: Calculated Hurst exponent value
        
    Returns:
        tuple: (regime category, intensity level, description)
    """
    # (Previous implementation remains the same)
    # [The entire detailed_regime_classification function from the previous script]
    
    return ("UNKNOWN", 0, "Placeholder")

# New function to calculate Hurst from tick-level data within each 30-minute block
def calculate_tick_level_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    
    query = f"""
    SELECT 
        date_trunc('30 minutes', created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours') AS time_block,
        final_price,
        price_magnitude,
        price_direction
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
    result = calculate_tick_level_hurst(token)
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
        all_times.update(df.index.tolist())
    
    all_times = sorted(all_times)
    
    # Create a multi-index DataFrame for the table
    table_data = {}
    
    # For each token, add its Hurst values
    for token, df in token_results.items():
        # Add to table data
        table_data[token] = df['Hurst']
    
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
    st.markdown("## Hurst Exponent Table (Tick-Level, 30min blocks)")
    st.markdown("### Color Legend: <span style='color:red'>Red = Mean Reversion</span>, <span style='color:gray'>Gray = Random Walk</span>, <span style='color:green'>Green = Trending</span>", unsafe_allow_html=True)
    
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Add summary statistics
    st.subheader("Current Market Overview")
    
    # Get the most recent Hurst value for each token
    latest_values = {}
    for token, df in token_results.items():
        if not df.empty and not df['Hurst'].isna().all():
            latest = df['Hurst'].iloc[-1]
            regime = df.iloc[-1]['regime_desc']
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
    
    This table shows the Hurst exponent values calculated from tick-level data, aggregated into 30-minute blocks for all selected tokens over the last 24 hours.
    
    **Key Changes:**
    - Hurst exponent is now calculated using ALL tick data within each 30-minute block
    - Captures the microstructure behavior of each 30-minute interval
    - Provides a more granular view of market dynamics
    
    **Color coding:**
    - **Red** (Hurst < 0.4): The token is showing mean-reverting behavior during that time period
    - **Gray** (Hurst 0.4-0.6): The token is behaving like a random walk (no clear pattern)
    - **Green** (Hurst > 0.6): The token is showing trending behavior
    
    **The intensity of the color indicates the strength of the pattern:**
    - Darker red = Stronger mean-reversion
    - Darker green = Stronger trending
    
    **Technical details:**
    - Each Hurst value is calculated using ALL tick data within a 30-minute block
    - Multiple calculation methods are used to ensure robustness
    - Multiple data points provide a more comprehensive view of market behavior
    """)