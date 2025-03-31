import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
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

# Diagnostic function to check data availability
def check_data_availability():
    try:
        with engine.connect() as connection:
            # Check basic table information
            query = text("""
                SELECT 
                    pair_name, 
                    COUNT(*) as record_count, 
                    MIN(created_at) as earliest_record, 
                    MAX(created_at) as latest_record
                FROM public.oracle_price_log
                GROUP BY pair_name
                ORDER BY record_count DESC
                LIMIT 50;
            """)
            
            result = connection.execute(query)
            data = result.fetchall()
            
            # Display diagnostic information
            st.write("Data Availability Diagnostic:")
            for row in data:
                st.write(f"Pair: {row[0]}, Records: {row[1]}, "
                         f"Earliest: {row[2]}, Latest: {row[3]}")
            
            return [row[0] for row in data]
    
    except Exception as e:
        st.error(f"Error checking data availability: {e}")
        return []

# Hurst Calculation Function (simplified for diagnosis)
def calculate_comprehensive_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=1)
    
    query = text(f"""
    SELECT 
        final_price,
        created_at
    FROM public.oracle_price_log 
    WHERE 
        pair_name = :token AND
        created_at BETWEEN :start_time AND :end_time
    ORDER BY created_at;
    """)
    
    try:
        # Fetch all tick-level data
        with engine.connect() as connection:
            df = pd.read_sql(
                query, 
                connection, 
                params={
                    'token': token, 
                    'start_time': start_time, 
                    'end_time': end_time
                }
            )
        
        if df.empty:
            st.warning(f"No data found for {token}")
            return None
        
        # Basic Hurst calculation if data exists
        prices = df['final_price'].values
        
        if len(prices) < 10:
            st.warning(f"Insufficient data for {token}: {len(prices)} points")
            return None
        
        # Simple Hurst estimation
        log_returns = np.diff(np.log(prices))
        autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        hurst = 0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4))
        
        # Create dummy DataFrame with standardized structure
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
        dummy_df = pd.DataFrame(
            index=time_labels, 
            columns=['Hurst', 'regime', 'regime_desc']
        )
        
        # Fill with the calculated Hurst
        dummy_df['Hurst'] = hurst
        
        # Classify regime
        if hurst < 0.4:
            regime = "MEAN-REVERT"
            desc = "Mean-reverting"
        elif hurst > 0.6:
            regime = "TREND"
            desc = "Trending"
        else:
            regime = "NOISE"
            desc = "Random walk"
        
        dummy_df['regime'] = regime
        dummy_df['regime_desc'] = desc
        
        return dummy_df
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

# Main Streamlit app
st.title("Daily Hurst Table (Diagnostic)")
st.subheader("All Trading Pairs - Last 24 Hours")

# Diagnostic button
if st.button("Check Data Availability"):
    available_tokens = check_data_availability()
    st.write(f"Found {len(available_tokens)} tokens with data")

# Generate Hurst Table button
if st.button("Generate Hurst Table"):
    # Fetch tokens with data
    available_tokens = check_data_availability()
    
    if not available_tokens:
        st.error("No tokens found with data")
        st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results
    token_results = {}

    # Calculate for each token
    for i, token in enumerate(available_tokens):
        # Update progress
        progress_bar.progress((i+1) / len(available_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(available_tokens)})")
        
        # Calculate Hurst
        result = calculate_comprehensive_hurst(token)
        if result is not None:
            token_results[token] = result

    # Finalize progress
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(token_results)}/{len(available_tokens)} tokens")

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