import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, UTC

st.set_page_config(
    page_title="Daily Hurst Table (Multi-Window Analysis)",
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

def calculate_multi_window_hurst(prices, num_windows=5, window_overlap=0.5):
    """
    Calculate Hurst exponent using multiple overlapping windows.
    
    Args:
        prices (array-like): Price series
        num_windows (int): Number of windows to divide the data
        window_overlap (float): Overlap between windows (0-1)
    
    Returns:
        float: Average Hurst estimate across windows
    """
    if len(prices) < 20:  # Minimum data requirement
        return 0.5
    
    # Compute window size and overlap
    total_length = len(prices)
    window_size = total_length // num_windows
    overlap_size = int(window_size * window_overlap)
    
    # Store Hurst estimates
    hurst_estimates = []
    
    for i in range(num_windows):
        # Compute start and end indices with overlap
        start = max(0, i * (window_size - overlap_size))
        end = min(total_length, start + window_size)
        
        # Extract window prices
        window_prices = prices[start:end]
        
        # Skip if window too small
        if len(window_prices) < 10:
            continue
        
        try:
            # Calculate log returns
            log_returns = np.diff(np.log(window_prices))
            
            # Multiple estimation methods
            hurst_methods = []
            
            # 1. Autocorrelation method
            try:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                hurst_methods.append(0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4)))
            except:
                pass
            
            # 2. Variance ratio method
            try:
                # Calculate variance at different lags
                lags = [1, 2, 4]
                var_ratios = []
                for lag in lags:
                    var_ratio = np.var(log_returns[lag:]) / np.var(log_returns)
                    var_ratios.append(var_ratio)
                
                # Estimate Hurst from variance ratios
                hurst_var = 0.5 + np.mean(var_ratios) / 2
                hurst_methods.append(hurst_var)
            except:
                pass
            
            # 3. Simple correlation of cumulative returns
            try:
                cumulative_returns = np.cumsum(log_returns)
                cum_corr = np.corrcoef(cumulative_returns[:-1], cumulative_returns[1:])[0, 1]
                hurst_methods.append(0.5 + (np.sign(cum_corr) * min(abs(cum_corr), 0.4)))
            except:
                pass
            
            # Aggregate estimates
            if hurst_methods:
                hurst_estimates.append(np.median(hurst_methods))
        
        except Exception as e:
            continue
    
    # Return median of estimates or default
    return np.median(hurst_estimates) if hurst_estimates else 0.5

def calculate_comprehensive_hurst(token):
    # Fetch data from DB for the last 24 hours
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=1)
    
    query = text(f"""
    WITH time_blocks AS (
        SELECT 
            generate_series(
                date_trunc('day', :start_time::timestamp) + INTERVAL '8 hours',
                date_trunc('day', :start_time::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_start,
            generate_series(
                date_trunc('day', :start_time::timestamp) + INTERVAL '8 hours',
                date_trunc('day', :start_time::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_end
    )
    SELECT 
        time_blocks.block_start AS time_block,
        final_price,
        created_at
    FROM time_blocks
    JOIN public.oracle_price_log ON 
        pair_name = :token AND
        created_at >= time_blocks.block_start AND
        created_at < time_blocks.block_end
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
                    'start_time': start_time
                }
            )
        
        if df.empty:
            st.warning(f"No data found for {token}")
            return None
        
        # Group by time blocks and calculate Hurst
        def process_block(block_data):
            if len(block_data) < 10:
                return np.nan
            
            prices = block_data['final_price'].values
            return calculate_multi_window_hurst(prices)
        
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
                LIMIT 100;
            """)
            
            result = connection.execute(query)
            data = result.fetchall()
            
            return [row[0] for row in data]
    
    except Exception as e:
        st.error(f"Error checking data availability: {e}")
        return []

# Main Streamlit app
st.title("Daily Hurst Table (Multi-Window Analysis)")
st.subheader("All Trading Pairs - Last 24 Hours")

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
        
        # Summary Statistics
        st.subheader("Market Regime Overview")
        
        # Prepare latest Hurst values
        latest_values = {}
        for token, df in token_results.items():
            if not df.empty and not df['Hurst'].isna().all():
                latest = df['Hurst'].iloc[-1]
                regime = df['regime'].iloc[-1]
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
                        st.markdown(f"- **{token}**: {value:.2f}")
                else:
                    st.markdown("*No tokens in this category*")
            
            with col2:
                st.markdown("### Random Walk Tokens")
                rw_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if 0.4 <= v <= 0.6]
                rw_tokens.sort(key=lambda x: x[1])  # Sort by Hurst value
                
                if rw_tokens:
                    for token, value, regime in rw_tokens:
                        st.markdown(f"- **{token}**: {value:.2f}")
                else:
                    st.markdown("*No tokens in this category*")
            
            with col3:
                st.markdown("### Trending Tokens")
                tr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v > 0.6]
                tr_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by Hurst value (descending)
                
                if tr_tokens:
                    for token, value, regime in tr_tokens:
                        st.markdown(f"- **{token}**: {value:.2f}")
                else:
                    st.markdown("*No tokens in this category*")
    else:
        st.warning("No tokens could be processed.")