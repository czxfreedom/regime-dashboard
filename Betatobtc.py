# Save this as pages/07_Market_Response_Analysis.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Market Response Analysis",
    page_icon="📊",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("Market Response Analysis")
st.subheader("Analyzing how altcoins respond to Bitcoin movements")

# Global parameters
atr_periods = 14  # Standard ATR uses 14 periods
timeframe = "30min"  # Using 30-minute intervals as requested
lookback_days = 1  # 24 hours
expected_points = 48  # Expected data points per pair over 24 hours (2 per hour × 24 hours)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to get partition tables for a date range
def get_partition_tables(start_date, end_date, engine):
    """
    Get list of order book partition tables that need to be queried based on date range.
    Returns a list of table names (oracle_order_book_level_price_data_partition_YYYYMMDD)
    """
    # Convert to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str) and end_date:
        end_date = pd.to_datetime(end_date)
    elif end_date is None:
        end_date = datetime.now()
        
    # Ensure timezone is removed
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
        
    # Generate list of dates between start and end
    current_date = start_date
    dates = []
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # Create table names from dates
    table_names = [f"oracle_order_book_level_price_data_partition_{date}" for date in dates]
    
    # Verify which tables actually exist in the database
    existing_tables = []
    
    with engine.connect() as conn:
        for table in table_names:
            # Check if table exists
            query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                );
            """)
            
            result = conn.execute(query, {"table_name": table}).scalar()
            
            if result:
                existing_tables.append(table)
    
    if not existing_tables:
        print(f"Warning: No order book partition tables found for the date range {start_date.date()} to {end_date.date()}")
    else:
        print(f"Found {len(existing_tables)} order book partition tables: {', '.join(existing_tables)}")
        
    return existing_tables

# Function to fetch data from partition tables
def fetch_data_from_partitions(token, start_time_utc, end_time_utc, engine):
    """
    Fetch data from multiple partition tables based on date range.
    """
    # Get partition tables for the date range
    partition_tables = get_partition_tables(start_time_utc, end_time_utc, engine)
    
    # Query each table and combine results
    all_data = []
    
    with engine.connect() as conn:
        for table in partition_tables:
            query = f"""
            SELECT 
                created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
                final_price, 
                pair_name
            FROM public.{table}
            WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name = '{token}';
            """
            
            try:
                print(f"Querying {table} for {token}")
                df = pd.read_sql(query, conn)
                print(f"Found {len(df)} rows for {token} in {table}")
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"Error querying {table} for {token}: {e}")
    
    # Also try the main table if it exists
    try:
        query = f"""
        SELECT 
            created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
            final_price, 
            pair_name
        FROM public.oracle_price_log
        WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{token}';
        """
        
        with engine.connect() as conn:
            main_df = pd.read_sql(query, conn)
            if not main_df.empty:
                print(f"Found {len(main_df)} rows for {token} in oracle_price_log")
                all_data.append(main_df)
    except Exception as e:
        print(f"Error querying oracle_price_log: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates()
        return combined_df
    else:
        return pd.DataFrame()

# Fetch all available tokens from DB
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_all_tokens():
    # Get current time for date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days + 1)
    
    # Get partition tables
    partition_tables = get_partition_tables(start_time, end_time, engine)
    
    # Query each partition table for distinct pair names
    all_tokens = set()
    
    for table in partition_tables:
        try:
            query = f"""
            SELECT DISTINCT pair_name FROM public.{table}
            ORDER BY pair_name;
            """
            
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
                all_tokens.update(df['pair_name'].tolist())
        except Exception as e:
            print(f"Error fetching tokens from {table}: {e}")
    
    # Also check the main table
    try:
        query = """
        SELECT DISTINCT pair_name FROM public.oracle_price_log
        ORDER BY pair_name;
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            all_tokens.update(df['pair_name'].tolist())
    except Exception as e:
        print(f"Error fetching tokens from oracle_price_log: {e}")
    
    all_tokens_list = sorted(list(all_tokens))
    
    if not all_tokens_list:
        return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback
    
    return all_tokens_list

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

# Identify the BTC token if present
btc_token = next((t for t in selected_tokens if "BTC" in t), None)
if not btc_token and len(selected_tokens) > 0:
    btc_token = selected_tokens[0]  # Use the first token as reference if BTC not available
    st.info(f"Using {btc_token} as the reference token since BTC was not found")

# Function to generate aligned 30-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 30-minute time blocks for past 24 hours,
    aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
    """
    # Round down to the nearest 30-minute mark
    if current_time.minute < 30:
        # Round down to XX:00
        latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Round down to XX:30
        latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(48):  # 24 hours of 30-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Calculate ATR for a given DataFrame
def calculate_atr(ohlc_df, period=14):
    """
    Calculate the Average True Range (ATR) for a price series
    """
    df = ohlc_df.copy()
    
    # Calculate High-Low, High-Close(prev), and Low-Close(prev)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    
    # Calculate the True Range
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate the ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

# Calculate beta using numpy
def calculate_beta(token_returns, ref_returns):
    """
    Calculate the beta coefficient
    beta = cov(token, reference) / var(reference)
    """
    # Drop NaN values and align the series
    clean_data = pd.concat([token_returns, ref_returns], axis=1).dropna()
    
    if len(clean_data) < 3:  # Need at least 3 data points for meaningful calculation
        return None, None, None
    
    try:
        token_rets = clean_data.iloc[:, 0].values
        ref_rets = clean_data.iloc[:, 1].values
        
        # Calculate covariance and variance
        covariance = np.cov(token_rets, ref_rets, ddof=1)[0, 1]
        ref_variance = np.var(ref_rets, ddof=1)
        
        if ref_variance == 0:
            return None, None, None
            
        # Calculate beta
        beta = covariance / ref_variance
        
        # Calculate correlation
        correlation = np.corrcoef(token_rets, ref_rets)[0, 1]
        r_squared = correlation ** 2
        
        # Calculate alpha (intercept)
        alpha = np.mean(token_rets) - beta * np.mean(ref_rets)
        
        return beta, alpha, r_squared
    except Exception as e:
        print(f"Error calculating beta: {e}")
        return None, None, None

# Fetch price data and calculate metrics
@st.cache_data(ttl=600, show_spinner="Calculating metrics...")
def fetch_and_calculate_metrics(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days + 1)  # Extra day for calculations
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Query data from partitioned tables
    df = fetch_data_from_partitions(token, start_time_utc, end_time_utc, engine)
    
    if df.empty:
        print(f"[{token}] No data found.")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample to get OHLC data
    df_resampled = df['final_price'].resample(timeframe).ohlc().dropna()
    if df_resampled.empty:
        print(f"[{token}] No OHLC data after resampling.")
        return None
    
    # Calculate returns for beta calculation
    df_resampled['returns'] = df_resampled['close'].pct_change() * 100  # percentage returns
    
    # Calculate ATR
    df_atr = calculate_atr(df_resampled, period=atr_periods)
    
    # Create a DataFrame with the results
    metrics_df = df_atr[['open', 'high', 'low', 'close', 'atr', 'returns']].copy()
    metrics_df['token'] = token
    metrics_df['original_datetime'] = metrics_df.index
    metrics_df['time_label'] = metrics_df.index.strftime('%H:%M')
    
    # Calculate the 24-hour average ATR
    metrics_df['avg_24h_atr'] = metrics_df['atr'].mean()
    
    print(f"[{token}] Successful Metrics Calculation")
    return metrics_df

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate metrics for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_metrics(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create tabs for ATR Ratio and Beta analysis
tab1, tab2 = st.tabs(["ATR Ratio Analysis", "Beta Analysis"])

# Helper function to categorize volatility
def get_volatility_category(ratio):
    if ratio < 0.5:
        return "Much Less Volatile"
    elif ratio < 0.9:
        return "Less Volatile"
    elif ratio < 1.1:
        return "Similar to Reference"
    elif ratio < 2.0:
        return "More Volatile"
    else:
        return "Much More Volatile"

# Helper function to categorize beta
def get_beta_category(beta):
    if beta < 0.3:
        return "Very Low Response"
    elif beta < 0.7:
        return "Low Response"
    elif beta < 1.3:
        return "Medium Response"
    elif beta < 2.0:
        return "High Response"
    else:
        return "Very High Response"

# TAB 1: ATR RATIO ANALYSIS
with tab1:
    st.header("ATR Ratio Analysis")
    st.write(f"Analyzing how volatile each token is compared to {btc_token}")
    
    if token_results and btc_token in token_results:
        reference_df = token_results[btc_token]
        
        # Create table data for ATR ratios
        ratio_table_data = {}
        for token, df in token_results.items():
            if token != btc_token:  # Skip reference token as we're comparing others to it
                # Merge with reference token ATR data on the time index
                merged_df = pd.merge(
                    df[['time_label', 'atr']], 
                    reference_df[['time_label', 'atr']], 
                    on='time_label', 
                    suffixes=('', '_ref')
                )
                
                # Calculate the ratio
                merged_df['atr_ratio'] = merged_df['atr'] / merged_df['atr_ref']
                
                # Series with time_label as index and atr_ratio as values
                ratio_series = merged_df.set_index('time_label')['atr_ratio']
                ratio_table_data[token] = ratio_series
        
        # Create DataFrame with all token ratios
        ratio_table = pd.DataFrame(ratio_table_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(ratio_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)
        
        # Reindex with the ordered times if they exist
        if ordered_times:
            ratio_table = ratio_table.reindex(ordered_times)
        
        # Function to color cells based on ATR ratio
        def color_ratio_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0.5:  # Much less volatile than reference
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Less volatile than reference
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to reference
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # More volatile than reference
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Much more volatile than reference
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        styled_table = ratio_table.style.applymap(color_ratio_cells).format("{:.2f}")
        st.markdown(f"## ATR Ratio Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown(f"### Reference Token: {btc_token}")
        st.markdown("### Color Legend: <span style='color:blue'>Much Less Volatile</span>, <span style='color:lightblue'>Less Volatile</span>, <span style='color:black'>Similar to Reference</span>, <span style='color:orange'>More Volatile</span>, <span style='color:red'>Much More Volatile</span>", unsafe_allow_html=True)
        st.markdown(f"Values shown as ratio of token's ATR to {btc_token}'s ATR within each 30-minute interval")
        st.dataframe(styled_table, height=700, use_container_width=True)
        
        # Create ranking table based on average ATR ratio
        st.subheader("ATR Ratio Ranking (24-Hour Average, Descending Order)")
        
        ranking_data = []
        for token, ratio_series in ratio_table_data.items():
            if not ratio_series.empty:
                avg_ratio = ratio_series.mean()
                min_ratio = ratio_series.min()
                max_ratio = ratio_series.max()
                range_ratio = max_ratio - min_ratio
                std_ratio = ratio_series.std()
                cv_ratio = std_ratio / avg_ratio if avg_ratio > 0 else 0  # Coefficient of variation
                
                # Calculate time periods where token outperforms reference
                outperformance_periods = (ratio_series > 1.5).sum()
                outperformance_pct = (outperformance_periods / len(ratio_series)) * 100 if len(ratio_series) > 0 else 0
                
                ranking_data.append({
                    'Token': token,
                    'Avg ATR Ratio': round(avg_ratio, 2),
                    'Max ATR Ratio': round(max_ratio, 2),
                    'Min ATR Ratio': round(min_ratio, 2),
                    'Range': round(range_ratio, 2),
                    'Std Dev': round(std_ratio, 2),
                    'CoV': round(cv_ratio, 2),
                    'Outperform %': round(outperformance_pct, 1),
                    'Volatility Category': get_volatility_category(avg_ratio)
                })
        
        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            # Sort by average ATR ratio (high to low)
            ranking_df = ranking_df.sort_values(by='Avg ATR Ratio', ascending=False)
            # Add rank column
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
            
            # Reset the index to remove it
            ranking_df = ranking_df.reset_index(drop=True)
            
            # Display the styled dataframe
            st.dataframe(ranking_df, height=500, use_container_width=True)
            
            # Create breakout amplification chart
            st.subheader("Breakout Amplification Analysis")
            
            # Create a bar chart of tokens by average ATR ratio
            tokens_by_ratio = ranking_df.sort_values(by='Avg ATR Ratio', ascending=False).head(15)
            
            fig = px.bar(
                tokens_by_ratio, 
                x='Token', 
                y='Avg ATR Ratio', 
                title=f'Top 15 Tokens by Average ATR Ratio to {btc_token}',
                labels={'Avg ATR Ratio': f'Average ATR Ratio (Token/{btc_token})', 'Token': 'Token'},
                color='Avg ATR Ratio',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize ATR ratio variability
            tokens_by_variability = ranking_df.sort_values(by='CoV', ascending=False).head(15)
            
            fig = px.bar(
                tokens_by_variability, 
                x='Token', 
                y='CoV', 
                title='Top 15 Tokens by ATR Ratio Variability',
                labels={'CoV': 'Coefficient of Variation (Std/Mean)', 'Token': 'Token'},
                color='CoV',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # ATR Ratio explainer
            with st.expander("Understanding ATR Ratio Analysis"):
                st.markdown(f"""
                ### How to Read the ATR Ratio Matrix
                This matrix shows the ATR (Average True Range) of each cryptocurrency pair compared to {btc_token}'s ATR over the last 24 hours using 30-minute intervals.
                
                **The ATR Ratio is calculated as:**
                ```
                ATR Ratio = Token's ATR / {btc_token}'s ATR
                ```
                
                **What the ratios mean:**
                - **Ratio = 1.0**: The token has the same volatility as {btc_token}
                - **Ratio > 1.0**: The token is more volatile than {btc_token}
                - **Ratio < 1.0**: The token is less volatile than {btc_token}
                
                **Color coding:**
                - **Blue** (< 0.5): Much less volatile than {btc_token}
                - **Light Blue** (0.5 - 0.9): Less volatile than {btc_token}
                - **White** (0.9 - 1.1): Similar volatility to {btc_token}
                - **Orange** (1.1 - 2.0): More volatile than {btc_token}
                - **Red** (> 2.0): Much more volatile than {btc_token}
                
                ### Trading Applications
                
                **Breakout Identification:**
                When {btc_token} shows a significant price movement or breakout:
                
                1. Tokens with consistently high ATR ratios (greater than 1.5) are likely to show even larger moves
                2. Tokens with high "Outperform %" values consistently amplify {btc_token}'s movements
                3. The "Range" value shows how much the token's behavior varies
                """)
        else:
            st.warning("No ranking data available.")
    else:
        st.error(f"Reference token data ({btc_token}) is required for ATR ratio calculations. Please ensure it is selected and data is available.")

# TAB 2: BETA ANALYSIS
with tab2:
    st.header("Beta Analysis")
    st.write(f"Analyzing how much each token moves per 1% move in {btc_token} (accounting for correlation)")
    
    if token_results and btc_token in token_results:
        reference_returns = token_results[btc_token]['returns']
        
        # Create table data for betas
        beta_table_data = {}
        beta_values = {}
        
        for token, df in token_results.items():
            if token != btc_token:  # Skip reference token as we're comparing others to it
                token_returns = df['returns']
                
                # Calculate overall beta for the entire period
                overall_beta, overall_alpha, overall_r_squared = calculate_beta(token_returns, reference_returns)
                
                # Calculate correlation
                clean_data = pd.concat([token_returns, reference_returns], axis=1).dropna()
                overall_corr = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1]) if len(clean_data) > 1 else np.nan
                
                # Calculate rolling betas for each time period
                beta_by_time = {}
                
                # Group data by time label (hour:minute)
                time_groups = df.groupby('time_label')
                
                for time_label, group in time_groups:
                    # Get reference data for the same time period
                    ref_group = token_results[btc_token][token_results[btc_token]['time_label'] == time_label]
                    
                    if not group.empty and not ref_group.empty:
                        # Calculate beta for this time period
                        period_beta, _, _ = calculate_beta(group['returns'], ref_group['returns'])
                        beta_by_time[time_label] = period_beta
                
                # Convert to series
                beta_series = pd.Series(beta_by_time)
                beta_table_data[token] = beta_series
                
                # Store overall metrics
                beta_values[token] = {
                    'beta': overall_beta,
                    'alpha': overall_alpha,
                    'r_squared': overall_r_squared,
                    'correlation': overall_corr
                }
        
        # Create DataFrame with all token betas
        beta_table = pd.DataFrame(beta_table_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(beta_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # Reindex with the ordered times if they exist
        if ordered_times:
            beta_table = beta_table.reindex(ordered_times)
        
        # Function to color cells based on beta value
        def color_beta_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0:  # Negative beta (moves opposite to reference)
                return 'background-color: rgba(255, 0, 255, 0.7); color: white'  # Purple
            elif val < 0.5:  # Low beta
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Moderate beta
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to reference
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # High beta
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Very high beta
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        styled_beta_table = beta_table.style.applymap(color_beta_cells).format("{:.2f}")
        st.markdown(f"## Beta Coefficient Table (30-minute intervals, Last 24 hours, Singapore Time)")
        st.markdown(f"### Reference Token: {btc_token}")
        st.markdown("### Color Legend: <span style='color:purple'>Negative Beta</span>, <span style='color:blue'>Low Beta</span>, <span style='color:lightblue'>Moderate Beta</span>, <span style='color:black'>Similar to Reference</span>, <span style='color:orange'>High Beta</span>, <span style='color:red'>Very High Beta</span>", unsafe_allow_html=True)
        st.markdown(f"Values shown as Beta coefficient (how much token moves per 1% move in {btc_token})")
        st.dataframe(styled_beta_table, height=700, use_container_width=True)
        
        # Create ranking table based on overall beta
        st.subheader("Beta Coefficient Ranking (24-Hour, Descending Order)")
        
        beta_ranking_data = []
        for token, values in beta_values.items():
            beta = values['beta']
            r_squared = values['r_squared']
            alpha = values['alpha']
            correlation = values['correlation']
            
            # Skip if beta calculation failed
            if beta is None:
                continue
                
            # Get statistics for this token if available
            if token in beta_table_data:
                rolling_betas = beta_table_data[token].dropna()
                max_beta = rolling_betas.max() if not rolling_betas.empty else np.nan
                min_beta = rolling_betas.min() if not rolling_betas.empty else np.nan
                beta_range = max_beta - min_beta if not np.isnan(max_beta) and not np.isnan(min_beta) else np.nan
                beta_std = rolling_betas.std() if not rolling_betas.empty else np.nan
            else:
                max_beta = min_beta = beta_range = beta_std = np.nan
            
            beta_ranking_data.append({
                'Token': token,
                'Beta': round(beta, 2),
                'Alpha (%)': round(alpha, 2) if alpha is not None else np.nan,
                'R²': round(r_squared, 2) if r_squared is not None else np.nan,
                'Correlation': round(correlation, 2),
                'Max Beta': round(max_beta, 2),
                'Min Beta': round(min_beta, 2),
                'Beta Range': round(beta_range, 2),
                'Beta Std Dev': round(beta_std, 2),
                'Response Category': get_beta_category(beta)
            })
        
        if beta_ranking_data:
            beta_ranking_df = pd.DataFrame(beta_ranking_data)
            # Sort by beta (high to low)
            beta_ranking_df = beta_ranking_df.sort_values(by='Beta', ascending=False)
            # Add rank column
            beta_ranking_df.insert(0, 'Rank', range(1, len(beta_ranking_df) + 1))
            
            # Reset the index to remove it
            beta_ranking_df = beta_ranking_df.reset_index(drop=True)
            
            # Display the styled dataframe
            st.dataframe(beta_ranking_df, height=500, use_container_width=True)
            
            # Create visualization
            st.subheader("Beta vs. Correlation Analysis")
            
            # Scatter plot of beta vs correlation
            fig = px.scatter(
                beta_ranking_df,
                x='Beta',
                y='Correlation',
                title=f'Beta vs. Correlation with {btc_token}',
                labels={
                    'Beta': 'Beta Coefficient',
                    'Correlation': f'Correlation with {btc_token}'
                },
                color='R²',
                size='Beta Range',
                hover_name='Token',
                color_continuous_scale='Viridis'
            )
            
            # Add a vertical line at x=1 (same as reference)
            fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Same as Reference")
            
            # Add a horizontal line at y=0 (no correlation)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Correlation")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bar chart of top betas
            fig2 = px.bar(
                beta_ranking_df.head(15),
                x='Token',
                y='Beta',
                title='Top 15 Tokens by Beta Coefficient',
                color='Beta',
                color_continuous_scale='Reds'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Alpha Analysis
            st.subheader("Alpha Analysis - Tokens that Outperform Reference")
            
            # Sort by Alpha
            alpha_df = beta_ranking_df.sort_values(by='Alpha (%)', ascending=False)
            
            # Create bar chart of top alphas
            fig3 = px.bar(
                alpha_df.head(15),
                x='Token',
                y='Alpha (%)',
                title=f'Top 15 Tokens by Alpha (Outperformance vs {btc_token})',
                color='Alpha (%)',
                color_continuous_scale='Greens'
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Beta explainer
            with st.expander("Understanding Beta Analysis"):
                st.markdown(f"""
                ### How to Read the Beta Coefficient Matrix
                
                This matrix shows how much each token moves relative to a 1% move in {btc_token}, taking into account the correlation between the token and {btc_token}.
                
                **The Beta Coefficient is calculated using covariance and variance:**
                ```
                Beta = Covariance(Token, {btc_token}) / Variance({btc_token})
                ```
                
                **What the beta values mean:**
                - **Beta = 1.0**: The token moves exactly the same as {btc_token} (1% when it moves 1%)
                - **Beta = 2.0**: The token moves twice as much as {btc_token} (2% when it moves 1%)
                - **Beta = 0.5**: The token moves half as much as {btc_token} (0.5% when it moves 1%)
                - **Beta < 0**: The token moves in the opposite direction to {btc_token}
                
                **Other important metrics:**
                - **Alpha**: The token's excess return over what would be predicted by {btc_token}'s movements alone
                - **R²**: How well {btc_token}'s movements explain the token's movements (higher means stronger relationship)
                - **Correlation**: Linear correlation between token and {btc_token} returns
                
                ### Trading Applications
                
                **For Breakout Trading:**
                1. Look for tokens with high Beta (>1.5) AND high Correlation (>0.7) for amplified moves in the same direction
                2. Tokens with high Beta but lower Correlation may move strongly but less predictably
                3. Tokens with high positive Alpha tend to outperform {btc_token} over time
                
                **Risk Management:**
                - Higher Beta tokens will experience larger drawdowns when {btc_token} falls
                - Beta Range shows how consistent the relationship is (lower means more consistent)
                """)
        else:
            st.warning("No beta data available.")
    else:
        st.error(f"Reference token data ({btc_token}) is required for beta calculations. Please ensure it is selected and data is available.")