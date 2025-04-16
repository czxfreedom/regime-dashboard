# Save this as pages/07_Market_Response_Analysis.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
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

# Fetch all available tokens from DB
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
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

# Check if BTC is in the selected tokens, if not add it
if "BTC" not in selected_tokens:
    selected_tokens = ["BTC"] + selected_tokens
    st.info("Added BTC to selection as it's required for ratio calculations")

# Function to generate aligned 30-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 30-minute time blocks for past 24 hours,
    aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
    """
    # Round down to the nearest 30-minute mark
    minute = current_time.minute
    nearest_30min = (minute // 30) * 30
    latest_complete_block_end = current_time.replace(minute=nearest_30min, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(2 * 24):  # 24 hours of 30-minute blocks (2 per hour)
        block_end = latest_complete_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Calculate ATR for a given DataFrame
def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) for a price series
    """
    df = df.copy()
    df['previous_close'] = df['price'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['previous_close'])
    df['low_close'] = abs(df['low'] - df['previous_close'])
    
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

# Calculate beta using numpy instead of statsmodels
def calculate_beta_numpy(token_returns, btc_returns):
    """
    Calculate the beta coefficient using numpy:
    beta = cov(token, btc) / var(btc)
    """
    # Drop NaN values
    clean_data = pd.concat([token_returns, btc_returns], axis=1).dropna()
    
    if len(clean_data) < 3:  # Need at least 3 data points for calculation
        return None, None, None
    
    x = clean_data.iloc[:, 1].values  # BTC returns
    y = clean_data.iloc[:, 0].values  # Token returns
    
    try:
        # Calculate covariance and variance
        covariance = np.cov(y, x)[0, 1]
        variance = np.var(x, ddof=1)
        
        if variance == 0:
            return None, None, None
            
        # Calculate beta
        beta = covariance / variance
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(y, x)[0, 1]
        r_squared = correlation ** 2
        
        # Calculate alpha (intercept term)
        mean_y = np.mean(y)
        mean_x = np.mean(x)
        alpha = mean_y - beta * mean_x
        
        return beta, alpha, r_squared
    except:
        return None, None, None

# Fetch price data and calculate metrics
@st.cache_data(ttl=600, show_spinner="Calculating metrics...")
def fetch_and_calculate_metrics(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days + 1)  # Extra day for ATR calculation
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Query needs to get OHLC data for proper ATR calculation
    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price AS price, 
        pair_name,
        -- You may need to adapt these fields based on your actual database schema
        -- If your DB doesn't store OHLC directly, you'll need to calculate them
        final_price AS high,
        final_price AS low,
        final_price AS close
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_name = '{token}';
    """
    try:
        print(f"[{token}] Executing query: {query}")
        df = pd.read_sql(query, engine)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No data found.")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Resample to get OHLC data at desired timeframe
        df_resampled = df['price'].resample(timeframe).ohlc()
        df_resampled['pair_name'] = token
        
        # Calculate returns
        df_resampled['returns'] = df_resampled['close'].pct_change() * 100  # percentage returns
        
        # Calculate ATR
        df_atr = calculate_atr(df_resampled, period=atr_periods)
        
        # Create a DataFrame with the results
        metrics_df = df_atr[['pair_name', 'atr', 'returns']].copy()
        metrics_df['original_datetime'] = metrics_df.index
        metrics_df['time_label'] = metrics_df.index.strftime('%Y-%m-%d %H:%M')
        
        # Calculate the 24-hour average ATR
        metrics_df['avg_24h_atr'] = metrics_df['atr'].mean()
        
        print(f"[{token}] Successful Metrics Calculation")
        return metrics_df
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

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
        return "Similar to BTC"
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
    st.write("Analyzing how volatile each token is compared to Bitcoin")
    
    if token_results and "BTC" in token_results:
        btc_atr = token_results["BTC"]
        
        # Create table data for ATR ratios
        ratio_table_data = {}
        for token, df in token_results.items():
            if token != "BTC":  # Skip BTC as we're comparing others to it
                # Merge with BTC ATR data on the time index
                merged_df = pd.merge(
                    df, 
                    btc_atr[['atr', 'time_label']], 
                    on='time_label', 
                    suffixes=('', '_btc')
                )
                
                # Calculate the ratio
                merged_df['atr_ratio'] = merged_df['atr'] / merged_df['atr_btc']
                
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
        
        # Reindex with the ordered times
        ratio_table = ratio_table.reindex(ordered_times)
        
        # Function to color cells based on ATR ratio
        def color_ratio_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0.5:  # Much less volatile than BTC
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Less volatile than BTC
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to BTC
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # More volatile than BTC
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Much more volatile than BTC
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        styled_table = ratio_table.style.applymap(color_ratio_cells).format("{:.2f}")
        st.markdown("## ATR Ratio Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:blue'>Much Less Volatile</span>, <span style='color:lightblue'>Less Volatile</span>, <span style='color:black'>Similar to BTC</span>, <span style='color:orange'>More Volatile</span>, <span style='color:red'>Much More Volatile</span>", unsafe_allow_html=True)
        st.markdown("Values shown as ratio of token's ATR to BTC's ATR within each 30-minute interval")
        st.dataframe(styled_table, height=700, use_container_width=True)
        
        # Create ranking table based on average ATR ratio
        st.subheader("ATR Ratio Ranking (24-Hour Average, Descending Order)")
        
        ranking_data = []
        for token, ratio_series in ratio_table_data.items():
            avg_ratio = ratio_series.mean()
            min_ratio = ratio_series.min()
            max_ratio = ratio_series.max()
            range_ratio = max_ratio - min_ratio
            std_ratio = ratio_series.std()
            cv_ratio = std_ratio / avg_ratio if avg_ratio > 0 else 0  # Coefficient of variation
            
            # Calculate time periods where token outperforms BTC the most
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
                title='Top 15 Tokens by Average ATR Ratio to BTC',
                labels={'Avg ATR Ratio': 'Average ATR Ratio (Token/BTC)', 'Token': 'Token'},
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
                st.markdown("""
                ### How to Read the ATR Ratio Matrix
                This matrix shows the ATR (Average True Range) of each cryptocurrency pair compared to Bitcoin's ATR over the last 24 hours using 30-minute intervals.
                
                **The ATR Ratio is calculated as:**
                ```
                ATR Ratio = Token's ATR / Bitcoin's ATR
                ```
                
                **What the ratios mean:**
                - **Ratio = 1.0**: The token has the same volatility as Bitcoin
                - **Ratio > 1.0**: The token is more volatile than Bitcoin
                - **Ratio < 1.0**: The token is less volatile than Bitcoin
                
                **Color coding:**
                - **Blue** (< 0.5): Much less volatile than Bitcoin
                - **Light Blue** (0.5 - 0.9): Less volatile than Bitcoin
                - **White** (0.9 - 1.1): Similar volatility to Bitcoin
                - **Orange** (1.1 - 2.0): More volatile than Bitcoin
                - **Red** (> 2.0): Much more volatile than Bitcoin
                
                ### Trading Applications
                
                **Breakout Identification:**
                When Bitcoin shows a significant price movement or breakout:
                
                1. Tokens with consistently high ATR ratios (greater than 1.5) are likely to show even larger moves
                2. Tokens with high "Outperform %" values consistently amplify Bitcoin's movements
                3. The "Range" value shows how much the token's behavior varies
                """)
        else:
            st.warning("No ranking data available.")
    else:
        st.error("BTC data is required for ATR ratio calculations. Please ensure BTC is selected and data is available.")

# TAB 2: BETA ANALYSIS
with tab2:
    st.header("Beta Analysis")
    st.write("Analyzing how much each token moves per 1% move in Bitcoin (accounting for correlation)")
    
    if token_results and "BTC" in token_results:
        btc_returns = token_results["BTC"]['returns']
        
        # Create table data for betas
        beta_table_data = {}
        beta_values = {}
        
        for token, df in token_results.items():
            if token != "BTC":  # Skip BTC as we're comparing others to it
                token_returns = df['returns']
                
                # Calculate rolling betas over a window of data points
                beta_windows = {}
                corr_windows = {}
                window_size = 12  # 6 hours of data (12 half-hour periods)
                
                for i in range(len(aligned_time_blocks) - window_size + 1):
                    end_idx = i
                    start_idx = i + window_size - 1
                    
                    if start_idx >= len(aligned_time_blocks):
                        continue
                        
                    window_start = aligned_time_blocks[start_idx][0]
                    window_end = aligned_time_blocks[end_idx][1]
                    
                    # Filter returns for the window
                    btc_window = btc_returns[(btc_returns.index >= window_start) & (btc_returns.index <= window_end)]
                    token_window = token_returns[(token_returns.index >= window_start) & (token_returns.index <= window_end)]
                    
                    # Calculate beta for the window using numpy
                    beta, alpha, r_squared = calculate_beta_numpy(token_window, btc_window)
                    
                    # Calculate correlation
                    if not token_window.empty and not btc_window.empty:
                        clean_data = pd.concat([token_window, btc_window], axis=1).dropna()
                        if len(clean_data) > 1:
                            corr = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1])
                        else:
                            corr = np.nan
                    else:
                        corr = np.nan
                    
                    # Store the beta for this time block
                    time_label = aligned_time_blocks[end_idx][2]
                    beta_windows[time_label] = beta if beta is not None else np.nan
                    corr_windows[time_label] = corr
                
                # Convert to series
                beta_series = pd.Series(beta_windows)
                beta_table_data[token] = beta_series
                
                # Calculate overall beta for the entire period
                overall_beta, overall_alpha, overall_r_squared = calculate_beta_numpy(token_returns, btc_returns)
                
                # Calculate overall correlation
                clean_data = pd.concat([token_returns, btc_returns], axis=1).dropna()
                if len(clean_data) > 1:
                    overall_corr = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1])
                else:
                    overall_corr = np.nan
                
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
            elif val < 0:  # Negative beta (moves opposite to BTC)
                return 'background-color: rgba(255, 0, 255, 0.7); color: white'  # Purple
            elif val < 0.5:  # Low beta
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Moderate beta
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to BTC
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # High beta
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Very high beta
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        styled_beta_table = beta_table.style.applymap(color_beta_cells).format("{:.2f}")
        st.markdown("## Beta Coefficient Table (6-hour rolling window, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:purple'>Negative Beta</span>, <span style='color:blue'>Low Beta</span>, <span style='color:lightblue'>Moderate Beta</span>, <span style='color:black'>Similar to BTC</span>, <span style='color:orange'>High Beta</span>, <span style='color:red'>Very High Beta</span>", unsafe_allow_html=True)
        st.markdown("Values shown as Beta coefficient (how much token moves per 1% move in BTC)")
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
                
            # Get rolling betas for this token if available
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
                title='Beta vs. Correlation with BTC',
                labels={
                    'Beta': 'Beta Coefficient',
                    'Correlation': 'Correlation with BTC'
                },
                color='R²',
                size='Beta Range',
                hover_name='Token',
                color_continuous_scale='Viridis'
            )
            
            # Add a vertical line at x=1 (same as BTC)
            fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Same as BTC")
            
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
            st.subheader("Alpha Analysis - Tokens that Outperform BTC")
            
            # Sort by Alpha
            alpha_df = beta_ranking_df.sort_values(by='Alpha (%)', ascending=False)
            
            # Create bar chart of top alphas
            fig3 = px.bar(
                alpha_df.head(15),
                x='Token',
                y='Alpha (%)',
                title='Top 15 Tokens by Alpha (Outperformance vs BTC)',
                color='Alpha (%)',
                color_continuous_scale='Greens'
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Beta explainer
            with st.expander("Understanding Beta Analysis"):
                st.markdown("""
                ### How to Read the Beta Coefficient Matrix
                
                This matrix shows how much each token moves relative to a 1% move in Bitcoin, taking into account the correlation between the token and Bitcoin.
                
                **The Beta Coefficient is calculated using covariance and variance:**
                ```
                Beta = Covariance(Token, Bitcoin) / Variance(Bitcoin)
                ```
                
                **What the beta values mean:**
                - **Beta = 1.0**: The token moves exactly the same as Bitcoin (1% when BTC moves 1%)
                - **Beta = 2.0**: The token moves twice as much as Bitcoin (2% when BTC moves 1%)
                - **Beta = 0.5**: The token moves half as much as Bitcoin (0.5% when BTC moves 1%)
                - **Beta < 0**: The token moves in the opposite direction to Bitcoin
                
                **Other important metrics:**
                - **Alpha**: The token's excess return over what would be predicted by Bitcoin's movements alone
                - **R²**: How well Bitcoin's movements explain the token's movements (higher means stronger relationship)
                - **Correlation**: Linear correlation between token and Bitcoin returns
                
                ### Trading Applications
                
                **For Breakout Trading:**
                1. Look for tokens with high Beta (>1.5) AND high Correlation (>0.7) for amplified moves in the same direction
                2. Tokens with high Beta but lower Correlation may move strongly but less predictably
                3. Tokens with high positive Alpha tend to outperform Bitcoin over time
                
                **Risk Management:**
                - Higher Beta tokens will experience larger drawdowns when Bitcoin falls
                - Beta Range shows how consistent the relationship is (lower means more consistent)
                """)
        else:
            st.warning("No beta data available.")
    else:
        st.error("BTC data is required for beta calculations. Please ensure BTC is selected and data is available.")