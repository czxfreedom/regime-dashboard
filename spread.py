# Save this as pages/06_Exchange_Fee_Comparison.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Exchange Fee Comparison",
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
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Exchange Fee Comparison (10min Intervals)")
st.subheader("Transaction Fees Across Exchanges - Last 24 Hours (Singapore Time)")

# Define parameters
lookback_days = 1  # 24 hours
interval_minutes = 10  # 10-minute intervals
singapore_timezone = pytz.timezone('Asia/Singapore')

# Correct exchange names from the database
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "mexcFuture", "okxFuture", "surfFuture"]
exchanges_display = {
    "binanceFuture": "Binance",
    "gateFuture": "Gate",
    "hyperliquidFuture": "Hyperliquid",
    "mexcFuture": "MEXC",
    "okxFuture": "OKX",
    "surfFuture": "SurfFuture"
}

# Fetch all available tokens from DB
@st.cache_data(show_spinner="Fetching tokens...")
def fetch_all_tokens():
    query = """
    SELECT DISTINCT pair_name 
    FROM oracle_exchange_fee 
    ORDER BY pair_name
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default fallback

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

# Function to convert time string to sortable minutes value
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Fetch fee data for a token with proper interval calculation
@st.cache_data(ttl=600, show_spinner="Calculating exchange fees...")
def fetch_fee_data(token):
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # Query to get the fee data
        query = f"""
        SELECT 
            time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
            pair_name,
            source,
            total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name = '{token}'
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        ORDER BY 
            timestamp ASC,
            source
        """
        
        print(f"Executing query for {token}...")
        df = pd.read_sql(query, engine)
        print(f"Query executed for {token}. Results: {len(df)} rows")
        
        if df.empty:
            print(f"[{token}] No fee data found.")
            return None, None

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Debug: Check what data we got
        print(f"[{token}] Data fetched. Shape: {df.shape}")
        print(f"[{token}] Unique sources: {df['source'].unique()}")
        if not df.empty:
            print(f"[{token}] First few rows: {df.head()}")
        else:
            print(f"[{token}] DataFrame is empty")
        
        # Prepare data structure for interval processing
        fee_data = {}
        avg_data = []
        
        # Process each exchange separately
        for exchange in exchanges:
            exchange_df = df[df['source'] == exchange].copy()
            if not exchange_df.empty:
                exchange_df = exchange_df.set_index('timestamp')
                
                # First, resample to 1-minute to fill any gaps
                minute_df = exchange_df['total_fee'].resample('1min').sum().fillna(0)
                
                # Now create 10-minute rolling windows and calculate averages
                # For each point at XX:X0, average the previous 10 minutes (including the current minute)
                rolling_mean = minute_df.rolling(window=interval_minutes).mean()
                
                # Select only the points at 10-minute intervals (XX:00, XX:10, XX:20, etc.)
                interval_points = rolling_mean[rolling_mean.index.minute % interval_minutes == 0]
                
                if not interval_points.empty:
                    # Get last 24 hours
                    last_24h = interval_points.iloc[-144:]  # 144 10-minute intervals in 24 hours
                    fee_data[exchanges_display[exchange]] = last_24h
                    
                    # Prepare data for average calculation
                    exchange_avg_df = pd.DataFrame(last_24h)
                    exchange_avg_df['source'] = exchange
                    exchange_avg_df['pair_name'] = token
                    exchange_avg_df.reset_index(inplace=True)
                    avg_data.append(exchange_avg_df)
                
        # If no valid data found for any exchange
        if not fee_data:
            print(f"[{token}] No valid data after processing.")
            return None, None
            
        # Calculate average fee across all exchanges
        if avg_data:
            all_exchanges_df = pd.concat(avg_data)
            all_exchanges_df = all_exchanges_df.set_index('timestamp')
            avg_fee = all_exchanges_df.groupby(all_exchanges_df.index)['total_fee'].mean()
            fee_data['Average'] = avg_fee
            
        # Create time labels
        result_df = pd.DataFrame(fee_data)
        result_df['time_label'] = result_df.index.strftime('%H:%M')
        
        # Calculate daily average for each exchange
        daily_avgs = {}
        for exchange in result_df.columns:
            if exchange != 'time_label':
                daily_avgs[exchange] = result_df[exchange].mean()
        
        return result_df, daily_avgs
            
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate fees for each token
token_results = {}
token_daily_avgs = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result, daily_avgs = fetch_fee_data(token)
        if result is not None and daily_avgs is not None:
            token_results[token] = result
            token_daily_avgs[token] = daily_avgs
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Color function for fee values
def color_fee(val):
    if pd.isna(val):
        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
    elif val < 0.01:  # Very low fee
        return 'background-color: rgba(0, 200, 0, 0.7); color: black'
    elif val < 0.05:  # Low fee
        return 'background-color: rgba(100, 255, 100, 0.7); color: black'
    elif val < 0.1:  # Medium fee
        return 'background-color: rgba(255, 255, 0, 0.7); color: black'
    elif val < 0.5:  # High fee
        return 'background-color: rgba(255, 165, 0, 0.7); color: black'
    else:  # Very high fee
        return 'background-color: rgba(255, 0, 0, 0.7); color: white'

# Color function for daily average cells (different color scheme)
def color_daily_avg(val):
    if pd.isna(val):
        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
    elif val < 0.01:  # Excellent
        return 'background-color: rgba(75, 0, 130, 0.7); color: white'  # Purple
    elif val < 0.05:  # Very good
        return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
    elif val < 0.1:  # Good
        return 'background-color: rgba(0, 128, 128, 0.7); color: white'  # Teal
    elif val < 0.5:  # Average
        return 'background-color: rgba(210, 105, 30, 0.7); color: white'  # Brown
    else:  # Poor
        return 'background-color: rgba(128, 0, 0, 0.7); color: white'  # Maroon

# Create a consolidated summary table at the top
if token_daily_avgs:
    st.markdown("## Overall Daily Average Fees by Token and Exchange")
    
    # Create a DataFrame to hold all daily averages
    all_averages_data = []
    
    for token, averages in token_daily_avgs.items():
        row_data = {'Token': token}
        
        # Add each exchange's average
        for exchange, avg in averages.items():
            if exchange != 'Average':  # Skip the overall average for now
                row_data[exchange] = avg
        
        # Calculate and add the non-surf average (average excluding 'SurfFuture')
        exchange_avgs = [v for k, v in averages.items() 
                         if k != 'Average' and k != 'SurfFuture' and not pd.isna(v)]
        if exchange_avgs:
            row_data['Non-Surf Avg'] = sum(exchange_avgs) / len(exchange_avgs)
        else:
            row_data['Non-Surf Avg'] = float('nan')
            
        all_averages_data.append(row_data)
    
    # Create DataFrame and sort by token name
    all_averages_df = pd.DataFrame(all_averages_data)
    if not all_averages_df.empty and 'Token' in all_averages_df.columns:
        all_averages_df = all_averages_df.sort_values(by='Token')
        
        # Ensure all exchange columns are present, fill with NaN if missing
        exchange_cols = []
        for ex in exchanges:
            display_name = exchanges_display[ex]
            if display_name != 'SurfFuture':
                exchange_cols.append(display_name)
        exchange_cols.append('Non-Surf Avg')
        
        for col in exchange_cols:
            if col not in all_averages_df.columns:
                all_averages_df[col] = float('nan')
        
        # Reorder columns to put Token first, then exchanges
        cols_order = ['Token'] + [col for col in exchange_cols if col in all_averages_df.columns]
        all_averages_df = all_averages_df[cols_order]
        
        # Format numbers and apply styling
        styled_averages = all_averages_df.style.applymap(
            color_daily_avg, 
            subset=[col for col in exchange_cols if col in all_averages_df.columns]
        )
        
        # Display the summary table
        st.dataframe(styled_averages, height=min(600, 80 + 35 * len(all_averages_df)), use_container_width=True)
        
        # Calculate and display the best exchange based on average fees
        exchange_overall_avgs = {}
        for col in exchange_cols:
            if col != 'Non-Surf Avg' and col in all_averages_df.columns:
                avg_value = all_averages_df[col].mean()
                if not pd.isna(avg_value):
                    exchange_overall_avgs[col] = avg_value
        
        if exchange_overall_avgs:
            best_exchange = min(exchange_overall_avgs.items(), key=lambda x: x[1])
            st.info(f"Best exchange overall based on average fees: **{best_exchange[0]}** (Average fee: {best_exchange[1]:.4f})")

        # Add some explanation
        st.markdown("""
        This table shows the daily average fee for each token across different exchanges. 
        The 'Non-Surf Avg' column shows the average fee excluding SurfFuture.
        
        **Color coding**:
        - **Purple** (< 0.01): Excellent
        - **Blue** (0.01-0.05): Very good
        - **Teal** (0.05-0.1): Good
        - **Brown** (0.1-0.5): Average
        - **Maroon** (> 0.5): Poor
        """)
        
        st.markdown("---")  # Add a separator before individual token tables
        st.markdown("## Detailed Analysis by Token")
        st.markdown("Each token has its own table showing fees at 10-minute intervals.")

# Create display for each token
if token_results:
    # For each token, create a table showing all exchanges
    for token, df in token_results.items():
        st.markdown(f"## {token} Exchange Fee Comparison")
        
        # Set the time_label as index to display in rows
        table_df = df.set_index('time_label')
        
        # Sort time labels
        all_times = sorted(table_df.index, key=time_to_minutes, reverse=True)
        table_df = table_df.reindex(all_times)
        
        # Format with 4 decimal places
        formatted_df = table_df.applymap(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        # Apply styling
        styled_df = table_df.style.applymap(color_fee)
        
        # Display the table
        st.dataframe(styled_df, height=500, use_container_width=True)
        
        # Create a summary for this token
        st.markdown("### Exchange Comparison Summary")
        
        # Get the daily averages for this token
        daily_avgs = token_daily_avgs[token]
        
        summary_data = {}
        for column in table_df.columns:
            col_data = table_df[column].dropna()
            if not col_data.empty:
                summary_data[column] = {
                    'Daily Average Fee': daily_avgs[column],
                    'Min Fee': col_data.min(),
                    'Max Fee': col_data.max(),
                    'Data Points': len(col_data)
                }
                
        if summary_data:
            summary_df = pd.DataFrame(summary_data).T
            summary_df = summary_df.round(4)
                
            # Reset index to display exchange names as a column
            summary_df = summary_df.reset_index()
            summary_df.columns = ['Exchange', 'Daily Average Fee', 'Min Fee', 'Max Fee', 'Data Points']
            
            # Style the summary table with special colors for the daily average
            styled_summary = summary_df.style.applymap(
                color_daily_avg, 
                subset=['Daily Average Fee']
            )
            
            # Display the summary table
            st.dataframe(styled_summary, height=200, use_container_width=True)
            
            # Calculate best exchange (lowest average fee)
            exchanges_in_data = [ex for ex in summary_data.keys() if ex != 'Average']
            if exchanges_in_data:
                best_exchange = min(
                    exchanges_in_data,
                    key=lambda x: summary_data[x]['Daily Average Fee']
                )
                st.info(f"Best exchange for {token} based on daily average fee: **{best_exchange}**")
        
        st.markdown("---")  # Add a separator between tokens
    
    # Create an overall comparison of all tokens
    st.markdown("## Overall Token Fee Comparison")
    
    all_token_averages = {}
    for token, avgs in token_daily_avgs.items():
        if 'Average' in avgs:
            all_token_averages[token] = avgs['Average']
    
    if all_token_averages:
        # Create and sort the overall comparison DataFrame
        overall_df = pd.DataFrame.from_dict(all_token_averages, orient='index', columns=['Daily Average Fee'])
        overall_df = overall_df.sort_values(by='Daily Average Fee')
        
        # Reset index to display token names as a column
        overall_df = overall_df.reset_index()
        overall_df.columns = ['Token', 'Daily Average Fee']
        
        # Style the overall table
        styled_overall = overall_df.style.applymap(
            color_daily_avg, 
            subset=['Daily Average Fee']
        )
        
        # Display the overall comparison
        st.dataframe(styled_overall, height=400, use_container_width=True)
        
        # Create a bar chart to visualize the comparison
        tokens = list(all_token_averages.keys())
        values = list(all_token_averages.values())
        
        # Sort for better visualization
        sorted_data = sorted(zip(tokens, values), key=lambda x: x[1])
        sorted_tokens, sorted_values = zip(*sorted_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_tokens,
                y=sorted_values,
                marker_color='lightseagreen'
            )
        ])
        
        fig.update_layout(
            title="Daily Average Fee Comparison Across Tokens",
            xaxis_title="Token",
            yaxis_title="Daily Average Fee",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.warning("No valid fee data available for the selected tokens.")
    # Add more diagnostic information
    st.error("Please check the following:")
    st.write("1. Make sure the oracle_exchange_fee table has data for the selected time period")
    st.write("2. Verify the exchange names and column names in the database")
    st.write("3. Check the logs for more detailed error messages")

with st.expander("Understanding the Exchange Fee Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the average fee rates for each exchange and trading pair over the last 24 hours at 10-minute intervals.
    
    **Fee Rate Calculation**: 
    Each value represents the average fee over a 10-minute window. For example, the value at 15:20 is the average of fees from 15:11 to 15:20.
    
    **Time Intervals**:
    Data points occur at 10-minute marks (XX:00, XX:10, XX:20, XX:30, XX:40, XX:50).
    
    **Color coding for intervals**:
    - **Dark Green** (< 0.01): Very low fees
    - **Light Green** (0.01-0.05): Low fees
    - **Yellow** (0.05-0.1): Medium fees
    - **Orange** (0.1-0.5): High fees
    - **Red** (> 0.5): Very high fees
    
    **Color coding for daily averages** (different color scheme):
    - **Purple** (< 0.01): Excellent
    - **Blue** (0.01-0.05): Very good
    - **Teal** (0.05-0.1): Good
    - **Brown** (0.1-0.5): Average
    - **Maroon** (> 0.5): Poor
    
    **Average Column**: 
    The "Average" column represents the mean fee rate across all exchanges for each time period.
    
    **Summary Table**:
    For each token, a summary table shows:
    - Daily average fee across all time periods
    - Minimum and maximum fees observed
    - Number of data points available
    
    **Missing values (light gray cells)** indicate periods where no data was available for that exchange.
    """)