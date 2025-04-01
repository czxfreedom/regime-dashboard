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

# Apply some custom CSS for better styling
st.markdown("""
<style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
    }
    .header-style {
        font-size:28px !important;
        font-weight: bold;
        color: #1E88E5;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .subheader-style {
        font-size:22px !important;
        font-weight: bold;
        color: #1976D2;
        padding: 5px 0;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="header-style">Exchange Fee Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">Transaction Fees Across All Exchanges - Last 24 Hours (Singapore Time)</div>', unsafe_allow_html=True)

# Define parameters
lookback_days = 1  # 24 hours
interval_minutes = 10  # 10-minute intervals
singapore_timezone = pytz.timezone('Asia/Singapore')

# Correct exchange names from the database - reordered to put SurfFuture last
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "okxFuture", "mexcFuture", "surfFuture"]
exchanges_display = {
    "binanceFuture": "Binance",
    "gateFuture": "Gate",
    "hyperliquidFuture": "Hyperliquid",
    "okxFuture": "OKX",
    "mexcFuture": "MEXC",
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
st.markdown('<div class="medium-font">Select Tokens to Compare</div>', unsafe_allow_html=True)
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
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Function to convert time string to sortable minutes value
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Fetch fee data for each token over time
@st.cache_data(ttl=600, show_spinner="Calculating exchange fees...")
def fetch_token_fee_data(token):
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
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None, None

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
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
        print(f"[{token}] Error processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Fetch summary fee data for all tokens
@st.cache_data(ttl=600, show_spinner="Calculating total fees...")
def fetch_summary_fee_data(token):
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
            source,
            SUM(total_fee) as total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name = '{token}'
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        GROUP BY 
            source
        ORDER BY 
            source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        # Process each exchange separately
        fee_data = {}
        
        for _, row in df.iterrows():
            exchange = row['source']
            if exchange in exchanges:
                fee_data[exchanges_display[exchange]] = row['total_fee']
        
        # If no valid data found for any exchange
        if not fee_data:
            return None
            
        # Calculate average fee across all exchanges (excluding SurfFuture)
        non_surf_fees = [v for k, v in fee_data.items() if k != 'SurfFuture']
        if non_surf_fees:
            fee_data['Avg (Non-Surf)'] = sum(non_surf_fees) / len(non_surf_fees)
        
        return fee_data
            
    except Exception as e:
        print(f"[{token}] Error processing summary: {e}")
        import traceback
        traceback.print_exc()
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate fees for each token
token_summary_results = {}
token_detailed_results = {}
token_daily_avgs = {}

for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress(i / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        
        # Get summary data for overview table
        summary_result = fetch_summary_fee_data(token)
        if summary_result is not None:
            token_summary_results[token] = summary_result
        
        # Get detailed data for individual token tables
        detailed_result, daily_avgs = fetch_token_fee_data(token)
        if detailed_result is not None and daily_avgs is not None:
            token_detailed_results[token] = detailed_result
            token_daily_avgs[token] = daily_avgs
            
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_summary_results)}/{len(selected_tokens)} tokens successfully")

# Create a consolidated summary table at the top
if token_summary_results:
    st.markdown('<div class="header-style">Overall Exchange Fee Comparison</div>', unsafe_allow_html=True)
    
    # Create a DataFrame to hold all fees
    all_fees_data = []
    
    for token, fees in token_summary_results.items():
        row_data = {'Token': token}
        row_data.update(fees)
        all_fees_data.append(row_data)
    
    # Create DataFrame and sort by token name
    all_fees_df = pd.DataFrame(all_fees_data)
    
    if not all_fees_df.empty and 'Token' in all_fees_df.columns:
        all_fees_df = all_fees_df.sort_values(by='Token')
        
        # Find if we need to scale the values for better readability
        fee_values = all_fees_df.select_dtypes(include=[np.number])
        mean_fee = fee_values.mean().mean()
        
        # Determine scale factor based on mean fee value
        scale_factor = 1
        scale_label = ""
        
        if mean_fee < 0.001:
            scale_factor = 1000
            scale_label = "× 1,000"
        elif mean_fee < 0.0001:
            scale_factor = 10000
            scale_label = "× 10,000"
        elif mean_fee < 0.00001:
            scale_factor = 100000
            scale_label = "× 100,000"
        
        # Apply scaling if needed
        if scale_factor > 1:
            for col in all_fees_df.columns:
                if col != 'Token' and all_fees_df[col].dtype != 'object':
                    all_fees_df[col] = all_fees_df[col] * scale_factor
            
            st.markdown(f"<div class='info-box'><b>Note:</b> All fee values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
        
        # Make sure columns are in the desired order with SurfFuture at the end
        desired_order = ['Token']
        for ex in exchanges:
            display_name = exchanges_display[ex]
            if display_name in all_fees_df.columns:
                desired_order.append(display_name)
        if 'Avg (Non-Surf)' in all_fees_df.columns:
            desired_order.append('Avg (Non-Surf)')
        
        # Reorder columns according to the specified order
        ordered_columns = [col for col in desired_order if col in all_fees_df.columns]
        all_fees_df = all_fees_df[ordered_columns]
        
        # Round values to 2 decimal places for display
        for col in all_fees_df.columns:
            if col != 'Token' and all_fees_df[col].dtype != 'object':
                all_fees_df[col] = all_fees_df[col].round(2)
        
        # Display the summary table with dynamic height to show all tokens without scrolling
        token_count = len(all_fees_df)
        table_height = max(100 + 35 * token_count, 200)  # Minimum height of 200px
        st.dataframe(all_fees_df, height=table_height, use_container_width=True)
        
        # Calculate and display the best exchange based on average fees
        exchange_overall_avgs = {}
        for col in all_fees_df.columns:
            if col not in ['Token', 'Avg (Non-Surf)']:
                avg_value = all_fees_df[col].mean()
                if not pd.isna(avg_value):
                    exchange_overall_avgs[col] = avg_value
        
        if exchange_overall_avgs:
            best_exchange = min(exchange_overall_avgs.items(), key=lambda x: x[1])
            st.info(f"🏆 **Best exchange overall**: {best_exchange[0]} (Average fee: {best_exchange[1]:.2f})")

        # Add visualization - Bar chart comparing average fees by exchange
        st.markdown('<div class="subheader-style">Average Fee by Exchange</div>', unsafe_allow_html=True)
        
        avg_by_exchange = {}
        for col in all_fees_df.columns:
            if col not in ['Token', 'Avg (Non-Surf)']:
                avg_by_exchange[col] = all_fees_df[col].mean()
        
        if avg_by_exchange:
            # Sort exchanges by average fee
            sorted_exchanges = sorted(avg_by_exchange.items(), key=lambda x: x[1])
            exchanges_sorted = [x[0] for x in sorted_exchanges]
            fees_sorted = [x[1] for x in sorted_exchanges]
            
            # Create a colorful bar chart
            colors = ['#1a9850', '#66bd63', '#fee08b', '#f46d43', '#d73027']
            exchange_colors = [colors[min(i, len(colors)-1)] for i in range(len(sorted_exchanges))]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=exchanges_sorted,
                    y=fees_sorted,
                    marker_color=exchange_colors,
                    text=[f"{x:.2f}" for x in fees_sorted],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Average Fee Comparison Across Exchanges {scale_label}",
                xaxis_title="Exchange",
                yaxis_title=f"Average Fee {scale_label}",
                height=400,
                font=dict(size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add visualization - Best exchange for each token
        st.markdown('<div class="subheader-style">Best Exchange by Token</div>', unsafe_allow_html=True)
        
        best_exchanges = {}
        for idx, row in all_fees_df.iterrows():
            fee_cols = [c for c in row.index if c != 'Token' and c != 'Avg (Non-Surf)']
            if fee_cols:
                fees = row[fee_cols]
                min_fee = fees.min()
                best_ex = fees.idxmin()
                best_exchanges[row['Token']] = (best_ex, min_fee)
        
        # Count the number of "wins" for each exchange
        exchange_wins = {}
        for ex in [c for c in all_fees_df.columns if c != 'Token' and c != 'Avg (Non-Surf)']:
            exchange_wins[ex] = sum(1 for _, (best_ex, _) in best_exchanges.items() if best_ex == ex)
        
        # Create a pie chart of wins
        fig = go.Figure(data=[go.Pie(
            labels=list(exchange_wins.keys()),
            values=list(exchange_wins.values()),
            textinfo='label+percent',
            marker=dict(colors=[
                '#1a9850', '#66bd63', '#fee08b', '#f46d43', '#d73027', '#8c510a'
            ]),
            hole=.3
        )])
        
        fig.update_layout(
            title="Exchange with Lowest Fees (Number of Tokens)",
            height=400,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Display individual token tables with minimal formatting        
if token_detailed_results:
    st.markdown('<div class="header-style">Detailed Analysis by Token</div>', unsafe_allow_html=True)
    
    # For each token, create a table showing all exchanges
    for token, df in token_detailed_results.items():
        st.markdown(f"## {token} Exchange Fee Comparison")
        
        # Set the time_label as index to display in rows
        table_df = df.copy()
        table_df = table_df.set_index('time_label')
        
        # Sort time labels
        all_times = sorted(table_df.index, key=time_to_minutes, reverse=True)
        table_df = table_df.reindex(all_times)
        
        # Find if we need to scale the values for better readability
        mean_fee = table_df.mean().mean()
        
        # Determine scale factor based on mean fee value
        scale_factor = 1
        scale_label = ""
        
        if mean_fee < 0.001:
            scale_factor = 1000
            scale_label = "× 1,000"
        elif mean_fee < 0.0001:
            scale_factor = 10000
            scale_label = "× 10,000"
        elif mean_fee < 0.00001:
            scale_factor = 100000
            scale_label = "× 100,000"
        
        # Apply scaling if needed
        if scale_factor > 1:
            for col in table_df.columns:
                table_df[col] = table_df[col] * scale_factor
            st.markdown(f"<div class='info-box'><b>Note:</b> All fee values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
        
        # Reorder columns to put SurfFuture at the end
        desired_order = []
        for ex in exchanges:
            display_name = exchanges_display[ex]
            if display_name in table_df.columns:
                desired_order.append(display_name)
        if 'Average' in table_df.columns:
            desired_order.append('Average')
        
        # Reorder columns according to the specified order
        ordered_columns = [col for col in desired_order if col in table_df.columns]
        table_df = table_df[ordered_columns]
        
        # Format to 6 decimal places
        formatted_df = table_df.round(6)
        
        # Display the table with minimal formatting
        st.dataframe(formatted_df, height=500, use_container_width=True)
        
        # Create a summary for this token
        st.markdown("### Exchange Comparison Summary")
        
        # Get the daily averages for this token
        daily_avgs = token_daily_avgs[token]
        
        summary_data = {}
        for column in table_df.columns:
            col_data = table_df[column].dropna()
            if not col_data.empty:
                summary_data[column] = {
                    'Average Fee': col_data.mean(),
                    'Min Fee': col_data.min(),
                    'Max Fee': col_data.max(),
                    'Data Points': len(col_data)
                }
                
        if summary_data:
            # Create a summary dataframe
            summary_rows = []
            for exchange, metrics in summary_data.items():
                summary_rows.append({
                    'Exchange': exchange,
                    'Average Fee': round(metrics['Average Fee'], 6),
                    'Min Fee': round(metrics['Min Fee'], 6),
                    'Max Fee': round(metrics['Max Fee'], 6),
                    'Data Points': metrics['Data Points']
                })
            summary_df = pd.DataFrame(summary_rows)
            
            # Display the summary dataframe
            st.dataframe(summary_df, height=200, use_container_width=True)
            
            # Calculate best exchange (lowest average fee)
            exchanges_in_data = [ex for ex in summary_data.keys() if ex != 'Average']
            if exchanges_in_data:
                best_exchange = min(
                    exchanges_in_data,
                    key=lambda x: summary_data[x]['Average Fee']
                )
                st.info(f"Best exchange for {token} based on average fee: **{best_exchange}**")
        
        st.markdown("---")  # Add a separator between tokens
        
else:
    st.warning("No valid fee data available for the selected tokens.")
    # Add more diagnostic information
    st.error("Please check the following:")
    st.write("1. Make sure the oracle_exchange_fee table has data for the selected time period")
    st.write("2. Verify the exchange names and column names in the database")
    st.write("3. Check the logs for more detailed error messages")

with st.expander("Understanding the Exchange Fee Comparison"):
    st.markdown("""
    ### About This Dashboard
    
    This dashboard compares the fees charged by different exchanges for trading various cryptocurrency pairs. 
    
    ### Key Features:
    
    - **Summary Table**: Shows total fees for each token and exchange.
    
    - **Detailed Token Tables**: For each token, you can see the fees at 10-minute intervals throughout the day.
    
    - **Visualizations**:
      - The bar chart shows the average fee across all tokens for each exchange
      - The pie chart shows which exchange offers the lowest fees for the most tokens
    
    - **Best Exchange**: For each token and overall, the exchange with the lowest fees is identified.
    
    ### Note on Scaling:
    
    If fee values are very small, they may be multiplied by a scaling factor for better readability. The scaling factor is indicated with each table.
    """)