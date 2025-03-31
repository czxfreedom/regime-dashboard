# Save this as pages/06_Exchange_Spread_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Exchange Spread Comparison",
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
st.title("Exchange Spread Comparison (30min)")
st.subheader("Bid-Ask Spreads Across Binance, Hyperliquid, and Gate - Last 24 Hours (Singapore Time)")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
expected_points = 48  # Expected data points per pair over 24 hours
singapore_timezone = pytz.timezone('Asia/Singapore')
exchanges = ["Binance", "Hyperliquid", "Gate"]

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
        return ["BTC", "ETH", "SOL"]  # Default fallback

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

# Fetch spread data for a token with 30min timeframe
@st.cache_data(ttl=600, show_spinner="Calculating spreads...")
def fetch_spread_data(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Query to get the bid-ask data
    query = f"""
    SELECT 
        time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
        pair_name,
        source,
        best_bid,
        best_ask,
        (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 100 AS spread_pct
    FROM 
        oracle_orderbook_snapshot
    WHERE 
        pair_name = '{token}'
        AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    ORDER BY 
        timestamp DESC,
        source
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print(f"[{token}] No spread data found.")
            return None

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare data structure for 30-min resampling
        spread_data = {}
        avg_data = []
        
        # Process each exchange separately
        for exchange in exchanges:
            exchange_df = df[df['source'] == exchange].copy()
            if not exchange_df.empty:
                exchange_df = exchange_df.set_index('timestamp')
                # Resample to 30-minute periods
                resampled = exchange_df['spread_pct'].resample('30min').mean().dropna()
                
                if not resampled.empty:
                    # Get last 24 hours
                    last_24h = resampled.iloc[-48:]
                    spread_data[exchange] = last_24h
                    
                    # Prepare data for average calculation
                    exchange_df = pd.DataFrame(last_24h)
                    exchange_df['source'] = exchange
                    exchange_df['pair_name'] = token
                    exchange_df.reset_index(inplace=True)
                    avg_data.append(exchange_df)
                
        # If no valid data found for any exchange
        if not spread_data:
            return None
            
        # Calculate average spread across all exchanges
        if avg_data:
            all_exchanges_df = pd.concat(avg_data)
            all_exchanges_df = all_exchanges_df.set_index('timestamp')
            avg_spread = all_exchanges_df.groupby(all_exchanges_df.index)['spread_pct'].mean()
            spread_data['Average'] = avg_spread
            
        # Create time labels
        result_df = pd.DataFrame(spread_data)
        result_df['time_label'] = result_df.index.strftime('%H:%M')
        
        return result_df
            
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate spreads for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_spread_data(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create display for each token
if token_results:
    # Color function for spread cells
    def color_spread(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val < 0.05:  # Very tight spread (<0.05%)
            return 'background-color: rgba(0, 200, 0, 0.7); color: black'
        elif val < 0.1:  # Tight spread (<0.1%)
            return 'background-color: rgba(100, 255, 100, 0.7); color: black'
        elif val < 0.2:  # Normal spread (<0.2%)
            return 'background-color: rgba(255, 255, 0, 0.7); color: black'
        elif val < 0.5:  # Wide spread (<0.5%)
            return 'background-color: rgba(255, 165, 0, 0.7); color: black'
        else:  # Very wide spread (≥0.5%)
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    # For each token, create a table showing all exchanges
    for token, df in token_results.items():
        st.markdown(f"## {token} Spread Comparison")
        
        # Set the time_label as index to display in rows
        table_df = df.set_index('time_label')
        
        # Sort time labels
        all_times = sorted(table_df.index, key=time_to_minutes, reverse=True)
        table_df = table_df.reindex(all_times)
        
        # Format as percentage with 4 decimal places
        formatted_df = table_df.applymap(lambda x: f"{x:.4f}%" if not pd.isna(x) else "N/A")
        
        # Apply styling
        styled_df = table_df.style.applymap(color_spread)
        
        # Display the table
        st.dataframe(styled_df, height=500, use_container_width=True)
        
        # Create a summary for this token
        st.markdown("### Exchange Comparison Summary")
        
        summary_data = {}
        for column in table_df.columns:
            col_data = table_df[column].dropna()
            if not col_data.empty:
                summary_data[column] = {
                    'Average Spread (%)': col_data.mean(),
                    'Min Spread (%)': col_data.min(),
                    'Max Spread (%)': col_data.max(),
                    'Data Points': len(col_data)
                }
                
        if summary_data:
            summary_df = pd.DataFrame(summary_data).T
            summary_df = summary_df.round(4)
            
            # Format as percentage
            for col in ['Average Spread (%)', 'Min Spread (%)', 'Max Spread (%)']:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}%")
                
            # Reset index to display exchange names as a column
            summary_df = summary_df.reset_index()
            summary_df.columns = ['Exchange', 'Average Spread (%)', 'Min Spread (%)', 'Max Spread (%)', 'Data Points']
            
            # Display the summary table
            st.dataframe(summary_df, height=200, use_container_width=True)
            
            # Calculate best exchange (lowest average spread)
            if 'Average' in summary_data:
                best_exchange = min(
                    [ex for ex in summary_data.keys() if ex != 'Average'], 
                    key=lambda x: summary_data[x]['Average Spread (%)']
                )
                st.info(f"Best exchange for {token} based on average spread: **{best_exchange}**")
        
        st.markdown("---")  # Add a separator between tokens
    
    # Create an overall comparison of all tokens
    st.markdown("## Overall Token Spread Comparison")
    
    all_token_averages = {}
    for token, df in token_results.items():
        if 'Average' in df.columns:
            avg_spread = df['Average'].mean()
            all_token_averages[token] = avg_spread
    
    if all_token_averages:
        # Create and sort the overall comparison DataFrame
        overall_df = pd.DataFrame.from_dict(all_token_averages, orient='index', columns=['Average Spread (%)'])
        overall_df = overall_df.sort_values(by='Average Spread (%)')
        
        # Reset index to display token names as a column
        overall_df = overall_df.reset_index()
        overall_df.columns = ['Token', 'Average Spread (%)']
        
        # Format as percentage
        overall_df['Average Spread (%)'] = overall_df['Average Spread (%)'].apply(lambda x: f"{x:.4f}%")
        
        # Display the overall comparison
        st.dataframe(overall_df, height=400, use_container_width=True)
        
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
            title="Average Spread Comparison Across Tokens",
            xaxis_title="Token",
            yaxis_title="Average Spread (%)",
            yaxis_tickformat='.4%',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.warning("No valid spread data available for the selected tokens.")

with st.expander("Understanding the Exchange Spread Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the bid-ask spread as a percentage for each exchange and trading pair over the last 24 hours using 30-minute bars.
    Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
    
    **Spread Calculation**: 
    Spread % = (Ask Price - Bid Price) / ((Ask Price + Bid Price) / 2) * 100
    
    **Color coding:**
    - **Dark Green** (< 0.05%): Very tight spread
    - **Light Green** (0.05-0.1%): Tight spread
    - **Yellow** (0.1-0.2%): Normal spread
    - **Orange** (0.2-0.5%): Wide spread
    - **Red** (> 0.5%): Very wide spread
    
    **Average Column**: 
    The "Average" column represents the mean spread across all exchanges for each time period.
    
    **Summary Table**:
    For each token, a summary table shows the average, minimum, and maximum spreads across the 24-hour period for each exchange.
    
    **Missing values (light gray cells)** indicate periods where no data was available for that exchange.
    """)