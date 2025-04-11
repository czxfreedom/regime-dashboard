# Save this as pages/06_Direction_Changes_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Direction Changes Analysis",
    page_icon="🔄",
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
st.title("Direction Changes Analysis (10min Intervals)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters
timeframe = "10min"  # Using 10-minute intervals as requested
lookback_days = 1  # 24 hours
expected_points = 144  # Expected data points per pair over 24 hours (6 per hour × 24 hours)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch all available tokens from DB
@st.cache_data(show_spinner="Fetching tokens...")
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

# Function to generate aligned 10-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 10-minute time blocks for past 24 hours,
    aligned with standard 10-minute intervals (e.g., 4:00-4:10, 4:10-4:20)
    """
    # Round down to the nearest 10-minute mark
    minute = current_time.minute
    nearest_10min = (minute // 10) * 10
    latest_complete_block_end = current_time.replace(minute=nearest_10min, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(6 * 24):  # 24 hours of 10-minute blocks (6 per hour)
        block_end = latest_complete_block_end - timedelta(minutes=i*10)
        block_start = block_end - timedelta(minutes=10)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Direction change classification
def classify_direction_changes(changes_count, interval_minutes=10):
    """
    Classify the number of direction changes within a 10-minute period
    """
    # Calculate theoretical max changes per minute (based on 500ms data)
    max_changes_per_minute = 60 / 0.5  # 120 potential changes per minute
    max_changes = max_changes_per_minute * interval_minutes * 0.5  # Theoretical maximum (50% of potential)
    
    if pd.isna(changes_count) or changes_count < 0:
        return ("UNKNOWN", 0, "Insufficient data")
    elif changes_count == 0:
        return ("NONE", 0, "No direction changes")
    elif changes_count < max_changes * 0.25:  # Less than 25% of theoretical max
        return ("LOW", 1, "Few direction changes")
    elif changes_count < max_changes * 0.50:  # Between 25% and 50% of theoretical max
        return ("MEDIUM", 2, "Moderate direction changes")
    elif changes_count < max_changes * 0.75:  # Between 50% and 75% of theoretical max
        return ("HIGH", 3, "Many direction changes")
    else:  # More than 75% of theoretical max
        return ("EXTREME", 4, "Extreme number of direction changes")

# Analyze price runs (similar to your CryptoMarketAnalyzer.analyze_price_runs method)
def analyze_price_runs(df):
    print("Analyzing price runs...")
    direction = df['direction'].values

    runs = {'up': [], 'down': [], 'flat': []}
    current_direction = 0 if direction[0] == 0 else (1 if direction[0] > 0 else -1)
    current_run = 1

    for i in range(1, len(direction)):
        if direction[i] == 0:
            continue
        elif direction[i] == current_direction:
            current_run += 1
        else:
            if current_direction == 1:
                runs['up'].append(current_run)
            elif current_direction == -1:
                runs['down'].append(current_run)
            elif current_direction == 0:
                runs['flat'].append(current_run)

            current_run = 1
            current_direction = direction[i]

    if current_direction == 1:
        runs['up'].append(current_run)
    elif current_direction == -1:
        runs['down'].append(current_run)
    elif current_direction == 0:
        runs['flat'].append(current_run)

    result = {
        'up_runs': {
            'count': len(runs['up']),
            'average_length': np.mean(runs['up']) if runs['up'] else 0,
            'max_length': max(runs['up']) if runs['up'] else 0
        },
        'down_runs': {
            'count': len(runs['down']),
            'average_length': np.mean(runs['down']) if runs['down'] else 0,
            'max_length': max(runs['down']) if runs['down'] else 0
        },
        'flat_runs': {
            'count': len(runs['flat']),
            'average_length': np.mean(runs['flat']) if runs['flat'] else 0,
            'max_length': max(runs['flat']) if runs['flat'] else 0
        },
        'total_runs': len(runs['up']) + len(runs['down']) + len(runs['flat'])
    }

    return result

# Fetch price data and calculate direction changes
@st.cache_data(ttl=600, show_spinner="Calculating direction changes...")
def fetch_and_calculate_direction_changes(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price AS price, 
        pair_name
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
        
        # Calculate basic metrics (using same logic as CryptoMarketAnalyzer)
        df['price_change'] = df['price'].diff()
        df['abs_price_change'] = df['price_change'].abs()
        
        # Calculate direction
        df['direction'] = np.sign(df['price_change']).fillna(0)
        
        # Calculate direction changes (same logic as your code)
        df['prev_direction'] = df['direction'].shift(1).fillna(0)
        df['direction_change'] = ((df['direction'] != df['prev_direction']) &
                                 (df['direction'] != 0) &
                                 (df['prev_direction'] != 0)).astype(int)
        
        # Calculate the number of direction changes in each 10-minute window
        direction_changes = df['direction_change'].resample('10min', closed='left', label='left').sum()
        
        # Create a DataFrame with the results
        changes_df = direction_changes.to_frame(name='direction_changes')
        changes_df['original_datetime'] = changes_df.index
        changes_df['time_label'] = changes_df.index.strftime('%H:%M')
        
        # Calculate the 24-hour average
        changes_df['avg_24h_changes'] = changes_df['direction_changes'].mean()
        
        # Classify direction changes
        changes_df['changes_info'] = changes_df['direction_changes'].apply(classify_direction_changes)
        changes_df['changes_regime'] = changes_df['changes_info'].apply(lambda x: x[0])
        changes_df['changes_desc'] = changes_df['changes_info'].apply(lambda x: x[2])
        
        # Also classify the 24-hour average
        changes_df['avg_changes_info'] = changes_df['avg_24h_changes'].apply(classify_direction_changes)
        changes_df['avg_changes_regime'] = changes_df['avg_changes_info'].apply(lambda x: x[0])
        changes_df['avg_changes_desc'] = changes_df['avg_changes_info'].apply(lambda x: x[2])
        
        # Calculate direction change rate (changes per minute)
        changes_df['changes_per_minute'] = changes_df['direction_changes'] / 10
        changes_df['avg_changes_per_minute'] = changes_df['avg_24h_changes'] / 10
        
        # Additional metrics
        # Calculate the percentage of time spent in each direction
        uptime_pct = (df[df['direction'] > 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        downtime_pct = (df[df['direction'] < 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        flattime_pct = (df[df['direction'] == 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        
        changes_df['uptime_pct'] = uptime_pct
        changes_df['downtime_pct'] = downtime_pct
        changes_df['flattime_pct'] = flattime_pct
        
        # Calculate price runs
        runs = analyze_price_runs(df)
        avg_run_length = (runs['up_runs']['average_length'] + runs['down_runs']['average_length']) / 2
        changes_df['avg_run_length'] = avg_run_length
        changes_df['up_run_avg'] = runs['up_runs']['average_length']
        changes_df['down_run_avg'] = runs['down_runs']['average_length']
        changes_df['up_run_count'] = runs['up_runs']['count']
        changes_df['down_run_count'] = runs['down_runs']['count']
        
        print(f"[{token}] Successful Direction Changes Calculation")
        return changes_df
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

# Calculate direction changes for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_direction_changes(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    # Create table data
    table_data = {}
    for token, df in token_results.items():
        changes_series = df.set_index('time_label')['direction_changes']
        table_data[token] = changes_series
    
    # Create DataFrame with all tokens
    changes_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(changes_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    changes_table = changes_table.reindex(ordered_times)
    
    # Function to color cells based on number of direction changes
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val == 0:
            return 'background-color: rgba(240, 240, 240, 0.7); color: black'  # Light grey for no changes
        elif val < 50:  # Low number of changes - green
            intensity = max(0, min(255, int(255 * val / 50)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 100:  # Medium number of changes - yellow
            intensity = max(0, min(255, int(255 * (val - 50) / 50)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < 200:  # High number of changes - orange
            intensity = max(0, min(255, int(255 * (val - 100) / 100)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Extreme number of changes - red
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    styled_table = changes_table.style.applymap(color_cells)
    st.markdown("## Direction Changes Table (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Few Changes</span>, <span style='color:#aaaa00'>Moderate</span>, <span style='color:orange'>Many</span>, <span style='color:red'>Extreme</span>", unsafe_allow_html=True)
    st.markdown("Values shown as absolute count of direction changes within each 10-minute interval")
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create ranking table based on average direction changes
    st.subheader("Direction Changes Ranking (24-Hour Average, Descending Order)")
    
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_changes' in df.columns and not df['avg_24h_changes'].isna().all():
            avg_changes = df['avg_24h_changes'].iloc[0]  # All rows have the same avg value
            changes_regime = df['avg_changes_desc'].iloc[0]
            max_changes = df['direction_changes'].max()
            min_changes = df['direction_changes'].min()
            # Calculate changes per minute
            avg_changes_per_min = df['avg_changes_per_minute'].iloc[0]
            
            # Get directional bias
            uptime_pct = df['uptime_pct'].iloc[0]
            downtime_pct = df['downtime_pct'].iloc[0]
            flattime_pct = df['flattime_pct'].iloc[0]
            
            # Get run information
            avg_run_length = df['avg_run_length'].iloc[0]
            up_run_avg = df['up_run_avg'].iloc[0]
            down_run_avg = df['down_run_avg'].iloc[0]
            
            # Determine directional bias text
            if uptime_pct > downtime_pct * 1.5:
                directional_bias = "Strong Upward"
            elif uptime_pct > downtime_pct * 1.1:
                directional_bias = "Slight Upward"
            elif downtime_pct > uptime_pct * 1.5:
                directional_bias = "Strong Downward"
            elif downtime_pct > uptime_pct * 1.1:
                directional_bias = "Slight Downward"
            else:
                directional_bias = "Neutral"
            
            ranking_data.append({
                'Token': token,
                'Avg Changes': round(avg_changes, 1),
                'Changes/Min': round(avg_changes_per_min, 1),
                'Regime': changes_regime,
                'Max Changes': round(max_changes),
                'Min Changes': round(min_changes),
                'Range': round(max_changes - min_changes),
                'Avg Run Length': round(avg_run_length, 1),
                'Up Run Avg': round(up_run_avg, 1),
                'Down Run Avg': round(down_run_avg, 1),
                'Uptime %': round(uptime_pct, 1),
                'Downtime %': round(downtime_pct, 1),
                'Flattime %': round(flattime_pct, 1),
                'Directional Bias': directional_bias
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        # Sort by average changes (high to low)
        ranking_df = ranking_df.sort_values(by='Avg Changes', ascending=False)
        # Add rank column
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        
        # Reset the index to remove it
        ranking_df = ranking_df.reset_index(drop=True)
        
        # Format ranking table with colors
        def color_regime(val):
            if 'Few' in val or 'None' in val:
                return 'color: green'
            elif 'Moderate' in val:
                return 'color: #aaaa00'
            elif 'Many' in val:
                return 'color: orange'
            elif 'Extreme' in val:
                return 'color: red'
            return ''
        
        def color_changes(val):
            if pd.isna(val):
                return ''
            elif val < 50:
                return 'color: green'
            elif val < 100:
                return 'color: #aaaa00'
            elif val < 200:
                return 'color: orange'
            else:
                return 'color: red'
        
        def color_directional_bias(val):
            if 'Strong Upward' in val:
                return 'color: #008800; font-weight: bold'
            elif 'Slight Upward' in val:
                return 'color: #008800'
            elif 'Strong Downward' in val:
                return 'color: #880000; font-weight: bold'
            elif 'Slight Downward' in val:
                return 'color: #880000'
            else:
                return 'color: #888888'
        
        # Apply styling
        styled_ranking = ranking_df.style\
            .applymap(color_regime, subset=['Regime'])\
            .applymap(color_changes, subset=['Avg Changes', 'Max Changes', 'Min Changes'])\
            .applymap(color_directional_bias, subset=['Directional Bias'])
        
        # Display the styled dataframe
        st.dataframe(styled_ranking, height=500, use_container_width=True)
    else:
        st.warning("No ranking data available.")
    
    # Identify tokens with high direction changes volatility
    st.subheader("Tokens with Highly Variable Direction Changes")
    
    # Calculate the coefficient of variation (std/mean) for direction changes
    variability_data = []
    for token, df in token_results.items():
        if not df.empty and 'direction_changes' in df.columns:
            changes_std = df['direction_changes'].std()
            changes_mean = df['direction_changes'].mean()
            changes_cv = changes_std / changes_mean if changes_mean > 0 else 0
            
            max_changes = df['direction_changes'].max()
            min_changes = df['direction_changes'].min()
            changes_range = max_changes - min_changes
            
            variability_data.append({
                'Token': token,
                'CoV': round(changes_cv, 2),  # Coefficient of Variation
                'Std Dev': round(changes_std, 1),
                'Mean Changes': round(changes_mean, 1),
                'Range': int(changes_range),
                'Max Changes': int(max_changes),
                'Min Changes': int(min_changes)
            })
    
    if variability_data:
        variability_df = pd.DataFrame(variability_data)
        # Sort by Coefficient of Variation (high to low)
        variability_df = variability_df.sort_values(by='CoV', ascending=False)
        
        # Reset the index to remove it
        variability_df = variability_df.reset_index(drop=True)
        
        # Display the dataframe
        st.dataframe(variability_df, height=300, use_container_width=True)
        
        # Create a bar chart of the top 10 most variable tokens
        top_variable_tokens = variability_df.head(10)
        
        fig = px.bar(
            top_variable_tokens, 
            x='Token', 
            y='CoV', 
            title='Top 10 Tokens by Direction Change Variability',
            labels={'CoV': 'Coefficient of Variation (Std/Mean)', 'Token': 'Token'},
            color='CoV',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No variability data available.")
    
    # Direction Changes Distribution
    st.subheader("24-Hour Direction Changes Overview (Singapore Time)")
    avg_changes = {}
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_changes' in df.columns and not df['avg_24h_changes'].isna().all():
            avg = df['avg_24h_changes'].iloc[0]
            regime = df['avg_changes_desc'].iloc[0]
            avg_changes[token] = (avg, regime)
    
    if avg_changes:
        # Calculate statistics for the different regimes
        few_changes = sum(1 for v, r in avg_changes.values() if 'Few' in r or 'None' in r)
        moderate_changes = sum(1 for v, r in avg_changes.values() if 'Moderate' in r)
        many_changes = sum(1 for v, r in avg_changes.values() if 'Many' in r)
        extreme_changes = sum(1 for v, r in avg_changes.values() if 'Extreme' in r)
        total = few_changes + moderate_changes + many_changes + extreme_changes
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Few Changes", f"{few_changes} ({few_changes/total*100:.1f}%)")
        col2.metric("Moderate Changes", f"{moderate_changes} ({moderate_changes/total*100:.1f}%)")
        col3.metric("Many Changes", f"{many_changes} ({many_changes/total*100:.1f}%)")
        col4.metric("Extreme Changes", f"{extreme_changes} ({extreme_changes/total*100:.1f}%)")
        
        labels = ['Few Changes', 'Moderate Changes', 'Many Changes', 'Extreme Changes']
        values = [few_changes, moderate_changes, many_changes, extreme_changes]
        colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
        fig.update_layout(
            title="Direction Changes Distribution (Singapore Time)",
            height=400,
            font=dict(color="#000000", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create columns for each direction changes category
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.markdown("### Tokens with Few Direction Changes")
            fc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Few' in r or 'None' in r]
            fc_tokens.sort(key=lambda x: x[1])
            if fc_tokens:
                for token, value, regime in fc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Tokens with Moderate Direction Changes")
            mc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Moderate' in r]
            mc_tokens.sort(key=lambda x: x[1])
            if mc_tokens:
                for token, value, regime in mc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### Tokens with Many Direction Changes")
            hc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Many' in r]
            hc_tokens.sort(key=lambda x: x[1])
            if hc_tokens:
                for token, value, regime in hc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col4:
            st.markdown("### Tokens with Extreme Direction Changes")
            ec_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Extreme' in r]
            ec_tokens.sort(key=lambda x: x[1], reverse=True)
            if ec_tokens:
                for token, value, regime in ec_tokens:
                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
    else:
        st.warning("No direction changes data available for the selected tokens.")
        
    # Add some simple visualizations for directional bias
    st.subheader("Directional Bias Analysis")
    
    # Collect directional bias data
    directional_data = []
    for token, df in token_results.items():
        if not df.empty and 'uptime_pct' in df.columns:
            uptime = df['uptime_pct'].iloc[0]
            downtime = df['downtime_pct'].iloc[0]
            flattime = df['flattime_pct'].iloc[0]
            
            # Calculate uptrend vs downtrend ratio
            if downtime > 0:
                trend_ratio = uptime / downtime
            else:
                trend_ratio = float('inf')
                
            directional_data.append({
                'Token': token,
                'Uptime %': round(uptime, 1),
                'Downtime %': round(downtime, 1),
                'Flattime %': round(flattime, 1),
                'Up/Down Ratio': round(trend_ratio, 2) if trend_ratio != float('inf') else float('inf')
            })
    
    if directional_data:
        # Create directional bias dataframe
        dir_df = pd.DataFrame(directional_data)
        
        # Sort by Up/Down Ratio
        dir_df = dir_df.sort_values(by='Up/Down Ratio', ascending=False)
        
        # Display the dataframe
        st.dataframe(dir_df, height=300, use_container_width=True)
        
        # Create visualization of up vs down time
        if len(dir_df) > 0:
            # Prepare data for stacked bar chart
            plot_data = []
            for _, row in dir_df.head(10).iterrows():  # Take top 10 tokens
                plot_data.append({
                    'Token': row['Token'],
                    'Direction': 'Up',
                    'Percentage': row['Uptime %']
                })
                plot_data.append({
                    'Token': row['Token'],
                    'Direction': 'Down',
                    'Percentage': row['Downtime %']
                })
                plot_data.append({
                    'Token': row['Token'],
                    'Direction': 'Flat',
                    'Percentage': row['Flattime %']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            fig = px.bar(
                plot_df,
                x='Token',
                y='Percentage',
                color='Direction',
                title='Directional Bias: Time Spent in Each Direction (Top 10 by Up/Down Ratio)',
                color_discrete_map={'Up': 'green', 'Down': 'red', 'Flat': 'gray'},
                barmode='stack'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No directional bias data available.")
    
    # Analyze relationship between run length and direction changes
    st.subheader("Run Length vs Direction Changes")
    
    run_vs_changes_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_run_length' in df.columns and 'avg_24h_changes' in df.columns:
            run_vs_changes_data.append({
                'Token': token,
                'Avg Run Length': df['avg_run_length'].iloc[0],
                'Avg Direction Changes': df['avg_24h_changes'].iloc[0]
            })
    
    if run_vs_changes_data:
        rc_df = pd.DataFrame(run_vs_changes_data)
        
        # Create a scatter plot
        fig = px.scatter(
            rc_df, 
            x='Avg Run Length', 
            y='Avg Direction Changes',
            text='Token',
            title='Relationship Between Run Length and Direction Changes',
            labels={
                'Avg Run Length': 'Average Run Length (consecutive price movements in same direction)',
                'Avg Direction Changes': 'Average Direction Changes per 10min'
            },
            color='Avg Direction Changes',
            color_continuous_scale='Viridis',
            size='Avg Run Length',
            size_max=20
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = rc_df['Avg Run Length'].corr(rc_df['Avg Direction Changes'])
        st.write(f"**Correlation between Run Length and Direction Changes: {corr:.3f}**")
        st.write("""
        **Note:** A negative correlation is expected as shorter runs typically mean more frequent direction changes.
        """)
    else:
        st.warning("Insufficient data to analyze run length vs direction changes.")
    
    # Heat Map Analysis - shows time periods with most direction changes across all tokens
    st.subheader("Direction Changes Heat Map (When do price directions change most?)")
    
    # Prepare data for the heatmap
    if len(token_results) > 0 and len(changes_table) > 0:
        # Calculate the average direction changes per time period across all tokens
        avg_changes_by_time = changes_table.mean(axis=1).to_frame('avg_changes')
        avg_changes_by_time = avg_changes_by_time.sort_index()
        
        # Get the hour from the index (time label)
        avg_changes_by_time['hour'] = avg_changes_by_time.index.str.split(':').str[0].astype(int)
        
        # Group by hour and calculate mean
        hourly_changes = avg_changes_by_time.groupby('hour')['avg_changes'].mean().reset_index()
        
        # Create a bar chart
        fig = px.bar(
            hourly_changes,
            x='hour',
            y='avg_changes',
            title='Average Direction Changes by Hour of Day (Singapore Time)',
            labels={'hour': 'Hour of Day (24h)', 'avg_changes': 'Avg Direction Changes'},
            color='avg_changes',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a text explanation
        max_hour = hourly_changes.loc[hourly_changes['avg_changes'].idxmax()]
        min_hour = hourly_changes.loc[hourly_changes['avg_changes'].idxmin()]
        
        st.write(f"""
        **Analysis of Direction Changes by Time of Day:**
        
        - The most active hour for direction changes is **{max_hour['hour']}:00** with an average of **{max_hour['avg_changes']:.1f}** changes per 10-minute period.
        - The least active hour is **{min_hour['hour']}:00** with an average of **{min_hour['avg_changes']:.1f}** changes per 10-minute period.
        - This may correspond to market open/close times or periods of higher trading activity.
        """)
    else:
        st.warning("Insufficient data to create the heat map analysis.")
    
    # Final Summary
    st.subheader("Executive Summary of Direction Changes Analysis")
    
    # Generate a simple text summary
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        top_changing_token = ranking_df.sort_values(by='Avg Changes', ascending=False).iloc[0]
        lowest_changing_token = ranking_df.sort_values(by='Avg Changes', ascending=True).iloc[0]
        
        most_variable_token = None
        if variability_data:
            var_df = pd.DataFrame(variability_data)
            most_variable_token = var_df.sort_values(by='CoV', ascending=False).iloc[0]
        
        st.write(f"""
        ### Key Findings:
        
        1. **{top_changing_token['Token']}** shows the highest average direction changes with **{top_changing_token['Avg Changes']}** changes per 10-minute period.
        
        2. **{lowest_changing_token['Token']}** shows the lowest average direction changes with **{lowest_changing_token['Avg Changes']}** changes per 10-minute period.
        
        3. Direction changes occur most frequently during the **{max_hour['hour']}:00** hour (Singapore time).
        """)
        
        if most_variable_token is not None:
            st.write(f"""
        4. **{most_variable_token['Token']}** shows the most variable direction change behavior with a Coefficient of Variation of **{most_variable_token['CoV']}**.
            """)
    else:
        st.warning("Insufficient data to generate an executive summary.")

with st.expander("Understanding the Direction Changes Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the number of price direction changes for all selected tokens over the last 24 hours using 10-minute intervals.
    Each row represents a specific 10-minute time period, with times shown in Singapore time. The table is sorted with the most recent 10-minute period at the top.
    
    **Color coding:**
    - **Green** (< 50 changes): Few direction changes
    - **Yellow** (50-100 changes): Moderate direction changes
    - **Orange** (100-200 changes): Many direction changes
    - **Red** (> 200 changes): Extreme number of direction changes
    
    **The intensity of the color indicates the number of direction changes:**
    - Darker green = Fewer changes
    - Darker red = More changes
    
    **Ranking Table:**
    The ranking table sorts tokens by their 24-hour average direction changes from highest to lowest.
    
    ### Direction Change Metric Details
    Direction changes occur when the price movement switches from upward to downward or vice versa. A direction change is counted when:
    
    1. The previous price movement was positive (price increasing)
    2. The current price movement is negative (price decreasing)
    
    OR
    
    1. The previous price movement was negative (price decreasing) 
    2. The current price movement is positive (price increasing)
    
    The calculation ignores flat periods (no price change) and only counts true reversals in price movement.
    
    **Metrics explained:**
    - **Direction Changes**: Raw count of direction changes in the given time period
    - **Changes/Min**: Average number of direction changes per minute
    - **Directional Bias**: Overall tendency of price movement (Upward, Neutral, or Downward)
    - **Up/Down Ratio**: Ratio of time spent in upward movement vs. downward movement
    - **Avg Run Length**: Average number of consecutive price movements in the same direction
    
    **Note on data granularity:**
    The underlying data is sampled at ~500ms intervals, giving a theoretical maximum of 1,200 data points per 10-minute period. The analysis examines each price change to identify genuine direction reversals.
    """)