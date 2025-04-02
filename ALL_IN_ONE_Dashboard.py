
import streamlit as st

st.set_page_config(page_title="ALL-IN-ONE Dashboard", layout="wide")
st.title("📊 ALL-IN-ONE Dashboard")

# Global refresh button
if st.button("🔄 Refresh All Tabs"):
    st.cache_data.clear()
    st.experimental_rerun()

tab_names = ["Macro View", "Cumulative PnL", "PnL and Trades", "Regime Matrix", "Spread Analysis", "Vol & Hurst"]
tabs = st.tabs(tab_names)


def render_macro_view():
    # Save this as pages/04_Daily_Hurst_Table.py in your Streamlit app folder

    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz

    st.set_page_config(
        page_title="Daily Hurst Table",
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
    st.title("Daily Hurst Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    rolling_window = 20  # Window size for Hurst calculation
    expected_points = 48  # Expected data points per pair over 24 hours
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

    # Universal Hurst calculation function
    def universal_hurst(ts):
        print(f"universal_hurst called with ts: {ts}")
        print(f"Type of ts: {type(ts)}")
        if isinstance(ts, (list, np.ndarray, pd.Series)) and len(ts) > 0:
            print(f"First few values of ts: {ts[:5]}")

        if ts is None:
            print("ts is None")
            return np.nan

        if isinstance(ts, pd.Series) and ts.empty:
            print("ts is empty series")
            return np.nan

        if not isinstance(ts, (list, np.ndarray, pd.Series)):
            print(f"ts is not a list, NumPy array, or Series. Type: {type(ts)}")
            return np.nan

        try:
            ts = np.array(ts, dtype=float)
        except Exception as e:
            print(f"ts cannot be converted to float: {e}")
            return np.nan

        if len(ts) < 10 or np.any(~np.isfinite(ts)):
            print(f"ts length < 10 or non-finite values: {ts}")
            return np.nan

        # Convert to returns - using log returns handles any scale of asset
        epsilon = 1e-10
        adjusted_ts = ts + epsilon
        log_returns = np.diff(np.log(adjusted_ts))

        # If all returns are exactly zero (completely flat price), return 0.5
        if np.all(log_returns == 0):
            return 0.5

        # Use multiple methods and average for robustness
        hurst_estimates = []

        # Method 1: Rescaled Range (R/S) Analysis
        try:
            max_lag = min(len(log_returns) // 4, 40)
            lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
            rs_values = []
            for lag in lags:
                segments = len(log_returns) // lag
                if segments < 1:
                    continue
                rs_by_segment = []
                for i in range(segments):
                    segment = log_returns[i*lag:(i+1)*lag]
                    if len(segment) < lag // 2:
                        continue
                    mean_return = np.mean(segment)
                    std_return = np.std(segment)
                    if std_return == 0:
                        continue
                    cumdev = np.cumsum(segment - mean_return)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = std_return
                    rs_by_segment.append(r / s)
                if rs_by_segment:
                    rs_values.append((lag, np.mean(rs_by_segment)))
            if len(rs_values) >= 4:
                lags_log = np.log10([x[0] for x in rs_values])
                rs_log = np.log10([x[1] for x in rs_values])
                poly = np.polyfit(lags_log, rs_log, 1)
                h_rs = poly[0]
                hurst_estimates.append(h_rs)
        except Exception as e:
            print(f"Error in R/S calculation: {e}")
            pass

        # Method 2: Variance Method
        try:
            max_lag = min(len(log_returns) // 4, 40)
            lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
            var_values = []
            for lag in lags:
                if lag >= len(log_returns):
                    continue
                lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
                if len(lagged_returns) < 2:
                    continue
                var = np.var(lagged_returns)
                if var > 0:
                    var_values.append((lag, var))
            if len(var_values) >= 4:
                lags_log = np.log10([x[0] for x in var_values])
                var_log = np.log10([x[1] for x in var_values])
                poly = np.polyfit(lags_log, var_log, 1)
                h_var = (poly[0] + 1) / 2
                hurst_estimates.append(h_var)
        except Exception as e:
            print(f"Error in Variance calculation: {e}")
            pass

        # Fallback to autocorrelation method if other methods fail
        if not hurst_estimates and len(log_returns) > 1:
            try:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
                hurst_estimates.append(h_acf)
            except Exception as e:
                print(f"Error in Autocorrelation calculation: {e}")
                pass

        # If we have estimates, aggregate them and constrain to 0-1 range
        if hurst_estimates:
            valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
            if not valid_estimates and hurst_estimates:
                valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
            if valid_estimates:
                return np.median(valid_estimates)

        return 0.5

    # Detailed regime classification function
    def detailed_regime_classification(hurst):
        if pd.isna(hurst):
            return ("UNKNOWN", 0, "Insufficient data")
        elif hurst < 0.2:
            return ("MEAN-REVERT", 3, "Strong mean-reversion")
        elif hurst < 0.3:
            return ("MEAN-REVERT", 2, "Moderate mean-reversion")
        elif hurst < 0.4:
            return ("MEAN-REVERT", 1, "Mild mean-reversion")
        elif hurst < 0.45:
            return ("NOISE", 1, "Slight mean-reversion bias")
        elif hurst <= 0.55:
            return ("NOISE", 0, "Pure random walk")
        elif hurst < 0.6:
            return ("NOISE", 1, "Slight trending bias")
        elif hurst < 0.7:
            return ("TREND", 1, "Mild trending")
        elif hurst < 0.8:
            return ("TREND", 2, "Moderate trending")
        else:
            return ("TREND", 3, "Strong trending")

    # Function to generate proper 30-minute time blocks for the past 24 hours
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
            blocks.append(block_label)

        return blocks

    # Generate aligned time blocks
    aligned_time_blocks = generate_aligned_time_blocks(now_sg)

    # Fetch and calculate Hurst for a token with 30min timeframe
    @st.cache_data(ttl=600, show_spinner="Calculating Hurst exponents...")
    def fetch_and_calculate_hurst(token):
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
            final_price, 
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

            print(f"[{token}] First few rows:\n{df.head()}")
            print(f"[{token}] DataFrame columns and types:\n{df.info()}")

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
            if one_min_ohlc.empty:
                print(f"[{token}] No OHLC data after resampling.")
                return None

            print(f"[{token}] one_min_ohlc head:\n{one_min_ohlc.head()}")
            print(f"[{token}] one_min_ohlc info:\n{one_min_ohlc.info()}")

            # Apply universal_hurst to the 'close' prices directly
            one_min_ohlc['Hurst'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(universal_hurst)

            # Resample to exactly 30min intervals aligned with clock
            thirty_min_hurst = one_min_ohlc['Hurst'].resample('30min', closed='left', label='left').mean().dropna()

            if thirty_min_hurst.empty:
                print(f"[{token}] No 30-min Hurst data.")
                return None

            last_24h_hurst = thirty_min_hurst.tail(48)  # Get up to last 48 periods (24 hours)
            last_24h_hurst = last_24h_hurst.to_frame()

            # Store original datetime index for reference
            last_24h_hurst['original_datetime'] = last_24h_hurst.index

            # Format time label to match our aligned blocks (HH:MM format)
            last_24h_hurst['time_label'] = last_24h_hurst.index.strftime('%H:%M')

            # Calculate regime information
            last_24h_hurst['regime_info'] = last_24h_hurst['Hurst'].apply(detailed_regime_classification)
            last_24h_hurst['regime'] = last_24h_hurst['regime_info'].apply(lambda x: x[0])
            last_24h_hurst['regime_desc'] = last_24h_hurst['regime_info'].apply(lambda x: x[2])

            print(f"[{token}] Successful Calculation")
            return last_24h_hurst
        except Exception as e:
            st.error(f"Error processing {token}: {e}")
            print(f"[{token}] Error processing: {e}")
            return None

    # Show progress bar while calculating
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Calculate Hurst for each token
    token_results = {}
    for i, token in enumerate(selected_tokens):
        try:  # Added try-except around token processing
            progress_bar.progress((i) / len(selected_tokens))
            status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
            result = fetch_and_calculate_hurst(token)
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
            hurst_series = df.set_index('time_label')['Hurst']
            table_data[token] = hurst_series

        # Create DataFrame with all tokens
        hurst_table = pd.DataFrame(table_data)

        # Apply the time blocks in the proper order (most recent first)
        available_times = set(hurst_table.index)
        ordered_times = [t for t in aligned_time_blocks if t in available_times]

        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)

        # Reindex with the ordered times
        hurst_table = hurst_table.reindex(ordered_times)
        hurst_table = hurst_table.round(2)

        def color_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;' # Grey for missing
            elif val < 0.4:
                intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
                return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
            elif val > 0.6:
                intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            else:
                return 'background-color: rgba(200, 200, 200, 0.5); color: black' # Lighter gray

        styled_table = hurst_table.style.applymap(color_cells)
        st.markdown("## Hurst Exponent Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:red'>Mean Reversion</span>, <span style='color:gray'>Random Walk</span>, <span style='color:green'>Trending</span>", unsafe_allow_html=True)
        st.dataframe(styled_table, height=700, use_container_width=True)

        st.subheader("Current Market Overview (Singapore Time)")

        # Use the first aligned time block that has data for each token
        latest_values = {}
        for token, df in token_results.items():
            if not df.empty and not df['Hurst'].isna().all():
                # Try to find the most recent time block in our aligned blocks
                for block_time in aligned_time_blocks[:5]:  # Check the 5 most recent blocks
                    latest_data = df[df['time_label'] == block_time]
                    if not latest_data.empty:
                        latest = latest_data['Hurst'].iloc[0]
                        regime = latest_data['regime_desc'].iloc[0]
                        latest_values[token] = (latest, regime)
                        break

                # If no match in aligned blocks, use the most recent data point
                if token not in latest_values:
                    latest = df['Hurst'].iloc[-1]
                    regime = df['regime_desc'].iloc[-1]
                    latest_values[token] = (latest, regime)

        if latest_values:
            mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
            random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
            trending = sum(1 for v, r in latest_values.values() if v > 0.6)
            total = mean_reverting + random_walk + trending

            if total > 0:  # Avoid division by zero
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean-Reverting", f"{mean_reverting} ({mean_reverting/total*100:.1f}%)", delta=f"{mean_reverting/total*100:.1f}%")
                col2.metric("Random Walk", f"{random_walk} ({random_walk/total*100:.1f}%)", delta=f"{random_walk/total*100:.1f}%")
                col3.metric("Trending", f"{trending} ({trending/total*100:.1f}%)", delta=f"{trending/total*100:.1f}%")

                labels = ['Mean-Reverting', 'Random Walk', 'Trending']
                values = [mean_reverting, random_walk, trending]
                colors = ['rgba(255,100,100,0.8)', 'rgba(200,200,200,0.8)', 'rgba(100,255,100,0.8)'] # Slightly more opaque

                fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)]) # Added black borders
                fig.update_layout(
                    title="Current Market Regime Distribution (Singapore Time)",
                    height=400,
                    font=dict(color="#000000", size=12),  # Set default font color and size
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Mean-Reverting Tokens")
                    mr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v < 0.4]
                    mr_tokens.sort(key=lambda x: x[1])
                    if mr_tokens:
                        for token, value, regime in mr_tokens:
                            st.markdown(f"- **{token}**: <span style='color:red'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                    else:
                        st.markdown("*No tokens in this category*")

                with col2:
                    st.markdown("### Random Walk Tokens")
                    rw_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if 0.4 <= v <= 0.6]
                    rw_tokens.sort(key=lambda x: x[1])
                    if rw_tokens:
                        for token, value, regime in rw_tokens:
                            st.markdown(f"- **{token}**: <span style='color:gray'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                    else:
                        st.markdown("*No tokens in this category*")

                with col3:
                    st.markdown("### Trending Tokens")
                    tr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v > 0.6]
                    tr_tokens.sort(key=lambda x: x[1], reverse=True)
                    if tr_tokens:
                        for token, value, regime in tr_tokens:
                            st.markdown(f"- **{token}**: <span style='color:green'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                    else:
                        st.markdown("*No tokens in this category*")
            else:
                st.warning("No valid data found for analysis.")
        else:
            st.warning("No data available for the selected tokens.")

    with st.expander("Understanding the Daily Hurst Table"):
        st.markdown("""
        ### How to Read This Table
        This table shows the Hurst exponent values for all selected tokens over the last 24 hours using 30-minute bars.
        Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
        **Color coding:**
        - **Red** (Hurst < 0.4): The token is showing mean-reverting behavior during that time period
        - **Gray** (Hurst 0.4-0.6): The token is behaving like a random walk (no clear pattern)
        - **Green** (Hurst > 0.6): The token is showing trending behavior
        **The intensity of the color indicates the strength of the pattern:**
        - Darker red = Stronger mean-reversion
        - Darker green = Stronger trending
        **Technical details:**
        - Each Hurst value is calculated by applying a rolling window of 20 one-minute bars to the closing prices, and then averaging the Hurst values of 30 one-minute bars.
        - Values are calculated using multiple methods (R/S Analysis, Variance Method, and Autocorrelation)
        - Missing values (light gray cells) indicate insufficient data for calculation
        """)

    # Add an expandable section to view the exact time blocks being analyzed
    with st.expander("View Time Blocks Being Analyzed"):
        time_blocks_df = pd.DataFrame(aligned_time_blocks, columns=['Block Start Time'])
        st.dataframe(time_blocks_df)

def render_cumulative_pnl():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz

    st.set_page_config(
        page_title="Trading Pairs PNL Dashboard",
        page_icon="💰",
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
    st.title("Trading Pairs PNL Dashboard")
    st.subheader("Performance Analysis by Time Period (Singapore Time)")

    # Set up the timezone
    singapore_timezone = pytz.timezone('Asia/Singapore')

    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

    # Function to get Singapore midnight today, yesterday, and 7 days ago
    def get_time_boundaries():
        # Current time in Singapore
        now_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)

        # Today's midnight in Singapore
        today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)

        # Yesterday's midnight in Singapore
        yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)

        # 7 days ago midnight in Singapore
        week_ago_midnight_sg = today_midnight_sg - timedelta(days=7)

        # All time (use a far past date, e.g., 5 years ago)
        all_time_start_sg = today_midnight_sg.replace(year=today_midnight_sg.year-5)

        # Convert all times back to UTC for database queries
        today_midnight_utc = today_midnight_sg.astimezone(pytz.utc)
        yesterday_midnight_utc = yesterday_midnight_sg.astimezone(pytz.utc)
        day_before_yesterday_midnight_utc = (yesterday_midnight_sg - timedelta(days=1)).astimezone(pytz.utc)
        week_ago_midnight_utc = week_ago_midnight_sg.astimezone(pytz.utc)
        all_time_start_utc = all_time_start_sg.astimezone(pytz.utc)
        now_utc = now_sg.astimezone(pytz.utc)

        return {
            "today": {
                "start": today_midnight_utc,
                "end": now_utc,
                "label": f"Today ({today_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "yesterday": {
                "start": yesterday_midnight_utc,
                "end": today_midnight_utc,
                "label": f"Yesterday ({yesterday_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "day_before_yesterday": {
                "start": day_before_yesterday_midnight_utc,
                "end": yesterday_midnight_utc,
                "label": f"Day Before ({(yesterday_midnight_sg - timedelta(days=1)).strftime('%Y-%m-%d')})"
            },
            "this_week": {
                "start": week_ago_midnight_utc,
                "end": now_utc,
                "label": f"This Week (Last 7 Days)"
            },
            "all_time": {
                "start": all_time_start_utc,
                "end": now_utc,
                "label": "All Time"
            }
        }

    # Calculate time boundaries
    time_boundaries = get_time_boundaries()

    # Fetch all available pairs from DB
    @st.cache_data(ttl=600, show_spinner="Fetching pairs...")
    def fetch_all_pairs():
        query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                st.error("No pairs found in the database.")
                return []
            return df['pair_name'].tolist()
        except Exception as e:
            st.error(f"Error fetching pairs: {e}")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

    all_pairs = fetch_all_pairs()

    # UI Controls
    col1, col2 = st.columns([3, 1])

    with col1:
        # Let user select pairs to display (or select all)
        select_all = st.checkbox("Select All Pairs", value=True)

        if select_all:
            selected_pairs = all_pairs
        else:
            selected_pairs = st.multiselect(
                "Select Pairs", 
                all_pairs,
                default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
            )

    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()

    if not selected_pairs:
        st.warning("Please select at least one pair")
        st.stop()

    # Function to fetch PNL data for a specific time period
    @st.cache_data(ttl=600)
    def fetch_pnl_data(pair_name, start_time, end_time):
        """Fetch platform PNL data for a specific time period."""

        query = f"""
        WITH order_pnl AS (
          -- Calculate platform order PNL
          SELECT
            COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (0, 1, 2, 3, 4)
        ),

        fee_data AS (
          -- Calculate user fee payments
          SELECT
            COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_fee_mode" = 1
            AND "taker_way" IN (1, 3)
        ),

        funding_pnl AS (
          -- Calculate platform funding fee PNL
          SELECT
            COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" = 0
        ),

        rebate_data AS (
          -- Calculate platform rebate payments
          SELECT
            COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
          FROM
            "public"."user_cashbooks"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "remark" = '给邀请人返佣'
        ),

        trade_count AS (
          -- Calculate total number of trades
          SELECT
            COUNT(*) AS "total_trades"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
        )

        -- Final query: combine all data sources
        SELECT
          (SELECT "platform_order_pnl" FROM order_pnl) +
          (SELECT "user_fee_payments" FROM fee_data) +
          (SELECT "platform_funding_pnl" FROM funding_pnl) +
          (SELECT "platform_rebate_payments" FROM rebate_data) AS "platform_total_pnl",
          (SELECT "total_trades" FROM trade_count) AS "total_trades"
        """

        try:
            print(f"[{pair_name}] Executing PNL query for period {start_time} to {end_time}")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] PNL query executed. Result: {df.iloc[0]['platform_total_pnl']}")

            if df.empty:
                return {"pnl": 0, "trades": 0}

            return {
                "pnl": float(df.iloc[0]['platform_total_pnl']),
                "trades": int(df.iloc[0]['total_trades'])
            }
        except Exception as e:
            st.error(f"Error processing PNL for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing PNL: {e}")
            return {"pnl": 0, "trades": 0}

    # Show progress bar while calculating
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Gather PNL data for all pairs and time periods
    results = {}
    periods = ["today", "yesterday", "day_before_yesterday", "this_week", "all_time"]

    for i, pair_name in enumerate(selected_pairs):
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")

        pair_data = {"pair_name": pair_name}

        for period in periods:
            start_time = time_boundaries[period]["start"]
            end_time = time_boundaries[period]["end"]

            # Fetch PNL data for this pair and time period
            period_data = fetch_pnl_data(pair_name, start_time, end_time)

            # Store the results
            pair_data[f"{period}_pnl"] = period_data["pnl"]
            pair_data[f"{period}_trades"] = period_data["trades"]

        results[pair_name] = pair_data

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(results)}/{len(selected_pairs)} pairs successfully")

    # Create DataFrame from results
    pnl_df = pd.DataFrame([results[pair] for pair in selected_pairs])

    # If DataFrame is empty, show warning and stop
    if pnl_df.empty:
        st.warning("No PNL data found for the selected pairs and time periods.")
        st.stop()

    # Reformat the DataFrame for display
    display_df = pd.DataFrame({
        'Trading Pair': pnl_df['pair_name'],
        'Today PNL (USD)': pnl_df['today_pnl'].round(2),
        'Today Trades': pnl_df['today_trades'],
        'Yesterday PNL (USD)': pnl_df['yesterday_pnl'].round(2),
        'Yesterday Trades': pnl_df['yesterday_trades'],
        'Day Before PNL (USD)': pnl_df['day_before_yesterday_pnl'].round(2),
        'Day Before Trades': pnl_df['day_before_yesterday_trades'],
        'Week PNL (USD)': pnl_df['this_week_pnl'].round(2),
        'Week Trades': pnl_df['this_week_trades'],
        'All Time PNL (USD)': pnl_df['all_time_pnl'].round(2),
        'All Time Trades': pnl_df['all_time_trades'],
    })

    # Function to format display DataFrame
    def format_display_df(df):
        # Add derived columns
        if 'Today PNL (USD)' in df.columns and 'Today Trades' in df.columns:
            df['Today PNL/Trade'] = (df['Today PNL (USD)'] / df['Today Trades']).round(2)
            df['Today PNL/Trade'] = df['Today PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)

        if 'Yesterday PNL (USD)' in df.columns and 'Yesterday Trades' in df.columns:
            df['Yesterday PNL/Trade'] = (df['Yesterday PNL (USD)'] / df['Yesterday Trades']).round(2)
            df['Yesterday PNL/Trade'] = df['Yesterday PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)

        if 'Day Before PNL (USD)' in df.columns and 'Day Before Trades' in df.columns:
            df['Day Before PNL/Trade'] = (df['Day Before PNL (USD)'] / df['Day Before Trades']).round(2)
            df['Day Before PNL/Trade'] = df['Day Before PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)

        if 'Week PNL (USD)' in df.columns and 'Week Trades' in df.columns:
            df['Week PNL/Trade'] = (df['Week PNL (USD)'] / df['Week Trades']).round(2)
            df['Week PNL/Trade'] = df['Week PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)

        if 'All Time PNL (USD)' in df.columns and 'All Time Trades' in df.columns:
            df['All Time PNL/Trade'] = (df['All Time PNL (USD)'] / df['All Time Trades']).round(2)
            df['All Time PNL/Trade'] = df['All Time PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)

        # Calculate daily average
        if 'All Time PNL (USD)' in df.columns:
            # Assuming All Time is 5 years (1825 days) - this is an approximation
            df['Avg Daily PNL'] = (df['All Time PNL (USD)'] / 1825).round(2)

        return df

    # Format the display DataFrame
    display_df = format_display_df(display_df)

    # Sort DataFrame by Today's PNL (descending)
    display_df = display_df.sort_values(by='Today PNL (USD)', ascending=False)

    # Function to color cells based on value
    def color_pnl_cells(val):
        """Color cells based on PNL value."""
        if pd.isna(val) or val == 0:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
        elif val < -1000:  # Large negative PNL (loss) - red
            return 'background-color: rgba(255, 0, 0, 0.9); color: white'
        elif val < 0:  # Small negative PNL (loss) - light red
            intensity = max(0, min(255, int(255 * abs(val) / 1000)))
            return f'background-color: rgba(255, {180-intensity}, {180-intensity}, 0.7); color: black'
        elif val < 1000:  # Small positive PNL (profit) - light green
            intensity = max(0, min(255, int(255 * val / 1000)))
            return f'background-color: rgba({180-intensity}, 255, {180-intensity}, 0.7); color: black'
        else:  # Large positive PNL (profit) - green
            return 'background-color: rgba(0, 200, 0, 0.8); color: black'

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Main Dashboard", "Detailed View", "Statistics & Insights"])

    with tab1:
        # Main Dashboard View
        st.subheader("PNL Overview by Trading Pair")

        # Create a simplified display DataFrame for the main dashboard
        main_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']]

        # Apply styling
        styled_df = main_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Week PNL (USD)': '${:,.2f}',
            'All Time PNL (USD)': '${:,.2f}'
        })

        # Display the styled DataFrame
        st.dataframe(styled_df, height=600, use_container_width=True)

        # Create summary cards
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_today_pnl = display_df['Today PNL (USD)'].sum()
            st.metric(
                "Total Today PNL", 
                f"${total_today_pnl:,.2f}", 
                delta=f"{(total_today_pnl - display_df['Yesterday PNL (USD)'].sum()):,.2f}"
            )

        with col2:
            total_yesterday_pnl = display_df['Yesterday PNL (USD)'].sum()
            st.metric(
                "Total Yesterday PNL", 
                f"${total_yesterday_pnl:,.2f}"
            )

        with col3:
            total_week_pnl = display_df['Week PNL (USD)'].sum()
            daily_avg = total_week_pnl / 7
            st.metric(
                "Week PNL (7 days)", 
                f"${total_week_pnl:,.2f}",
                delta=f"${daily_avg:,.2f}/day"
            )

        with col4:
            total_all_time_pnl = display_df['All Time PNL (USD)'].sum()
            st.metric(
                "All Time PNL", 
                f"${total_all_time_pnl:,.2f}"
            )

        # Create a visualization of top and bottom performers today
        st.subheader("Today's Top Performers")

        # Filter out zero PNL pairs
        non_zero_today = display_df[display_df['Today PNL (USD)'] != 0].copy()

        # Get top 5 and bottom 5 performers
        top_5 = non_zero_today.nlargest(5, 'Today PNL (USD)')
        bottom_5 = non_zero_today.nsmallest(5, 'Today PNL (USD)')

        # Plot top and bottom performers
        fig = go.Figure()

        # Top performers
        fig.add_trace(go.Bar(
            x=top_5['Trading Pair'],
            y=top_5['Today PNL (USD)'],
            name='Top Performers',
            marker_color='green'
        ))

        # Bottom performers
        fig.add_trace(go.Bar(
            x=bottom_5['Trading Pair'],
            y=bottom_5['Today PNL (USD)'],
            name='Bottom Performers',
            marker_color='red'
        ))

        fig.update_layout(
            title="Top and Bottom Performers Today",
            xaxis_title="Trading Pair",
            yaxis_title="PNL (USD)",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Detailed View
        st.subheader("Detailed PNL and Trade Data")

        # Create a detailed display DataFrame
        detailed_df = display_df[['Trading Pair', 
                                  'Today PNL (USD)', 'Today Trades', 'Today PNL/Trade',
                                  'Yesterday PNL (USD)', 'Yesterday Trades', 'Yesterday PNL/Trade',
                                  'Week PNL (USD)', 'Week Trades', 'Week PNL/Trade',
                                  'All Time PNL (USD)', 'All Time Trades', 'All Time PNL/Trade']]

        # Apply styling
        styled_detailed_df = detailed_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)',
                    'Today PNL/Trade', 'Yesterday PNL/Trade', 'Week PNL/Trade', 'All Time PNL/Trade']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Week PNL (USD)': '${:,.2f}',
            'All Time PNL (USD)': '${:,.2f}',
            'Today PNL/Trade': '${:,.2f}',
            'Yesterday PNL/Trade': '${:,.2f}',
            'Week PNL/Trade': '${:,.2f}',
            'All Time PNL/Trade': '${:,.2f}'
        })

        # Display the styled DataFrame
        st.dataframe(styled_detailed_df, height=600, use_container_width=True)

        # Show day-to-day comparison
        st.subheader("Day-to-Day PNL Comparison")

        day_comparison_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)']].copy()
        day_comparison_df['Day-to-Day Change'] = day_comparison_df['Today PNL (USD)'] - day_comparison_df['Yesterday PNL (USD)']
        day_comparison_df['Yesterday Change'] = day_comparison_df['Yesterday PNL (USD)'] - day_comparison_df['Day Before PNL (USD)']

        # Sort by day-to-day change
        day_comparison_df = day_comparison_df.sort_values(by='Day-to-Day Change', ascending=False)

        # Apply styling
        styled_day_comparison = day_comparison_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Day-to-Day Change', 'Yesterday Change']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Day Before PNL (USD)': '${:,.2f}',
            'Day-to-Day Change': '${:,.2f}',
            'Yesterday Change': '${:,.2f}'
        })

        # Display the styled DataFrame
        st.dataframe(styled_day_comparison, height=400, use_container_width=True)

        # Create a visualization for day-to-day comparison
        # Get top 10 pairs by absolute day-to-day change
        top_change = day_comparison_df.reindex(day_comparison_df['Day-to-Day Change'].abs().sort_values(ascending=False).index).head(10)

        fig = go.Figure()

        # Add bars for each day
        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Day Before PNL (USD)'],
            name='Day Before Yesterday',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Yesterday PNL (USD)'],
            name='Yesterday',
            marker_color='royalblue'
        ))

        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Today PNL (USD)'],
            name='Today',
            marker_color='darkblue'
        ))

        fig.update_layout(
            title="Top 10 Pairs by Change - 3-Day PNL Comparison",
            xaxis_title="Trading Pair",
            yaxis_title="PNL (USD)",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Statistics & Insights
        st.subheader("PNL Statistics and Insights")

        # Create a statistics DataFrame
        stats_df = pd.DataFrame({
            'Metric': [
                'Total Trading Pairs',
                'Profitable Pairs Today',
                'Unprofitable Pairs Today',
                'Most Profitable Pair Today',
                'Least Profitable Pair Today',
                'Highest PNL/Trade Today',
                'Average PNL/Trade (All Pairs)',
                'Total Platform PNL Today',
                'Total Platform PNL Yesterday',
                'Week-to-Date PNL',
                'Estimated Monthly PNL (based on week)',
                'Total Trades Today',
                'Total Trades Yesterday',
                'Week-to-Date Trades'
            ],
            'Value': [
                len(display_df),
                len(display_df[display_df['Today PNL (USD)'] > 0]),
                len(display_df[display_df['Today PNL (USD)'] < 0]),
                display_df.loc[display_df['Today PNL (USD)'].idxmax()]['Trading Pair'] if not display_df.empty else 'N/A',
                display_df.loc[display_df['Today PNL (USD)'].idxmin()]['Trading Pair'] if not display_df.empty else 'N/A',
                f"${display_df['Today PNL/Trade'].max():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
                f"${display_df['Today PNL/Trade'].mean():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
                f"${display_df['Today PNL (USD)'].sum():.2f}",
                f"${display_df['Yesterday PNL (USD)'].sum():.2f}",
                f"${display_df['Week PNL (USD)'].sum():.2f}",
                f"${(display_df['Week PNL (USD)'].sum() / 7 * 30):.2f}",
                f"{display_df['Today Trades'].sum():,}",
                f"{display_df['Yesterday Trades'].sum():,}",
                f"{display_df['Week Trades'].sum():,}"
            ]
        })

        # Display statistics
        st.dataframe(stats_df, hide_index=True, height=400, use_container_width=True)

        # Visualize PNL breakdown by time period
        st.subheader("PNL Breakdown by Time Period")

        # For Top 10 Pairs
        top_10_pairs = display_df.nlargest(10, 'Week PNL (USD)')['Trading Pair'].tolist()
        top_10_df = display_df[display_df['Trading Pair'].isin(top_10_pairs)].copy()

        # Prepare data for stacked bar chart
        chart_data = []
        for pair in top_10_pairs:
            pair_data = top_10_df[top_10_df['Trading Pair'] == pair].iloc[0]
            chart_data.append({
                'Trading Pair': pair,
                'Today': pair_data['Today PNL (USD)'],
                'Yesterday': pair_data['Yesterday PNL (USD)'],
                'Rest of Week': pair_data['Week PNL (USD)'] - pair_data['Today PNL (USD)'] - pair_data['Yesterday PNL (USD)']
            })

        chart_df = pd.DataFrame(chart_data)

        # Create the stacked bar chart
        fig = px.bar(
            chart_df,
            x='Trading Pair',
            y=['Today', 'Yesterday', 'Rest of Week'],
            title='PNL Breakdown for Top 10 Pairs',
            labels={'value': 'PNL (USD)', 'variable': 'Time Period'},
            barmode='group'
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Visualize cumulative PNL distribution
        st.subheader("Cumulative PNL Distribution")

        # Prepare data for cumulative chart
        sorted_df = display_df.sort_values(by='All Time PNL (USD)', ascending=False).copy()
        sorted_df['Cumulative PNL'] = sorted_df['All Time PNL (USD)'].cumsum()
        sorted_df['Contribution %'] = (sorted_df['All Time PNL (USD)'] / sorted_df['All Time PNL (USD)'].sum() * 100)
        sorted_df['Cumulative Contribution %'] = sorted_df['Contribution %'].cumsum()

        # Create the cumulative chart
        fig = go.Figure()

        # Add the bar chart for individual contribution
        fig.add_trace(go.Bar(
            x=sorted_df['Trading Pair'].head(20),
            y=sorted_df['All Time PNL (USD)'].head(20),
            name='Individual PNL',
            marker_color='lightblue'
        ))

        # Add the line chart for cumulative contribution
        fig.add_trace(go.Scatter(
            x=sorted_df['Trading Pair'].head(20),
            y=sorted_df['Cumulative Contribution %'].head(20),
            name='Cumulative Contribution %',
            yaxis='y2',
            marker_color='darkblue',
            line=dict(width=3)
        ))

        # Update layout with two y-axes
        fig.update_layout(
            title="Top 20 Pairs - Individual and Cumulative PNL Contribution",
            xaxis=dict(title="Trading Pair"),
            yaxis=dict(
                title="Individual PNL (USD)",
                titlefont=dict(color="lightblue"),
                tickfont=dict(color="lightblue")
            ),
            yaxis2=dict(
                title="Cumulative Contribution %",
                titlefont=dict(color="darkblue"),
                tickfont=dict(color="darkblue"),
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show performance trends
        st.subheader("PNL Trend Analysis")

        # Calculate week-over-week change
        trend_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)']].copy()
        trend_df['Daily Avg This Week'] = trend_df['Week PNL (USD)'] / 7
        trend_df['3-Day Total'] = trend_df['Today PNL (USD)'] + trend_df['Yesterday PNL (USD)'] + trend_df['Day Before PNL (USD)']
        trend_df['3-Day Daily Avg'] = trend_df['3-Day Total'] / 3
        trend_df['Performance Trend'] = (trend_df['3-Day Daily Avg'] / trend_df['Daily Avg This Week']).round(2)

        # Remove pairs with no activity
        trend_df = trend_df[trend_df['Week PNL (USD)'] != 0].copy()

        # Sort by performance trend (descending)
        trend_df = trend_df.sort_values(by='Performance Trend', ascending=False)

        # Apply styling
        styled_trend_df = trend_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)', '3-Day Total', '3-Day Daily Avg', 'Daily Avg This Week']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Day Before PNL (USD)': '${:,.2f}',
            'Week PNL (USD)': '${:,.2f}',
            '3-Day Total': '${:,.2f}',
            '3-Day Daily Avg': '${:,.2f}',
            'Daily Avg This Week': '${:,.2f}',
            'Performance Trend': '{:.2f}x'
        })

        # Add conditional formatting for Performance Trend
        def color_trend(val):
            if val > 1.5:
                return 'background-color: rgba(0, 200, 0, 0.8); color: black; font-weight: bold'
            elif val > 1.1:
                return 'background-color: rgba(150, 255, 150, 0.7); color: black'
            elif val < 0.5:
                return 'background-color: rgba(255, 0, 0, 0.9); color: white; font-weight: bold'
            elif val < 0.9:
                return 'background-color: rgba(255, 150, 150, 0.7); color: black'
            else:
                return 'background-color: rgba(255, 255, 200, 0.7); color: black'  # Neutral/stable

        styled_trend_df = styled_trend_df.applymap(color_trend, subset=['Performance Trend'])

        # Display the trend analysis
        st.markdown("### Recent Performance Trends (3-Day vs Weekly Average)")
        st.markdown("Performance Trend > 1: Recent performance better than weekly average")
        st.markdown("Performance Trend < 1: Recent performance worse than weekly average")

        st.dataframe(styled_trend_df, height=500, use_container_width=True)

        # Add explanatory text
        with st.expander("Understanding the PNL Dashboard"):
            st.markdown("""
            ## 📊 How to Use This PNL Dashboard

            This dashboard shows platform profit and loss (PNL) across all selected trading pairs over different time periods.

            ### Time Periods
            - **Today**: From midnight Singapore time (SGT) until now
            - **Yesterday**: Full 24 hours from midnight to midnight SGT
            - **This Week**: Last 7 days including today
            - **All Time**: Cumulative PNL since records began

            ### Color Coding
            - 🟩 **Green**: Profit (darker green for higher profits)
            - 🟥 **Red**: Loss (darker red for higher losses)
            - ⬜ **Grey**: No activity/zero PNL

            ### Key Metrics
            - **PNL (USD)**: Platform's profit/loss in USD for each time period
            - **Trades**: Number of trades executed in each time period
            - **PNL/Trade**: Average profit per trade

            ### Dashboard Tabs
            - **Main Dashboard**: Quick overview of PNL by time period
            - **Detailed View**: Complete breakdown including trade counts and per-trade metrics
            - **Statistics & Insights**: Trends, correlations, and deeper analysis

            ### Performance Trends
            - **Performance Trend > 1**: Recent performance (3-day avg) is better than the weekly average
            - **Performance Trend < 1**: Recent performance is worse than the weekly average

            ### Technical Details
            - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
            - All values are shown in USD
            - The dashboard refreshes when you click the "Refresh Data" button
            - Singapore timezone (UTC+8) is used throughout
            """
            )

    # Add footer with last update time
    st.markdown("---")
    st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")

def render_pnl_and_trades():
    # Save this as pages/06_Trades_PNL_Table.py in your Streamlit app folder
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz

    st.set_page_config(
        page_title="User Trades & Platform PNL Table",
        page_icon="💰",
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
    st.title("User Trades & Platform PNL Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    expected_points = 48  # Expected data points per pair over 24 hours (24 hours * 2 intervals per hour)
    singapore_timezone = pytz.timezone('Asia/Singapore')

    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

    # Set thresholds for highlighting
    high_trade_count_threshold = 100  # Number of trades considered "high activity"
    high_pnl_threshold = 1000  # Platform PNL amount considered "high" (in USD)
    low_pnl_threshold = -1000  # Platform PNL amount considered "low" (in USD)

    # Fetch all available pairs from DB
    @st.cache_data(ttl=600, show_spinner="Fetching pairs...")
    def fetch_all_pairs():
        query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                st.error("No pairs found in the database.")
                return []
            return df['pair_name'].tolist()
        except Exception as e:
            st.error(f"Error fetching pairs: {e}")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

    all_pairs = fetch_all_pairs()

    # UI Controls
    col1, col2 = st.columns([3, 1])

    with col1:
        # Let user select pairs to display (or select all)
        select_all = st.checkbox("Select All Pairs", value=True)

        if select_all:
            selected_pairs = all_pairs
        else:
            selected_pairs = st.multiselect(
                "Select Pairs", 
                all_pairs,
                default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
            )

    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()

    if not selected_pairs:
        st.warning("Please select at least one pair")
        st.stop()

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

    # Fetch trades data for the past 24 hours in 30min intervals
    @st.cache_data(ttl=600, show_spinner="Fetching trade counts...")
    def fetch_trade_counts(pair_name):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)

        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # Updated query to use trade_fill_fresh with consistent time handling
        # Explicitly adding 8 hours to UTC timestamps to match Singapore time
        query = f"""
        SELECT
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
            INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30) 
            AS timestamp,
            COUNT(*) AS trade_count
        FROM public.trade_fill_fresh
        WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
        AND taker_way IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
        GROUP BY
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
            INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30)
        ORDER BY timestamp
        """

        try:
            print(f"[{pair_name}] Executing trade count query")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] Trade count query executed. DataFrame shape: {df.shape}")

            if df.empty:
                print(f"[{pair_name}] No trade data found.")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Format time label to match our aligned blocks (HH:MM format)
            df['time_label'] = df.index.strftime('%H:%M')

            return df
        except Exception as e:
            st.error(f"Error processing trade counts for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing trade counts: {e}")
            return None

    # Fetch platform PNL data for the past 24 hours in 30min intervals
    @st.cache_data(ttl=600, show_spinner="Calculating platform PNL...")
    def fetch_platform_pnl(pair_name):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)

        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # This query combines order PNL, fee data, funding PNL, and rebate data in 30-minute intervals
        query = f"""
        WITH time_intervals AS (
          -- Generate 30-minute intervals for the past 24 hours
          SELECT
            generate_series(
              date_trunc('hour', '{start_time_utc}'::timestamp) + 
              INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 30),
              '{end_time_utc}'::timestamp,
              INTERVAL '30 minutes'
            ) AS "UTC+8"
        ),

        order_pnl AS (
          -- Calculate platform order PNL
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (0, 1, 2, 3, 4)
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),

        fee_data AS (
          -- Calculate user fee payments
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_fee_mode" = 1
            AND "taker_way" IN (1, 3)
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),

        funding_pnl AS (
          -- Calculate platform funding fee PNL
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" = 0
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),

        rebate_data AS (
          -- Calculate platform rebate payments
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
          FROM
            "public"."user_cashbooks"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "remark" = '给邀请人返佣'
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        )

        -- Final query: combine all data sources
        SELECT
          t."UTC+8" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp",
          COALESCE(o."platform_order_pnl", 0) +
          COALESCE(f."user_fee_payments", 0) +
          COALESCE(ff."platform_funding_pnl", 0) +
          COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
        FROM
          time_intervals t
        LEFT JOIN
          order_pnl o ON t."UTC+8" = o."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          fee_data f ON t."UTC+8" = f."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          funding_pnl ff ON t."UTC+8" = ff."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          rebate_data r ON t."UTC+8" = r."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        ORDER BY
          t."UTC+8" DESC
        """

        try:
            print(f"[{pair_name}] Executing platform PNL query")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] Platform PNL query executed. DataFrame shape: {df.shape}")

            if df.empty:
                print(f"[{pair_name}] No PNL data found.")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Format time label to match our aligned blocks (HH:MM format)
            df['time_label'] = df.index.strftime('%H:%M')

            return df
        except Exception as e:
            st.error(f"Error processing platform PNL for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing platform PNL: {e}")
            return None

    # Combine Trade Count and Platform PNL data for visualization
    def combine_data(trade_data, pnl_data):
        if trade_data is None and pnl_data is None:
            return None

        # Create a DataFrame with time blocks as index
        time_blocks = pd.DataFrame(index=[block[2] for block in aligned_time_blocks])

        # Add trade count data if available
        if trade_data is not None and not trade_data.empty:
            for time_label in time_blocks.index:
                # Find matching rows in trade_data by time_label
                matching_rows = trade_data[trade_data['time_label'] == time_label]
                if not matching_rows.empty:
                    time_blocks.at[time_label, 'trade_count'] = matching_rows['trade_count'].sum()

        # Add PNL data if available
        if pnl_data is not None and not pnl_data.empty:
            for time_label in time_blocks.index:
                # Find matching rows in pnl_data by time_label
                matching_rows = pnl_data[pnl_data['time_label'] == time_label]
                if not matching_rows.empty:
                    time_blocks.at[time_label, 'platform_pnl'] = matching_rows['platform_total_pnl'].sum()

        # Fill NaN values with 0
        time_blocks.fillna(0, inplace=True)

        return time_blocks

    # Show progress bar while calculating
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Calculate trade count and platform PNL for each pair
    pair_results = {}
    for i, pair_name in enumerate(selected_pairs):
        try:
            progress_bar.progress((i) / len(selected_pairs))
            status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")

            # Fetch trade count data
            trade_data = fetch_trade_counts(pair_name)

            # Fetch platform PNL data
            pnl_data = fetch_platform_pnl(pair_name)

            # Combine data
            combined_data = combine_data(trade_data, pnl_data)

            if combined_data is not None:
                pair_results[pair_name] = combined_data
        except Exception as e:
            st.error(f"Error processing pair {pair_name}: {e}")
            print(f"Error processing pair {pair_name} in main loop: {e}")

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

    # Create tables for display - Trade Count Table
    if pair_results:
        # Create trade count table data
        trade_count_data = {}
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns:
                trade_count_data[pair_name] = df['trade_count']

        # Create DataFrame with all pairs
        trade_count_table = pd.DataFrame(trade_count_data)

        # Apply the time blocks in the proper order (most recent first)
        available_times = set(trade_count_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]

        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)

        # Reindex with the ordered times
        trade_count_table = trade_count_table.reindex(ordered_times)

        # Round to integers - trade counts should be whole numbers
        trade_count_table = trade_count_table.round(0).astype('Int64')  # Using Int64 to handle NaN values properly

        def color_trade_cells(val):
            if pd.isna(val) or val == 0:
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
            elif val < 10:  # Low activity
                intensity = max(0, min(255, int(255 * val / 10)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            elif val < 50:  # Medium activity
                intensity = max(0, min(255, int(255 * (val - 10) / 40)))
                return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
            elif val < high_trade_count_threshold:  # High activity
                intensity = max(0, min(255, int(255 * (val - 50) / 50)))
                return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
            else:  # Very high activity
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'

        styled_trade_table = trade_count_table.style.applymap(color_trade_cells)
        st.markdown("## User Trades Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:green'>Low Activity</span>, <span style='color:#aaaa00'>Medium Activity</span>, <span style='color:orange'>High Activity</span>, <span style='color:red'>Very High Activity</span>", unsafe_allow_html=True)
        st.markdown("Values shown as number of trades per 30-minute period")
        st.dataframe(styled_trade_table, height=700, use_container_width=True)

        # Create Platform PNL table data
        pnl_data = {}
        for pair_name, df in pair_results.items():
            if 'platform_pnl' in df.columns:
                if df['platform_pnl'].abs().sum() > 0:
                    pnl_data[pair_name] = df['platform_pnl']

        # Create DataFrame with all pairs
        pnl_table = pd.DataFrame(pnl_data)

        # Apply the time blocks in the proper order (most recent first)
        pnl_table = pnl_table.reindex(ordered_times)

        # Round to 2 decimal places for display
        pnl_table = pnl_table.round(0).astype(int)

        def color_pnl_cells(val):
            if pd.isna(val) or val == 0:
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
            elif val < low_pnl_threshold:  # Large negative PNL (loss) - red
                return f'background-color: rgba(255, 0, 0, 0.9); color: white'
            elif val < 0:  # Small negative PNL (loss) - light red
                intensity = max(0, min(255, int(255 * abs(val) / abs(low_pnl_threshold))))
                return f'background-color: rgba(255, {100-intensity}, {100-intensity}, 0.9); color: black'
            elif val < high_pnl_threshold:  # Small positive PNL (profit) - light green
                intensity = max(0, min(255, int(255 * val / high_pnl_threshold)))
                return f'background-color: rgba({100-intensity}, 180, {100-intensity}, 0.9); color: black'
            else:  # Large positive PNL (profit) - green
                return 'background-color: rgba(0, 120, 0, 0.7); color: black'

        styled_pnl_table = pnl_table.style.applymap(color_pnl_cells)
        st.markdown("## Platform PNL Table (USD, 30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:red'>Loss</span>, <span style='color:#ff9999'>Small Loss</span>, <span style='color:#99ff99'>Small Profit</span>, <span style='color:green'>Large Profit</span>", unsafe_allow_html=True)
        st.markdown("Values shown in USD")
        st.dataframe(styled_pnl_table, height=700, use_container_width=True)

        # Create summary tables with improved legibility
        st.subheader("Summary Statistics (Last 24 Hours)")

        # Add a separator
        st.markdown("---")

        # Prepare Trades Summary data
        trades_summary = {}
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns:
                total_trades = df['trade_count'].sum()
                max_trades = df['trade_count'].max()
                max_trades_time = df['trade_count'].idxmax() if max_trades > 0 else "N/A"

                trades_summary[pair_name] = {
                    'Total Trades': int(total_trades),
                    'Max Trades in 30min': int(max_trades),
                    'Busiest Time': max_trades_time if max_trades > 0 else "N/A",
                    'Avg Trades per 30min': round(df['trade_count'].mean(), 1)
                }

        # Trading Activity Summary with improved formatting
        if trades_summary:
            # Convert to DataFrame and sort
            trades_summary_df = pd.DataFrame(trades_summary).T
            trades_summary_df = trades_summary_df.sort_values(by='Total Trades', ascending=False)

            # Format the dataframe for better legibility
            trades_summary_df = trades_summary_df.rename(columns={
                'Total Trades': '📊 Total Trades',
                'Max Trades in 30min': '⏱️ Max Trades (30min)',
                'Busiest Time': '🕒 Busiest Time',
                'Avg Trades per 30min': '📈 Avg Trades/30min'
            })

            # Add a clear section header
            st.markdown("### 📊 Trading Activity Summary")

            # Use a larger height and make sure the table has proper spacing
            st.dataframe(
                trades_summary_df.style.format({
                    '📈 Avg Trades/30min': '{:.1f}'
                }).set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=350,
                use_container_width=True
            )

        # Add spacing between tables
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Prepare PNL Summary data
        pnl_summary = {}
        for pair_name, df in pair_results.items():
            if 'platform_pnl' in df.columns:
                total_pnl = df['platform_pnl'].sum()
                max_pnl = df['platform_pnl'].max()
                min_pnl = df['platform_pnl'].min()
                max_pnl_time = df['platform_pnl'].idxmax() if abs(max_pnl) > 0 else "N/A"
                min_pnl_time = df['platform_pnl'].idxmin() if abs(min_pnl) > 0 else "N/A"

                pnl_summary[pair_name] = {
                    'Total PNL (USD)': round(total_pnl, 2),
                    'Max Profit in 30min': round(max_pnl, 2),
                    'Max Profit Time': max_pnl_time if abs(max_pnl) > 0 else "N/A",
                    'Max Loss in 30min': round(min_pnl, 2),
                    'Max Loss Time': min_pnl_time if abs(min_pnl) > 0 else "N/A",
                    'Avg PNL per 30min': round(df['platform_pnl'].mean(), 2)
                }

        # PNL Summary with improved formatting
        if pnl_summary:
            # Convert to DataFrame and sort
            pnl_summary_df = pd.DataFrame(pnl_summary).T
            pnl_summary_df = pnl_summary_df.sort_values(by='Total PNL (USD)', ascending=False)

            # Format the dataframe for better legibility
            pnl_summary_df = pnl_summary_df.rename(columns={
                'Total PNL (USD)': '💰 Total PNL (USD)',
                'Max Profit in 30min': '📈 Max Profit (30min)',
                'Max Profit Time': '⏱️ Max Profit Time',
                'Max Loss in 30min': '📉 Max Loss (30min)',
                'Max Loss Time': '⏱️ Max Loss Time', 
                'Avg PNL per 30min': '📊 Avg PNL/30min'
            })

            # Add a clear section header
            st.markdown("### 💰 Platform PNL Summary")

            # Style the dataframe for better legibility
            styled_pnl_df = pnl_summary_df.style.format({
                '💰 Total PNL (USD)': '${:,.2f}',
                '📈 Max Profit (30min)': '${:,.2f}',
                '📉 Max Loss (30min)': '${:,.2f}',
                '📊 Avg PNL/30min': '${:,.2f}'
            })

            # Apply conditional formatting
            def highlight_profits(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                return ''

            styled_pnl_df = styled_pnl_df.applymap(highlight_profits, subset=['💰 Total PNL (USD)', '📈 Max Profit (30min)', '📉 Max Loss (30min)', '📊 Avg PNL/30min'])

            # Use a larger height and make sure the table has proper spacing
            st.dataframe(
                styled_pnl_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=350,
                use_container_width=True
            )

        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Combined Analysis with improved formatting
        st.subheader("Combined Analysis: Trading Activity vs. Platform PNL")
        # Extract periods with both high activity and significant PNL
        high_activity_periods = []
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                # Find time periods with both significant activity and significant PNL
                for time_label, row in df.iterrows():
                    trade_count = row['trade_count']
                    pnl = row['platform_pnl']

                    # Check if this is a noteworthy period
                    if (trade_count >= high_trade_count_threshold or 
                        pnl >= high_pnl_threshold or pnl <= low_pnl_threshold):
                        high_activity_periods.append({
                            'Pair': pair_name,
                            'Time': time_label,
                            'Trade Count': int(trade_count),
                            'Platform PNL (USD)': round(pnl, 2),
                            'Revenue per Trade (USD)': round(pnl / trade_count, 2) if trade_count > 0 else 0
                        })

        # Extract periods with both high activity and significant PNL with better formatting
        if high_activity_periods:
            # Convert to DataFrame
            high_activity_df = pd.DataFrame(high_activity_periods)

            # Rename columns for clarity
            high_activity_df = high_activity_df.rename(columns={
                'Pair': '🔄 Trading Pair',
                'Time': '⏰ Time Period',
                'Trade Count': '📊 Number of Trades',
                'Platform PNL (USD)': '💰 PNL (USD)',
                'Revenue per Trade (USD)': '💸 PNL/Trade (USD)'
            })

            # Sort by Trade Count (highest first)
            high_activity_df = high_activity_df.sort_values(by='📊 Number of Trades', ascending=False)

            # Style the dataframe
            styled_activity_df = high_activity_df.style.format({
                '💰 PNL (USD)': '${:,.2f}',
                '💸 PNL/Trade (USD)': '${:,.2f}'
            })

            # Apply conditional formatting
            def highlight_pnl(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                return ''

            styled_activity_df = styled_activity_df.applymap(highlight_pnl, subset=['💰 PNL (USD)', '💸 PNL/Trade (USD)'])

            # Add a clear section header
            st.markdown("### 🔍 High Activity Periods")

            # Display the dataframe with improved styling
            st.dataframe(
                styled_activity_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=350,
                use_container_width=True
            )

            # Create visual representation
            col1, col2 = st.columns(2)

            with col1:
                # Top 10 trading periods by volume
                top_trading_periods = high_activity_df.head(10)
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_trading_periods.iterrows()],
                        y=top_trading_periods['📊 Number of Trades'],
                        marker_color='blue'
                    )
                ])
                fig.update_layout(
                    title="Top 10 Trading Periods by Volume",
                    xaxis_title="Pair and Time",
                    yaxis_title="Number of Trades",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Top 10 PNL periods
                top_pnl_periods = high_activity_df.sort_values(by='💰 PNL (USD)', ascending=False).head(10)
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_pnl_periods.iterrows()],
                        y=top_pnl_periods['💰 PNL (USD)'],
                        marker_color='green'
                    )
                ])
                fig.update_layout(
                    title="Top 10 Trading Periods by Platform PNL",
                    xaxis_title="Pair and Time",
                    yaxis_title="Platform PNL (USD)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Overall Trade Count vs. PNL correlation analysis
            st.subheader("Correlation Analysis: Trade Count vs. Platform PNL")

            correlation_data = []
            for pair_name, df in pair_results.items():
                if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                    # Calculate correlation between trade count and PNL
                    correlation = df['trade_count'].corr(df['platform_pnl'])
                    # Filter out rows with zero trades
                    non_zero_trades = df[df['trade_count'] > 0]
                    # Calculate average PNL per trade
                    avg_pnl_per_trade = non_zero_trades['platform_pnl'].sum() / non_zero_trades['trade_count'].sum() if non_zero_trades['trade_count'].sum() > 0 else 0

                    correlation_data.append({
                        'Pair': pair_name,
                        'Correlation': round(correlation, 3) if not pd.isna(correlation) else 0,
                        'Total Trades': int(df['trade_count'].sum()),
                        'Total PNL (USD)': round(df['platform_pnl'].sum(), 2),
                        'Avg PNL per Trade (USD)': round(avg_pnl_per_trade, 3)
                    })

            if correlation_data:
                # Convert to DataFrame
                correlation_df = pd.DataFrame(correlation_data)

                # Sort by correlation (highest first)
                correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

                # Format the dataframe for better legibility
                correlation_df = correlation_df.rename(columns={
                    'Pair': '🔄 Trading Pair',
                    'Correlation': '📊 Correlation',
                    'Total Trades': '📈 Total Trades',
                    'Total PNL (USD)': '💰 Total PNL (USD)',
                    'Avg PNL per Trade (USD)': '💸 Avg PNL/Trade (USD)'
                })

                # Style the dataframe
                styled_correlation_df = correlation_df.style.format({
                    '📊 Correlation': '{:.3f}',
                    '💰 Total PNL (USD)': '${:,.2f}',
                    '💸 Avg PNL/Trade (USD)': '${:,.3f}'
                })

                # Apply conditional formatting
                def highlight_correlation(val):
                    if isinstance(val, (int, float)):
                        if val > 0.5:
                            return 'color: green; font-weight: bold'
                        elif val < -0.5:
                            return 'color: red; font-weight: bold'
                    return ''

                styled_correlation_df = styled_correlation_df.applymap(highlight_correlation, subset=['📊 Correlation'])
                styled_correlation_df = styled_correlation_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 Avg PNL/Trade (USD)'])

                # Create columns for display
                col1, col2 = st.columns([2, 3])

                with col1:
                    # Display the correlation table with improved styling
                    st.markdown("### 📊 Trade Count vs. PNL Correlation by Pair")
                    st.dataframe(
                        styled_correlation_df.set_properties(**{
                            'font-size': '16px',
                            'text-align': 'center',
                            'background-color': '#f0f2f6'
                        }),
                        height=400,
                        use_container_width=True
                    )

                with col2:
                    # Create a scatter plot to visualize the correlation
                    # Gather all data points for the scatter plot
                    scatter_data = []
                    for pair_name, df in pair_results.items():
                        if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                            for time_label, row in df.iterrows():
                                if row['trade_count'] > 0:  # Only include periods with trades
                                    scatter_data.append({
                                        'Pair': pair_name,
                                        'Trade Count': int(row['trade_count']),
                                        'Platform PNL (USD)': round(row['platform_pnl'], 2)
                                    })

                    if scatter_data:
                        scatter_df = pd.DataFrame(scatter_data)

                        # Create a scatter plot using Plotly
                        fig = px.scatter(
                            scatter_df, 
                            x='Trade Count', 
                            y='Platform PNL (USD)',
                            color='Pair',
                            title='Trade Count vs. Platform PNL Correlation',
                            hover_data=['Pair', 'Trade Count', 'Platform PNL (USD)'],

                        )

                        fig.update_layout(
                            height=500,
                            xaxis_title="Number of Trades",
                            yaxis_title="Platform PNL (USD)",
                        )

                        st.plotly_chart(fig, use_container_width=True)

        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Time-based Analysis
        st.subheader("Time-based Analysis")

        # Analyze trading patterns over time
        hourly_patterns = {}
        for time_block in ordered_times:
            # Extract hour from time block
            hour = int(time_block.split(':')[0])

            # Initialize if this hour is not yet in the dictionary
            if hour not in hourly_patterns:
                hourly_patterns[hour] = {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'count': 0
                }

            # Sum up trades and PNL for this hour across all pairs
            for pair_name, df in pair_results.items():
                if time_block in df.index:
                    if 'trade_count' in df.columns:
                        hourly_patterns[hour]['total_trades'] += df.at[time_block, 'trade_count']
                    if 'platform_pnl' in df.columns:
                        hourly_patterns[hour]['total_pnl'] += df.at[time_block, 'platform_pnl']
                    hourly_patterns[hour]['count'] += 1

        # Convert to DataFrame for display
        hourly_patterns_df = pd.DataFrame([
            {
                'Hour (SG Time)': f"{hour:02d}:00-{hour:02d}:59",
                'Avg Trades': round(data['total_trades'] / data['count'] if data['count'] > 0 else 0, 1),
                'Avg PNL (USD)': round(data['total_pnl'] / data['count'] if data['count'] > 0 else 0, 2),
                'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'] if data['total_trades'] > 0 else 0, 3)
            }
            for hour, data in hourly_patterns.items()
        ])

        # Sort by hour for display
        hourly_patterns_df = hourly_patterns_df.sort_values(by='Hour (SG Time)')

        # Format the dataframe for better legibility
        hourly_patterns_df = hourly_patterns_df.rename(columns={
            'Hour (SG Time)': '🕒 Hour (SG Time)',
            'Avg Trades': '📊 Avg Trades',
            'Avg PNL (USD)': '💰 Avg PNL (USD)',
            'PNL per Trade (USD)': '💸 PNL/Trade (USD)'
        })

        # Style the dataframe
        styled_hourly_df = hourly_patterns_df.style.format({
            '📊 Avg Trades': '{:.1f}',
            '💰 Avg PNL (USD)': '${:,.2f}',
            '💸 PNL/Trade (USD)': '${:,.3f}'
        })

        # Apply conditional formatting
        styled_hourly_df = styled_hourly_df.applymap(highlight_pnl, subset=['💰 Avg PNL (USD)', '💸 PNL/Trade (USD)'])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🕒 Hourly Trading Patterns")
            st.dataframe(
                styled_hourly_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=500,
                use_container_width=True
            )

        with col2:
            # Create charts for hourly patterns
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=hourly_patterns_df['🕒 Hour (SG Time)'],
                y=hourly_patterns_df['📊 Avg Trades'],
                name='Avg Trades',
                marker_color='blue'
            ))

            fig.add_trace(go.Scatter(
                x=hourly_patterns_df['🕒 Hour (SG Time)'],
                y=hourly_patterns_df['💰 Avg PNL (USD)'],
                name='Avg PNL (USD)',
                yaxis='y2',
                mode='lines+markers',
                marker_color='green',
                line=dict(width=3)
            ))

            # Update layout with two y-axes
            fig.update_layout(
                title="Hourly Trading Activity and PNL (Singapore Time)",
                xaxis=dict(title="Hour"),
                yaxis=dict(
                    title="Avg Number of Trades",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue")
                ),
                yaxis2=dict(
                    title="Avg PNL (USD)",
                    titlefont=dict(color="green"),
                    tickfont=dict(color="green"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # PNL Breakdown Analysis
        st.subheader("Platform Profit Distribution")

        # Calculate platform total profit
        total_platform_profit = sum(df['platform_pnl'].sum() for pair, df in pair_results.items() if 'platform_pnl' in df.columns)

        # Calculate per-pair contribution to total profit
        profit_distribution = []
        for pair_name, df in pair_results.items():
            if 'platform_pnl' in df.columns:
                pair_pnl = df['platform_pnl'].sum()
                contribution_pct = 100 * pair_pnl / total_platform_profit if total_platform_profit != 0 else 0

                profit_distribution.append({
                    'Pair': pair_name,
                    'Total PNL (USD)': round(pair_pnl, 0),
                    'Contribution (%)': round(contribution_pct, 2)
                })

        if profit_distribution:
            # Sort by total PNL (highest first)
            profit_distribution_df = pd.DataFrame(profit_distribution)
            profit_distribution_df = profit_distribution_df.sort_values(by='Total PNL (USD)', ascending=False)

            # Format the dataframe for better legibility
            profit_distribution_df = profit_distribution_df.rename(columns={
                'Pair': '🔄 Trading Pair',
                'Total PNL (USD)': '💰 Total PNL (USD)',
                'Contribution (%)': '📊 Contribution (%)'
            })

            # Apply styling to make numbers clearer
            styled_profit_df = profit_distribution_df.style.format({
                '💰 Total PNL (USD)': '${:,.0f}',  # Format as integers with comma for thousands
                '📊 Contribution (%)': '{:+.2f}%'  # Show with + or - sign and 2 decimal places
            })

            # Conditionally color the cells based on values
            def color_pnl_and_contribution(val, column):
                if column == '💰 Total PNL (USD)':
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                    return ''
                elif column == '📊 Contribution (%)':
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                    return ''
                return ''

            # Apply the styling function
            styled_profit_df = styled_profit_df.applymap(
                lambda x: color_pnl_and_contribution(x, '💰 Total PNL (USD)'), 
                subset=['💰 Total PNL (USD)']
             ).applymap(
                lambda x: color_pnl_and_contribution(x, '📊 Contribution (%)'), 
                subset=['📊 Contribution (%)']
            )

            # Display the styled dataframe
            st.markdown("### 💰 Profit Contribution by Pair")
            st.dataframe(
                styled_profit_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=500,
                use_container_width=True
            )

            # Create a pie chart visualizing contribution
            top_pairs = profit_distribution_df.head(10)

            # Filter to only positive contributions for the pie chart
            positive_pairs = top_pairs[top_pairs['💰 Total PNL (USD)'] > 0]
            if not positive_pairs.empty:
                fig = px.pie(
                    positive_pairs, 
                    values='💰 Total PNL (USD)', 
                    names='🔄 Trading Pair',
                    title='Top 10 Pairs by Positive PNL Contribution',
                    hole=0.4
                )

                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Identify Most Profitable Time Periods
        st.subheader("Most Profitable Time Periods")

        # Calculate profitability for each time period across all pairs
        time_period_profit = {}
        for time_block in ordered_times:
            time_period_profit[time_block] = {
                'total_pnl': 0,
                'total_trades': 0,
                'pair_breakdown': {}
            }

            for pair_name, df in pair_results.items():
                if time_block in df.index:
                    if 'platform_pnl' in df.columns:
                        pair_pnl = df.at[time_block, 'platform_pnl']
                        time_period_profit[time_block]['total_pnl'] += pair_pnl
                        time_period_profit[time_block]['pair_breakdown'][pair_name] = pair_pnl

                    if 'trade_count' in df.columns:
                        time_period_profit[time_block]['total_trades'] += df.at[time_block, 'trade_count']

        # Convert to DataFrame for display
        time_profit_df = pd.DataFrame([
            {
                'Time Period': time_block,
                'Total PNL (USD)': round(data['total_pnl'], 2),
                'Total Trades': int(data['total_trades']),
                'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'], 3) if data['total_trades'] > 0 else 0,
                'Top Contributing Pair': max(data['pair_breakdown'].items(), key=lambda x: x[1])[0] if data['pair_breakdown'] else "None"
            }
            for time_block, data in time_period_profit.items()
        ])

        # Format the dataframe for better legibility
        time_profit_df = time_profit_df.rename(columns={
            'Time Period': '⏰ Time Period',
            'Total PNL (USD)': '💰 Total PNL (USD)',
            'Total Trades': '📊 Total Trades',
            'PNL per Trade (USD)': '💸 PNL/Trade (USD)',
            'Top Contributing Pair': '🔄 Top Pair'
        })

        # Sort by total PNL (highest first)
        time_profit_df = time_profit_df.sort_values(by='💰 Total PNL (USD)', ascending=False)

        # Style the dataframe
        styled_time_profit_df = time_profit_df.style.format({
            '💰 Total PNL (USD)': '${:,.2f}',
            '💸 PNL/Trade (USD)': '${:,.3f}'
        })

        # Apply conditional formatting
        styled_time_profit_df = styled_time_profit_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 PNL/Trade (USD)'])

        col1, col2 = st.columns([2, 2])

        with col1:
            # Show top profitable time periods
            st.markdown("### 📈 Top 10 Most Profitable Time Periods")

            # Style the entire DataFrame
            full_styled_df = styled_time_profit_df.set_properties(**{
             'font-size': '16px',
             'text-align': 'center',
             'background-color': '#f0f2f6'
        }   )

            # Display only the top 10 rows
            st.dataframe(
               full_styled_df.data.head(10),
               height=300,
               use_container_width=True
            )

        with col2:
        # Show bottom profitable (loss-making) time periods
        st.markdown("### 📉 Top 10 Least Profitable Time Periods")

        # Get the 10 least profitable periods and sort them
        bottom_10 = time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')

        # Style the bottom 10 with the same approach as the top 10
        bottom_styled_df = styled_time_profit_df.set_properties(**{
            'font-size': '16px',
            'text-align': 'center',
            'background-color': '#f0f2f6'
        })

        # Display the filtered data
        st.dataframe(
            bottom_styled_df.data.tail(10).sort_values(by='💰 Total PNL (USD)'),
            height=300,
            use_container_width=True
        )

        # Create visualization of top profitable and loss-making periods
        fig = go.Figure()

        # Top 10 profitable periods
        fig.add_trace(go.Bar(
            x=time_profit_df.head(10)['⏰ Time Period'],
            y=time_profit_df.head(10)['💰 Total PNL (USD)'],
            name='Top Profitable Periods',
            marker_color='green'
        ))

        # Bottom 10 profitable periods
        bottom_10 = time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')
        fig.add_trace(go.Bar(
            x=bottom_10['⏰ Time Period'],
            y=bottom_10['💰 Total PNL (USD)'],
            name='Least Profitable Periods',
            marker_color='red'
        ))

        fig.update_layout(
            title="Most and Least Profitable Time Periods",
            xaxis_title="Time Period (SG Time)",
            yaxis_title="Total PNL (USD)",
            height=500,
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation for dashboard
        with st.expander("Understanding the Trading & PNL Dashboard"):
            st.markdown("""
            ## 📊 How to Use This Dashboard

            This dashboard shows trading activity and platform profit/loss (PNL) across all selected trading pairs using 30-minute intervals over the past 24 hours (Singapore time).

            ### Main Tables
            - **User Trades Table**: Shows the number of trades completed in each 30-minute period
            - **Platform PNL Table**: Shows the platform's profit/loss in each 30-minute period

            ### Color Coding
            - **Trades Table**: 
              - 🟩 Green: Low activity
              - 🟨 Yellow: Medium activity
              - 🟧 Orange: High activity
              - 🟥 Red: Very high activity

            - **PNL Table**: 
              - 🟥 Red: Significant loss
              - 🟠 Light red: Small loss
              - 🟢 Light green: Small profit
              - 🟩 Green: Significant profit

            ### Key Insights
            - **Trading Activity Summary**: See which pairs have the highest trading volume
            - **Platform PNL Summary**: Identify which pairs are most profitable
            - **High Activity Periods**: Spot times of significant trading and PNL outcomes
            - **Correlation Analysis**: Understand the relationship between trade volume and profitability
            - **Time-based Analysis**: Discover trading patterns throughout the day
            - **Profit Distribution**: See which pairs contribute most to overall profit
            - **Time Period Analysis**: Identify the most and least profitable time periods

            ### Technical Details
            - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
            - All values are shown in USD
            - The dashboard refreshes when you click the "Refresh Data" button
            - Singapore timezone (UTC+8) is used throughout
            """
            )

    else:
        st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")

def render_regime_matrix():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone
    import plotly.graph_objects as go
    import plotly.express as px
    from sqlalchemy import create_engine
    import concurrent.futures
    from functools import lru_cache

    # --- Setup ---
    st.set_page_config(layout="wide")
    st.title("📈 Currency Pair Trend Matrix Dashboard")

    # Create tabs for Matrix View, Summary Table, Filters/Settings, and Global Summary
    tab1, tab2, tab3, tab4 = st.tabs(["Matrix View", "Pair-Specific Summary Table", "Filter by Regime", "Global Regime Summary"])

    # --- DB CONFIG ---
    @st.cache_resource
    def get_database_connection():
        db_config = st.secrets["database"]
        db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(db_uri)

    engine = get_database_connection()

    # --- Detailed Regime Color Map with Intensities ---
    color_map = {
        "MEAN-REVERT": {
            3: "rgba(255,0,0,0.7)",      # Strong Mean-Reversion
            2: "rgba(255,50,50,0.6)",    # Moderate Mean-Reversion
            1: "rgba(255,100,100,0.5)",  # Mild Mean-Reversion
            0: "rgba(255,150,150,0.4)"   # (Not used)
        },
        "NOISE": {
            0: "rgba(200,200,200,0.5)",  # Pure Random Walk
            1: "rgba(220,220,255,0.4)"   # Slight bias (either direction)
        },
        "TREND": {
            3: "rgba(0,180,0,0.7)",      # Strong Trend
            2: "rgba(50,200,50,0.6)",    # Moderate Trend
            1: "rgba(100,220,100,0.5)",  # Mild Trend
            0: "rgba(150,255,150,0.4)"   # (Not used)
        }
    }

    # Emoji indicators for regimes
    regime_emojis = {
        "Strong mean-reversion": "⬇️⬇️⬇️",
        "Moderate mean-reversion": "⬇️⬇️",
        "Mild mean-reversion": "⬇️",
        "Slight mean-reversion bias": "↘️",
        "Pure random walk": "↔️",
        "Slight trending bias": "↗️",
        "Mild trending": "⬆️",
        "Moderate trending": "⬆️⬆️",
        "Strong trending": "⬆️⬆️⬆️",
        "Insufficient data": "❓",
    }

    # --- Optimized Hurst & Regime Logic ---
    @lru_cache(maxsize=128)
    def universal_hurst(ts_tuple):
        """
        A universal Hurst exponent calculation that works for any asset class.

        Args:
            ts_tuple: Time series of prices as a tuple (for caching)

        Returns:
            float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
        """
        # Convert tuple to numpy array
        ts = np.array(ts_tuple, dtype=float)

        # Basic data validation
        if len(ts) < 10 or np.any(~np.isfinite(ts)):
            return np.nan

        # Convert to returns - using log returns handles any scale of asset
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        adjusted_ts = ts + epsilon
        log_returns = np.diff(np.log(adjusted_ts))

        # If all returns are exactly zero (completely flat price), return 0.5
        if np.all(log_returns == 0):
            return 0.5

        # Use lag-1 autocorrelation as primary method (fastest)
        try:
            if len(log_returns) > 1:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
                return max(0, min(1, h_acf))  # Constrain to [0,1]
        except:
            pass

        # If autocorrelation fails, return random walk assumption
        return 0.5

    def batch_calculate_hurst(df, window_size):
        """
        Calculate Hurst exponent for all windows at once instead of rolling

        Args:
            df: DataFrame with 'close' column
            window_size: Size of the rolling window

        Returns:
            Series: Hurst values for each window
        """
        n = len(df)
        if n < window_size:
            return pd.Series([np.nan] * n)

        hurst_values = []

        # For first window-1 positions, we don't have enough data
        hurst_values.extend([np.nan] * (window_size - 1))

        # Calculate Hurst for each complete window
        for i in range(window_size - 1, n):
            window = df.iloc[i - window_size + 1:i + 1]['close'].values
            hurst = universal_hurst(tuple(window))
            hurst_values.append(hurst)

        return pd.Series(hurst_values, index=df.index)

    # --- Calculate Hurst confidence ---
    def hurst_confidence(ts):
        """Calculate confidence score for Hurst estimation (0-100%)"""
        ts = np.array(ts)

        # Simple factors affecting confidence
        len_factor = min(1.0, len(ts) / 50)
        var = np.var(ts) if len(ts) > 1 else 0
        var_factor = min(1.0, var / 1e-4) if var > 0 else 0

        # Simple confidence calculation
        confidence = np.mean([len_factor, var_factor]) * 100
        return round(confidence)

    def detailed_regime_classification(hurst):
        """
        Provides a more detailed regime classification including intensity levels.

        Args:
            hurst: Calculated Hurst exponent value

        Returns:
            tuple: (regime category, intensity level, description)
        """
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

    def get_recommended_settings(timeframe):
        """Returns recommended lookback and window settings for a given timeframe"""
        recommendations = {
            "30s": {"lookback_min": 1, "lookback_ideal": 2, "window_min": 30, "window_ideal": 50},
            "15min": {"lookback_min": 2, "lookback_ideal": 3, "window_min": 20, "window_ideal": 30},
            "30min": {"lookback_min": 3, "lookback_ideal": 4, "window_min": 20, "window_ideal": 30},
            "1h": {"lookback_min": 5, "lookback_ideal": 7, "window_min": 20, "window_ideal": 30},
            "4h": {"lookback_min": 10, "lookback_ideal": 14, "window_min": 20, "window_ideal": 30},
            "6h": {"lookback_min": 14, "lookback_ideal": 21, "window_min": 20, "window_ideal": 30}
        }

        return recommendations.get(timeframe, {"lookback_min": 3, "lookback_ideal": 7, "window_min": 20, "window_ideal": 30})

    # --- Bulk data fetching ---
    @st.cache_data(ttl=300)  # Cache data for 5 minutes
    def fetch_price_data_bulk(pairs, lookback_days):
        """
        Fetch price data for multiple pairs in a single query

        Args:
            pairs: List of pairs to fetch
            lookback_days: Days to look back

        Returns:
            DataFrame: Prices for all pairs
        """
        if not pairs:
            return pd.DataFrame()

        # Format the IN clause with proper SQL escaping
        pairs_str = "','".join(pairs)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)

        query = f"""
        SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, 
               final_price, pair_name
        FROM public.oracle_price_log
        WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
        AND pair_name IN ('{pairs_str}');
        """

        try:
            df = pd.read_sql(query, engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()

    def process_pair_data(df, timeframe, rolling_window):
        """
        Process data for a single pair and timeframe

        Args:
            df: DataFrame with price data for a single pair
            timeframe: Timeframe for resampling
            rolling_window: Size of rolling window

        Returns:
            DataFrame: Processed OHLC data with Hurst values
        """
        if df.empty:
            return None

        # Convert timestamp and sort
        df = df.set_index('timestamp').sort_index()

        # Resample to OHLC
        ohlc = df['final_price'].resample(timeframe).ohlc().dropna()

        # If not enough data for the window size, return None
        if len(ohlc) < rolling_window:
            return None

        # Calculate Hurst with optimized batch calculation
        ohlc['Hurst'] = batch_calculate_hurst(ohlc, rolling_window)

        # Calculate confidence only for the last value to save processing
        last_window = ohlc['close'].iloc[-rolling_window:].values if len(ohlc) >= rolling_window else []
        last_confidence = hurst_confidence(last_window) if len(last_window) == rolling_window else 0
        ohlc['confidence'] = 0  # Initialize
        if len(ohlc) >= rolling_window:
            ohlc.iloc[-1, ohlc.columns.get_loc('confidence')] = last_confidence

        # Apply the enhanced regime classification only to the last value
        last_hurst = ohlc['Hurst'].iloc[-1] if not ohlc.empty else np.nan
        last_regime_info = detailed_regime_classification(last_hurst)

        # Initialize regime columns
        ohlc['regime'] = np.nan
        ohlc['intensity'] = np.nan
        ohlc['regime_desc'] = np.nan

        # Set the last values
        if not ohlc.empty:
            ohlc.iloc[-1, ohlc.columns.get_loc('regime')] = last_regime_info[0]
            ohlc.iloc[-1, ohlc.columns.get_loc('intensity')] = last_regime_info[1]
            ohlc.iloc[-1, ohlc.columns.get_loc('regime_desc')] = last_regime_info[2]

        return ohlc

    # --- Sidebar Parameters ---
    @st.cache_data
    def fetch_token_list():
        query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
        df = pd.read_sql(query, engine)
        return df['pair_name'].tolist()

    # Keep only the most essential controls in the sidebar
    all_pairs = fetch_token_list()
    selected_pairs = st.sidebar.multiselect("Select Currency Pairs", all_pairs, default=all_pairs[:5] if len(all_pairs) >= 5 else all_pairs)
    timeframes = ["30s","15min", "30min", "1h", "4h", "6h"]
    selected_timeframes = st.sidebar.multiselect("Select Timeframes", timeframes, default=["15min", "1h"] if "15min" in timeframes and "1h" in timeframes else timeframes[:2] if len(timeframes) >= 2 else timeframes)

    # IMPORTANT: Define sliders in sidebar (essential settings)
    col1, col2 = st.sidebar.columns(2)
    lookback_days = col1.slider("Lookback (Days)", 1, 30, 7)  # Default to 7 days
    rolling_window = col2.slider("Rolling Window (Bars)", 20, 100, 30)

    # Display dynamic recommendations in sidebar
    if selected_timeframes:
        st.sidebar.markdown("### Recommended Settings")
        settings_text = ""
        for tf in selected_timeframes:
            rec = get_recommended_settings(tf)
            settings_text += f"""
            - **{tf}**: {rec['lookback_min']}-{rec['lookback_ideal']} day lookback, {rec['window_min']}-{rec['window_ideal']} bar window
            """

        st.sidebar.markdown(settings_text)

        # Auto-suggestion for current settings
        recommended_lookbacks = []
        recommended_windows = []

        for tf in selected_timeframes:
            rec = get_recommended_settings(tf)
            recommended_lookbacks.append(rec["lookback_min"])
            recommended_windows.append(rec["window_ideal"])

        # Only proceed if we have recommendations
        if recommended_lookbacks and recommended_windows:
            rec_lookback = max(recommended_lookbacks)
            rec_window = min(recommended_windows)

            if lookback_days < rec_lookback:
                st.sidebar.warning(f"⚠️ Current lookback ({lookback_days} days) may be too short for {max(selected_timeframes, key=lambda x: get_recommended_settings(x)['lookback_min'])}")

                # Add a button to auto-apply the recommended settings
                if st.sidebar.button(f"Apply Recommended Settings ({rec_lookback} days lookback, {rec_window} bar window)"):
                    lookback_days = rec_lookback
                    rolling_window = rec_window
                    st.experimental_rerun()

    # --- Color Code Legend ---
    with st.sidebar.expander("Legend: Regime Colors", expanded=True):
        st.markdown("""
        ### Mean-Reverting
        - <span style='background-color:rgba(255,0,0,0.7);padding:3px'>**Strong Mean-Reverting ⬇️⬇️⬇️**</span>  
        - <span style='background-color:rgba(255,50,50,0.6);padding:3px'>**Moderate Mean-Reverting ⬇️⬇️**</span>  
        - <span style='background-color:rgba(255,100,100,0.5);padding:3px'>**Mild Mean-Reverting ⬇️**</span>  
        - <span style='background-color:rgba(255,150,150,0.4);padding:3px'>**Slight Mean-Reverting Bias ↘️**</span>  

        ### Random/Noise
        - <span style='background-color:rgba(200,200,200,0.5);padding:3px'>**Pure Random Walk ↔️**</span>  

        ### Trending
        - <span style='background-color:rgba(150,255,150,0.4);padding:3px'>**Slight Trending Bias ↗️**</span>  
        - <span style='background-color:rgba(100,220,100,0.5);padding:3px'>**Mild Trending ⬆️**</span>  
        - <span style='background-color:rgba(50,200,50,0.6);padding:3px'>**Moderate Trending ⬆️⬆️**</span>  
        - <span style='background-color:rgba(0,180,0,0.7);padding:3px'>**Strong Trending ⬆️⬆️⬆️**</span>  
        """, unsafe_allow_html=True)

    # --- Batch Data Processing ---
    @st.cache_data(ttl=300)
    def get_hurst_data_batch(pairs, timeframes, lookback_days, rolling_window):
        """
        Process Hurst data for multiple pairs and timeframes in batch

        Args:
            pairs: List of pairs to process
            timeframes: List of timeframes to process
            lookback_days: Days to look back
            rolling_window: Size of rolling window

        Returns:
            dict: Processed data for each pair and timeframe
        """
        if not pairs or not timeframes:
            return {}

        # Fetch all price data at once
        bulk_data = fetch_price_data_bulk(pairs, lookback_days)

        if bulk_data.empty:
            return {}

        # Process each pair and timeframe
        results = {}

        for pair in pairs:
            results[pair] = {}
            # Filter data for this pair
            pair_data = bulk_data[bulk_data['pair_name'] == pair].copy()

            if pair_data.empty:
                continue

            # Drop the pair_name column as it's no longer needed
            pair_data = pair_data.drop(columns=['pair_name'])

            for tf in timeframes:
                # Process data for this timeframe
                ohlc = process_pair_data(pair_data, tf, rolling_window)
                results[pair][tf] = ohlc

        return results

    # --- Analyze Data & Generate Summary ---
    def generate_summary_data(batch_results):
        """
        Generate summary data from batch results

        Args:
            batch_results: Dict of processed data for each pair and timeframe

        Returns:
            list: Summary data for each pair
        """
        summary_data = []

        for pair, timeframe_data in batch_results.items():
            pair_data = {"Pair": pair}

            for tf, ohlc in timeframe_data.items():
                if ohlc is None or ohlc.empty or pd.isna(ohlc['Hurst'].iloc[-1]):
                    pair_data[tf] = {"Hurst": np.nan, "Regime": "UNKNOWN", "Description": "Insufficient data"}
                else:
                    # Get last values
                    pair_data[tf] = {
                        "Hurst": ohlc['Hurst'].iloc[-1],
                        "Regime": ohlc['regime'].iloc[-1],
                        "Description": ohlc['regime_desc'].iloc[-1],
                        "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], ""),
                        "Valid_Pct": (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100,
                        "Confidence": ohlc['confidence'].iloc[-1]
                    }

            summary_data.append(pair_data)

        return summary_data

    # --- Process data in background ---
    if selected_pairs and selected_timeframes:
        # Cache data loading with a status indicator
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.processing_started = False

        if not st.session_state.processing_started:
            st.session_state.processing_started = True
            with st.spinner("Loading data and calculating Hurst exponents..."):
                # Get batch results once
                batch_results = get_hurst_data_batch(selected_pairs, selected_timeframes, lookback_days, rolling_window)
                st.session_state.batch_results = batch_results

                # Generate summary data
                summary_data = generate_summary_data(batch_results)
                st.session_state.summary_data = summary_data

                st.session_state.data_loaded = True

    # --- Display Matrix View ---
    with tab1:
        # Add a refresh button at the top of the Matrix View tab
        refresh_clicked = st.button("Refresh Analysis")
        if refresh_clicked:
            st.session_state.data_loaded = False
            st.session_state.processing_started = False
            st.experimental_rerun()

        if not selected_pairs or not selected_timeframes:
            st.warning("Please select at least one pair and timeframe")
        elif not st.session_state.data_loaded:
            st.info("Processing data... please wait")
        else:
            # Get batch results from session state
            batch_results = st.session_state.batch_results

            # Check if current settings match recommendations
            needs_adjustment = False
            problematic_timeframes = []

            for tf in selected_timeframes:
                rec = get_recommended_settings(tf)
                if lookback_days < rec["lookback_min"] or rolling_window > rec["window_ideal"] * 1.5:
                    needs_adjustment = True
                    problematic_timeframes.append(tf)

            if needs_adjustment and problematic_timeframes:
                recommendation_text = "Recommended settings:\n"
                for tf in problematic_timeframes:
                    rec = get_recommended_settings(tf)
                    recommendation_text += f"- {tf}: {rec['lookback_min']}-{rec['lookback_ideal']} day lookback, {rec['window_min']}-{rec['window_ideal']} bar window\n"

                st.warning(f"""
                ⚠️ **Your current settings may result in insufficient data for some timeframes.**

                {recommendation_text}

                Current settings: {lookback_days} days lookback, {rolling_window} bar window
                """)

            # Display pairs and charts
            for pair in selected_pairs:
                if pair not in batch_results:
                    continue

                st.markdown(f"### 📌 {pair}")
                cols = st.columns(len(selected_timeframes))

                for i, tf in enumerate(selected_timeframes):
                    with cols[i]:
                        st.markdown(f"**{tf}**")

                        if tf not in batch_results[pair]:
                            st.write("No data")
                            continue

                        ohlc = batch_results[pair][tf]

                        if ohlc is None or ohlc.empty:
                            st.write("No data")
                            continue

                        # Calculate data quality metrics
                        valid_data_pct = (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100

                        # Diagnostic information for insufficient data
                        if valid_data_pct < 30:
                            if len(ohlc) < rolling_window * 2:
                                suggestion = "⚠️ Need more data. Increase lookback period."
                            elif rolling_window > 40:
                                suggestion = "⚠️ Window too large. Try a smaller rolling window."
                            else:
                                suggestion = f"⚠️ Low valid data ({valid_data_pct:.1f}%)."
                            st.warning(suggestion)

                        # Check if we have a valid Hurst value
                        if pd.isna(ohlc['Hurst'].iloc[-1]):
                            st.error("Insufficient data for Hurst calculation")
                            continue

                        # Chart 
                        fig = go.Figure()

                        # Background regime color - simplified for performance
                        # Only color the last few bars for performance
                        last_regime = ohlc['regime'].iloc[-1] if not pd.isna(ohlc['regime'].iloc[-1]) else "UNKNOWN"
                        last_intensity = ohlc['intensity'].iloc[-1] if not pd.isna(ohlc['intensity'].iloc[-1]) else 0

                        if last_regime in color_map and last_intensity in color_map[last_regime]:
                            shade_color = color_map[last_regime][last_intensity]
                        else:
                            shade_color = "rgba(200,200,200,0.3)"

                        # Add a background for the entire chart based on last regime
                        fig.add_shape(
                            type="rect",
                            x0=ohlc.index[0],
                            y0=0,
                            x1=ohlc.index[-1],
                            y1=1,
                            yref="paper",
                            fillcolor=shade_color,
                            opacity=0.5,
                            layer="below",
                            line_width=0
                        )

                        # Price line
                        fig.add_trace(go.Scatter(
                            x=ohlc.index, 
                            y=ohlc['close'], 
                            mode='lines', 
                            line=dict(color='black', width=1.5), 
                            name='Price'))

                        # Add Hurst line on secondary y-axis - only for last few values
                        # Subsample the data for better performance
                        max_points = 100  # Maximum points to plot for performance
                        step = max(1, len(ohlc) // max_points)

                        # Only add Hurst line if we have calculated values
                        valid_hurst = ohlc[ohlc['Hurst'].notna()]
                        if not valid_hurst.empty:
                            # Subsample for performance
                            subsampled_hurst = valid_hurst.iloc[::step]

                            fig.add_trace(go.Scatter(
                                x=subsampled_hurst.index,
                                y=subsampled_hurst['Hurst'],
                                mode='lines',
                                line=dict(color='blue', width=2, dash='dot'),
                                name='Hurst',
                                yaxis='y2'
                            ))

                        # Determine color based on regime
                        if last_regime == "MEAN-REVERT":
                             title_color = "red"
                        elif last_regime == "TREND":
                             title_color = "green"
                        else:
                             title_color = "gray"

                        # Current regime info
                        current_hurst = ohlc['Hurst'].iloc[-1]
                        current_desc = ohlc['regime_desc'].iloc[-1] if not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown"

                        # Add emoji to description
                        emoji = regime_emojis.get(current_desc, "")
                        display_text = f"{current_desc} {emoji}" if not pd.isna(current_hurst) else "Unknown"
                        hurst_text = f"Hurst: {current_hurst:.2f}" if not pd.isna(current_hurst) else "Hurst: n/a"

                         # Add data quality info
                        quality_text = f"Valid data: {valid_data_pct:.1f}%"

                        fig.update_layout(
                            title=dict(
                                text=f"<b>{display_text}</b><br><sub>{hurst_text} | {quality_text}</sub>",
                                font=dict(color=title_color, size=14, family="Arial, sans-serif")
                            ),
                            margin=dict(l=5, r=5, t=60, b=5),
                            height=220,
                            hovermode="x unified",
                            yaxis=dict(
                                title="Price",
                                titlefont=dict(size=10),
                                showgrid=True,
                                gridcolor='rgba(230,230,230,0.5)'
                            ),
                            yaxis2=dict(
                                title="Hurst",
                                titlefont=dict(color="blue", size=10),
                                tickfont=dict(color="blue", size=8),
                                anchor="x",
                                overlaying="y",
                                side="right",
                                range=[0, 1],
                                showgrid=False
                            ),
                            xaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(230,230,230,0.5)'
                            ),
                            showlegend=False,
                            plot_bgcolor='white'
                        )

                        # Add reference lines for Hurst thresholds
                        fig.add_shape(
                             type="line",
                             x0=ohlc.index[0],
                             y0=0.4,
                             x1=ohlc.index[-1],
                             y1=0.4,
                            line=dict(color="red", width=1.5, dash="dash"),
                            yref="y2"
                        )

                        fig.add_shape(
                            type="line",
                            x0=ohlc.index[0],
                            y0=0.6,
                            x1=ohlc.index[-1],
                            y1=0.6,
                            line=dict(color="green", width=1.5, dash="dash"),
                            yref="y2"
                        )

                        st.plotly_chart(fig, use_container_width=True) 

    # --- Summary Table ---
    with tab2:
        if not selected_pairs or not selected_timeframes:
            st.warning("Please select at least one pair and timeframe")
        elif not st.session_state.data_loaded:
            st.info("Processing data... please wait")
        else:
            # Get summary data from session state
            summary_data = st.session_state.summary_data

            if not summary_data:
                st.warning("No summary data available")
            else:
                st.subheader("🔍 Pair-Specific Summary Table")

                # Recommended settings based on current data
                st.info(f"""
                ### Data Quality & Recommended Settings

                - **Low timeframes (15min)**: 2-3 day lookback, 20-30 bar window
                - **Medium timeframes (1h)**: 5-7 day lookback, 20-30 bar window
                - **High timeframes (6h)**: 14+ day lookback, 20-30 bar window

                Your current settings: **{lookback_days} days lookback** with **{rolling_window} bar window**
                """.format(lookback_days, rolling_window))

                # Create a formatted HTML table for better visualization
                html_table = "<table style='width:100%; border-collapse: collapse;'>"

                # Header row
                html_table += "<tr style='background-color:#f2f2f2;'>"
                html_table += "<th style='padding:10px; border:1px solid #ddd;'>Pair</th>"

                for tf in selected_timeframes:
                    html_table += f"<th style='padding:10px; border:1px solid #ddd;'>{tf}</th>"

                html_table += "</tr>"

                # Data rows
                for item in summary_data:
                    html_table += "<tr>"
                    html_table += f"<td style='padding:10px; border:1px solid #ddd; font-weight:bold;'>{item['Pair']}</td>"

                    for tf in selected_timeframes:
                        if tf in item:
                            regime_data = item[tf]

                            # Determine cell background color based on regime
                            if "Regime" in regime_data:
                                if regime_data["Regime"] == "MEAN-REVERT":
                                    bg_color = "rgba(255,200,200,0.5)"
                                elif regime_data["Regime"] == "TREND":
                                    bg_color = "rgba(200,255,200,0.5)"
                                else:
                                    bg_color = "rgba(220,220,220,0.3)"
                            else:
                                bg_color = "rgba(220,220,220,0.3)"

                            # Format Hurst value
                            hurst_val = f"{regime_data['Hurst']:.2f}" if "Hurst" in regime_data and not pd.isna(regime_data["Hurst"]) else "n/a"

                            # Add emoji if available
                            emoji = regime_data.get("Emoji", "")

                            # Add data quality info if available
                            valid_pct = regime_data.get("Valid_Pct", 0)
                            quality_text = ""
                            if valid_pct < 30 and valid_pct > 0:
                                quality_text = f"<br><small style='color:orange;'>Low quality: {valid_pct:.1f}%</small>"

                            html_table += f"<td style='padding:10px; border:1px solid #ddd; background-color:{bg_color};'>"
                            html_table += f"{regime_data['Description']} {emoji}<br><small>Hurst: {hurst_val}</small>{quality_text}"
                            html_table += "</td>"
                        else:
                            html_table += "<td style='padding:10px; border:1px solid #ddd;'>No data</td>"

                    html_table += "</tr>"

                html_table += "</table>"

                # Display the HTML table
                st.markdown(html_table, unsafe_allow_html=True)

                # Add a downloadable CSV version
                csv_data = []
                header = ["Pair"]

                for tf in selected_timeframes:
                    header.append(f"{tf}_Regime")
                    header.append(f"{tf}_Hurst")
                    header.append(f"{tf}_Valid_Pct")

                csv_data.append(header)

                for item in summary_data:
                    row = [item["Pair"]]

                    for tf in selected_timeframes:
                        if tf in item:
                            row.append(item[tf]["Description"])
                            row.append(item[tf]["Hurst"] if "Hurst" in item[tf] and not pd.isna(item[tf]["Hurst"]) else "")
                            row.append(item[tf].get("Valid_Pct", ""))
                        else:
                            row.append("No data")
                            row.append("")
                            row.append("")

                    csv_data.append(row)

                # Convert to DataFrame for download
                csv_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])

                # Add download button
                st.download_button(
                    label="Download as CSV",
                    data=csv_df.to_csv(index=False),
                    file_name=f"market_regimes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

    # --- Filter by Regime Tab ---
    with tab3:
        st.header("Find Currency Pairs by Regime")

        st.info("""
        This tab allows you to search across **all currency pairs** in the database to find those 
        matching specific market regimes, regardless of the pairs selected in the sidebar.
        """)

        # Select timeframe
        filter_timeframe = st.selectbox(
            "Select Timeframe to Analyze",
            timeframes,
            index=timeframes.index("1h") if "1h" in timeframes else 0
        )

        # Select regime to filter by
        filter_regime = st.multiselect(
            "Show Only Pairs with These Regimes:", 
            ["All Regimes","Strong mean-reversion", "Moderate mean-reversion", "Mild mean-reversion", "Slight mean-reversion bias",
             "Pure random walk", 
             "Slight trending bias", "Mild trending", "Moderate trending", "Strong trending"],
            default=["All Regimes"]
        )

        # Sorting options
        sort_option_filter = st.selectbox(
            "Sort Results By:",
            ["Name", "Most Trending (Highest Hurst)", "Most Mean-Reverting (Lowest Hurst)", "Data Quality"],
            index=0
        )

        # Data quality filter
        min_data_quality = st.slider("Minimum Data Quality (%)", 0, 100, 30)

        # Use current lookback/window or set custom ones
        use_custom_params = st.checkbox("Use Custom Parameters (instead of sidebar settings)", value=False)

        if use_custom_params:
            custom_col1, custom_col2 = st.columns(2)
            custom_lookback = custom_col1.slider("Custom Lookback (Days)", 1, 30, 
                                                get_recommended_settings(filter_timeframe)["lookback_ideal"])
            custom_window = custom_col2.slider("Custom Window (Bars)", 20, 100, 
                                              get_recommended_settings(filter_timeframe)["window_ideal"])

        # Button to run the filter
        if st.button("Find Matching Pairs"):
            # Show a spinner while processing
            with st.spinner("Analyzing all currency pairs..."):
                # Get the complete list of pairs from database 
                all_available_pairs = fetch_token_list()

                # Determine which parameters to use
                if use_custom_params:
                    actual_lookback = custom_lookback
                    actual_window = custom_window
                else:
                    actual_lookback = lookback_days
                    actual_window = rolling_window

                # Batch process all pairs with parallelization
                # Show loading visualization
                progress_bar = st.progress(0)

                # Process in batches for better performance
                batch_size = 5  # Process 5 pairs at a time
                num_batches = (len(all_available_pairs) + batch_size - 1) // batch_size

                regime_results = []

                for i in range(0, num_batches):
                    # Update progress
                    progress_bar.progress(i / num_batches)

                    # Get batch of pairs
                    batch_start = i * batch_size
                    batch_end = min((i + 1) * batch_size, len(all_available_pairs))
                    batch_pairs = all_available_pairs[batch_start:batch_end]

                    # Process batch
                    batch_results = get_hurst_data_batch(batch_pairs, [filter_timeframe], actual_lookback, actual_window)

                    # Extract results
                    for pair in batch_pairs:
                        if pair not in batch_results or filter_timeframe not in batch_results[pair]:
                            regime_results.append({
                                "Pair": pair,
                                "Regime": "Insufficient data",
                                "Hurst": np.nan,
                                "Data Quality": 0,
                                "Emoji": "❓"
                            })
                            continue

                        ohlc = batch_results[pair][filter_timeframe]

                        if ohlc is None or ohlc.empty or pd.isna(ohlc['Hurst'].iloc[-1]):
                            regime_results.append({
                                "Pair": pair,
                                "Regime": "Insufficient data",
                                "Hurst": np.nan,
                                "Data Quality": 0,
                                "Emoji": "❓"
                            })
                            continue

                        # Calculate data quality
                        data_quality = (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100

                        regime_info = {
                            "Pair": pair,
                            "Regime": ohlc['regime_desc'].iloc[-1] if not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown",
                            "Hurst": ohlc['Hurst'].iloc[-1],
                            "Data Quality": data_quality,
                            "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], "")
                        }

                        if "All Regimes" in filter_regime or not filter_regime or regime_info["Regime"] in filter_regime:
                            regime_results.append(regime_info)

                # Complete progress
                progress_bar.progress(1.0)

                # Filter by data quality
                regime_results = [
                    result for result in regime_results 
                    if result['Data Quality'] >= min_data_quality
                ]

                # Sorting logic
                if sort_option_filter == "Most Trending (Highest Hurst)":
                    regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else -np.inf, reverse=True)
                elif sort_option_filter == "Most Mean-Reverting (Lowest Hurst)":
                    regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else np.inf)
                elif sort_option_filter == "Data Quality":
                    regime_results.sort(key=lambda x: x["Data Quality"], reverse=True)
                else:  # Sort by name
                    regime_results.sort(key=lambda x: x["Pair"])

                # Display results
                if regime_results:
                    st.success(f"Found {len(regime_results)} matching pairs")

                    # Create a DataFrame for better display
                    results_df = pd.DataFrame(regime_results)

                    # Define a function to apply styling to the table
                    def highlight_regimes(val):
                        color = "white"
                        if "mean-reversion" in str(val).lower():
                            color = "rgba(255,200,200,0.5)"
                        elif "trend" in str(val).lower():
                            color = "rgba(200,255,200,0.5)"
                        elif "random" in str(val).lower():
                            color = "rgba(220,220,220,0.5)"
                        return f'background-color: {color}'

                    # Apply styling and display the table
                    st.dataframe(
                        results_df.style.applymap(highlight_regimes, subset=['Regime']),
                        height=600
                    )

                    # Option to select these pairs in the main view
                    if st.button(f"Add these {len(regime_results)} pairs to sidebar selection"):
                        # Get the pairs to add
                        pairs_to_add = [item["Pair"] for item in regime_results]
                        # Convert to set to avoid duplicates and combine with existing selection
                        updated_pairs = list(set(selected_pairs + pairs_to_add))
                        # Update session state to persist across reruns
                        if 'selected_pairs' not in st.session_state:
                            st.session_state.selected_pairs = updated_pairs
                        else:
                            st.session_state.selected_pairs = updated_pairs
                        st.experimental_rerun()
                else:
                    st.warning("No currency pairs match your filter criteria")

    # --- Global Regime Summary Tab ---
    # --- Global Regime Summary Tab ---
    # --- Global Regime Summary Tab ---
    # --- Global Regime Summary Tab ---
    # --- Global Regime Summary Tab ---
    with tab4:
        st.header("Global Regime Summary")

        # Add a simple debug message
        st.write("This tab calculates regimes for all pairs across selected timeframes")

        # Independent controls for global summary
        col1, col2, col3 = st.columns(3)

        with col1:
            global_timeframes = st.multiselect(
                "Select Timeframes", 
                timeframes, 
                default=[timeframes[0]] if timeframes else [],
                key="global_timeframes_select_unique"
            )

        with col2:
            global_lookback = st.slider(
                "Lookback (Days)", 
                1, 30, 3,
                key="global_lookback_slider_unique"
            )

        with col3:
            global_window = st.slider(
                "Rolling Window (Bars)", 
                20, 100, 25,
                key="global_window_slider_unique"
            )

        # Ensure we have a session state key for this tab
        if 'global_results' not in st.session_state:
            st.session_state.global_results = None

        # Add a debug option
        debug_mode = st.checkbox("Enable debug mode", value=False, key="global_debug_unique")

        # Define styling function properly (outside of any conditional blocks)
        def style_regime_table(df):
            # Create a copy with styled values
            styled_df = df.copy()

            # Define CSS for the entire dataframe
            styles = [
                # Increase overall font size
                dict(selector="th", props=[("font-size", "16px"), ("font-weight", "bold"), 
                      ("background-color", "#f2f2f2"), ("padding", "12px")]),
                dict(selector="td", props=[("font-size", "15px"), ("padding", "10px")])
            ]

            # Apply styling based on regime
            def color_regimes(val):
                if pd.isna(val):
                    return 'background-color: #f0f0f0; color: #666666;'

                if "mean-reversion" in str(val).lower():
                    intensity = "strong" if "strong" in str(val).lower() else \
                               "moderate" if "moderate" in str(val).lower() else \
                               "mild" if "mild" in str(val).lower() else "slight"

                    if intensity == "strong":
                        return 'background-color: rgba(255,0,0,0.7); color: white; font-weight: bold;'
                    elif intensity == "moderate":
                        return 'background-color: rgba(255,50,50,0.6); color: white; font-weight: bold;'
                    elif intensity == "mild":
                        return 'background-color: rgba(255,100,100,0.5); color: black;'
                    else:  # slight
                        return 'background-color: rgba(255,150,150,0.4); color: black;'

                elif "trending" in str(val).lower():
                    intensity = "strong" if "strong" in str(val).lower() else \
                               "moderate" if "moderate" in str(val).lower() else \
                               "mild" if "mild" in str(val).lower() else "slight"

                    if intensity == "strong":
                        return 'background-color: rgba(0,180,0,0.7); color: white; font-weight: bold;'
                    elif intensity == "moderate":
                        return 'background-color: rgba(50,200,50,0.6); color: white; font-weight: bold;'
                    elif intensity == "mild":
                        return 'background-color: rgba(100,220,100,0.5); color: black;'
                    else:  # slight
                        return 'background-color: rgba(150,255,150,0.4); color: black;'

                elif "random" in str(val).lower():
                    return 'background-color: rgba(200,200,200,0.5); color: black;'

                elif "no data" in str(val).lower() or "insufficient" in str(val).lower():
                    return 'background-color: #f0f0f0; color: #999999; font-style: italic;'

                else:
                    return ''

            # Return the styled dataframe
            return styled_df.style.set_table_styles(styles).applymap(color_regimes)

        # Generate button with a unique key
        generate_button = st.button("Generate Global Regime Summary", key="unique_global_generate_btn")

        if debug_mode:
            st.write(f"Button state: {generate_button}")

        if generate_button:
            if debug_mode:
                st.write("Button clicked!")

            if not global_timeframes:
                st.warning("Please select at least one timeframe")
            else:
                try:
                    # Create a placeholder for the progress bar
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)

                    # Get all pairs for testing
                    all_pairs = fetch_token_list()
                    if debug_mode:
                        st.write(f"Found {len(all_pairs)} pairs")
                        if debug_mode:
                            all_pairs = all_pairs[:10]  # Limit to 10 pairs for debugging
                            st.write(f"Using {len(all_pairs)} pairs for debug")

                    # Process in smaller batches for reliability
                    batch_size = 5
                    rows = []

                    # Process each batch
                    for i in range(0, len(all_pairs), batch_size):
                        batch = all_pairs[i:i+batch_size]
                        progress_bar.progress(i / len(all_pairs) if len(all_pairs) > 0 else 0)

                        if debug_mode:
                            st.write(f"Processing batch {i//batch_size + 1}/{(len(all_pairs) + batch_size - 1)//batch_size}")

                        # Get data for this batch
                        for pair in batch:
                            row = {"Pair": pair}

                            # Process each timeframe
                            for tf in global_timeframes:
                                # Simplified logic - just calculate the Hurst value directly
                                try:
                                    # Get price data
                                    end_time = datetime.now(timezone.utc)
                                    start_time = end_time - timedelta(days=global_lookback)

                                    query = f"""
                                    SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, 
                                           final_price 
                                    FROM public.oracle_price_log 
                                    WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
                                    AND pair_name = '{pair}';
                                    """

                                    df = pd.read_sql(query, engine)

                                    if df.empty:
                                        row[tf] = "No data"
                                        continue

                                    # Process data
                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    df = df.set_index('timestamp').sort_index()

                                    # Resample to OHLC
                                    ohlc = df['final_price'].resample(tf).ohlc().dropna()

                                    if len(ohlc) < global_window:
                                        row[tf] = "Insufficient data"
                                        continue

                                    # Calculate Hurst on the last window
                                    last_window = ohlc['close'].tail(global_window).values
                                    hurst = universal_hurst(tuple(last_window))

                                    if pd.isna(hurst):
                                        row[tf] = "Calculation failed"
                                        continue

                                    # Get regime
                                    regime_info = detailed_regime_classification(hurst)
                                    regime_desc = regime_info[2]
                                    emoji = regime_emojis.get(regime_desc, "")

                                    # Store result
                                    row[tf] = f"{regime_desc} {emoji} (H:{hurst:.2f})"

                                except Exception as e:
                                    if debug_mode:
                                        st.write(f"Error processing {pair} - {tf}: {str(e)}")
                                    row[tf] = f"Error: {str(e)[:20]}"

                            rows.append(row)

                    # Complete progress
                    progress_bar.progress(1.0)

                    # Create DataFrame
                    results_df = pd.DataFrame(rows)

                    # Store in session state
                    st.session_state.global_results = results_df

                    # Success message
                    st.success(f"Analysis complete for {len(rows)} pairs")

                    # Display the enhanced table with increased height
                    st.dataframe(
                        style_regime_table(results_df),
                        height=800,  # Increase height to show more rows
                        use_container_width=True
                    )

                    # Add option to view full-screen
                    st.markdown("""
                    <style>
                        .fullscreen-button {
                            background-color: #4CAF50;
                            border: none;
                            color: white;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin: 4px 2px;
                            cursor: pointer;
                            border-radius: 4px;
                        }
                    </style>

                    <a href="#" class="fullscreen-button" onclick="document.querySelector('iframe[title=\"streamlit_app\"]').requestFullscreen(); return false;">View Table Fullscreen</a>
                    """, unsafe_allow_html=True)

                    # Add download options
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download as CSV",
                        data=csv,
                        file_name=f"global_regime_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="global_download_btn"
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())

        # Display previous results if they exist
        elif st.session_state.global_results is not None:
            st.success("Showing previously generated results")

            # Display using the same styling
            st.dataframe(
                style_regime_table(st.session_state.global_results),
                height=800,
                use_container_width=True
            )


def render_spread_analysis():
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
        .highlight-text {
            color: #1E88E5;
            font-weight: bold;
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

    # Modified exchange list: Remove MEXC and OKX, put SurfFuture at the end
    exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "surfFuture"]
    exchanges_display = {
        "binanceFuture": "Binance",
        "gateFuture": "Gate",
        "hyperliquidFuture": "Hyperliquid",
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
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
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

            # Create time labels
            timestamps = []
            for exchange, series in fee_data.items():
                timestamps.extend(series.index)
            unique_timestamps = sorted(set(timestamps))

            # Create DataFrame with all timestamps and exchanges
            result_df = pd.DataFrame(index=unique_timestamps)
            for exchange, series in fee_data.items():
                result_df[exchange] = pd.Series(series.values, index=series.index)

            # Add time label
            result_df['time_label'] = result_df.index.strftime('%H:%M')

            # Calculate non-surf average (Binance, Gate, Hyperliquid)
            non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
            # Only include columns that exist in the DataFrame
            non_surf_columns = [col for col in non_surf_columns if col in result_df.columns]

            if non_surf_columns:
                # Calculate row by row to avoid operating on all columns
                non_surf_avg = []
                for idx, row in result_df.iterrows():
                    values = [row[col] for col in non_surf_columns if not pd.isna(row[col])]
                    if values:
                        non_surf_avg.append(sum(values) / len(values))
                    else:
                        non_surf_avg.append(np.nan)
                result_df['Avg (Non-Surf)'] = non_surf_avg

            # Calculate daily average for each exchange - column by column
            daily_avgs = {}
            for column in result_df.columns:
                if column != 'time_label':
                    values = result_df[column].dropna().values
                    if len(values) > 0:
                        daily_avgs[column] = sum(values) / len(values)
                    else:
                        daily_avgs[column] = np.nan

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

            # Query to get the fee data - only for the exchanges we want
            query = f"""
            SELECT 
                source,
                SUM(total_fee) as total_fee
            FROM 
                oracle_exchange_fee
            WHERE 
                pair_name = '{token}'
                AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
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

            # Calculate average fee across non-SurfFuture exchanges
            non_surf_fees = []
            for k, v in fee_data.items():
                if k in ['Binance', 'Gate', 'Hyperliquid']:
                    non_surf_fees.append(v)

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

            # Calculate scaling factor based on selected columns
            scale_factor = 1
            scale_label = ""

            # Find numeric columns
            numeric_cols = []
            for col in all_fees_df.columns:
                if col != 'Token':
                    try:
                        # Check if column can be treated as numeric
                        all_fees_df[col] = pd.to_numeric(all_fees_df[col], errors='coerce')
                        numeric_cols.append(col)
                    except:
                        pass

            # Calculate mean for scaling
            if numeric_cols:
                values = []
                for col in numeric_cols:
                    values.extend(all_fees_df[col].dropna().tolist())

                if values:
                    mean_fee = sum(values) / len(values)

                    # Determine scale factor based on mean fee value
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
                for col in numeric_cols:
                    all_fees_df[col] = all_fees_df[col] * scale_factor

                st.markdown(f"<div class='info-box'><b>Note:</b> All fee values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)

            # Make sure columns are in the desired order with SurfFuture at the end
            desired_order = ['Token', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']

            # Reorder columns according to the specified order
            ordered_columns = [col for col in desired_order if col in all_fees_df.columns]
            all_fees_df = all_fees_df[ordered_columns]

            # Round values to 2 decimal places for display
            for col in numeric_cols:
                if col in all_fees_df.columns:
                    all_fees_df[col] = all_fees_df[col].round(2)

            # Display the summary table with dynamic height to show all tokens without scrolling
            token_count = len(all_fees_df)
            table_height = max(100 + 35 * token_count, 200)  # Minimum height of 200px
            st.dataframe(all_fees_df, height=table_height, use_container_width=True)

            # Check if SurfFuture has lower fees than average
            if 'SurfFuture' in all_fees_df.columns and 'Avg (Non-Surf)' in all_fees_df.columns:
                # Extract values for comparison, skipping NaN
                surf_values = []
                nonsurf_values = []

                for idx, row in all_fees_df.iterrows():
                    if not pd.isna(row['SurfFuture']) and not pd.isna(row['Avg (Non-Surf)']):
                        surf_values.append(row['SurfFuture'])
                        nonsurf_values.append(row['Avg (Non-Surf)'])

                if surf_values and nonsurf_values:
                    surf_avg = sum(surf_values) / len(surf_values)
                    non_surf_avg = sum(nonsurf_values) / len(nonsurf_values)

                    if surf_avg < non_surf_avg:
                        st.success(f"🏆 **SurfFuture has tighter spreads overall**: SurfFuture ({surf_avg:.2f}) vs Other Exchanges ({non_surf_avg:.2f})")
                    else:
                        st.info(f"SurfFuture average: {surf_avg:.2f}, Other Exchanges average: {non_surf_avg:.2f}")

            # Add visualization - Bar chart comparing average fees by exchange
            st.markdown('<div class="subheader-style">Average Fee by Exchange</div>', unsafe_allow_html=True)

            # Calculate average by exchange
            avg_by_exchange = {}
            for col in all_fees_df.columns:
                if col not in ['Token', 'Avg (Non-Surf)'] and col in numeric_cols:
                    values = all_fees_df[col].dropna().tolist()
                    if values:
                        avg_by_exchange[col] = sum(values) / len(values)

            if avg_by_exchange:
                # Sort exchanges by average fee
                sorted_exchanges = sorted(avg_by_exchange.items(), key=lambda x: x[1])
                exchanges_sorted = [x[0] for x in sorted_exchanges]
                fees_sorted = [x[1] for x in sorted_exchanges]

                # Create a colorful bar chart
                colors = ['#1a9850', '#66bd63', '#fee08b', '#f46d43']
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

            # Calculate best exchange for each token
            best_exchanges = {}
            for idx, row in all_fees_df.iterrows():
                exchange_cols = [c for c in row.index if c not in ['Token', 'Avg (Non-Surf)'] and c in numeric_cols]
                if exchange_cols:
                    # Extract values and exchanges, skipping NaN
                    valid_fees = {}
                    for col in exchange_cols:
                        if not pd.isna(row[col]):
                            valid_fees[col] = row[col]

                    if valid_fees:
                        best_ex = min(valid_fees.items(), key=lambda x: x[1])
                        best_exchanges[row['Token']] = best_ex

            # Count the number of "wins" for each exchange
            exchange_wins = {}
            for ex in [c for c in all_fees_df.columns if c not in ['Token', 'Avg (Non-Surf)'] and c in numeric_cols]:
                exchange_wins[ex] = sum(1 for _, (best_ex, _) in best_exchanges.items() if best_ex == ex)

            # Create a pie chart of wins
            if exchange_wins:
                fig = go.Figure(data=[go.Pie(
                    labels=list(exchange_wins.keys()),
                    values=list(exchange_wins.values()),
                    textinfo='label+percent',
                    marker=dict(colors=['#66bd63', '#fee08b', '#f46d43', '#1a9850']),
                    hole=.3
                )])

                fig.update_layout(
                    title="Exchange with Lowest Fees (Number of Tokens)",
                    height=400,
                    font=dict(size=14)
                )

                st.plotly_chart(fig, use_container_width=True)

    # Display individual token tables with modified layout
    if token_detailed_results:
        st.markdown('<div class="header-style">Detailed Analysis by Token</div>', unsafe_allow_html=True)

        # For each token, create a table showing all exchanges
        for token, df in token_detailed_results.items():
            st.markdown(f"## {token} Exchange Fee Comparison")

            # Set the time_label as index to display in rows
            table_df = df.copy()

            # Determine scale factor
            scale_factor = 1
            scale_label = ""

            # Find numeric columns
            numeric_cols = []
            for col in table_df.columns:
                if col != 'time_label':
                    try:
                        # Check if column can be treated as numeric
                        table_df[col] = pd.to_numeric(table_df[col], errors='coerce')
                        numeric_cols.append(col)
                    except:
                        pass

            # Calculate mean for scaling
            if numeric_cols:
                values = []
                for col in numeric_cols:
                    values.extend(table_df[col].dropna().tolist())

                if values:
                    mean_fee = sum(values) / len(values)

                    # Determine scale factor based on mean fee value
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
                for col in numeric_cols:
                    table_df[col] = table_df[col] * scale_factor
                st.markdown(f"<div class='info-box'><b>Note:</b> All fee values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)

            # Reorder columns to the specified layout
            desired_order = ['time_label', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']

            # Filter only columns that exist
            ordered_columns = [col for col in desired_order if col in table_df.columns]
            table_df = table_df[ordered_columns]

            # Sort by time label in descending order
            table_df = table_df.sort_values(by='time_label', key=lambda x: x.map(time_to_minutes), ascending=False)

            # Format to 6 decimal places
            formatted_df = table_df.copy()
            for col in numeric_cols:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(6)

            # Display the table with the time_label as the first column
            st.dataframe(formatted_df, height=500, use_container_width=True)

            # Check if SurfFuture has tighter spread than the average of the other exchanges
            if 'SurfFuture' in table_df.columns and 'Avg (Non-Surf)' in table_df.columns:
                # Extract values for comparison, skipping NaN
                surf_values = []
                nonsurf_values = []

                for idx, row in table_df.iterrows():
                    if not pd.isna(row['SurfFuture']) and not pd.isna(row['Avg (Non-Surf)']):
                        surf_values.append(row['SurfFuture'])
                        nonsurf_values.append(row['Avg (Non-Surf)'])

                if surf_values and nonsurf_values:
                    surf_avg = sum(surf_values) / len(surf_values)
                    non_surf_avg = sum(nonsurf_values) / len(nonsurf_values)

                    if surf_avg < non_surf_avg:
                        st.success(f"✅ SURF Spread tighter: SurfFuture ({surf_avg:.6f}) < Non-Surf Average ({non_surf_avg:.6f})")
                    else:
                        st.info(f"SurfFuture ({surf_avg:.6f}) ≥ Non-Surf Average ({non_surf_avg:.6f})")

            # Create a summary for this token
            st.markdown("### Exchange Comparison Summary")

            # Get the daily averages for this token
            daily_avgs = token_daily_avgs[token]

            # Build summary data manually
            summary_data = []
            for exchange in ordered_columns:
                if exchange != 'time_label' and exchange in daily_avgs:
                    row_data = {
                        'Exchange': exchange,
                        'Average Fee': daily_avgs[exchange]
                    }
                    if not pd.isna(row_data['Average Fee']):
                        # Apply scaling
                        if scale_factor > 1:
                            row_data['Average Fee'] = row_data['Average Fee'] * scale_factor

                        # Round
                        row_data['Average Fee'] = round(row_data['Average Fee'], 6)

                        summary_data.append(row_data)

            if summary_data:
                # Create a DataFrame from the data
                summary_df = pd.DataFrame(summary_data)

                # Display it
                st.dataframe(summary_df, height=200, use_container_width=True)

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

        - **Column Order**: The tables display data in this order: Time, Binance, Gate, Hyperliquid, Average (Non-Surf), SurfFuture

        - **SURF Spread Indicator**: For each token, we indicate when SurfFuture has tighter spreads than the average of other exchanges.

        - **Visualizations**:
          - The bar chart shows the average fee across all tokens for each exchange
          - The pie chart shows which exchange offers the lowest fees for the most tokens

        ### Note on Scaling:

        If fee values are very small, they may be multiplied by a scaling factor for better readability. The scaling factor is indicated with each table.
        """)

def render_vol_and_hurst():
    # Save this as pages/05_Daily_Volatility_Table.py in your Streamlit app folder

    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz

    st.set_page_config(
        page_title="Daily Volatility Table",
        page_icon="📈",
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
    st.title("Daily Volatility Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    rolling_window = 20  # Window size for volatility calculation
    expected_points = 48  # Expected data points per pair over 24 hours
    singapore_timezone = pytz.timezone('Asia/Singapore')

    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

    # Set extreme volatility threshold
    extreme_vol_threshold = 1.0  # 100% annualized volatility

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

    # Function to calculate various volatility metrics
    def calculate_volatility_metrics(price_series):
        if price_series is None or len(price_series) < 2:
            return {
                'realized_vol': np.nan,
                'parkinson_vol': np.nan,
                'gk_vol': np.nan,
                'rs_vol': np.nan
            }

        try:
            # Calculate log returns
            log_returns = np.diff(np.log(price_series))

            # 1. Standard deviation of returns (realized volatility)
            realized_vol = np.std(log_returns) * np.sqrt(252 * 48)  # Annualized volatility (30min bars)

            # For other volatility metrics, need OHLC data
            # For simplicity, we'll focus on realized volatility for now
            # But the structure allows adding more volatility metrics

            return {
                'realized_vol': realized_vol,
                'parkinson_vol': np.nan,  # Placeholder for Parkinson volatility
                'gk_vol': np.nan,         # Placeholder for Garman-Klass volatility
                'rs_vol': np.nan          # Placeholder for Rogers-Satchell volatility
            }
        except Exception as e:
            print(f"Error in volatility calculation: {e}")
            return {
                'realized_vol': np.nan,
                'parkinson_vol': np.nan,
                'gk_vol': np.nan,
                'rs_vol': np.nan
            }

    # Volatility classification function
    def classify_volatility(vol):
        if pd.isna(vol):
            return ("UNKNOWN", 0, "Insufficient data")
        elif vol < 0.30:  # 30% annualized volatility threshold for low volatility
            return ("LOW", 1, "Low volatility")
        elif vol < 0.60:  # 60% annualized volatility threshold for medium volatility
            return ("MEDIUM", 2, "Medium volatility")
        elif vol < 1.00:  # 100% annualized volatility threshold for high volatility
            return ("HIGH", 3, "High volatility")
        else:
            return ("EXTREME", 4, "Extreme volatility")

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

    # Fetch and calculate volatility for a token with 30min timeframe
    @st.cache_data(ttl=600, show_spinner="Calculating volatility metrics...")
    def fetch_and_calculate_volatility(token):
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
            final_price, 
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

            # Create 1-minute OHLC data
            one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
            if one_min_ohlc.empty:
                print(f"[{token}] No OHLC data after resampling.")
                return None

            # Calculate rolling volatility on 1-minute data
            one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(
                lambda x: calculate_volatility_metrics(x)['realized_vol']
            )

            # Resample to exactly 30min intervals aligned with clock
            thirty_min_vol = one_min_ohlc['realized_vol'].resample('30min', closed='left', label='left').mean().dropna()

            if thirty_min_vol.empty:
                print(f"[{token}] No 30-min volatility data.")
                return None

            # Get last 24 hours (48 30-minute bars)
            last_24h_vol = thirty_min_vol.tail(48)  # Get up to last 48 periods (24 hours)
            last_24h_vol = last_24h_vol.to_frame()

            # Store original datetime index for reference
            last_24h_vol['original_datetime'] = last_24h_vol.index

            # Format time label to match our aligned blocks (HH:MM format)
            last_24h_vol['time_label'] = last_24h_vol.index.strftime('%H:%M')

            # Calculate 24-hour average volatility
            last_24h_vol['avg_24h_vol'] = last_24h_vol['realized_vol'].mean()

            # Classify volatility
            last_24h_vol['vol_info'] = last_24h_vol['realized_vol'].apply(classify_volatility)
            last_24h_vol['vol_regime'] = last_24h_vol['vol_info'].apply(lambda x: x[0])
            last_24h_vol['vol_desc'] = last_24h_vol['vol_info'].apply(lambda x: x[2])

            # Also classify the 24-hour average
            last_24h_vol['avg_vol_info'] = last_24h_vol['avg_24h_vol'].apply(classify_volatility)
            last_24h_vol['avg_vol_regime'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[0])
            last_24h_vol['avg_vol_desc'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[2])

            # Flag extreme volatility events
            last_24h_vol['is_extreme'] = last_24h_vol['realized_vol'] >= extreme_vol_threshold

            print(f"[{token}] Successful Volatility Calculation")
            return last_24h_vol
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

    # Calculate volatility for each token
    token_results = {}
    for i, token in enumerate(selected_tokens):
        try:
            progress_bar.progress((i) / len(selected_tokens))
            status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
            result = fetch_and_calculate_volatility(token)
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
            vol_series = df.set_index('time_label')['realized_vol']
            table_data[token] = vol_series

        # Create DataFrame with all tokens
        vol_table = pd.DataFrame(table_data)

        # Apply the time blocks in the proper order (most recent first)
        available_times = set(vol_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]

        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)

        # Reindex with the ordered times
        vol_table = vol_table.reindex(ordered_times)

        # Convert from decimal to percentage and round to 1 decimal place
        vol_table = (vol_table * 100).round(1)

        def color_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 30:  # Low volatility - green
                intensity = max(0, min(255, int(255 * val / 30)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            elif val < 60:  # Medium volatility - yellow
                intensity = max(0, min(255, int(255 * (val - 30) / 30)))
                return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
            elif val < 100:  # High volatility - orange
                intensity = max(0, min(255, int(255 * (val - 60) / 40)))
                return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
            else:  # Extreme volatility - red
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'

        styled_table = vol_table.style.applymap(color_cells)
        st.markdown("## Volatility Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
        st.markdown("Values shown as annualized volatility percentage")
        st.dataframe(styled_table, height=700, use_container_width=True)

        # Create ranking table based on average volatility
        st.subheader("Volatility Ranking (24-Hour Average, Descending Order)")

        ranking_data = []
        for token, df in token_results.items():
            if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
                avg_vol = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
                vol_regime = df['avg_vol_desc'].iloc[0]
                max_vol = df['realized_vol'].max()
                min_vol = df['realized_vol'].min()
                ranking_data.append({
                    'Token': token,
                    'Avg Vol (%)': (avg_vol * 100).round(1),
                    'Regime': vol_regime,
                    'Max Vol (%)': (max_vol * 100).round(1),
                    'Min Vol (%)': (min_vol * 100).round(1),
                    'Vol Range (%)': ((max_vol - min_vol) * 100).round(1)
                })

        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            # Sort by average volatility (high to low)
            ranking_df = ranking_df.sort_values(by='Avg Vol (%)', ascending=False)
            # Add rank column
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))

            # Reset the index to remove it
            ranking_df = ranking_df.reset_index(drop=True)

            # Format ranking table with colors
            def color_regime(val):
                if 'Low' in val:
                    return 'color: green'
                elif 'Medium' in val:
                    return 'color: #aaaa00'
                elif 'High' in val:
                    return 'color: orange'
                elif 'Extreme' in val:
                    return 'color: red'
                return ''

            def color_value(val):
                if pd.isna(val):
                    return ''
                elif val < 30:
                    return 'color: green'
                elif val < 60:
                    return 'color: #aaaa00'
                elif val < 100:
                    return 'color: orange'
                else:
                    return 'color: red'

            # Apply styling
            styled_ranking = ranking_df.style\
                .applymap(color_regime, subset=['Regime'])\
                .applymap(color_value, subset=['Avg Vol (%)', 'Max Vol (%)', 'Min Vol (%)'])

            # Display the styled dataframe
            st.dataframe(styled_ranking, height=500, use_container_width=True)
        else:
            st.warning("No ranking data available.")

        # Identify and display extreme volatility events
        st.subheader("Extreme Volatility Events (>= 100% Annualized)")

        extreme_events = []
        for token, df in token_results.items():
            if not df.empty and 'is_extreme' in df.columns:
                extreme_periods = df[df['is_extreme']]
                for idx, row in extreme_periods.iterrows():
                    # Safely access values with explicit casting to avoid attribute errors
                    vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                    time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"

                    extreme_events.append({
                        'Token': token,
                        'Time': time_label,
                        'Volatility (%)': round(vol_value * 100, 1),
                        'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M')
                    })

        if extreme_events:
            extreme_df = pd.DataFrame(extreme_events)
            # Sort by volatility (highest first)
            extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)

            # Reset the index to remove it
            extreme_df = extreme_df.reset_index(drop=True)

            # Display the dataframe
            st.dataframe(extreme_df, height=300, use_container_width=True)

            # Create a more visually appealing list of extreme events
            st.markdown("### Extreme Volatility Events Detail")

            # Only process top 10 events if there are any
            top_events = extreme_events[:min(10, len(extreme_events))]
            for i, event in enumerate(top_events):
                token = event['Token']
                time = event['Time']
                vol = event['Volatility (%)']
                date = event['Full Timestamp'].split(' ')[0]

                st.markdown(f"**{i+1}. {token}** at **{time}** on {date}: <span style='color:red; font-weight:bold;'>{vol}%</span> volatility", unsafe_allow_html=True)

            if len(extreme_events) > 10:
                st.markdown(f"*... and {len(extreme_events) - 10} more extreme events*")

        else:
            st.info("No extreme volatility events detected in the selected tokens.")

        # 24-Hour Average Volatility Distribution
        st.subheader("24-Hour Average Volatility Overview (Singapore Time)")
        avg_values = {}
        for token, df in token_results.items():
            if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
                avg = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
                regime = df['avg_vol_desc'].iloc[0]
                avg_values[token] = (avg, regime)

        if avg_values:
            low_vol = sum(1 for v, r in avg_values.values() if v < 0.3)
            medium_vol = sum(1 for v, r in avg_values.values() if 0.3 <= v < 0.6)
            high_vol = sum(1 for v, r in avg_values.values() if 0.6 <= v < 1.0)
            extreme_vol = sum(1 for v, r in avg_values.values() if v >= 1.0)
            total = low_vol + medium_vol + high_vol + extreme_vol

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Low Vol", f"{low_vol} ({low_vol/total*100:.1f}%)")
            col2.metric("Medium Vol", f"{medium_vol} ({medium_vol/total*100:.1f}%)")
            col3.metric("High Vol", f"{high_vol} ({high_vol/total*100:.1f}%)")
            col4.metric("Extreme Vol", f"{extreme_vol} ({extreme_vol/total*100:.1f}%)")

            labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
            values = [low_vol, medium_vol, high_vol, extreme_vol]
            colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
            fig.update_layout(
                title="24-Hour Average Volatility Distribution (Singapore Time)",
                height=400,
                font=dict(color="#000000", size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Create columns for each volatility category
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### Low Average Volatility Tokens")
                lv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v < 0.3]
                lv_tokens.sort(key=lambda x: x[1])
                if lv_tokens:
                    for token, value, regime in lv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")

            with col2:
                st.markdown("### Medium Average Volatility Tokens")
                mv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.3 <= v < 0.6]
                mv_tokens.sort(key=lambda x: x[1])
                if mv_tokens:
                    for token, value, regime in mv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")

            with col3:
                st.markdown("### High Average Volatility Tokens")
                hv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.6 <= v < 1.0]
                hv_tokens.sort(key=lambda x: x[1])
                if hv_tokens:
                    for token, value, regime in hv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")

            with col4:
                st.markdown("### Extreme Average Volatility Tokens")
                ev_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v >= 1.0]
                ev_tokens.sort(key=lambda x: x[1], reverse=True)
                if ev_tokens:
                    for token, value, regime in ev_tokens:
                        st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")
        else:
            st.warning("No average volatility data available for the selected tokens.")

    with st.expander("Understanding the Volatility Table"):
        st.markdown("""
        ### How to Read This Table
        This table shows annualized volatility values for all selected tokens over the last 24 hours using 30-minute bars.
        Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.

        **Color coding:**
        - **Green** (< 30%): Low volatility
        - **Yellow** (30-60%): Medium volatility
        - **Orange** (60-100%): High volatility
        - **Red** (> 100%): Extreme volatility

        **The intensity of the color indicates the strength of the volatility:**
        - Darker green = Lower volatility
        - Darker red = Higher volatility

        **Ranking Table:**
        The ranking table sorts tokens by their 24-hour average volatility from highest to lowest.

        **Extreme Volatility Events:**
        These are specific 30-minute periods where a token's annualized volatility exceeded 100%.

        **Technical details:**
        - Volatility is calculated as the standard deviation of log returns, annualized to represent the expected price variation over a year
        - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
        - The calculation uses a rolling window of 20 one-minute price points
        - The 24-hour average section shows the mean volatility across all 48 30-minute periods
        - Missing values (light gray cells) indicate insufficient data for calculation
        """)


tab_functions = {
    "Macro View": render_macro_view,
    "Cumulative PnL": render_cumulative_pnl,
    "PnL and Trades": render_pnl_and_trades,
    "Regime Matrix": render_regime_matrix,
    "Spread Analysis": render_spread_analysis,
    "Vol & Hurst": render_vol_and_hurst
}

for tab, name in zip(tabs, tab_names):
    with tab:
        st.markdown(f"### {name}")
        if st.button(f"↻ Refresh {name}", key=name):
            st.cache_data.clear()
            st.experimental_rerun()
        tab_functions[name]()
