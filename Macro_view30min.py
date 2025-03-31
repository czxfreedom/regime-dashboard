import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta, timezone
import pytz

st.set_page_config(
    page_title="Daily Hurst Table",
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

# Timezone for Singapore
tz_sg = pytz.timezone("Asia/Singapore")

def get_available_pairs():
    query = "SELECT DISTINCT pair_name FROM public.tick_data"
    return pd.read_sql(query, engine)['pair_name'].tolist()

def get_tick_data(pair, start_time, end_time):
    query = f"""
        SELECT timestamp, price 
        FROM public.tick_data
        WHERE pair_name = '{pair}' 
            AND timestamp >= '{start_time}'
            AND timestamp < '{end_time}'
        ORDER BY timestamp
    """
    return pd.read_sql(query, engine, parse_dates=['timestamp'])

def calculate_hurst_exponent(series):
    # Calculate Hurst Exponent using Rescaled Range (R/S) analysis
    # Implementation details omitted for brevity
    pass

def process_pair_data(pair, start_time, end_time):
    result = []
    current_time = start_time
    while current_time < end_time:
        block_end_time = current_time + timedelta(minutes=30)
        block_result = []
        
        for _ in range(6):  # 6 x 5-minute intervals in 30 minutes
            interval_end_time = current_time + timedelta(minutes=5)
            tick_data = get_tick_data(pair, current_time, interval_end_time)
            if not tick_data.empty:
                hurst = calculate_hurst_exponent(tick_data['price'])
                block_result.append(hurst)
            current_time = interval_end_time
        
        if block_result:
            avg_hurst = np.mean(block_result)
            result.append((block_end_time, avg_hurst))
        
    return pd.DataFrame(result, columns=['timestamp', 'hurst']).set_index('timestamp')

# Main App
st.title("Daily Hurst Table")
st.subheader("Market Regime Analysis")

if st.button("Generate Hurst Table"):
    pairs = get_available_pairs()
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = {}

    start_time = datetime.now(tz_sg).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)

    for i, pair in enumerate(pairs):
        progress_bar.progress((i+1)/len(pairs))
        status_text.text(f"Processing {pair} ({i+1}/{len(pairs)})")

        try:
            pair_result = process_pair_data(pair, start_time, end_time)
            results[pair] = pair_result
        except Exception as e:
            st.error(f"Error processing {pair}: {e}")

    progress_bar.progress(1.0)
    status_text.text("Processing complete")

    if results:
        hurst_table = pd.concat(results, axis=1)
        hurst_table.columns = pd.MultiIndex.from_product([hurst_table.columns, ['Hurst']])
        hurst_table = hurst_table.applymap(lambda x: round(x, 2) if not pd.isna(x) else x)
        hurst_table.index = hurst_table.index.tz_localize('UTC').tz_convert(tz_sg)
        hurst_table.index.name = "Time (SGT)"

        def color_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5'
            elif val < 0.4:
                intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
                return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
            elif val > 0.6:
                intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            else:
                return 'background-color: rgba(200, 200, 200, 0.5); color: black'

        styled_table = hurst_table.style.applymap(color_cells)
        st.dataframe(styled_table, height=700, use_container_width=True)

        # Market Regime Summary
        st.subheader("Market Regime Overview")
        latest_values = hurst_table.iloc[-1]  # Latest values for each pair

        mean_reverting, random_walk, trending = [], [], []

        for pair, hurst in latest_values.iteritems():
            if not pd.isna(hurst):
                if hurst < 0.4:
                    mean_reverting.append((pair, hurst))
                elif hurst > 0.6:
                    trending.append((pair, hurst))
                else:
                    random_walk.append((pair, hurst))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Mean-Reverting Pairs")
            if mean_reverting:
                mean_reverting.sort(key=lambda x: x[1])
                for pair, hurst in mean_reverting:
                    st.markdown(f"- **{pair}**: {hurst:.2f}")
            else:
                st.markdown("*No mean-reverting pairs*")

        with col2:
            st.markdown("### Random Walk Pairs")
            if random_walk:
                random_walk.sort(key=lambda x: x[1])
                for pair, hurst in random_walk:
                    st.markdown(f"- **{pair}**: {hurst:.2f}")
            else:
                st.markdown("*No random walk pairs*")

        with col3:
            st.markdown("### Trending Pairs")
            if trending:
                trending.sort(key=lambda x: x[1], reverse=True)
                for pair, hurst in trending:
                    st.markdown(f"- **{pair}**: {hurst:.2f}")
            else:
                st.markdown("*No trending pairs*")

        st.subheader("Market Regime Distribution")
        regime_counts = {
            'Mean-Reverting': len(mean_reverting),
            'Random Walk': len(random_walk),
            'Trending': len(trending)
        }

        fig = go.Figure(data=[go.Pie(
            labels=list(regime_counts.keys()),
            values=list(regime_counts.values()),
            hole=0.3,
            marker_colors=['red', 'gray', 'green']
        )])
        fig.update_layout(title="Market Regime Distribution Across Pairs")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Market Statistics")
        all_hursts = latest_values[~pd.isna(latest_values)]
        st.metric("Average Hurst", f"{all_hursts.mean():.2f}")
        st.metric("Median Hurst", f"{all_hursts.median():.2f}")

        st.subheader("Analysis & Insights")
        st.markdown("""
        - **Hurst < 0.4** indicates a **mean-reverting regime** – good for counter-trend or reversion strategies.
        - **Hurst ~ 0.5** shows a **random walk**, indicating uncertainty and potentially range-bound or noisy markets.
        - **Hurst > 0.6** suggests a **trending market** – better for breakout and momentum strategies.
        - Distribution skew toward one regime often signals macro market behavior (e.g., volatility clusters or trend persistence).
        - Use this data with volatility, volume, and funding data to develop adaptive trading systems.
        """)

    else:
        st.warning("No data processed")