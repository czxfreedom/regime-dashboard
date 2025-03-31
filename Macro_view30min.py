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

def get_available_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log"
    return pd.read_sql(query, engine)['pair_name'].tolist()

# Main App
st.title("Daily Hurst Table")
st.subheader("Market Regime Analysis")

if st.button("Generate Hurst Table"):
    tokens = get_available_tokens()
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = {}

    for i, token in enumerate(tokens):
        progress_bar.progress((i+1)/len(tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(tokens)})")

        try:
            token_result = process_token_data(token)
            if token_result is not None:
                if isinstance(token_result.index, pd.DatetimeIndex):
                    token_result.index = token_result.index.tz_localize('UTC').tz_convert(tz_sg)
                    token_result.sort_index(ascending=False, inplace=True)  # Latest first
                results[token] = token_result
        except Exception as e:
            st.error(f"Error processing {token}: {e}")

    progress_bar.progress(1.0)
    status_text.text("Processing complete")

    if results:
        hurst_table = pd.DataFrame(results).round(2)
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
        latest_values = {}
        for token, df in results.items():
            if not df.empty and not df.isna().all():
                latest = df.dropna().iloc[0]  # Already sorted descending
                latest_values[token] = latest

        mean_reverting, random_walk, trending = [], [], []

        for token, hurst in latest_values.items():
            if hurst < 0.4:
                mean_reverting.append((token, hurst))
            elif hurst > 0.6:
                trending.append((token, hurst))
            else:
                random_walk.append((token, hurst))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Mean-Reverting Tokens")
            if mean_reverting:
                mean_reverting.sort(key=lambda x: x[1])
                for token, hurst in mean_reverting:
                    st.markdown(f"- **{token}**: {hurst:.2f}")
            else:
                st.markdown("*No mean-reverting tokens*")

        with col2:
            st.markdown("### Random Walk Tokens")
            if random_walk:
                random_walk.sort(key=lambda x: x[1])
                for token, hurst in random_walk:
                    st.markdown(f"- **{token}**: {hurst:.2f}")
            else:
                st.markdown("*No random walk tokens*")

        with col3:
            st.markdown("### Trending Tokens")
            if trending:
                trending.sort(key=lambda x: x[1], reverse=True)
                for token, hurst in trending:
                    st.markdown(f"- **{token}**: {hurst:.2f}")
            else:
                st.markdown("*No trending tokens*")

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
        fig.update_layout(title="Market Regime Distribution Across Tokens")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Market Statistics")
        all_hursts = [h for _, h in latest_values.items()]
        st.metric("Average Hurst", f"{np.mean(all_hursts):.2f}")
        st.metric("Median Hurst", f"{np.median(all_hursts):.2f}")

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
