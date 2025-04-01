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
singapore_timezone = pytz.timezone('Asia/Singapore')
extreme_vol_threshold = 1.0  # 100% annualized volatility

# Fetch all available tokens from DB
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
    select_all = st.checkbox("Select All Tokens", value=True)
    if select_all:
        selected_tokens = all_tokens
    else:
        selected_tokens = st.multiselect("Select Tokens", all_tokens, default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens)

with col2:
    if st.button("Refresh Data"):
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

def calculate_volatility_metrics(price_series):
    if price_series is None or len(price_series) < 2:
        return {'realized_vol': np.nan}
    try:
        log_returns = np.diff(np.log(price_series))
        realized_vol = np.std(log_returns) * np.sqrt(252 * 48)
        return {'realized_vol': realized_vol}
    except Exception as e:
        print(f"Error in volatility calculation: {e}")
        return {'realized_vol': np.nan}

def classify_volatility(vol):
    if pd.isna(vol):
        return ("UNKNOWN", 0, "Insufficient data")
    elif vol < 0.30:
        return ("LOW", 1, "Low volatility")
    elif vol < 0.60:
        return ("MEDIUM", 2, "Medium volatility")
    elif vol < 1.00:
        return ("HIGH", 3, "High volatility")
    else:
        return ("EXTREME", 4, "Extreme volatility")

def fetch_and_calculate_volatility(token):
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    query = f"""
    SELECT created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, final_price
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}' AND pair_name = '{token}';
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            return None

        one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(lambda x: calculate_volatility_metrics(x)['realized_vol'])
        thirty_min_vol = one_min_ohlc['realized_vol'].resample('30min').mean().dropna()
        if thirty_min_vol.empty:
            return None

        last_24h_vol = thirty_min_vol.iloc[-48:]
        last_24h_vol = last_24h_vol.to_frame()
        last_24h_vol['time_label'] = last_24h_vol.index.strftime('%H:%M')
        last_24h_vol['avg_24h_vol'] = last_24h_vol['realized_vol'].mean()
        last_24h_vol['vol_info'] = last_24h_vol['realized_vol'].apply(classify_volatility)
        last_24h_vol['vol_regime'] = last_24h_vol['vol_info'].apply(lambda x: x[0])
        last_24h_vol['vol_desc'] = last_24h_vol['vol_info'].apply(lambda x: x[2])
        last_24h_vol['avg_vol_info'] = last_24h_vol['avg_24h_vol'].apply(classify_volatility)
        last_24h_vol['avg_vol_regime'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[0])
        last_24h_vol['avg_vol_desc'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[2])
        last_24h_vol['is_extreme'] = last_24h_vol['realized_vol'] >= extreme_vol_threshold
        return last_24h_vol
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

token_results = {}
for token in selected_tokens:
    result = fetch_and_calculate_volatility(token)
    if result is not None:
        token_results[token] = result

if token_results:
    all_times = sorted(token_results[list(token_results.keys())[0]].index, reverse=True)
    table_data = {}
    for token, df in token_results.items():
        vol_series = df.set_index('time_label')['realized_vol']
        table_data[token] = vol_series
    vol_table = pd.DataFrame(table_data).reindex(all_times)
    vol_table = (vol_table * 100).round(1)

    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'
        elif val < 30:
            intensity = max(0, min(255, int(255 * val / 30)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 60:
            intensity = max(0, min(255, int(255 * (val - 30) / 30)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < 100:
            intensity = max(0, min(255, int(255 * (val - 60) / 40)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'

    styled_table = vol_table.style.applymap(color_cells)
    st.markdown("## Volatility Table (30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
    st.markdown("Values shown as annualized volatility percentage")
    st.dataframe(styled_table, height=700, use_container_width=True)

    st.subheader("Volatility Ranking (24-Hour Average, Descending Order)")
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
            avg_vol = df['avg_24h_vol'].iloc[0]
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
        ranking_df = pd.DataFrame(ranking_data).sort_values(by='Avg Vol (%)', ascending=False).reset_index(drop=True)
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))

        def color_regime(val):
            if 'Low' in val: return 'color: green'
            elif 'Medium' in val: return 'color: #aaaa00'
            elif 'High' in val: return 'color: orange'
            elif 'Extreme' in val: return 'color: red'
            return ''

        def color_value(val):
            if pd.isna(val): return ''
            elif val < 30: return 'color: green'
            elif val < 60: return 'color: #aaaa00'
            elif val < 100: return 'color: orange'
            else: return 'color: red'

        styled_ranking = ranking_df.style.applymap(color_regime, subset=['Regime']).applymap(color_value, subset=['Avg Vol (%)', 'Max Vol (%)', 'Min Vol (%)'])
        st.dataframe(styled_ranking, height=500, use_container_width=True)
    else:
        st.warning("No ranking data available.")

    st.subheader("Extreme Volatility Events (>= 100% Annualized)")
    extreme_events = []
    for token, df in token_results.items():
        if not df.empty and 'is_extreme' in df.columns:
            extreme_periods = df[df['is_extreme']]
            for idx, row in extreme_periods.iterrows():
                vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                extreme_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M')
                })

    if extreme_events:
        extreme_df = pd.DataFrame(extreme_events).sort_values(by='Volatility (%)', ascending=False).reset_index(drop=True)
        st.dataframe(extreme_df, height=300, use_container_width=True)
        st.markdown("### Extreme Volatility Events Detail")
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

    st.subheader("24-Hour Average Volatility Overview (Singapore Time)")
    avg_values = {}
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
            avg = df['avg_24h_vol'].iloc[0]
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
        fig.update_layout(title="24-Hour Average Volatility Distribution (Singapore Time)", height=400, font=dict(color="#000000", size=12))
        st.plotly_chart(fig, use_container_width=True)

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