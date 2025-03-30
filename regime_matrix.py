import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- Helper Functions (Shared with Dashboard) ---
def universal_hurst(ts):
    ts = np.array(ts)
    if len(ts) < 20 or np.std(ts) == 0:
        return np.nan
    lags = range(2, 20)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        if len(diff) == 0 or np.std(diff) == 0:
            return np.nan
        tau.append(np.std(diff))
    if any(t == 0 for t in tau):
        return np.nan
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

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

# --- Setup ---
st.set_page_config(layout="wide")
st.title("Currency Pair Trend Matrix Dashboard")

# Regime Color Legend ---
st.markdown("""
### Regime Color Code:
<ul>
  <li><span style='color: red; font-weight:bold'>⬤</span> <strong>Strong Mean-Revert</strong></li>
  <li><span style='color: #FF6666; font-weight:bold'>⬤</span> Moderate Mean-Revert</li>
  <li><span style='color: #FF9999; font-weight:bold'>⬤</span> Mild Mean-Revert</li>
  <li><span style='color: gray; font-weight:bold'>⬤</span> Noise / Random Walk</li>
  <li><span style='color: #99FF99; font-weight:bold'>⬤</span> Mild Trending</li>
  <li><span style='color: #66FF66; font-weight:bold'>⬤</span> Moderate Trending</li>
  <li><span style='color: green; font-weight:bold'>⬤</span> <strong>Strong Trending</strong></li>
</ul>
""", unsafe_allow_html=True)

# --- DB CONFIG ---
db_config = st.secrets["database"]
db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(db_uri)

# --- Parameters ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

all_pairs = fetch_token_list()
selected_pairs = st.multiselect("Select Currency Pairs", all_pairs, default=all_pairs[:5])

# Timeframes
timeframes = ["15min", "30min", "1h", "4h", "6h"]
selected_timeframes = st.multiselect("Select Timeframes", timeframes, default=["15min", "1h", "6h"])

# Common settings
col1, col2 = st.columns(2)
with col1:
    lookback_days = st.slider("Lookback (Days)", 1, 30, 7)
with col2:
    rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 50)

if not selected_pairs or not selected_timeframes:
    st.warning("Please select at least one pair and timeframe")
    st.stop()

def get_hurst_data(pair, timeframe, lookback_days, rolling_window):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    query = f"""
        SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price
        FROM public.oracle_price_log
        WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
        AND pair_name = '{pair}';
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    ohlc = df['final_price'].resample(timeframe).ohlc().dropna()
    ohlc['Hurst'] = ohlc['close'].rolling(rolling_window).apply(universal_hurst)
    ohlc['regime_info'] = ohlc['Hurst'].apply(detailed_regime_classification)
    ohlc['regime'] = ohlc['regime_info'].apply(lambda x: x[0])
    ohlc['desc'] = ohlc['regime_info'].apply(lambda x: x[2])
    return ohlc

# --- Render Grid ---
for pair in selected_pairs:
    st.markdown(f"## {pair}")
    cols = st.columns(len(selected_timeframes))
    for i, timeframe in enumerate(selected_timeframes):
        with cols[i]:
            st.markdown(f"### {timeframe}")
            ohlc = get_hurst_data(pair, timeframe, lookback_days, rolling_window)
            if ohlc is None or ohlc.empty:
                st.write("No data available")
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ohlc.index, y=ohlc['close'], mode='lines', name='Price', line=dict(color='black')))

            for j in range(1, len(ohlc)):
                r = ohlc['regime'].iloc[j-1]
                desc = ohlc['desc'].iloc[j-1]
                color_map = {
                    "MEAN-REVERT": ['rgba(255,0,0,0.3)', 'rgba(255,100,100,0.3)', 'rgba(255,150,150,0.3)'],
                    "NOISE": ['rgba(200,200,200,0.3)'],
                    "TREND": ['rgba(0,255,0,0.3)', 'rgba(100,255,100,0.3)', 'rgba(150,255,150,0.3)']
                }
                if pd.isna(r):
                    continue
                shade_color = color_map[r][0] if r != "NOISE" else color_map[r][0]
                fig.add_vrect(x0=ohlc.index[j-1], x1=ohlc.index[j], fillcolor=shade_color, opacity=0.5, line_width=0)

            if not ohlc.empty and not pd.isna(ohlc['Hurst'].iloc[-1]):
                h = ohlc['Hurst'].iloc[-1]
                r = ohlc['regime'].iloc[-1]
                label_color = "red" if r == "MEAN-REVERT" else ("green" if r == "TREND" else "gray")
                fig.update_layout(title=f"{r} | H={h:.2f}", title_font_color=label_color)

            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                              xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
