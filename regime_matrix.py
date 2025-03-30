import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- Setup ---
st.set_page_config(layout="wide")
st.title("📈 Currency Pair Trend Matrix Dashboard")

# --- Regime Color Map with Intensities ---
color_map = {
    "MEAN-REVERT": ["rgba(255,0,0,0.4)", "rgba(255,100,100,0.4)", "rgba(255,150,150,0.4)"],
    "TREND": ["rgba(0,200,0,0.4)", "rgba(100,255,100,0.4)", "rgba(150,255,150,0.4)"],
    "NOISE": ["rgba(200,200,200,0.3)"]
}

# --- DB CONFIG ---
db_config = st.secrets["database"]
db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(db_uri)

# --- Hurst & Regime Logic ---
def universal_hurst(ts):
    ts = np.array(ts)
    if len(ts) < 20 or np.std(ts) == 0:
        return np.nan
    lags = range(2, 20)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 0]
    if len(tau) < 2:
        return np.nan
    poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
    return poly[0]

def detailed_regime_classification(hurst):
    if pd.isna(hurst):
        return ("UNKNOWN", 0, "Insufficient data")
    elif hurst < 0.2:
        return ("MEAN-REVERT", 0, "Strong Mean-Reversion")
    elif hurst < 0.3:
        return ("MEAN-REVERT", 1, "Moderate Mean-Reversion")
    elif hurst < 0.4:
        return ("MEAN-REVERT", 2, "Mild Mean-Reversion")
    elif hurst < 0.6:
        return ("NOISE", 0, "Random / Unclear")
    elif hurst < 0.7:
        return ("TREND", 2, "Mild Trend")
    elif hurst < 0.8:
        return ("TREND", 1, "Moderate Trend")
    else:
        return ("TREND", 0, "Strong Trend")

# --- Sidebar Parameters ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

all_pairs = fetch_token_list()
selected_pairs = st.multiselect("Select Currency Pairs", all_pairs, default=all_pairs[:3])
timeframes = ["15min", "1h", "6h"]
selected_timeframes = st.multiselect("Select Timeframes", timeframes, default=timeframes)
col1, col2 = st.columns(2)
lookback_days = col1.slider("Lookback (Days)", 1, 30, 3)
rolling_window = col2.slider("Rolling Window (Bars)", 20, 100, 23)

# --- Color Code Legend ---
with st.expander("Legend: Regime Colors", expanded=True):
    st.markdown("""
    - <span style='background-color:rgba(255,0,0,0.6);padding:3px'>**Mean-Revert**</span>  
    - <span style='background-color:rgba(0,200,0,0.6);padding:3px'>**Trending**</span>  
    - <span style='background-color:rgba(200,200,200,0.4);padding:3px'>**Noise / Unknown**</span>
    """, unsafe_allow_html=True)

# --- Data Fetching ---
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
    ohlc['intensity'] = ohlc['regime_info'].apply(lambda x: x[1])
    ohlc['regime_desc'] = ohlc['regime_info'].apply(lambda x: x[2])

    return ohlc

# --- Display Matrix ---
for pair in selected_pairs:
    st.subheader(f"📌 {pair}")
    cols = st.columns(len(selected_timeframes))

    for i, tf in enumerate(selected_timeframes):
        with cols[i]:
            st.markdown(f"**{tf}**")
            ohlc = get_hurst_data(pair, tf, lookback_days, rolling_window)

            if ohlc is None or ohlc.empty:
                st.write("No data")
                continue

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ohlc.index, y=ohlc['close'], mode='lines', line=dict(color='black'), name='Price'))

            # Background regime color
            for j in range(1, len(ohlc)):
                r = ohlc['regime'].iloc[j-1]
                intensity = ohlc['intensity'].iloc[j-1]
                shade_color = color_map.get(r, ("rgba(200,200,200,0.3)",))[intensity if r in color_map and len(color_map[r]) > intensity else 0]

                fig.add_vrect(
                    x0=ohlc.index[j-1], x1=ohlc.index[j],
                    fillcolor=shade_color, opacity=0.5,
                    layer="below", line_width=0
                )

            current_hurst = ohlc['Hurst'].iloc[-1]
            current_regime = ohlc['regime'].iloc[-1]
            display_text = f"{current_regime} | H={current_hurst:.2f}" if not pd.isna(current_hurst) else "UNKNOWN | H=nan"
            fig.update_layout(
                title=dict(
                    text=f"<b>{display_text}</b>",
                    font=dict(color="red" if current_regime == "MEAN-REVERT" else ("green" if current_regime == "TREND" else "gray"))
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=200,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
