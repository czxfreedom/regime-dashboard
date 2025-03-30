import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- Setup ---
st.title("Currency Pair Trend Matrix Dashboard")

# --- DB CONFIG ---
db_config = st.secrets["database"]
db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(db_uri)

# --- Parameters ---
# Get all pairs
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
lookback_days = st.slider("Lookback (Days)", 1, 30, 7)
rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 50)

# --- Helper Functions ---
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

def get_hurst_data(pair, timeframe, lookback_days, rolling_window):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    query = f"""
    SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
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

# --- Render Matrix ---
st.subheader("Trend Regime Matrix")
if not selected_pairs or not selected_timeframes:
    st.warning("Please select at least one pair and timeframe")
    st.stop()

for pair in selected_pairs:
    st.markdown(f"### {pair}")
    cols = st.columns(len(selected_timeframes))

    for i, timeframe in enumerate(selected_timeframes):
        with cols[i]:
            st.markdown(f"**{timeframe}**")
            ohlc = get_hurst_data(pair, timeframe, lookback_days, rolling_window)

            if ohlc is None or ohlc.empty:
                st.write("No data available")
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ohlc.index,
                y=ohlc['close'],
                line=dict(color='gray', width=1),
                name='Price'
            ))

            for j in range(1, len(ohlc)):
                regime = ohlc['regime'].iloc[j-1]
                intensity = ohlc['intensity'].iloc[j-1]
                desc = ohlc['regime_desc'].iloc[j-1]

                if regime == "MEAN-REVERT":
                    shades = ['rgba(255,200,200,0.6)', 'rgba(255,100,100,0.6)', 'rgba(255,0,0,0.6)']
                    color = shades[intensity-1]
                elif regime == "TREND":
                    shades = ['rgba(200,255,200,0.6)', 'rgba(100,255,100,0.6)', 'rgba(0,200,0,0.6)']
                    color = shades[intensity-1]
                else:
                    color = 'rgba(220,220,220,0.5)'

                fig.add_vrect(
                    x0=ohlc.index[j-1],
                    x1=ohlc.index[j],
                    fillcolor=color,
                    opacity=0.8,
                    layer="below",
                    line_width=0
                )

            if not ohlc.empty and not pd.isna(ohlc['Hurst'].iloc[-1]):
                current_hurst = ohlc['Hurst'].iloc[-1]
                current_regime = ohlc['regime'].iloc[-1]
                color = "red" if current_regime == "MEAN-REVERT" else "green" if current_regime == "TREND" else "gray"
                fig.update_layout(title=f"{current_regime} | H={current_hurst:.2f}", title_font_color=color)

            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )

            st.plotly_chart(fig, use_container_width=True)

# Add Regime Legend
st.markdown("---")
st.markdown("""
**Regime Color Code:**
- 🟥 **Mean-Revert**
- 🟩 **Trending**
- ⬜ **Noise/Unknown**
""")
