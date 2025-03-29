import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

# --- UI Sidebar ---
st.title("Rolling tHurst Exponent Dashboard")

# --- Fetch token list from DB ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

token_list = fetch_token_list()
selected_token = st.selectbox("Select Token", token_list, index=0)
timeframe = st.selectbox("Timeframe", ["30s", "15min", "30min", "1h", "6h"], index=2)
lookback_days = st.slider("Lookback (Days)", 1, 30, 2)
rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 20)

# --- Determine Bars per Hour ---
bars_per_hour = {"30s": 120, "15min": 4, "30min": 2, "1h": 1, "6h": 1/6}[timeframe]
expected_bars = lookback_days * 24 * bars_per_hour

if expected_bars < rolling_window + 10:
    st.warning("⚠️ Not enough data for this rolling window. Increase lookback or reduce window.")

# --- Fetch Oracle Price Data ---
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=lookback_days)

query = f"""
SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
FROM public.oracle_price_log
WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
AND pair_name = '{selected_token}';
"""
df = pd.read_sql(query, engine)

if df.empty:
    st.warning("No data found for selected pair and timeframe.")
    st.stop()

# --- Preprocess ---
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# --- Resample to OHLC ---
ohlc = df['final_price'].resample(timeframe).ohlc().dropna()

# --- More Robust Hurst Calculation ---
def compute_hurst(ts):
    ts = np.array(ts)
    if len(ts) < 10 or np.std(ts) < 1e-8:
        return np.nan
    lags = range(2, min(len(ts) - 1, 10))
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        if std_diff < 1e-8:
            continue
        tau.append(std_diff)
    if len(tau) < 4:
        return np.nan
    try:
        poly = np.polyfit(np.log(list(lags[:len(tau)])), np.log(tau), 1)
        hurst = poly[0]
        if hurst < -1 or hurst > 2:
            return np.nan
        return hurst
    except:
        return np.nan


# --- Regime Classification ---
def classify_regime(hurst):
    if pd.isna(hurst):
        return "UNKNOWN"
    elif hurst < 0.4:
        return "MEAN-REVERT"
    elif hurst > 0.6:
        return "TREND"
    else:
        return "NOISE"

# --- Compute Rolling tHurst and Regime ---
ohlc['tHurst'] = ohlc['close'].rolling(rolling_window).apply(compute_hurst)
ohlc['regime'] = ohlc['tHurst'].apply(classify_regime)

# --- Chart Output ---
st.subheader(f"Rolling tHurst for {selected_token} ({timeframe})")
if ohlc['tHurst'].dropna().empty:
    st.warning("⚠️ No valid tHurst values computed. Price may be too flat or data too sparse.")
else:
    fig = px.line(
        ohlc.reset_index(),
        x='timestamp',
        y='tHurst',
        title=f"Rolling tHurst for {selected_token} ({timeframe})",
        markers=True
    )
    fig.update_layout(
        yaxis_title="tHurst Exponent",
        xaxis_title="Time",
        yaxis_range=[0, 1],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Table Display ---
st.markdown("### Regime Table (Most Recent 100 Bars)")
st.dataframe(ohlc.sort_index(ascending=False).head(100))