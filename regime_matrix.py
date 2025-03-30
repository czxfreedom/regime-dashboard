import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- Setup ---
st.set_page_config(layout="wide")
st.title("Currency Pair Trend Matrix Dashboard")

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
lookback_days = st.slider("Lookback (Days)", 1, 30, 7)
rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 50)

# --- Hurst Calculation ---
def universal_hurst(ts):
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
        if hurst < 0 or hurst > 1.5:
            return np.nan
        return hurst
    except:
        return np.nan

def detailed_regime_classification(h):
    if pd.isna(h):
        return ('UNKNOWN', '')
    elif h < 0.4:
        return ('MEAN-REVERT', 'red')
    elif h > 0.6:
        return ('TREND', 'green')
    else:
        return ('NOISE', 'gray')

# --- Get Hurst Data ---
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
    ohlc['regime_color'] = ohlc['regime_info'].apply(lambda x: x[1])
    return ohlc

# --- Collect regime summary table ---
summary_data = []

# --- Render Matrix ---
if not selected_pairs or not selected_timeframes:
    st.warning("Please select at least one pair and one timeframe")
    st.stop()

st.subheader("Trend Regime Visual Matrix")

for pair in selected_pairs:
    st.markdown(f"### {pair}")
    cols = st.columns(len(selected_timeframes))

    for i, timeframe in enumerate(selected_timeframes):
        with cols[i]:
            st.markdown(f"**{timeframe}**")
            ohlc = get_hurst_data(pair, timeframe, lookback_days, rolling_window)

            if ohlc is None or ohlc.empty:
                st.write("⚠️ No data available for this timeframe.")
                summary_data.append((pair, timeframe, 'NO DATA'))
                continue

            if ohlc['Hurst'].dropna().empty:
                st.write("⚠️ No valid Hurst values. Try increasing lookback or reducing window.")
                summary_data.append((pair, timeframe, 'NO HURST'))
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ohlc.index,
                y=ohlc['close'],
                line=dict(color='gray', width=1),
                name='Price'
            ))

            for j in range(1, len(ohlc)):
                regime = ohlc['regime'].iloc[j - 1]
                color = {
                    "MEAN-REVERT": "rgba(255,0,0,0.2)",
                    "TREND": "rgba(0,200,0,0.2)",
                    "NOISE": "rgba(200,200,200,0.3)"
                }.get(regime, "rgba(150,150,150,0.1)")
                fig.add_vrect(
                    x0=ohlc.index[j - 1], x1=ohlc.index[j],
                    fillcolor=color, opacity=0.3, layer="below", line_width=0
                )

            current_hurst = ohlc['Hurst'].iloc[-1]
            current_regime = ohlc['regime'].iloc[-1]
            regime_color = {
                "MEAN-REVERT": "red",
                "TREND": "green",
                "NOISE": "gray",
                "UNKNOWN": "black"
            }.get(current_regime, "black")

            summary_data.append((pair, timeframe, f"{current_regime} ({current_hurst:.2f})" if not pd.isna(current_hurst) else "UNKNOWN"))

            fig.update_layout(
                title=f"{current_regime} | H={current_hurst:.2f}" if not pd.isna(current_hurst) else "Regime Unknown",
                title_font_color=regime_color,
                height=180, width=250,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )

            st.plotly_chart(fig, use_container_width=True)

# --- Tabular Summary ---
st.subheader("📊 Regime Table Summary")
summary_df = pd.DataFrame(summary_data, columns=["Pair", "Timeframe", "Regime"])
pivot_df = summary_df.pivot(index="Pair", columns="Timeframe", values="Regime").fillna("-")
st.dataframe(pivot_df, use_container_width=True, height=300)
