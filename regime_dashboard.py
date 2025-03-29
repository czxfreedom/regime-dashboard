import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.title("Rolling Hurst Exponent Dashboard")

# --- Fetch token list from DB ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

token_list = fetch_token_list()
selected_token = st.selectbox("Select Token", token_list, index=0)
timeframe = st.selectbox("Timeframe", ["30s", "15min", "30min", "1h", "6h"], index=2)

col1, col2 = st.columns(2)
with col1:
    lookback_days = st.slider("Lookback (Days)", 1, 30, 2)
with col2:
    rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 20)

# --- Determine Bars per Hour and calculate expected data points ---
bars_per_hour = {"30s": 120, "15min": 4, "30min": 2, "1h": 1, "6h": 1/6}[timeframe]
expected_bars = int(lookback_days * 24 * bars_per_hour)
expected_points = max(0, expected_bars - rolling_window + 1)  # Points that can be plotted

# Show data point information
st.info(f"📊 Data Point Information: Based on your settings, expecting ~{expected_bars} total bars and ~{expected_points} plotted Hurst values.")

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

# --- Improved Hurst Calculation ---
def improved_hurst(ts):
    """
    A more robust Hurst exponent calculation using multiple methods.
    
    Args:
        ts: Time series (numpy array or list)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    ts = np.array(ts)
    
    # Check if we have enough data
    if len(ts) < 20 or np.std(ts) < 1e-8:
        return np.nan
    
    # Method 1: R/S Analysis
    def rs_analysis():
        # Create range of lag values
        lags = range(2, min(len(ts) // 4, 20))
        tau = []
        
        # Calculate R/S for each lag
        for lag in lags:
            # Split ts into chunks of size lag
            segments = len(ts) // lag
            if segments < 1:
                continue
                
            # Calculate R/S for each segment and average
            rs_values = []
            for i in range(segments):
                segment = ts[i*lag:(i+1)*lag]
                
                # Get cumulative deviation from mean
                mean = np.mean(segment)
                cum_dev = np.cumsum(segment - mean)
                
                # Calculate range and standard deviation
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(segment)
                
                if s > 1e-8:  # Avoid division by zero
                    rs_values.append(r/s)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) < 4:
            return np.nan
            
        # Fit log-log relationship
        try:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0]
        except:
            return np.nan
    
    # Method 2: Variance method
    def variance_method():
        # Create range of lag values
        lags = range(2, min(len(ts) // 4, 20))
        tau = []
        
        # Calculate variance of differences for each lag
        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            var = np.var(diff)
            if var > 1e-8:  # Avoid near-zero variance
                tau.append(var)
        
        if len(tau) < 4:
            return np.nan
            
        # Variance should increase as lag^(2H)
        try:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0] / 2  # Divide by 2 as we're using variance
        except:
            return np.nan
    
    # Calculate using both methods
    h_rs = rs_analysis()
    h_var = variance_method()
    
    # Combine results
    valid_results = [h for h in [h_rs, h_var] if not np.isnan(h)]
    if not valid_results:
        return np.nan
    
    # Average the valid results
    h_avg = np.mean(valid_results)
    
    # Constrain to reasonable range (0 to 1)
    # Values outside this range usually indicate calculation issues
    if h_avg < 0:
        return max(0, min(0.2, 0.2 + h_avg))  # Map negative values to 0-0.2 range
    elif h_avg > 1:
        return min(1, max(0.8, 0.8 + (h_avg - 1) * 0.2))  # Map values >1 to 0.8-1 range
    else:
        return h_avg

# --- Calculate Hurst confidence ---
def hurst_confidence(ts):
    """Calculate confidence score for Hurst estimation (0-100%)"""
    ts = np.array(ts)
    
    # Factors affecting confidence
    factors = []
    
    # 1. Length of time series
    len_factor = min(1.0, len(ts) / 50)
    factors.append(len_factor)
    
    # 2. Variance in the series
    var = np.var(ts)
    var_factor = min(1.0, var / 1e-4) if var > 0 else 0
    factors.append(var_factor)
    
    # 3. Trend consistency
    diff = np.diff(ts)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    consistency = 1.0 - min(1.0, sign_changes / (len(diff) - 1))
    factors.append(consistency)
    
    # Combine factors
    confidence = np.mean(factors) * 100
    return round(confidence)

# --- Enhanced Regime Classification with Intensity Levels ---
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

# --- Compute Rolling Hurst, Confidence and Regime ---
ohlc['Hurst'] = ohlc['close'].rolling(rolling_window).apply(improved_hurst)
ohlc['confidence'] = ohlc['close'].rolling(rolling_window).apply(hurst_confidence)
ohlc['regime_info'] = ohlc['Hurst'].apply(detailed_regime_classification)
ohlc['regime'] = ohlc['regime_info'].apply(lambda x: x[0])
ohlc['intensity'] = ohlc['regime_info'].apply(lambda x: x[1])
ohlc['regime_desc'] = ohlc['regime_info'].apply(lambda x: x[2])

# --- Display Actual Data Point Metrics ---
actual_bars = len(ohlc)
actual_points = len(ohlc.dropna(subset=['Hurst']))
st.success(f"✅ Actual Data: {actual_bars} bars collected, {actual_points} valid Hurst values calculated")

# --- Plots ---
st.subheader(f"Rolling Hurst for {selected_token} ({timeframe})")

if ohlc['Hurst'].dropna().empty:
    st.warning("⚠️ No valid Hurst values computed. Price may be too flat or data too sparse.")
else:
    # Create two plots - one for price, one for Hurst
    fig = go.Figure()
    fig2 = go.Figure()
    df_plot = ohlc.reset_index()
    
    # Define color map for regimes with intensity
    def get_regime_color(regime, intensity):
        """Generate colors based on regime and intensity level"""
        if regime == "MEAN-REVERT":
            # Red colors of increasing intensity
            colors = ['#FFCCCC', '#FF9999', '#FF6666', '#FF0000']
            return colors[intensity]
        elif regime == "TREND":
            # Green colors of increasing intensity
            colors = ['#CCFFCC', '#99FF99', '#66FF66', '#00FF00']
            return colors[intensity]
        elif regime == "NOISE":
            # Gray with slight color bias based on intensity
            if intensity == 0:  # Pure random
                return '#CCCCCC'
            elif intensity > 0:  # Trending bias
                return '#CCFFEE'
            else:  # Mean-reversion bias
                return '#FFCCEE'
        else:
            return '#EEEEEE'  # Unknown
    
    # Generate colors for each point in the dataframe
    colors = [get_regime_color(row['regime'], row['intensity']) 
            for _, row in df_plot.iterrows()]
    
    # Plot 1: Hurst exponent with confidence
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], 
        y=df_plot['Hurst'],
        mode='lines+markers', 
        name='Hurst',
        line=dict(color='blue'),
        marker=dict(
            size=6,
            color=colors,
            opacity=df_plot['confidence']/100
        ),
        text=df_plot['regime_desc'],  # Add hover text with regime description
        hovertemplate="<b>%{text}</b><br>Hurst: %{y:.3f}<br>Time: %{x}<extra></extra>"
    ))
    
    # Add regime bands with improved annotations
    fig.add_hrect(y0=0, y1=0.2, fillcolor="red", opacity=0.3, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.1, text="Strong Mean-Reversion", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(255,0,0,0.3)")
    
    fig.add_hrect(y0=0.2, y1=0.3, fillcolor="red", opacity=0.2, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.25, text="Moderate Mean-Reversion", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(255,0,0,0.2)")
    
    fig.add_hrect(y0=0.3, y1=0.4, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.35, text="Mild Mean-Reversion", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(255,0,0,0.1)")
    
    fig.add_hrect(y0=0.4, y1=0.6, fillcolor="gray", opacity=0.1, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.5, text="Random/Noise", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(128,128,128,0.1)")
    
    fig.add_hrect(y0=0.6, y1=0.7, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.65, text="Mild Trending", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(0,255,0,0.1)")
    
    fig.add_hrect(y0=0.7, y1=0.8, fillcolor="green", opacity=0.2, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.75, text="Moderate Trending", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(0,255,0,0.2)")
    
    fig.add_hrect(y0=0.8, y1=1, fillcolor="green", opacity=0.3, layer="below", line_width=0)
    fig.add_annotation(x=df_plot['timestamp'].iloc[0], y=0.9, text="Strong Trending", showarrow=False, 
                      font=dict(color="black", size=10), bgcolor="rgba(0,255,0,0.3)")
    
    # Add horizontal line at 0.5 to highlight the random walk threshold
    fig.add_shape(type="line", x0=df_plot['timestamp'].iloc[0], y0=0.5, 
                 x1=df_plot['timestamp'].iloc[-1], y1=0.5,
                 line=dict(color="black", width=1, dash="dash"))
    
    fig.update_layout(
        yaxis_title="Hurst Exponent",
        xaxis_title="Time",
        height=400,
        title=f"Rolling Hurst for {selected_token} ({timeframe})",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Legend for Hurst values
    fig.add_annotation(
        x=1.02,  # Position outside the right edge of the plot
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Hurst Value Meaning:</b><br>0.0-0.2: Strong Mean-Reversion<br>0.2-0.3: Moderate Mean-Reversion<br>0.3-0.4: Mild Mean-Reversion<br>0.4-0.6: Random/Noise<br>0.6-0.7: Mild Trending<br>0.7-0.8: Moderate Trending<br>0.8-1.0: Strong Trending",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Plot 2: Price chart with regime background
    # Create candlestick chart
    fig2.add_trace(go.Candlestick(
        x=df_plot['timestamp'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name="Price"
    ))
    
    # Add colored background for different regimes
    for i in range(1, len(df_plot)):
        if not pd.isna(df_plot['regime'].iloc[i-1]):
            regime = df_plot['regime'].iloc[i-1]
            intensity = df_plot['intensity'].iloc[i-1]
            color = get_regime_color(regime, intensity)
            fig2.add_vrect(
                x0=df_plot['timestamp'].iloc[i-1],
                x1=df_plot['timestamp'].iloc[i],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0
            )
    
    fig2.update_layout(
        yaxis_title="Price",
        xaxis_title="Time",
        height=400,
        title=f"Price Chart with Regime Overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Render plots
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Show Confidence Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Valid coverage
    valid_pct = round(ohlc['Hurst'].notna().mean() * 100, 1)
    col1.metric("✅ Valid Hurst Coverage", f"{valid_pct}%")
    
    # Average confidence
    avg_conf = round(ohlc['confidence'].mean(), 1)
    col2.metric("🎯 Avg Confidence", f"{avg_conf}%")
    
    # Current regime
    current_regime_desc = ohlc['regime_desc'].iloc[-1] if not ohlc.empty and not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown"
    col3.metric("🔍 Current Regime", current_regime_desc)
    
    # Window/Data Ratio
    window_data_ratio = round(rolling_window / actual_bars * 100, 1) if actual_bars > 0 else 0
    col4.metric("⚖️ Window/Data Ratio", f"{window_data_ratio}%", 
               delta="Good" if 10 <= window_data_ratio <= 50 else "Adjust",
               delta_color="normal" if 10 <= window_data_ratio <= 50 else "off")
    
    if window_data_ratio > 50:
        st.warning("⚠️ Rolling window is too large relative to data size. Consider reducing window size.")
    elif window_data_ratio < 10:
        st.warning("⚠️ Rolling window may be too small for reliable Hurst estimation. Consider increasing window size.")
    
    if valid_pct < 20:
        st.warning("⚠️ Low Hurst coverage — increase lookback or reduce rolling window.")

# --- Regime Distribution Analysis ---
if not ohlc['Hurst'].dropna().empty:
    with st.expander("Regime Distribution Analysis"):
        # Count occurrences of each regime description
        regime_counts = ohlc['regime_desc'].value_counts().reset_index()
        regime_counts.columns = ['Regime', 'Count']
        
        # Calculate percentages
        total = regime_counts['Count'].sum()
        regime_counts['Percentage'] = (regime_counts['Count'] / total * 100).round(1)
        
        # Create horizontal bar chart
        fig_dist = px.bar(
            regime_counts, 
            y='Regime', 
            x='Percentage',
            orientation='h',
            color='Regime',
            text='Percentage',
            labels={'Percentage': 'Percentage of Time (%)'},
            color_discrete_map={
                'Strong mean-reversion': '#FF0000',
                'Moderate mean-reversion': '#FF6666',
                'Mild mean-reversion': '#FF9999',
                'Slight mean-reversion bias': '#FFCCEE',
                'Pure random walk': '#CCCCCC',
                'Slight trending bias': '#CCFFEE', 
                'Mild trending': '#99FF99',
                'Moderate trending': '#66FF66',
                'Strong trending': '#00FF00',
                'Insufficient data': '#EEEEEE'
            }
        )
        
        fig_dist.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_dist.update_layout(title="Distribution of Regime Types")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Dominant regime
        if not regime_counts.empty:
            dominant_regime = regime_counts.iloc[0]['Regime']
            dominant_pct = regime_counts.iloc[0]['Percentage']
            st.info(f"📊 Dominant Regime: **{dominant_regime}** ({dominant_pct}% of the time)")
            
            # Trading suggestion based on dominant regime
            if "Strong mean-reversion" in dominant_regime or "Moderate mean-reversion" in dominant_regime:
                st.success("💡 Trading Suggestion: Consider mean-reversion strategies (buy low, sell high)")
            elif "Strong trending" in dominant_regime or "Moderate trending" in dominant_regime:
                st.success("💡 Trading Suggestion: Consider trend-following strategies (follow the trend direction)")
            elif "Pure random walk" in dominant_regime or "Noise" in dominant_regime:
                st.warning("💡 Trading Suggestion: Market appears mostly random. Consider reducing position size or using other indicators.")

# --- Table Display ---
st.markdown("### Regime Table (Most Recent 100 Bars)")
display_df = ohlc[['open', 'high', 'low', 'close', 'Hurst', 'confidence', 'regime_desc']].copy()
display_df['Hurst'] = display_df['Hurst'].round(3)
display_df['confidence'] = display_df['confidence'].round(1)
st.dataframe(display_df.sort_index(ascending=False).head(100))

# --- Explanation ---
with st.expander("Understanding Hurst Exponent and Dashboard"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Interpreting the Hurst Exponent
        
        The Hurst exponent measures the long-term memory of a time series:
        
        **Mean-Reverting (H < 0.4)**
        - **Strong (0.0-0.2)**: Very strong pullbacks to mean
        - **Moderate (0.2-0.3)**: Consistent mean-reversion
        - **Mild (0.3-0.4)**: Weak mean-reversion tendency
        
        **Random/Noisy (H 0.4-0.6)**
        - **Near 0.5**: Random walk, no correlation to past
        
        **Trending (H > 0.6)**
        - **Mild (0.6-0.7)**: Weak trend persistence
        - **Moderate (0.7-0.8)**: Steady trend persistence
        - **Strong (0.8-1.0)**: Very strong trend persistence
        """)
    
    with col2:
        st.markdown("""
        ### Dashboard Components
        
        **Settings:**
        - **Lookback**: How far back to collect price data
        - **Rolling Window**: How many bars to use for each Hurst calculation
        
        **Charts:**
        - **Hurst Chart**: Shows Hurst values over time with colored bands indicating regimes
        - **Price Chart**: Shows price with background colored by regime
        
        **Metrics:**
        - **Valid Coverage**: Percentage of time with valid Hurst values
        - **Avg Confidence**: Average reliability of calculations
        - **Window/Data Ratio**: Rolling window size relative to data size
        """)
    
    st.markdown("""
    ### Trading Applications
    
    - **Mean-Reverting Regimes**: Look for overbought/oversold conditions, use oscillators like RSI
    - **Trending Regimes**: Use trend-following indicators like moving averages, MACD
    - **Random/Noisy Regimes**: Reduce position sizes, look for clearer setups
    
    ### Optimal Parameters
    
    - **Rolling Window**: 20-30 bars for short-term, 50-100 for long-term regime detection
    - **Lookback**: At least 3-5x the rolling window size for sufficient data
    - **Window/Data Ratio**: Aim for 20-30% for balanced sensitivity/reliability
    """)