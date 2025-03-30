import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- Setup ---
st.set_page_config(layout="wide")
st.title("📈 Currency Pair Trend Matrix Dashboard")

# Create tabs for Matrix View, Summary Table, and Filters/Settings
tab1, tab2, tab3 = st.tabs(["Matrix View", "Summary Table", "Filters & Settings"])

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(db_uri)

# --- Detailed Regime Color Map with Intensities ---
color_map = {
    "MEAN-REVERT": {
        0: "rgba(255,0,0,0.7)",      # Strong Mean-Reversion
        1: "rgba(255,50,50,0.6)",    # Moderate Mean-Reversion
        2: "rgba(255,100,100,0.5)",  # Mild Mean-Reversion
        3: "rgba(255,150,150,0.4)"   # Slight Mean-Reversion bias
    },
    "NOISE": {
        0: "rgba(200,200,200,0.5)",  # Pure Random Walk
        1: "rgba(220,220,255,0.4)"   # Slight bias
    },
    "TREND": {
        0: "rgba(0,180,0,0.7)",      # Strong Trend
        1: "rgba(50,200,50,0.6)",    # Moderate Trend
        2: "rgba(100,220,100,0.5)",  # Mild Trend
        3: "rgba(150,255,150,0.4)"   # Slight Trending bias
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

# --- Hurst & Regime Logic ---
def universal_hurst(ts):
    """
    A universal Hurst exponent calculation that works for any asset class.
    
    Args:
        ts: Time series of prices (numpy array or list)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    # Convert to numpy array and ensure floating point
    try:
        ts = np.array(ts, dtype=float)
    except:
        return np.nan  # Return NaN if conversion fails
        
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
    
    # Use multiple methods and average for robustness
    hurst_estimates = []
    
    # Method 1: Rescaled Range (R/S) Analysis
    try:
        # Create range of lags - adaptive based on data length
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        rs_values = []
        for lag in lags:
            # Reshape returns into segments
            segments = len(log_returns) // lag
            if segments < 1:
                continue
                
            # Calculate R/S for each segment
            rs_by_segment = []
            for i in range(segments):
                segment = log_returns[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:  # Skip if segment is too short
                    continue
                    
                # Get mean and standard deviation
                mean_return = np.mean(segment)
                std_return = np.std(segment)
                
                if std_return == 0:  # Skip if no variation
                    continue
                    
                # Calculate cumulative deviation from mean
                cumdev = np.cumsum(segment - mean_return)
                
                # Calculate R/S statistic
                r = np.max(cumdev) - np.min(cumdev)
                s = std_return
                
                rs_by_segment.append(r / s)
            
            if rs_by_segment:
                rs_values.append((lag, np.mean(rs_by_segment)))
        
        # Need at least 4 points for reliable regression
        if len(rs_values) >= 4:
            lags_log = np.log10([x[0] for x in rs_values])
            rs_log = np.log10([x[1] for x in rs_values])
            
            # Calculate Hurst exponent from slope
            poly = np.polyfit(lags_log, rs_log, 1)
            h_rs = poly[0]
            hurst_estimates.append(h_rs)
    except:
        pass
    
    # If we have estimates, aggregate them and constrain to 0-1 range
    if hurst_estimates:
        # Remove any extreme outliers
        valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
        
        # If no valid estimates remain after filtering, use all estimates but constrain them
        if not valid_estimates and hurst_estimates:
            valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
        
        # If we have valid estimates, return their median (more robust than mean)
        if valid_estimates:
            return np.median(valid_estimates)
    
    # If all methods fail, return 0.5 (random walk assumption)
    return 0.5

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
        return ("MEAN-REVERT", 0, "Strong mean-reversion")
    
    # Moderate mean reversion
    elif hurst < 0.3:
        return ("MEAN-REVERT", 1, "Moderate mean-reversion")
    
    # Mild mean reversion
    elif hurst < 0.4:
        return ("MEAN-REVERT", 2, "Mild mean-reversion")
    
    # Noisy/Random zone
    elif hurst < 0.45:
        return ("MEAN-REVERT", 3, "Slight mean-reversion bias")
    elif hurst <= 0.55:
        return ("NOISE", 0, "Pure random walk")
    elif hurst < 0.6:
        return ("TREND", 3, "Slight trending bias")
    
    # Mild trend
    elif hurst < 0.7:
        return ("TREND", 2, "Mild trending")
    
    # Moderate trend
    elif hurst < 0.8:
        return ("TREND", 1, "Moderate trending")
    
    # Strong trend
    else:
        return ("TREND", 0, "Strong trending")
    
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

# --- Sidebar Parameters ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

# Keep only the most essential controls in the sidebar
all_pairs = fetch_token_list()
selected_pairs = st.sidebar.multiselect("Select Currency Pairs", all_pairs, default=all_pairs[:5])
timeframes = ["30s","15min", "30min", "1h", "4h", "6h"]
selected_timeframes = st.sidebar.multiselect("Select Timeframes", timeframes, default=["15min", "1h", "6h"])

# IMPORTANT: Define sliders in sidebar (essential settings)
col1, col2 = st.sidebar.columns(2)
lookback_days = col1.slider("Lookback (Days)", 1, 30, 14)  # Default to 14 for better results
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

# --- Troubleshooting Guide ---
with st.sidebar.expander("Troubleshooting 'Insufficient Data'", expanded=False):
    st.markdown("""
    ### Fixing "Insufficient Data" Issues
    
    If you're seeing many "Insufficient Data" messages:
    
    1. **Reduce Rolling Window**: Try a smaller rolling window (20-30 bars instead of 50+)
    2. **Increase Lookback**: Extend lookback period to provide more data points
    3. **Try Lower Timeframes**: Lower timeframes (15min, 30min) have more data points
    4. **Check Pair Liquidity**: Some pairs may have sparse or irregular data
    
    **Optimal Settings**:
    - For 15min timeframe: 2-3 day lookback with 20-30 bar window
    - For 1h timeframe: 5-7 day lookback with 20-30 bar window
    - For 6h timeframe: 14+ day lookback with 20-30 bar window
    
    **Technical Note**: Hurst calculation needs at least 10 valid price changes with sufficient variance.
    """)

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

# --- Move Filtering and Sorting to a dedicated tab ---
with tab3:
    st.header("Filtering & Sorting Settings")
    
    st.subheader("Filter by Regime")
    regime_filter = st.multiselect(
        "Show only currency pairs that exhibit these regimes in any timeframe:", 
        ["Strong mean-reversion", "Moderate mean-reversion", "Mild mean-reversion", "Slight mean-reversion bias",
         "Pure random walk", 
         "Slight trending bias", "Mild trending", "Moderate trending", "Strong trending"],
        default=[]
    )
    
    st.subheader("Sort Pairs By")
    sort_option = st.selectbox(
        "Order currency pairs according to:",
        ["Name", "Most Trending", "Most Mean-Reverting", "Regime Consistency"]
    )
    
    # Add a button to apply filters
    apply_button = st.button("Apply Filters and Sorting")
    
    # Provide some helpful guidance
    st.info("""
    **How to use filters:**
    
    1. Select one or more regimes to filter pairs that exhibit those characteristics
    2. Choose a sorting method to organize the results
    3. Click 'Apply Filters and Sorting' to update the Matrix and Summary views
    
    Filtering shows only currency pairs that have at least one timeframe matching any of your selected regimes.
    """)

# --- Data Fetching ---
@st.cache_data(ttl=300)  # Cache data for 5 minutes
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

# --- Collect all data for summary table ---
@st.cache_data
def generate_summary_data():
    summary_data = []
    
    for pair in selected_pairs:
        pair_data = {"Pair": pair}
        
        for tf in selected_timeframes:
            ohlc = get_hurst_data(pair, tf, lookback_days, rolling_window)
            
            if ohlc is None or ohlc.empty or pd.isna(ohlc['Hurst'].iloc[-1]):
                pair_data[tf] = {"Hurst": np.nan, "Regime": "UNKNOWN", "Description": "Insufficient data"}
            else:
                pair_data[tf] = {
                    "Hurst": ohlc['Hurst'].iloc[-1],
                    "Regime": ohlc['regime'].iloc[-1],
                    "Description": ohlc['regime_desc'].iloc[-1],
                    "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], ""),
                    "Valid_Pct": (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100
                }
        
        summary_data.append(pair_data)
    
    # Apply sorting
    if sort_option == "Most Trending":
        # Calculate average Hurst and sort descending
        for item in summary_data:
            item["avg_hurst"] = np.mean([item[tf]["Hurst"] for tf in selected_timeframes if tf in item and "Hurst" in item[tf] and not pd.isna(item[tf]["Hurst"])])
        summary_data.sort(key=lambda x: x.get("avg_hurst", 0), reverse=True)
    elif sort_option == "Most Mean-Reverting":
        # Calculate average Hurst and sort ascending (lower Hurst = more mean-reverting)
        for item in summary_data:
            item["avg_hurst"] = np.mean([item[tf]["Hurst"] for tf in selected_timeframes if tf in item and "Hurst" in item[tf] and not pd.isna(item[tf]["Hurst"])])
        summary_data.sort(key=lambda x: x.get("avg_hurst", 1))
    elif sort_option == "Regime Consistency":
        # Sort by how consistent regimes are across timeframes
        for item in summary_data:
            regimes = [item[tf]["Regime"] for tf in selected_timeframes if tf in item and "Regime" in item[tf]]
            item["consistency"] = len(set(regimes)) if regimes else 0
        summary_data.sort(key=lambda x: x.get("consistency", 3))
    
    # Apply filtering
    if regime_filter:
        filtered_data = []
        for item in summary_data:
            # Check if any timeframe matches the filter
            match = False
            for tf in selected_timeframes:
                if tf in item and "Description" in item[tf] and item[tf]["Description"] in regime_filter:
                    match = True
                    break
            if match:
                filtered_data.append(item)
        summary_data = filtered_data
    
    return summary_data

# Get summary data
if selected_pairs and selected_timeframes:
    summary_data = generate_summary_data()
else:
    summary_data = []

# --- Display Matrix View ---
with tab1:
    if not selected_pairs or not selected_timeframes:
        st.warning("Please select at least one pair and timeframe")
    else:
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
            st.markdown(f"### 📌 {pair}")
            cols = st.columns(len(selected_timeframes))

            for i, tf in enumerate(selected_timeframes):
                with cols[i]:
                    st.markdown(f"**{tf}**")
                    ohlc = get_hurst_data(pair, tf, lookback_days, rolling_window)

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

                    # Chart
                    fig = go.Figure()
                    
                    # Price line
                    fig.add_trace(go.Scatter(
                        x=ohlc.index, 
                        y=ohlc['close'], 
                        mode='lines', 
                        line=dict(color='black', width=1.5), 
                        name='Price'))

                    # Background regime color with improved visualization
                    for j in range(1, len(ohlc)):
                        if pd.isna(ohlc['regime'].iloc[j-1]) or pd.isna(ohlc['intensity'].iloc[j-1]):
                            continue
                            
                        r = ohlc['regime'].iloc[j-1]
                        intensity = ohlc['intensity'].iloc[j-1]
                        
                        if r in color_map and intensity in color_map[r]:
                            shade_color = color_map[r][intensity]
                        else:
                            shade_color = "rgba(200,200,200,0.3)"

                        fig.add_vrect(
                            x0=ohlc.index[j-1], x1=ohlc.index[j],
                            fillcolor=shade_color, opacity=0.7,
                            layer="below", line_width=0
                        )

                    # Add Hurst line on secondary y-axis
                    fig.add_trace(go.Scatter(
                        x=ohlc.index,
                        y=ohlc['Hurst'],
                        mode='lines',
                        line=dict(color='blue', width=1, dash='dot'),
                        name='Hurst',
                        yaxis='y2'
                    ))
                    
                    # Current regime info
                    current_hurst = ohlc['Hurst'].iloc[-1]
                    current_regime = ohlc['regime'].iloc[-1]
                    current_desc = ohlc['regime_desc'].iloc[-1]
                    
                    # Determine color based on regime
                    if current_regime == "MEAN-REVERT":
                        title_color = "red"
                    elif current_regime == "TREND":
                        title_color = "green"
                    else:
                        title_color = "gray"
                    
                    # Add emoji to description
                    emoji = regime_emojis.get(current_desc, "")
                    display_text = f"{current_desc} {emoji}" if not pd.isna(current_hurst) else "Unknown"
                    hurst_text = f"Hurst: {current_hurst:.2f}" if not pd.isna(current_hurst) else "Hurst: n/a"
                    
                    # Add data quality info
                    quality_text = f"Valid data: {valid_data_pct:.1f}%"
                    
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{display_text}</b><br><sub>{hurst_text} | {quality_text}</sub>",
                            font=dict(color=title_color, size=14)
                        ),
                        margin=dict(l=5, r=5, t=60, b=5),
                        height=220,
                        hovermode="x unified",
                        yaxis=dict(
                            title="Price",
                            titlefont=dict(size=10),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.3)'
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
                            gridcolor='rgba(230,230,230,0.3)'
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
                        line=dict(color="red", width=1, dash="dash"),
                        yref="y2"
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=ohlc.index[0],
                        y0=0.6,
                        x1=ohlc.index[-1],
                        y1=0.6,
                        line=dict(color="green", width=1, dash="dash"),
                        yref="y2"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# --- Summary Table ---
with tab2:
    if not summary_data:
        st.warning("Please select at least one pair and timeframe to generate summary data")
    else:
        st.subheader("🔍 Market Regime Summary Table")
        
        # Recommended settings based on current data
        st.info("""
        ### Data Quality & Recommended Settings
        
        - **Low timeframes (15min)**: 2-3 day lookback, 20-30 bar window
        - **Medium timeframes (1h)**: 5-7 day lookback, 20-30 bar window
        - **High timeframes (6h)**: 14+ day lookback, 20-30 bar window
        
        Your current settings: **{} days lookback** with **{} bar window**
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
                    if regime_data["Regime"] == "MEAN-REVERT":
                        bg_color = "rgba(255,200,200,0.5)"
                    elif regime_data["Regime"] == "TREND":
                        bg_color = "rgba(200,255,200,0.5)"
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
        
        # Add dashboard statistics
        st.subheader("Dashboard Statistics")
        
        # Count regimes
        regime_counts = {}
        valid_count = 0
        invalid_count = 0
        
        for item in summary_data:
            for tf in selected_timeframes:
                if tf in item and "Description" in item[tf]:
                    desc = item[tf]["Description"]
                    regime_counts[desc] = regime_counts.get(desc, 0) + 1
                    if desc == "Insufficient data":
                        invalid_count += 1
                    else:
                        valid_count += 1
        
        # Display validity stats
        total = valid_count + invalid_count
        if total > 0:
            valid_pct = (valid_count / total) * 100
            
            if valid_pct < 50:
                st.warning(f"⚠️ Low data quality: Only {valid_pct:.1f}% of data points have valid regimes. Try adjusting parameters.")
                
                # Suggest improvements
                if rolling_window > 40:
                    st.info("📌 Suggestion: Try reducing your rolling window to 20-30 bars")
                
                # Suggest different timeframe/lookback combinations
                if "6h" in selected_timeframes and lookback_days < 10:
                    st.info("📌 Suggestion: For 6h timeframe, increase lookback to at least 14 days")
                elif "1h" in selected_timeframes and lookback_days < 5:
                    st.info("📌 Suggestion: For 1h timeframe, increase lookback to 5-7 days")
            else:
                st.success(f"✅ Good data quality: {valid_pct:.1f}% of data points have valid regimes")
        
        # Add heatmap visualization of the current regimes
        st.subheader("Regime Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = []
        
        for item in summary_data:
            for tf in selected_timeframes:
                if tf in item and "Hurst" in item[tf] and not pd.isna(item[tf]["Hurst"]):
                    heatmap_data.append({
                        "Pair": item["Pair"],
                        "Timeframe": tf,
                        "Hurst": item[tf]["Hurst"],
                        "Regime": item[tf]["Description"],
                        "Valid_Pct": item[tf].get("Valid_Pct", 0)
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Create a pivot table for the heatmap
            heatmap_pivot = heatmap_df.pivot(index="Pair", columns="Timeframe", values="Hurst")
            
            # Create heatmap using Plotly
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Timeframe", y="Pair", color="Hurst Value"),
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                color_continuous_scale="RdBu_r",  # Red for mean-reversion, Blue for trending
                range_color=[0, 1],
                aspect="auto",
                height=max(300, len(selected_pairs) * 30)
            )
            
            # Add text annotations with regime descriptions
            for pair_idx, pair in enumerate(heatmap_pivot.index):
                for tf_idx, tf in enumerate(heatmap_pivot.columns):
                    regime_desc = ""
                    for item in summary_data:
                        if item["Pair"] == pair and tf in item and "Description" in item[tf]:
                            regime_desc = item[tf]["Description"]
                            emoji = item[tf].get("Emoji", "")
                            break
                    
                    hurst_val = heatmap_pivot.iloc[pair_idx, tf_idx]
                    if not pd.isna(hurst_val):
                        fig.add_annotation(
                            x=tf,
                            y=pair,
                            text=f"{regime_desc} {emoji}<br>H={hurst_val:.2f}",
                            showarrow=False,
                            font=dict(
                                color="black" if 0.3 < hurst_val < 0.7 else "white",
                                size=10
                            )
                        )
            
            fig.update_layout(
                title="Hurst Exponent Heatmap Across Pairs and Timeframes",
                margin=dict(l=5, r=5, t=40, b=5),
                coloraxis_colorbar=dict(
                    title="Hurst Value",
                    tickvals=[0, 0.4, 0.5, 0.6, 1],
                    ticktext=["0 (Strong Mean-Rev)", "0.4", "0.5 (Random)", "0.6", "1 (Strong Trend)"]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional visualization - Regime Distribution
            st.subheader("Regime Distribution")
            regime_counts = heatmap_df['Regime'].value_counts().reset_index()
            regime_counts.columns = ['Regime', 'Count']
            
            # Create ordered categories for proper sorting
            regime_order = [
                "Strong mean-reversion", "Moderate mean-reversion", "Mild mean-reversion", "Slight mean-reversion bias",
                "Pure random walk",
                "Slight trending bias", "Mild trending", "Moderate trending", "Strong trending"
            ]
            
            # Filter to keep only regimes in our order list
            regime_counts = regime_counts[regime_counts['Regime'].isin(regime_order)]
            
            # Create categorical type with our custom order
            regime_counts['Regime'] = pd.Categorical(
                regime_counts['Regime'],
                categories=regime_order,
                ordered=True
            )
            
            # Sort by our custom order
            regime_counts = regime_counts.sort_values('Regime')
            
            # Create color map for bars
            colors = []
            for regime in regime_counts['Regime']:
                if "mean-reversion" in regime.lower():
                    if "strong" in regime.lower():
                        colors.append("rgba(255,0,0,0.8)")
                    elif "moderate" in regime.lower():
                        colors.append("rgba(255,50,50,0.7)")
                    elif "mild" in regime.lower():
                        colors.append("rgba(255,100,100,0.6)")
                    else:
                        colors.append("rgba(255,150,150,0.5)")
                elif "random" in regime.lower():
                    colors.append("rgba(180,180,180,0.7)")
                elif "trending" in regime.lower():
                    if "strong" in regime.lower():
                        colors.append("rgba(0,180,0,0.8)")
                    elif "moderate" in regime.lower():
                        colors.append("rgba(50,200,50,0.7)")
                    elif "mild" in regime.lower():
                        colors.append("rgba(100,220,100,0.6)")
                    else:
                        colors.append("rgba(150,255,150,0.5)")
                else:
                    colors.append("rgba(200,200,200,0.5)")
            
            fig = px.bar(
                regime_counts,
                y='Regime',
                x='Count',
                orientation='h',
                labels={'Count': 'Number of Pairs/Timeframes', 'Regime': ''},
                color_discrete_sequence=colors,
                text='Count'
            )
            
            fig.update_traces(textposition='outside')
            
            fig.update_layout(
                title="Distribution of Market Regimes",
                height=400,
                showlegend=False,
                xaxis=dict(title="Count"),
                yaxis=dict(title=""),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data quality assessment
            st.subheader("Data Quality Assessment")
            
            # Get valid percentage distribution
            valid_pcts = [item[tf].get("Valid_Pct", 0) for item in summary_data for tf in selected_timeframes if tf in item and "Valid_Pct" in item[tf]]
            
            if valid_pcts:
                # Create a histogram of valid percentages
                fig = px.histogram(
                    valid_pcts, 
                    nbins=10,
                    labels={'value': 'Valid Data Percentage', 'count': 'Number of Pair/Timeframe Combinations'},
                    title="Distribution of Valid Data Percentages",
                    color_discrete_sequence=['rgba(0,100,200,0.6)']
                )
                
                # Add a vertical line for the mean
                mean_valid = np.mean(valid_pcts)
                fig.add_vline(
                    x=mean_valid,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_valid:.1f}%",
                    annotation_position="top right"
                )
                
                # Add reference lines for quality thresholds
                fig.add_vline(
                    x=30,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Poor (<30%)",
                    annotation_position="bottom right"
                )
                
                fig.add_vline(
                    x=70,
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Good (>70%)",
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data quality summary
                low_quality = len([x for x in valid_pcts if x < 30])
                medium_quality = len([x for x in valid_pcts if 30 <= x < 70])
                high_quality = len([x for x in valid_pcts if x >= 70])
                total = len(valid_pcts)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Low Quality Data", f"{low_quality} ({low_quality/total*100:.1f}%)", delta=f"{low_quality} pairs/timeframes", delta_color="inverse")
                col2.metric("Medium Quality Data", f"{medium_quality} ({medium_quality/total*100:.1f}%)", delta=f"{medium_quality} pairs/timeframes")
                col3.metric("High Quality Data", f"{high_quality} ({high_quality/total*100:.1f}%)", delta=f"{high_quality} pairs/timeframes", delta_color="normal")
                
                if low_quality / total > 0.5:
                    st.warning("""
                    ### ⚠️ Data Quality Alert
                    
                    Over 50% of your pair/timeframe combinations have low quality data (under 30% valid values).
                    Consider adjusting your parameters with the suggested values in the troubleshooting guide.
                    """)
        else:
            st.info("Not enough data to generate heatmap")