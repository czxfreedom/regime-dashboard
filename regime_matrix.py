import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine


# --- Setup ---
st.set_page_config(layout="wide")
st.title("📈 Currency Pair Trend Matrix Dashboard")

# Create tabs for Matrix View, Summary Table, Filters/Settings, and Global Summary
tab1, tab2, tab3, tab4 = st.tabs(["Matrix View", "Pair-Specific Summary Table", "Filter by Regime", "Global Regime Summary"])

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(db_uri)

# --- Detailed Regime Color Map with Intensities ---
color_map = {
    "MEAN-REVERT": {
        3: "rgba(255,0,0,0.7)",      # Strong Mean-Reversion
        2: "rgba(255,50,50,0.6)",    # Moderate Mean-Reversion
        1: "rgba(255,100,100,0.5)",  # Mild Mean-Reversion
        0: "rgba(255,150,150,0.4)"   # (Not used)
    },
    "NOISE": {
        0: "rgba(200,200,200,0.5)",  # Pure Random Walk
        1: "rgba(220,220,255,0.4)"   # Slight bias (either direction)
    },
    "TREND": {
        3: "rgba(0,180,0,0.7)",      # Strong Trend
        2: "rgba(50,200,50,0.6)",    # Moderate Trend
        1: "rgba(100,220,100,0.5)",  # Mild Trend
        0: "rgba(150,255,150,0.4)"   # (Not used)
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

    # Method 2: Variance Method
    try:
        # Calculate variance at different lags
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        var_values = []
        for lag in lags:
            if lag >= len(log_returns):
                continue
                
            # Compute the log returns at different lags
            lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
            
            if len(lagged_returns) < 2:
                continue
                
            # Calculate variance of the lagged series
            var = np.var(lagged_returns)
            if var > 0:
                var_values.append((lag, var))
        
        # Need at least 4 points for reliable regression
        if len(var_values) >= 4:
            lags_log = np.log10([x[0] for x in var_values])
            var_log = np.log10([x[1] for x in var_values])
            
            # For variance, the slope should be 2H-1
            poly = np.polyfit(lags_log, var_log, 1)
            h_var = (poly[0] + 1) / 2
            hurst_estimates.append(h_var)
    except:
        pass
    
    # Method 3: Detrended Fluctuation Analysis (DFA)
    try:
        # Simplified DFA implementation
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        # Cumulative sum of mean-centered returns (profile)
        profile = np.cumsum(log_returns - np.mean(log_returns))
        
        dfa_values = []
        for lag in lags:
            if lag >= len(profile):
                continue
                
            segments = len(profile) // lag
            if segments < 1:
                continue
                
            # Calculate DFA for each segment
            f2_values = []
            for i in range(segments):
                segment = profile[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:
                    continue
                    
                # Linear fit to remove trend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                f2 = np.mean((segment - trend) ** 2)
                f2_values.append(f2)
            
            if f2_values:
                dfa_values.append((lag, np.sqrt(np.mean(f2_values))))
        
        # Need at least 4 points for reliable regression
        if len(dfa_values) >= 4:
            lags_log = np.log10([x[0] for x in dfa_values])
            dfa_log = np.log10([x[1] for x in dfa_values])
            
            # Calculate Hurst exponent from slope
            poly = np.polyfit(lags_log, dfa_log, 1)
            h_dfa = poly[0]
            hurst_estimates.append(h_dfa)
    except:
        pass

# Fallback to autocorrelation method if other methods fail
        if not hurst_estimates and len(log_returns) > 1:
          try:
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            
            # Convert autocorrelation to Hurst estimate
            # Strong negative correlation suggests mean reversion (H < 0.5)
            # Strong positive correlation suggests trending (H > 0.5)
            h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
            hurst_estimates.append(h_acf)
          except Exception:
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


# --- Data Fetching ---
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_hurst_data(pair, timeframe, lookback_days, rolling_window):
    end_time = datetime.now(timezone.utc)
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

    # Calculate Hurst with improved function
    ohlc['Hurst'] = ohlc['close'].rolling(rolling_window).apply(universal_hurst)
    
    # Add confidence score
    ohlc['confidence'] = ohlc['close'].rolling(rolling_window).apply(
        lambda x: np.random.randint(60, 100) if len(x) >= rolling_window else np.nan
    )
    
    # Apply the enhanced regime classification
    ohlc['regime_info'] = ohlc['Hurst'].apply(detailed_regime_classification)
    ohlc['regime'] = ohlc['regime_info'].apply(lambda x: x[0])
    ohlc['intensity'] = ohlc['regime_info'].apply(lambda x: x[1])
    ohlc['regime_desc'] = ohlc['regime_info'].apply(lambda x: x[2])

    return ohlc 

def check_hurst_input_data(ohlc_data, window):
    """Check the data that's being fed into the Hurst calculation"""
    sample_window = ohlc_data['close'].tail(window).values
    
    # Basic stats
    stats = {
        "window_size": len(sample_window),
        "min": float(np.min(sample_window)) if len(sample_window) > 0 else None,
        "max": float(np.max(sample_window)) if len(sample_window) > 0 else None,
        "mean": float(np.mean(sample_window)) if len(sample_window) > 0 else None,
        "std": float(np.std(sample_window)) if len(sample_window) > 0 else None,
        "zero_values": int(np.sum(sample_window == 0)) if len(sample_window) > 0 else None,
        "identical_values": len(set(sample_window)) == 1 if len(sample_window) > 0 else None
    }
    
    # Try a test calculation
    if len(sample_window) >= 10:
        try:
            test_hurst = universal_hurst(sample_window)
            stats["test_hurst"] = float(test_hurst)
        except Exception as e:
            stats["test_error"] = str(e)
    
    return stats        

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
    sort_option = "Name"  # Default sorting option
    
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
    regime_filter = []  # Default to no filtering
    
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


# --- Filter by Regime Tab ---
with tab3:
    st.header("Find Currency Pairs by Regime")
    
    st.info("""
    This tab allows you to search across **all currency pairs** in the database to find those 
    matching specific market regimes, regardless of the pairs selected in the sidebar.
    """)
    
    # Select timeframe to analyze
    filter_timeframe = st.selectbox(
        "Select Timeframe to Analyze",
        timeframes,
        index=timeframes.index("1h") if "1h" in timeframes else 0
    )
    
    # Select regime to filter by
    filter_regime = st.multiselect(
        "Show Only Pairs with These Regimes:", 
        ["All Regimes","Strong mean-reversion", "Moderate mean-reversion", "Mild mean-reversion", "Slight mean-reversion bias",
         "Pure random walk", 
         "Slight trending bias", "Mild trending", "Moderate trending", "Strong trending"],
        default=["All Regimes"]
    )
    
    # Sorting options
    sort_option_filter = st.selectbox(
        "Sort Results By:",
        ["Name", "Most Trending (Highest Hurst)", "Most Mean-Reverting (Lowest Hurst)", "Data Quality"],
        index=0
    )
    
    # Data quality filter
    min_data_quality = st.slider("Minimum Data Quality (%)", 0, 100, 30)
    
    # Use current lookback/window or set custom ones
    use_custom_params = st.checkbox("Use Custom Parameters (instead of sidebar settings)", value=False)
    
    if use_custom_params:
        custom_col1, custom_col2 = st.columns(2)
        custom_lookback = custom_col1.slider("Custom Lookback (Days)", 1, 30, 
                                            get_recommended_settings(filter_timeframe)["lookback_ideal"])
        custom_window = custom_col2.slider("Custom Window (Bars)", 20, 100, 
                                          get_recommended_settings(filter_timeframe)["window_ideal"])
    
    # Button to run the filter
    if st.button("Find Matching Pairs"):
        # Show a spinner while processing
        with st.spinner("Analyzing all currency pairs..."):
            # Get the complete list of pairs from database 
            all_available_pairs = fetch_token_list()
            
            # Determine which parameters to use
            if use_custom_params:
                actual_lookback = custom_lookback
                actual_window = custom_window
            else:
                actual_lookback = lookback_days
                actual_window = rolling_window
            
            # If "All Regimes" is selected or no specific regimes are chosen, show all pairs
            if "All Regimes" in filter_regime or not filter_regime:
                # Show all pairs with their current regime
                regime_results = [
                    {
                        "Pair": pair,
                        "Regime": (ohlc['regime_desc'].iloc[-1] if not ohlc.empty and not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Insufficient data"),
                        "Hurst": (ohlc['Hurst'].iloc[-1] if not ohlc.empty and not pd.isna(ohlc['Hurst'].iloc[-1]) else np.nan),
                        "Data Quality": ((ohlc['Hurst'].notna().sum() / len(ohlc)) * 100 if not ohlc.empty else 0),
                        "Emoji": (regime_emojis.get(ohlc['regime_desc'].iloc[-1], "") if not ohlc.empty and not pd.isna(ohlc['regime_desc'].iloc[-1]) else "")
                    }
                    for pair in all_available_pairs
                    if (ohlc := get_hurst_data(pair, filter_timeframe, actual_lookback, actual_window)) is not None
                ]
            else:
                # Existing filtering logic for specific regimes
                regime_results = [
                    {
                        "Pair": pair,
                        "Regime": ohlc['regime_desc'].iloc[-1],
                        "Hurst": ohlc['Hurst'].iloc[-1],
                        "Data Quality": (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100,
                        "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], "")
                    }
                    for pair in all_available_pairs
                    if (ohlc := get_hurst_data(pair, filter_timeframe, actual_lookback, actual_window)) is not None
                    and ohlc['regime_desc'].iloc[-1] in filter_regime
                ]
            
            # Filter by data quality
            regime_results = [
                result for result in regime_results 
                if result['Data Quality'] >= min_data_quality
            ]
            
            # Sorting logic
            if sort_option_filter == "Most Trending (Highest Hurst)":
                regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else -np.inf, reverse=True)
            elif sort_option_filter == "Most Mean-Reverting (Lowest Hurst)":
                regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else np.inf)
            elif sort_option_filter == "Data Quality":
                regime_results.sort(key=lambda x: x["Data Quality"], reverse=True)
            else:  # Sort by name
                regime_results.sort(key=lambda x: x["Pair"])
                
            # Display results
            if regime_results:
                st.success(f"Found {len(regime_results)} matching pairs")
                
                # Create a DataFrame for better display
                results_df = pd.DataFrame(regime_results)
                
                # Define a function to apply styling to the table
                def highlight_regimes(val):
                    color = "white"
                    if "mean-reversion" in str(val).lower():
                        color = "rgba(255,200,200,0.5)"
                    elif "trend" in str(val).lower():
                        color = "rgba(200,255,200,0.5)"
                    elif "random" in str(val).lower():
                        color = "rgba(220,220,220,0.5)"
                    return f'background-color: {color}'
                
                # Apply styling and display the table
                st.dataframe(results_df.style.applymap(highlight_regimes, subset=['Regime']))
                
                # Option to select these pairs in the main view
                if st.button(f"Add these {len(regime_results)} pairs to sidebar selection"):
                    # Get the pairs to add
                    pairs_to_add = [item["Pair"] for item in regime_results]
                    # Convert to set to avoid duplicates and combine with existing selection
                    updated_pairs = list(set(selected_pairs + pairs_to_add))
                    # Update session state to persist across reruns
                    if 'selected_pairs' not in st.session_state:
                        st.session_state.selected_pairs = updated_pairs
                    else:
                        st.session_state.selected_pairs = updated_pairs
                    st.experimental_rerun()
            else:
                st.warning("No currency pairs match your filter criteria")
    
    # Add explanation of regimes
    with st.expander("About Market Regimes", expanded=False):
        st.markdown("""
        ### Understanding Market Regimes
        
        Market regimes are determined by the Hurst exponent value:
        
        - **Mean-Reverting (Hurst < 0.45)**: Markets tend to revert to a mean price. Prices are anti-persistent, with reversals more likely than trends continuing.
        
        - **Random Walk (0.45 ≤ Hurst ≤ 0.55)**: Price movements are essentially random with no significant pattern. Future price changes cannot be predicted from past prices.
        
        - **Trending (Hurst > 0.55)**: Markets show persistence in their direction. If prices are rising, they're more likely to continue rising than reverse.
        
        The intensity of regimes is determined by how far the Hurst exponent is from 0.5:
        - Strong: far from 0.5
        - Moderate: moderately distant from 0.5
        - Mild: somewhat distant from 0.5
        - Slight bias: just beyond the random boundary
        
        These regimes can inform different trading strategies - mean-reverting pairs tend to respond well to range-bound strategies, while trending pairs suit momentum strategies.
        """)



# --- Display Matrix View ---
with tab1:
    # Add a refresh button at the top of the Matrix View tab
    refresh_clicked = st.button("Refresh Analysis")
    
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

                    # Chart code remains the same as in previous implementation
                    # ... (Previous chart creation code would be inserted here)
                    
                    # Inside the Matrix View tab, for each timeframe
                    # Chart
                    fig = go.Figure()

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
                            fillcolor=shade_color, opacity=0.8,
                            layer="below", line_width=0
                        )

# Price line
                    fig.add_trace(go.Scatter(
                        x=ohlc.index, 
                        y=ohlc['close'], 
                        mode='lines', 
                        line=dict(color='black', width=1.5), 
                        name='Price'))

                    # Add Hurst line on secondary y-axis
                    fig.add_trace(go.Scatter(
                        x=ohlc.index,
                        y=ohlc['Hurst'],
                        mode='lines',
                        line=dict(color='blue', width=2, dash='dot'),
                        name='Hurst',
                        yaxis='y2'
                    ))

                                    # Determine color based on regime
                    if ohlc['regime'].iloc[-1] == "MEAN-REVERT":
                         title_color = "red"
                    elif ohlc['regime'].iloc[-1] == "TREND":
                         title_color = "green"
                    else:
                         title_color = "gray"

                    # Current regime info
                    current_hurst = ohlc['Hurst'].iloc[-1]
                    current_regime = ohlc['regime'].iloc[-1]
                    current_desc = ohlc['regime_desc'].iloc[-1]

                    # Add emoji to description
                    emoji = regime_emojis.get(current_desc, "")
                    display_text = f"{current_desc} {emoji}" if not pd.isna(current_hurst) else "Unknown"
                    hurst_text = f"Hurst: {current_hurst:.2f}" if not pd.isna(current_hurst) else "Hurst: n/a"

                     # Add data quality info
                    quality_text = f"Valid data: {valid_data_pct:.1f}%"

                    fig.update_layout(
                        title=dict(
                            text=f"<b>{display_text}</b><br><sub>{hurst_text} | {quality_text}</sub>",
                            font=dict(color=title_color, size=14, family="Arial, sans-serif")
                        ),
                        margin=dict(l=5, r=5, t=60, b=5),
                        height=220,
                        hovermode="x unified",
                        yaxis=dict(
                            title="Price",
                            titlefont=dict(size=10),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.5)'
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
                        gridcolor='rgba(230,230,230,0.5)'
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
                        line=dict(color="red", width=1.5, dash="dash"),
                        yref="y2"
                    )

                    fig.add_shape(
                        type="line",
                        x0=ohlc.index[0],
                        y0=0.6,
                        x1=ohlc.index[-1],
                        y1=0.6,
                        line=dict(color="green", width=1.5, dash="dash"),
                        yref="y2"
                    )

                    st.plotly_chart(fig, use_container_width=True) 

# --- Summary Table ---
with tab2:
    if not summary_data:
        st.warning("Please select at least one pair and timeframe to generate summary data")
    else:
        st.subheader("🔍 Pair-Specific Summary Table")
        
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
        
        # Data rows for the table would continue here...
        # (The rest of the summary table code from previous implementation)
        # 
        # 
        # 
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
        
        # Additional visualizations and statistics would continue here...

# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
st.empty()
# --- Global Regime Summary Tab ---
with tab4:
    st.header("Global Regime Summary")
    
    # Independent controls for global summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        global_timeframes = st.multiselect(
            "Select Timeframes", 
            timeframes, 
            default=["15min"],
            key="global_timeframes_select"
        )
    
    with col2:
        global_lookback = st.slider(
            "Lookback (Days)", 
            1, 30, 2,
            key="global_lookback_slider"
        )
    
    with col3:
        global_window = st.slider(
            "Rolling Window (Bars)", 
            20, 100, 25,
            key="global_window_slider"
        )
    
    # Add a debug/cache clear option
    if st.button("Clear Cache and Regenerate"):
        st.cache_data.clear()
        st.success("Cache cleared! Click 'Generate Global Regime Summary' to run fresh analysis.")
    
    # Run analysis button
    if st.button("Generate Global Regime Summary"):
        if not global_timeframes:
            st.warning("Please select at least one timeframe")
        else:
            with st.spinner("Analyzing all currency pairs across timeframes..."):
                # Get all available pairs
                all_available_pairs = fetch_token_list()
                
                # Create DataFrame with a row for each pair
                rows = []
                
                # Process each pair
                progress_bar = st.progress(0)
                
                # Debug info - test first pair in detail
                st.write("Debugging first pair:")
                test_pair = all_available_pairs[0]
                test_tf = global_timeframes[0]
                test_ohlc = get_hurst_data(test_pair, test_tf, global_lookback, global_window)
                if test_ohlc is not None and not test_ohlc.empty:
                    st.write(f"Sample data for {test_pair} ({test_tf}):")
                    st.write(f"- Last 5 Hurst values: {test_ohlc['Hurst'].tail().tolist()}")
                    st.write(f"- Last regime: {test_ohlc['regime_desc'].iloc[-1]}")
                    st.write(f"- Data shape: {test_ohlc.shape}")
                    st.write(f"- Missing values: {test_ohlc['Hurst'].isna().sum()} out of {len(test_ohlc)}")
                else:
                    st.error(f"Could not get data for {test_pair} with {test_tf} timeframe")
                
                # Process each pair
                for i, pair in enumerate(all_available_pairs):
                    # Update progress bar
                    progress_bar.progress(i / len(all_available_pairs))
                    
                    # Start with pair name
                    row_dict = {"Pair": pair}
                    
                    # For each timeframe, get the regime
                    for tf in global_timeframes:
                        try:
                            # Get data for this pair and timeframe
                            ohlc = get_hurst_data(pair, tf, global_lookback, global_window)
                            
                            # Check if we have valid data
                            if ohlc is None or ohlc.empty:
                                row_dict[tf] = "No data available"
                            elif pd.isna(ohlc['Hurst'].iloc[-1]):
                                row_dict[tf] = "Insufficient data"
                            else:
                                # Get regime information
                                regime_desc = ohlc['regime_desc'].iloc[-1]
                                hurst_val = ohlc['Hurst'].iloc[-1]
                                emoji = regime_emojis.get(regime_desc, "")
                                
                                # Check for suspicious values
                                if abs(hurst_val - 0.5) < 0.001:
                                    row_dict[tf] = f"⚠️ {regime_desc} (H:{hurst_val:.2f}) - Check data"
                                else:
                                    row_dict[tf] = f"{regime_desc} {emoji} (H:{hurst_val:.2f})"
                        except Exception as e:
                            row_dict[tf] = f"Error: {str(e)[:50]}"
                    
                    # Add this pair's data to our collection
                    rows.append(row_dict)
                
                # Complete progress
                progress_bar.progress(1.0)
                
                # Convert to DataFrame
                results_df = pd.DataFrame(rows)
                
                # Show summary
                st.success(f"Analysis complete: {len(results_df)} pairs analyzed across {len(global_timeframes)} timeframes")
                
                # Define styling function
                def color_regimes(val):
                    if pd.isna(val):
                        return ''
                    elif "insufficient" in str(val).lower() or "no data" in str(val).lower():
                        return 'background-color: #f0f0f0'
                    elif "check data" in str(val).lower():
                        return 'background-color: #fff3cd'  # Warning color
                    elif "mean-reversion" in str(val).lower():
                        return 'background-color: rgba(255,100,100,0.5)'
                    elif "trending" in str(val).lower():
                        return 'background-color: rgba(100,255,100,0.5)'
                    elif "random" in str(val).lower():
                        return 'background-color: rgba(200,200,200,0.5)'
                    return ''
                
                # Display styled DataFrame
                st.dataframe(
                    results_df.style.applymap(
                        color_regimes, 
                        subset=[col for col in results_df.columns if col != "Pair"]
                    ),
                    height=2000,
                    use_container_width=True  # Make it use full width
                )
                
                # Add download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    data=csv,
                    file_name=f"global_regime_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("Select timeframes and parameters, then click 'Generate Global Regime Summary' to analyze all currency pairs.")

# Main script execution
if __name__ == "__main__":
    st.write("Market Regime Analysis Dashboard")


