import streamlit as st
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Trading Parameters Optimizer",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS styling
st.markdown("""
<style>
    .header-style {font-size:24px !important; font-weight: bold; padding: 10px 0;}
    .subheader-style {font-size:20px !important; font-weight: bold; padding: 5px 0;}
    .info-box {background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; border: 1px solid #e9ecef;}
    .dataframe th, .dataframe td {text-align: center !important; font-family: monospace;}
    .dataframe th:first-child, .dataframe td:first-child {text-align: left !important; font-family: inherit;}
    .positive-change {color: green; font-weight: bold;}
    .negative-change {color: red; font-weight: bold;}
    .high-range {background-color: rgba(255, 152, 0, 0.2);}
    .extreme-range {background-color: rgba(244, 67, 54, 0.2);}
    .success-message {background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #c3e6cb;}
    .warning-message {background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ffeeba;}
    .parameter-controls {background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0;}
    .rollbit-comparison {background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0; border: 1px solid #bbdefb;}
    .parameter-group {border-left: 4px solid #1976D2; padding-left: 10px; margin-bottom: 10px;}
    .rollbit-param {background-color: #ffecb3; font-weight: bold;}
    .surf-param {background-color: #c8e6c9; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Constants and Configuration
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture"]
exchanges_display = {
    "binanceFuture": "Binance", "gateFuture": "Gate", "hyperliquidFuture": "Hyperliquid"
}
singapore_timezone = pytz.timezone('Asia/Singapore')
lookback_days = 1
weekly_lookback_days = 7
scale_factor = 10000
scale_label = "× 10,000"

# Database Configuration
@st.cache_resource
def init_connection():
    try:
        db_config = st.secrets["database"]
        db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(db_uri)
    except Exception as e:
        st.sidebar.error(f"Error connecting to the database: {e}")
        return None

# Utility Functions
def is_major(token):
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    return any(major in token for major in majors)

def get_depth_level(token):
    return ("50K", "fee1") if is_major(token) else ("20K", "fee1")

def parse_leverage_config(leverage_config):
    if not leverage_config or leverage_config == '[]':
        return 100
    try:
        if isinstance(leverage_config, str):
            config = json.loads(leverage_config)
        else:
            config = leverage_config
        if isinstance(config, list) and len(config) > 0:
            max_lev = max([item.get('leverage', 0) for item in config])
            return max_lev if max_lev > 0 else 100
    except Exception as e:
        print(f"Error parsing leverage config: {e}")
        return 100

def format_percent(value):
    return f"{value*100:.2f}%"

def format_number(value):
    return f"{int(value):,}"

def format_float(value, decimals=4):
    return f"{value:.{decimals}f}"

def format_with_change_indicator(old_value, new_value, is_percent=False, decimals=2):
    if is_percent:
        formatted_old = format_percent(old_value)
        formatted_new = format_percent(new_value)
    elif decimals != 2:
        formatted_old = format_float(old_value, decimals)
        formatted_new = format_float(new_value, decimals)
    else:
        formatted_old = format_number(old_value)
        formatted_new = format_number(new_value)
    
    if old_value == 0:
        change_pct = 0
    else:
        change_pct = ((new_value - old_value) / old_value) * 100
    
    change_indicator = f"({change_pct:+.2f}%)"
    
    if new_value > old_value:
        return f"{formatted_new} <span class='positive-change'>{change_indicator}</span>"
    elif new_value < old_value:
        return f"{formatted_new} <span class='negative-change'>{change_indicator}</span>"
    else:
        return formatted_new

def format_spread_range(low, high, current, baseline):
    range_value = high - low
    if range_value > 0:
        position_pct = (current - low) / range_value * 100
    else:
        position_pct = 50
    
    css_class = ""
    if position_pct > 80 or position_pct < 20:
        css_class = "high-range"
    if current > high or current < low:
        css_class = "extreme-range"
    
    return f"<span class='{css_class}'>{low:.2f} - {high:.2f} [{range_value:.2f}]</span>"

def format_range_as_percent_of_current(low, high, current):
    range_value = high - low
    if current > 0:
        range_percent = (range_value / current) * 100
        if range_percent > 100:
            return f"<span style='color:#E53935;font-weight:bold'>{range_percent:.2f}%</span>"
        elif range_percent > 50:
            return f"<span style='color:#FFA726;font-weight:bold'>{range_percent:.2f}%</span>"
        elif range_percent > 25:
            return f"<span style='color:#43A047;font-weight:bold'>{range_percent:.2f}%</span>"
        else:
            return f"<span style='color:#1E88E5;font-weight:bold'>{range_percent:.2f}%</span>"
    else:
        return "N/A"

# Data Fetching Functions
@st.cache_data(ttl=600)
def fetch_all_tokens():
    try:
        engine = init_connection()
        if not engine:
            return []
        query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs WHERE status = 1 ORDER BY pair_name"
        df = pd.read_sql(query, engine)
        return df['pair_name'].tolist() if not df.empty else []
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_daily_spread_averages(tokens):
    try:
        engine = init_connection()
        if not engine:
            return None
        
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        tokens_str = "', '".join(tokens)
        query = f"""
        SELECT 
            pair_name, source,
            AVG(fee1) as avg_fee1, AVG(fee2) as avg_fee2, 
            AVG(fee3) as avg_fee3, AVG(fee4) as avg_fee4,
            AVG(total_fee) as avg_total_fee
        FROM oracle_exchange_fee
        WHERE 
            pair_name IN ('{tokens_str}')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        GROUP BY pair_name, source
        ORDER BY pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
            
    except Exception as e:
        st.error(f"Error fetching daily spread averages: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_weekly_spread_ranges(tokens):
    try:
        engine = init_connection()
        if not engine:
            return None
            
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=weekly_lookback_days)
        
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        tokens_str = "', '".join(tokens)
        query = f"""
        WITH spread_data AS (
            SELECT 
                pair_name, source, time_group, fee1, fee2, fee3, fee4, total_fee
            FROM oracle_exchange_fee
            WHERE 
                pair_name IN ('{tokens_str}')
                AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        ),
        aggregated_data AS (
            SELECT 
                pair_name, time_group, AVG(fee1) as avg_spread
            FROM spread_data
            GROUP BY pair_name, time_group
        )
        SELECT 
            pair_name,
            MIN(avg_spread) as weekly_low,
            MAX(avg_spread) as weekly_high,
            AVG(avg_spread) as weekly_avg
        FROM aggregated_data
        GROUP BY pair_name
        ORDER BY pair_name
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
            
    except Exception as e:
        st.error(f"Error fetching weekly spread ranges: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_current_parameters():
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT
            pair_name, buffer_rate, position_multiplier,
            rate_multiplier, rate_exponent,
            max_leverage, leverage_config, status
        FROM public.trade_pool_pairs
        WHERE status = 1
        ORDER BY pair_name
        """
        
        df = pd.read_sql(query, engine)
        if 'max_leverage' not in df.columns or df['max_leverage'].isna().all():
            df['max_leverage'] = df['leverage_config'].apply(parse_leverage_config)
        
        # Add default values for new parameters if they don't exist
        if 'rate_multiplier' not in df.columns:
            df['rate_multiplier'] = 6.0  # Default rate multiplier
        if 'rate_exponent' not in df.columns:
            df['rate_exponent'] = 2.0  # Default rate exponent
            
        return df
    except Exception as e:
        st.error(f"Error fetching current parameters: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_rollbit_parameters():
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT 
            pair_name,
            bust_buffer as buffer_rate,
            position_multiplier,
            rate_multiplier,
            rate_exponent,
            created_at
        FROM 
            rollbit_pair_config
        WHERE 
            created_at = (SELECT max(created_at) FROM rollbit_pair_config)
        ORDER BY 
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

def get_non_surf_average_spread(spread_data, pair_name, fee_level='avg_fee1'):
    pair_data = spread_data[spread_data['pair_name'] == pair_name]
    valid_sources = [source for source in exchanges if source in pair_data['source'].values]
    filtered_data = pair_data[pair_data['source'].isin(valid_sources)]
    
    if not filtered_data.empty and fee_level in filtered_data.columns:
        fee_values = filtered_data[fee_level]
        avg_value = fee_values.mean()
        return avg_value
    return None

def calculate_recommended_params(current_params, current_spread, baseline_spread, 
                                sensitivities, significant_change_threshold):
    """Calculate recommended parameter values based on spread change ratio"""
    
    if current_spread is None or baseline_spread is None or baseline_spread <= 0:
        return current_params
    
    # Get current parameter values
    current_buffer_rate = current_params.get('buffer_rate', 0.01)
    current_position_multiplier = current_params.get('position_multiplier', 1000000)
    current_rate_multiplier = current_params.get('rate_multiplier', 6.0)
    current_rate_exponent = current_params.get('rate_exponent', 2.0)
    max_leverage = current_params.get('max_leverage', 100)
    
    # Get sensitivity parameters
    buffer_sensitivity = sensitivities.get('buffer_sensitivity', 0.5)
    position_sensitivity = sensitivities.get('position_sensitivity', 0.5)
    rate_multiplier_sensitivity = sensitivities.get('rate_multiplier_sensitivity', 0.5)
    rate_exponent_sensitivity = sensitivities.get('rate_exponent_sensitivity', 0.5)
    
    # Calculate spread change ratio
    spread_change_ratio = current_spread / baseline_spread
    
    # Check if the change is significant
    if abs(spread_change_ratio - 1.0) < significant_change_threshold:
        return current_params
    
    # Recommended values - FIXED to make the relationship correct:
    # 1. If spreads increase, buffer rates should increase (direct relationship)
    recommended_buffer_rate = current_buffer_rate * (spread_change_ratio ** buffer_sensitivity)
    
    # 2. If spreads increase, position multiplier should decrease (inverse relationship)
    recommended_position_multiplier = current_position_multiplier / (spread_change_ratio ** position_sensitivity)
    
    # 3. If spreads increase, rate multiplier should decrease (inverse relationship)
    recommended_rate_multiplier = current_rate_multiplier / (spread_change_ratio ** rate_multiplier_sensitivity)
    
    # 4. If spreads increase, rate exponent should increase (direct relationship)
    recommended_rate_exponent = current_rate_exponent * (spread_change_ratio ** rate_exponent_sensitivity)
    
    # Apply bounds to keep values in reasonable ranges
    max_buffer_rate = 0.9 / max_leverage if max_leverage > 0 else 0.009
    recommended_buffer_rate = max(0.001, min(max_buffer_rate, recommended_buffer_rate))
    recommended_position_multiplier = max(100000, min(10000000, recommended_position_multiplier))
    recommended_rate_multiplier = max(1.0, min(10.0, recommended_rate_multiplier))
    recommended_rate_exponent = max(1.0, min(5.0, recommended_rate_exponent))
    
    return {
        'buffer_rate': recommended_buffer_rate,
        'position_multiplier': recommended_position_multiplier,
        'rate_multiplier': recommended_rate_multiplier,
        'rate_exponent': recommended_rate_exponent
    }

def create_baseline_spreads(spread_data):
    try:
        engine = init_connection()
        if not engine:
            return "Database connection failed"
            
        create_table_query = """
        CREATE TABLE IF NOT EXISTS spread_baselines (
            pair_name VARCHAR(50) PRIMARY KEY,
            baseline_spread FLOAT NOT NULL,
            updated_at TIMESTAMP NOT NULL
        );
        """
        with engine.connect() as connection:
            connection.execute(text(create_table_query))
            connection.commit()
        
        success_count = 0
        error_count = 0
        
        for pair_name in spread_data['pair_name'].unique():
            depth_label, fee_level = get_depth_level(pair_name)
            fee_column = f"avg_{fee_level}"
            
            non_surf_avg = get_non_surf_average_spread(spread_data, pair_name, fee_column)
            
            if non_surf_avg is not None:
                upsert_query = f"""
                INSERT INTO spread_baselines (pair_name, baseline_spread, updated_at)
                VALUES ('{pair_name}', {non_surf_avg}, NOW())
                ON CONFLICT (pair_name) DO UPDATE 
                SET baseline_spread = EXCLUDED.baseline_spread, updated_at = EXCLUDED.updated_at;
                """
                
                try:
                    with engine.connect() as connection:
                        connection.execute(text(upsert_query))
                        connection.commit()
                    success_count += 1
                except Exception as e:
                    print(f"Error saving baseline for {pair_name}: {e}")
                    error_count += 1
        
        return f"Successfully reset {success_count} baselines" + (f", {error_count} errors" if error_count > 0 else "")
    except Exception as e:
        return f"Error creating baseline spreads: {e}"

@st.cache_data(ttl=600)
def get_baseline_spreads():
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = "SELECT pair_name, baseline_spread, updated_at FROM spread_baselines"
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching baseline spreads: {e}")
        return None

def update_trading_parameters(pair_name, params):
    try:
        engine = init_connection()
        if not engine:
            return False
        
        # Extract parameter values
        buffer_rate = params.get('buffer_rate')
        position_multiplier = params.get('position_multiplier')
        rate_multiplier = params.get('rate_multiplier')
        rate_exponent = params.get('rate_exponent')
        
        # Construct the SET clause based on available parameters
        set_clauses = []
        if buffer_rate is not None:
            set_clauses.append(f"buffer_rate = {buffer_rate}")
        if position_multiplier is not None:
            set_clauses.append(f"position_multiplier = {position_multiplier}")
        if rate_multiplier is not None:
            set_clauses.append(f"rate_multiplier = {rate_multiplier}")
        if rate_exponent is not None:
            set_clauses.append(f"rate_exponent = {rate_exponent}")
        
        # Add updated_at timestamp
        set_clauses.append("updated_at = NOW()")
        
        # Join clauses with commas
        set_clause = ", ".join(set_clauses)
        
        # Construct the full query
        query = f"""
        UPDATE public.trade_pool_pairs
        SET {set_clause}
        WHERE pair_name = '{pair_name}';
        """
        
        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        return True
    except Exception as e:
        st.error(f"Error updating parameters for {pair_name}: {e}")
        return False

# Main Application
def main():
    st.markdown('<div class="header-style">Trading Parameters Optimizer</div>', unsafe_allow_html=True)
    
    # Connect to the database
    engine = init_connection()
    if not engine:
        st.error("Database connection failed. Please check connection parameters.")
        
        # Add manual connection form
        with st.sidebar:
            st.header("Database Connection")
            db_user = st.text_input("Database Username")
            db_password = st.text_input("Database Password", type="password")
            db_host = st.text_input("Database Host")
            db_port = st.text_input("Database Port", "5432")
            db_name = st.text_input("Database Name")
            
            if st.button("Connect to Database"):
                if all([db_user, db_password, db_host, db_port, db_name]):
                    # Store connection info in session state
                    st.session_state.db_params = {
                        'user': db_user, 'password': db_password,
                        'host': db_host, 'port': db_port, 'database': db_name
                    }
                    st.experimental_rerun()
                else:
                    st.error("Please fill in all database connection fields")
                    
        return
    
    # Success message for DB connection
    st.sidebar.success("Connected to database successfully")
    
    # Fetch all tokens and current parameters
    all_tokens = fetch_all_tokens()
    current_params_df = fetch_current_parameters()
    
    # Fetch Rollbit parameters for comparison
    rollbit_params_df = fetch_rollbit_parameters()
    
    if not all_tokens:
        st.error("No tokens found in the database. Please check data availability.")
        return
        
    if current_params_df is None or current_params_df.empty:
        st.error("Failed to fetch current parameters. Please check database access.")
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Always select all tokens
    selected_tokens = all_tokens
    
    # Add parameter sensitivity controls to sidebar
    st.sidebar.header("Parameter Settings")
    
    # Initialize session state for parameter values if not already set
    if 'buffer_sensitivity' not in st.session_state:
        st.session_state['buffer_sensitivity'] = 0.5
    if 'position_sensitivity' not in st.session_state:
        st.session_state['position_sensitivity'] = 0.5
    if 'rate_multiplier_sensitivity' not in st.session_state:
        st.session_state['rate_multiplier_sensitivity'] = 0.5
    if 'rate_exponent_sensitivity' not in st.session_state:
        st.session_state['rate_exponent_sensitivity'] = 0.5
    if 'change_threshold' not in st.session_state:
        st.session_state['change_threshold'] = 0.05
    
    # Dropdown for buffer rate sensitivity
    buffer_sensitivity = st.sidebar.selectbox(
        "Buffer Rate Sensitivity",
        options=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        format_func=lambda x: f"{x} - {'Low' if x < 0.5 else 'Medium' if x < 1.0 else 'High'} Sensitivity",
        index=2,  # Default to 0.5
        key="buffer_sensitivity"
    )
    
    # Dropdown for position multiplier sensitivity 
    position_sensitivity = st.sidebar.selectbox(
        "Position Multiplier Sensitivity",
        options=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        format_func=lambda x: f"{x} - {'Low' if x < 0.5 else 'Medium' if x < 1.0 else 'High'} Sensitivity",
        index=2,  # Default to 0.5
        key="position_sensitivity"
    )
    
    # Add new dropdowns for rate multiplier and rate exponent sensitivity
    rate_multiplier_sensitivity = st.sidebar.selectbox(
        "Rate Multiplier Sensitivity",
        options=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        format_func=lambda x: f"{x} - {'Low' if x < 0.5 else 'Medium' if x < 1.0 else 'High'} Sensitivity",
        index=2,  # Default to 0.5
        key="rate_multiplier_sensitivity"
    )
    
    rate_exponent_sensitivity = st.sidebar.selectbox(
        "Rate Exponent Sensitivity",
        options=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        format_func=lambda x: f"{x} - {'Low' if x < 0.5 else 'Medium' if x < 1.0 else 'High'} Sensitivity",
        index=2,  # Default to 0.5
        key="rate_exponent_sensitivity"
    )
    
    # Dropdown for significant change threshold
    change_threshold = st.sidebar.selectbox(
        "Significant Change Threshold",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        format_func=lambda x: f"{x*100}% - {'Very Low' if x < 0.01 else 'Low' if x < 0.05 else 'Medium' if x < 0.1 else 'High'} Threshold",
        index=4,  # Default to 0.05 (5%)
        key="change_threshold"
    )
    
    # Show current parameter settings
    st.sidebar.markdown(f"""
    **Current Settings:**
    - Buffer Sensitivity: **{buffer_sensitivity}**
    - Position Sensitivity: **{position_sensitivity}**
    - Rate Multiplier Sensitivity: **{rate_multiplier_sensitivity}**
    - Rate Exponent Sensitivity: **{rate_exponent_sensitivity}**
    - Change Threshold: **{change_threshold*100}%**
    """)
    
    # Collect all sensitivity parameters
    sensitivity_params = {
        'buffer_sensitivity': buffer_sensitivity,
        'position_sensitivity': position_sensitivity,
        'rate_multiplier_sensitivity': rate_multiplier_sensitivity,
        'rate_exponent_sensitivity': rate_exponent_sensitivity
    }
    
    # Add a "Apply Recommendations" button
    apply_button = st.sidebar.button("Apply All Recommendations", use_container_width=True, 
                                 help="Apply all recommended parameter values to the system")
    
    # Add a "Reset Baselines" button
    reset_button = st.sidebar.button("Reset All Baselines to Current Spreads", use_container_width=True,
                                 help="Reset all baseline spreads to current market conditions")
    
    # Add a refresh button
    refresh_button = st.sidebar.button("Refresh Data", use_container_width=True)
    
    # Get spread data
    daily_avg_data = fetch_daily_spread_averages(selected_tokens)
    
    # Get weekly spread range data
    weekly_spread_data = fetch_weekly_spread_ranges(selected_tokens)
    
    # Display the parameter settings in the main area
    st.markdown("""
    <div class="parameter-controls">
        <h3>Parameter Settings</h3>
        <p>These settings control how the system responds to market spread changes:</p>
        <ul>
            <li><strong>Buffer Sensitivity:</strong> How strongly buffer rates respond to spread changes (higher = more responsive)</li>
            <li><strong>Position Sensitivity:</strong> How strongly position multipliers respond to spread changes (higher = more responsive)</li>
            <li><strong>Rate Multiplier Sensitivity:</strong> How strongly rate multipliers respond to spread changes (higher = more responsive)</li>
            <li><strong>Rate Exponent Sensitivity:</strong> How strongly rate exponents respond to spread changes (higher = more responsive)</li>
            <li><strong>Change Threshold:</strong> Minimum spread change needed before parameters are adjusted (lower = more frequent updates)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have baseline data
    baseline_spreads_df = get_baseline_spreads()
    if baseline_spreads_df is None or baseline_spreads_df.empty:
        st.warning("No baseline spreads found. Please use 'Reset All Baselines' button to establish baselines.")
        if reset_button:
            if daily_avg_data is not None and not daily_avg_data.empty:
                result = create_baseline_spreads(daily_avg_data)
                st.success(result)
                # Clear cache and refresh baseline data
                st.cache_data.clear()
                baseline_spreads_df = get_baseline_spreads()
            else:
                st.error("No spread data available for baseline reset")
    elif reset_button:
        if daily_avg_data is not None and not daily_avg_data.empty:
            result = create_baseline_spreads(daily_avg_data)
            st.success(result)
            # Clear cache and refresh baseline data
            st.cache_data.clear()
            baseline_spreads_df = get_baseline_spreads()
        else:
            st.error("No spread data available for baseline reset")
    
    if refresh_button:
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Display explanation
    st.markdown("""
    <div class="info-box">
    <b>Trading Parameters Optimization</b><br>
    This tool helps optimize buffer rates, position multipliers, rate multipliers, and rate exponents based on market spread conditions.<br>
    <b>Parameter Relationships:</b>
    <ul>
        <li>When spreads <b>increase</b>: Buffer rates and rate exponents <b>increase</b>, position multipliers and rate multipliers <b>decrease</b></li>
        <li>When spreads <b>decrease</b>: Buffer rates and rate exponents <b>decrease</b>, position multipliers and rate multipliers <b>increase</b></li>
    </ul>
    <b>Key Features:</b>
    <ul>
        <li>Weekly range percentage shows how volatile each token's spread is relative to its current value</li>
        <li>Higher weekly range % indicates tokens that experience larger spread fluctuations</li>
        <li>Comparison with Rollbit parameters helps benchmark our settings against industry standards</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Process data and create recommendations
    if daily_avg_data is not None and not daily_avg_data.empty and baseline_spreads_df is not None and not baseline_spreads_df.empty:
        # Create recommendations dataframe
        recommendations_data = []
        
        # Debug mode toggle
        debug_mode = False
        if debug_mode:
            debug_container = st.expander("Debug Information", expanded=False)
        
        for _, param_row in current_params_df.iterrows():
            pair_name = param_row['pair_name']
            current_buffer_rate = param_row['buffer_rate']
            current_position_multiplier = param_row['position_multiplier']
            current_rate_multiplier = param_row['rate_multiplier']
            current_rate_exponent = param_row['rate_exponent']
            max_leverage = param_row['max_leverage']
            
            # Get current market spread
            depth_label, fee_level = get_depth_level(pair_name)
            fee_column = f"avg_{fee_level}"
            
            current_spread = get_non_surf_average_spread(daily_avg_data, pair_name, fee_column)
            
            # Get baseline spread
            baseline_row = baseline_spreads_df[baseline_spreads_df['pair_name'] == pair_name]
            baseline_spread = baseline_row['baseline_spread'].iloc[0] if not baseline_row.empty else None
            
            # Get weekly spread range data
            weekly_row = None
            if weekly_spread_data is not None and not weekly_spread_data.empty:
                weekly_row = weekly_spread_data[weekly_spread_data['pair_name'] == pair_name]
            
            weekly_low = weekly_row['weekly_low'].iloc[0] if weekly_row is not None and not weekly_row.empty else None
            weekly_high = weekly_row['weekly_high'].iloc[0] if weekly_row is not None and not weekly_row.empty else None
            weekly_avg = weekly_row['weekly_avg'].iloc[0] if weekly_row is not None and not weekly_row.empty else None
            
            # Get Rollbit parameters if available
            rollbit_row = None
            if rollbit_params_df is not None and not rollbit_params_df.empty:
                # Try exact match first
                rollbit_row = rollbit_params_df[rollbit_params_df['pair_name'] == pair_name]
                
                # If no match, try without the /USDT suffix (Rollbit may use different format)
                if rollbit_row.empty and '/' in pair_name:
                    base_token = pair_name.split('/')[0]
                    rollbit_row = rollbit_params_df[rollbit_params_df['pair_name'].str.contains(base_token, case=False)]
            
            rollbit_buffer_rate = rollbit_row['buffer_rate'].iloc[0] if rollbit_row is not None and not rollbit_row.empty else None
            rollbit_position_multiplier = rollbit_row['position_multiplier'].iloc[0] if rollbit_row is not None and not rollbit_row.empty else None
            rollbit_rate_multiplier = rollbit_row['rate_multiplier'].iloc[0] if rollbit_row is not None and not rollbit_row.empty else None
            rollbit_rate_exponent = rollbit_row['rate_exponent'].iloc[0] if rollbit_row is not None and not rollbit_row.empty else None
            
            # Calculate recommended parameters
            if current_spread is not None and baseline_spread is not None:
                # Calculate spread change ratio for display
                spread_change_ratio = current_spread / baseline_spread
                
                # Show debug information if enabled
                if debug_mode:
                    with debug_container:
                        st.write(f"**{pair_name}**: Current Spread: {current_spread:.8f}, Baseline: {baseline_spread:.8f}")
                        st.write(f"Spread Ratio: {spread_change_ratio:.4f}, Threshold: {abs(spread_change_ratio - 1.0):.4f} vs {change_threshold:.4f}")
                
                # Current parameters dictionary
                current_params = {
                    'buffer_rate': current_buffer_rate,
                    'position_multiplier': current_position_multiplier,
                    'rate_multiplier': current_rate_multiplier,
                    'rate_exponent': current_rate_exponent,
                    'max_leverage': max_leverage
                }
                
                # Calculate recommendations using the dynamic parameters
                rec_params = calculate_recommended_params(
                    current_params, 
                    current_spread,
                    baseline_spread,
                    sensitivity_params,
                    change_threshold
                )
                
                # Extract recommended values
                rec_buffer = rec_params['buffer_rate']
                rec_position = rec_params['position_multiplier']
                rec_rate_multiplier = rec_params['rate_multiplier']
                rec_rate_exponent = rec_params['rate_exponent']
                
                # Format the message
                if current_spread > baseline_spread:
                    spread_note = f"↑ {(current_spread/baseline_spread - 1)*100:.2f}%"
                elif current_spread < baseline_spread:
                    spread_note = f"↓ {(1 - current_spread/baseline_spread)*100:.2f}%"
                else:
                    spread_note = "No change"
                
                # Calculate if there is a significant change in any parameter
                buffer_change_pct = abs((rec_buffer - current_buffer_rate) / current_buffer_rate) * 100 if current_buffer_rate > 0 else 0
                position_change_pct = abs((rec_position - current_position_multiplier) / current_position_multiplier) * 100 if current_position_multiplier > 0 else 0
                rate_multiplier_change_pct = abs((rec_rate_multiplier - current_rate_multiplier) / current_rate_multiplier) * 100 if current_rate_multiplier > 0 else 0
                rate_exponent_change_pct = abs((rec_rate_exponent - current_rate_exponent) / current_rate_exponent) * 100 if current_rate_exponent > 0 else 0
                
                significant_change = (buffer_change_pct > 1 or 
                                     position_change_pct > 1 or 
                                     rate_multiplier_change_pct > 1 or 
                                     rate_exponent_change_pct > 1)
                
                recommendations_data.append({
                    'pair_name': pair_name,
                    'token_type': 'Major' if is_major(pair_name) else 'Altcoin',
                    'depth_tier': depth_label,
                    'max_leverage': max_leverage,
                    'current_spread': current_spread * scale_factor,
                    'baseline_spread': baseline_spread * scale_factor,
                    'weekly_low': weekly_low * scale_factor if weekly_low is not None else None,
                    'weekly_high': weekly_high * scale_factor if weekly_high is not None else None, 
                    'weekly_avg': weekly_avg * scale_factor if weekly_avg is not None else None,
                    'weekly_range': (weekly_high - weekly_low) * scale_factor if weekly_high is not None and weekly_low is not None else None,
                    'spread_change': spread_note,
                    'spread_change_ratio': spread_change_ratio,  # For sorting
                    
                    # Current SURF parameters
                    'current_buffer_rate': current_buffer_rate,
                    'current_position_multiplier': current_position_multiplier,
                    'current_rate_multiplier': current_rate_multiplier,
                    'current_rate_exponent': current_rate_exponent,
                    
                    # Recommended parameters
                    'recommended_buffer_rate': rec_buffer,
                    'recommended_position_multiplier': rec_position,
                    'recommended_rate_multiplier': rec_rate_multiplier,
                    'recommended_rate_exponent': rec_rate_exponent,
                    
                    # Rollbit parameters for comparison
                    'rollbit_buffer_rate': rollbit_buffer_rate,
                    'rollbit_position_multiplier': rollbit_position_multiplier,
                    'rollbit_rate_multiplier': rollbit_rate_multiplier,
                    'rollbit_rate_exponent': rollbit_rate_exponent,
                    
                    'significant_change': significant_change
                })
        
        if recommendations_data:
            # Create DataFrame
            rec_df = pd.DataFrame(recommendations_data)
            
            # Sort by token_type and then by pair_name
            rec_df = rec_df.sort_values(by=['token_type', 'pair_name'])
            
            # First show a summary
            total_pairs = len(rec_df)
            pairs_with_changes = len(rec_df[rec_df['significant_change']])
            
            if pairs_with_changes > 0:
                st.markdown(f"<div class='warning-message'>⚠️ {pairs_with_changes} out of {total_pairs} pairs have significant parameter changes recommended</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-message'>✅ All parameters are within optimal ranges</div>", unsafe_allow_html=True)
            
            # Create tabs for different views
            tabs = st.tabs(["All Pairs", "Changed Parameters Only", "Rollbit Comparison", "Major Tokens", "Altcoin Tokens"])
            
            with tabs[0]:  # All Pairs
                st.markdown("### All Trading Pairs")
                
                # Create display data
                display_data = []
                for _, row in rec_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    current_rate_mult_formatted = format_float(row['current_rate_multiplier'], 2)
                    rec_rate_mult_formatted = format_with_change_indicator(row['current_rate_multiplier'], row['recommended_rate_multiplier'], is_percent=False, decimals=2)
                    
                    current_rate_exp_formatted = format_float(row['current_rate_exponent'], 2)
                    rec_rate_exp_formatted = format_with_change_indicator(row['current_rate_exponent'], row['recommended_rate_exponent'], is_percent=False, decimals=2)
                    
                    # Format the weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                        # Calculate the weekly range as a percentage of current spread
                        weekly_range_pct = format_range_as_percent_of_current(
                            row['weekly_low'], 
                            row['weekly_high'], 
                            row['current_spread']
                        )
                    else:
                        weekly_range = "N/A"
                        weekly_range_pct = "N/A"
                    
                    display_data.append({
                        'Pair': row['pair_name'],
                        'Type': row['token_type'],
                        'Size': row['depth_tier'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Weekly Range %': weekly_range_pct,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted,
                        'Current Rate Mult.': current_rate_mult_formatted,
                        'Recommended Rate Mult.': rec_rate_mult_formatted,
                        'Current Rate Exp.': current_rate_exp_formatted,
                        'Recommended Rate Exp.': rec_rate_exp_formatted
                    })
                
                display_df = pd.DataFrame(display_data)
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Add legend for weekly range formatting
                st.markdown("""
                <div style="font-size: 0.85em; margin-top: 10px;">
                <strong>Weekly Range % Legend:</strong> Shows how large the weekly fluctuation range is compared to current spread
                <ul style="list-style-type: none; padding-left: 10px; margin-top: 5px;">
                    <li><span style="color:#1E88E5;font-weight:bold">Blue (< 25%)</span>: Very stable - small weekly fluctuations</li>
                    <li><span style="color:#43A047;font-weight:bold">Green (25-50%)</span>: Moderately stable - normal fluctuations</li>
                    <li><span style="color:#FFA726;font-weight:bold">Orange (50-100%)</span>: Volatile - large fluctuations</li>
                    <li><span style="color:#E53935;font-weight:bold">Red (> 100%)</span>: Highly volatile - extreme fluctuations</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tabs[1]:  # Changed Parameters Only
                st.markdown("### Pairs with Recommended Changes")
                
                # Filter for significant changes
                changed_df = rec_df[rec_df['significant_change']]
                
                if not changed_df.empty:
                    # Create display data for changed parameters
                    changed_display = []
                    for _, row in changed_df.iterrows():
                        current_buffer_formatted = format_percent(row['current_buffer_rate'])
                        rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                        
                        current_pos_formatted = format_number(row['current_position_multiplier'])
                        rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                        
                        current_rate_mult_formatted = format_float(row['current_rate_multiplier'], 2)
                        rec_rate_mult_formatted = format_with_change_indicator(row['current_rate_multiplier'], row['recommended_rate_multiplier'], is_percent=False, decimals=2)
                        
                        current_rate_exp_formatted = format_float(row['current_rate_exponent'], 2)
                        rec_rate_exp_formatted = format_with_change_indicator(row['current_rate_exponent'], row['recommended_rate_exponent'], is_percent=False, decimals=2)
                        
                        # Format the weekly range
                        if row['weekly_low'] is not None and row['weekly_high'] is not None:
                            weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                            # Calculate the weekly range as a percentage of current spread
                            weekly_range_pct = format_range_as_percent_of_current(
                                row['weekly_low'], 
                                row['weekly_high'], 
                                row['current_spread']
                            )
                        else:
                            weekly_range = "N/A"
                            weekly_range_pct = "N/A"
                        
                        changed_display.append({
                            'Pair': row['pair_name'],
                            'Type': row['token_type'],
                            'Size': row['depth_tier'],
                            'Market Spread': f"{row['current_spread']:.2f}",
                            'Baseline Spread': f"{row['baseline_spread']:.2f}",
                            'Weekly Range': weekly_range,
                            'Weekly Range %': weekly_range_pct,
                            'Spread Change': row['spread_change'],
                            'Current Buffer': current_buffer_formatted,
                            'Recommended Buffer': rec_buffer_formatted,
                            'Current Position Mult.': current_pos_formatted,
                            'Recommended Position Mult.': rec_pos_formatted,
                            'Current Rate Mult.': current_rate_mult_formatted,
                            'Recommended Rate Mult.': rec_rate_mult_formatted,
                            'Current Rate Exp.': current_rate_exp_formatted,
                            'Recommended Rate Exp.': rec_rate_exp_formatted
                        })
                    
                    changed_display_df = pd.DataFrame(changed_display)
                    st.write(changed_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("No significant parameter changes recommended at this time.")
                    
            with tabs[2]:  # Rollbit Comparison
                st.markdown("### Rollbit Parameter Comparison")
                
                # Filter for pairs with Rollbit data
                rollbit_comparison_df = rec_df.dropna(subset=['rollbit_buffer_rate'])
                
                if rollbit_comparison_df.empty:
                    st.info("No matching pairs found with Rollbit data for comparison.")
                else:
                    # Create a side-by-side comparison of SURF and Rollbit parameters
                    comparison_display = []
                    
                    for _, row in rollbit_comparison_df.iterrows():
                        # Format all parameters for display
                        surf_buffer = format_percent(row['current_buffer_rate'])
                        rollbit_buffer = format_percent(row['rollbit_buffer_rate'])
                        
                        surf_position = format_number(row['current_position_multiplier'])
                        rollbit_position = format_number(row['rollbit_position_multiplier'])
                        
                        surf_rate_mult = format_float(row['current_rate_multiplier'], 2)
                        rollbit_rate_mult = format_float(row['rollbit_rate_multiplier'], 2)
                        
                        surf_rate_exp = format_float(row['current_rate_exponent'], 2)
                        rollbit_rate_exp = format_float(row['rollbit_rate_exponent'], 2)
                        
                        # Calculate ratios (SURF vs Rollbit)
                        buffer_ratio = row['current_buffer_rate'] / row['rollbit_buffer_rate'] if row['rollbit_buffer_rate'] > 0 else 0
                        position_ratio = row['current_position_multiplier'] / row['rollbit_position_multiplier'] if row['rollbit_position_multiplier'] > 0 else 0
                        rate_mult_ratio = row['current_rate_multiplier'] / row['rollbit_rate_multiplier'] if row['rollbit_rate_multiplier'] > 0 else 0
                        rate_exp_ratio = row['current_rate_exponent'] / row['rollbit_rate_exponent'] if row['rollbit_rate_exponent'] > 0 else 0
                        
                        comparison_display.append({
                            'Pair': row['pair_name'],
                            'Type': row['token_type'],
                            'SURF Buffer': f"<span class='surf-param'>{surf_buffer}</span>",
                            'Rollbit Buffer': f"<span class='rollbit-param'>{rollbit_buffer}</span>",
                            'Buffer Ratio': f"{buffer_ratio:.2f}x",
                            'SURF Position': f"<span class='surf-param'>{surf_position}</span>",
                            'Rollbit Position': f"<span class='rollbit-param'>{rollbit_position}</span>",
                            'Position Ratio': f"{position_ratio:.2f}x",
                            'SURF Rate Mult': f"<span class='surf-param'>{surf_rate_mult}</span>",
                            'Rollbit Rate Mult': f"<span class='rollbit-param'>{rollbit_rate_mult}</span>",
                            'Rate Mult Ratio': f"{rate_mult_ratio:.2f}x",
                            'SURF Rate Exp': f"<span class='surf-param'>{surf_rate_exp}</span>",
                            'Rollbit Rate Exp': f"<span class='rollbit-param'>{rollbit_rate_exp}</span>",
                            'Rate Exp Ratio': f"{rate_exp_ratio:.2f}x"
                        })
                    
                    comparison_display_df = pd.DataFrame(comparison_display)
                    st.write(comparison_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Add explanations
                    st.markdown("""
                    <div class="rollbit-comparison">
                        <h4>Understanding the Comparison</h4>
                        <p>This tab compares SURF's current parameters with Rollbit's parameters for matching tokens:</p>
                        <ul>
                            <li><b>Buffer Ratio</b>: SURF buffer rate ÷ Rollbit buffer rate. Values > 1 mean SURF is more conservative.</li>
                            <li><b>Position Ratio</b>: SURF position multiplier ÷ Rollbit position multiplier. Values > 1 mean SURF allows larger positions.</li>
                            <li><b>Rate Multiplier Ratio</b>: SURF rate multiplier ÷ Rollbit rate multiplier. Values > 1 mean SURF has higher market impact factors.</li>
                            <li><b>Rate Exponent Ratio</b>: SURF rate exponent ÷ Rollbit rate exponent. Values > 1 mean SURF has steeper market impact curves.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create visualizations for parameter comparisons
                    st.markdown("### Parameter Comparison Visualizations")
                    
                    # Prepare data for visualization
                    viz_data = []
                    for _, row in rollbit_comparison_df.iterrows():
                        viz_data.append({
                            'Pair': row['pair_name'],
                            'Parameter': 'Buffer Rate (%)',
                            'SURF': row['current_buffer_rate'] * 100,
                            'Rollbit': row['rollbit_buffer_rate'] * 100
                        })
                        viz_data.append({
                            'Pair': row['pair_name'],
                            'Parameter': 'Position Multiplier (log10)',
                            'SURF': np.log10(row['current_position_multiplier']),
                            'Rollbit': np.log10(row['rollbit_position_multiplier'])
                        })
                        viz_data.append({
                            'Pair': row['pair_name'],
                            'Parameter': 'Rate Multiplier',
                            'SURF': row['current_rate_multiplier'],
                            'Rollbit': row['rollbit_rate_multiplier']
                        })
                        viz_data.append({
                            'Pair': row['pair_name'],
                            'Parameter': 'Rate Exponent',
                            'SURF': row['current_rate_exponent'],
                            'Rollbit': row['rollbit_rate_exponent']
                        })
                    
                    viz_df = pd.DataFrame(viz_data)
                    
                    # Create scatter plot for each parameter
                    for param in ['Buffer Rate (%)', 'Position Multiplier (log10)', 'Rate Multiplier', 'Rate Exponent']:
                        param_df = viz_df[viz_df['Parameter'] == param]
                        
                        # Calculate min/max for reference line
                        if not param_df.empty:
                            min_val = min(param_df['SURF'].min(), param_df['Rollbit'].min()) * 0.9
                            max_val = max(param_df['SURF'].max(), param_df['Rollbit'].max()) * 1.1
                            
                            fig = px.scatter(
                                param_df,
                                x='Rollbit',
                                y='SURF',
                                hover_name='Pair',
                                title=f"{param}: SURF vs Rollbit",
                                labels={
                                    'Rollbit': f'Rollbit {param}',
                                    'SURF': f'SURF {param}'
                                }
                            )
                            
                            # Add reference line (y=x)
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Equal Values'
                            ))
                            
                            # Add explanatory text
                            if param == 'Buffer Rate (%)':
                                fig.add_annotation(
                                    text="SURF higher",
                                    x=(min_val + max_val) / 2,
                                    y=max_val * 0.9,
                                    showarrow=False,
                                    font=dict(size=10)
                                )
                                fig.add_annotation(
                                    text="Rollbit higher",
                                    x=max_val * 0.9,
                                    y=(min_val + max_val) / 2,
                                    showarrow=False,
                                    font=dict(size=10)
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:  # Major Tokens
                st.markdown("### Major Tokens Only")
                
                # Filter for major tokens
                major_df = rec_df[rec_df['token_type'] == 'Major']
                
                # Display similar to All Pairs tab but filtered for Major tokens
                major_display = []
                for _, row in major_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    current_rate_mult_formatted = format_float(row['current_rate_multiplier'], 2)
                    rec_rate_mult_formatted = format_with_change_indicator(row['current_rate_multiplier'], row['recommended_rate_multiplier'], is_percent=False, decimals=2)
                    
                    current_rate_exp_formatted = format_float(row['current_rate_exponent'], 2)
                    rec_rate_exp_formatted = format_with_change_indicator(row['current_rate_exponent'], row['recommended_rate_exponent'], is_percent=False, decimals=2)
                    
                    # Format weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                        weekly_range_pct = format_range_as_percent_of_current(
                            row['weekly_low'], row['weekly_high'], row['current_spread']
                        )
                    else:
                        weekly_range = "N/A"
                        weekly_range_pct = "N/A"
                    
                    major_display.append({
                        'Pair': row['pair_name'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Weekly Range %': weekly_range_pct,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted,
                        'Current Rate Mult.': current_rate_mult_formatted,
                        'Recommended Rate Mult.': rec_rate_mult_formatted,
                        'Current Rate Exp.': current_rate_exp_formatted,
                        'Recommended Rate Exp.': rec_rate_exp_formatted
                    })
                
                major_display_df = pd.DataFrame(major_display)
                st.write(major_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            with tabs[4]:  # Altcoin Tokens
                st.markdown("### Altcoin Tokens Only")
                
                # Filter for altcoin tokens
                altcoin_df = rec_df[rec_df['token_type'] == 'Altcoin']
                
                # Display similar to Major Tokens tab but for Altcoins
                altcoin_display = []
                for _, row in altcoin_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    current_rate_mult_formatted = format_float(row['current_rate_multiplier'], 2)
                    rec_rate_mult_formatted = format_with_change_indicator(row['current_rate_multiplier'], row['recommended_rate_multiplier'], is_percent=False, decimals=2)
                    
                    current_rate_exp_formatted = format_float(row['current_rate_exponent'], 2)
                    rec_rate_exp_formatted = format_with_change_indicator(row['current_rate_exponent'], row['recommended_rate_exponent'], is_percent=False, decimals=2)
                    
                    # Format weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                        weekly_range_pct = format_range_as_percent_of_current(
                            row['weekly_low'], row['weekly_high'], row['current_spread']
                        )
                    else:
                        weekly_range = "N/A"
                        weekly_range_pct = "N/A"
                    
                    altcoin_display.append({
                        'Pair': row['pair_name'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Weekly Range %': weekly_range_pct,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted,
                        'Current Rate Mult.': current_rate_mult_formatted,
                        'Recommended Rate Mult.': rec_rate_mult_formatted,
                        'Current Rate Exp.': current_rate_exp_formatted,
                        'Recommended Rate Exp.': rec_rate_exp_formatted
                    })
                
                altcoin_display_df = pd.DataFrame(altcoin_display)
                st.write(altcoin_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Add visualizations
            st.markdown("### Spread and Parameter Visualizations")
            
            # Create summary visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a scatter plot with current spread vs weekly range
                try:
                    # Only include rows with valid weekly range data
                    range_df = rec_df.dropna(subset=['weekly_low', 'weekly_high'])
                    
                    if not range_df.empty:
                        # Calculate percentile within range for each pair
                        range_df['percentile'] = range_df.apply(
                            lambda row: ((row['current_spread'] - row['weekly_low']) / 
                                        (row['weekly_high'] - row['weekly_low'])) * 100 
                                        if row['weekly_high'] > row['weekly_low'] else 50, 
                            axis=1
                        )
                        
                        # Clamp percentiles to 0-100 range for display
                        range_df['display_percentile'] = range_df['percentile'].clip(0, 100)
                        
                        # For points outside the range, add an "outside" flag
                        range_df['outside_range'] = ((range_df['current_spread'] < range_df['weekly_low']) | 
                                                   (range_df['current_spread'] > range_df['weekly_high']))
                        
                        # Create color coding for percentile as strings to avoid groupby issues
                        range_df['color_code'] = pd.cut(
                            range_df['display_percentile'],
                            bins=[0, 20, 40, 60, 80, 100],
                            labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'],
                            include_lowest=True
                        ).astype(str)
                        
                        # Add a spot size based on change ratio
                        range_df['spot_size'] = (abs(range_df['spread_change_ratio'] - 1) * 20 + 5).clip(5, 15)
                        
                        # Create scatter plot
                        fig1 = px.scatter(
                            range_df,
                            x='pair_name',
                            y='display_percentile',
                            color='color_code',
                            size='spot_size',
                            hover_name='pair_name',
                            hover_data={
                                'pair_name': False,
                                'display_percentile': ':.1f',
                                'current_spread': ':.2f',
                                'weekly_low': ':.2f',
                                'weekly_high': ':.2f',
                                'spot_size': False
                            },
                            labels={
                                'display_percentile': 'Position in Weekly Range (%)',
                                'pair_name': 'Trading Pair',
                                'color_code': 'Range Position'
                            },
                            title="Current Spread Position in Weekly Range",
                            color_discrete_map={
                                'Very Low': '#1565C0',
                                'Low': '#42A5F5',
                                'Normal': '#66BB6A',
                                'High': '#FFA726',
                                'Very High': '#E53935'
                            }
                        )
                        
                        # Add reference lines and bands
                        fig1.add_shape(
                            type="line", x0=-0.5, x1=len(range_df['pair_name'].unique()) - 0.5,
                            y0=0, y1=0, line=dict(color="rgba(50, 50, 50, 0.2)", width=1, dash="dot")
                        )
                        
                        fig1.add_shape(
                            type="line", x0=-0.5, x1=len(range_df['pair_name'].unique()) - 0.5,
                            y0=100, y1=100, line=dict(color="rgba(50, 50, 50, 0.2)", width=1, dash="dot")
                        )
                        
                        fig1.add_shape(
                            type="rect", x0=-0.5, x1=len(range_df['pair_name'].unique()) - 0.5,
                            y0=20, y1=80, fillcolor="rgba(0, 200, 0, 0.1)", line=dict(width=0), layer="below"
                        )
                        
                        # Update layout
                        fig1.update_layout(
                            yaxis_range=[-10, 110],
                            xaxis_tickangle=-45,
                            height=500
                        )
                        
                        # Annotate exterior points with markers
                        outside_range_df = range_df[range_df['outside_range']]
                        if not outside_range_df.empty:
                            for _, row in outside_range_df.iterrows():
                                fig1.add_annotation(
                                    x=row['pair_name'],
                                    y=row['display_percentile'],
                                    text="⚠️",
                                    showarrow=False,
                                    font=dict(size=16)
                                )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("No weekly range data available for visualization")
                except Exception as e:
                    st.error(f"Error creating the weekly range chart: {str(e)}")
            
            with col2:
                # Create multiple scatter plots showing parameter relationships
                try:
                    # Create tabs for different parameter visualizations
                    param_viz_tabs = st.tabs(["Buffer Rate", "Position Mult.", "Rate Mult.", "Rate Exp."])
                    
                    # Buffer Rate visualization
                    with param_viz_tabs[0]:
                        fig_buffer = px.scatter(
                            rec_df,
                            x='current_buffer_rate',
                            y='recommended_buffer_rate',
                            color='token_type',
                            hover_name='pair_name',
                            labels={
                                'current_buffer_rate': 'Current Buffer Rate',
                                'recommended_buffer_rate': 'Recommended Buffer Rate',
                                'token_type': 'Token Type'
                            },
                            title="Current vs Recommended Buffer Rates"
                        )
                        
                        # Add diagonal reference line
                        max_val = max(rec_df['current_buffer_rate'].max(), rec_df['recommended_buffer_rate'].max())
                        fig_buffer.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='No Change'
                        ))
                        
                        fig_buffer.update_layout(
                            xaxis_title="Current Buffer Rate",
                            yaxis_title="Recommended Buffer Rate",
                            height=400
                        )
                        
                        st.plotly_chart(fig_buffer, use_container_width=True)
                    
                    # Position Multiplier visualization
                    with param_viz_tabs[1]:
                        fig_pos = px.scatter(
                            rec_df,
                            x='current_position_multiplier',
                            y='recommended_position_multiplier',
                            color='token_type',
                            hover_name='pair_name',
                            labels={
                                'current_position_multiplier': 'Current Position Multiplier',
                                'recommended_position_multiplier': 'Recommended Position Multiplier',
                                'token_type': 'Token Type'
                            },
                            title="Current vs Recommended Position Multipliers"
                        )
                        
                        # Add diagonal reference line
                        max_val = max(rec_df['current_position_multiplier'].max(), rec_df['recommended_position_multiplier'].max())
                        fig_pos.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='No Change'
                        ))
                        
                        fig_pos.update_layout(
                            xaxis_title="Current Position Multiplier",
                            yaxis_title="Recommended Position Multiplier",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pos, use_container_width=True)
                    
                    # Rate Multiplier visualization
                    with param_viz_tabs[2]:
                        fig_rate_mult = px.scatter(
                            rec_df,
                            x='current_rate_multiplier',
                            y='recommended_rate_multiplier',
                            color='token_type',
                            hover_name='pair_name',
                            labels={
                                'current_rate_multiplier': 'Current Rate Multiplier',
                                'recommended_rate_multiplier': 'Recommended Rate Multiplier',
                                'token_type': 'Token Type'
                            },
                            title="Current vs Recommended Rate Multipliers"
                        )
                        
                        # Add diagonal reference line
                        max_val = max(rec_df['current_rate_multiplier'].max(), rec_df['recommended_rate_multiplier'].max())
                        fig_rate_mult.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='No Change'
                        ))
                        
                        fig_rate_mult.update_layout(
                            xaxis_title="Current Rate Multiplier",
                            yaxis_title="Recommended Rate Multiplier",
                            height=400
                        )
                        
                        st.plotly_chart(fig_rate_mult, use_container_width=True)
                    
                    # Rate Exponent visualization
                    with param_viz_tabs[3]:
                        fig_rate_exp = px.scatter(
                            rec_df,
                            x='current_rate_exponent',
                            y='recommended_rate_exponent',
                            color='token_type',
                            hover_name='pair_name',
                            labels={
                                'current_rate_exponent': 'Current Rate Exponent',
                                'recommended_rate_exponent': 'Recommended Rate Exponent',
                                'token_type': 'Token Type'
                            },
                            title="Current vs Recommended Rate Exponents"
                        )
                        
                        # Add diagonal reference line
                        max_val = max(rec_df['current_rate_exponent'].max(), rec_df['recommended_rate_exponent'].max())
                        fig_rate_exp.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='No Change'
                        ))
                        
                        fig_rate_exp.update_layout(
                            xaxis_title="Current Rate Exponent",
                            yaxis_title="Recommended Rate Exponent",
                            height=400
                        )
                        
                        st.plotly_chart(fig_rate_exp, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating parameter visualizations: {str(e)}")
            
            # Add a visualization for weekly range percentage distribution
            st.markdown("### Weekly Range Analysis")
            
            # Create a histogram of weekly range percentages
            try:
                if 'weekly_range' in rec_df.columns and 'current_spread' in rec_df.columns:
                    # Calculate weekly range as percentage of current spread
                    range_percent_df = rec_df.copy()
                    range_percent_df['weekly_range_pct'] = range_percent_df.apply(
                        lambda row: (row['weekly_range'] / row['current_spread']) * 100 if pd.notnull(row['weekly_range']) and row['current_spread'] > 0 else np.nan,
                        axis=1
                    )
                    
                    # Drop rows with missing data
                    range_percent_df = range_percent_df.dropna(subset=['weekly_range_pct'])
                    
                    if not range_percent_df.empty:
                        # Create histogram
                        fig3 = px.histogram(
                            range_percent_df,
                            x='weekly_range_pct',
                            color='token_type',
                            nbins=20,
                            title="Distribution of Weekly Range as Percentage of Current Spread",
                            labels={'weekly_range_pct': 'Weekly Range (% of Current Spread)', 'token_type': 'Token Type'},
                            color_discrete_map={'Major': '#1E88E5', 'Altcoin': '#FFA726'}
                        )
                        
                        # Add vertical reference lines for volatility thresholds
                        fig3.add_vline(x=25, line_dash="dash", line_color="#1E88E5", annotation_text="Very Stable (25%)")
                        fig3.add_vline(x=50, line_dash="dash", line_color="#43A047", annotation_text="Moderately Stable (50%)")
                        fig3.add_vline(x=100, line_dash="dash", line_color="#FFA726", annotation_text="Volatile (100%)")
                        
                        # Update layout
                        fig3.update_layout(
                            xaxis_title="Weekly Range as % of Current Spread",
                            yaxis_title="Number of Tokens",
                            height=400,
                            bargap=0.1
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Create a scatter plot comparing weekly range % vs spread change %
                        fig4 = px.scatter(
                            range_percent_df,
                            x='weekly_range_pct',
                            y=range_percent_df['spread_change_ratio'].apply(lambda x: abs(x - 1) * 100),  # Convert to % difference
                            color='token_type',
                            hover_name='pair_name',
                            size='current_spread',
                            title="Correlation Between Weekly Range and Spread Change",
                            labels={
                                'weekly_range_pct': 'Weekly Range (% of Current Spread)',
                                'y': 'Current Spread Change from Baseline (%)',
                                'token_type': 'Token Type'
                            },
                            color_discrete_map={'Major': '#1E88E5', 'Altcoin': '#FFA726'}
                        )
                        
                        # Update layout
                        fig4.update_layout(
                            xaxis_title="Weekly Range as % of Current Spread",
                            yaxis_title="Spread Change from Baseline (%)",
                            height=500
                        )
                        
                        # Add reference grid for different zones
                        fig4.add_shape(type="rect", x0=0, x1=50, y0=0, y1=5, fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0), layer="below")
                        fig4.add_shape(type="rect", x0=50, x1=100, y0=5, y1=10, fillcolor="rgba(255, 255, 0, 0.1)", line=dict(width=0), layer="below")
                        fig4.add_shape(type="rect", x0=100, x1=200, y0=10, y1=100, fillcolor="rgba(255, 165, 0, 0.1)", line=dict(width=0), layer="below")
                        
                        # Add threshold line
                        fig4.add_hline(y=change_threshold*100, line_dash="dash", line_color="red", 
                                      annotation_text=f"Change Threshold ({change_threshold*100}%)")
                        
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # Recommendations title
                        st.markdown("### Recommended Change Thresholds by Token Volatility")
                        
                        # Group tokens by volatility category
                        volatility_categories = [
                            {'name': 'Very Stable', 'min': 0, 'max': 25, 'rec_threshold': 0.1, 'color': '#1E88E5'},
                            {'name': 'Moderately Stable', 'min': 25, 'max': 50, 'rec_threshold': 0.05, 'color': '#43A047'},
                            {'name': 'Volatile', 'min': 50, 'max': 100, 'rec_threshold': 0.03, 'color': '#FFA726'},
                            {'name': 'Highly Volatile', 'min': 100, 'max': float('inf'), 'rec_threshold': 0.01, 'color': '#E53935'}
                        ]
                        
                        # Create a DataFrame to store token category and recommended threshold
                        threshold_recs = []
                        
                        for _, row in range_percent_df.iterrows():
                            # Find which category this token belongs to
                            for category in volatility_categories:
                                if category['min'] <= row['weekly_range_pct'] < category['max']:
                                    recommended_threshold = category['rec_threshold']
                                    volatility_category = category['name']
                                    color = category['color']
                                    break
                            else:
                                # Default fallback (should not happen)
                                recommended_threshold = 0.05
                                volatility_category = 'Unknown'
                                color = '#9E9E9E'
                            
                            # Add to recommendations
                            threshold_recs.append({
                                'Pair': row['pair_name'],
                                'Type': row['token_type'],
                                'Weekly Range %': f"{row['weekly_range_pct']:.2f}%",
                                'Volatility Category': volatility_category,
                                'Recommended Threshold': f"{recommended_threshold*100:.1f}%",
                                'Current Threshold': f"{change_threshold*100:.1f}%",
                                'Color': color
                            })
                        
                        # Create DataFrame and sort by volatility
                        threshold_df = pd.DataFrame(threshold_recs)
                        volatility_order = {'Very Stable': 0, 'Moderately Stable': 1, 'Volatile': 2, 'Highly Volatile': 3}
                        threshold_df['sort_order'] = threshold_df['Volatility Category'].map(volatility_order)
                        threshold_df = threshold_df.sort_values(['sort_order', 'Pair'])
                        
                        # Create HTML table with colored cells
                        html_rows = []
                        html_rows.append("<table style='width:100%; border-collapse:collapse;'>")
                        html_rows.append("<tr style='background-color:#f2f2f2;'>")
                        html_rows.append("<th style='text-align:left; padding:8px;'>Pair</th>")
                        html_rows.append("<th style='text-align:center; padding:8px;'>Type</th>")
                        html_rows.append("<th style='text-align:center; padding:8px;'>Weekly Range %</th>")
                        html_rows.append("<th style='text-align:center; padding:8px;'>Volatility Category</th>")
                        html_rows.append("<th style='text-align:center; padding:8px;'>Recommended Threshold</th>")
                        html_rows.append("<th style='text-align:center; padding:8px;'>Current Threshold</th>")
                        html_rows.append("</tr>")
                        
                        for _, row in threshold_df.iterrows():
                            html_rows.append("<tr>")
                            html_rows.append(f"<td style='text-align:left; padding:8px;'>{row['Pair']}</td>")
                            html_rows.append(f"<td style='text-align:center; padding:8px;'>{row['Type']}</td>")
                            html_rows.append(f"<td style='text-align:center; padding:8px;'>{row['Weekly Range %']}</td>")
                            html_rows.append(f"<td style='text-align:center; padding:8px; background-color:{row['Color']}; color:white;'>{row['Volatility Category']}</td>")
                            html_rows.append(f"<td style='text-align:center; padding:8px;'>{row['Recommended Threshold']}</td>")
                            html_rows.append(f"<td style='text-align:center; padding:8px;'>{row['Current Threshold']}</td>")
                            html_rows.append("</tr>")
                        
                        html_rows.append("</table>")
                        html_table = "".join(html_rows)
                        
                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        # Create summary statistics
                        volatility_counts = threshold_df['Volatility Category'].value_counts().reset_index()
                        volatility_counts.columns = ['Category', 'Count']
                        volatility_counts['Color'] = volatility_counts['Category'].map({
                            'Very Stable': '#1E88E5', 'Moderately Stable': '#43A047',
                            'Volatile': '#FFA726', 'Highly Volatile': '#E53935'
                        })
                        volatility_counts['sort_order'] = volatility_counts['Category'].map(volatility_order)
                        volatility_counts = volatility_counts.sort_values('sort_order')
                        
                        # Create bar chart of volatility distribution
                        fig6 = px.bar(
                            volatility_counts,
                            x='Count',
                            y='Category',
                            orientation='h',
                            title="Distribution of Tokens by Volatility Category",
                            color='Category',
                            color_discrete_map={
                                'Very Stable': '#1E88E5', 'Moderately Stable': '#43A047',
                                'Volatile': '#FFA726', 'Highly Volatile': '#E53935'
                            }
                        )
                        
                        # Update layout
                        fig6.update_layout(
                            xaxis_title="Number of Tokens",
                            yaxis_title="Volatility Category",
                            yaxis_categoryorder='array',
                            yaxis_categoryarray=['Highly Volatile', 'Volatile', 'Moderately Stable', 'Very Stable'],
                            height=300,
                            showlegend=False
                        )
                        
                        # Add value labels
                        fig6.update_traces(texttemplate='%{x}', textposition='outside')
                        
                        st.plotly_chart(fig6, use_container_width=True)
                    else:
                        st.info("Insufficient data for weekly range percentage analysis")
                else:
                    st.info("Weekly range data not available. Check data quality or refresh the data.")
            except Exception as e:
                st.error(f"Error in weekly range analysis: {str(e)}")
            
            # Handle the "Apply Recommendations" button
            if apply_button:
                st.markdown("### Applying Recommendations")
                
                # Filter for pairs with significant changes
                to_update = rec_df[rec_df['significant_change']]
                
                if not to_update.empty:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Apply changes pair by pair
                    success_count = 0
                    error_count = 0
                    
                    for i, (_, row) in enumerate(to_update.iterrows()):
                        pair_name = row['pair_name']
                        
                        # Create parameters dictionary
                        params = {
                            'buffer_rate': row['recommended_buffer_rate'],
                            'position_multiplier': row['recommended_position_multiplier'],
                            'rate_multiplier': row['recommended_rate_multiplier'],
                            'rate_exponent': row['recommended_rate_exponent']
                        }
                        
                        # Update in the database
                        if update_trading_parameters(pair_name, params):
                            success_count += 1
                        else:
                            error_count += 1
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(to_update))
                    
                    # Show results
                    if error_count == 0:
                        st.success(f"Successfully updated parameters for {success_count} pairs")
                    else:
                        st.warning(f"Updated {success_count} pairs, but encountered {error_count} errors")
                    
                    # Clear cache to refresh data
                    st.cache_data.clear()
                else:
                    st.info("No significant parameter changes to apply")
        else:
            st.warning("Unable to generate recommendations. Check data quality.")
    else:
        missing = []
        if daily_avg_data is None or daily_avg_data.empty:
            missing.append("spread data")
        if baseline_spreads_df is None or baseline_spreads_df.empty:
            missing.append("baseline data")
            
        st.warning(f"Missing {' and '.join(missing)}. Please ensure both are available.")

if __name__ == "__main__":
    main()
                                                                                  