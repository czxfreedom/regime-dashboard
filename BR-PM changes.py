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
    .header-style {
        font-size:24px !important;
        font-weight: bold;
        padding: 10px 0;
    }
    .subheader-style {
        font-size:20px !important;
        font-weight: bold;
        padding: 5px 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    /* Center numeric columns in dataframes */
    .dataframe th, .dataframe td {
        text-align: center !important;
        font-family: monospace;
    }
    /* First column (Token) remains left-aligned */
    .dataframe th:first-child, .dataframe td:first-child {
        text-align: left !important;
        font-family: inherit;
    }
    /* Highlight changes */
    .positive-change {
        color: green;
        font-weight: bold;
    }
    .negative-change {
        color: red;
        font-weight: bold;
    }
    /* Highlight spread ranges */
    .high-range {
        background-color: rgba(255, 152, 0, 0.2);
    }
    .extreme-range {
        background-color: rgba(244, 67, 54, 0.2);
    }
    /* Success message */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #c3e6cb;
    }
    /* Warning message */
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #ffeeba;
    }
    /* Parameter controls section */
    .parameter-controls {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants and Configuration ---
# Define exchanges for spread calculation
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture"]
exchanges_display = {
    "binanceFuture": "Binance",
    "gateFuture": "Gate",
    "hyperliquidFuture": "Hyperliquid"
}

# Define time parameters
singapore_timezone = pytz.timezone('Asia/Singapore')
lookback_days = 1  # Default to 1 day
weekly_lookback_days = 7  # For weekly range

# Use a fixed scale factor for consistency
scale_factor = 10000
scale_label = "× 10,000"

# --- Database Configuration ---
@st.cache_resource
def init_connection():
    try:
        # Try to get database config from secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        return create_engine(db_uri)
    except Exception as e:
        st.sidebar.error(f"Error connecting to the database: {e}")
        return None

# --- Utility Functions ---
def is_major(token):
    """Determine if a token is a major token"""
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
            return True
    return False

def get_depth_level(token):
    """Get the appropriate depth level based on token type"""
    if is_major(token):
        return "50K", "fee1"  # 50K for majors
    else:
        return "20K", "fee1"  # 20K for altcoins

def parse_leverage_config(leverage_config):
    """Parse the leverage_config JSON to extract max leverage info"""
    if not leverage_config or leverage_config == '[]':
        return 100  # Default max leverage if not specified
    
    try:
        if isinstance(leverage_config, str):
            config = json.loads(leverage_config)
        else:
            config = leverage_config
            
        if isinstance(config, list) and len(config) > 0:
            # Find the maximum leverage value in the config
            max_lev = max([item.get('leverage', 0) for item in config])
            return max_lev if max_lev > 0 else 100
    except Exception as e:
        print(f"Error parsing leverage config: {e}")
        return 100  # Default

def format_percent(value):
    """Format value as percentage with 2 decimal places"""
    return f"{value*100:.2f}%"

def format_number(value):
    """Format large numbers with commas"""
    return f"{int(value):,}"

def format_with_change_indicator(old_value, new_value, is_percent=False):
    """Format a value with color coding and change indicator"""
    if is_percent:
        formatted_old = format_percent(old_value)
        formatted_new = format_percent(new_value)
        change_pct = ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
    else:
        formatted_old = format_number(old_value) 
        formatted_new = format_number(new_value)
        change_pct = ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
    
    change_indicator = f"({change_pct:+.2f}%)"
    
    if new_value > old_value:
        return f"{formatted_new} <span class='positive-change'>{change_indicator}</span>"
    elif new_value < old_value:
        return f"{formatted_new} <span class='negative-change'>{change_indicator}</span>"
    else:
        return formatted_new

def format_spread_range(low, high, current, baseline):
    """Format the spread range with indicators for position in range"""
    range_value = high - low
    # Calculate percentile position of current in range
    if range_value > 0:
        position_pct = (current - low) / range_value * 100
    else:
        position_pct = 50  # Default if there's no range
    
    # Determine if current is near high/low
    css_class = ""
    if position_pct > 80:
        css_class = "high-range"  # Near high
    elif position_pct < 20:
        css_class = "high-range"  # Near low
    
    # Check if current is outside the range (extreme value)
    if current > high or current < low:
        css_class = "extreme-range"
    
    # Format with range and indicator
    return f"<span class='{css_class}'>{low:.2f} - {high:.2f} [{range_value:.2f}]</span>"

# --- Data Fetching Functions ---
# Note: We're not using the engine as a cache key parameter anymore
@st.cache_data(ttl=600)
def fetch_all_tokens():
    """Fetch all active tokens from the database"""
    try:
        engine = init_connection()
        if not engine:
            return []
            
        query = """
        SELECT DISTINCT pair_name 
        FROM public.trade_pool_pairs
        WHERE status = 1
        ORDER BY pair_name
        """
        
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_daily_spread_averages(tokens):
    """Fetch daily spread averages for multiple tokens (last 24 hours)"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Create placeholders for tokens
        tokens_str = "', '".join(tokens)

        # Query to get average fee data for all selected tokens
        query = f"""
        SELECT 
            pair_name,
            source,
            AVG(fee1) as avg_fee1,
            AVG(fee2) as avg_fee2,
            AVG(fee3) as avg_fee3,
            AVG(fee4) as avg_fee4,
            AVG(total_fee) as avg_total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name IN ('{tokens_str}')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        return df
            
    except Exception as e:
        st.error(f"Error fetching daily spread averages: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_weekly_spread_ranges(tokens):
    """Fetch weekly high/low/average spread data for multiple tokens"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=weekly_lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Create placeholders for tokens
        tokens_str = "', '".join(tokens)

        # Query to get high/low/avg spread data for all selected tokens
        query = f"""
        WITH spread_data AS (
            SELECT 
                pair_name,
                source,
                time_group,
                fee1,
                fee2,
                fee3,
                fee4,
                total_fee
            FROM 
                oracle_exchange_fee
            WHERE 
                pair_name IN ('{tokens_str}')
                AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        ),
        aggregated_data AS (
            -- Calculate average spread for each pair, source, and time point
            SELECT 
                pair_name,
                time_group,
                AVG(fee1) as avg_spread
            FROM 
                spread_data
            GROUP BY 
                pair_name, time_group
        )
        -- Calculate high, low, avg for each pair
        SELECT 
            pair_name,
            MIN(avg_spread) as weekly_low,
            MAX(avg_spread) as weekly_high,
            AVG(avg_spread) as weekly_avg
        FROM 
            aggregated_data
        GROUP BY 
            pair_name
        ORDER BY 
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        return df
            
    except Exception as e:
        st.error(f"Error fetching weekly spread ranges: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_current_parameters():
    """Fetch current trading parameters from the database"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT
            pair_name,
            buffer_rate,
            position_multiplier,
            max_leverage,
            leverage_config,
            status
        FROM
            public.trade_pool_pairs
        WHERE
            status = 1
        ORDER BY
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        
        # Add max_leverage column from leverage_config if it doesn't exist
        if 'max_leverage' not in df.columns or df['max_leverage'].isna().all():
            df['max_leverage'] = df['leverage_config'].apply(parse_leverage_config)
        
        return df
    except Exception as e:
        st.error(f"Error fetching current parameters: {e}")
        return None

def get_non_surf_average_spread(spread_data, pair_name, fee_level='avg_fee1'):
    """Calculate the average spread across non-SurfFuture exchanges for a specific pair"""
    # Filter for the specific pair
    pair_data = spread_data[spread_data['pair_name'] == pair_name]
    
    # Keep only supported exchanges
    valid_sources = [source for source in exchanges if source in pair_data['source'].values]
    filtered_data = pair_data[pair_data['source'].isin(valid_sources)]
    
    if not filtered_data.empty and fee_level in filtered_data.columns:
        fee_values = filtered_data[fee_level]
        avg_value = fee_values.mean()
        return avg_value
    return None

def calculate_recommended_params(current_buffer_rate, current_position_multiplier, current_spread, baseline_spread, max_leverage, buffer_sensitivity, position_sensitivity, significant_change_threshold):
    """
    Calculate recommended buffer rate and position multiplier based on spread changes
    
    Args:
        current_buffer_rate: Current buffer rate in the system
        current_position_multiplier: Current position multiplier in the system
        current_spread: Current non-Surf average spread
        baseline_spread: Baseline non-Surf average spread to compare against
        max_leverage: Maximum leverage allowed for this trading pair
        buffer_sensitivity: How sensitive buffer rate is to spread changes
        position_sensitivity: How sensitive position multiplier is to spread changes
        significant_change_threshold: Minimum % change required to trigger adjustments
        
    Returns:
        Tuple of (recommended_buffer_rate, recommended_position_multiplier)
    """
    if current_spread is None or baseline_spread is None or baseline_spread <= 0:
        return current_buffer_rate, current_position_multiplier
    
    # Calculate relative change in spread compared to baseline
    spread_change_ratio = current_spread / baseline_spread
    
    # Only make changes if the spread has changed significantly
    if abs(spread_change_ratio - 1.0) < significant_change_threshold:
        return current_buffer_rate, current_position_multiplier
    
    # Calculate recommended buffer rate (direct relationship with spread)
    # Higher spread -> higher buffer rate
    recommended_buffer_rate = current_buffer_rate * (spread_change_ratio ** buffer_sensitivity)
    
    # Calculate recommended position multiplier (inverse relationship with spread)
    # Higher spread -> lower position multiplier
    recommended_position_multiplier = current_position_multiplier / (spread_change_ratio ** position_sensitivity)
    
    # Apply bounds to prevent extreme values
    # Buffer rate should never exceed 1/max_leverage (to avoid immediate liquidations)
    max_buffer_rate = 0.9 / max_leverage if max_leverage > 0 else 0.009
    recommended_buffer_rate = max(0.001, min(max_buffer_rate, recommended_buffer_rate))
    
    # Position multiplier bounds
    recommended_position_multiplier = max(100000, min(10000000, recommended_position_multiplier))
    
    return recommended_buffer_rate, recommended_position_multiplier

def create_baseline_spreads(spread_data):
    """
    Create or update baseline spreads in the database
    """
    try:
        engine = init_connection()
        if not engine:
            return "Database connection failed"
            
        # Create table if it doesn't exist
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
        
        # For each pair, calculate and save the baseline spread
        success_count = 0
        error_count = 0
        
        for pair_name in spread_data['pair_name'].unique():
            # Calculate non-Surf average spread
            depth_label, fee_level = get_depth_level(pair_name)
            fee_column = f"avg_{fee_level}"
            
            non_surf_avg = get_non_surf_average_spread(spread_data, pair_name, fee_column)
            
            if non_surf_avg is not None:
                # Insert or update baseline
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
    """Get baseline spreads from the database"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = "SELECT pair_name, baseline_spread, updated_at FROM spread_baselines"
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching baseline spreads: {e}")
        return None

def update_trading_parameters(pair_name, buffer_rate, position_multiplier):
    """Update buffer_rate and position_multiplier for a trading pair"""
    try:
        engine = init_connection()
        if not engine:
            return False
            
        query = f"""
        UPDATE public.trade_pool_pairs
        SET 
            buffer_rate = {buffer_rate},
            position_multiplier = {position_multiplier},
            updated_at = NOW()
        WHERE pair_name = '{pair_name}';
        """
        
        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        return True
    except Exception as e:
        st.error(f"Error updating parameters for {pair_name}: {e}")
        return False

# --- Main Application ---
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
                        'user': db_user,
                        'password': db_password,
                        'host': db_host,
                        'port': db_port,
                        'database': db_name
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
    - Change Threshold: **{change_threshold*100}%**
    """)
    
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
            <li><strong>Change Threshold:</strong> Minimum spread change needed before parameters are adjusted (lower = more frequent updates)</li>
        </ul>
        <p>Adjust these settings in the sidebar to fine-tune the recommendations.</p>
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
    This tool helps optimize buffer rates and position multipliers based on market spread conditions.<br><br>
    <b>How it works:</b>
    <ul>
        <li>We monitor non-SurfFuture spreads at 50K (majors) / 20K (altcoins) sizes as a baseline</li>
        <li>When spreads increase above baseline, buffer rates increase and position multipliers decrease</li>
        <li>When spreads decrease below baseline, buffer rates decrease and position multipliers increase</li>
        <li>Parameters are kept within safe bounds based on max leverage and other constraints</li>
        <li>Weekly spread ranges provide context about whether current spreads are at unusual levels</li>
    </ul>
    <b>Actions:</b>
    <ul>
        <li>Use "Apply All Recommendations" to update parameters in the system</li>
        <li>Use "Reset All Baselines" to establish new baselines based on current market conditions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Process data and create recommendations
    if daily_avg_data is not None and not daily_avg_data.empty and baseline_spreads_df is not None and not baseline_spreads_df.empty:
        # Create recommendations dataframe
        recommendations_data = []
        
        # Uncomment to enable debug mode
        debug_mode = False
        if debug_mode:
            debug_container = st.expander("Debug Information", expanded=False)
        
        for _, param_row in current_params_df.iterrows():
            pair_name = param_row['pair_name']
            current_buffer_rate = param_row['buffer_rate']
            current_position_multiplier = param_row['position_multiplier']
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
            
            # Calculate recommended parameters
            if current_spread is not None and baseline_spread is not None:
                # Calculate spread change ratio for display
                spread_change_ratio = current_spread / baseline_spread
                
                # Show debug information if enabled
                if debug_mode:
                    with debug_container:
                        st.write(f"**{pair_name}**: Current Spread: {current_spread:.8f}, Baseline: {baseline_spread:.8f}")
                        st.write(f"Spread Ratio: {spread_change_ratio:.4f}, Threshold: {abs(spread_change_ratio - 1.0):.4f} vs {change_threshold:.4f}")
                
                # Calculate recommendations using the dynamic parameters
                rec_buffer, rec_position = calculate_recommended_params(
                    current_buffer_rate, 
                    current_position_multiplier,
                    current_spread,
                    baseline_spread,
                    max_leverage,
                    buffer_sensitivity,
                    position_sensitivity,
                    change_threshold
                )
                
                # Format the message
                if current_spread > baseline_spread:
                    spread_note = f"↑ {(current_spread/baseline_spread - 1)*100:.2f}%"
                elif current_spread < baseline_spread:
                    spread_note = f"↓ {(1 - current_spread/baseline_spread)*100:.2f}%"
                else:
                    spread_note = "No change"
                
                # Calculate if there is a significant change
                buffer_change_pct = abs((rec_buffer - current_buffer_rate) / current_buffer_rate) * 100 if current_buffer_rate > 0 else 0
                position_change_pct = abs((rec_position - current_position_multiplier) / current_position_multiplier) * 100 if current_position_multiplier > 0 else 0
                significant_change = buffer_change_pct > 1 or position_change_pct > 1
                
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
                    'spread_change': spread_note,
                    'spread_change_ratio': spread_change_ratio,  # For sorting
                    'current_buffer_rate': current_buffer_rate,
                    'recommended_buffer_rate': rec_buffer,
                    'current_position_multiplier': current_position_multiplier,
                    'recommended_position_multiplier': rec_position,
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
            tabs = st.tabs(["All Pairs", "Changed Parameters Only", "Major Tokens", "Altcoin Tokens"])
            
            with tabs[0]:  # All Pairs
                st.markdown("### All Trading Pairs")
                
                # Create display data
                display_data = []
                for _, row in rec_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    # Format the weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                    else:
                        weekly_range = "N/A"
                    
                    display_data.append({
                        'Pair': row['pair_name'],
                        'Type': row['token_type'],
                        'Size': row['depth_tier'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted
                    })
                
                display_df = pd.DataFrame(display_data)
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Add legend for weekly range formatting
                st.markdown("""
                <div style="font-size: 0.85em; margin-top: 10px;">
                <strong>Weekly Range Legend:</strong>
                <ul style="list-style-type: none; padding-left: 10px; margin-top: 5px;">
                    <li><span class="high-range" style="padding: 2px 5px;">Yellow Background</span>: Current spread is near the high or low of the weekly range (20% from extremes)</li>
                    <li><span class="extreme-range" style="padding: 2px 5px;">Red Background</span>: Current spread is outside the weekly range (extreme value)</li>
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
                        
                        # Format the weekly range
                        if row['weekly_low'] is not None and row['weekly_high'] is not None:
                            weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                        else:
                            weekly_range = "N/A"
                        
                        changed_display.append({
                            'Pair': row['pair_name'],
                            'Type': row['token_type'],
                            'Size': row['depth_tier'],
                            'Market Spread': f"{row['current_spread']:.2f}",
                            'Baseline Spread': f"{row['baseline_spread']:.2f}",
                            'Weekly Range': weekly_range,
                            'Spread Change': row['spread_change'],
                            'Current Buffer': current_buffer_formatted,
                            'Recommended Buffer': rec_buffer_formatted,
                            'Current Position Mult.': current_pos_formatted,
                            'Recommended Position Mult.': rec_pos_formatted
                        })
                    
                    changed_display_df = pd.DataFrame(changed_display)
                    st.write(changed_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Add legend for weekly range formatting
                    st.markdown("""
                    <div style="font-size: 0.85em; margin-top: 10px;">
                    <strong>Weekly Range Legend:</strong>
                    <ul style="list-style-type: none; padding-left: 10px; margin-top: 5px;">
                        <li><span class="high-range" style="padding: 2px 5px;">Yellow Background</span>: Current spread is near the high or low of the weekly range (20% from extremes)</li>
                        <li><span class="extreme-range" style="padding: 2px 5px;">Red Background</span>: Current spread is outside the weekly range (extreme value)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant parameter changes recommended at this time.")
            
            with tabs[2]:  # Major Tokens
                st.markdown("### Major Tokens Only")
                
                # Filter for major tokens
                major_df = rec_df[rec_df['token_type'] == 'Major']
                
                # Create display data for major tokens
                major_display = []
                for _, row in major_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    # Format the weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                    else:
                        weekly_range = "N/A"
                    
                    major_display.append({
                        'Pair': row['pair_name'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted
                    })
                
                major_display_df = pd.DataFrame(major_display)
                st.write(major_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Add legend for weekly range formatting
                st.markdown("""
                <div style="font-size: 0.85em; margin-top: 10px;">
                <strong>Weekly Range Legend:</strong>
                <ul style="list-style-type: none; padding-left: 10px; margin-top: 5px;">
                    <li><span class="high-range" style="padding: 2px 5px;">Yellow Background</span>: Current spread is near the high or low of the weekly range (20% from extremes)</li>
                    <li><span class="extreme-range" style="padding: 2px 5px;">Red Background</span>: Current spread is outside the weekly range (extreme value)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tabs[3]:  # Altcoin Tokens
                st.markdown("### Altcoin Tokens Only")
                
                # Filter for altcoin tokens
                altcoin_df = rec_df[rec_df['token_type'] == 'Altcoin']
                
                # Create display data for altcoin tokens
                altcoin_display = []
                for _, row in altcoin_df.iterrows():
                    current_buffer_formatted = format_percent(row['current_buffer_rate'])
                    rec_buffer_formatted = format_with_change_indicator(row['current_buffer_rate'], row['recommended_buffer_rate'], is_percent=True)
                    
                    current_pos_formatted = format_number(row['current_position_multiplier'])
                    rec_pos_formatted = format_with_change_indicator(row['current_position_multiplier'], row['recommended_position_multiplier'])
                    
                    # Format the weekly range
                    if row['weekly_low'] is not None and row['weekly_high'] is not None:
                        weekly_range = format_spread_range(row['weekly_low'], row['weekly_high'], row['current_spread'], row['baseline_spread'])
                    else:
                        weekly_range = "N/A"
                    
                    altcoin_display.append({
                        'Pair': row['pair_name'],
                        'Market Spread': f"{row['current_spread']:.2f}",
                        'Baseline Spread': f"{row['baseline_spread']:.2f}",
                        'Weekly Range': weekly_range,
                        'Spread Change': row['spread_change'],
                        'Current Buffer': current_buffer_formatted,
                        'Recommended Buffer': rec_buffer_formatted,
                        'Current Position Mult.': current_pos_formatted,
                        'Recommended Position Mult.': rec_pos_formatted
                    })
                
                altcoin_display_df = pd.DataFrame(altcoin_display)
                st.write(altcoin_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Add legend for weekly range formatting
                st.markdown("""
                <div style="font-size: 0.85em; margin-top: 10px;">
                <strong>Weekly Range Legend:</strong>
                <ul style="list-style-type: none; padding-left: 10px; margin-top: 5px;">
                    <li><span class="high-range" style="padding: 2px 5px;">Yellow Background</span>: Current spread is near the high or low of the weekly range (20% from extremes)</li>
                    <li><span class="extreme-range" style="padding: 2px 5px;">Red Background</span>: Current spread is outside the weekly range (extreme value)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Add visualizations
            st.markdown("### Spread and Parameter Visualizations")
            
            # Create summary visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a scatter plot with current spread vs weekly range
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
                    
                    # Create color coding for percentile
                    range_df['color_code'] = pd.cut(
                        range_df['display_percentile'],
                        bins=[0, 20, 40, 60, 80, 100],
                        labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
                    )
                    
                    # Add a spot size based on change ratio
                    range_df['spot_size'] = (abs(range_df['spread_change_ratio'] - 1) * 20 + 5).clip(5, 15)
                    
                    # Create scatter plot
                    fig1 = px.scatter(
                        range_df,
                        x='pair_name',
                        y='percentile',
                        color='color_code',
                        size='spot_size',
                        hover_name='pair_name',
                        hover_data={
                            'pair_name': False,
                            'percentile': ':.1f',
                            'current_spread': ':.2f',
                            'weekly_low': ':.2f',
                            'weekly_high': ':.2f',
                            'spot_size': False
                        },
                        labels={
                            'percentile': 'Position in Weekly Range (%)',
                            'pair_name': 'Trading Pair',
                            'color_code': 'Range Position'
                        },
                        title="Current Spread Position in Weekly Range",
                        color_discrete_map={
                            'Very Low': '#1565C0',   # Dark blue
                            'Low': '#42A5F5',        # Light blue
                            'Normal': '#66BB6A',     # Green
                            'High': '#FFA726',       # Orange
                            'Very High': '#E53935'   # Red
                        }
                    )
                    
                    # Add reference lines
                    fig1.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(range_df['pair_name'].unique()) - 0.5,
                        y0=0,
                        y1=0,
                        line=dict(color="rgba(50, 50, 50, 0.2)", width=1, dash="dot")
                    )
                    
                    fig1.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(range_df['pair_name'].unique()) - 0.5,
                        y0=100,
                        y1=100,
                        line=dict(color="rgba(50, 50, 50, 0.2)", width=1, dash="dot")
                    )
                    
                    # Add a band for the "normal" range (20-80%)
                    fig1.add_shape(
                        type="rect",
                        x0=-0.5,
                        x1=len(range_df['pair_name'].unique()) - 0.5,
                        y0=20,
                        y1=80,
                        fillcolor="rgba(0, 200, 0, 0.1)",
                        line=dict(width=0),
                        layer="below"
                    )
                    
                    # Update layout
                    fig1.update_layout(
                        yaxis_range=[-10, 110],  # Add margin for points outside range
                        xaxis_tickangle=-45,
                        height=500
                    )
                    
                    # Annotate exterior points with markers
                    for idx, row in range_df[range_df['outside_range']].iterrows():
                        fig1.add_annotation(
                            x=row['pair_name'],
                            y=row['display_percentile'],
                            text="⚠️",
                            showarrow=False,
                            font=dict(size=16)
                        )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Add explanatory text
                    st.markdown("""
                    **Understanding the Weekly Range Chart:**
                    - **0%**: At or below the lowest spread seen this week
                    - **50%**: Middle of the weekly range
                    - **100%**: At or above the highest spread seen this week
                    - **⚠️ Marker**: Indicates spread is currently outside the weekly range
                    - **Dot Size**: Larger dots indicate greater deviation from baseline
                    """)
                else:
                    st.info("No weekly range data available for visualization")
            
            with col2:
                # Create a scatter plot showing current buffer rate vs recommended
                fig2 = px.scatter(
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
                
                # Add diagonal reference line (no change)
                max_val = max(rec_df['current_buffer_rate'].max(), rec_df['recommended_buffer_rate'].max())
                fig2.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='No Change'
                ))
                
                fig2.update_layout(
                    xaxis_title="Current Buffer Rate",
                    yaxis_title="Recommended Buffer Rate",
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Add a visualization for spread distribution
            st.markdown("### Spread Distribution Analysis")
            
            # Create a box plot of current spreads vs baselines by token type
            box_df = rec_df.copy()
            
            # Create long-format data for boxplot
            box_data = []
            for _, row in box_df.iterrows():
                # Current spread
                box_data.append({
                    'Pair': row['pair_name'],
                    'Token Type': row['token_type'],
                    'Spread Type': 'Current',
                    'Spread Value': row['current_spread']
                })
                # Baseline spread
                box_data.append({
                    'Pair': row['pair_name'],
                    'Token Type': row['token_type'],
                    'Spread Type': 'Baseline', 
                    'Spread Value': row['baseline_spread']
                })
                # Weekly low (if available)
                if pd.notnull(row['weekly_low']):
                    box_data.append({
                        'Pair': row['pair_name'],
                        'Token Type': row['token_type'],
                        'Spread Type': 'Weekly Low',
                        'Spread Value': row['weekly_low']
                    })
                # Weekly high (if available)
                if pd.notnull(row['weekly_high']):
                    box_data.append({
                        'Pair': row['pair_name'],
                        'Token Type': row['token_type'],
                        'Spread Type': 'Weekly High',
                        'Spread Value': row['weekly_high']
                    })
            
            box_plot_df = pd.DataFrame(box_data)
            
            if not box_plot_df.empty:
                fig3 = px.box(
                    box_plot_df,
                    x='Token Type',
                    y='Spread Value',
                    color='Spread Type',
                    title="Spread Distribution by Token Type",
                    labels={
                        'Spread Value': f'Spread ({scale_label})',
                        'Token Type': 'Token Type',
                        'Spread Type': 'Spread Type'
                    },
                    color_discrete_map={
                        'Current': '#FF9800',
                        'Baseline': '#2196F3',
                        'Weekly Low': '#4CAF50',
                        'Weekly High': '#F44336'
                    }
                )
                
                fig3.update_layout(height=500)
                st.plotly_chart(fig3, use_container_width=True)
            
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
                        rec_buffer = row['recommended_buffer_rate']
                        rec_position = row['recommended_position_multiplier']
                        
                        # Update in the database
                        if update_trading_parameters(pair_name, rec_buffer, rec_position):
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
            
            # Add explanatory content
            with st.expander("How Parameter Optimization Works"):
                st.markdown("""
                ### How Parameter Optimization Works
                
                #### Key Concepts
                
                1. **Buffer Rate**: Used in calculating trigger prices for busts and stop losses:
                   ```
                   P_trigger = P_close * (1 + trade_sign*bust_buffer)
                   ```
                  - Higher buffer rate means more distance to liquidation
                  - This provides greater protection during volatile periods
                
                2. **Position Multiplier**: Used in market impact calculations:
                   ```
                   P_close(T) = P(t) + ((1 - base_rate) / (1 + 1/abs((P(T)/P(t) - 1)*rate_multiplier)^rate_exponent + bet_amount*bet_multiplier/(10^6*abs(P(T)/P(t) - 1)*position_multiplier)))*(P(T) - P(t))
                   ```
                  - Lower position multiplier increases market impact
                  - This reduces PnL for large winning positions
                
                #### Optimization Logic
                
                - **When Market Spreads Increase**:
                  - Buffer rate increases (more protection against volatility)
                  - Position multiplier decreases (more impact on large positions)
                  - Result: Better protection for the exchange during volatile periods
                
                - **When Market Spreads Decrease**:
                  - Buffer rate decreases (less protection needed)
                  - Position multiplier increases (less impact on positions)
                  - Result: Better user experience during stable periods
                
                #### Parameter Controls
                
                - **Buffer Sensitivity**: Controls how strongly buffer rates react to spread changes
                  - Higher values make the system more aggressive in adjusting buffer rates
                
                - **Position Sensitivity**: Controls how strongly position multipliers react to spread changes
                  - Higher values make the system more aggressive in adjusting position multipliers
                
                - **Change Threshold**: Minimum spread difference required before recommending changes
                  - Lower values make the system more responsive to smaller market changes
                
                #### Weekly Range Context
                
                The weekly range data provides important market context:
                
                - **Weekly Low**: Lowest spread observed over the past 7 days
                - **Weekly High**: Highest spread observed over the past 7 days
                - **Range**: Difference between high and low (shows volatility)
                - **Position in Range**: Where the current spread sits in the weekly range
                  - Near bottom (0-20%): Unusually tight spreads, might widen soon
                  - Middle (20-80%): Normal market conditions
                  - Near top (80-100%): Unusually wide spreads, might tighten soon
                  - Outside range: Extreme market conditions, needs special attention
                
                #### Safety Constraints
                
                - **Buffer Rate Limit**: Never exceeds `0.9/max_leverage` to avoid immediate liquidations
                - **Position Multiplier Bounds**: Between 100,000 and 10,000,000
                
                The system automatically maintains these optimizations while keeping all parameters within safe bounds.
                """)
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