import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(
    page_title="Exchange Parameter Optimization Dashboard",
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
    .param-group {
        border: 2px solid #1976D2;
        border-radius: 5px;
        margin-bottom: 15px;
        overflow: hidden;
    }
    .param-header {
        background-color: #1976D2;
        color: white;
        padding: 8px;
        font-weight: bold;
        text-align: center;
    }
    .param-table {
        width: 100%;
        border-collapse: collapse;
    }
    .param-table th {
        background-color: #f2f2f2;
        padding: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .param-table td {
        padding: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .surf-value {
        background-color: rgba(200, 230, 201, 0.7);
        font-weight: bold;
    }
    .rollbit-value {
        background-color: rgba(255, 236, 179, 0.7);
        font-weight: bold;
    }
    .ratio-value {
        background-color: rgba(225, 245, 254, 0.7);
    }
    /* Center numeric columns */
    .dataframe th, .dataframe td {
        text-align: center !important;
        font-family: monospace;
    }
    /* First column remains left-aligned */
    .dataframe th:first-child, .dataframe td:first-child {
        text-align: left !important;
        font-family: inherit;
    }
    /* Formula display */
    .formula-box {
        background-color: #f1f8e9;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
        border-left: 5px solid #7cb342;
        overflow-x: auto;
    }
    .formula {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    /* Parameter card styles */
    .param-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .param-card-header {
        background-color: #1976D2;
        color: white;
        padding: 10px 15px;
        font-weight: bold;
    }
    .param-card-body {
        padding: 15px;
    }
    /* Error/warning message */
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #d32f2f;
    }
    /* Navigation bar styling */
    .nav-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .nav-item {
        background-color: #f5f5f5;
        padding: 8px 15px;
        border-radius: 5px;
        cursor: pointer;
        border: 1px solid #e0e0e0;
        transition: all 0.3s;
    }
    .nav-item:hover, .nav-item.active {
        background-color: #1976D2;
        color: white;
    }
    /* Range indicator styling */
    .range-indicator {
        margin-top: 10px;
        padding: 5px;
        border-radius: 5px;
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
    }
    /* Highlight for current value in range */
    .current-in-range {
        font-weight: bold;
        color: #1976D2;
    }
    /* Styling for spread range info */
    .spread-range-info {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .range-bar {
        flex-grow: 1;
        height: 20px;
        background: linear-gradient(to right, #4caf50, #ffeb3b, #f44336);
        position: relative;
        border-radius: 3px;
        margin: 0 10px;
    }
    .range-marker {
        position: absolute;
        width: 4px;
        height: 28px;
        background-color: #000;
        top: -4px;
    }
    .range-label {
        font-size: 12px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

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
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.sidebar.error(f"Error connecting to the database: {e}")
        return None

# --- Utility Functions ---
def format_percent(value):
    """Format a value as a percentage with 2 decimal places"""
    if pd.isna(value) or value is None or value == 0:
        return "N/A"
    return f"{value * 100:.2f}%"

def format_number(value):
    """Format a number with comma separation"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:,.0f}"

def format_float(value, decimals=2):
    """Format a float with specified decimal places"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def is_major(token):
    """Determine if a token is a major token"""
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
            return True
    return False

def safe_division(a, b, default=0.0):
    """Safely divide two numbers, handling zeros and None values"""
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return default
    return a / b

def check_null_or_zero(value):
    """Check if a value is NULL, None, NaN, or zero"""
    if value is None or pd.isna(value) or value == 0:
        return True
    return False

def compute_weekly_spread_range(token, engine):
    """Compute the weekly spread range for a specific token"""
    try:
        # Get Singapore timezone for consistent time handling
        singapore_timezone = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        
        # Get start of previous week (7 days ago)
        start_time_sg = now_sg - timedelta(days=7)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Query to get all spread values from the past week
        query = f"""
        SELECT 
            time_group,
            fee1
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name = '{token}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        ORDER BY 
            time_group ASC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None, None, None
        
        # Calculate min, max and average spread
        min_spread = df['fee1'].min()
        max_spread = df['fee1'].max()
        avg_spread = df['fee1'].mean()
        
        return min_spread, max_spread, avg_spread
    
    except Exception as e:
        print(f"Error computing weekly spread range for {token}: {e}")
        return None, None, None

# --- Data Fetching Functions ---
@st.cache_data(ttl=600)
def fetch_current_parameters():
    """Fetch current parameters from the database"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT
            pair_name,
            buffer_rate,
            position_multiplier,
            rate_multiplier,
            rate_exponent,
            max_leverage
        FROM
            public.trade_pool_pairs
        WHERE
            status = 1  -- Only active pairs
        ORDER BY
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        
        # Fill NaN values with reasonable defaults
        if not df.empty:
            df['rate_multiplier'] = df['rate_multiplier'].fillna(6.0)
            df['rate_exponent'] = df['rate_exponent'].fillna(2.0)
            df['max_leverage'] = df['max_leverage'].fillna(100)
            
            # Ensure buffer_rate has no zeros (replace with NaN for "N/A" display)
            df['buffer_rate'] = df['buffer_rate'].replace(0, np.nan)
            
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching current parameters: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_rollbit_parameters():
    """Fetch Rollbit parameters for comparison"""
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
        
        # Fill NaN values with reasonable defaults
        if not df.empty:
            df['rate_multiplier'] = df['rate_multiplier'].fillna(6.0)
            df['rate_exponent'] = df['rate_exponent'].fillna(2.0)
            
            # Ensure buffer_rate has no zeros (replace with NaN for "N/A" display)
            df['buffer_rate'] = df['buffer_rate'].replace(0, np.nan)
            
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_market_spread_data():
    """Fetch current market spread data for all tokens"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone
        singapore_timezone = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=1)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        query = f"""
        SELECT 
            pair_name,
            source,
            AVG(fee1) as avg_fee1
        FROM 
            oracle_exchange_fee
        WHERE 
            source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching market spread data: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_weekly_spread_data():
    """Fetch weekly spread data for all tokens"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone
        singapore_timezone = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=7)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        query = f"""
        SELECT 
            pair_name,
            source,
            MIN(fee1) as min_fee1,
            MAX(fee1) as max_fee1,
            AVG(fee1) as avg_fee1
        FROM 
            oracle_exchange_fee
        WHERE 
            source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        
        # Aggregate across exchanges to get overall min/max/avg
        if not df.empty:
            aggregated = df.groupby('pair_name').agg({
                'min_fee1': 'min',
                'max_fee1': 'max',
                'avg_fee1': 'mean'
            }).reset_index()
            
            return aggregated
        
        return None
    except Exception as e:
        st.error(f"Error fetching weekly spread data: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_spread_baselines():
    """Fetch spread baselines for comparison"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT 
            pair_name,
            baseline_spread,
            updated_at
        FROM 
            spread_baselines
        ORDER BY 
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching spread baselines: {e}")
        return None

def save_spread_baseline(pair_name, baseline_spread):
    """Save a new spread baseline to the database"""
    try:
        engine = init_connection()
        if not engine:
            return False
            
        query = """
        INSERT INTO spread_baselines (pair_name, baseline_spread, updated_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (pair_name) DO UPDATE 
        SET baseline_spread = EXCLUDED.baseline_spread, 
            updated_at = EXCLUDED.updated_at
        """
        
        with engine.connect() as conn:
            conn.execute(query, (pair_name, baseline_spread, datetime.now()))
            conn.commit()
            
        return True
    except Exception as e:
        st.error(f"Error saving baseline spread for {pair_name}: {e}")
        return False

def reset_all_baselines(market_data_df):
    """Reset all baselines to current market spreads"""
    if market_data_df is None or market_data_df.empty:
        return False
    
    # Calculate current spreads
    current_spreads = calculate_current_spreads(market_data_df)
    
    success_count = 0
    error_count = 0
    
    # Update each baseline in the database
    for pair, spread in current_spreads.items():
        if save_spread_baseline(pair, spread):
            success_count += 1
        else:
            error_count += 1
    
    return success_count > 0, success_count, error_count

def calculate_current_spreads(market_data):
    """Calculate current average non-SurfFuture spread for each token"""
    if market_data is None or market_data.empty:
        return {}
    
    # Group by pair_name and calculate average spread across all exchanges
    current_spreads = {}
    for pair, group in market_data.groupby('pair_name'):
        current_spreads[pair] = group['avg_fee1'].mean()
    
    return current_spreads

def calculate_recommended_params(current_params, current_spread, baseline_spread, weekly_data,
                              sensitivities, significant_change_threshold=0.05):
    """Calculate recommended parameter values based on spread change ratio and weekly range with proper inverse relationships"""
    
    # Handle cases with missing data
    if current_spread is None or baseline_spread is None or baseline_spread <= 0:
        return current_params
    
    # Get current parameter values (with safety checks)
    current_buffer_rate = current_params.get('buffer_rate')
    if check_null_or_zero(current_buffer_rate):
        current_buffer_rate = 0.01  # Default if zero or null
        
    current_position_multiplier = current_params.get('position_multiplier')
    if check_null_or_zero(current_position_multiplier):
        current_position_multiplier = 1000000  # Default if zero or null
        
    current_rate_multiplier = current_params.get('rate_multiplier')
    if check_null_or_zero(current_rate_multiplier):
        current_rate_multiplier = 6.0  # Default if zero or null
        
    current_rate_exponent = current_params.get('rate_exponent')
    if check_null_or_zero(current_rate_exponent):
        current_rate_exponent = 2.0  # Default if zero or null
        
    max_leverage = current_params.get('max_leverage')
    if check_null_or_zero(max_leverage):
        max_leverage = 100  # Default if zero or null
    
    # Get weekly min/max spread if available
    min_weekly_spread = weekly_data.get('min_fee1')
    max_weekly_spread = weekly_data.get('max_fee1')
    avg_weekly_spread = weekly_data.get('avg_fee1')
    
    # Calculate weekly range ratio (how far current spread is within the weekly range)
    weekly_range_ratio = 0.5  # Default to middle of the range
    if not check_null_or_zero(min_weekly_spread) and not check_null_or_zero(max_weekly_spread):
        if max_weekly_spread > min_weekly_spread:
            # Normalize current_spread to [0,1] within the weekly range
            weekly_range_ratio = (current_spread - min_weekly_spread) / (max_weekly_spread - min_weekly_spread)
            # Clamp to [0,1]
            weekly_range_ratio = max(0, min(1, weekly_range_ratio))
    
    # Get sensitivity parameters
    buffer_sensitivity = sensitivities.get('buffer_sensitivity', 0.5)
    position_sensitivity = sensitivities.get('position_sensitivity', 0.5)
    rate_multiplier_sensitivity = sensitivities.get('rate_multiplier_sensitivity', 0.5)
    rate_exponent_sensitivity = sensitivities.get('rate_exponent_sensitivity', 0.5)
    
    # Calculate spread change ratio
    spread_change_ratio = current_spread / baseline_spread
    
    # Adjust spread_change_ratio based on where we are in the weekly range
    # If near the top of the range, strengthen adjustments; if near bottom, weaken them
    range_adjusted_ratio = spread_change_ratio * (0.5 + weekly_range_ratio * 0.5)
    
    # Check if the change is significant
    if abs(range_adjusted_ratio - 1.0) < significant_change_threshold:
        return current_params
    
    # Fixed parameter relationships:
    # 1. If spreads increase, buffer rates should increase (direct relationship)
    recommended_buffer_rate = current_buffer_rate * (range_adjusted_ratio ** buffer_sensitivity)
    
    # 2. If spreads increase, position multiplier should decrease (inverse relationship)
    recommended_position_multiplier = current_position_multiplier / (range_adjusted_ratio ** position_sensitivity)
    
    # 3. If spreads increase, rate multiplier should decrease (inverse relationship)
    recommended_rate_multiplier = current_rate_multiplier / (range_adjusted_ratio ** rate_multiplier_sensitivity)
    
    # 4. If spreads increase, rate exponent should increase (direct relationship)
    recommended_rate_exponent = current_rate_exponent * (range_adjusted_ratio ** rate_exponent_sensitivity)
    
    # Apply bounds to keep values in reasonable ranges
    max_buffer_rate = 0.9 / max_leverage if max_leverage > 0 else 0.009
    recommended_buffer_rate = max(0.001, min(max_buffer_rate, recommended_buffer_rate))
    recommended_position_multiplier = max(100000, min(10000000, recommended_position_multiplier))
    recommended_rate_multiplier = max(1.0, min(15.0, recommended_rate_multiplier))
    recommended_rate_exponent = max(1.0, min(5.0, recommended_rate_exponent))
    
    return {
        'buffer_rate': recommended_buffer_rate,
        'position_multiplier': recommended_position_multiplier,
        'rate_multiplier': recommended_rate_multiplier,
        'rate_exponent': recommended_rate_exponent,
        'weekly_range_ratio': weekly_range_ratio  # Return where we are in the weekly range
    }

def generate_recommendations(current_params_df, market_data_df, weekly_data_df, baselines_df, sensitivities):
    """Generate parameter recommendations based on market data and weekly range"""
    if current_params_df is None or market_data_df is None or baselines_df is None:
        return None
    
    # Convert DataFrames to more convenient formats
    current_params = {}
    for _, row in current_params_df.iterrows():
        pair_name = row['pair_name']
        current_params[pair_name] = {
            'buffer_rate': row['buffer_rate'],
            'position_multiplier': row['position_multiplier'],
            'rate_multiplier': row.get('rate_multiplier', 6.0),
            'rate_exponent': row.get('rate_exponent', 2.0),
            'max_leverage': row.get('max_leverage', 100)
        }
    
    # Calculate current spreads
    current_spreads = calculate_current_spreads(market_data_df)
    
    # Create baselines dictionary
    baselines = {}
    for _, row in baselines_df.iterrows():
        baselines[row['pair_name']] = row['baseline_spread']
    
    # Create weekly data dictionary
    weekly_data = {}
    if weekly_data_df is not None:
        for _, row in weekly_data_df.iterrows():
            weekly_data[row['pair_name']] = {
                'min_fee1': row['min_fee1'],
                'max_fee1': row['max_fee1'],
                'avg_fee1': row['avg_fee1']
            }
    
    # Create recommendations DataFrame
    recommendations = []
    
    significant_change_threshold = 0.05  # 5% change threshold
    
    for pair, params in current_params.items():
        if pair in current_spreads and pair in baselines:
            current_spread = current_spreads[pair]
            baseline_spread = baselines[pair]
            
            # Get weekly data for this pair or empty dict if not available
            token_weekly_data = weekly_data.get(pair, {})
            
            recommended = calculate_recommended_params(
                params, 
                current_spread, 
                baseline_spread,
                token_weekly_data,
                sensitivities,
                significant_change_threshold
            )
            
            # Calculate changes with safety checks
            buffer_change = 0
            if not check_null_or_zero(params['buffer_rate']) and not check_null_or_zero(recommended['buffer_rate']):
                buffer_change = ((recommended['buffer_rate'] - params['buffer_rate']) / params['buffer_rate']) * 100
            
            position_change = 0
            if not check_null_or_zero(params['position_multiplier']) and not check_null_or_zero(recommended['position_multiplier']):
                position_change = ((recommended['position_multiplier'] - params['position_multiplier']) / params['position_multiplier']) * 100
            
            rate_mult_change = 0
            if not check_null_or_zero(params['rate_multiplier']) and not check_null_or_zero(recommended['rate_multiplier']):
                rate_mult_change = ((recommended['rate_multiplier'] - params['rate_multiplier']) / params['rate_multiplier']) * 100
            
            rate_exp_change = 0
            if not check_null_or_zero(params['rate_exponent']) and not check_null_or_zero(recommended['rate_exponent']):
                rate_exp_change = ((recommended['rate_exponent'] - params['rate_exponent']) / params['rate_exponent']) * 100
            
            # Add weekly data to the recommendations
            min_weekly_spread = token_weekly_data.get('min_fee1', None)
            max_weekly_spread = token_weekly_data.get('max_fee1', None)
            avg_weekly_spread = token_weekly_data.get('avg_fee1', None)
            weekly_range_ratio = recommended.get('weekly_range_ratio', 0.5)
            
            recommendations.append({
                'pair_name': pair,
                'token_type': 'Major' if is_major(pair) else 'Altcoin',
                'current_buffer_rate': params['buffer_rate'],
                'recommended_buffer_rate': recommended['buffer_rate'],
                'buffer_change': buffer_change,
                'current_position_multiplier': params['position_multiplier'],
                'recommended_position_multiplier': recommended['position_multiplier'],
                'position_change': position_change,
                'current_rate_multiplier': params['rate_multiplier'],
                'recommended_rate_multiplier': recommended['rate_multiplier'],
                'rate_multiplier_change': rate_mult_change,
                'current_rate_exponent': params['rate_exponent'],
                'recommended_rate_exponent': recommended['rate_exponent'],
                'rate_exponent_change': rate_exp_change,
                'current_spread': current_spread,
                'baseline_spread': baseline_spread,
                'spread_change_ratio': safe_division(current_spread, baseline_spread, 1.0),
                'min_weekly_spread': min_weekly_spread,
                'max_weekly_spread': max_weekly_spread,
                'avg_weekly_spread': avg_weekly_spread,
                'weekly_range_ratio': weekly_range_ratio
            })
    
    return pd.DataFrame(recommendations)

def add_rollbit_comparison(rec_df, rollbit_df):
    """Add Rollbit parameter data to recommendations DataFrame for comparison
       with improved handling of zero/null values"""
    if rec_df is None or rollbit_df is None or rec_df.empty or rollbit_df.empty:
        return rec_df
    
    # Create mapping of Rollbit parameters
    rollbit_params = {}
    for _, row in rollbit_df.iterrows():
        # Clean up pair name to match format in rec_df
        pair_name = row['pair_name']
        
        rollbit_params[pair_name] = {
            'buffer_rate': row['buffer_rate'],
            'position_multiplier': row['position_multiplier'],
            'rate_multiplier': row['rate_multiplier'],
            'rate_exponent': row['rate_exponent'],
        }
    
    # Add Rollbit parameters to recommendations DataFrame
    for i, row in rec_df.iterrows():
        pair_name = row['pair_name']
        
        if pair_name in rollbit_params:
            # Safely add Rollbit parameters (handling NULL/zero values)
            rec_df.at[i, 'rollbit_buffer_rate'] = rollbit_params[pair_name]['buffer_rate']
            rec_df.at[i, 'rollbit_position_multiplier'] = rollbit_params[pair_name]['position_multiplier']
            rec_df.at[i, 'rollbit_rate_multiplier'] = rollbit_params[pair_name]['rate_multiplier']
            rec_df.at[i, 'rollbit_rate_exponent'] = rollbit_params[pair_name]['rate_exponent']
    
    return rec_df

def render_rollbit_comparison(rollbit_comparison_df):
    """Render the Rollbit comparison tab with improved formatting and error handling for NULL/zero values"""
    
    if rollbit_comparison_df is None or rollbit_comparison_df.empty:
        st.info("No matching pairs found with Rollbit data for comparison.")
        return
    
    # Buffer Rate Table
    st.markdown("<h3>Buffer Rate Comparison</h3>", unsafe_allow_html=True)
    buffer_html = """
    <div class="param-group">
        <div class="param-header">Buffer Rate</div>
        <table class="param-table">
            <tr>
                <th>Pair</th>
                <th>Type</th>
                <th>SURF Buffer</th>
                <th>Rollbit Buffer</th>
                <th>Buffer Ratio</th>
            </tr>
    """
    
    for _, row in rollbit_comparison_df.iterrows():
        # Handle NULL/NaN values to avoid display issues
        surf_buffer = format_percent(row.get('current_buffer_rate'))
        rollbit_buffer = format_percent(row.get('rollbit_buffer_rate'))
        
        # Calculate ratio safely
        if (not check_null_or_zero(row.get('current_buffer_rate')) and 
            not check_null_or_zero(row.get('rollbit_buffer_rate'))):
            buffer_ratio = f"{row['current_buffer_rate']/row['rollbit_buffer_rate']:.2f}x"
        else:
            buffer_ratio = "N/A"
        
        buffer_html += f"""
        <tr>
            <td>{row['pair_name']}</td>
            <td>{row['token_type']}</td>
            <td class="surf-value">{surf_buffer}</td>
            <td class="rollbit-value">{rollbit_buffer}</td>
            <td class="ratio-value">{buffer_ratio}</td>
        </tr>
        """
    
    buffer_html += """
        </table>
    </div>
    """
    
    st.markdown(buffer_html, unsafe_allow_html=True)
    
    # Position Multiplier Table
    st.markdown("<h3>Position Multiplier Comparison</h3>", unsafe_allow_html=True)
    
    position_html = """
    <div class="param-group">
        <div class="param-header">Position Multiplier</div>
        <table class="param-table">
            <tr>
                <th>Pair</th>
                <th>Type</th>
                <th>SURF Position Mult.</th>
                <th>Rollbit Position Mult.</th>
                <th>Position Ratio</th>
            </tr>
    """
    
    for _, row in rollbit_comparison_df.iterrows():
        surf_position = format_number(row.get('current_position_multiplier'))
        rollbit_position = format_number(row.get('rollbit_position_multiplier'))
        
        # Calculate ratio safely
        if (not check_null_or_zero(row.get('current_position_multiplier')) and 
            not check_null_or_zero(row.get('rollbit_position_multiplier'))):
            position_ratio = f"{row['current_position_multiplier']/row['rollbit_position_multiplier']:.2f}x"
        else:
            position_ratio = "N/A"
        
        position_html += f"""
        <tr>
            <td>{row['pair_name']}</td>
            <td>{row['token_type']}</td>
            <td class="surf-value">{surf_position}</td>
            <td class="rollbit-value">{rollbit_position}</td>
            <td class="ratio-value">{position_ratio}</td>
        </tr>
        """
    
    position_html += """
        </table>
    </div>
    """
    
    st.markdown(position_html, unsafe_allow_html=True)
    
    # Rate Multiplier Table
    st.markdown("<h3>Rate Multiplier Comparison</h3>", unsafe_allow_html=True)
    
    rate_mult_html = """
    <div class="param-group">
        <div class="param-header">Rate Multiplier</div>
        <table class="param-table">
            <tr>
                <th>Pair</th>
                <th>Type</th>
                <th>SURF Rate Mult.</th>
                <th>Rollbit Rate Mult.</th>
                <th>Rate Mult. Ratio</th>
            </tr>
    """
    
    for _, row in rollbit_comparison_df.iterrows():
        surf_rate_mult = format_float(row.get('current_rate_multiplier'), 2)
        rollbit_rate_mult = format_float(row.get('rollbit_rate_multiplier'), 2)
        
        # Calculate ratio safely
        if (not check_null_or_zero(row.get('current_rate_multiplier')) and 
            not check_null_or_zero(row.get('rollbit_rate_multiplier'))):
            rate_mult_ratio = f"{row['current_rate_multiplier']/row['rollbit_rate_multiplier']:.2f}x"
        else:
            rate_mult_ratio = "N/A"
        
        rate_mult_html += f"""
        <tr>
            <td>{row['pair_name']}</td>
            <td>{row['token_type']}</td>
            <td class="surf-value">{surf_rate_mult}</td>
            <td class="rollbit-value">{rollbit_rate_mult}</td>
            <td class="ratio-value">{rate_mult_ratio}</td>
        </tr>
        """
    
    rate_mult_html += """
        </table>
    </div>
    """
    
    st.markdown(rate_mult_html, unsafe_allow_html=True)
    
    # Rate Exponent Table
    st.markdown("<h3>Rate Exponent Comparison</h3>", unsafe_allow_html=True)
    
    rate_exp_html = """
    <div class="param-group">
        <div class="param-header">Rate Exponent</div>
        <table class="param-table">
            <tr>
                <th>Pair</th>
                <th>Type</th>
                <th>SURF Rate Exp.</th>
                <th>Rollbit Rate Exp.</th>
                <th>Rate Exp. Ratio</th>
            </tr>
    """
    
    for _, row in rollbit_comparison_df.iterrows():
        surf_rate_exp = format_float(row.get('current_rate_exponent'), 2)
        rollbit_rate_exp = format_float(row.get('rollbit_rate_exponent'), 2)
        
        # Calculate ratio safely
        if (not check_null_or_zero(row.get('current_rate_exponent')) and 
            not check_null_or_zero(row.get('rollbit_rate_exponent'))):
            rate_exp_ratio = f"{row['current_rate_exponent']/row['rollbit_rate_exponent']:.2f}x"
        else:
            rate_exp_ratio = "N/A"
        
        rate_exp_html += f"""
        <tr>
            <td>{row['pair_name']}</td>
            <td>{row['token_type']}</td>
            <td class="surf-value">{surf_rate_exp}</td>
            <td class="rollbit-value">{rollbit_rate_exp}</td>
            <td class="ratio-value">{rate_exp_ratio}</td>
        </tr>
        """
    
    rate_exp_html += """
        </table>
    </div>
    """
    
    st.markdown(rate_exp_html, unsafe_allow_html=True)
    
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
        <p><i>Note: "N/A" is displayed when either SURF or Rollbit has null, zero, or missing values for comparison.</i></p>
    </div>
    """, unsafe_allow_html=True)

def render_market_impact_formula():
    """Display the market impact formula with explanation"""
    st.markdown("""
    <h3>Market Impact Formula</h3>
    <p>
        When users trade on our exchange, large orders impact the closing price for winning trades. 
        This formula replicates the market impact of large taking orders in a consistent and predictable way.
    </p>
    
    <div class="formula-box">
        <div class="formula">
            P_close(T) = P(t) + ((1 - base_rate) / (1 + 1/abs((P(T)/P(t) - 1)*rate_multiplier)^rate_exponent + bet_amount*bet_multiplier/(10^6*abs(P(T)/P(t) - 1)*position_multiplier)))*(P(T) - P(t))
        </div>
    </div>
    
    <h4>Parameter Effects:</h4>
    <ul>
        <li><b>Buffer Rate</b>: Controls the safety margin for positions. Higher values reduce leverage but increase stability.</li>
        <li><b>Position Multiplier</b>: Controls maximum position size. Higher values allow larger positions with less impact.</li>
        <li><b>Rate Multiplier</b>: Controls market impact sensitivity. Lower values increase the market impact.</li>
        <li><b>Rate Exponent</b>: Controls how steeply market impact increases with size. Higher values mean more aggressive scaling.</li>
    </ul>
    
    <p>
        These parameters work together to create a fair and sustainable fee model that accurately represents market conditions.
    </p>
    """, unsafe_allow_html=True)

def render_parameter_simulation():
    """Render interactive parameter simulation"""
    st.markdown("<h3>Parameter Impact Simulation</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <p>This simulator helps visualize how parameter changes affect market impact and closing prices for trades.</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Parameter inputs
        buffer_rate = st.slider("Buffer Rate (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100
        position_multiplier = st.slider("Position Multiplier", min_value=100000, max_value=10000000, value=1000000, step=100000)
    
    with col2:
        rate_multiplier = st.slider("Rate Multiplier", min_value=1.0, max_value=15.0, value=6.0, step=0.5)
        rate_exponent = st.slider("Rate Exponent", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    
    # Additional parameters for simulation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_price = st.number_input("Initial Price (P(t))", min_value=100.0, max_value=100000.0, value=50000.0, step=1000.0)
    
    with col2:
        final_price = st.number_input("Final Price (P(T))", min_value=100.0, max_value=100000.0, value=55000.0, step=1000.0)
    
    with col3:
        bet_amount = st.number_input("Bet Amount", min_value=100, max_value=1000000, value=10000, step=1000)
    
    # Fixed parameters
    base_rate = 0.01  # 1% base rate
    bet_multiplier = 1.0  # Default bet multiplier
    
    # Calculate market impact for different position sizes
    position_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    impacts = []
    
    for amount in position_sizes:
        # Calculate price change percentage
        price_change_pct = abs(final_price/initial_price - 1)
        
        # Calculate denominator parts
        exponent_part = (1/abs(price_change_pct * rate_multiplier))**rate_exponent
        size_part = amount * bet_multiplier / (10**6 * abs(price_change_pct) * position_multiplier)
        
        # Calculate market impact factor
        impact_factor = (1 - base_rate) / (1 + exponent_part + size_part)
        
        # Calculate effective price and impact percentage
        if final_price > initial_price:  # Long position
            effective_price = initial_price + impact_factor * (final_price - initial_price)
            raw_profit_pct = (final_price - initial_price) / initial_price * 100
            effective_profit_pct = (effective_price - initial_price) / initial_price * 100
            impact_pct = (raw_profit_pct - effective_profit_pct) / raw_profit_pct * 100 if raw_profit_pct != 0 else 0
        else:  # Short position
            effective_price = initial_price + impact_factor * (final_price - initial_price)
            raw_profit_pct = (initial_price - final_price) / initial_price * 100
            effective_profit_pct = (initial_price - effective_price) / initial_price * 100
            impact_pct = (raw_profit_pct - effective_profit_pct) / raw_profit_pct * 100 if raw_profit_pct != 0 else 0
        
        impacts.append({
            'Position Size': amount,
            'Effective Price': effective_price,
            'Market Impact': impact_pct,
            'Raw P&L %': raw_profit_pct,
            'Effective P&L %': effective_profit_pct
        })
    
    # Convert to DataFrame for display
    impact_df = pd.DataFrame(impacts)
    
    # Display results
    st.markdown("<h4>Market Impact Simulation Results</h4>", unsafe_allow_html=True)
    
    # Format the DataFrame for display
    formatted_df = pd.DataFrame({
        'Position Size': impact_df['Position Size'].apply(lambda x: f"${x:,}"),
        'Raw P&L %': impact_df['Raw P&L %'].apply(lambda x: f"{x:.2f}%"),
        'Effective P&L %': impact_df['Effective P&L %'].apply(lambda x: f"{x:.2f}%"),
        'Impact %': impact_df['Market Impact'].apply(lambda x: f"{x:.2f}%"),
        'Effective Price': impact_df['Effective Price'].apply(lambda x: f"${x:,.2f}")
    })
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Create visualization of market impact
    fig = go.Figure()
    
    # Add bar chart for market impact percentage
    fig.add_trace(go.Bar(
        x=[f"${size:,}" for size in impact_df['Position Size']],
        y=impact_df['Market Impact'],
        name='Market Impact %',
        marker_color='red'
    ))
    
    # Add line chart for effective P&L percentage
    fig.add_trace(go.Scatter(
        x=[f"${size:,}" for size in impact_df['Position Size']],
        y=impact_df['Effective P&L %'],
        name='Effective P&L %',
        mode='lines+markers',
        marker=dict(size=8, color='blue'),
        yaxis='y2'
    ))
    
    # Update layout with two y-axes
    fig.update_layout(
        title="Market Impact vs. Position Size",
        xaxis_title="Position Size (USD)",
        yaxis=dict(
            title="Market Impact %",
            titlefont=dict(color="red"),
            tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Effective P&L %",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying="y",
            side="right"
        ),
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation of results
    st.markdown("""
    <div class="info-box">
        <h4>How to Interpret Results</h4>
        <p>This simulation shows how different position sizes affect market impact and effective P&L:</p>
        <ul>
            <li><b>Raw P&L %:</b> The unrealized profit percentage without market impact.</li>
            <li><b>Effective P&L %:</b> The realized profit percentage after market impact.</li>
            <li><b>Impact %:</b> The percentage of potential profit reduced by market impact.</li>
            <li><b>Effective Price:</b> The actual closing price after market impact.</li>
        </ul>
        <p>Adjusting the parameters above will show how they influence market impact across different position sizes.</p>
    </div>
    """, unsafe_allow_html=True)

def render_weekly_spread_range(pair_name, min_spread, max_spread, current_spread, avg_spread=None):
    """Render a visual representation of where the current spread is within weekly range"""
    
    # Check if we have all the data needed
    if (check_null_or_zero(min_spread) or check_null_or_zero(max_spread) or 
        check_null_or_zero(current_spread)):
        return f"<div class='info-box'>Weekly spread range data is not available for {pair_name}</div>"
    
    # Calculate position as percentage
    if max_spread > min_spread:
        position_pct = (current_spread - min_spread) / (max_spread - min_spread) * 100
        position_pct = max(0, min(100, position_pct))  # Clamp to [0,100]
    else:
        position_pct = 50  # Default to middle if min==max
    
    # Format the spreads for display (multiplied by 10000 for readability)
    min_spread_fmt = f"{min_spread*10000:.2f}"
    max_spread_fmt = f"{max_spread*10000:.2f}"
    current_spread_fmt = f"{current_spread*10000:.2f}"
    avg_spread_fmt = f"{avg_spread*10000:.2f}" if avg_spread is not None else "N/A"
    
    # Generate the HTML
    html = f"""
    <div class='info-box'>
        <h4>Weekly Spread Range for {pair_name}</h4>
        <div class='spread-range-info'>
            <span class='range-label'>Min: {min_spread_fmt}</span>
            <div class='range-bar'>
                <div class='range-marker' style='left: {position_pct}%;'></div>
            </div>
            <span class='range-label'>Max: {max_spread_fmt}</span>
        </div>
        <div>
            <strong>Current Spread:</strong> {current_spread_fmt} 
            (at {position_pct:.1f}% of weekly range)
        </div>
        <div>
            <strong>Average Spread:</strong> {avg_spread_fmt}
        </div>
    </div>
    """
    
    return html

# --- Main Application ---
def main():
    st.markdown('<div class="header-style">Exchange Parameter Optimization Dashboard</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Controls")

    # Add a refresh button
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

    # Add a reset baselines button
    if st.sidebar.button("Reset Baselines to Current Spreads", use_container_width=True):
        market_data_df = fetch_market_spread_data()
        if market_data_df is not None and not market_data_df.empty:
            success, count, errors = reset_all_baselines(market_data_df)
            if success:
                st.sidebar.success(f"Successfully reset {count} baselines")
                # Clear cache to refresh data
                st.cache_data.clear()
            else:
                st.sidebar.error(f"Failed to reset baselines. {errors} errors occurred.")
        else:
            st.sidebar.error("No market data available to reset baselines")

    # Add adjustment sensitivity controls
    st.sidebar.header("Adjustment Sensitivity")
    buffer_sensitivity = st.sidebar.slider("Buffer Rate Sensitivity", 0.1, 1.0, 0.5, 0.1)
    position_sensitivity = st.sidebar.slider("Position Multiplier Sensitivity", 0.1, 1.0, 0.5, 0.1)
    rate_multiplier_sensitivity = st.sidebar.slider("Rate Multiplier Sensitivity", 0.1, 1.0, 0.5, 0.1)
    rate_exponent_sensitivity = st.sidebar.slider("Rate Exponent Sensitivity", 0.1, 1.0, 0.5, 0.1)

    # Set sensitivities
    sensitivities = {
        'buffer_sensitivity': buffer_sensitivity,
        'position_sensitivity': position_sensitivity,
        'rate_multiplier_sensitivity': rate_multiplier_sensitivity,
        'rate_exponent_sensitivity': rate_exponent_sensitivity
    }

    # Create simple tab navigation
    tab_options = ["Overview", "Parameter Recommendations", "Rollbit Comparison", "Impact Simulation", "Formula Explanation"]
    tab_cols = st.columns(len(tab_options))
    selected_tab = None

    for i, tab in enumerate(tab_options):
        with tab_cols[i]:
            if st.button(tab, key=f"tab_{i}", use_container_width=True):
                selected_tab = tab

    # Default tab if none selected
    if selected_tab is None:
        selected_tab = "Overview"

    st.markdown(f"### {selected_tab}", unsafe_allow_html=True)
    st.markdown("---")

    # Fetch current data
    current_params_df = fetch_current_parameters()
    market_data_df = fetch_market_spread_data()
    weekly_data_df = fetch_weekly_spread_data()
    baselines_df = fetch_spread_baselines()
    rollbit_df = fetch_rollbit_parameters()

    # Generate recommendations
    if current_params_df is not None and market_data_df is not None and baselines_df is not None:
        # Generate recommendations with updated sensitivities and weekly data
        rec_df = generate_recommendations(current_params_df, market_data_df, weekly_data_df, baselines_df, sensitivities)
        
        # Add Rollbit comparison data if available
        if rollbit_df is not None:
            rec_df = add_rollbit_comparison(rec_df, rollbit_df)
        
        # Render the selected tab content
        if selected_tab == "Overview":
            st.markdown("""
            <div class="info-box">
                <h4>Dashboard Overview</h4>
                <p>This dashboard helps optimize trading parameters based on market conditions. Key features:</p>
                <ul>
                    <li><b>Parameter Recommendations</b>: Auto-generated recommendations based on market spread changes and weekly ranges</li>
                    <li><b>Rollbit Comparison</b>: Compare your parameters with Rollbit's settings</li>
                    <li><b>Impact Simulation</b>: Visualize how parameters affect market impact and P&L</li>
                    <li><b>Reset Baselines</b>: Set current market spreads as new baselines (sidebar)</li>
                </ul>
                <p>Use the sensitivity sliders in the sidebar to control how aggressively parameters adapt to market changes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display current status metrics
            if rec_df is not None and not rec_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_pairs = len(rec_df)
                    st.metric("Total Trading Pairs", total_pairs)
                
                with col2:
                    significant_changes = len(rec_df[(abs(rec_df['buffer_change']) > 1.0) | 
                                               (abs(rec_df['position_change']) > 1.0)])
                    st.metric("Pairs Needing Adjustment", significant_changes)
                
                with col3:
                    avg_spread_change = rec_df['spread_change_ratio'].mean()
                    spread_direction = "↑" if avg_spread_change > 1.0 else "↓"
                    st.metric("Avg Spread Change", f"{spread_direction} {abs(avg_spread_change-1)*100:.2f}%")
            
            # Show current parameters for major pairs
            st.markdown("### Current Parameters for Major Tokens", unsafe_allow_html=True)
            
            if current_params_df is not None and not current_params_df.empty:
                major_pairs = current_params_df[current_params_df['pair_name'].apply(is_major)]
                
                if not major_pairs.empty:
                    # Format for display
                    display_df = pd.DataFrame({
                        'Token': major_pairs['pair_name'],
                        'Buffer Rate': major_pairs['buffer_rate'].apply(lambda x: format_percent(x)),
                        'Position Multiplier': major_pairs['position_multiplier'].apply(lambda x: format_number(x)),
                        'Rate Multiplier': major_pairs['rate_multiplier'].apply(lambda x: format_float(x)),
                        'Rate Exponent': major_pairs['rate_exponent'].apply(lambda x: format_float(x))
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No major tokens configured")
            
            # Show baseline vs current spreads
            st.markdown("### Market Spread Analysis", unsafe_allow_html=True)
            
            if baselines_df is not None and market_data_df is not None:
                # Create spread comparison table
                current_spreads = calculate_current_spreads(market_data_df)
                
                spread_data = []
                for _, row in baselines_df.iterrows():
                    pair = row['pair_name']
                    if pair in current_spreads:
                        # Get weekly range data if available
                        min_weekly = None
                        max_weekly = None
                        avg_weekly = None
                        
                        if weekly_data_df is not None and not weekly_data_df.empty:
                            weekly_row = weekly_data_df[weekly_data_df['pair_name'] == pair]
                            if not weekly_row.empty:
                                min_weekly = weekly_row.iloc[0]['min_fee1']
                                max_weekly = weekly_row.iloc[0]['max_fee1']
                                avg_weekly = weekly_row.iloc[0]['avg_fee1']
                        
                        # Calculate position in range
                        range_position = None
                        if min_weekly is not None and max_weekly is not None and min_weekly < max_weekly:
                            range_position = (current_spreads[pair] - min_weekly) / (max_weekly - min_weekly)
                            range_position = max(0, min(1, range_position))  # Clamp to [0,1]
                        
                        spread_data.append({
                            'Token': pair,
                            'Type': 'Major' if is_major(pair) else 'Altcoin',
                            'Baseline Spread': row['baseline_spread'],
                            'Current Spread': current_spreads[pair],
                            'Change Ratio': safe_division(current_spreads[pair], row['baseline_spread'], 1.0),
                            'Change %': (safe_division(current_spreads[pair], row['baseline_spread'], 1.0) - 1) * 100,
                            'Min Weekly': min_weekly,
                            'Max Weekly': max_weekly,
                            'Avg Weekly': avg_weekly,
                            'Range Position': range_position
                        })
                
                if spread_data:
                    spread_df = pd.DataFrame(spread_data)
                    spread_df = spread_df.sort_values('Change %', ascending=False)
                    
                    # Format for display
                    display_df = pd.DataFrame({
                        'Token': spread_df['Token'],
                        'Type': spread_df['Type'],
                        'Baseline Spread': spread_df['Baseline Spread'].apply(lambda x: f"{x*10000:.2f}"),
                        'Current Spread': spread_df['Current Spread'].apply(lambda x: f"{x*10000:.2f}"),
                        'Change': spread_df['Change %'].apply(lambda x: f"{x:+.2f}%"),
                        'Range Position': spread_df['Range Position'].apply(
                            lambda x: f"{x*100:.0f}%" if x is not None else "N/A"
                        )
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show weekly range for a selected token
                    if len(spread_df) > 0:
                        st.markdown("### Weekly Spread Range Analysis", unsafe_allow_html=True)
                        
                        # Token selector
                        selected_token = st.selectbox(
                            "Select a token to view its weekly spread range:",
                            spread_df['Token'].tolist()
                        )
                        
                        # Find the selected token in the data
                        token_data = spread_df[spread_df['Token'] == selected_token].iloc[0]
                        
                        # Render the weekly range visualization
                        weekly_html = render_weekly_spread_range(
                            token_data['Token'],
                            token_data['Min Weekly'],
                            token_data['Max Weekly'],
                            token_data['Current Spread'],
                            token_data['Avg Weekly']
                        )
                        
                        st.markdown(weekly_html, unsafe_allow_html=True)
                    
                    # Create visualization of spread changes
                    fig = px.bar(
                        spread_df.head(15),  # Top 15 for readability
                        x='Token',
                        y='Change %',
                        color='Type',
                        title="Biggest Spread Changes (Baseline vs Current)",
                        color_discrete_map={
                            'Major': '#1976D2',
                            'Altcoin': '#FFA000'
                        }
                    )
                    
                    # Add reference line at 0
                    fig.add_shape(
                        type="line",
                        x0=-0.5, y0=0,
                        x1=len(spread_df.head(15))-0.5, y1=0,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        yaxis_title="Spread Change %"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No spread comparison data available")
            
        elif selected_tab == "Parameter Recommendations":
            if rec_df is not None and not rec_df.empty:
                # Create summary cards
                st.markdown("#### Recommended Parameter Changes")
                
                # Get counts for significant changes
                buffer_increases = len(rec_df[rec_df['buffer_change'] > 1.0])
                buffer_decreases = len(rec_df[rec_df['buffer_change'] < -1.0])
                position_increases = len(rec_df[rec_df['position_change'] > 1.0])
                position_decreases = len(rec_df[rec_df['position_change'] < -1.0])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Buffer Increases", buffer_increases)
                with col2:
                    st.metric("Buffer Decreases", buffer_decreases)
                with col3:
                    st.metric("Position Increases", position_increases)
                with col4:
                    st.metric("Position Decreases", position_decreases)
                
                # Display pairs with significant changes
                st.markdown("#### Pairs with Significant Parameter Changes")
                
                significant_changes = rec_df[
                    (abs(rec_df['buffer_change']) > 1.0) | 
                    (abs(rec_df['position_change']) > 1.0) |
                    (abs(rec_df['rate_multiplier_change']) > 1.0) |
                    (abs(rec_df['rate_exponent_change']) > 1.0)
                ].copy()
                
                if not significant_changes.empty:
                    # Format for display
                    display_df = pd.DataFrame({
                        'Pair': significant_changes['pair_name'],
                        'Type': significant_changes['token_type'],
                        'Current Buffer': significant_changes['current_buffer_rate'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"),
                        'Rec. Buffer': significant_changes['recommended_buffer_rate'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"),
                        'Buffer Δ': significant_changes['buffer_change'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"),
                        'Current Pos.': significant_changes['current_position_multiplier'].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"),
                        'Rec. Pos.': significant_changes['recommended_position_multiplier'].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"),
                        'Pos. Δ': significant_changes['position_change'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show weekly range context for selected pair
                    if len(significant_changes) > 0:
                        selected_pair = st.selectbox(
                            "Select a pair to view its weekly spread context:",
                            significant_changes['pair_name'].tolist()
                        )
                        
                        pair_data = significant_changes[significant_changes['pair_name'] == selected_pair].iloc[0]
                        
                        # Render the weekly range visualization with explanation of its impact on recommendations
                        weekly_html = render_weekly_spread_range(
                            pair_data['pair_name'],
                            pair_data['min_weekly_spread'],
                            pair_data['max_weekly_spread'],
                            pair_data['current_spread'],
                            pair_data['avg_weekly_spread']
                        )
                        
                        st.markdown(weekly_html, unsafe_allow_html=True)
                        
                        # Explain how this affects the recommendation
                        st.markdown("""
                        <div class="info-box">
                            <h4>How Weekly Range Affects Recommendations</h4>
                            <p>When current spread is near the top of the weekly range, parameter changes are amplified because high spreads are more likely temporary market conditions.</p>
                            <p>When current spread is near the bottom of the range, parameter changes are more conservative because low spreads might not persist.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant parameter changes recommended.")
                    
                # Create visualizations
                st.markdown("#### Parameter Change Distribution")
                
                fig = go.Figure()
                
                # Add histogram for buffer rate changes
                fig.add_trace(go.Histogram(
                    x=rec_df['buffer_change'].dropna(),
                    name='Buffer Rate Changes',
                    nbinsx=20,
                    marker_color='blue',
                    opacity=0.7
                ))
                
                # Add histogram for position multiplier changes
                fig.add_trace(go.Histogram(
                    x=rec_df['position_change'].dropna(),
                    name='Position Multiplier Changes',
                    nbinsx=20,
                    marker_color='green',
                    opacity=0.7
                ))
                
                # Add reference line at 0
                fig.add_shape(
                    type="line",
                    x0=0, y0=0,
                    x1=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Update layout
                fig.update_layout(
                    title="Distribution of Parameter Changes",
                    xaxis_title="Percent Change (%)",
                    yaxis_title="Number of Tokens",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed parameter table
                st.markdown("#### Complete Parameter Recommendations")
                
                # Sort by absolute buffer change (descending)
                rec_df['abs_buffer_change'] = rec_df['buffer_change'].abs()
                sorted_df = rec_df.sort_values(by='abs_buffer_change', ascending=False)
                
                # Display detailed table
                display_df = pd.DataFrame({
                    'Pair': sorted_df['pair_name'],
                    'Type': sorted_df['token_type'],
                    'Current Buffer': sorted_df['current_buffer_rate'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"),
                    'Rec. Buffer': sorted_df['recommended_buffer_rate'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"),
                    'Buffer Δ': sorted_df['buffer_change'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"),
                    'Current Pos.': sorted_df['current_position_multiplier'].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"),
                    'Rec. Pos.': sorted_df['recommended_position_multiplier'].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"),
                    'Pos. Δ': sorted_df['position_change'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"),
                    'Rate Mult.': sorted_df['current_rate_multiplier'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"),
                    'Rec. Rate Mult.': sorted_df['recommended_rate_multiplier'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"),
                    'Rate Exp.': sorted_df['current_rate_exponent'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"),
                    'Rec. Rate Exp.': sorted_df['recommended_rate_exponent'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                })
                
                st.dataframe(display_df, use_container_width=True)
                
                # Create scatter plot of spread change ratio vs parameter changes
                st.markdown("#### Spread Change Impact on Parameters")
                
                fig = go.Figure()
                
                # Add scatter plot for buffer rate changes
                fig.add_trace(go.Scatter(
                    x=sorted_df['spread_change_ratio'].dropna(),
                    y=sorted_df['buffer_change'].dropna(),
                    mode='markers',
                    name='Buffer Rate Change',
                    marker=dict(size=8, color='blue', opacity=0.7)
                ))
                
                # Add scatter plot for position multiplier changes
                fig.add_trace(go.Scatter(
                    x=sorted_df['spread_change_ratio'].dropna(),
                    y=sorted_df['position_change'].dropna(),
                    mode='markers',
                    name='Position Multiplier Change',
                    marker=dict(size=8, color='green', opacity=0.7)
                ))
                
                # Add reference lines
                fig.add_shape(
                    type="line",
                    x0=0.5, y0=0,
                    x1=2.0, y1=0,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=1.0, y0=-50,
                    x1=1.0, y1=50,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Update layout
                fig.update_layout(
                    title="Parameter Changes vs Spread Change Ratio",
                    xaxis_title="Spread Change Ratio (Current/Baseline)",
                    yaxis_title="Parameter Change (%)",
                    height=500,
                    xaxis=dict(range=[0.5, 2.0]),
                    yaxis=dict(range=[-50, 50])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recommendation data available. Please check database connection and try refreshing.")
                
        elif selected_tab == "Rollbit Comparison":
            # Filter for pairs with Rollbit data
            if rec_df is not None and not rec_df.empty:
                # Create a copy to avoid any pandas warnings
                rollbit_comparison_df = rec_df.copy()
                
                # Filter rows with Rollbit data
                if 'rollbit_buffer_rate' in rollbit_comparison_df.columns:
                    rollbit_comparison_df = rollbit_comparison_df.dropna(subset=['rollbit_buffer_rate', 'rollbit_position_multiplier'], how='all')
                    
                    # Use the rendering function
                    render_rollbit_comparison(rollbit_comparison_df)
                else:
                    st.warning("Rollbit comparison data not found in the recommendations. Make sure Rollbit data is being fetched correctly.")
            else:
                st.warning("No data available for Rollbit comparison. Please check database connection and try refreshing.")
                
        elif selected_tab == "Impact Simulation":
            render_parameter_simulation()
            
        elif selected_tab == "Formula Explanation":
            render_market_impact_formula()
    else:
        st.error("Failed to load required data. Please check database connection and try refreshing.")

    # Add footer with explanatory information
    with st.expander("Understanding Parameter Optimization"):
        st.markdown("""
        ### About This Dashboard
        
        This dashboard helps optimize trading parameters based on market conditions:
        
        ### Key Parameters
        
        - **Buffer Rate**: Percentage of the position that must be maintained as margin for safety.
          - When spreads increase, buffer rate should increase to account for higher volatility risk.
          
        - **Position Multiplier**: Factor that determines the maximum position size per unit of margin.
          - When spreads increase, position multiplier should decrease to limit exposure.
          
        - **Rate Multiplier**: Factor that affects the market impact (funding rate) for position sizes.
          - When spreads increase, rate multiplier typically decreases to avoid excessive funding rates.
          
        - **Rate Exponent**: Exponent that determines how quickly market impact increases with position size.
          - When spreads increase, rate exponent typically increases to more aggressively limit large positions.
        
        ### Methodology
        
        1. The dashboard compares current market spreads with stored baseline spreads
        2. Weekly spread ranges are used to contextualize how significant current spreads are
        3. Parameter adjustments are proportional to the relative change in spread and where it falls in the weekly range
        4. Sensitivity controls adjust how aggressively parameters respond to spread changes
        5. Rollbit parameters are shown for comparison with a major competitor
        
        ### Weekly Spread Range
        
        The dashboard now shows where the current spread falls within the weekly range (min to max):
        - If current spread is near the weekly maximum, parameter changes are amplified
        - If current spread is near the weekly minimum, parameter changes are more conservative
        - This helps avoid overreacting to temporary market conditions
        
        ### Market Impact Formula
        
        ```
        P_close(T) = P(t) + ((1 - base_rate) / (1 + 1/abs((P(T)/P(t) - 1)*rate_multiplier)^rate_exponent + bet_amount*bet_multiplier/(10^6*abs(P(T)/P(t) - 1)*position_multiplier)))*(P(T) - P(t))
        ```
        
        Where:
        - P(t) is the opening price
        - P(T) is the market price at close time
        - rate_multiplier, rate_exponent, and position_multiplier are the parameters we optimize
        
        ### Interpretation
        
        - Parameters with significant recommended changes may need manual adjustment
        - The scatter plot shows how spread changes correlate with parameter recommendations
        - Consider weekly spread range context when evaluating parameter changes
        - Consider both market conditions and competitor settings when finalizing parameters
        - Use the Impact Simulation to visualize exactly how your settings affect winning trade prices
        """)

if __name__ == "__main__":
    main()