import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import traceback
import json
import os

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
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #d32f2f;
    }
    .success-message {
        color: #2e7d32;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #2e7d32;
    }
    .confirm-button {
        background-color: #f44336;
        color: white;
        font-weight: bold;
    }
    .action-button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
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

# --- Session State Management ---
def init_session_state():
    """Initialize session state variables"""
    if 'backup_params' not in st.session_state:
        st.session_state.backup_params = None
    if 'has_applied_recommendations' not in st.session_state:
        st.session_state.has_applied_recommendations = False
    if 'show_confirm_dialog' not in st.session_state:
        st.session_state.show_confirm_dialog = False

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
            df['max_leverage'] = df['max_leverage'].fillna(100)
            
            # Note: we'll keep buffer_rate as NaN if it's actually NULL or 0
            # This way we can distinguish between actual values and missing values
            
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
        SELECT * 
        FROM rollbit_pair_config 
        WHERE created_at = (SELECT max(created_at) FROM rollbit_pair_config)
        """
        
        df = pd.read_sql(query, engine)
        
        # Ensure we have the required columns and rename if needed
        if not df.empty:
            # Ensure we have bust_buffer to use as buffer_rate
            if 'bust_buffer' in df.columns and 'buffer_rate' not in df.columns:
                df['buffer_rate'] = df['bust_buffer']
            
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
        
        # Use SQLAlchemy text() for parameterized queries
        query = text("""
        INSERT INTO spread_baselines (pair_name, baseline_spread, updated_at)
        VALUES (:pair_name, :baseline_spread, :updated_at)
        ON CONFLICT (pair_name) DO UPDATE 
        SET baseline_spread = EXCLUDED.baseline_spread, 
            updated_at = EXCLUDED.updated_at
        """)
        
        # Execute with parameters
        with engine.connect() as conn:
            conn.execute(
                query, 
                {"pair_name": pair_name, "baseline_spread": baseline_spread, "updated_at": datetime.now()}
            )
            conn.commit()
            
        return True
    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"Error saving baseline spread for {pair_name}: {e}\n\nDetails: {error_details}")
        return False

def apply_parameter_recommendations(recommendations_df):
    """Apply recommended parameters to the database"""
    if recommendations_df is None or recommendations_df.empty:
        return False, "No recommendations to apply"
    
    try:
        engine = init_connection()
        if not engine:
            return False, "Database connection error"
        
        # Create a backup of current parameters before applying changes
        backup_params = {}
        
        # Apply buffer rate and position multiplier recommendations
        success_count = 0
        error_count = 0
        errors = []
        
        for _, row in recommendations_df.iterrows():
            pair_name = row['pair_name']
            current_buffer = row['current_buffer_rate']
            current_position = row['current_position_multiplier']
            buffer_change = row['buffer_change']
            position_change = row['position_change']
            
            # Store current values for backup
            backup_params[pair_name] = {
                'buffer_rate': current_buffer,
                'position_multiplier': current_position
            }
            
            # Get recommended values
            buffer_rate = row['recommended_buffer_rate']
            position_multiplier = row['recommended_position_multiplier']
            
            # Skip rows with null values or where recommended is same as current
            if check_null_or_zero(buffer_rate) or pd.isna(current_buffer):
                continue
                
            try:
                # Update query with parameter binding for security
                query = text("""
                UPDATE public.trade_pool_pairs
                SET 
                    buffer_rate = :buffer_rate,
                    position_multiplier = :position_multiplier,
                    updated_at = :updated_at
                WHERE 
                    pair_name = :pair_name
                """)
                
                # Execute with parameters
                with engine.connect() as conn:
                    conn.execute(
                        query, 
                        {
                            "buffer_rate": buffer_rate,
                            "position_multiplier": position_multiplier,
                            "updated_at": datetime.now(),
                            "pair_name": pair_name
                        }
                    )
                    conn.commit()
                    
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"Error updating {pair_name}: {str(e)}")
        
        # Save backup to session state
        st.session_state.backup_params = backup_params
        st.session_state.has_applied_recommendations = True
        
        # Clear cache to refresh data
        st.cache_data.clear()
        
        if error_count > 0:
            error_message = "\n".join(errors[:5])
            if len(errors) > 5:
                error_message += f"\n...and {len(errors) - 5} more errors"
            return success_count > 0, f"Applied {success_count} recommendations with {error_count} errors.\n{error_message}"
        else:
            return True, f"Successfully applied {success_count} recommendations"
            
    except Exception as e:
        return False, f"Error applying recommendations: {str(e)}"

def undo_parameter_changes():
    """Undo the most recent parameter changes"""
    if not st.session_state.backup_params:
        return False, "No backup parameters available to restore"
    
    try:
        engine = init_connection()
        if not engine:
            return False, "Database connection error"
        
        success_count = 0
        error_count = 0
        errors = []
        
        for pair_name, params in st.session_state.backup_params.items():
            buffer_rate = params['buffer_rate']
            position_multiplier = params['position_multiplier']
            
            # Skip rows with null values
            if check_null_or_zero(buffer_rate) and check_null_or_zero(position_multiplier):
                continue
                
            try:
                # Update query with parameter binding for security
                query = text("""
                UPDATE public.trade_pool_pairs
                SET 
                    buffer_rate = :buffer_rate,
                    position_multiplier = :position_multiplier,
                    updated_at = :updated_at
                WHERE 
                    pair_name = :pair_name
                """)
                
                # Execute with parameters
                with engine.connect() as conn:
                    conn.execute(
                        query, 
                        {
                            "buffer_rate": buffer_rate if not pd.isna(buffer_rate) else None,
                            "position_multiplier": position_multiplier if not pd.isna(position_multiplier) else None,
                            "updated_at": datetime.now(),
                            "pair_name": pair_name
                        }
                    )
                    conn.commit()
                    
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"Error restoring {pair_name}: {str(e)}")
        
        # Reset flags 
        st.session_state.has_applied_recommendations = False
        st.session_state.backup_params = None
        
        # Clear cache to refresh data
        st.cache_data.clear()
        
        if error_count > 0:
            error_message = "\n".join(errors[:5])
            if len(errors) > 5:
                error_message += f"\n...and {len(errors) - 5} more errors"
            return success_count > 0, f"Restored {success_count} parameters with {error_count} errors.\n{error_message}"
        else:
            return True, f"Successfully restored {success_count} parameters to their previous values"
            
    except Exception as e:
        return False, f"Error restoring parameters: {str(e)}"

def reset_all_baselines(market_data_df):
    """Reset all baselines to current market spreads"""
    if market_data_df is None or market_data_df.empty:
        return False, 0, 0
    
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

def calculate_recommended_params(current_params, current_spread, baseline_spread, 
                               sensitivities, significant_change_threshold=0.05):
    """Calculate recommended parameter values based on spread change ratio"""
    
    # Handle cases with missing data
    if current_spread is None or baseline_spread is None or baseline_spread <= 0:
        return {
            'buffer_rate': current_params.get('buffer_rate'),
            'position_multiplier': current_params.get('position_multiplier')
        }
    
    # Get current parameter values
    current_buffer_rate = current_params.get('buffer_rate')
    current_position_multiplier = current_params.get('position_multiplier')
    max_leverage = current_params.get('max_leverage', 100)
    
    # If current buffer rate is N/A or zero, return N/A for recommended buffer rate
    if check_null_or_zero(current_buffer_rate) or pd.isna(current_buffer_rate):
        return {
            'buffer_rate': None,
            'position_multiplier': current_position_multiplier
        }
    
    # If current position multiplier is N/A or zero, use a default but don't recommend changes
    if check_null_or_zero(current_position_multiplier) or pd.isna(current_position_multiplier):
        return {
            'buffer_rate': current_buffer_rate,
            'position_multiplier': None
        }
    
    # Get sensitivity parameters
    buffer_sensitivity = sensitivities.get('buffer_sensitivity', 0.5)
    position_sensitivity = sensitivities.get('position_sensitivity', 0.5)
    
    # Calculate spread change ratio (how current spread compares to baseline)
    spread_change_ratio = current_spread / baseline_spread
    
    # Check if the change is significant
    if abs(spread_change_ratio - 1.0) < significant_change_threshold:
        return {
            'buffer_rate': current_buffer_rate,
            'position_multiplier': current_position_multiplier
        }
    
    # ========== CORE PARAMETER ADJUSTMENT EQUATIONS ==========
    
    # 1. Buffer Rate (direct relationship):
    # When spreads increase, buffer rate increases
    recommended_buffer_rate = current_buffer_rate * (spread_change_ratio ** buffer_sensitivity)
    
    # 2. Position Multiplier (inverse relationship):
    # When spreads increase, position multiplier decreases
    recommended_position_multiplier = current_position_multiplier / (spread_change_ratio ** position_sensitivity)
    
    # Apply bounds based on the provided constraints
    # Buffer rate: 0 to 70% of 1/max_leverage
    buffer_upper_bound = 0.7 / max_leverage if max_leverage > 0 else 0.007
    recommended_buffer_rate = max(0.0, min(buffer_upper_bound, recommended_buffer_rate))
    
    # Position multiplier: 1 to 15000
    recommended_position_multiplier = max(1, min(15000, recommended_position_multiplier))
    
    return {
        'buffer_rate': recommended_buffer_rate,
        'position_multiplier': recommended_position_multiplier
    }

def generate_recommendations(current_params_df, market_data_df, baselines_df, sensitivities):
    """Generate parameter recommendations based on market data"""
    if current_params_df is None or market_data_df is None or baselines_df is None:
        return None
    
    # Convert DataFrames to more convenient formats
    current_params = {}
    for _, row in current_params_df.iterrows():
        pair_name = row['pair_name']
        current_params[pair_name] = {
            'buffer_rate': row['buffer_rate'],
            'position_multiplier': row['position_multiplier'],
            'max_leverage': row.get('max_leverage', 100)
        }
    
    # Calculate current spreads
    current_spreads = calculate_current_spreads(market_data_df)
    
    # Create baselines dictionary
    baselines = {}
    for _, row in baselines_df.iterrows():
        baselines[row['pair_name']] = row['baseline_spread']
    
    # Create recommendations DataFrame
    recommendations = []
    
    significant_change_threshold = 0.05  # 5% change threshold
    
    for pair, params in current_params.items():
        if pair in current_spreads and pair in baselines:
            current_spread = current_spreads[pair]
            baseline_spread = baselines[pair]
            
            recommended = calculate_recommended_params(
                params, 
                current_spread, 
                baseline_spread,
                sensitivities,
                significant_change_threshold
            )
            
            # Calculate changes with safety checks
            buffer_change = 0
            if (not check_null_or_zero(params['buffer_rate']) and 
                not check_null_or_zero(recommended['buffer_rate']) and
                not pd.isna(params['buffer_rate']) and 
                not pd.isna(recommended['buffer_rate'])):
                buffer_change = ((recommended['buffer_rate'] - params['buffer_rate']) / params['buffer_rate']) * 100
            
            position_change = 0
            if (not check_null_or_zero(params['position_multiplier']) and 
                not check_null_or_zero(recommended['position_multiplier']) and
                not pd.isna(params['position_multiplier']) and 
                not pd.isna(recommended['position_multiplier'])):
                # Only show position change if the rounded values are different
                if round(recommended['position_multiplier']) != round(params['position_multiplier']):
                    position_change = ((recommended['position_multiplier'] - params['position_multiplier']) / params['position_multiplier']) * 100
            
            # Calculate spread change percentage
            spread_change_pct = 0
            if current_spread > 0 and baseline_spread > 0:
                spread_change_pct = ((current_spread / baseline_spread) - 1) * 100
            
            recommendations.append({
                'pair_name': pair,
                'token_type': 'Major' if is_major(pair) else 'Altcoin',
                'current_buffer_rate': params['buffer_rate'],
                'recommended_buffer_rate': recommended['buffer_rate'],
                'buffer_change': buffer_change,
                'current_position_multiplier': params['position_multiplier'],
                'recommended_position_multiplier': recommended['position_multiplier'],
                'position_change': position_change,
                'current_spread': current_spread,
                'baseline_spread': baseline_spread,
                'spread_change_ratio': safe_division(current_spread, baseline_spread, 1.0),
                'spread_change_pct': spread_change_pct,
                'max_leverage': params['max_leverage']
            })
    
    return pd.DataFrame(recommendations)

def add_rollbit_comparison(rec_df, rollbit_df):
    """Add Rollbit parameter data to recommendations DataFrame for comparison"""
    if rec_df is None or rollbit_df is None or rec_df.empty or rollbit_df.empty:
        return rec_df
    
    # Create mapping of Rollbit parameters
    rollbit_params = {}
    for _, row in rollbit_df.iterrows():
        pair_name = row['pair_name']
        
        # Extract the parameters, handling potential column name differences
        buffer_rate = row.get('buffer_rate', row.get('bust_buffer', np.nan))
        position_multiplier = row.get('position_multiplier', np.nan)
        
        rollbit_params[pair_name] = {
            'buffer_rate': buffer_rate,
            'position_multiplier': position_multiplier
        }
    
    # Add Rollbit parameters to recommendations DataFrame
    for i, row in rec_df.iterrows():
        pair_name = row['pair_name']
        
        if pair_name in rollbit_params:
            # Safely add Rollbit parameters (handling NULL/zero values)
            rec_df.at[i, 'rollbit_buffer_rate'] = rollbit_params[pair_name]['buffer_rate']
            rec_df.at[i, 'rollbit_position_multiplier'] = rollbit_params[pair_name]['position_multiplier']
    
    return rec_df

def render_complete_parameter_table(rec_df, sort_by="pair_name"):
    """Render the complete parameter table with all pairs"""
    
    if rec_df is None or rec_df.empty:
        st.warning("No parameter data available.")
        return
    
    # Sort the DataFrame based on sort option
    if sort_by == "pair_name":
        sorted_df = rec_df.sort_values("pair_name")
    elif sort_by == "buffer_change":
        sorted_df = rec_df.sort_values("buffer_change", ascending=False)
    elif sort_by == "position_change":
        sorted_df = rec_df.sort_values("position_change", ascending=False)
    elif sort_by == "spread_change_ratio":
        sorted_df = rec_df.sort_values("spread_change_pct", ascending=False)
    else:
        sorted_df = rec_df
    
    # Highlight significant changes
    def highlight_changes(val):
        """Highlight significant changes in the parameters"""
        if isinstance(val, str) and "%" in val:
            try:
                num_val = float(val.strip('%').replace('+', '').replace('-', ''))
                if num_val > 5.0:
                    return 'background-color: #ffcccc'  # Red for significant changes
                elif num_val > 2.0:
                    return 'background-color: #ffffcc'  # Yellow for moderate changes
            except:
                pass
        return ''
    
    # Create a formatted DataFrame for display
    display_df = pd.DataFrame({
        'Pair': sorted_df['pair_name'],
        'Type': sorted_df['token_type'],
        'Current Spread': sorted_df['current_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Baseline Spread': sorted_df['baseline_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Spread Change': sorted_df['spread_change_pct'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Current Buffer': sorted_df['current_buffer_rate'].apply(
            lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Recommended Buffer': sorted_df.apply(
            lambda row: f"{row['recommended_buffer_rate']*100:.2f}%" 
            if not pd.isna(row['recommended_buffer_rate']) and not pd.isna(row['current_buffer_rate']) 
            else "N/A", 
            axis=1
        ),
        'Buffer Change': sorted_df['buffer_change'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) and abs(x) > 0.01 else "±0.00%"
        ),
        'Current Position': sorted_df['current_position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Recommended Position': sorted_df.apply(
            lambda row: f"{row['recommended_position_multiplier']:,.0f}" 
            if not pd.isna(row['recommended_position_multiplier']) and not pd.isna(row['current_position_multiplier']) 
            else "N/A", 
            axis=1
        ),
        'Position Change': sorted_df['position_change'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) and abs(x) > 0.01 else "±0.00%"
        ),
        'Max Leverage': sorted_df['max_leverage'].apply(
            lambda x: f"{x:.0f}x" if not pd.isna(x) else "N/A"
        )
    })
    
    # Display with highlighting
    st.dataframe(
        display_df.style.applymap(highlight_changes, subset=['Buffer Change', 'Position Change', 'Spread Change']),
        use_container_width=True
    )
    
    # Add a color legend below the table
    st.markdown("""
    <div style="margin-top: 10px; font-size: 0.8em;">
        <span style="background-color: #ffcccc; padding: 3px 8px;">Red</span>: Major adjustment needed (>5%)
        <span style="margin-left: 15px; background-color: #ffffcc; padding: 3px 8px;">Yellow</span>: Moderate adjustment needed (>2%)
    </div>
    """, unsafe_allow_html=True)

def render_significant_changes_summary(rec_df):
    """Render a summary of pairs with significant parameter changes"""
    
    if rec_df is None or rec_df.empty:
        return
    
    # Filter pairs with significant changes (either buffer or position)
    significant_df = rec_df[
        (abs(rec_df['buffer_change']) > 2.0) | 
        (abs(rec_df['position_change']) > 2.0)
    ].copy()
    
    if significant_df.empty:
        st.info("No pairs have significant parameter changes at this time.")
        return
    
    # Sort by most significant changes (using absolute buffer change as primary sort)
    significant_df['abs_buffer_change'] = significant_df['buffer_change'].abs()
    significant_df = significant_df.sort_values('abs_buffer_change', ascending=False)
    
    # Display summary table
    st.markdown("### Pairs Requiring Adjustment")
    
    # Create a formatted DataFrame for display
    display_df = pd.DataFrame({
        'Pair': significant_df['pair_name'],
        'Current Spread': significant_df['current_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Spread Change': significant_df['spread_change_pct'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Current Buffer': significant_df['current_buffer_rate'].apply(
            lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Recommended Buffer': significant_df.apply(
            lambda row: f"{row['recommended_buffer_rate']*100:.2f}%" 
            if not pd.isna(row['recommended_buffer_rate']) and not pd.isna(row['current_buffer_rate']) 
            else "N/A", 
            axis=1
        ),
        'Buffer Change': significant_df['buffer_change'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) and abs(x) > 0.01 else "±0.00%"
        ),
        'Current Position': significant_df['current_position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Recommended Position': significant_df.apply(
            lambda row: f"{row['recommended_position_multiplier']:,.0f}" 
            if not pd.isna(row['recommended_position_multiplier']) and not pd.isna(row['current_position_multiplier']) 
            else "N/A", 
            axis=1
        ),
        'Position Change': significant_df['position_change'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) and abs(x) > 0.01 else "±0.00%"
        ),
        'Max Leverage': significant_df['max_leverage'].apply(
            lambda x: f"{x:.0f}x" if not pd.isna(x) else "N/A"
        )
    })
    
    st.dataframe(display_df, use_container_width=True)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_significant = len(significant_df)
        st.metric("Pairs Needing Adjustment", total_significant)
    
    with col2:
        buffer_increases = len(significant_df[significant_df['buffer_change'] > 2.0])
        st.metric("Buffer Increases", buffer_increases)
    
    with col3:
        buffer_decreases = len(significant_df[significant_df['buffer_change'] < -2.0])
        st.metric("Buffer Decreases", buffer_decreases)
    
    with col4:
        position_changes = len(significant_df[abs(significant_df['position_change']) > 2.0])
        st.metric("Position Changes", position_changes)

def render_rollbit_comparison(comparison_df):
    """Render the Rollbit comparison tab"""
    
    if comparison_df is None or comparison_df.empty:
        st.info("No matching pairs found with Rollbit data for comparison.")
        return
    
    # Filter to pairs that have Rollbit data
    rollbit_df = comparison_df.dropna(subset=['rollbit_buffer_rate', 'rollbit_position_multiplier'])
    
    if rollbit_df.empty:
        st.info("No matching pairs found with Rollbit data for comparison.")
        return
    
    # Display parameter comparison
    st.markdown("### Buffer Rate Comparison")
    
    # Create buffer rate comparison table
    buffer_df = pd.DataFrame({
        'Pair': rollbit_df['pair_name'],
        'Type': rollbit_df['token_type'],
        'SURF Buffer': rollbit_df['current_buffer_rate'].apply(
            lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Rollbit Buffer': rollbit_df['rollbit_buffer_rate'].apply(
            lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
        )
    })
    
    # Add buffer ratio column
    buffer_ratio = []
    for _, row in rollbit_df.iterrows():
        if (not check_null_or_zero(row.get('current_buffer_rate')) and 
            not check_null_or_zero(row.get('rollbit_buffer_rate'))):
            buffer_ratio.append(f"{row['current_buffer_rate']/row['rollbit_buffer_rate']:.2f}x")
        else:
            buffer_ratio.append("N/A")
    
    buffer_df['Buffer Ratio'] = buffer_ratio
    
    # Display buffer rate comparison
    st.dataframe(buffer_df, use_container_width=True)
    
    # Position Multiplier Comparison
    st.markdown("### Position Multiplier Comparison")
    
    # Create position multiplier comparison table
    position_df = pd.DataFrame({
        'Pair': rollbit_df['pair_name'],
        'Type': rollbit_df['token_type'],
        'SURF Position Mult.': rollbit_df['current_position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Rollbit Position Mult.': rollbit_df['rollbit_position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        )
    })
    
    # Add position ratio column
    position_ratio = []
    for _, row in rollbit_df.iterrows():
        if (not check_null_or_zero(row.get('current_position_multiplier')) and 
            not check_null_or_zero(row.get('rollbit_position_multiplier'))):
            position_ratio.append(f"{row['current_position_multiplier']/row['rollbit_position_multiplier']:.2f}x")
        else:
            position_ratio.append("N/A")
    
    position_df['Position Ratio'] = position_ratio
    
    # Display position multiplier comparison
    st.dataframe(position_df, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Understanding the Comparison
    
    This tab compares SURF's current parameters with Rollbit's parameters for matching tokens:
    
    - **Buffer Ratio**: SURF buffer rate ÷ Rollbit buffer rate. Values > 1 mean SURF is more conservative.
    - **Position Ratio**: SURF position multiplier ÷ Rollbit position multiplier. Values > 1 mean SURF allows larger positions.
    
    *Note: "N/A" is displayed when either SURF or Rollbit has null, zero, or missing values for comparison.*
    """)

def render_overview():
    """Render the overview tab with explanations"""
    
    st.markdown("### Dashboard Overview")
    
    st.markdown("""
    This dashboard helps optimize trading parameters based on market conditions:
    
    ### Key Parameters
    
    - **Buffer Rate**: Percentage of the position that must be maintained as margin for safety.
      - When spreads increase, buffer rate should increase to account for higher volatility risk.
      - When spreads decrease, buffer rate should decrease accordingly.
      - Parameter bounds: 0 to 70% of 1/max_leverage
      
    - **Position Multiplier**: Factor that determines the maximum position size per unit of margin.
      - When spreads increase, position multiplier should decrease to limit exposure.
      - When spreads decrease, position multiplier should increase accordingly.
      - Parameter bounds: 1 to 15,000
    
    ### Parameter Adjustment Equations
    
    - Buffer Rate = Current Buffer Rate × (Spread Change Ratio ^ Buffer Sensitivity)
    - Position Multiplier = Current Position Multiplier ÷ (Spread Change Ratio ^ Position Sensitivity)
    
    Where Spread Change Ratio = Current Spread ÷ Baseline Spread
    
    ### Dashboard Controls
    
    - **Refresh Data**: Updates current spreads and parameters from the database
    - **Reset Baselines**: Sets current market spreads as new baselines
    - **Apply Recommendations**: Updates database with recommended parameters
    - **Undo Latest Changes**: Reverts parameters to values before last applied recommendations
    
    ### Methodology
    
    1. The dashboard compares current market spreads with stored baseline spreads
    2. Parameter adjustments are proportional to the relative change in spread
    3. Sensitivity controls adjust how aggressively parameters respond to spread changes
    4. Rollbit parameters are shown for comparison with a major competitor
    
    ### Market Impact Formula
    
    ```
    P_close(T) = P(t) + ((1 - base_rate) / (1 + 1/abs((P(T)/P(t) - 1)*rate_multiplier)^rate_exponent + bet_amount*bet_multiplier/(10^6*abs(P(T)/P(t) - 1)*position_multiplier)))*(P(T) - P(t))
    ```
    
    Where:
    - P(t) is the opening price
    - P(T) is the market price at close time
    - position_multiplier is the parameter we optimize
    """)

# --- Main Application ---
def main():
    # Initialize session state
    init_session_state()
    
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

    # Set sensitivities
    sensitivities = {
        'buffer_sensitivity': buffer_sensitivity,
        'position_sensitivity': position_sensitivity
    }

    # Create simplified tab navigation
    tabs = st.tabs(["Complete Parameter Table", "Rollbit Comparison", "Overview"])
    
    # Fetch data
    current_params_df = fetch_current_parameters()
    market_data_df = fetch_market_spread_data()
    baselines_df = fetch_spread_baselines()
    rollbit_df = fetch_rollbit_parameters()

    # Generate recommendations
    if current_params_df is not None and market_data_df is not None and baselines_df is not None:
        # Generate recommendations with the selected sensitivities
        rec_df = generate_recommendations(current_params_df, market_data_df, baselines_df, sensitivities)
        
        # Add Rollbit comparison data if available
        if rollbit_df is not None:
            rec_df = add_rollbit_comparison(rec_df, rollbit_df)
        
        # Render the appropriate tab content
        with tabs[0]:  # Complete Parameter Table
            # Add sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=[
                    "Pair Name", 
                    "Buffer Change", 
                    "Position Change", 
                    "Spread Change Ratio"
                ],
                index=0
            )
            
            # Add recommendation application and undo buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                apply_button = st.button("Apply Recommendations", key="apply_button", use_container_width=True)
                
                if apply_button:
                    st.session_state.show_confirm_dialog = True
            
            with col2:
                # Always show the Undo button if we have backup params, regardless of has_applied_recommendations flag
                if st.session_state.backup_params:
                    if st.button("Undo Latest Changes", key="undo_button", use_container_width=True):
                        success, message = undo_parameter_changes()
                        if success:
                            st.success(message)
                            # Refresh data
                            st.cache_data.clear()
                            st.experimental_rerun()
                        else:
                            st.error(message)
            
            # Show confirmation dialog if needed
            if st.session_state.show_confirm_dialog:
                st.warning("Are you sure you want to apply all recommendations to the database?")
                confirm_col1, confirm_col2 = st.columns([1, 1])
                
                with confirm_col1:
                    if st.button("Yes, Apply Changes", key="confirm_yes"):
                        success, message = apply_parameter_recommendations(rec_df)
                        if success:
                            st.success(message)
                            # Reset confirmation flag
                            st.session_state.show_confirm_dialog = False
                            # Refresh data
                            st.cache_data.clear()
                            st.experimental_rerun()
                        else:
                            st.error(message)
                
                with confirm_col2:
                    if st.button("No, Cancel", key="confirm_no"):
                        st.session_state.show_confirm_dialog = False
                        st.experimental_rerun()
            
            # Show pairs requiring adjustment first
            render_significant_changes_summary(rec_df)
            
            # Show complete parameter comparison table
            st.markdown("### Complete Parameter Comparison Table")
            render_complete_parameter_table(rec_df, sort_by)
            
            # Add info about parameter constraints
            st.markdown("""
            ### Parameter Constraints
            - **Buffer Rate**: Values are constrained between 0 and 70% of 1/max_leverage
            - **Position Multiplier**: Values are constrained between 1 and 15,000
            """)
            
        with tabs[1]:  # Rollbit Comparison
            render_rollbit_comparison(rec_df)
            
        with tabs[2]:  # Overview
            render_overview()
            
    else:
        st.error("Failed to load required data. Please check database connection and try refreshing.")

if __name__ == "__main__":
    main()