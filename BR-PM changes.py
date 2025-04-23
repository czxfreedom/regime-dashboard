import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

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
</style>
""", unsafe_allow_html=True)

# --- Database Configuration ---
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
        
        # Provide manual connection option
        st.sidebar.header("Database Connection")
        db_user = st.sidebar.text_input("Database Username")
        db_password = st.sidebar.text_input("Database Password", type="password")
        db_host = st.sidebar.text_input("Database Host")
        db_port = st.sidebar.text_input("Database Port", "5432")
        db_name = st.sidebar.text_input("Database Name")
        
        if st.sidebar.button("Connect to Database"):
            try:
                db_uri = (
                    f"postgresql+psycopg2://{db_user}:{db_password}"
                    f"@{db_host}:{db_port}/{db_name}"
                )
                return create_engine(db_uri)
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {e}")
                return None
        return None

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

# Use a fixed scale factor for consistency
scale_factor = 10000
scale_label = "× 10,000"

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
        import json
        config = json.loads(leverage_config)
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

# --- Data Fetching Functions ---
@st.cache_data(ttl=600)
def fetch_all_tokens(engine):
    """Fetch all active tokens from the database"""
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_pool_pairs
    WHERE status = 1
    ORDER BY pair_name
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_daily_spread_averages(engine, tokens):
    """Fetch daily spread averages for multiple tokens (last 24 hours)"""
    try:
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
def fetch_current_parameters(engine):
    """Fetch current trading parameters from the database"""
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
    
    try:
        df = pd.read_sql(query, engine)
        
        # Add max_leverage column from leverage_config if it doesn't exist
        if 'max_leverage' not in df.columns:
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

def calculate_recommended_params(current_buffer_rate, current_position_multiplier, current_spread, baseline_spread, max_leverage):
    """
    Calculate recommended buffer rate and position multiplier based on spread changes
    
    Args:
        current_buffer_rate: Current buffer rate in the system
        current_position_multiplier: Current position multiplier in the system
        current_spread: Current non-Surf average spread
        baseline_spread: Baseline non-Surf average spread to compare against
        max_leverage: Maximum leverage allowed for this trading pair
        
    Returns:
        Tuple of (recommended_buffer_rate, recommended_position_multiplier)
    """
    if current_spread is None or baseline_spread is None or baseline_spread <= 0:
        return current_buffer_rate, current_position_multiplier
    
    # Calculate relative change in spread compared to baseline
    spread_change_ratio = current_spread / baseline_spread
    
    # Only make changes if the spread has changed significantly
    significant_change_threshold = 0.05  # 5% change required before parameter adjustments
    
    if abs(spread_change_ratio - 1.0) < significant_change_threshold:
        return current_buffer_rate, current_position_multiplier
    
    # Sensitivity parameters
    buffer_sensitivity = 0.5  # How sensitive buffer rate is to spread changes
    position_sensitivity = 0.5  # How sensitive position multiplier is to spread changes
    
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

def create_baseline_spreads(engine, spread_data):
    """
    Create or update baseline spreads in the database
    """
    try:
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS spread_baselines (
            pair_name VARCHAR(50) PRIMARY KEY,
            baseline_spread FLOAT NOT NULL,
            updated_at TIMESTAMP NOT NULL
        );
        """
        with engine.connect() as connection:
            connection.execute(create_table_query)
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
                        connection.execute(upsert_query)
                        connection.commit()
                    success_count += 1
                except Exception as e:
                    print(f"Error saving baseline for {pair_name}: {e}")
                    error_count += 1
        
        return f"Successfully reset {success_count} baselines" + (f", {error_count} errors" if error_count > 0 else "")
    except Exception as e:
        return f"Error creating baseline spreads: {e}"

@st.cache_data(ttl=600)
def get_baseline_spreads(engine):
    """Get baseline spreads from the database"""
    query = "SELECT pair_name, baseline_spread, updated_at FROM spread_baselines"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching baseline spreads: {e}")
        return None

def update_trading_parameters(engine, pair_name, buffer_rate, position_multiplier):
    """Update buffer_rate and position_multiplier for a trading pair"""
    query = f"""
    UPDATE public.trade_pool_pairs
    SET 
        buffer_rate = {buffer_rate},
        position_multiplier = {position_multiplier},
        updated_at = NOW()
    WHERE pair_name = '{pair_name}';
    """
    
    try:
        with engine.connect() as connection:
            connection.execute(query)
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
        st.error("Please connect to the database to continue")
        return
    
    # Success message for DB connection
    st.sidebar.success("Connected to database successfully")
    
    # Fetch all tokens and current parameters
    all_tokens = fetch_all_tokens(engine)
    current_params_df = fetch_current_parameters(engine)
    
    if all_tokens and not current_params_df.empty:
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Always select all tokens
        selected_tokens = all_tokens
        
        # Add a "Apply Recommendations" button
        apply_button = st.sidebar.button("Apply All Recommendations", use_container_width=True, 
                                     help="Apply all recommended parameter values to the system")
        
        # Add a "Reset Baselines" button
        reset_button = st.sidebar.button("Reset All Baselines to Current Spreads", use_container_width=True,
                                     help="Reset all baseline spreads to current market conditions")
        
        # Add a refresh button
        refresh_button = st.sidebar.button("Refresh Data", use_container_width=True)
        
        # Get spread data
        daily_avg_data = fetch_daily_spread_averages(engine, selected_tokens)
        
        # Check if we have baseline data
        baseline_spreads_df = get_baseline_spreads(engine)
        if baseline_spreads_df is None or baseline_spreads_df.empty:
            st.warning("No baseline spreads found. Please use 'Reset All Baselines' button to establish baselines.")
            if reset_button:
                if daily_avg_data is not None and not daily_avg_data.empty:
                    result = create_baseline_spreads(engine, daily_avg_data)
                    st.success(result)
                    # Clear cache and refresh baseline data
                    st.cache_data.clear()
                    baseline_spreads_df = get_baseline_spreads(engine)
                else:
                    st.error("No spread data available for baseline reset")
        elif reset_button:
            if daily_avg_data is not None and not daily_avg_data.empty:
                result = create_baseline_spreads(engine, daily_avg_data)
                st.success(result)
                # Clear cache and refresh baseline data
                st.cache_data.clear()
                baseline_spreads_df = get_baseline_spreads(engine)
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
                
                # Calculate recommended parameters
                if current_spread is not None and baseline_spread is not None:
                    rec_buffer, rec_position = calculate_recommended_params(
                        current_buffer_rate, 
                        current_position_multiplier,
                        current_spread,
                        baseline_spread,
                        max_leverage
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
                        'spread_change': spread_note,
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
                        
                        display_data.append({
                            'Pair': row['pair_name'],
                            'Type': row['token_type'],
                            'Size': row['depth_tier'],
                            'Market Spread': f"{row['current_spread']:.2f}",
                            'Baseline Spread': f"{row['baseline_spread']:.2f}",
                            'Spread Change': row['spread_change'],
                            'Current Buffer': current_buffer_formatted,
                            'Recommended Buffer': rec_buffer_formatted,
                            'Current Position Mult.': current_pos_formatted,
                            'Recommended Position Mult.': rec_pos_formatted
                        })
                    
                    display_df = pd.DataFrame(display_data)
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
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
                            
                            changed_display.append({
                                'Pair': row['pair_name'],
                                'Type': row['token_type'],
                                'Size': row['depth_tier'],
                                'Market Spread': f"{row['current_spread']:.2f}",
                                'Baseline Spread': f"{row['baseline_spread']:.2f}",
                                'Spread Change': row['spread_change'],
                                'Current Buffer': current_buffer_formatted,
                                'Recommended Buffer': rec_buffer_formatted,
                                'Current Position Mult.': current_pos_formatted,
                                'Recommended Position Mult.': rec_pos_formatted
                            })
                        
                        changed_display_df = pd.DataFrame(changed_display)
                        st.write(changed_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
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
                        
                        major_display.append({
                            'Pair': row['pair_name'],
                            'Market Spread': f"{row['current_spread']:.2f}",
                            'Baseline Spread': f"{row['baseline_spread']:.2f}",
                            'Spread Change': row['spread_change'],
                            'Current Buffer': current_buffer_formatted,
                            'Recommended Buffer': rec_buffer_formatted,
                            'Current Position Mult.': current_pos_formatted,
                            'Recommended Position Mult.': rec_pos_formatted
                        })
                    
                    major_display_df = pd.DataFrame(major_display)
                    st.write(major_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
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
                        
                        altcoin_display.append({
                            'Pair': row['pair_name'],
                            'Market Spread': f"{row['current_spread']:.2f}",
                            'Baseline Spread': f"{row['baseline_spread']:.2f}",
                            'Spread Change': row['spread_change'],
                            'Current Buffer': current_buffer_formatted,
                            'Recommended Buffer': rec_buffer_formatted,
                            'Current Position Mult.': current_pos_formatted,
                            'Recommended Position Mult.': rec_pos_formatted
                        })
                    
                    altcoin_display_df = pd.DataFrame(altcoin_display)
                    st.write(altcoin_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
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
                            if update_trading_parameters(engine, pair_name, rec_buffer, rec_position):
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
                    
                    #### Safety Constraints
                    
                    - **Buffer Rate Limit**: Never exceeds `0.9/max_leverage` to avoid immediate liquidations
                    - **Position Multiplier Bounds**: Between 100,000 and 10,000,000
                    - **Significant Change Threshold**: Parameters only update if spread changes by at least 5%
                    
                    The system automatically maintains these optimizations while keeping all parameters within safe bounds.
                    """)
            else:
                st.warning("Unable to generate recommendations. Check data quality.")
        else:
            st.warning("Missing spread data or baseline data. Please ensure both are available.")
    else:
        st.error("Failed to load tokens or current parameters. Please check the database connection.")

if __name__ == "__main__":
    main()