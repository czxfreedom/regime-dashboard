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
    page_title="Parameter Optimization Dashboard",
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
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"

def format_number(value):
    """Format a number with comma separation"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"

def format_float(value, decimals=2):
    """Format a float with specified decimal places"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def is_major(token):
    """Determine if a token is a major token"""
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
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
                               sensitivities, significant_change_threshold):
    """Calculate recommended parameter values based on spread change ratio with proper inverse relationships"""
    
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
    
    # Fixed parameter relationships:
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

def generate_recommendations(current_params_df, market_data_df, baselines_df):
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
    
    # Create recommendations DataFrame
    recommendations = []
    
    # Sensitivity parameters
    sensitivities = {
        'buffer_sensitivity': 0.5,
        'position_sensitivity': 0.5,
        'rate_multiplier_sensitivity': 0.5,
        'rate_exponent_sensitivity': 0.5
    }
    
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
            
            # Calculate changes
            buffer_change = ((recommended['buffer_rate'] - params['buffer_rate']) / params['buffer_rate']) * 100
            position_change = ((recommended['position_multiplier'] - params['position_multiplier']) / params['position_multiplier']) * 100
            rate_mult_change = ((recommended['rate_multiplier'] - params['rate_multiplier']) / params['rate_multiplier']) * 100
            rate_exp_change = ((recommended['rate_exponent'] - params['rate_exponent']) / params['rate_exponent']) * 100
            
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
                'spread_change_ratio': current_spread / baseline_spread if baseline_spread > 0 else 1.0
            })
    
    return pd.DataFrame(recommendations)

def add_rollbit_comparison(rec_df, rollbit_df):
    """Add Rollbit parameter data to recommendations DataFrame for comparison"""
    if rec_df is None or rollbit_df is None:
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
            rec_df.at[i, 'rollbit_buffer_rate'] = rollbit_params[pair_name]['buffer_rate']
            rec_df.at[i, 'rollbit_position_multiplier'] = rollbit_params[pair_name]['position_multiplier']
            rec_df.at[i, 'rollbit_rate_multiplier'] = rollbit_params[pair_name]['rate_multiplier']
            rec_df.at[i, 'rollbit_rate_exponent'] = rollbit_params[pair_name]['rate_exponent']
    
    return rec_df

def render_rollbit_comparison(rollbit_comparison_df):
    """Render the Rollbit comparison tab with improved formatting"""
    
    if rollbit_comparison_df.empty:
        st.info("No matching pairs found with Rollbit data for comparison.")
        return
    
    # Buffer Rate Table
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
        surf_buffer = format_percent(row['current_buffer_rate'])
        rollbit_buffer = format_percent(row['rollbit_buffer_rate'])
        buffer_ratio = f"{row['current_buffer_rate']/row['rollbit_buffer_rate']:.2f}x" if row['rollbit_buffer_rate'] > 0 else "N/A"
        
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
        surf_position = format_number(row['current_position_multiplier'])
        rollbit_position = format_number(row['rollbit_position_multiplier'])
        position_ratio = f"{row['current_position_multiplier']/row['rollbit_position_multiplier']:.2f}x" if row['rollbit_position_multiplier'] > 0 else "N/A"
        
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
        surf_rate_mult = format_float(row['current_rate_multiplier'], 2)
        rollbit_rate_mult = format_float(row['rollbit_rate_multiplier'], 2)
        rate_mult_ratio = f"{row['current_rate_multiplier']/row['rollbit_rate_multiplier']:.2f}x" if row['rollbit_rate_multiplier'] > 0 else "N/A"
        
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
        surf_rate_exp = format_float(row['current_rate_exponent'], 2)
        rollbit_rate_exp = format_float(row['rollbit_rate_exponent'], 2)
        rate_exp_ratio = f"{row['current_rate_exponent']/row['rollbit_rate_exponent']:.2f}x" if row['rollbit_rate_exponent'] > 0 else "N/A"
        
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
    </div>
    """, unsafe_allow_html=True)

# --- Main Application ---
st.markdown('<div class="header-style">Parameter Optimization Dashboard</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Controls")

# Add a refresh button
if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

# Add adjustment sensitivity controls
st.sidebar.header("Adjustment Sensitivity")
buffer_sensitivity = st.sidebar.slider("Buffer Rate Sensitivity", 0.1, 1.0, 0.5, 0.1)
position_sensitivity = st.sidebar.slider("Position Multiplier Sensitivity", 0.1, 1.0, 0.5, 0.1)
rate_multiplier_sensitivity = st.sidebar.slider("Rate Multiplier Sensitivity", 0.1, 1.0, 0.5, 0.1)
rate_exponent_sensitivity = st.sidebar.slider("Rate Exponent Sensitivity", 0.1, 1.0, 0.5, 0.1)

# Fetch current data
current_params_df = fetch_current_parameters()
market_data_df = fetch_market_spread_data()
baselines_df = fetch_spread_baselines()
rollbit_df = fetch_rollbit_parameters()

# Generate recommendations
if current_params_df is not None and market_data_df is not None and baselines_df is not None:
    # Set sensitivities based on sidebar inputs
    sensitivities = {
        'buffer_sensitivity': buffer_sensitivity,
        'position_sensitivity': position_sensitivity,
        'rate_multiplier_sensitivity': rate_multiplier_sensitivity,
        'rate_exponent_sensitivity': rate_exponent_sensitivity
    }
    
    # Generate recommendations with updated sensitivities
    rec_df = generate_recommendations(current_params_df, market_data_df, baselines_df)
    
    # Add Rollbit comparison data if available
    if rollbit_df is not None:
        rec_df = add_rollbit_comparison(rec_df, rollbit_df)
    
    # Create tabs for different views
    tabs = st.tabs(["Recommendations Overview", "Parameter Details", "Rollbit Comparison"])
    
    # Tab 1: Recommendations Overview
    with tabs[0]:
        st.markdown("### Parameter Recommendations Overview")
        
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
                (abs(rec_df['position_change']) > 1.0)
            ].copy()
            
            if not significant_changes.empty:
                # Format for display
                display_df = pd.DataFrame({
                    'Pair': significant_changes['pair_name'],
                    'Token Type': significant_changes['token_type'],
                    'Current Buffer': significant_changes['current_buffer_rate'].apply(lambda x: f"{x*100:.2f}%"),
                    'Recommended Buffer': significant_changes['recommended_buffer_rate'].apply(lambda x: f"{x*100:.2f}%"),
                    'Buffer Change': significant_changes['buffer_change'].apply(lambda x: f"{x:+.2f}%"),
                    'Current Position': significant_changes['current_position_multiplier'].apply(lambda x: f"{x:,.0f}"),
                    'Recommended Position': significant_changes['recommended_position_multiplier'].apply(lambda x: f"{x:,.0f}"),
                    'Position Change': significant_changes['position_change'].apply(lambda x: f"{x:+.2f}%")
                })
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No significant parameter changes recommended.")
                
            # Create visualizations
            st.markdown("#### Parameter Change Distribution")
            
            fig = go.Figure()
            
            # Add histogram for buffer rate changes
            fig.add_trace(go.Histogram(
                x=rec_df['buffer_change'],
                name='Buffer Rate Changes',
                nbinsx=20,
                marker_color='blue',
                opacity=0.7
            ))
            
            # Add histogram for position multiplier changes
            fig.add_trace(go.Histogram(
                x=rec_df['position_change'],
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
        else:
            st.warning("No recommendation data available")
    
    # Tab 2: Parameter Details
    with tabs[1]:
        st.markdown("### Detailed Parameter Recommendations")
        
        if rec_df is not None and not rec_df.empty:
            # Sort by absolute buffer change (descending)
            rec_df['abs_buffer_change'] = rec_df['buffer_change'].abs()
            sorted_df = rec_df.sort_values(by='abs_buffer_change', ascending=False)
            
            # Display detailed table
            display_df = pd.DataFrame({
                'Pair': sorted_df['pair_name'],
                'Type': sorted_df['token_type'],
                'Current Buffer': sorted_df['current_buffer_rate'].apply(lambda x: f"{x*100:.2f}%"),
                'Rec. Buffer': sorted_df['recommended_buffer_rate'].apply(lambda x: f"{x*100:.2f}%"),
                'Buffer Δ': sorted_df['buffer_change'].apply(lambda x: f"{x:+.2f}%"),
                'Current Pos.': sorted_df['current_position_multiplier'].apply(lambda x: f"{x:,.0f}"),
                'Rec. Pos.': sorted_df['recommended_position_multiplier'].apply(lambda x: f"{x:,.0f}"),
                'Pos. Δ': sorted_df['position_change'].apply(lambda x: f"{x:+.2f}%"),
                'Rate Mult.': sorted_df['current_rate_multiplier'].apply(lambda x: f"{x:.2f}"),
                'Rec. Rate Mult.': sorted_df['recommended_rate_multiplier'].apply(lambda x: f"{x:.2f}"),
                'Rate Exp.': sorted_df['current_rate_exponent'].apply(lambda x: f"{x:.2f}"),
                'Rec. Rate Exp.': sorted_df['recommended_rate_exponent'].apply(lambda x: f"{x:.2f}")
            })
            
            st.dataframe(display_df, use_container_width=True)
            
            # Create scatter plot of spread change ratio vs parameter changes
            st.markdown("#### Spread Change Impact on Parameters")
            
            fig = go.Figure()
            
            # Add scatter plot for buffer rate changes
            fig.add_trace(go.Scatter(
                x=sorted_df['spread_change_ratio'],
                y=sorted_df['buffer_change'],
                mode='markers',
                name='Buffer Rate Change',
                marker=dict(size=8, color='blue', opacity=0.7)
            ))
            
            # Add scatter plot for position multiplier changes
            fig.add_trace(go.Scatter(
                x=sorted_df['spread_change_ratio'],
                y=sorted_df['position_change'],
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
            st.warning("No parameter details available")
    
    # Tab 3: Rollbit Comparison
    with tabs[2]:
        st.markdown("### Rollbit Parameter Comparison")
        
        # Filter for pairs with Rollbit data
        rollbit_comparison_df = rec_df.dropna(subset=['rollbit_buffer_rate'])
        
        # Use the rendering function
        render_rollbit_comparison(rollbit_comparison_df)
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
    2. Parameter adjustments are proportional to the relative change in spread
    3. Sensitivity controls adjust how aggressively parameters respond to spread changes
    4. Rollbit parameters are shown for comparison with a major competitor
    
    ### Interpretation
    
    - Parameters with significant recommended changes may need manual adjustment
    - The scatter plot shows how spread changes correlate with parameter recommendations
    - Consider both market conditions and competitor settings when finalizing parameters
    """)