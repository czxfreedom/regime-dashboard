# Fixes for the Rollbit comparison display and parameter calculations

# 1. Updated function to calculate recommended parameters with proper relationships
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

# 2. Updated Rollbit parameters fetching function with correct column names
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

# 3. Updated code for the Rollbit comparison tab with better formatting
def render_rollbit_comparison(rollbit_comparison_df):
    """Render the Rollbit comparison tab with improved formatting"""
    
    if rollbit_comparison_df.empty:
        st.info("No matching pairs found with Rollbit data for comparison.")
        return
    
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)
    
    # Create grouped tables for each parameter type
    st.markdown("<h3>Buffer Rate Comparison</h3>", unsafe_allow_html=True)
    
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

# 4. Update the Rollbit tab in the main dashboard to use the new render function
def update_rollbit_tab(tabs):
    with tabs[2]:  # Rollbit Comparison tab
        st.markdown("### Rollbit Parameter Comparison")
        
        # Filter for pairs with Rollbit data
        rollbit_comparison_df = rec_df.dropna(subset=['rollbit_buffer_rate'])
        
        # Use the new rendering function
        render_rollbit_comparison(rollbit_comparison_df)