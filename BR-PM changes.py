# Add these imports to the top of your file if not already present
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import traceback
import json
import os
import math  # Added for the new implementation
import numpy as np
from sqlalchemy import text

# --- Database Functions ---
@st.cache_data(ttl=600)
def fetch_spread_weekly_stats():
    """Fetch weekly spread statistics"""
    try:
        engine = init_connection()
        if not engine:
            return None
        query = """
        SELECT 
            pair_name,
            min_spread,
            max_spread,
            std_dev,
            updated_at
        FROM spread_weekly_stats
        """
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching weekly spread stats: {e}")
        return None

def update_weekly_spread_stats(market_data_df):
    """Update weekly spread statistics based on current market data"""
    if market_data_df is None or market_data_df.empty:
        return False, "No market data available"
    
    try:
        engine = init_connection()
        if not engine:
            return False, "Database connection error"
        
        # Calculate current spreads by pair
        current_spreads = calculate_current_spreads(market_data_df)
        
        # Fetch existing stats
        existing_stats_df = fetch_spread_weekly_stats()
        existing_stats = {}
        
        if existing_stats_df is not None and not existing_stats_df.empty:
            for _, row in existing_stats_df.iterrows():
                existing_stats[row['pair_name']] = {
                    'min_spread': row['min_spread'],
                    'max_spread': row['max_spread'],
                    'std_dev': row['std_dev']
                }
        
        # Update statistics for each pair
        success_count = 0
        error_count = 0
        
        for pair_name, current_spread in current_spreads.items():
            try:
                # Get existing stats or initialize new ones
                if pair_name in existing_stats:
                    stats = existing_stats[pair_name]
                    
                    # Update min and max
                    min_spread = min(stats['min_spread'], current_spread)
                    max_spread = max(stats['max_spread'], current_spread)
                    
                    # Estimate std_dev as 1/4 of the range (normal distribution approximation)
                    std_dev = (max_spread - min_spread) / 4.0
                    if std_dev <= 0:
                        std_dev = current_spread * 0.05  # Fallback: 5% of current spread
                else:
                    # Initialize new stats with reasonable range around current value
                    min_spread = current_spread * 0.9  # 10% below current
                    max_spread = current_spread * 1.1  # 10% above current
                    std_dev = current_spread * 0.05     # 5% of current spread
                
                # Upsert the stats
                query = text("""
                INSERT INTO spread_weekly_stats 
                    (pair_name, min_spread, max_spread, std_dev, updated_at)
                VALUES 
                    (:pair_name, :min_spread, :max_spread, :std_dev, :updated_at)
                ON CONFLICT (pair_name) DO UPDATE 
                SET 
                    min_spread = LEAST(spread_weekly_stats.min_spread, :min_spread),
                    max_spread = GREATEST(spread_weekly_stats.max_spread, :max_spread),
                    std_dev = :std_dev,
                    updated_at = :updated_at
                """)
                
                with engine.connect() as conn:
                    conn.execute(
                        query, 
                        {
                            "pair_name": pair_name,
                            "min_spread": min_spread,
                            "max_spread": max_spread,
                            "std_dev": std_dev,
                            "updated_at": datetime.now()
                        }
                    )
                    conn.commit()
                
                success_count += 1
            except Exception as e:
                error_count += 1
        
        return success_count > 0, f"Updated {success_count} pairs with {error_count} errors"
    except Exception as e:
        return False, f"Error updating weekly stats: {str(e)}"

# --- Modified Parameter Calculation Function ---
def calculate_recommended_params(current_params, current_spread, baseline_spread, 
                              weekly_stats, sensitivities, 
                              significant_change_threshold=1.5):  # Z-score threshold
    """Calculate recommended parameter values using z-score approach"""
    
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
    
    # Get weekly statistics with fallbacks
    weekly_min = weekly_stats.get('min_spread', baseline_spread * 0.8)
    weekly_max = weekly_stats.get('max_spread', baseline_spread * 1.2)
    weekly_std_dev = weekly_stats.get('std_dev')
    
    # If std_dev is missing, estimate it as 1/4 of the range (normal distribution approximation)
    if weekly_std_dev is None or pd.isna(weekly_std_dev) or weekly_std_dev == 0:
        weekly_std_dev = (weekly_max - weekly_min) / 4.0
        if weekly_std_dev <= 0:
            weekly_std_dev = baseline_spread * 0.05  # Fallback: 5% of baseline
    
    # Calculate z-score: how many std devs away from baseline is current spread
    z_score = (current_spread - baseline_spread) / weekly_std_dev if weekly_std_dev > 0 else 0
    
    # Check if change is significant based on z-score magnitude
    if abs(z_score) < significant_change_threshold:
        return {
            'buffer_rate': current_buffer_rate,
            'position_multiplier': current_position_multiplier
        }
    
    # Get sensitivity parameters
    buffer_sensitivity = sensitivities.get('buffer_sensitivity', 0.5)
    position_sensitivity = sensitivities.get('position_sensitivity', 0.5)
    
    # Calculate change factor based on z-score
    # Use tanh to create a bounded factor (between 0.5 and 2.0 for typical z-scores)
    change_direction = 1 if z_score > 0 else -1
    z_factor = 1.0 + (change_direction * min(abs(z_score) / 5.0, 0.5))
    
    # Calculate new parameters
    recommended_buffer_rate = current_buffer_rate * (z_factor ** buffer_sensitivity)
    recommended_position_multiplier = current_position_multiplier / (z_factor ** position_sensitivity)
    
    # Apply bounds based on the provided constraints
    buffer_upper_bound = 0.7 / max_leverage if max_leverage > 0 else 0.007
    recommended_buffer_rate = max(0.0, min(buffer_upper_bound, recommended_buffer_rate))
    recommended_position_multiplier = max(1, min(15000, recommended_position_multiplier))
    
    return {
        'buffer_rate': recommended_buffer_rate,
        'position_multiplier': recommended_position_multiplier
    }

# --- Modified Recommendations Generator ---
def generate_recommendations(current_params_df, market_data_df, baselines_df, weekly_stats_df, sensitivities):
    """Generate parameter recommendations based on market data and weekly statistics"""
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
    
    # Create weekly stats dictionary
    weekly_stats = {}
    if weekly_stats_df is not None and not weekly_stats_df.empty:
        for _, row in weekly_stats_df.iterrows():
            weekly_stats[row['pair_name']] = {
                'min_spread': row['min_spread'],
                'max_spread': row['max_spread'],
                'std_dev': row['std_dev']
            }
    
    # Create recommendations DataFrame
    recommendations = []
    
    for pair, params in current_params.items():
        if pair in current_spreads and pair in baselines:
            current_spread = current_spreads[pair]
            baseline_spread = baselines[pair]
            
            # Get weekly stats for this pair or empty dict
            pair_weekly_stats = weekly_stats.get(pair, {})
            
            recommended = calculate_recommended_params(
                params, 
                current_spread, 
                baseline_spread,
                pair_weekly_stats,
                sensitivities,
                significant_change_threshold=1.5  # Z-score threshold
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
            
            # Calculate spread stats
            spread_change_pct = 0
            if current_spread > 0 and baseline_spread > 0:
                spread_change_pct = ((current_spread / baseline_spread) - 1) * 100
            
            # Calculate z-score if possible
            z_score = None
            if pair in weekly_stats and 'std_dev' in weekly_stats[pair] and weekly_stats[pair]['std_dev'] > 0:
                z_score = (current_spread - baseline_spread) / weekly_stats[pair]['std_dev']
            
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
                'max_leverage': params['max_leverage'],
                'z_score': z_score,
                'weekly_min': pair_weekly_stats.get('min_spread'),
                'weekly_max': pair_weekly_stats.get('max_spread'),
                'weekly_std_dev': pair_weekly_stats.get('std_dev')
            })
    
    return pd.DataFrame(recommendations)

# --- Modified Parameter Table Rendering ---
def render_complete_parameter_table(rec_df, sort_by="pair_name"):
    """Render the complete parameter table with all pairs including z-score"""
    
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
    elif sort_by == "z_score":
        sorted_df = rec_df.sort_values("z_score", key=abs, ascending=False)
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
    
    def highlight_zscore(val):
        """Highlight significant z-scores"""
        if isinstance(val, str) and not val == "N/A":
            try:
                num_val = abs(float(val))
                if num_val > 2.0:
                    return 'background-color: #ffcccc'  # Red for significant z-score
                elif num_val > 1.5:
                    return 'background-color: #ffffcc'  # Yellow for moderate z-score
            except:
                pass
        return ''
    
    # Create a formatted DataFrame for display
    display_df = pd.DataFrame({
        'Pair': sorted_df['pair_name'],
        'Type': sorted_df['token_type'],
        'Current Spread': sorted_df['current_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Baseline Spread': sorted_df['baseline_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Z-Score': sorted_df['z_score'].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
        ),
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
    
    # Style the dataframe with highlighting
    styled_df = display_df.style\
        .applymap(highlight_changes, subset=['Buffer Change', 'Position Change', 'Spread Change'])\
        .applymap(highlight_zscore, subset=['Z-Score'])
    
    # Display with highlighting
    st.dataframe(styled_df, use_container_width=True)
    
    # Add a color legend below the table
    st.markdown("""
    <div style="margin-top: 10px; font-size: 0.8em;">
        <span style="background-color: #ffcccc; padding: 3px 8px;">Red</span>: Major adjustment needed (Z-Score > 2.0 or change > 5%)
        <span style="margin-left: 15px; background-color: #ffffcc; padding: 3px 8px;">Yellow</span>: Moderate adjustment needed (Z-Score > 1.5 or change > 2%)
    </div>
    """, unsafe_allow_html=True)

# --- Main Application Modifications ---
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
    
    # Add weekly stats update button
    if st.sidebar.button("Update Weekly Spread Stats", use_container_width=True):
        market_data_df = fetch_market_spread_data()
        if market_data_df is not None and not market_data_df.empty:
            success, message = update_weekly_spread_stats(market_data_df)
            if success:
                st.sidebar.success(message)
                # Clear cache to refresh data
                st.cache_data.clear()
            else:
                st.sidebar.error(message)
        else:
            st.sidebar.error("No market data available to update weekly stats")

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
    weekly_stats_df = fetch_spread_weekly_stats()  # Add this line to fetch weekly stats

    # Generate recommendations
    if current_params_df is not None and market_data_df is not None and baselines_df is not None:
        # Generate recommendations with the selected sensitivities
        rec_df = generate_recommendations(current_params_df, market_data_df, baselines_df, weekly_stats_df, sensitivities)
        
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
                    "Spread Change Ratio",
                    "Z-Score"  # Add Z-Score as sort option
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

# --- Create the spread_weekly_stats table if it doesn't exist ---
def create_weekly_stats_table():
    """Create the spread_weekly_stats table if it doesn't exist"""
    try:
        engine = init_connection()
        if not engine:
            return False
        
        query = """
        CREATE TABLE IF NOT EXISTS spread_weekly_stats (
            pair_name VARCHAR(50) PRIMARY KEY,
            min_spread NUMERIC,
            max_spread NUMERIC,
            std_dev NUMERIC,
            updated_at TIMESTAMP
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
        
        return True
    except Exception as e:
        st.error(f"Error creating weekly stats table: {e}")
        return False

# --- Additional Overview Tab Content ---
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
    
    ### Z-Score Based Parameter Adjustment
    
    - Z-Score = (Current Spread - Baseline Spread) / Weekly Standard Deviation
    - Changes are only applied when Z-Score is above threshold (1.5)
    - This approach accounts for the typical volatility range of each token
    - Tokens with naturally high spread volatility require larger absolute changes to trigger adjustments
    
    ### Dashboard Controls
    
    - **Refresh Data**: Updates current spreads and parameters from the database
    - **Reset Baselines**: Sets current market spreads as new baselines
    - **Update Weekly Stats**: Updates the weekly spread range statistics used for Z-score calculation
    - **Apply Recommendations**: Updates database with recommended parameters
    - **Undo Latest Changes**: Reverts parameters to values before last applied recommendations
    """)

# Call table creation function during initial setup
if 'table_created' not in st.session_state:
    if create_weekly_stats_table():
        st.session_state.table_created = True