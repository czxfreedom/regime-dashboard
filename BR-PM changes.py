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

# Apply custom CSS styling (keeping your existing styling)
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
    /* Additional styles for Excel-like table appearance */
    .excel-table {
        border-collapse: collapse;
        width: 100%;
    }
    .excel-table th {
        background-color: #f2f2f2;
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
        font-weight: bold;
    }
    .excel-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .excel-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    /* Color highlighting for different rows */
    .red-row {
        background-color: #ffcccc !important;
    }
    .green-row {
        background-color: #ccffcc !important;
    }
    .yellow-row {
        background-color: #ffffcc !important;
    }
    /* Add rest of your existing styles */
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

# --- Keep all your existing utility functions ---
# (format_percent, format_number, format_float, is_major, safe_division, check_null_or_zero, etc.)

# --- Data Fetching Functions (keep all your existing ones) ---

# --- Main Application ---
def main():
    st.markdown('<div class="header-style">Exchange Parameter Optimization Dashboard</div>', unsafe_allow_html=True)

    # Sidebar controls (keeping your existing sidebar elements)
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

    # Fetch all the necessary data
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
        
        # NEW: Create an Excel-like table format that matches your mockup
        st.markdown("## Parameter Comparison & Recommendations")
        
        # Filter to only show major tokens for a cleaner initial view
        major_tokens = rec_df[rec_df['token_type'] == 'Major'].copy()
        
        # Create selectors for viewing options
        col1, col2 = st.columns(2)
        with col1:
            view_option = st.selectbox(
                "View tokens:",
                ["Major Tokens Only", "All Tokens", "Tokens Needing Adjustment"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Pair Name", "Buffer Change (%)", "Position Change (%)", "Spread Change Ratio"]
            )
        
        # Filter based on view option
        if view_option == "Major Tokens Only":
            display_df = rec_df[rec_df['token_type'] == 'Major'].copy()
        elif view_option == "Tokens Needing Adjustment":
            display_df = rec_df[
                (abs(rec_df['buffer_change']) > 1.0) | 
                (abs(rec_df['position_change']) > 1.0)
            ].copy()
        else:
            display_df = rec_df.copy()
        
        # Sort based on selection
        if sort_by == "Pair Name":
            display_df = display_df.sort_values("pair_name")
        elif sort_by == "Buffer Change (%)":
            display_df = display_df.sort_values("buffer_change", ascending=False)
        elif sort_by == "Position Change (%)":
            display_df = display_df.sort_values("position_change", ascending=False)
        elif sort_by == "Spread Change Ratio":
            display_df = display_df.sort_values("spread_change_ratio", ascending=False)
        
        # Create Excel-like table with highlighting
        excel_table = """
        <div style="overflow-x: auto;">
        <table class="excel-table">
            <tr>
                <th>Token</th>
                <th>Type</th>
                <th>Baseline Spread</th>
                <th>Current Spread</th>
                <th>Spread Change</th>
                <th>Weekly Range Position</th>
                <th>Current Buffer Rate</th>
                <th>Rollbit Buffer Rate</th>
                <th>Recommended Buffer Rate</th>
                <th>Current Position Mult.</th>
                <th>Rollbit Position Mult.</th>
                <th>Recommended Position Mult.</th>
            </tr>
        """
        
        for _, row in display_df.iterrows():
            # Determine row highlighting
            row_class = ""
            if abs(row.get('buffer_change', 0)) > 5.0 or abs(row.get('position_change', 0)) > 5.0:
                row_class = "red-row"
            elif abs(row.get('buffer_change', 0)) > 2.0 or abs(row.get('position_change', 0)) > 2.0:
                row_class = "yellow-row"
            
            # Format values
            baseline_spread = format_percent(row.get('baseline_spread'))
            current_spread = format_percent(row.get('current_spread'))
            spread_change = f"{(row.get('spread_change_ratio', 1) - 1) * 100:+.2f}%" if not pd.isna(row.get('spread_change_ratio')) else "N/A"
            
            # Calculate range position
            range_position = "N/A"
            if not pd.isna(row.get('weekly_range_ratio')):
                range_position = f"{row['weekly_range_ratio'] * 100:.0f}%"
            
            # Format rates and multipliers
            current_buffer = format_percent(row.get('current_buffer_rate'))
            rollbit_buffer = format_percent(row.get('rollbit_buffer_rate')) if 'rollbit_buffer_rate' in row else "N/A"
            rec_buffer = format_percent(row.get('recommended_buffer_rate'))
            
            current_pos = format_number(row.get('current_position_multiplier'))
            rollbit_pos = format_number(row.get('rollbit_position_multiplier')) if 'rollbit_position_multiplier' in row else "N/A"
            rec_pos = format_number(row.get('recommended_position_multiplier'))
            
            excel_table += f"""
            <tr class="{row_class}">
                <td>{row['pair_name']}</td>
                <td>{row['token_type']}</td>
                <td>{baseline_spread}</td>
                <td>{current_spread}</td>
                <td>{spread_change}</td>
                <td>{range_position}</td>
                <td>{current_buffer}</td>
                <td>{rollbit_buffer}</td>
                <td>{rec_buffer}</td>
                <td>{current_pos}</td>
                <td>{rollbit_pos}</td>
                <td>{rec_pos}</td>
            </tr>
            """
        
        excel_table += """
        </table>
        </div>
        <div style="margin-top: 10px;">
            <span style="background-color: #ffcccc; padding: 3px 8px;">Red</span>: Major adjustment needed (>5%)
            <span style="margin-left: 15px; background-color: #ffffcc; padding: 3px 8px;">Yellow</span>: Moderate adjustment needed (>2%)
        </div>
        """
        
        st.markdown(excel_table, unsafe_allow_html=True)
        
        # Add visualization section
        st.markdown("## Parameter Change Visualization")
        
        # Create visualization of spread changes vs parameter changes
        fig = px.scatter(
            display_df,
            x="spread_change_ratio",
            y="buffer_change",
            color="token_type",
            hover_name="pair_name",
            size=abs(display_df['buffer_change'])+1,  # Adding 1 to ensure all points have some size
            color_discrete_map={
                'Major': '#1976D2',
                'Altcoin': '#FFA000'
            },
            title="Buffer Rate Changes vs Spread Change Ratio"
        )
        
        # Add reference lines
        fig.add_shape(
            type="line",
            x0=0.5, y0=0,
            x1=2.0, y1=0,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=1.0, y0=-20,
            x1=1.0, y1=20,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Spread Change Ratio (Current/Baseline)",
            yaxis_title="Buffer Rate Change (%)",
            height=500,
            xaxis=dict(range=[0.5, 2.0]),
            yaxis=dict(range=[-20, 20])
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        # Create second visualization for position multiplier
        with col2:
            fig2 = px.scatter(
                display_df,
                x="spread_change_ratio",
                y="position_change",
                color="token_type",
                hover_name="pair_name",
                size=abs(display_df['position_change'])+1,
                color_discrete_map={
                    'Major': '#1976D2',
                    'Altcoin': '#FFA000'
                },
                title="Position Multiplier Changes vs Spread Change Ratio"
            )
            
            # Add reference lines
            fig2.add_shape(
                type="line",
                x0=0.5, y0=0,
                x1=2.0, y1=0,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=1.0, y0=-20,
                x1=1.0, y1=20,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Update layout
            fig2.update_layout(
                xaxis_title="Spread Change Ratio (Current/Baseline)",
                yaxis_title="Position Multiplier Change (%)",
                height=500,
                xaxis=dict(range=[0.5, 2.0]),
                yaxis=dict(range=[-20, 20])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Weekly spread range analysis for a selected token
        st.markdown("## Weekly Spread Range Analysis")
        
        # Token selector
        selected_token = st.selectbox(
            "Select a token to view its weekly spread range:",
            display_df['pair_name'].tolist()
        )
        
        # Find the selected token in the data
        token_data = display_df[display_df['pair_name'] == selected_token].iloc[0]
        
        # Render the weekly range visualization
        weekly_html = render_weekly_spread_range(
            token_data['pair_name'],
            token_data.get('min_weekly_spread'),
            token_data.get('max_weekly_spread'),
            token_data.get('current_spread'),
            token_data.get('avg_weekly_spread')
        )
        
        st.markdown(weekly_html, unsafe_allow_html=True)
        
        # Parameter impact explanation
        st.markdown("""
        ## Parameter Impact Explanation
        
        ### How Parameters Affect Trading
        
        <div class="info-box">
            <h4>Buffer Rate</h4>
            <p>Controls the safety margin for positions. Higher buffer rates reduce leverage but increase stability.</p>
            <ul>
                <li>When market spreads increase, buffer rates should increase to maintain safety</li>
                <li>When market spreads decrease, buffer rates can decrease to improve capital efficiency</li>
            </ul>
            
            <h4>Position Multiplier</h4>
            <p>Controls maximum position size. Higher position multipliers allow larger positions with less impact.</p>
            <ul>
                <li>When market spreads increase, position multipliers should decrease to limit exposure</li>
                <li>When market spreads decrease, position multipliers can increase to allow larger positions</li>
            </ul>
            
            <h4>Formula for Parameter Adjustments</h4>
            <p>The following formulas determine parameter adjustments:</p>
            <ul>
                <li>Buffer Rate = Current Buffer Rate × (Spread Change Ratio<sup>Buffer Sensitivity</sup>)</li>
                <li>Position Multiplier = Current Position Multiplier ÷ (Spread Change Ratio<sup>Position Sensitivity</sup>)</li>
            </ul>
            <p>Where Spread Change Ratio = Current Spread ÷ Baseline Spread</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Market Impact Simulation
        st.markdown("## Parameter Impact Simulation")
        render_parameter_simulation()  # Using your existing function
        
    else:
        st.error("Failed to load required data. Please check database connection and try refreshing.")

if __name__ == "__main__":
    main()