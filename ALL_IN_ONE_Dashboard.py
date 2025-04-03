import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
import importlib
import os
import sys

# Add module path to sys.path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# App Configuration
st.set_page_config(
    page_title="Unified Regime Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for sidebar and tabs
st.markdown("""
<style>
    .header-style {
        font-size:26px !important;
        font-weight: bold;
        padding: 10px 0;
        text-align: center;
    }
    .subheader-style {
        font-size:20px !important;
        font-weight: bold;
        padding: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 15px;
        background-color: #f5f5f5;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4682B4;
        color: white;
    }
    .refresh-btn {
        font-size: 16px;
        margin-bottom: 10px;
    }
    .last-refresh {
        font-size: 12px;
        font-style: italic;
        margin-bottom: 20px;
    }
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .st-emotion-cache-16txtl3 h4 {
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Configuration ---
def connect_to_database():
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
        # If secrets not available, allow manual entry
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
                engine = create_engine(db_uri)
                st.sidebar.success("Connected to database successfully")
                return engine
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {e}")
                st.stop()
        else:
            st.error("Please connect to the database to continue")
            st.stop()

# Establish database connection
engine = connect_to_database()

# --- Dashboard Modules ---
# Import all dashboard modules
# These should be your existing Python files with modifications for integration
try:
    import Spread_matrix as spread_matrix_module
    import Macro_view as macro_view_module
    import Volandhurst as volandhurst_module
    import Pnlandtrades as pnlandtrades_module
    import platformpnlcumulative as platformpnl_module
except ImportError:
    st.error("Some dashboard modules could not be imported. Please check file paths and names.")

# --- Main Application ---
def main():
    # Title and current time
    singapore_timezone = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    st.markdown('<div class="header-style">Unified Regime Dashboard Suite</div>', unsafe_allow_html=True)
    st.markdown(f"Current Singapore Time: **{now_sg.strftime('%Y-%m-%d %H:%M:%S')}**")
    
    # Sidebar controls
    st.sidebar.markdown('<div class="sidebar-header">Dashboard Controls</div>', unsafe_allow_html=True)
    
    # Global refresh button
    if st.sidebar.button("🔄 Refresh All Dashboards", use_container_width=True, key="global_refresh"):
        # Clear all cached data
        st.cache_data.clear()
        st.experimental_rerun()
    
    st.sidebar.markdown(f'<div class="last-refresh">Last global refresh: {now_sg.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "Exchange Spread Matrix", 
        "Macro View (Hurst)", 
        "Vol & Hurst", 
        "PnL & Trades", 
        "PnL Cumulative",
        "All-In-One"
    ])
    
    # Define a function to handle the content for each tab
    with tabs[0]:  # Exchange Spread Matrix
        st.markdown("## Exchange Spread Matrix")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_spread_matrix"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_spread_matrix_dashboard()
    
    with tabs[1]:  # Macro View (Hurst)
        st.markdown("## Macro View (Hurst Exponent)")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_macro_view"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_macro_view_dashboard()
    
    with tabs[2]:  # Vol & Hurst
        st.markdown("## Volatility & Hurst")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_vol_hurst"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_volandhurst_dashboard()
    
    with tabs[3]:  # PnL & Trades
        st.markdown("## PnL & Trades")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_pnl_trades"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_pnlandtrades_dashboard()
    
    with tabs[4]:  # PnL Cumulative
        st.markdown("## Platform PnL Cumulative")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_platform_pnl"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_platformpnl_dashboard()
    
    with tabs[5]:  # All-In-One
        st.markdown("## All-In-One Dashboard")
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("🔄", key="refresh_all_in_one"):
                # Clear cache for this specific module
                st.cache_data.clear()
                st.experimental_rerun()
                
        render_all_in_one_dashboard()

# --- Individual Dashboard Renderers ---
def render_spread_matrix_dashboard():
    """Render the Exchange Spread Matrix dashboard"""
    try:
        # Create a container for the content
        container = st.container()
        
        # Fetch all tokens (can be reused by multiple dashboards)
        all_tokens = fetch_all_tokens(engine)
        
        # Replace the main component functions used in the original script
        original_st_set_page_config = st.set_page_config
        original_st_title = st.title
        original_st_subheader = st.subheader
        
        # Mock these functions to prevent them from running in the tab
        st.set_page_config = lambda **kwargs: None
        st.title = lambda *args, **kwargs: None
        st.subheader = lambda *args, **kwargs: None
        
        # Run the dashboard code with the container as the context
        with container:
            # Set up token selection in the sidebar for this tab if needed
            # In this case, we'll just use all tokens since that's what the original does
            selected_tokens = all_tokens
            
            # Inject custom data into the module's namespace
            spread_matrix_module.engine = engine
            spread_matrix_module.all_tokens = all_tokens
            spread_matrix_module.selected_tokens = selected_tokens
            
            # Call the main functions from the module
            # Fetch daily spread data
            daily_avg_data = spread_matrix_module.fetch_daily_spread_averages(selected_tokens)
            if daily_avg_data is None or daily_avg_data.empty:
                st.warning("No spread data available for the selected time period.")
                return
            
            # Calculate matrix data for all fee levels
            matrix_data = spread_matrix_module.calculate_matrix_data(daily_avg_data)
            if matrix_data is None or not matrix_data:
                st.warning("Unable to process spread data. Check log for details.")
                return
            
            # Create tabs for different spread analyses
            subtab1, subtab2, subtab3 = st.tabs(["50K/20K Analysis", "100K/50K Analysis", "Spread By Size"])
            
            # Fill tabs with content
            with subtab1:
                st.markdown('<div class="subheader-style">50K/20K Spread Analysis</div>', unsafe_allow_html=True)
                
                # Display explanation of depth tiers
                st.markdown("""
                <div class="info-box">
                <b>Trading Size Definition:</b><br>
                • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K<br>
                • <b>Altcoin tokens</b>: 20K<br>
                <br>
                This tab shows daily averages of 10-minute spread data points at 50K/20K size.
                </div>
                """, unsafe_allow_html=True)
                
                if 'fee1' in matrix_data:
                    df = matrix_data['fee1'].copy()
                    scale_factor = spread_matrix_module.scale_factor
                    scale_label = spread_matrix_module.scale_label
                    
                    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                    
                    # Apply color coding and formatting similar to the original module
                    # ... (rest of the tab1 content from spread_matrix_module)
                    numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col] * scale_factor
                    
                    # Format and display the dataframe
                    display_df = df.copy()
                    display_df['Token Type'] = display_df['pair_name'].apply(
                        lambda x: 'Major' if spread_matrix_module.is_major(x) else 'Altcoin'
                    )
                    display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
                    
                    # Define column order
                    desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
                    if 'Improvement %' in display_df.columns:
                        desired_order.append('Improvement %')
                    ordered_columns = [col for col in desired_order if col in display_df.columns]
                    
                    if 'Surf Better' in display_df.columns:
                        ordered_columns.append('Surf Better')
                    
                    display_df = display_df[ordered_columns]
                    display_df = display_df.rename(columns={'pair_name': 'Token'})
                    
                    # Apply formatting to numeric columns
                    color_df = display_df.copy()
                    for col in numeric_cols:
                        if col in color_df.columns and col != 'Token Type':
                            color_df[col] = color_df[col].apply(lambda x: spread_matrix_module.color_code_value(x) if not pd.isna(x) else "")
                    
                    if 'Improvement %' in color_df.columns:
                        color_df['Improvement %'] = color_df['Improvement %'].apply(
                            lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                            (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
                        )
                    
                    # Display the table with HTML formatting
                    html_table = color_df.to_html(escape=False, index=False)
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Continue with visualizations...
                    # Example: Pie chart for SurfFuture performance
                    if 'Surf Better' in df.columns:
                        st.markdown("### SurfFuture Performance Analysis")
                        
                        # Count tokens where SurfFuture is better
                        surf_better_count = df['Surf Better'].sum()
                        total_count = len(df)
                        surf_worse_count = total_count - surf_better_count
                        
                        # Create pie chart
                        fig = px.pie(
                            values=[surf_better_count, surf_worse_count],
                            names=['SurfFuture Better', 'Other Exchanges Better'],
                            title="Proportion of Tokens Where SurfFuture Has Better Spreads",
                            color_discrete_sequence=['#4CAF50', '#FFC107'],
                            hole=0.4
                        )
                        
                        # Update layout
                        fig.update_layout(
                            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                            margin=dict(t=60, b=60, l=20, r=20),
                            height=400
                        )
                        
                        # Display percentage text in middle
                        better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
                        fig.add_annotation(
                            text=f"{better_percentage:.1f}%<br>Better",
                            x=0.5, y=0.5,
                            font_size=20,
                            showarrow=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
            with subtab2:
                # Similar implementation for subtab2 (100K/50K Analysis)
                st.markdown('<div class="subheader-style">100K/50K Spread Analysis</div>', unsafe_allow_html=True)
                # ... implementation similar to subtab1 but using 'fee2' data
            
            with subtab3:
                # Similar implementation for subtab3 (Spread By Size)
                st.markdown('<div class="subheader-style">Spread Analysis by Size</div>', unsafe_allow_html=True)
                # ... implementation for spread by size analysis
        
        # Restore the original functions
        st.set_page_config = original_st_set_page_config
        st.title = original_st_title
        st.subheader = original_st_subheader
        
    except Exception as e:
        st.error(f"Error rendering Spread Matrix dashboard: {e}")

def render_macro_view_dashboard():
    """Render the Macro View (Hurst) dashboard"""
    try:
        # Create a container for the content
        container = st.container()
        
        # Get all tokens
        all_tokens = fetch_all_tokens(engine)
        
        # Replace main Streamlit functions
        original_st_set_page_config = st.set_page_config
        original_st_title = st.title
        original_st_subheader = st.subheader
        
        # Mock these functions to prevent them from running in the tab
        st.set_page_config = lambda **kwargs: None
        st.title = lambda *args, **kwargs: None
        st.subheader = lambda *args, **kwargs: None
        
        # Run the dashboard code with the container as the context
        with container:
            # Create controls for token selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Let user select tokens to display (or select all)
                select_all = st.checkbox("Select All Tokens", value=True)
                
                if select_all:
                    selected_tokens = all_tokens
                else:
                    selected_tokens = st.multiselect(
                        "Select Tokens", 
                        all_tokens,
                        default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens
                    )
            
            # Set up the module with our engine and selected tokens
            macro_view_module.engine = engine
            
            # Show progress bar while calculating
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate Hurst for each token
            token_results = {}
            for i, token in enumerate(selected_tokens):
                try:
                    progress_bar.progress((i) / len(selected_tokens))
                    status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
                    result = macro_view_module.fetch_and_calculate_hurst(token)
                    if result is not None:
                        token_results[token] = result
                except Exception as e:
                    st.error(f"Error processing token {token}: {e}")
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")
            
            # Create Hurst exponent table
            if token_results:
                # Create table data
                table_data = {}
                for token, df in token_results.items():
                    hurst_series = df.set_index('time_label')['Hurst']
                    table_data[token] = hurst_series
                
                # Create DataFrame with all tokens
                hurst_table = pd.DataFrame(table_data)
                
                # Apply the time blocks in the proper order
                singapore_timezone = pytz.timezone('Asia/Singapore')
                now_utc = datetime.now(pytz.utc)
                now_sg = now_utc.astimezone(singapore_timezone)
                aligned_time_blocks = macro_view_module.generate_aligned_time_blocks(now_sg)
                
                available_times = set(hurst_table.index)
                ordered_times = [t for t in aligned_time_blocks if t in available_times]
                
                if not ordered_times and available_times:
                    ordered_times = sorted(list(available_times), reverse=True)
                
                # Reindex with the ordered times
                hurst_table = hurst_table.reindex(ordered_times)
                hurst_table = hurst_table.round(2)
                
                def color_cells(val):
                    if pd.isna(val):
                        return 'background-color: #f5f5f5; color: #666666;' # Grey for missing
                    elif val < 0.4:
                        intensity = max(0, min(255, int(255 * (0.4 - val) / 0.4)))
                        return f'background-color: rgba(255, {255-intensity}, {255-intensity}, 0.7); color: black'
                    elif val > 0.6:
                        intensity = max(0, min(255, int(255 * (val - 0.6) / 0.4)))
                        return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
                    else:
                        return 'background-color: rgba(200, 200, 200, 0.5); color: black' # Lighter gray
                
                styled_table = hurst_table.style.applymap(color_cells)
                st.markdown("## Hurst Exponent Table (30min timeframe, Last 24 hours, Singapore Time)")
                st.markdown("### Color Legend: <span style='color:red'>Mean Reversion</span>, <span style='color:gray'>Random Walk</span>, <span style='color:green'>Trending</span>", unsafe_allow_html=True)
                st.dataframe(styled_table, height=700, use_container_width=True)
                
                # Display market overview
                st.subheader("Current Market Overview (Singapore Time)")
                
                # Calculate market regime distributions and create visualizations
                # ... (implementation following the same logic as in the original)
                # Calculate latest values for each token
                latest_values = {}
                for token, df in token_results.items():
                    if not df.empty and not df['Hurst'].isna().all():
                        for block_time in aligned_time_blocks[:5]:
                            latest_data = df[df['time_label'] == block_time]
                            if not latest_data.empty:
                                latest = latest_data['Hurst'].iloc[0]
                                regime = latest_data['regime_desc'].iloc[0]
                                latest_values[token] = (latest, regime)
                                break
                        
                        # Fallback to most recent data point
                        if token not in latest_values:
                            latest = df['Hurst'].iloc[-1]
                            regime = df['regime_desc'].iloc[-1]
                            latest_values[token] = (latest, regime)
                
                if latest_values:
                    mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
                    random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
                    trending = sum(1 for v, r in latest_values.values() if v > 0.6)
                    total = mean_reverting + random_walk + trending
                    
                    if total > 0:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean-Reverting", f"{mean_reverting} ({mean_reverting/total*100:.1f}%)", delta=f"{mean_reverting/total*100:.1f}%")
                        col2.metric("Random Walk", f"{random_walk} ({random_walk/total*100:.1f}%)", delta=f"{random_walk/total*100:.1f}%")
                        col3.metric("Trending", f"{trending} ({trending/total*100:.1f}%)", delta=f"{trending/total*100:.1f}%")
                        
                        # Create pie chart
                        labels = ['Mean-Reverting', 'Random Walk', 'Trending']
                        values = [mean_reverting, random_walk, trending]
                        colors = ['rgba(255,100,100,0.8)', 'rgba(200,200,200,0.8)', 'rgba(100,255,100,0.8)']
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values, 
                            marker=dict(colors=colors, line=dict(color='#000000', width=2)), 
                            textinfo='label+percent', 
                            hole=.3
                        )])
                        
                        fig.update_layout(
                            title="Current Market Regime Distribution (Singapore Time)",
                            height=400,
                            font=dict(color="#000000", size=12),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display tokens in each category
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("### Mean-Reverting Tokens")
                            mr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v < 0.4]
                            mr_tokens.sort(key=lambda x: x[1])
                            if mr_tokens:
                                for token, value, regime in mr_tokens:
                                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                            else:
                                st.markdown("*No tokens in this category*")
                        
                        with col2:
                            st.markdown("### Random Walk Tokens")
                            rw_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if 0.4 <= v <= 0.6]
                            rw_tokens.sort(key=lambda x: x[1])
                            if rw_tokens:
                                for token, value, regime in rw_tokens:
                                    st.markdown(f"- **{token}**: <span style='color:gray'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                            else:
                                st.markdown("*No tokens in this category*")
                        
                        with col3:
                            st.markdown("### Trending Tokens")
                            tr_tokens = [(t, v, r) for t, (v, r) in latest_values.items() if v > 0.6]
                            tr_tokens.sort(key=lambda x: x[1], reverse=True)
                            if tr_tokens:
                                for token, value, regime in tr_tokens:
                                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.2f}</span> ({regime})", unsafe_allow_html=True)
                            else:
                                st.markdown("*No tokens in this category*")
                
            # Add help expander
            with st.expander("Understanding the Daily Hurst Table"):
                st.markdown("""
                ### How to Read This Table
                This table shows the Hurst exponent values for all selected tokens over the last 24 hours using 30-minute bars.
                Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
                **Color coding:**
                - **Red** (Hurst < 0.4): The token is showing mean-reverting behavior during that time period
                - **Gray** (Hurst 0.4-0.6): The token is behaving like a random walk (no clear pattern)
                - **Green** (Hurst > 0.6): The token is showing trending behavior
                **The intensity of the color indicates the strength of the pattern:**
                - Darker red = Stronger mean-reversion
                - Darker green = Stronger trending
                **Technical details:**
                - Each Hurst value is calculated by applying a rolling window of 20 one-minute bars to the closing prices, and then averaging the Hurst values of 30 one-minute bars.
                - Values are calculated using multiple methods (R/S Analysis, Variance Method, and Autocorrelation)
                - Missing values (light gray cells) indicate insufficient data for calculation
                """)
        
        # Restore the original functions
        st.set_page_config = original_st_set_page_config
        st.title = original_st_title
        st.subheader = original_st_subheader
        
    except Exception as e:
        st.error(f"Error rendering Macro View dashboard: {e}")

def render_volandhurst_dashboard():
    """Render the Vol & Hurst dashboard"""
    try:
        # Create a container for the content
        container = st.container()
        
        # Get all tokens
        all_tokens = fetch_all_tokens(engine)
        
        # Replace main Streamlit functions
        original_st_set_page_config = st.set_page_config
        original_st_title = st.title
        original_st_subheader = st.subheader
        
        # Mock these functions to prevent them from running in the tab
        st.set_page_config = lambda **kwargs: None
        st.title = lambda *args, **kwargs: None
        st.subheader = lambda *args, **kwargs: None
        
        # Run the dashboard code with the container as the context
        with container:
            # Create token selection controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Let user select tokens to display (or select all)
                select_all = st.checkbox("Select All Tokens", value=True, key="vol_hurst_select_all")
                
                if select_all:
                    selected_tokens = all_tokens
                else:
                    selected_tokens = st.multiselect(
                        "Select Tokens", 
                        all_tokens,
                        default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens,
                        key="vol_hurst_multiselect"
                    )
            
            # Set up the module with our engine
            volandhurst_module.engine = engine
            
            # Create tabs for Volatility and Hurst
            subtab1, subtab2 = st.tabs(["Volatility Analysis", "Combined Vol & Hurst"])
            
            with subtab1:
                # Volatility tab
                st.markdown("## Volatility Analysis (30min timeframe, Last 24 hours)")
                
                # Show progress bar while calculating
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Calculate volatility for each token
                token_vol_results = {}
                for i, token in enumerate(selected_tokens):
                    try:
                        progress_bar.progress((i) / len(selected_tokens))
                        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
                        result = volandhurst_module.fetch_and_calculate_volatility(token)
                        if result is not None:
                            token_vol_results[token] = result
                    except Exception as e:
                        st.error(f"Error processing token {token}: {e}")
                
                # Final progress update
                progress_bar.progress(1.0)
                status_text.text(f"Processed {len(token_vol_results)}/{len(selected_tokens)} tokens successfully")
                
                # Create volatility table
                if token_vol_results:
                    # Create table data
                    vol_table_data = {}
                    for token, df in token_vol_results.items():
                        vol_series = df.set_index('time_label')['realized_vol']
                        vol_table_data[token] = vol_series
                    
                    # Create DataFrame with all tokens
                    vol_table = pd.DataFrame(vol_table_data)
                    
                    # Apply the time blocks in the proper order
                    singapore_timezone = pytz.timezone('Asia/Singapore')
                    now_utc = datetime.now(pytz.utc)
                    now_sg = now_utc.astimezone(singapore_timezone)
                    aligned_time_blocks = volandhurst_module.generate_aligned_time_blocks(now_sg)
                    time_block_labels = [block[2] for block in aligned_time_blocks]
                    
                    available_times = set(vol_table.index)
                    ordered_times = [t for t in time_block_labels if t in available_times]
                    
                    if not ordered_times and available_times:
                        ordered_times = sorted(list(available_times), reverse=True)
                    
                    # Reindex with the ordered times
                    vol_table = vol_table.reindex(ordered_times)
                    
                    # Convert from decimal to percentage and round to 1 decimal place
                    vol_table = (vol_table * 100).round(1)
                    
                    def color_cells(val):
                        if pd.isna(val):
                            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
                        elif val < 30:  # Low volatility - green
                            intensity = max(0, min(255, int(255 * val / 30)))
                            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
                        elif val < 60:  # Medium volatility - yellow
                            intensity = max(0, min(255, int(255 * (val - 30) / 30)))
                            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
                        elif val < 100:  # High volatility - orange
                            intensity = max(0, min(255, int(255 * (val - 60) / 40)))
                            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
                        else:  # Extreme volatility - red
                            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
                    
                    styled_table = vol_table.style.applymap(color_cells)
                    st.markdown("### Volatility Table")
                    st.markdown("#### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
                    st.markdown("Values shown as annualized volatility percentage")
                    st.dataframe(styled_table, height=600, use_container_width=True)
                    
                    # Show volatility ranking
                    st.subheader("Volatility Ranking (24-Hour Average, Descending Order)")
                    
                    # Calculate ranking data
                    ranking_data = []
                    for token, df in token_vol_results.items():
                        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
                            avg_vol = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
                            vol_regime = df['avg_vol_desc'].iloc[0]
                            max_vol = df['realized_vol'].max()
                            min_vol = df['realized_vol'].min()
                            ranking_data.append({
                                'Token': token,
                                'Avg Vol (%)': (avg_vol * 100).round(1),
                                'Regime': vol_regime,
                                'Max Vol (%)': (max_vol * 100).round(1),
                                'Min Vol (%)': (min_vol * 100).round(1),
                                'Vol Range (%)': ((max_vol - min_vol) * 100).round(1)
                            })
                    
                    if ranking_data:
                        ranking_df = pd.DataFrame(ranking_data)
                        ranking_df = ranking_df.sort_values(by='Avg Vol (%)', ascending=False)
                        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
                        
                        # Reset the index to remove it
                        ranking_df = ranking_df.reset_index(drop=True)
                        
                        # Apply styling for the ranking table
                        def color_regime(val):
                            if 'Low' in val:
                                return 'color: green'
                            elif 'Medium' in val:
                                return 'color: #aaaa00'
                            elif 'High' in val:
                                return 'color: orange'
                            elif 'Extreme' in val:
                                return 'color: red'
                            return ''
                        
                        def color_value(val):
                            if pd.isna(val):
                                return ''
                            elif val < 30:
                                return 'color: green'
                            elif val < 60:
                                return 'color: #aaaa00'
                            elif val < 100:
                                return 'color: orange'
                            else:
                                return 'color: red'
                        
                        # Apply styling
                        styled_ranking = ranking_df.style\
                            .applymap(color_regime, subset=['Regime'])\
                            .applymap(color_value, subset=['Avg Vol (%)', 'Max Vol (%)', 'Min Vol (%)'])
                        
                        # Display the styled dataframe
                        st.dataframe(styled_ranking, height=500, use_container_width=True)
                    
                    # Identify extreme volatility events
                    st.subheader("Extreme Volatility Events (>= 100% Annualized)")
                    
                    extreme_events = []
                    for token, df in token_vol_results.items():
                        if not df.empty and 'is_extreme' in df.columns:
                            extreme_periods = df[df['is_extreme']]
                            for idx, row in extreme_periods.iterrows():
                                vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                                
                                extreme_events.append({
                                    'Token': token,
                                    'Time': time_label,
                                    'Volatility (%)': round(vol_value * 100, 1),
                                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M')
                                })
                    
                    if extreme_events:
                        extreme_df = pd.DataFrame(extreme_events)
                        extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)
                        extreme_df = extreme_df.reset_index(drop=True)
                        
                        # Display the dataframe
                        st.dataframe(extreme_df, height=300, use_container_width=True)
                        
                        # Create a more visually appealing list of extreme events
                        st.markdown("### Extreme Volatility Events Detail")
                        
                        # Only process top 10 events if there are any
                        top_events = extreme_events[:min(10, len(extreme_events))]
                        for i, event in enumerate(top_events):
                            token = event['Token']
                            time = event['Time']
                            vol = event['Volatility (%)']
                            date = event['Full Timestamp'].split(' ')[0]
                            
                            st.markdown(f"**{i+1}. {token}** at **{time}** on {date}: <span style='color:red; font-weight:bold;'>{vol}%</span> volatility", unsafe_allow_html=True)
                        
                        if len(extreme_events) > 10:
                            st.markdown(f"*... and {len(extreme_events) - 10} more extreme events*")
                    else:
                        st.info("No extreme volatility events detected in the selected tokens.")
                    
                    # Show volatility distribution
                    st.subheader("24-Hour Average Volatility Overview")
                    avg_values = {}
                    for token, df in token_vol_results.items():
                        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
                            avg = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
                            regime = df['avg_vol_desc'].iloc[0]
                            avg_values[token] = (avg, regime)
                    
                    if avg_values:
                        low_vol = sum(1 for v, r in avg_values.values() if v < 0.3)
                        medium_vol = sum(1 for v, r in avg_values.values() if 0.3 <= v < 0.6)
                        high_vol = sum(1 for v, r in avg_values.values() if 0.6 <= v < 1.0)
                        extreme_vol = sum(1 for v, r in avg_values.values() if v >= 1.0)
                        total = low_vol + medium_vol + high_vol + extreme_vol
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Low Vol", f"{low_vol} ({low_vol/total*100:.1f}%)")
                        col2.metric("Medium Vol", f"{medium_vol} ({medium_vol/total*100:.1f}%)")
                        col3.metric("High Vol", f"{high_vol} ({high_vol/total*100:.1f}%)")
                        col4.metric("Extreme Vol", f"{extreme_vol} ({extreme_vol/total*100:.1f}%)")
                        
                        # Create pie chart
                        labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
                        values = [low_vol, medium_vol, high_vol, extreme_vol]
                        colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values, 
                            marker=dict(colors=colors, line=dict(color='#000000', width=2)), 
                            textinfo='label+percent', 
                            hole=.3
                        )])
                        
                        fig.update_layout(
                            title="24-Hour Average Volatility Distribution",
                            height=400,
                            font=dict(color="#000000", size=12),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            with subtab2:
                # Combined Vol & Hurst tab - displays correlation between volatility and Hurst exponent
                st.markdown("## Combined Volatility & Hurst Analysis")
                st.markdown("This tab shows the relationship between volatility and market regime (Hurst exponent).")
                
                # Run Hurst calculations similar to the Macro View tab if needed
                # For this example, we'll assume Hurst calculations are already done in the other tab
                
                # Combine the data from volatility and Hurst calculations
                if token_vol_results and 'token_results' in locals():  # Use the Hurst results if they exist
                    st.subheader("Correlation between Volatility and Market Regime")
                    
                    # Prepare combined data
                    combined_data = []
                    for token in set(token_vol_results.keys()) & set(token_results.keys()):
                        vol_df = token_vol_results[token]
                        hurst_df = token_results[token]
                        
                        # Get average values for each token
                        if not vol_df.empty and 'avg_24h_vol' in vol_df.columns:
                            avg_vol = vol_df['avg_24h_vol'].iloc[0]
                            
                            # Get the latest Hurst value
                            latest_hurst = None
                            if not hurst_df.empty and 'Hurst' in hurst_df.columns:
                                # Try to find latest value that isn't NaN
                                for idx in range(len(hurst_df)):
                                    if not pd.isna(hurst_df['Hurst'].iloc[idx]):
                                        latest_hurst = hurst_df['Hurst'].iloc[idx]
                                        break
                            
                            if latest_hurst is not None:
                                combined_data.append({
                                    'Token': token,
                                    'Avg Volatility (%)': round(avg_vol * 100, 1),
                                    'Hurst Exponent': round(latest_hurst, 2),
                                    'Vol Regime': vol_df['avg_vol_desc'].iloc[0],
                                    'Market Regime': hurst_df['regime_desc'].iloc[0] if 'regime_desc' in hurst_df.columns else "Unknown"
                                })
                    
                    if combined_data:
                        # Create DataFrame
                        combined_df = pd.DataFrame(combined_data)
                        
                        # Create scatter plot
                        fig = px.scatter(
                            combined_df,
                            x='Hurst Exponent',
                            y='Avg Volatility (%)',
                            color='Token',
                            hover_data=['Token', 'Vol Regime', 'Market Regime'],
                            title="Volatility vs. Hurst Exponent",
                            labels={'Avg Volatility (%)': 'Annualized Volatility (%)', 'Hurst Exponent': 'Hurst Exponent'},
                        )
                        
                        # Add vertical lines for regime boundaries
                        fig.add_vline(x=0.4, line_width=1, line_dash="dash", line_color="red")
                        fig.add_vline(x=0.6, line_width=1, line_dash="dash", line_color="green")
                        
                        # Add horizontal lines for volatility boundaries
                        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
                        fig.add_hline(y=60, line_width=1, line_dash="dash", line_color="orange")
                        fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="red")
                        
                        # Add annotations
                        fig.add_annotation(x=0.2, y=115, text="Mean-Reverting", showarrow=False, font=dict(color="red"))
                        fig.add_annotation(x=0.5, y=115, text="Random Walk", showarrow=False, font=dict(color="black"))
                        fig.add_annotation(x=0.8, y=115, text="Trending", showarrow=False, font=dict(color="green"))
                        
                        fig.add_annotation(x=0.9, y=15, text="Low Vol", showarrow=False, font=dict(color="green"))
                        fig.add_annotation(x=0.9, y=45, text="Medium Vol", showarrow=False, font=dict(color="orange"))
                        fig.add_annotation(x=0.9, y=80, text="High Vol", showarrow=False, font=dict(color="orange"))
                        fig.add_annotation(x=0.9, y=110, text="Extreme Vol", showarrow=False, font=dict(color="red"))
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            xaxis=dict(range=[0, 1]),
                            yaxis=dict(range=[0, 120]),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display the data table
                        st.subheader("Combined Metrics by Token")
                        
                        # Apply styling
                        def color_hurst(val):
                            if pd.isna(val):
                                return ''
                            elif val < 0.4:
                                return 'color: red'
                            elif val > 0.6:
                                return 'color: green'
                            else:
                                return 'color: black'  # Random walk
                        
                        def color_vol(val):
                            if pd.isna(val):
                                return ''
                            elif val < 30:
                                return 'color: green'
                            elif val < 60:
                                return 'color: #aaaa00'
                            elif val < 100:
                                return 'color: orange'
                            else:
                                return 'color: red'
                        
                        styled_combined = combined_df.style\
                            .applymap(color_hurst, subset=['Hurst Exponent'])\
                            .applymap(color_vol, subset=['Avg Volatility (%)'])
                        
                        st.dataframe(styled_combined, height=500, use_container_width=True)
                        
                    else:
                        st.warning("Insufficient data to create the combined analysis.")
                else:
                    st.warning("Please run both Volatility and Hurst analyses to see the combined view.")
            
            # Add explanatory information
            with st.expander("Understanding Volatility Metrics"):
                st.markdown("""
                ### How to Read the Volatility Table
                This table shows annualized volatility values for all selected tokens over the last 24 hours using 30-minute bars.
                Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
                
                **Color coding:**
                - **Green** (< 30%): Low volatility
                - **Yellow** (30-60%): Medium volatility
                - **Orange** (60-100%): High volatility
                - **Red** (> 100%): Extreme volatility
                
                **Technical details:**
                - Volatility is calculated as the standard deviation of log returns, annualized to represent the expected price variation over a year
                - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
                - The calculation uses a rolling window of 20 one-minute price points
                - The 24-hour average section shows the mean volatility across all 48 30-minute periods
                - Missing values (light gray cells) indicate insufficient data for calculation
                """)
        
        # Restore the original functions
        st.set_page_config = original_st_set_page_config
        st.title = original_st_title
        st.subheader = original_st_subheader
        
    except Exception as e:
        st.error(f"Error rendering Vol & Hurst dashboard: {e}")

def render_pnlandtrades_dashboard():
    """Render the PnL & Trades dashboard"""
    try:
        # Create a container for the content
        container = st.container()
        
        # Get all trading pairs
        all_pairs = fetch_all_pairs(engine)
        
        # Replace main Streamlit functions
        original_st_set_page_config = st.set_page_config
        original_st_title = st.title
        original_st_subheader = st.subheader
        
        # Mock these functions to prevent them from running in the tab
        st.set_page_config = lambda **kwargs: None
        st.title = lambda *args, **kwargs: None
        st.subheader = lambda *args, **kwargs: None
        
        # Run the dashboard code with the container as the context
        with container:
            # Create controls for pair selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Let user select pairs to display (or select all)
                select_all = st.checkbox("Select All Pairs", value=True, key="pnl_trades_select_all")
                
                if select_all:
                    selected_pairs = all_pairs
                else:
                    selected_pairs = st.multiselect(
                        "Select Pairs", 
                        all_pairs,
                        default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs,
                        key="pnl_trades_multiselect"
                    )
            
            # Set up the module with our engine
            pnlandtrades_module.engine = engine
            
            if not selected_pairs:
                st.warning("Please select at least one pair")
                return
            
            # Show progress bar while calculating
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate trade count and platform PNL for each pair
            pair_results = {}
            for i, pair_name in enumerate(selected_pairs):
                try:
                    progress_bar.progress((i) / len(selected_pairs))
                    status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
                    
                    # Fetch trade count data
                    trade_data = pnlandtrades_module.fetch_trade_counts(pair_name)
                    
                    # Fetch platform PNL data
                    pnl_data = pnlandtrades_module.fetch_platform_pnl(pair_name)
                    
                    # Combine data
                    combined_data = pnlandtrades_module.combine_data(trade_data, pnl_data)
                    
                    if combined_data is not None:
                        pair_results[pair_name] = combined_data
                except Exception as e:
                    st.error(f"Error processing pair {pair_name}: {e}")
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")
            
            # Create trades and PNL tables
            if pair_results:
                singapore_timezone = pytz.timezone('Asia/Singapore')
                now_utc = datetime.now(pytz.utc)
                now_sg = now_utc.astimezone(singapore_timezone)
                aligned_time_blocks = pnlandtrades_module.generate_aligned_time_blocks(now_sg)
                time_block_labels = [block[2] for block in aligned_time_blocks]
                
                # Create trade count table
                trade_count_data = {}
                for pair_name, df in pair_results.items():
                    if 'trade_count' in df.columns:
                        trade_count_data[pair_name] = df['trade_count']
                
                # Create DataFrame with all pairs
                trade_count_table = pd.DataFrame(trade_count_data)
                
                # Apply the time blocks in the proper order
                available_times = set(trade_count_table.index)
                ordered_times = [t for t in time_block_labels if t in available_times]
                
                # If no matches are found in aligned blocks, fallback to the available times
                if not ordered_times and available_times:
                    ordered_times = sorted(list(available_times), reverse=True)
                
                # Reindex with the ordered times
                trade_count_table = trade_count_table.reindex(ordered_times)
                
                # Round to integers - trade counts should be whole numbers
                trade_count_table = trade_count_table.round(0).astype('Int64')  # Using Int64 to handle NaN values properly
                
                # Display the trades table with styling
                def color_trade_cells(val):
                    if pd.isna(val) or val == 0:
                        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
                    elif val < 10:  # Low activity
                        intensity = max(0, min(255, int(255 * val / 10)))
                        return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
                    elif val < 50:  # Medium activity
                        intensity = max(0, min(255, int(255 * (val - 10) / 40)))
                        return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
                    elif val < 100:  # High activity
                        intensity = max(0, min(255, int(255 * (val - 50) / 50)))
                        return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
                    else:  # Very high activity
                        return 'background-color: rgba(255, 0, 0, 0.7); color: white'
                
                styled_trade_table = trade_count_table.style.applymap(color_trade_cells)
                st.markdown("## User Trades Table (30min timeframe, Last 24 hours)")
                st.markdown("### Color Legend: <span style='color:green'>Low Activity</span>, <span style='color:#aaaa00'>Medium Activity</span>, <span style='color:orange'>High Activity</span>, <span style='color:red'>Very High Activity</span>", unsafe_allow_html=True)
                st.markdown("Values shown as number of trades per 30-minute period")
                st.dataframe(styled_trade_table, height=600, use_container_width=True)
                
                # Create Platform PNL table
                pnl_data = {}
                for pair_name, df in pair_results.items():
                    if 'platform_pnl' in df.columns:
                        if df['platform_pnl'].abs().sum() > 0:
                            pnl_data[pair_name] = df['platform_pnl']
                
                # Create DataFrame with all pairs
                pnl_table = pd.DataFrame(pnl_data)
                
                # Apply the time blocks in the proper order
                pnl_table = pnl_table.reindex(ordered_times)
                
                # Round to 2 decimal places for display
                pnl_table = pnl_table.round(0).astype(int)
                
                # Apply styling
                def color_pnl_cells(val):
                    if pd.isna(val) or val == 0:
                        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
                    elif val < -1000:  # Large negative PNL (loss) - red
                        return f'background-color: rgba(255, 0, 0, 0.9); color: white'
                    elif val < 0:  # Small negative PNL (loss) - light red
                        intensity = max(0, min(255, int(255 * abs(val) / 1000)))
                        return f'background-color: rgba(255, {100-intensity}, {100-intensity}, 0.9); color: black'
                    elif val < 1000:  # Small positive PNL (profit) - light green
                        intensity = max(0, min(255, int(255 * val / 1000)))
                        return f'background-color: rgba({100-intensity}, 180, {100-intensity}, 0.9); color: black'
                    else:  # Large positive PNL (profit) - green
                        return 'background-color: rgba(0, 120, 0, 0.7); color: black'
                
                styled_pnl_table = pnl_table.style.applymap(color_pnl_cells)
                st.markdown("## Platform PNL Table (USD, 30min timeframe, Last 24 hours)")
                st.markdown("### Color Legend: <span style='color:red'>Loss</span>, <span style='color:#ff9999'>Small Loss</span>, <span style='color:#99ff99'>Small Profit</span>, <span style='color:green'>Large Profit</span>", unsafe_allow_html=True)
                st.markdown("Values shown in USD")
                st.dataframe(styled_pnl_table, height=600, use_container_width=True)
                
                # Create summary tables with improved legibility
                st.subheader("Summary Statistics (Last 24 Hours)")
                
                # Add a separator
                st.markdown("---")
                
                # Prepare Trades Summary data
                trades_summary = {}
                for pair_name, df in pair_results.items():
                    if 'trade_count' in df.columns:
                        total_trades = df['trade_count'].sum()
                        max_trades = df['trade_count'].max()
                        max_trades_time = df['trade_count'].idxmax() if max_trades > 0 else "N/A"
                        
                        trades_summary[pair_name] = {
                            'Total Trades': int(total_trades),
                            'Max Trades in 30min': int(max_trades),
                            'Busiest Time': max_trades_time if max_trades > 0 else "N/A",
                            'Avg Trades per 30min': round(df['trade_count'].mean(), 1)
                        }
                
                # Trading Activity Summary with improved formatting
                if trades_summary:
                    # Convert to DataFrame and sort
                    trades_summary_df = pd.DataFrame(trades_summary).T
                    trades_summary_df = trades_summary_df.sort_values(by='Total Trades', ascending=False)
                    
                    # Format the dataframe for better legibility
                    trades_summary_df = trades_summary_df.rename(columns={
                        'Total Trades': '📊 Total Trades',
                        'Max Trades in 30min': '⏱️ Max Trades (30min)',
                        'Busiest Time': '🕒 Busiest Time',
                        'Avg Trades per 30min': '📈 Avg Trades/30min'
                    })
                    
                    # Add a clear section header
                    st.markdown("### 📊 Trading Activity Summary")
                    
                    # Display the styled dataframe
                    st.dataframe(
                        trades_summary_df.style.format({
                            '📈 Avg Trades/30min': '{:.1f}'
                        }).set_properties(**{
                            'font-size': '16px',
                            'text-align': 'center',
                            'background-color': '#f0f2f6'
                            }),
                        height=350,
                        use_container_width=True
                    )
                
                # Prepare PNL Summary data
                pnl_summary = {}
                for pair_name, df in pair_results.items():
                    if 'platform_pnl' in df.columns:
                        total_pnl = df['platform_pnl'].sum()
                        max_pnl = df['platform_pnl'].max()
                        min_pnl = df['platform_pnl'].min()
                        max_pnl_time = df['platform_pnl'].idxmax() if abs(max_pnl) > 0 else "N/A"
                        min_pnl_time = df['platform_pnl'].idxmin() if abs(min_pnl) > 0 else "N/A"
                        
                        pnl_summary[pair_name] = {
                            'Total PNL (USD)': round(total_pnl, 2),
                            'Max Profit in 30min': round(max_pnl, 2),
                            'Max Profit Time': max_pnl_time if abs(max_pnl) > 0 else "N/A",
                            'Max Loss in 30min': round(min_pnl, 2),
                            'Max Loss Time': min_pnl_time if abs(min_pnl) > 0 else "N/A",
                            'Avg PNL per 30min': round(df['platform_pnl'].mean(), 2)
                        }
                
                # PNL Summary with improved formatting
                if pnl_summary:
                    # Convert to DataFrame and sort
                    pnl_summary_df = pd.DataFrame(pnl_summary).T
                    pnl_summary_df = pnl_summary_df.sort_values(by='Total PNL (USD)', ascending=False)
                    
                    # Format the dataframe for better legibility
                    pnl_summary_df = pnl_summary_df.rename(columns={
                        'Total PNL (USD)': '💰 Total PNL (USD)',
                        'Max Profit in 30min': '📈 Max Profit (30min)',
                        'Max Profit Time': '⏱️ Max Profit Time',
                        'Max Loss in 30min': '📉 Max Loss (30min)',
                        'Max Loss Time': '⏱️ Max Loss Time', 
                        'Avg PNL per 30min': '📊 Avg PNL/30min'
                    })
                    
                    # Style the dataframe for better legibility
                    styled_pnl_df = pnl_summary_df.style.format({
                        '💰 Total PNL (USD)': '${:,.2f}',
                        '📈 Max Profit (30min)': '${:,.2f}',
                        '📉 Max Loss (30min)': '${:,.2f}',
                        '📊 Avg PNL/30min': '${:,.2f}'
                    })
                    
                    # Apply conditional formatting
                    def highlight_profits(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                return 'color: green; font-weight: bold'
                            elif val < 0:
                                return 'color: red; font-weight: bold'
                        return ''
                    
                    styled_pnl_df = styled_pnl_df.applymap(highlight_profits, subset=['💰 Total PNL (USD)', '📈 Max Profit (30min)', '📉 Max Loss (30min)', '📊 Avg PNL/30min'])
                    
                    # Add a clear section header
                    st.markdown("### 💰 Platform PNL Summary")
                    
                    # Display the styled dataframe
                    st.dataframe(
                        styled_pnl_df.set_properties(**{
                            'font-size': '16px',
                            'text-align': 'center',
                            'background-color': '#f0f2f6'
                        }),
                        height=350,
                        use_container_width=True
                    )
                
                # Add visualizations for top performers
                st.subheader("Top Performers")
                
                if pnl_summary:
                    # Get top 5 pairs by PNL
                    top_pnl_pairs = list(pnl_summary_df.sort_values(by='💰 Total PNL (USD)', ascending=False).head(5).index)
                    
                    # Create bar chart
                    top_pnl_data = []
                    for pair in top_pnl_pairs:
                        top_pnl_data.append({
                            'Pair': pair,
                            'PNL (USD)': pnl_summary[pair]['Total PNL (USD)']
                        })
                    
                    if top_pnl_data:
                        top_pnl_df = pd.DataFrame(top_pnl_data)
                        
                        fig = px.bar(
                            top_pnl_df,
                            x='Pair',
                            y='PNL (USD)',
                            title="Top 5 Pairs by PNL",
                            color='PNL (USD)',
                            color_continuous_scale=['red', 'green'],
                            text='PNL (USD)'
                        )
                        
                        fig.update_layout(
                            height=400,
                            xaxis_title="Trading Pair",
                            yaxis_title="Total PNL (USD)"
                        )
                        
                        fig.update_traces(
                            texttemplate='%{y:$.2f}',
                            textposition='outside'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Add help expander
            with st.expander("Understanding the PNL & Trades Dashboard"):
                st.markdown("""
                ## 📊 How to Use This Dashboard
                
                This dashboard shows trading activity and platform profit/loss (PNL) across all selected trading pairs using 30-minute intervals over the past 24 hours (Singapore time).
                
                ### Main Tables
                - **User Trades Table**: Shows the number of trades completed in each 30-minute period
                - **Platform PNL Table**: Shows the platform's profit/loss in each 30-minute period
                
                ### Color Coding
                - **Trades Table**: 
                  - 🟩 Green: Low activity
                  - 🟨 Yellow: Medium activity
                  - 🟧 Orange: High activity
                  - 🟥 Red: Very high activity
                
                - **PNL Table**: 
                  - 🟥 Red: Significant loss
                  - 🟠 Light red: Small loss
                  - 🟢 Light green: Small profit
                  - 🟩 Green: Significant profit
                
                ### Key Insights
                - **Trading Activity Summary**: See which pairs have the highest trading volume
                - **Platform PNL Summary**: Identify which pairs are most profitable
                
                ### Technical Details
                - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
                - All values are shown in USD
                - The dashboard refreshes when you click the "Refresh" button
                - Singapore timezone (UTC+8) is used throughout
                """)
        
        # Restore the original functions
        st.set_page_config = original_st_set_page_config
        st.title = original_st_title
        st.subheader = original_st_subheader
        
    except Exception as e:
        st.error(f"Error rendering PnL & Trades dashboard: {e}")

def render_platformpnl_dashboard():
    """Render the Platform PnL Cumulative dashboard"""
    try:
        # Create a container for the content
        container = st.container()
        
        # Get all trading pairs
        all_pairs = fetch_all_pairs(engine)
        
        # Replace main Streamlit functions
        original_st_set_page_config = st.set_page_config
        original_st_title = st.title
        original_st_subheader = st.subheader
        
        # Mock these functions to prevent them from running in the tab
        st.set_page_config = lambda **kwargs: None
        st.title = lambda *args, **kwargs: None
        st.subheader = lambda *args, **kwargs: None
        
        # Run the dashboard code with the container as the context
        with container:
            # Create controls for pair selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Let user select pairs to display (or select all)
                select_all = st.checkbox("Select All Pairs", value=True, key="platform_pnl_select_all")
                
                if select_all:
                    selected_pairs = all_pairs
                else:
                    selected_pairs = st.multiselect(
                        "Select Pairs", 
                        all_pairs,
                        default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs,
                        key="platform_pnl_multiselect"
                    )
            
            # Set up the module with our engine
            platformpnl_module.engine = engine
            
            if not selected_pairs:
                st.warning("Please select at least one pair")
                return
            
            # Calculate time boundaries
            time_boundaries = platformpnl_module.get_time_boundaries()
            
            # Show progress bar while calculating
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Gather PNL data for all pairs and time periods
            results = {}
            periods = ["today", "yesterday", "day_before_yesterday", "this_week", "all_time"]
            
            for i, pair_name in enumerate(selected_pairs):
                progress_bar.progress((i) / len(selected_pairs))
                status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
                
                pair_data = {"pair_name": pair_name}
                
                for period in periods:
                    start_time = time_boundaries[period]["start"]
                    end_time = time_boundaries[period]["end"]
                    
                    # Fetch PNL data for this pair and time period
                    period_data = platformpnl_module.fetch_pnl_data(pair_name, start_time, end_time)
                    
                    # Store the results
                    pair_data[f"{period}_pnl"] = period_data["pnl"]
                    pair_data[f"{period}_trades"] = period_data["trades"]
                
                results[pair_name] = pair_data
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processed {len(results)}/{len(selected_pairs)} pairs successfully")
            
            # Create DataFrame from results
            pnl_df = pd.DataFrame([results[pair] for pair in selected_pairs])
            
            # If DataFrame is empty, show warning and stop
            if pnl_df.empty:
                st.warning("No PNL data found for the selected pairs and time periods.")
                return
            
            # Reformat the DataFrame for display
            display_df = pd.DataFrame({
                'Trading Pair': pnl_df['pair_name'],
                'Today PNL (USD)': pnl_df['today_pnl'].round(2),
                'Today Trades': pnl_df['today_trades'],
                'Yesterday PNL (USD)': pnl_df['yesterday_pnl'].round(2),
                'Yesterday Trades': pnl_df['yesterday_trades'],
                'Day Before PNL (USD)': pnl_df['day_before_yesterday_pnl'].round(2),
                'Day Before Trades': pnl_df['day_before_yesterday_trades'],
                'Week PNL (USD)': pnl_df['this_week_pnl'].round(2),
                'Week Trades': pnl_df['this_week_trades'],
                'All Time PNL (USD)': pnl_df['all_time_pnl'].round(2),
                'All Time Trades': pnl_df['all_time_trades'],
            })
            
            # Format the display DataFrame
            display_df = platformpnl_module.format_display_df(display_df)
            
            # Sort DataFrame by Today's PNL (descending)
            display_df = display_df.sort_values(by='Today PNL (USD)', ascending=False)
            
            # Create tabs for different views
            subtab1, subtab2, subtab3 = st.tabs(["Main Dashboard", "Detailed View", "Statistics & Insights"])
            
            with subtab1:
                # Main Dashboard View
                st.subheader("PNL Overview by Trading Pair")
                
                # Create a simplified display DataFrame for the main dashboard
                main_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']]
                
                # Apply styling
                def color_pnl_cells(val):
                    """Color cells based on PNL value."""
                    if pd.isna(val) or val == 0:
                        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
                    elif val < -1000:  # Large negative PNL (loss) - red
                        return 'background-color: rgba(255, 0, 0, 0.9); color: white'
                    elif val < 0:  # Small negative PNL (loss) - light red
                        intensity = max(0, min(255, int(255 * abs(val) / 1000)))
                        return f'background-color: rgba(255, {180-intensity}, {180-intensity}, 0.7); color: black'
                    elif val < 1000:  # Small positive PNL (profit) - light green
                        intensity = max(0, min(255, int(255 * val / 1000)))
                        return f'background-color: rgba({180-intensity}, 255, {180-intensity}, 0.7); color: black'
                    else:  # Large positive PNL (profit) - green
                        return 'background-color: rgba(0, 200, 0, 0.8); color: black'
                
                styled_df = main_df.style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']
                ).format({
                    'Today PNL (USD)': '${:,.2f}',
                    'Yesterday PNL (USD)': '${:,.2f}',
                    'Week PNL (USD)': '${:,.2f}',
                    'All Time PNL (USD)': '${:,.2f}'
                })
                
                # Display the styled DataFrame
                st.dataframe(styled_df, height=600, use_container_width=True)
                
                # Create summary cards
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_today_pnl = display_df['Today PNL (USD)'].sum()
                    st.metric(
                        "Total Today PNL", 
                        f"${total_today_pnl:,.2f}", 
                        delta=f"{(total_today_pnl - display_df['Yesterday PNL (USD)'].sum()):,.2f}"
                    )
                
                with col2:
                    total_yesterday_pnl = display_df['Yesterday PNL (USD)'].sum()
                    st.metric(
                        "Total Yesterday PNL", 
                        f"${total_yesterday_pnl:,.2f}"
                    )
                
                with col3:
                    total_week_pnl = display_df['Week PNL (USD)'].sum()
                    daily_avg = total_week_pnl / 7
                    st.metric(
                        "Week PNL (7 days)", 
                        f"${total_week_pnl:,.2f}",
                        delta=f"${daily_avg:,.2f}/day"
                    )
                
                with col4:
                    total_all_time_pnl = display_df['All Time PNL (USD)'].sum()
                    st.metric(
                        "All Time PNL", 
                        f"${total_all_time_pnl:,.2f}"
                    )
                
                # Create a visualization of top and bottom performers today
                st.subheader("Today's Top Performers")
                
                # Filter out zero PNL pairs
                non_zero_today = display_df[display_df['Today PNL (USD)'] != 0].copy()
                
                # Get top 5 and bottom 5 performers
                top_5 = non_zero_today.nlargest(5, 'Today PNL (USD)')
                bottom_5 = non_zero_today.nsmallest(5, 'Today PNL (USD)')
                
                # Plot top and bottom performers
                fig = go.Figure()
                
                # Top performers
                fig.add_trace(go.Bar(
                    x=top_5['Trading Pair'],
                    y=top_5['Today PNL (USD)'],
                    name='Top Performers',
                    marker_color='green'
                ))
                
                # Bottom performers
                fig.add_trace(go.Bar(
                    x=bottom_5['Trading Pair'],
                    y=bottom_5['Today PNL (USD)'],
                    name='Bottom Performers',
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title="Top and Bottom Performers Today",
                    xaxis_title="Trading Pair",
                    yaxis_title="PNL (USD)",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                # Detailed View
                st.subheader("Detailed PNL and Trade Data")
                
                # Create a detailed display DataFrame
                detailed_df = display_df[['Trading Pair', 
                                        'Today PNL (USD)', 'Today Trades', 'Today PNL/Trade',
                                        'Yesterday PNL (USD)', 'Yesterday Trades', 'Yesterday PNL/Trade',
                                        'Week PNL (USD)', 'Week Trades', 'Week PNL/Trade',
                                        'All Time PNL (USD)', 'All Time Trades', 'All Time PNL/Trade']]
                
                # Apply styling
                styled_detailed_df = detailed_df.style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)',
                            'Today PNL/Trade', 'Yesterday PNL/Trade', 'Week PNL/Trade', 'All Time PNL/Trade']
                ).format({
                    'Today PNL (USD)': '${:,.2f}',
                    'Yesterday PNL (USD)': '${:,.2f}',
                    'Week PNL (USD)': '${:,.2f}',
                    'All Time PNL (USD)': '${:,.2f}',
                    'Today PNL/Trade': '${:,.2f}',
                    'Yesterday PNL/Trade': '${:,.2f}',
                    'Week PNL/Trade': '${:,.2f}',
                    'All Time PNL/Trade': '${:,.2f}'
                })
                
                # Display the styled DataFrame
                st.dataframe(styled_detailed_df, height=600, use_container_width=True)
                
                # Additional visualizations can be added here
            
            with subtab3:
                # Statistics & Insights
                st.subheader("PNL Statistics and Insights")
                
                # Create a statistics DataFrame
                stats_df = pd.DataFrame({
                    'Metric': [
                        'Total Trading Pairs',
                        'Profitable Pairs Today',
                        'Unprofitable Pairs Today',
                        'Most Profitable Pair Today',
                        'Least Profitable Pair Today',
                        'Highest PNL/Trade Today',
                        'Average PNL/Trade (All Pairs)',
                        'Total Platform PNL Today',
                        'Total Platform PNL Yesterday',
                        'Week-to-Date PNL',
                        'Estimated Monthly PNL (based on week)',
                        'Total Trades Today',
                        'Total Trades Yesterday',
                        'Week-to-Date Trades'
                    ],
                    'Value': [
                        len(display_df),
                        len(display_df[display_df['Today PNL (USD)'] > 0]),
                        len(display_df[display_df['Today PNL (USD)'] < 0]),
                        display_df.loc[display_df['Today PNL (USD)'].idxmax()]['Trading Pair'] if not display_df.empty else 'N/A',
                        display_df.loc[display_df['Today PNL (USD)'].idxmin()]['Trading Pair'] if not display_df.empty else 'N/A',
                        f"${display_df['Today PNL/Trade'].max():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
                        f"${display_df['Today PNL/Trade'].mean():.2f}" if 'Today PNL/Trade' in display_df.columns else 'N/A',
                        f"${display_df['Today PNL (USD)'].sum():.2f}",
                        f"${display_df['Yesterday PNL (USD)'].sum():.2f}",
                        f"${display_df['Week PNL (USD)'].sum():.2f}",
                        f"${(display_df['Week PNL (USD)'].sum() / 7 * 30):.2f}",
                        f"{display_df['Today Trades'].sum():,}",
                        f"{display_df['Yesterday Trades'].sum():,}",
                        f"{display_df['Week Trades'].sum():,}"
                    ]
                })
                
                # Display statistics
                st.dataframe(stats_df, hide_index=True, height=400, use_container_width=True)
                
                # Visualize PNL breakdown by time period
                st.subheader("PNL Breakdown by Time Period")
                
                # For Top 10 Pairs
                top_10_pairs = display_df.nlargest(10, 'Week PNL (USD)')['Trading Pair'].tolist()
                top_10_df = display_df[display_df['Trading Pair'].isin(top_10_pairs)].copy()
                
                # Prepare data for stacked bar chart
                chart_data = []
                for pair in top_10_pairs:
                    pair_data = top_10_df[top_10_df['Trading Pair'] == pair].iloc[0]
                    chart_data.append({
                        'Trading Pair': pair,
                        'Today': pair_data['Today PNL (USD)'],
                        'Yesterday': pair_data['Yesterday PNL (USD)'],
                        'Rest of Week': pair_data['Week PNL (USD)'] - pair_data['Today PNL (USD)'] - pair_data['Yesterday PNL (USD)']
                    })
                
                chart_df = pd.DataFrame(chart_data)
                
                # Create the stacked bar chart
                fig = px.bar(
                    chart_df,
                    x='Trading Pair',
                    y=['Today', 'Yesterday', 'Rest of Week'],
                    title='PNL Breakdown for Top 10 Pairs',
                    labels={'value': 'PNL (USD)', 'variable': 'Time Period'},
                    barmode='group'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of the dashboard
            with st.expander("Understanding the PNL Dashboard"):
                st.markdown("""
                ## 📊 How to Use This PNL Dashboard
                
                This dashboard shows platform profit and loss (PNL) across all selected trading pairs over different time periods.
                
                ### Time Periods
                - **Today**: From midnight Singapore time (SGT) until now
                - **Yesterday**: Full 24 hours from midnight to midnight SGT
                - **This Week**: Last 7 days including today
                - **All Time**: Cumulative PNL since records began
                
                ### Color Coding
                - 🟩 **Green**: Profit (darker green for higher profits)
                - 🟥 **Red**: Loss (darker red for higher losses)
                - ⬜ **Grey**: No activity/zero PNL
                
                ### Key Metrics
                - **PNL (USD)**: Platform's profit/loss in USD for each time period
                - **Trades**: Number of trades executed in each time period
                - **PNL/Trade**: Average profit per trade
                
                ### Dashboard Tabs
                - **Main Dashboard**: Quick overview of PNL by time period
                - **Detailed View**: Complete breakdown including trade counts and per-trade metrics
                - **Statistics & Insights**: Trends, correlations, and deeper analysis
                """)
        
        # Restore the original functions
        st.set_page_config = original_st_set_page_config
        st.title = original_st_title
        st.subheader = original_st_subheader
        
    except Exception as e:
        st.error(f"Error rendering Platform PNL dashboard: {e}")

def render_all_in_one_dashboard():
    """Render a consolidated view of key metrics from all dashboards"""
    try:
        # Create a container for the content
        container = st.container()
        
        with container:
            st.markdown("## All-In-One Dashboard Overview")
            st.markdown("This dashboard provides a comprehensive view of key metrics from all dashboards in one place.")
            
            # Create two columns for top metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Market Overview")
                # This would show a summary of market statistics, regime distributions, etc.
                
                # Placeholder for market data - you would integrate actual data here
                st.info("Market summary will be integrated here, showing the distribution of market regimes, volatility levels, and spread metrics.")
                
                # Example visualization - replace with actual data integration
                labels = ['Mean-Reverting', 'Random Walk', 'Trending']
                values = [30, 40, 30]  # Example values
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=.3, 
                    marker_colors=['red', 'gray', 'green']
                )])
                
                fig.update_layout(
                    title="Current Market Regime Distribution",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 💰 Platform Performance")
                # This would show a summary of PNL and trading metrics
                
                # Placeholder for PNL data - you would integrate actual data here
                st.info("Platform performance summary will be integrated here, showing total PNL, trade counts, and top performing pairs.")
                
                # Example visualization - replace with actual data integration
                x = ['Today', 'Yesterday', 'Day Before']
                y = [5000, 4500, 6000]  # Example values
                
                fig = go.Figure(data=[go.Bar(
                    x=x, 
                    y=y, 
                    marker_color=['green', 'green', 'green']
                )])
                
                fig.update_layout(
                    title="Platform PNL by Day",
                    height=300,
                    yaxis_title="PNL (USD)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create tabs for different metric categories
            subtab1, subtab2, subtab3 = st.tabs(["Market Metrics", "Trading Activity", "Opportunity Finder"])
            
            with subtab1:
                st.markdown("### Market Metrics Dashboard")
                
                # Create columns for key metric displays
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown("#### Spread Analysis")
                    st.info("Spread metrics will be displayed here, showing the best opportunities based on exchange price differences.")
                
                with metric_col2:
                    st.markdown("#### Volatility Analysis")
                    st.info("Volatility metrics will be displayed here, highlighting tokens with unusual price movement patterns.")
                
                with metric_col3:
                    st.markdown("#### Hurst Analysis")
                    st.info("Hurst exponent metrics will be displayed here, indicating which tokens are trending vs. mean-reverting.")
            
            with subtab2:
                st.markdown("### Trading Activity Dashboard")
                
                # Create columns for trading metrics
                trade_col1, trade_col2 = st.columns(2)
                
                with trade_col1:
                    st.markdown("#### Recent Trading Volume")
                    st.info("Recent trading volumes will be displayed here, showing the most active tokens.")
                
                with trade_col2:
                    st.markdown("#### PNL Performance")
                    st.info("PNL performance metrics will be displayed here, highlighting the most profitable trading pairs.")
                
                # Additional trading metrics
                st.markdown("#### Trading Activity Heatmap")
                st.info("A heatmap showing trading activity by time of day will be displayed here.")
            
            with subtab3:
                st.markdown("### Opportunity Finder")
                st.markdown("This tab helps identify potential trading opportunities based on combined metrics from all dashboards.")
                
                # Create sliders for filtering opportunities
                st.subheader("Filter Criteria")
                
                vol_threshold = st.slider("Minimum Volatility (%)", 0, 100, 30, 5)
                regime_options = st.multiselect("Market Regime", ["Mean-Reverting", "Random Walk", "Trending"], ["Trending"])
                min_spread = st.slider("Minimum Spread Difference (bps)", 0, 50, 10, 1)
                
                # Button to find opportunities
                if st.button("Find Opportunities", key="find_opps"):
                    st.info("Opportunity finder will analyze cross-dashboard metrics and present potential trading setups based on your criteria.")
                    
                    # Placeholder for opportunity results
                    st.markdown("#### Potential Opportunities")
                    
                    # This would be an actual data table with opportunities
                    opportunity_data = {
                        "Token/Pair": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                        "Volatility": ["45%", "62%", "87%"],
                        "Hurst": ["0.65", "0.72", "0.58"],
                        "Best Spread": ["3.2 bps", "4.1 bps", "5.7 bps"],
                        "PnL Trend": ["↑", "↑", "→"],
                        "Strategy": ["Momentum", "Momentum", "Volatility Break"]
                    }
                    
                    opportunity_df = pd.DataFrame(opportunity_data)
                    st.dataframe(opportunity_df, use_container_width=True)
            
            # Disclaimer
            st.markdown("---")
            st.markdown("*Note: This All-In-One dashboard is a consolidated view. For more detailed analysis, please refer to the individual tabs.*")
            
    except Exception as e:
        st.error(f"Error rendering All-In-One dashboard: {e}")

# --- Utility Functions ---
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_all_tokens(_engine):
    """Fetch all available tokens from the database"""
    query = """
    SELECT DISTINCT pair_name 
    FROM oracle_exchange_fee 
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
        print(f"Error fetching tokens: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default fallback

@st.cache_data(ttl=600, show_spinner="Fetching pairs...")
def fetch_all_pairs(_engine):
    """Fetch all available trading pairs from the database"""
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

# --- Run the app ---
if __name__ == "__main__":
    main()
                        
            