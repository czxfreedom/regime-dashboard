
def render_tab_0():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    refresh = st.button("Refresh Data", key="refresh_button_0")
    if refresh:
        st.cache_data.clear()
        st.experimental_rerun()
    with st.container():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz
    
    
    # --- DB CONFIG ---
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()
    
    # --- UI Setup ---
    st.title("Trading Pairs PNL Dashboard")
    st.subheader("Performance Analysis by Time Period (Singapore Time)")
    
    # Set up the timezone
    singapore_timezone = pytz.timezone('Asia/Singapore')
    
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Function to get Singapore midnight today, yesterday, and 7 days ago
    def get_time_boundaries():
        # Current time in Singapore
        now_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)
        
        # Today's midnight in Singapore
        today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Yesterday's midnight in Singapore
        yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)
        
        # 7 days ago midnight in Singapore
        week_ago_midnight_sg = today_midnight_sg - timedelta(days=7)
        
        # All time (use a far past date, e.g., 5 years ago)
        all_time_start_sg = today_midnight_sg.replace(year=today_midnight_sg.year-5)
        
        # Convert all times back to UTC for database queries
        today_midnight_utc = today_midnight_sg.astimezone(pytz.utc)
        yesterday_midnight_utc = yesterday_midnight_sg.astimezone(pytz.utc)
        day_before_yesterday_midnight_utc = (yesterday_midnight_sg - timedelta(days=1)).astimezone(pytz.utc)
        week_ago_midnight_utc = week_ago_midnight_sg.astimezone(pytz.utc)
        all_time_start_utc = all_time_start_sg.astimezone(pytz.utc)
        now_utc = now_sg.astimezone(pytz.utc)
        
        return {
            "today": {
                "start": today_midnight_utc,
                "end": now_utc,
                "label": f"Today ({today_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "yesterday": {
                "start": yesterday_midnight_utc,
                "end": today_midnight_utc,
                "label": f"Yesterday ({yesterday_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "day_before_yesterday": {
                "start": day_before_yesterday_midnight_utc,
                "end": yesterday_midnight_utc,
                "label": f"Day Before ({(yesterday_midnight_sg - timedelta(days=1)).strftime('%Y-%m-%d')})"
            },
            "this_week": {
                "start": week_ago_midnight_utc,
                "end": now_utc,
                "label": f"This Week (Last 7 Days)"
            },
            "all_time": {
                "start": all_time_start_utc,
                "end": now_utc,
                "label": "All Time"
            }
        }
    
    # Calculate time boundaries
    time_boundaries = get_time_boundaries()
    
    # Fetch all available pairs from DB
    @st.cache_data(ttl=600, show_spinner="Fetching pairs...")
    def fetch_all_pairs():
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
    
    all_pairs = fetch_all_pairs()
    
    # UI Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Let user select pairs to display (or select all)
        select_all = st.checkbox("Select All Pairs", value=True)
        
        if select_all:
            selected_pairs = all_pairs
        else:
            selected_pairs = st.multiselect(
                "Select Pairs", 
                all_pairs,
                default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
    
    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    if not selected_pairs:
        st.warning("Please select at least one pair")
        st.stop()
    
    # Function to fetch PNL data for a specific time period
    @st.cache_data(ttl=600)
    def fetch_pnl_data(pair_name, start_time, end_time):
        """Fetch platform PNL data for a specific time period."""
        
        query = f"""
        WITH order_pnl AS (
          -- Calculate platform order PNL
          SELECT
            COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (0, 1, 2, 3, 4)
        ),
        
        fee_data AS (
          -- Calculate user fee payments
          SELECT
            COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_fee_mode" = 1
            AND "taker_way" IN (1, 3)
        ),
        
        funding_pnl AS (
          -- Calculate platform funding fee PNL
          SELECT
            COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" = 0
        ),
        
        rebate_data AS (
          -- Calculate platform rebate payments
          SELECT
            COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
          FROM
            "public"."user_cashbooks"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "remark" = '给邀请人返佣'
        ),
        
        trade_count AS (
          -- Calculate total number of trades
          SELECT
            COUNT(*) AS "total_trades"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time}' AND '{end_time}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
        
        -- Final query: combine all data sources
        SELECT
          (SELECT "platform_order_pnl" FROM order_pnl) +
          (SELECT "user_fee_payments" FROM fee_data) +
          (SELECT "platform_funding_pnl" FROM funding_pnl) +
          (SELECT "platform_rebate_payments" FROM rebate_data) AS "platform_total_pnl",
          (SELECT "total_trades" FROM trade_count) AS "total_trades"
        """
        
        try:
            print(f"[{pair_name}] Executing PNL query for period {start_time} to {end_time}")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] PNL query executed. Result: {df.iloc[0]['platform_total_pnl']}")
            
            if df.empty:
                return {"pnl": 0, "trades": 0}
            
            return {
                "pnl": float(df.iloc[0]['platform_total_pnl']),
                "trades": int(df.iloc[0]['total_trades'])
            }
        except Exception as e:
            st.error(f"Error processing PNL for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing PNL: {e}")
            return {"pnl": 0, "trades": 0}
    
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
            period_data = fetch_pnl_data(pair_name, start_time, end_time)
            
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
        st.stop()
    
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
    
    # Function to format display DataFrame
    def format_display_df(df):
        # Add derived columns
        if 'Today PNL (USD)' in df.columns and 'Today Trades' in df.columns:
            df['Today PNL/Trade'] = (df['Today PNL (USD)'] / df['Today Trades']).round(2)
            df['Today PNL/Trade'] = df['Today PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
        
        if 'Yesterday PNL (USD)' in df.columns and 'Yesterday Trades' in df.columns:
            df['Yesterday PNL/Trade'] = (df['Yesterday PNL (USD)'] / df['Yesterday Trades']).round(2)
            df['Yesterday PNL/Trade'] = df['Yesterday PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
        
        if 'Day Before PNL (USD)' in df.columns and 'Day Before Trades' in df.columns:
            df['Day Before PNL/Trade'] = (df['Day Before PNL (USD)'] / df['Day Before Trades']).round(2)
            df['Day Before PNL/Trade'] = df['Day Before PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
        
        if 'Week PNL (USD)' in df.columns and 'Week Trades' in df.columns:
            df['Week PNL/Trade'] = (df['Week PNL (USD)'] / df['Week Trades']).round(2)
            df['Week PNL/Trade'] = df['Week PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
        
        if 'All Time PNL (USD)' in df.columns and 'All Time Trades' in df.columns:
            df['All Time PNL/Trade'] = (df['All Time PNL (USD)'] / df['All Time Trades']).round(2)
            df['All Time PNL/Trade'] = df['All Time PNL/Trade'].replace([np.inf, -np.inf, np.nan], 0)
        
        # Calculate daily average
        if 'All Time PNL (USD)' in df.columns:
            # Assuming All Time is 5 years (1825 days) - this is an approximation
            df['Avg Daily PNL'] = (df['All Time PNL (USD)'] / 1825).round(2)
        
        return df
    
    # Format the display DataFrame
    display_df = format_display_df(display_df)
    
    # Sort DataFrame by Today's PNL (descending)
    display_df = display_df.sort_values(by='Today PNL (USD)', ascending=False)
    
    # Function to color cells based on value
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
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Main Dashboard", "Detailed View", "Statistics & Insights"])
    
    with tab1:
        # Main Dashboard View
        st.subheader("PNL Overview by Trading Pair")
        
        # Create a simplified display DataFrame for the main dashboard
        main_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Week PNL (USD)', 'All Time PNL (USD)']]
        
        # Apply styling
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
        
        with col2:
            total_yesterday_pnl = display_df['Yesterday PNL (USD)'].sum()
            st.metric(
                "Total Yesterday PNL", 
                f"${total_yesterday_pnl:,.2f}"
        
        with col3:
            total_week_pnl = display_df['Week PNL (USD)'].sum()
            daily_avg = total_week_pnl / 7
            st.metric(
                "Week PNL (7 days)", 
                f"${total_week_pnl:,.2f}",
                delta=f"${daily_avg:,.2f}/day"
        
        with col4:
            total_all_time_pnl = display_df['All Time PNL (USD)'].sum()
            st.metric(
                "All Time PNL", 
                f"${total_all_time_pnl:,.2f}"
        
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
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
        
        # Show day-to-day comparison
        st.subheader("Day-to-Day PNL Comparison")
        
        day_comparison_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)']].copy()
        day_comparison_df['Day-to-Day Change'] = day_comparison_df['Today PNL (USD)'] - day_comparison_df['Yesterday PNL (USD)']
        day_comparison_df['Yesterday Change'] = day_comparison_df['Yesterday PNL (USD)'] - day_comparison_df['Day Before PNL (USD)']
        
        # Sort by day-to-day change
        day_comparison_df = day_comparison_df.sort_values(by='Day-to-Day Change', ascending=False)
        
        # Apply styling
        styled_day_comparison = day_comparison_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Day-to-Day Change', 'Yesterday Change']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Day Before PNL (USD)': '${:,.2f}',
            'Day-to-Day Change': '${:,.2f}',
            'Yesterday Change': '${:,.2f}'
        })
        
        # Display the styled DataFrame
        st.dataframe(styled_day_comparison, height=400, use_container_width=True)
        
        # Create a visualization for day-to-day comparison
        # Get top 10 pairs by absolute day-to-day change
        top_change = day_comparison_df.reindex(day_comparison_df['Day-to-Day Change'].abs().sort_values(ascending=False).index).head(10)
        
        fig = go.Figure()
        
        # Add bars for each day
        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Day Before PNL (USD)'],
            name='Day Before Yesterday',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Yesterday PNL (USD)'],
            name='Yesterday',
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            x=top_change['Trading Pair'],
            y=top_change['Today PNL (USD)'],
            name='Today',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Top 10 Pairs by Change - 3-Day PNL Comparison",
            xaxis_title="Trading Pair",
            yaxis_title="PNL (USD)",
            barmode='group',
            height=500
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
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
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize cumulative PNL distribution
        st.subheader("Cumulative PNL Distribution")
        
        # Prepare data for cumulative chart
        sorted_df = display_df.sort_values(by='All Time PNL (USD)', ascending=False).copy()
        sorted_df['Cumulative PNL'] = sorted_df['All Time PNL (USD)'].cumsum()
        sorted_df['Contribution %'] = (sorted_df['All Time PNL (USD)'] / sorted_df['All Time PNL (USD)'].sum() * 100)
        sorted_df['Cumulative Contribution %'] = sorted_df['Contribution %'].cumsum()
        
        # Create the cumulative chart
        fig = go.Figure()
        
        # Add the bar chart for individual contribution
        fig.add_trace(go.Bar(
            x=sorted_df['Trading Pair'].head(20),
            y=sorted_df['All Time PNL (USD)'].head(20),
            name='Individual PNL',
            marker_color='lightblue'
        ))
        
        # Add the line chart for cumulative contribution
        fig.add_trace(go.Scatter(
            x=sorted_df['Trading Pair'].head(20),
            y=sorted_df['Cumulative Contribution %'].head(20),
            name='Cumulative Contribution %',
            yaxis='y2',
            marker_color='darkblue',
            line=dict(width=3)
        ))
        
        # Update layout with two y-axes
        fig.update_layout(
            title="Top 20 Pairs - Individual and Cumulative PNL Contribution",
            xaxis=dict(title="Trading Pair"),
            yaxis=dict(
                title="Individual PNL (USD)",
                titlefont=dict(color="lightblue"),
                tickfont=dict(color="lightblue")
            ),
            yaxis2=dict(
                title="Cumulative Contribution %",
                titlefont=dict(color="darkblue"),
                tickfont=dict(color="darkblue"),
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show performance trends
        st.subheader("PNL Trend Analysis")
        
        # Calculate week-over-week change
        trend_df = display_df[['Trading Pair', 'Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)']].copy()
        trend_df['Daily Avg This Week'] = trend_df['Week PNL (USD)'] / 7
        trend_df['3-Day Total'] = trend_df['Today PNL (USD)'] + trend_df['Yesterday PNL (USD)'] + trend_df['Day Before PNL (USD)']
        trend_df['3-Day Daily Avg'] = trend_df['3-Day Total'] / 3
        trend_df['Performance Trend'] = (trend_df['3-Day Daily Avg'] / trend_df['Daily Avg This Week']).round(2)
        
        # Remove pairs with no activity
        trend_df = trend_df[trend_df['Week PNL (USD)'] != 0].copy()
        
        # Sort by performance trend (descending)
        trend_df = trend_df.sort_values(by='Performance Trend', ascending=False)
        
        # Apply styling
        styled_trend_df = trend_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL (USD)', 'Yesterday PNL (USD)', 'Day Before PNL (USD)', 'Week PNL (USD)', '3-Day Total', '3-Day Daily Avg', 'Daily Avg This Week']
        ).format({
            'Today PNL (USD)': '${:,.2f}',
            'Yesterday PNL (USD)': '${:,.2f}',
            'Day Before PNL (USD)': '${:,.2f}',
            'Week PNL (USD)': '${:,.2f}',
            '3-Day Total': '${:,.2f}',
            '3-Day Daily Avg': '${:,.2f}',
            'Daily Avg This Week': '${:,.2f}',
            'Performance Trend': '{:.2f}x'
        })
        
        # Add conditional formatting for Performance Trend
        def color_trend(val):
            if val > 1.5:
                return 'background-color: rgba(0, 200, 0, 0.8); color: black; font-weight: bold'
            elif val > 1.1:
                return 'background-color: rgba(150, 255, 150, 0.7); color: black'
            elif val < 0.5:
                return 'background-color: rgba(255, 0, 0, 0.9); color: white; font-weight: bold'
            elif val < 0.9:
                return 'background-color: rgba(255, 150, 150, 0.7); color: black'
            else:
                return 'background-color: rgba(255, 255, 200, 0.7); color: black'  # Neutral/stable
        
        styled_trend_df = styled_trend_df.applymap(color_trend, subset=['Performance Trend'])
        
        # Display the trend analysis
        st.markdown("### Recent Performance Trends (3-Day vs Weekly Average)")
        st.markdown("Performance Trend > 1: Recent performance better than weekly average")
        st.markdown("Performance Trend < 1: Recent performance worse than weekly average")
        
        st.dataframe(styled_trend_df, height=500, use_container_width=True)
        
        # Add explanatory text
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
            
            ### Performance Trends
            - **Performance Trend > 1**: Recent performance (3-day avg) is better than the weekly average
            - **Performance Trend < 1**: Recent performance is worse than the weekly average
            
            ### Technical Details
            - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
            - All values are shown in USD
            - The dashboard refreshes when you click the "Refresh Data" button
            - Singapore timezone (UTC+8) is used throughout
            """
    
    # Add footer with last update time
    st.markdown("---")
    st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")


def render_tab_1():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    refresh = st.button("Refresh Data", key="refresh_button_1")
    if refresh:
        st.cache_data.clear()
        st.experimental_rerun()
    with st.container():
    # Save this as pages/06_Trades_PNL_Table.py in your Streamlit app folder
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz
    
    
    # --- DB CONFIG ---
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()
    
    # --- UI Setup ---
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("User Trades & Platform PNL Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")
    
    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    expected_points = 48  # Expected data points per pair over 24 hours (24 hours * 2 intervals per hour)
    singapore_timezone = pytz.timezone('Asia/Singapore')
    
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set thresholds for highlighting
    high_trade_count_threshold = 100  # Number of trades considered "high activity"
    high_pnl_threshold = 1000  # Platform PNL amount considered "high" (in USD)
    low_pnl_threshold = -1000  # Platform PNL amount considered "low" (in USD)
    
    # Fetch all available pairs from DB
    @st.cache_data(ttl=600, show_spinner="Fetching pairs...")
    def fetch_all_pairs():
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
    
    all_pairs = fetch_all_pairs()
    
    # UI Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Let user select pairs to display (or select all)
        select_all = st.checkbox("Select All Pairs", value=True)
        
        if select_all:
            selected_pairs = all_pairs
        else:
            selected_pairs = st.multiselect(
                "Select Pairs", 
                all_pairs,
                default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
    
    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    if not selected_pairs:
        st.warning("Please select at least one pair")
        st.stop()
    
    # Function to generate aligned 30-minute time blocks for the past 24 hours
    def generate_aligned_time_blocks(current_time):
        """
        Generate fixed 30-minute time blocks for past 24 hours,
        aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
        """
        # Round down to the nearest 30-minute mark
        if current_time.minute < 30:
            # Round down to XX:00
            latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
        else:
            # Round down to XX:30
            latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
        
        # Generate block labels for display
        blocks = []
        for i in range(48):  # 24 hours of 30-minute blocks
            block_end = latest_complete_block_end - timedelta(minutes=i*30)
            block_start = block_end - timedelta(minutes=30)
            block_label = f"{block_start.strftime('%H:%M')}"
            blocks.append((block_start, block_end, block_label))
        
        return blocks
    
    # Generate aligned time blocks
    aligned_time_blocks = generate_aligned_time_blocks(now_sg)
    time_block_labels = [block[2] for block in aligned_time_blocks]
    
    # Fetch trades data for the past 24 hours in 30min intervals
    @st.cache_data(ttl=600, show_spinner="Fetching trade counts...")
    def fetch_trade_counts(pair_name):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
    
        # Updated query to use trade_fill_fresh with consistent time handling
        # Explicitly adding 8 hours to UTC timestamps to match Singapore time
        query = f"""
        SELECT
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
            INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30) 
            AS timestamp,
            COUNT(*) AS trade_count
        FROM public.trade_fill_fresh
        WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
        AND taker_way IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
        GROUP BY
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
            INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30)
        ORDER BY timestamp
        """
        
        try:
            print(f"[{pair_name}] Executing trade count query")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] Trade count query executed. DataFrame shape: {df.shape}")
            
            if df.empty:
                print(f"[{pair_name}] No trade data found.")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Format time label to match our aligned blocks (HH:MM format)
            df['time_label'] = df.index.strftime('%H:%M')
            
            return df
        except Exception as e:
            st.error(f"Error processing trade counts for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing trade counts: {e}")
            return None
        
    # Fetch platform PNL data for the past 24 hours in 30min intervals
    @st.cache_data(ttl=600, show_spinner="Calculating platform PNL...")
    def fetch_platform_pnl(pair_name):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
    
        # This query combines order PNL, fee data, funding PNL, and rebate data in 30-minute intervals
        query = f"""
        WITH time_intervals AS (
          -- Generate 30-minute intervals for the past 24 hours
          SELECT
            generate_series(
              date_trunc('hour', '{start_time_utc}'::timestamp) + 
              INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 30),
              '{end_time_utc}'::timestamp,
              INTERVAL '30 minutes'
            ) AS "UTC+8"
        ),
        
        order_pnl AS (
          -- Calculate platform order PNL
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" IN (0, 1, 2, 3, 4)
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),
        
        fee_data AS (
          -- Calculate user fee payments
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_fee_mode" = 1
            AND "taker_way" IN (1, 3)
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),
        
        funding_pnl AS (
          -- Calculate platform funding fee PNL
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
          FROM
            "public"."trade_fill_fresh"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "taker_way" = 0
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        ),
        
        rebate_data AS (
          -- Calculate platform rebate payments
          SELECT
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
            COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
          FROM
            "public"."user_cashbooks"
          WHERE
            "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
            AND "remark" = '给邀请人返佣'
          GROUP BY
            date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
            INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
        
        -- Final query: combine all data sources
        SELECT
          t."UTC+8" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp",
          COALESCE(o."platform_order_pnl", 0) +
          COALESCE(f."user_fee_payments", 0) +
          COALESCE(ff."platform_funding_pnl", 0) +
          COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
        FROM
          time_intervals t
        LEFT JOIN
          order_pnl o ON t."UTC+8" = o."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          fee_data f ON t."UTC+8" = f."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          funding_pnl ff ON t."UTC+8" = ff."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        LEFT JOIN
          rebate_data r ON t."UTC+8" = r."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
        ORDER BY
          t."UTC+8" DESC
        """
        
        try:
            print(f"[{pair_name}] Executing platform PNL query")
            df = pd.read_sql(query, engine)
            print(f"[{pair_name}] Platform PNL query executed. DataFrame shape: {df.shape}")
            
            if df.empty:
                print(f"[{pair_name}] No PNL data found.")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Format time label to match our aligned blocks (HH:MM format)
            df['time_label'] = df.index.strftime('%H:%M')
            
            return df
        except Exception as e:
            st.error(f"Error processing platform PNL for {pair_name}: {e}")
            print(f"[{pair_name}] Error processing platform PNL: {e}")
            return None
    
    # Combine Trade Count and Platform PNL data for visualization
    def combine_data(trade_data, pnl_data):
        if trade_data is None and pnl_data is None:
            return None
        
        # Create a DataFrame with time blocks as index
        time_blocks = pd.DataFrame(index=[block[2] for block in aligned_time_blocks])
        
        # Add trade count data if available
        if trade_data is not None and not trade_data.empty:
            for time_label in time_blocks.index:
                # Find matching rows in trade_data by time_label
                matching_rows = trade_data[trade_data['time_label'] == time_label]
                if not matching_rows.empty:
                    time_blocks.at[time_label, 'trade_count'] = matching_rows['trade_count'].sum()
        
        # Add PNL data if available
        if pnl_data is not None and not pnl_data.empty:
            for time_label in time_blocks.index:
                # Find matching rows in pnl_data by time_label
                matching_rows = pnl_data[pnl_data['time_label'] == time_label]
                if not matching_rows.empty:
                    time_blocks.at[time_label, 'platform_pnl'] = matching_rows['platform_total_pnl'].sum()
        
        # Fill NaN values with 0
        time_blocks.fillna(0, inplace=True)
        
        return time_blocks
    
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
            trade_data = fetch_trade_counts(pair_name)
            
            # Fetch platform PNL data
            pnl_data = fetch_platform_pnl(pair_name)
            
            # Combine data
            combined_data = combine_data(trade_data, pnl_data)
            
            if combined_data is not None:
                pair_results[pair_name] = combined_data
        except Exception as e:
            st.error(f"Error processing pair {pair_name}: {e}")
            print(f"Error processing pair {pair_name} in main loop: {e}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")
    
    # Create tables for display - Trade Count Table
    if pair_results:
        # Create trade count table data
        trade_count_data = {}
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns:
                trade_count_data[pair_name] = df['trade_count']
        
        # Create DataFrame with all pairs
        trade_count_table = pd.DataFrame(trade_count_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(trade_count_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)
        
        # Reindex with the ordered times
        trade_count_table = trade_count_table.reindex(ordered_times)
        
        # Round to integers - trade counts should be whole numbers
        trade_count_table = trade_count_table.round(0).astype('Int64')  # Using Int64 to handle NaN values properly
        
        def color_trade_cells(val):
            if pd.isna(val) or val == 0:
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
            elif val < 10:  # Low activity
                intensity = max(0, min(255, int(255 * val / 10)))
                return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
            elif val < 50:  # Medium activity
                intensity = max(0, min(255, int(255 * (val - 10) / 40)))
                return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
            elif val < high_trade_count_threshold:  # High activity
                intensity = max(0, min(255, int(255 * (val - 50) / 50)))
                return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
            else:  # Very high activity
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'
        
        styled_trade_table = trade_count_table.style.applymap(color_trade_cells)
        st.markdown("## User Trades Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:green'>Low Activity</span>, <span style='color:#aaaa00'>Medium Activity</span>, <span style='color:orange'>High Activity</span>, <span style='color:red'>Very High Activity</span>", unsafe_allow_html=True)
        st.markdown("Values shown as number of trades per 30-minute period")
        st.dataframe(styled_trade_table, height=700, use_container_width=True)
        
        # Create Platform PNL table data
        pnl_data = {}
        for pair_name, df in pair_results.items():
            if 'platform_pnl' in df.columns:
                if df['platform_pnl'].abs().sum() > 0:
                    pnl_data[pair_name] = df['platform_pnl']
        
        # Create DataFrame with all pairs
        pnl_table = pd.DataFrame(pnl_data)
        
        # Apply the time blocks in the proper order (most recent first)
        pnl_table = pnl_table.reindex(ordered_times)
        
        # Round to 2 decimal places for display
        pnl_table = pnl_table.round(0).astype(int)
        
        def color_pnl_cells(val):
            if pd.isna(val) or val == 0:
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
            elif val < low_pnl_threshold:  # Large negative PNL (loss) - red
                return f'background-color: rgba(255, 0, 0, 0.9); color: white'
            elif val < 0:  # Small negative PNL (loss) - light red
                intensity = max(0, min(255, int(255 * abs(val) / abs(low_pnl_threshold))))
                return f'background-color: rgba(255, {100-intensity}, {100-intensity}, 0.9); color: black'
            elif val < high_pnl_threshold:  # Small positive PNL (profit) - light green
                intensity = max(0, min(255, int(255 * val / high_pnl_threshold)))
                return f'background-color: rgba({100-intensity}, 180, {100-intensity}, 0.9); color: black'
            else:  # Large positive PNL (profit) - green
                return 'background-color: rgba(0, 120, 0, 0.7); color: black'
        
        styled_pnl_table = pnl_table.style.applymap(color_pnl_cells)
        st.markdown("## Platform PNL Table (USD, 30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:red'>Loss</span>, <span style='color:#ff9999'>Small Loss</span>, <span style='color:#99ff99'>Small Profit</span>, <span style='color:green'>Large Profit</span>", unsafe_allow_html=True)
        st.markdown("Values shown in USD")
        st.dataframe(styled_pnl_table, height=700, use_container_width=True)
        
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
            
            # Use a larger height and make sure the table has proper spacing
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
    
        # Add spacing between tables
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
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
            
            # Add a clear section header
            st.markdown("### 💰 Platform PNL Summary")
            
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
            
            # Use a larger height and make sure the table has proper spacing
            st.dataframe(
                styled_pnl_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=350,
                use_container_width=True
    
        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
    
        # Combined Analysis with improved formatting
        st.subheader("Combined Analysis: Trading Activity vs. Platform PNL")
        # Extract periods with both high activity and significant PNL
        high_activity_periods = []
        for pair_name, df in pair_results.items():
            if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                # Find time periods with both significant activity and significant PNL
                for time_label, row in df.iterrows():
                    trade_count = row['trade_count']
                    pnl = row['platform_pnl']
                    
                    # Check if this is a noteworthy period
                    if (trade_count >= high_trade_count_threshold or 
                        pnl >= high_pnl_threshold or pnl <= low_pnl_threshold):
                        high_activity_periods.append({
                            'Pair': pair_name,
                            'Time': time_label,
                            'Trade Count': int(trade_count),
                            'Platform PNL (USD)': round(pnl, 2),
                            'Revenue per Trade (USD)': round(pnl / trade_count, 2) if trade_count > 0 else 0
                        })
        
        # Extract periods with both high activity and significant PNL with better formatting
        if high_activity_periods:
            # Convert to DataFrame
            high_activity_df = pd.DataFrame(high_activity_periods)
            
            # Rename columns for clarity
            high_activity_df = high_activity_df.rename(columns={
                'Pair': '🔄 Trading Pair',
                'Time': '⏰ Time Period',
                'Trade Count': '📊 Number of Trades',
                'Platform PNL (USD)': '💰 PNL (USD)',
                'Revenue per Trade (USD)': '💸 PNL/Trade (USD)'
            })
            
            # Sort by Trade Count (highest first)
            high_activity_df = high_activity_df.sort_values(by='📊 Number of Trades', ascending=False)
            
            # Style the dataframe
            styled_activity_df = high_activity_df.style.format({
                '💰 PNL (USD)': '${:,.2f}',
                '💸 PNL/Trade (USD)': '${:,.2f}'
            })
            
            # Apply conditional formatting
            def highlight_pnl(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                return ''
            
            styled_activity_df = styled_activity_df.applymap(highlight_pnl, subset=['💰 PNL (USD)', '💸 PNL/Trade (USD)'])
            
            # Add a clear section header
            st.markdown("### 🔍 High Activity Periods")
            
            # Display the dataframe with improved styling
            st.dataframe(
                styled_activity_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=350,
                use_container_width=True
            
            # Create visual representation
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 trading periods by volume
                top_trading_periods = high_activity_df.head(10)
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_trading_periods.iterrows()],
                        y=top_trading_periods['📊 Number of Trades'],
                        marker_color='blue'
    }
])
                fig.update_layout(
                    title="Top 10 Trading Periods by Volume",
                    xaxis_title="Pair and Time",
                    yaxis_title="Number of Trades",
                    height=400
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top 10 PNL periods
                top_pnl_periods = high_activity_df.sort_values(by='💰 PNL (USD)', ascending=False).head(10)
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{row['🔄 Trading Pair']} ({row['⏰ Time Period']})" for _, row in top_pnl_periods.iterrows()],
                        y=top_pnl_periods['💰 PNL (USD)'],
                        marker_color='green'
    )
])
                fig.update_layout(
                    title="Top 10 Trading Periods by Platform PNL",
                    xaxis_title="Pair and Time",
                    yaxis_title="Platform PNL (USD)",
                    height=400
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall Trade Count vs. PNL correlation analysis
            st.subheader("Correlation Analysis: Trade Count vs. Platform PNL")
            
            correlation_data = []
            for pair_name, df in pair_results.items():
                if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                    # Calculate correlation between trade count and PNL
                    correlation = df['trade_count'].corr(df['platform_pnl'])
                    # Filter out rows with zero trades
                    non_zero_trades = df[df['trade_count'] > 0]
                    # Calculate average PNL per trade
                    avg_pnl_per_trade = non_zero_trades['platform_pnl'].sum() / non_zero_trades['trade_count'].sum() if non_zero_trades['trade_count'].sum() > 0 else 0
                    
                    correlation_data.append({
                        'Pair': pair_name,
                        'Correlation': round(correlation, 3) if not pd.isna(correlation) else 0,
                        'Total Trades': int(df['trade_count'].sum()),
                        'Total PNL (USD)': round(df['platform_pnl'].sum(), 2),
                        'Avg PNL per Trade (USD)': round(avg_pnl_per_trade, 3)
                    })
            
            if correlation_data:
                # Convert to DataFrame
                correlation_df = pd.DataFrame(correlation_data)
                
                # Sort by correlation (highest first)
                correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
                
                # Format the dataframe for better legibility
                correlation_df = correlation_df.rename(columns={
                    'Pair': '🔄 Trading Pair',
                    'Correlation': '📊 Correlation',
                    'Total Trades': '📈 Total Trades',
                    'Total PNL (USD)': '💰 Total PNL (USD)',
                    'Avg PNL per Trade (USD)': '💸 Avg PNL/Trade (USD)'
                })
                
                # Style the dataframe
                styled_correlation_df = correlation_df.style.format({
                    '📊 Correlation': '{:.3f}',
                    '💰 Total PNL (USD)': '${:,.2f}',
                    '💸 Avg PNL/Trade (USD)': '${:,.3f}'
                })
                
                # Apply conditional formatting
                def highlight_correlation(val):
                    if isinstance(val, (int, float)):
                        if val > 0.5:
                            return 'color: green; font-weight: bold'
                        elif val < -0.5:
                            return 'color: red; font-weight: bold'
                    return ''
                
                styled_correlation_df = styled_correlation_df.applymap(highlight_correlation, subset=['📊 Correlation'])
                styled_correlation_df = styled_correlation_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 Avg PNL/Trade (USD)'])
                
                # Create columns for display
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    # Display the correlation table with improved styling
                    st.markdown("### 📊 Trade Count vs. PNL Correlation by Pair")
                    st.dataframe(
                        styled_correlation_df.set_properties(**{
                            'font-size': '16px',
                            'text-align': 'center',
                            'background-color': '#f0f2f6'
                        }),
                        height=400,
                        use_container_width=True
                
                with col2:
                    # Create a scatter plot to visualize the correlation
                    # Gather all data points for the scatter plot
                    scatter_data = []
                    for pair_name, df in pair_results.items():
                        if 'trade_count' in df.columns and 'platform_pnl' in df.columns:
                            for time_label, row in df.iterrows():
                                if row['trade_count'] > 0:  # Only include periods with trades
                                    scatter_data.append({
                                        'Pair': pair_name,
                                        'Trade Count': int(row['trade_count']),
                                        'Platform PNL (USD)': round(row['platform_pnl'], 2)
                                    })
                    
                    if scatter_data:
                        scatter_df = pd.DataFrame(scatter_data)
                        
                        # Create a scatter plot using Plotly
                        fig = px.scatter(
                            scatter_df, 
                            x='Trade Count', 
                            y='Platform PNL (USD)',
                            color='Pair',
                            title='Trade Count vs. Platform PNL Correlation',
                            hover_data=['Pair', 'Trade Count', 'Platform PNL (USD)'],
                            
                        
                        fig.update_layout(
                            height=500,
                            xaxis_title="Number of Trades",
                            yaxis_title="Platform PNL (USD)",
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Time-based Analysis
        st.subheader("Time-based Analysis")
        
        # Analyze trading patterns over time
        hourly_patterns = {}
        for time_block in ordered_times:
            # Extract hour from time block
            hour = int(time_block.split(':')[0])
            
            # Initialize if this hour is not yet in the dictionary
            if hour not in hourly_patterns:
                hourly_patterns[hour] = {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'count': 0
                }
            
            # Sum up trades and PNL for this hour across all pairs
            for pair_name, df in pair_results.items():
                if time_block in df.index:
                    if 'trade_count' in df.columns:
                        hourly_patterns[hour]['total_trades'] += df.at[time_block, 'trade_count']
                    if 'platform_pnl' in df.columns:
                        hourly_patterns[hour]['total_pnl'] += df.at[time_block, 'platform_pnl']
                    hourly_patterns[hour]['count'] += 1
        
        # Convert to DataFrame for display
        hourly_patterns_df = pd.DataFrame([
            {
                'Hour (SG Time)': f"{hour:02d}:00-{hour:02d}:59",
                'Avg Trades': round(data['total_trades'] / data['count'] if data['count'] > 0 else 0, 1),
                'Avg PNL (USD)': round(data['total_pnl'] / data['count'] if data['count'] > 0 else 0, 2),
                'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'] if data['total_trades'] > 0 else 0, 3)
            }
            for hour, data in hourly_patterns.items()
    }
])
        
        # Sort by hour for display
        hourly_patterns_df = hourly_patterns_df.sort_values(by='Hour (SG Time)')
        
        # Format the dataframe for better legibility
        hourly_patterns_df = hourly_patterns_df.rename(columns={
            'Hour (SG Time)': '🕒 Hour (SG Time)',
            'Avg Trades': '📊 Avg Trades',
            'Avg PNL (USD)': '💰 Avg PNL (USD)',
            'PNL per Trade (USD)': '💸 PNL/Trade (USD)'
        })
        
        # Style the dataframe
        styled_hourly_df = hourly_patterns_df.style.format({
            '📊 Avg Trades': '{:.1f}',
            '💰 Avg PNL (USD)': '${:,.2f}',
            '💸 PNL/Trade (USD)': '${:,.3f}'
        })
        
        # Apply conditional formatting
        styled_hourly_df = styled_hourly_df.applymap(highlight_pnl, subset=['💰 Avg PNL (USD)', '💸 PNL/Trade (USD)'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🕒 Hourly Trading Patterns")
            st.dataframe(
                styled_hourly_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=500,
                use_container_width=True
        
        with col2:
            # Create charts for hourly patterns
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=hourly_patterns_df['🕒 Hour (SG Time)'],
                y=hourly_patterns_df['📊 Avg Trades'],
                name='Avg Trades',
                marker_color='blue'
            ))
            
            fig.add_trace(go.Scatter(
                x=hourly_patterns_df['🕒 Hour (SG Time)'],
                y=hourly_patterns_df['💰 Avg PNL (USD)'],
                name='Avg PNL (USD)',
                yaxis='y2',
                mode='lines+markers',
                marker_color='green',
                line=dict(width=3)
            ))
            
            # Update layout with two y-axes
            fig.update_layout(
                title="Hourly Trading Activity and PNL (Singapore Time)",
                xaxis=dict(title="Hour"),
                yaxis=dict(
                    title="Avg Number of Trades",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue")
                ),
                yaxis2=dict(
                    title="Avg PNL (USD)",
                    titlefont=dict(color="green"),
                    tickfont=dict(color="green"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # PNL Breakdown Analysis
        st.subheader("Platform Profit Distribution")
        
        # Calculate platform total profit
        total_platform_profit = sum(df['platform_pnl'].sum() for pair, df in pair_results.items() if 'platform_pnl' in df.columns)
        
        # Calculate per-pair contribution to total profit
        profit_distribution = []
        for pair_name, df in pair_results.items():
            if 'platform_pnl' in df.columns:
                pair_pnl = df['platform_pnl'].sum()
                contribution_pct = 100 * pair_pnl / total_platform_profit if total_platform_profit != 0 else 0
                
                profit_distribution.append({
                    'Pair': pair_name,
                    'Total PNL (USD)': round(pair_pnl, 0),
                    'Contribution (%)': round(contribution_pct, 2)
                })
        
        if profit_distribution:
            # Sort by total PNL (highest first)
            profit_distribution_df = pd.DataFrame(profit_distribution)
            profit_distribution_df = profit_distribution_df.sort_values(by='Total PNL (USD)', ascending=False)
            
            # Format the dataframe for better legibility
            profit_distribution_df = profit_distribution_df.rename(columns={
                'Pair': '🔄 Trading Pair',
                'Total PNL (USD)': '💰 Total PNL (USD)',
                'Contribution (%)': '📊 Contribution (%)'
            })
            
            # Apply styling to make numbers clearer
            styled_profit_df = profit_distribution_df.style.format({
                '💰 Total PNL (USD)': '${:,.0f}',  # Format as integers with comma for thousands
                '📊 Contribution (%)': '{:+.2f}%'  # Show with + or - sign and 2 decimal places
            })
        
            # Conditionally color the cells based on values
            def color_pnl_and_contribution(val, column):
                if column == '💰 Total PNL (USD)':
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                    return ''
                elif column == '📊 Contribution (%)':
                    if val > 0:
                        return 'color: green; font-weight: bold'
                    elif val < 0:
                        return 'color: red; font-weight: bold'
                    return ''
                return ''
        
            # Apply the styling function
            styled_profit_df = styled_profit_df.applymap(
                lambda x: color_pnl_and_contribution(x, '💰 Total PNL (USD)'), 
                subset=['💰 Total PNL (USD)']
             ).applymap(
                lambda x: color_pnl_and_contribution(x, '📊 Contribution (%)'), 
                subset=['📊 Contribution (%)']
            
            # Display the styled dataframe
            st.markdown("### 💰 Profit Contribution by Pair")
            st.dataframe(
                styled_profit_df.set_properties(**{
                    'font-size': '16px',
                    'text-align': 'center',
                    'background-color': '#f0f2f6'
                }),
                height=500,
                use_container_width=True
            
            # Create a pie chart visualizing contribution
            top_pairs = profit_distribution_df.head(10)
            
            # Filter to only positive contributions for the pie chart
            positive_pairs = top_pairs[top_pairs['💰 Total PNL (USD)'] > 0]
            if not positive_pairs.empty:
                fig = px.pie(
                    positive_pairs, 
                    values='💰 Total PNL (USD)', 
                    names='🔄 Trading Pair',
                    title='Top 10 Pairs by Positive PNL Contribution',
                    hole=0.4
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Add spacing for the next section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Identify Most Profitable Time Periods
        st.subheader("Most Profitable Time Periods")
        
        # Calculate profitability for each time period across all pairs
        time_period_profit = {}
        for time_block in ordered_times:
            time_period_profit[time_block] = {
                'total_pnl': 0,
                'total_trades': 0,
                'pair_breakdown': {}
            }
            
            for pair_name, df in pair_results.items():
                if time_block in df.index:
                    if 'platform_pnl' in df.columns:
                        pair_pnl = df.at[time_block, 'platform_pnl']
                        time_period_profit[time_block]['total_pnl'] += pair_pnl
                        time_period_profit[time_block]['pair_breakdown'][pair_name] = pair_pnl
                    
                    if 'trade_count' in df.columns:
                        time_period_profit[time_block]['total_trades'] += df.at[time_block, 'trade_count']
        
        # Convert to DataFrame for display
        time_profit_df = pd.DataFrame([
            {
                'Time Period': time_block,
                'Total PNL (USD)': round(data['total_pnl'], 2),
                'Total Trades': int(data['total_trades']),
                'PNL per Trade (USD)': round(data['total_pnl'] / data['total_trades'], 3) if data['total_trades'] > 0 else 0,
                'Top Contributing Pair': max(data['pair_breakdown'].items(), key=lambda x: x[1])[0] if data['pair_breakdown'] else "None"}
            }
            for time_block, data in time_period_profit.items()
    }
])
        
        # Format the dataframe for better legibility
        time_profit_df = time_profit_df.rename(columns={
            'Time Period': '⏰ Time Period',
            'Total PNL (USD)': '💰 Total PNL (USD)',
            'Total Trades': '📊 Total Trades',
            'PNL per Trade (USD)': '💸 PNL/Trade (USD)',
            'Top Contributing Pair': '🔄 Top Pair'
        })
        
        # Sort by total PNL (highest first)
        time_profit_df = time_profit_df.sort_values(by='💰 Total PNL (USD)', ascending=False)
        
        # Style the dataframe
        styled_time_profit_df = time_profit_df.style.format({
            '💰 Total PNL (USD)': '${:,.2f}',
            '💸 PNL/Trade (USD)': '${:,.3f}'
        })
        
        # Apply conditional formatting
        styled_time_profit_df = styled_time_profit_df.applymap(highlight_pnl, subset=['💰 Total PNL (USD)', '💸 PNL/Trade (USD)'])
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            # Show top profitable time periods
            st.markdown("### 📈 Top 10 Most Profitable Time Periods")
    
            # Style the entire DataFrame
            full_styled_df = styled_time_profit_df.set_properties(**{
             'font-size': '16px',
             'text-align': 'center',
             'background-color': '#f0f2f6'
        }   )
    
            # Display only the top 10 rows
            st.dataframe(
               full_styled_df.data.head(10),
               height=300,
               use_container_width=True
        
        with col2:
            # Show bottom profitable (loss-making) time periods
            st.markdown("### 📉 Top 10 Least Profitable Time Periods")
        
            # Get the 10 least profitable periods and sort them
            bottom_10 = time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')
        
            # Style the bottom 10 with the same approach as the top 10
            bottom_styled_df = styled_time_profit_df.set_properties(**{
            'font-size': '16px',
            'text-align': 'center',
            'background-color': '#f0f2f6'
        })
        
            # Display the filtered data
            st.dataframe(
                bottom_styled_df.data.tail(10).sort_values(by='💰 Total PNL (USD)'),
                height=300,
                use_container_width=True
        
        # Create visualization of top profitable and loss-making periods
        fig = go.Figure()
        
        # Top 10 profitable periods
        fig.add_trace(go.Bar(
            x=time_profit_df.head(10)['⏰ Time Period'],
            y=time_profit_df.head(10)['💰 Total PNL (USD)'],
            name='Top Profitable Periods',
            marker_color='green'
        ))
        
        # Bottom 10 profitable periods
        bottom_10 = time_profit_df.tail(10).sort_values(by='💰 Total PNL (USD)')
        fig.add_trace(go.Bar(
            x=bottom_10['⏰ Time Period'],
            y=bottom_10['💰 Total PNL (USD)'],
            name='Least Profitable Periods',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Most and Least Profitable Time Periods",
            xaxis_title="Time Period (SG Time)",
            yaxis_title="Total PNL (USD)",
            height=500,
            barmode='group'
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation for dashboard
        with st.expander("Understanding the Trading & PNL Dashboard"):
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
            - **High Activity Periods**: Spot times of significant trading and PNL outcomes
            - **Correlation Analysis**: Understand the relationship between trade volume and profitability
            - **Time-based Analysis**: Discover trading patterns throughout the day
            - **Profit Distribution**: See which pairs contribute most to overall profit
            - **Time Period Analysis**: Identify the most and least profitable time periods
            
            ### Technical Details
            - PNL calculation includes order PNL, fee revenue, funding fees, and rebate payments
            - All values are shown in USD
            - The dashboard refreshes when you click the "Refresh Data" button
            - Singapore timezone (UTC+8) is used throughout
            """
    
    else:
        st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")


def render_tab_2():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    refresh = st.button("Refresh Data", key="refresh_button_2")
    if refresh:
        st.cache_data.clear()
        st.experimental_rerun()
    with st.container():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz
    
    # Page configuration
    
    # Apply custom CSS styling - more minimal design with centered numeric columns and color coding
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
        /* Simplified tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 15px;
            background-color: #f5f5f5;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4682B4;
            color: white;
        }
        
        /* Center numeric columns in dataframes */
        .dataframe th, .dataframe td {
            text-align: center !important;
        }
        /* First column (Token) remains left-aligned */
        .dataframe th:first-child, .dataframe td:first-child {
            text-align: left !important;
        }
        
        /* Color coding for spread values */
        .very-low-spread {
            background-color: rgba(0, 128, 0, 0.1) !important;
            color: #006600 !important;
            font-weight: bold;
        }
        .low-spread {
            background-color: rgba(0, 102, 204, 0.1) !important;
            color: #0066cc !important;
        }
        .medium-spread {
            background-color: rgba(255, 153, 0, 0.1) !important;
            color: #ff9900 !important;
        }
        .high-spread {
            background-color: rgba(204, 0, 0, 0.1) !important;
            color: #cc0000 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Database Configuration ---
    try:
        # Try to get database config from secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_uri)
        st.sidebar.success("Connected to database successfully")
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
                engine = create_engine(db_uri)
                st.sidebar.success("Connected to database successfully")
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {e}")
                st.stop()
        else:
            st.error("Please connect to the database to continue")
            st.stop()
    
    # --- Constants and Configuration ---
    # Define exchanges
    exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "surfFuture"]
    exchanges_display = {
        "binanceFuture": "Binance",
        "gateFuture": "Gate",
        "hyperliquidFuture": "Hyperliquid",
        "surfFuture": "SurfFuture"
    }
    
    # Define time parameters
    interval_minutes = 10  # 10-minute intervals
    singapore_timezone = pytz.timezone('Asia/Singapore')
    lookback_days = 1  # Fixed to 1 day
    
    # --- Utility Functions ---
    # Function to convert time string to sortable minutes value
    def time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    # Function to determine if a token is a major
    def is_major(token):
        majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
        for major in majors:
            if major in token:
                return True
        return False
    
    # Function to get appropriate depth tiers based on token type
    def get_depth_tiers(token):
        if is_major(token):
            return [50000, 100000, 200000, 500000]  # Majors: 50k, 100k, 200k, 500k
        else:
            return [20000, 50000, 100000, 200000]   # Altcoins: 20k, 50k, 100k, 200k
    
    # Function to get depth label based on token type
    def get_depth_label(fee_column, token):
        if is_major(token):
            depth_map = {
                'fee1': '50K', 'fee2': '100K', 'fee3': '200K', 'fee4': '500K',
                'avg_fee1': '50K', 'avg_fee2': '100K', 'avg_fee3': '200K', 'avg_fee4': '500K'
            }
        else:
            depth_map = {
                'fee1': '20K', 'fee2': '50K', 'fee3': '100K', 'fee4': '200K',
                'avg_fee1': '20K', 'avg_fee2': '50K', 'avg_fee3': '100K', 'avg_fee4': '200K'
            }
        return depth_map.get(fee_column, fee_column)
    
    # Function to apply color coding to spread values
    def color_code_value(value, thresholds=None):
        """Apply color coding to spread values"""
        if pd.isna(value):
            return value
        
        if thresholds is None:
            # Default thresholds - adjust based on your data
            thresholds = [0.0005, 0.001, 0.005]
            
        if value < thresholds[0]:
            return f'<span class="very-low-spread">{value:.6f}</span>'
        elif value < thresholds[1]:
            return f'<span class="low-spread">{value:.6f}</span>'
        elif value < thresholds[2]:
            return f'<span class="medium-spread">{value:.6f}</span>'
        else:
            return f'<span class="high-spread">{value:.6f}</span>'
    
    # --- Data Fetching Functions ---
    @st.cache_data(ttl=600, show_spinner="Fetching tokens...")
    def fetch_all_tokens():
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
    
    @st.cache_data(ttl=600, show_spinner="Fetching spread data...")
    def fetch_10min_spread_data(token):
        """Fetch 10-minute spread data for a specific token (last 24 hours)"""
        try:
            # Get current time in Singapore timezone
            now_utc = datetime.now(pytz.utc)
            now_sg = now_utc.astimezone(singapore_timezone)
            start_time_sg = now_sg - timedelta(days=lookback_days)
            
            # Convert back to UTC for database query
            start_time_utc = start_time_sg.astimezone(pytz.utc)
            end_time_utc = now_sg.astimezone(pytz.utc)
    
            # Query to get the fee data for the specified token
            # Special handling for SurfFuture: Use total_fee for all fee levels
            query = f"""
            SELECT 
                time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
                pair_name,
                source,
                CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee1 END as fee1,
                CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee2 END as fee2,
                CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee3 END as fee3,
                CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee4 END as fee4,
                total_fee
            FROM 
                oracle_exchange_fee
            WHERE 
                pair_name = '{token}'
                AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
            ORDER BY 
                timestamp ASC,
                source
            """
            
            df = pd.read_sql(query, engine)
            
            if df.empty:
                return None
    
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
                
        except Exception as e:
            print(f"[{token}] Error fetching spread data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @st.cache_data(ttl=600, show_spinner="Fetching daily spread averages...")
    def fetch_daily_spread_averages(tokens):
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
            # Special handling for SurfFuture: Use total_fee for all fee levels
            query = f"""
            SELECT 
                pair_name,
                source,
                CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee1) END as avg_fee1,
                CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee2) END as avg_fee2,
                CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee3) END as avg_fee3,
                CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee4) END as avg_fee4,
                AVG(total_fee) as avg_total_fee
            FROM 
                oracle_exchange_fee
            WHERE 
                pair_name IN ('{tokens_str}')
                AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
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
            print(f"Error fetching daily spread averages: {e}")
            return None
    
    def calculate_matrix_data(avg_data):
        """Transform the average data into matrix format for display"""
        if avg_data is None or avg_data.empty:
            return None
        
        # Create pivot tables for each fee level
        matrix_data = {}
        fee_columns = [col for col in ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4'] 
                      if col in avg_data.columns]
        
        if not fee_columns:
            st.error("Required fee columns not found in the data. Check database schema.")
            return None
        
        for fee_col in fee_columns:
            try:
                # Convert the long format to wide format (pivot)
                pivot_df = avg_data.pivot(index='pair_name', columns='source', values=fee_col).reset_index()
                
                # Rename columns to display names
                for source in exchanges:
                    if source in pivot_df.columns:
                        pivot_df = pivot_df.rename(columns={source: exchanges_display[source]})
                
                # Calculate non-surf average
                non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
                available_non_surf = [col for col in non_surf_columns if col in pivot_df.columns]
                
                if available_non_surf:
                    pivot_df['Avg (Non-Surf)'] = pivot_df[available_non_surf].mean(axis=1)
                
                # Add a column indicating if SurfFuture is better than non-surf avg
                if 'SurfFuture' in pivot_df.columns and 'Avg (Non-Surf)' in pivot_df.columns:
                    pivot_df['Surf Better'] = pivot_df['SurfFuture'] < pivot_df['Avg (Non-Surf)']
                    
                    # Calculate percentage improvement
                    pivot_df['Improvement %'] = ((pivot_df['Avg (Non-Surf)'] - pivot_df['SurfFuture']) / 
                                                pivot_df['Avg (Non-Surf)'] * 100).round(2)
                
                # Store the pivot table for this fee level
                matrix_data[fee_col.replace('avg_', '')] = pivot_df
            except Exception as e:
                st.error(f"Error processing {fee_col}: {e}")
        
        return matrix_data
    
    # --- Main Application ---
    st.markdown('<div class="header-style">Exchange Spread Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Fetch all available tokens
    all_tokens = fetch_all_tokens()
    
    # Sidebar - simplified, just refresh button
    st.sidebar.header("Controls")
    
    # Always select all tokens
    selected_tokens = all_tokens
    
    # Add a refresh button
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["50K/20K Analysis", "100K/50K Analysis", "Spread By Size"])
    
    # Get daily spread data (used in all tabs)
    daily_avg_data = fetch_daily_spread_averages(selected_tokens)
    if daily_avg_data is None or daily_avg_data.empty:
        st.warning("No spread data available for the selected time period.")
        st.stop()
    
    # Calculate matrix data for all fee levels
    matrix_data = calculate_matrix_data(daily_avg_data)
    if matrix_data is None or not matrix_data:
        st.warning("Unable to process spread data. Check log for details.")
        st.stop()
    
    # === TAB 1: 50K/20K ANALYSIS (fee1) ===
    with tab1:
        st.markdown('<div class="header-style">50K/20K Spread Analysis</div>', unsafe_allow_html=True)
        
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
            df = matrix_data['fee1']
            
            # Determine scale factor for better readability
            scale_factor = 1
            scale_label = ""
            
            # Calculate mean for scaling
            numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
            if numeric_cols:
                values = []
                for col in numeric_cols:
                    values.extend(df[col].dropna().tolist())
                
                if values:
                    mean_fee = sum(values) / len(values)
                    
                    # Determine scale factor based on mean fee value
                    if mean_fee < 0.001:
                        scale_factor = 1000
                        scale_label = "× 1,000"
                    elif mean_fee < 0.0001:
                        scale_factor = 10000
                        scale_label = "× 10,000"
                    elif mean_fee < 0.00001:
                        scale_factor = 100000
                        scale_label = "× 100,000"
            
            # Apply scaling if needed
            if scale_factor > 1:
                for col in numeric_cols:
                    df[col] = df[col] * scale_factor
                st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
            
            # Format and display the dataframe
            display_df = df.copy()
            
            # Round values for display
            for col in numeric_cols:
                display_df[col] = display_df[col].round(6)
            
            # Add token type column for clarity
            display_df['Token Type'] = display_df['pair_name'].apply(
                lambda x: 'Major' if is_major(x) else 'Altcoin'
            
            # Sort by token type and then by name
            display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
            
            # Define column order with SurfFuture at the end
            desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
            if 'Improvement %' in display_df.columns:
                desired_order.append('Improvement %')
            ordered_columns = [col for col in desired_order if col in display_df.columns]
            
            # Add Surf Better column if it exists (hidden, used for filtering)
            if 'Surf Better' in display_df.columns:
                ordered_columns.append('Surf Better')
            
            display_df = display_df[ordered_columns]
            
            # Rename columns for display
            display_df = display_df.rename(columns={'pair_name': 'Token'})
            
            # Apply color coding to numeric columns
            color_df = display_df.copy()
            for col in numeric_cols:
                if col in color_df.columns and col != 'Token Type':
                    # Determine thresholds based on column values
                    values = color_df[col].dropna().tolist()
                    if values:
                        # Dynamic thresholds based on percentiles
                        q1 = np.percentile(values, 25)
                        median = np.percentile(values, 50)
                        q3 = np.percentile(values, 75)
                        thresholds = [q1, median, q3]
                        
                        color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
            
            # Special formatting for improvement percentage
            if 'Improvement %' in color_df.columns:
                color_df['Improvement %'] = color_df['Improvement %'].apply(
                    lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                    (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            
            # Display the table with HTML formatting
            token_count = len(color_df)
            table_height = max(100 + 35 * token_count, 300)  # Minimum height of 300px
            
            # Convert to HTML for better formatting
            html_table = color_df.to_html(escape=False, index=False)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
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
                
                # Update layout
                fig.update_layout(
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                    margin=dict(t=60, b=60, l=20, r=20),
                    height=400
                
                # Display percentage text in middle
                better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
                fig.add_annotation(
                    text=f"{better_percentage:.1f}%<br>Better",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show summary of SurfFuture performance
            if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
                surf_values = df['SurfFuture'].dropna()
                nonsurf_values = df['Avg (Non-Surf)'].dropna()
                
                if not surf_values.empty and not nonsurf_values.empty:
                    # Match indices to compare only pairs with both values
                    common_indices = surf_values.index.intersection(nonsurf_values.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                        total_count = len(common_indices)
                        
                        # Calculate percentages
                        surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                        
                        # Display summary box
                        st.markdown(f"""
                        <div class="info-box">
                        <b>SurfFuture Performance Summary:</b><br>
                        • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Calculate averages
                        surf_avg = surf_values.mean()
                        nonsurf_avg = nonsurf_values.mean()
                        
                        if surf_avg < nonsurf_avg:
                            improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                            st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is {improvement:.2f}% lower than other exchanges ({nonsurf_avg:.6f})**")
                        
                # Calculate separate stats for majors and altcoins
                major_tokens_df = df[df['pair_name'].apply(is_major)]
                altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
                
                # For Major tokens
                if not major_tokens_df.empty:
                    surf_major = major_tokens_df['SurfFuture'].dropna()
                    nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                    
                    if not surf_major.empty and not nonsurf_major.empty:
                        common_indices = surf_major.index.intersection(nonsurf_major.index)
                        if len(common_indices) > 0:
                            surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                            total_count = len(common_indices)
                            
                            if total_count > 0:
                                surf_major_avg = surf_major.mean()
                                nonsurf_major_avg = nonsurf_major.mean()
                                
                                st.markdown(f"""
                                <div class="info-box">
                                <b>Major Tokens (50K):</b><br>
                                • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                                • SurfFuture average: {surf_major_avg:.6f} vs Non-Surf average: {nonsurf_major_avg:.6f}
                                </div>
                                """, unsafe_allow_html=True)
                
                # For Altcoin tokens
                if not altcoin_tokens_df.empty:
                    surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                    nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                    
                    if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                        common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                        if len(common_indices) > 0:
                            surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                            total_count = len(common_indices)
                            
                            if total_count > 0:
                                surf_altcoin_avg = surf_altcoin.mean()
                                nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                                
                                st.markdown(f"""
                                <div class="info-box">
                                <b>Altcoin Tokens (20K):</b><br>
                                • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                                • SurfFuture average: {surf_altcoin_avg:.6f} vs Non-Surf average: {nonsurf_altcoin_avg:.6f}
                                </div>
                                """, unsafe_allow_html=True)
                                
                # Bar chart comparing exchanges
                st.markdown("### Average Spread by Exchange")
                
                # Calculate average for each exchange
                exchange_avgs = {}
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                    if exchange in df.columns:
                        values = df[exchange].dropna()
                        if not values.empty:
                            exchange_avgs[exchange] = values.mean()
                
                if exchange_avgs:
                    # Create data frame for plotting
                    avg_df = pd.DataFrame({
                        'Exchange': list(exchange_avgs.keys()),
                        'Average Spread': list(exchange_avgs.values())
                    })
                    
                    # Sort by average spread (ascending)
                    avg_df = avg_df.sort_values('Average Spread')
                    
                    # Create bar chart
                    fig = px.bar(
                        avg_df,
                        x='Exchange',
                        y='Average Spread',
                        title=f"Average Spread by Exchange ({scale_label})",
                        color='Exchange',
                        text='Average Spread',
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    
                    # Format the bars
                    fig.update_traces(
                        texttemplate='%{y:.6f}',
                        textposition='outside'
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Exchange",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=400
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No fee1 (50K/20K) data available for analysis")
    
    # === TAB 2: 100K/50K ANALYSIS (fee2) ===
    with tab2:
        st.markdown('<div class="header-style">100K/50K Spread Analysis</div>', unsafe_allow_html=True)
        
        # Display explanation of depth tiers
        st.markdown("""
        <div class="info-box">
        <b>Trading Size Definition:</b><br>
        • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 100K<br>
        • <b>Altcoin tokens</b>: 50K<br>
        <br>
        This tab shows daily averages of 10-minute spread data points at 100K/50K size.""
        "</div>
        """, unsafe_allow_html=True)
        
        if 'fee2' in matrix_data:
            df = matrix_data['fee2']
            
            # Determine scale factor for better readability
            scale_factor = 1
            scale_label = ""
            
            # Calculate mean for scaling
            numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
            if numeric_cols:
                values = []
                for col in numeric_cols:
                    values.extend(df[col].dropna().tolist())
                
                if values:
                    mean_fee = sum(values) / len(values)
                    
                    # Determine scale factor based on mean fee value
                    if mean_fee < 0.001:
                        scale_factor = 1000
                        scale_label = "× 1,000"
                    elif mean_fee < 0.0001:
                        scale_factor = 10000
                        scale_label = "× 10,000"
                    elif mean_fee < 0.00001:
                        scale_factor = 100000
                        scale_label = "× 100,000"
            
            # Apply scaling if needed
            if scale_factor > 1:
                for col in numeric_cols:
                    df[col] = df[col] * scale_factor
                st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
            
            # Format and display the dataframe
            display_df = df.copy()
            
            # Round values for display
            for col in numeric_cols:
                display_df[col] = display_df[col].round(6)
            
            # Add token type column for clarity
            display_df['Token Type'] = display_df['pair_name'].apply(
                lambda x: 'Major' if is_major(x) else 'Altcoin'
            
            # Sort by token type and then by name
            display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
            
            # Define column order with SurfFuture at the end
            desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
            if 'Improvement %' in display_df.columns:
                desired_order.append('Improvement %')
            ordered_columns = [col for col in desired_order if col in display_df.columns]
            
            # Add Surf Better column if it exists (hidden, used for filtering)
            if 'Surf Better' in display_df.columns:
                ordered_columns.append('Surf Better')
            
            display_df = display_df[ordered_columns]
            
            # Rename columns for display
            display_df = display_df.rename(columns={'pair_name': 'Token'})
            
            # Apply color coding to numeric columns
            color_df = display_df.copy()
            for col in numeric_cols:
                if col in color_df.columns and col != 'Token Type':
                    # Determine thresholds based on column values
                    values = color_df[col].dropna().tolist()
                    if values:
                        # Dynamic thresholds based on percentiles
                        q1 = np.percentile(values, 25)
                        median = np.percentile(values, 50)
                        q3 = np.percentile(values, 75)
                        thresholds = [q1, median, q3]
                        
                        color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
            
            # Special formatting for improvement percentage
            if 'Improvement %' in color_df.columns:
                color_df['Improvement %'] = color_df['Improvement %'].apply(
                    lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                    (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            
            # Display the table with HTML formatting
            token_count = len(color_df)
            table_height = max(100 + 35 * token_count, 300)  # Minimum height of 300px
            
            # Convert to HTML for better formatting
            html_table = color_df.to_html(escape=False, index=False)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
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
                
                # Update layout
                fig.update_layout(
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                    margin=dict(t=60, b=60, l=20, r=20),
                    height=400
                
                # Display percentage text in middle
                better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
                fig.add_annotation(
                    text=f"{better_percentage:.1f}%<br>Better",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show summary of SurfFuture performance
            if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
                surf_values = df['SurfFuture'].dropna()
                nonsurf_values = df['Avg (Non-Surf)'].dropna()
                
                if not surf_values.empty and not nonsurf_values.empty:
                    # Match indices to compare only pairs with both values
                    common_indices = surf_values.index.intersection(nonsurf_values.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                        total_count = len(common_indices)
                        
                        # Calculate percentages
                        surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                        
                        # Display summary box
                        st.markdown(f"""
                        <div class="info-box">
                        <b>SurfFuture Performance Summary:</b><br>
                        • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Calculate averages
                        surf_avg = surf_values.mean()
                        nonsurf_avg = nonsurf_values.mean()
                        
                        if surf_avg < nonsurf_avg:
                            improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                            st.success(f"📉 **SurfFuture average spread ({surf_avg:.6f}) is {improvement:.2f}% lower than other exchanges ({nonsurf_avg:.6f})**")
                        
                # Calculate separate stats for majors and altcoins
                major_tokens_df = df[df['pair_name'].apply(is_major)]
                altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
                
                # For Major tokens
                if not major_tokens_df.empty:
                    surf_major = major_tokens_df['SurfFuture'].dropna()
                    nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                    
                    if not surf_major.empty and not nonsurf_major.empty:
                        common_indices = surf_major.index.intersection(nonsurf_major.index)
                        if len(common_indices) > 0:
                            surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                            total_count = len(common_indices)
                            
                            if total_count > 0:
                                surf_major_avg = surf_major.mean()
                                nonsurf_major_avg = nonsurf_major.mean()
                                
                                st.markdown(f"""
                                <div class="info-box">
                                <b>Major Tokens (100K):</b><br>
                                • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                                • SurfFuture average: {surf_major_avg:.6f} vs Non-Surf average: {nonsurf_major_avg:.6f}
                                </div>
                                """, unsafe_allow_html=True)
                
                # For Altcoin tokens
                if not altcoin_tokens_df.empty:
                    surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                    nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                    
                    if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                        common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                        if len(common_indices) > 0:
                            surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                            total_count = len(common_indices)
                            
                            if total_count > 0:
                                surf_altcoin_avg = surf_altcoin.mean()
                                nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                                
                                st.markdown(f"""
                                <div class="info-box">
                                <b>Altcoin Tokens (50K):</b><br>
                                • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                                • SurfFuture average: {surf_altcoin_avg:.6f} vs Non-Surf average: {nonsurf_altcoin_avg:.6f}
                                </div>
                                """, unsafe_allow_html=True)
                                
                # Bar chart comparing exchanges
                st.markdown("### Average Spread by Exchange")
                
                # Calculate average for each exchange
                exchange_avgs = {}
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                    if exchange in df.columns:
                        values = df[exchange].dropna()
                        if not values.empty:
                            exchange_avgs[exchange] = values.mean()
                
                if exchange_avgs:
                    # Create data frame for plotting
                    avg_df = pd.DataFrame({
                        'Exchange': list(exchange_avgs.keys()),
                        'Average Spread': list(exchange_avgs.values())
                    })
                    
                    # Sort by average spread (ascending)
                    avg_df = avg_df.sort_values('Average Spread')
                    
                    # Create bar chart
                    fig = px.bar(
                        avg_df,
                        x='Exchange',
                        y='Average Spread',
                        title=f"Average Spread by Exchange ({scale_label})",
                        color='Exchange',
                        text='Average Spread',
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    
                    # Format the bars
                    fig.update_traces(
                        texttemplate='%{y:.6f}',
                        textposition='outside'
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Exchange",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=400
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No fee2 (100K/50K) data available for analysis")
    
    # === TAB 3: SPREAD BY SIZE (ALL SIZES) ===
    with tab3:
        st.markdown('<div class="header-style">Spread Analysis by Size</div>', unsafe_allow_html=True)
        
        # Explanation of sizes
        st.markdown("""
        <div class="info-box">
        <b>Trading Size Definitions:</b><br>
        • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K<br>
        • <b>Altcoin tokens</b>: 20K, 50K, 100K, 200K<br>
        <br>
        Tables show daily averages of 10-minute spread data points across different size tiers.
        </div>
        """, unsafe_allow_html=True)
        
        if daily_avg_data is not None and not daily_avg_data.empty:
            # Group tokens as majors and altcoins for better organization
            major_tokens = [t for t in selected_tokens if is_major(t)]
            altcoin_tokens = [t for t in selected_tokens if not is_major(t)]
            
            # Create dictionary mapping for depths
            fee_depth_map_major = {
                'avg_fee1': '50K', 'avg_fee2': '100K', 'avg_fee3': '200K', 'avg_fee4': '500K'
            }
            fee_depth_map_altcoin = {
                'avg_fee1': '20K', 'avg_fee2': '50K', 'avg_fee3': '100K', 'avg_fee4': '200K'
            }
            
            # Check available fee columns
            available_fee_cols = [col for col in ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4'] 
                                  if col in daily_avg_data.columns]
            
            if not available_fee_cols:
                st.error("Required fee columns not found in the data. Check database schema.")
            else:
                # --- Average of all exchanges (excluding SurfFuture) ---
                st.markdown("### Average Spreads Across Exchanges")
                
                try:
                    # Create filtered data without surfFuture for comparison
                    non_surf_data = daily_avg_data[daily_avg_data['source'].isin(['binanceFuture', 'gateFuture', 'hyperliquidFuture'])].copy()
                    
                    if not non_surf_data.empty:
                        # Calculate average across exchanges for each pair and depth
                        avg_all_exchanges = non_surf_data.groupby(['pair_name'])[available_fee_cols].mean().reset_index()
                        
                        # Process majors
                        if major_tokens:
                            major_df = avg_all_exchanges[avg_all_exchanges['pair_name'].isin(major_tokens)].copy()
                            if not major_df.empty:
                                # Rename columns for display
                                major_df_display = major_df.copy()
                                for col in available_fee_cols:
                                    if col in fee_depth_map_major:
                                        major_df_display = major_df_display.rename(columns={col: fee_depth_map_major[col]})
                                
                                # Sort alphabetically
                                major_df_display = major_df_display.sort_values('pair_name')
                                
                                # Determine scale factor
                                depth_cols = [fee_depth_map_major[col] for col in available_fee_cols if col in fee_depth_map_major]
                                values = []
                                for col in depth_cols:
                                    if col in major_df_display.columns:
                                        values.extend(major_df_display[col].dropna().tolist())
                                
                                scale_factor = 1
                                scale_label = ""
                                if values:
                                    mean_fee = sum(values) / len(values)
                                    if mean_fee < 0.001:
                                        scale_factor = 1000
                                        scale_label = "× 1,000"
                                    elif mean_fee < 0.0001:
                                        scale_factor = 10000
                                        scale_label = "× 10,000"
                                    elif mean_fee < 0.00001:
                                        scale_factor = 100000
                                        scale_label = "× 100,000"
                                
                                # Apply scaling
                                if scale_factor > 1:
                                    for col in depth_cols:
                                        if col in major_df_display.columns:
                                            major_df_display[col] = major_df_display[col] * scale_factor
                                    
                                    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                
                                # Apply color coding
                                color_df = major_df_display.copy()
                                for col in depth_cols:
                                    if col in color_df.columns:
                                        # Determine thresholds based on column values
                                        col_values = color_df[col].dropna().tolist()
                                        if col_values:
                                            # Dynamic thresholds based on percentiles
                                            q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                            median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                            q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                            thresholds = [q1, median, q3]
                                            
                                            color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                
                                # Round values for display
                                for col in depth_cols:
                                    if col in major_df_display.columns:
                                        major_df_display[col] = major_df_display[col].round(6)
                                
                                # Rename pair_name column
                                color_df = color_df.rename(columns={'pair_name': 'Token'})
                                
                                st.markdown("#### Major Tokens - Average Across Exchanges")
                                
                                # Display the table with HTML formatting
                                html_table = color_df.to_html(escape=False, index=False)
                                st.markdown(html_table, unsafe_allow_html=True)
                                
                                # Create visualization - line chart showing spread by size
                                st.markdown("#### Spread vs. Size Relationship (Major Tokens)")
                                
                                # Calculate average spread for each size tier
                                size_averages = {}
                                for col in depth_cols:
                                    if col in major_df_display.columns:
                                        size_averages[col] = major_df_display[col].mean()
                                
                                if size_averages:
                                    # Create data frame for plotting
                                    sizes = list(size_averages.keys())
                                    spreads = list(size_averages.values())
                                    
                                    # Create line chart
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=sizes,
                                        y=spreads,
                                        mode='lines+markers',
                                        line=dict(color='#1E88E5', width=3),
                                        marker=dict(size=10, color='#1E88E5'),
                                        name='Average Spread'
                                    ))
                                    
                                    # Format the layout
                                    fig.update_layout(
                                        title=f"Relationship Between Size and Spread - Major Tokens ({scale_label})",
                                        xaxis_title="Size Tier",
                                        yaxis_title=f"Average Spread {scale_label}",
                                        height=400,
                                        xaxis=dict(
                                            tickfont=dict(size=14),
                                            tickmode='array',
                                            tickvals=sizes
                                        ),
                                        yaxis=dict(
                                            tickformat='.6f'
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Process altcoins
                        if altcoin_tokens:
                            altcoin_df = avg_all_exchanges[avg_all_exchanges['pair_name'].isin(altcoin_tokens)].copy()
                            if not altcoin_df.empty:
                                # Rename columns for display
                                altcoin_df_display = altcoin_df.copy()
                                for col in available_fee_cols:
                                    if col in fee_depth_map_altcoin:
                                        altcoin_df_display = altcoin_df_display.rename(columns={col: fee_depth_map_altcoin[col]})
                                
                                # Sort alphabetically
                                altcoin_df_display = altcoin_df_display.sort_values('pair_name')
                                
                                # Determine scale factor
                                depth_cols = [fee_depth_map_altcoin[col] for col in available_fee_cols if col in fee_depth_map_altcoin]
                                values = []
                                for col in depth_cols:
                                    if col in altcoin_df_display.columns:
                                        values.extend(altcoin_df_display[col].dropna().tolist())
                                
                                scale_factor = 1
                                scale_label = ""
                                if values:
                                    mean_fee = sum(values) / len(values)
                                    if mean_fee < 0.001:
                                        scale_factor = 1000
                                        scale_label = "× 1,000"
                                    elif mean_fee < 0.0001:
                                        scale_factor = 10000
                                        scale_label = "× 10,000"
                                    elif mean_fee < 0.00001:
                                        scale_factor = 100000
                                        scale_label = "× 100,000"
                                
                                # Apply scaling
                                if scale_factor > 1:
                                    for col in depth_cols:
                                        if col in altcoin_df_display.columns:
                                            altcoin_df_display[col] = altcoin_df_display[col] * scale_factor
                                    
                                    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                
                                # Apply color coding
                                color_df = altcoin_df_display.copy()
                                for col in depth_cols:
                                    if col in color_df.columns:
                                        # Determine thresholds based on column values
                                        col_values = color_df[col].dropna().tolist()
                                        if col_values:
                                            # Dynamic thresholds based on percentiles
                                            q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                            median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                            q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                            thresholds = [q1, median, q3]
                                            
                                            color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                
                                # Round values for display
                                for col in depth_cols:
                                    if col in altcoin_df_display.columns:
                                        altcoin_df_display[col] = altcoin_df_display[col].round(6)
                                
                                # Rename pair_name column
                                color_df = color_df.rename(columns={'pair_name': 'Token'})
                                
                                st.markdown("#### Altcoin Tokens - Average Across Exchanges")
                                
                                # Display the table with HTML formatting
                                html_table = color_df.to_html(escape=False, index=False)
                                st.markdown(html_table, unsafe_allow_html=True)
                                
                                # Create visualization - line chart showing spread by size
                                st.markdown("#### Spread vs. Size Relationship (Altcoin Tokens)")
                                
                                # Calculate average spread for each size tier
                                size_averages = {}
                                for col in depth_cols:
                                    if col in altcoin_df_display.columns:
                                        size_averages[col] = altcoin_df_display[col].mean()
                                
                                if size_averages:
                                    # Create data frame for plotting
                                    sizes = list(size_averages.keys())
                                    spreads = list(size_averages.values())
                                    
                                    # Create line chart
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=sizes,
                                        y=spreads,
                                        mode='lines+markers',
                                        line=dict(color='#FF9800', width=3),
                                        marker=dict(size=10, color='#FF9800'),
                                        name='Average Spread'
                                    ))
                                    
                                    # Format the layout
                                    fig.update_layout(
                                        title=f"Relationship Between Size and Spread - Altcoin Tokens ({scale_label})",
                                        xaxis_title="Size Tier",
                                        yaxis_title=f"Average Spread {scale_label}",
                                        height=400,
                                        xaxis=dict(
                                            tickfont=dict(size=14),
                                            tickmode='array',
                                            tickvals=sizes
                                        ),
                                        yaxis=dict(
                                            tickformat='.6f'
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                    # --- Individual exchange analysis (excluding SurfFuture) ---
                    st.markdown("### Individual Exchange Analysis")
                    
                    # Create tabs for each exchange (excluding SurfFuture)
                    non_surf_exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture"]
                    exchange_tabs = st.tabs([exchanges_display[ex] for ex in non_surf_exchanges])
                    
                    # Process each exchange
                    for i, exchange_source in enumerate(non_surf_exchanges):
                        with exchange_tabs[i]:
                            exchange_display_name = exchanges_display[exchange_source]
                            st.markdown(f"#### {exchange_display_name} Spreads Analysis")
                            
                            # Filter data for this exchange
                            exchange_data = daily_avg_data[daily_avg_data['source'] == exchange_source].copy()
                            
                            if not exchange_data.empty:
                                # --- Process majors ---
                                if major_tokens:
                                    major_ex_df = exchange_data[exchange_data['pair_name'].isin(major_tokens)].copy()
                                    if not major_ex_df.empty:
                                        # Create a display DataFrame with available columns
                                        columns_to_select = ['pair_name'] + [col for col in available_fee_cols if col in major_ex_df.columns]
                                        major_ex_display = major_ex_df[columns_to_select].copy()
                                        
                                        # Rename columns for display
                                        for col in available_fee_cols:
                                            if col in fee_depth_map_major and col in major_ex_display.columns:
                                                major_ex_display = major_ex_display.rename(columns={col: fee_depth_map_major[col]})
                                        
                                        # Sort alphabetically
                                        major_ex_display = major_ex_display.sort_values('pair_name')
                                        
                                        # Determine scale factor
                                        depth_cols = [fee_depth_map_major[col] for col in available_fee_cols if col in fee_depth_map_major]
                                        values = []
                                        for col in depth_cols:
                                            if col in major_ex_display.columns:
                                                values.extend(major_ex_display[col].dropna().tolist())
                                        
                                        scale_factor = 1
                                        scale_label = ""
                                        if values:
                                            mean_fee = sum(values) / len(values)
                                            if mean_fee < 0.001:
                                                scale_factor = 1000
                                                scale_label = "× 1,000"
                                            elif mean_fee < 0.0001:
                                                scale_factor = 10000
                                                scale_label = "× 10,000"
                                            elif mean_fee < 0.00001:
                                                scale_factor = 100000
                                                scale_label = "× 100,000"
                                        
                                        # Apply scaling
                                        if scale_factor > 1:
                                            for col in depth_cols:
                                                if col in major_ex_display.columns:
                                                    major_ex_display[col] = major_ex_display[col] * scale_factor
                                            
                                            st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                        
                                        # Apply color coding
                                        color_df = major_ex_display.copy()
                                        for col in depth_cols:
                                            if col in color_df.columns:
                                                # Determine thresholds based on column values
                                                col_values = color_df[col].dropna().tolist()
                                                if col_values:
                                                    # Dynamic thresholds based on percentiles
                                                    q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                                    median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                                    q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                                    thresholds = [q1, median, q3]
                                                    
                                                    color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                        
                                        # Round values for display
                                        for col in depth_cols:
                                            if col in major_ex_display.columns:
                                                major_ex_display[col] = major_ex_display[col].round(6)
                                        
                                        # Rename pair_name column
                                        color_df = color_df.rename(columns={'pair_name': 'Token'})
                                        
                                        st.markdown(f"##### Major Tokens - {exchange_display_name}")
                                        
                                        # Display the table with HTML formatting
                                        html_table = color_df.to_html(escape=False, index=False)
                                        st.markdown(html_table, unsafe_allow_html=True)
                                        
                                        # Create visualization - line chart showing spread by size
                                        st.markdown(f"##### Spread vs. Size Relationship - {exchange_display_name} (Major Tokens)")
                                        
                                        # Calculate average spread for each size tier
                                        size_averages = {}
                                        for col in depth_cols:
                                            if col in major_ex_display.columns:
                                                size_averages[col] = major_ex_display[col].mean()
                                        
                                        if size_averages:
                                            # Create data frame for plotting
                                            sizes = list(size_averages.keys())
                                            spreads = list(size_averages.values())
                                            
                                            # Create line chart
                                            fig = go.Figure()
                                            
                                            fig.add_trace(go.Scatter(
                                                x=sizes,
                                                y=spreads,
                                                mode='lines+markers',
                                                line=dict(color='#1E88E5', width=3),
                                                marker=dict(size=10, color='#1E88E5'),
                                                name='Average Spread'
                                            ))
                                            
                                            # Format the layout
                                            fig.update_layout(
                                                title=f"{exchange_display_name} - Spread vs. Size (Major Tokens) {scale_label}",
                                                xaxis_title="Size Tier",
                                                yaxis_title=f"Average Spread {scale_label}",
                                                height=400,
                                                xaxis=dict(
                                                    tickfont=dict(size=14),
                                                    tickmode='array',
                                                    tickvals=sizes
                                                ),
                                                yaxis=dict(
                                                    tickformat='.6f'
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                
                                # --- Process altcoins ---
                                if altcoin_tokens:
                                    altcoin_ex_df = exchange_data[exchange_data['pair_name'].isin(altcoin_tokens)].copy()
                                    if not altcoin_ex_df.empty:
                                        # Create a display DataFrame with available columns
                                        columns_to_select = ['pair_name'] + [col for col in available_fee_cols if col in altcoin_ex_df.columns]
                                        altcoin_ex_display = altcoin_ex_df[columns_to_select].copy()
                                        
                                        # Rename columns for display
                                        for col in available_fee_cols:
                                            if col in fee_depth_map_altcoin and col in altcoin_ex_display.columns:
                                                altcoin_ex_display = altcoin_ex_display.rename(columns={col: fee_depth_map_altcoin[col]})
                                        
                                        # Sort alphabetically
                                        altcoin_ex_display = altcoin_ex_display.sort_values('pair_name')
                                        
                                        # Determine scale factor
                                        depth_cols = [fee_depth_map_altcoin[col] for col in available_fee_cols if col in fee_depth_map_altcoin]
                                        values = []
                                        for col in depth_cols:
                                            if col in altcoin_ex_display.columns:
                                                values.extend(altcoin_ex_display[col].dropna().tolist())
                                        
                                        scale_factor = 1
                                        scale_label = ""
                                        if values:
                                            mean_fee = sum(values) / len(values)
                                            if mean_fee < 0.001:
                                                scale_factor = 1000
                                                scale_label = "× 1,000"
                                            elif mean_fee < 0.0001:
                                                scale_factor = 10000
                                                scale_label = "× 10,000"
                                            elif mean_fee < 0.00001:
                                                scale_factor = 100000
                                                scale_label = "× 100,000"
                                        
                                        # Apply scaling
                                        if scale_factor > 1:
                                            for col in depth_cols:
                                                if col in altcoin_ex_display.columns:
                                                    altcoin_ex_display[col] = altcoin_ex_display[col] * scale_factor
                                            
                                            st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                        
                                        # Apply color coding
                                        color_df = altcoin_ex_display.copy()
                                        for col in depth_cols:
                                            if col in color_df.columns:
                                                # Determine thresholds based on column values
                                                col_values = color_df[col].dropna().tolist()
                                                if col_values:
                                                    # Dynamic thresholds based on percentiles
                                                    q1 = np.percentile(col_values, 25) if len(col_values) >= 4 else min(col_values)
                                                    median = np.percentile(col_values, 50) if len(col_values) >= 2 else col_values[0]
                                                    q3 = np.percentile(col_values, 75) if len(col_values) >= 4 else max(col_values)
                                                    thresholds = [q1, median, q3]
                                                    
                                                    color_df[col] = color_df[col].apply(lambda x: color_code_value(x, thresholds) if not pd.isna(x) else "")
                                        
                                        # Round values for display
                                        for col in depth_cols:
                                            if col in altcoin_ex_display.columns:
                                                altcoin_ex_display[col] = altcoin_ex_display[col].round(6)
                                        
                                        # Rename pair_name column
                                        color_df = color_df.rename(columns={'pair_name': 'Token'})
                                        
                                        st.markdown(f"##### Altcoin Tokens - {exchange_display_name}")
                                        
                                        # Display the table with HTML formatting
                                        html_table = color_df.to_html(escape=False, index=False)
                                        st.markdown(html_table, unsafe_allow_html=True)
                                        
                                        # Create visualization - line chart showing spread by size
                                        st.markdown(f"##### Spread vs. Size Relationship - {exchange_display_name} (Altcoin Tokens)")
                                        
                                        # Calculate average spread for each size tier
                                        size_averages = {}
                                        for col in depth_cols:
                                            if col in altcoin_ex_display.columns:
                                                size_averages[col] = altcoin_ex_display[col].mean()
                                        
                                        if size_averages:
                                            # Create data frame for plotting
                                            sizes = list(size_averages.keys())
                                            spreads = list(size_averages.values())
                                            
                                            # Create line chart
                                            fig = go.Figure()
                                            
                                            fig.add_trace(go.Scatter(
                                                x=sizes,
                                                y=spreads,
                                                mode='lines+markers',
                                                line=dict(color='#FF9800', width=3),
                                                marker=dict(size=10, color='#FF9800'),
                                                name='Average Spread'
                                            ))
                                            
                                            # Format the layout
                                            fig.update_layout(
                                                title=f"{exchange_display_name} - Spread vs. Size (Altcoin Tokens) {scale_label}",
                                                xaxis_title="Size Tier",
                                                yaxis_title=f"Average Spread {scale_label}",
                                                height=400,
                                                xaxis=dict(
                                                    tickfont=dict(size=14),
                                                    tickmode='array',
                                                    tickvals=sizes
                                                ),
                                                yaxis=dict(
                                                    tickformat='.6f'
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No data available for {exchange_display_name}")
                                
                    # Cross-Exchange Comparison by Size
                    st.markdown("### Cross-Exchange Comparison by Size")
                    
                    # Group data by token and exchange for comparison
                    exchange_comparison = {}
                    
                    # Process major tokens
                    if major_tokens and non_surf_data is not None and not non_surf_data.empty:
                        # Filter for major tokens
                        major_data = non_surf_data[non_surf_data['pair_name'].isin(major_tokens)].copy()
                        
                        if not major_data.empty:
                            # Calculate average for each exchange and fee level
                            for exchange_source in non_surf_exchanges:
                                exchange_display_name = exchanges_display[exchange_source]
                                exchange_comparison[exchange_display_name] = {}
                                
                                # Filter for this exchange
                                exchange_df = major_data[major_data['source'] == exchange_source].copy()
                                
                                if not exchange_df.empty:
                                    # Calculate average for each fee level
                                    for i, fee_col in enumerate(available_fee_cols):
                                        if fee_col in exchange_df.columns:
                                            values = exchange_df[fee_col].dropna()
                                            if not values.empty:
                                                # Map to correct size label
                                                size_label = fee_depth_map_major.get(fee_col, f'Size {i+1}')
                                                exchange_comparison[exchange_display_name][size_label] = values.mean()
                            
                            # Create a comparison chart if we have data
                            if exchange_comparison:
                                # Prepare data for plotting
                                comparison_data = []
                                for exchange, sizes in exchange_comparison.items():
                                    for size, value in sizes.items():
                                        comparison_data.append({
                                            'Exchange': exchange,
                                            'Size': size,
                                            'Spread': value
                                        })
                                
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    
                                    # Determine scale factor
                                    values = comparison_df['Spread'].tolist()
                                    scale_factor = 1
                                    scale_label = ""
                                    if values:
                                        mean_fee = sum(values) / len(values)
                                        if mean_fee < 0.001:
                                            scale_factor = 1000
                                            scale_label = "× 1,000"
                                        elif mean_fee < 0.0001:
                                            scale_factor = 10000
                                            scale_label = "× 10,000"
                                        elif mean_fee < 0.00001:
                                            scale_factor = 100000
                                            scale_label = "× 100,000"
                                    
                                    # Apply scaling
                                    if scale_factor > 1:
                                        comparison_df['Spread'] = comparison_df['Spread'] * scale_factor
                                        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                    
                                    # Create grouped bar chart
                                    st.markdown("#### Major Tokens - Exchange Comparison by Size")
                                    
                                    # Sort by size tiers for better visualization
                                    size_order = ['50K', '100K', '200K', '500K']
                                    comparison_df['Size'] = pd.Categorical(comparison_df['Size'], categories=size_order, ordered=True)
                                    comparison_df = comparison_df.sort_values('Size')
                                    
                                    fig = px.bar(
                                        comparison_df,
                                        x='Size',
                                        y='Spread',
                                        color='Exchange',
                                        barmode='group',
                                        title=f"Exchange Comparison by Size - Major Tokens {scale_label}",
                                        color_discrete_map={
                                            'Binance': '#2196F3',
                                            'Gate': '#FFC107',
                                            'Hyperliquid': '#FF5722'
                                        }
                                    
                                    # Format the layout
                                    fig.update_layout(
                                        xaxis_title="Size Tier",
                                        yaxis_title=f"Average Spread {scale_label}",
                                        height=500,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Process altcoin tokens
                    exchange_comparison = {}
                    if altcoin_tokens and non_surf_data is not None and not non_surf_data.empty:
                        # Filter for altcoin tokens
                        altcoin_data = non_surf_data[non_surf_data['pair_name'].isin(altcoin_tokens)].copy()
                        
                        if not altcoin_data.empty:
                            # Calculate average for each exchange and fee level
                            for exchange_source in non_surf_exchanges:
                                exchange_display_name = exchanges_display[exchange_source]
                                exchange_comparison[exchange_display_name] = {}
                                
                                # Filter for this exchange
                                exchange_df = altcoin_data[altcoin_data['source'] == exchange_source].copy()
                                
                                if not exchange_df.empty:
                                    # Calculate average for each fee level
                                    for i, fee_col in enumerate(available_fee_cols):
                                        if fee_col in exchange_df.columns:
                                            values = exchange_df[fee_col].dropna()
                                            if not values.empty:
                                                # Map to correct size label
                                                size_label = fee_depth_map_altcoin.get(fee_col, f'Size {i+1}')
                                                exchange_comparison[exchange_display_name][size_label] = values.mean()
                            
                            # Create a comparison chart if we have data
                            if exchange_comparison:
                                # Prepare data for plotting
                                comparison_data = []
                                for exchange, sizes in exchange_comparison.items():
                                    for size, value in sizes.items():
                                        comparison_data.append({
                                            'Exchange': exchange,
                                            'Size': size,
                                            'Spread': value
                                        })
                                
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    
                                    # Determine scale factor
                                    values = comparison_df['Spread'].tolist()
                                    scale_factor = 1
                                    scale_label = ""
                                    if values:
                                        mean_fee = sum(values) / len(values)
                                        if mean_fee < 0.001:
                                            scale_factor = 1000
                                            scale_label = "× 1,000"
                                        elif mean_fee < 0.0001:
                                            scale_factor = 10000
                                            scale_label = "× 10,000"
                                        elif mean_fee < 0.00001:
                                            scale_factor = 100000
                                            scale_label = "× 100,000"
                                    
                                    # Apply scaling
                                    if scale_factor > 1:
                                        comparison_df['Spread'] = comparison_df['Spread'] * scale_factor
                                        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
                                    
                                    # Create grouped bar chart
                                    st.markdown("#### Altcoin Tokens - Exchange Comparison by Size")
                                    
                                    # Sort by size tiers for better visualization
                                    size_order = ['20K', '50K', '100K', '200K']
                                    comparison_df['Size'] = pd.Categorical(comparison_df['Size'], categories=size_order, ordered=True)
                                    comparison_df = comparison_df.sort_values('Size')
                                    
                                    fig = px.bar(
                                        comparison_df,
                                        x='Size',
                                        y='Spread',
                                        color='Exchange',
                                        barmode='group',
                                        title=f"Exchange Comparison by Size - Altcoin Tokens {scale_label}",
                                        color_discrete_map={
                                            'Binance': '#2196F3',
                                            'Gate': '#FFC107',
                                            'Hyperliquid': '#FF5722'
                                        }
                                    
                                    # Format the layout
                                    fig.update_layout(
                                        xaxis_title="Size Tier",
                                        yaxis_title=f"Average Spread {scale_label}",
                                        height=500,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing spread data by size: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            st.warning("No daily average spread data available for the selected tokens and time period.")
    
    # Footer with explanatory information
    with st.expander("Understanding Exchange Spreads"):
        st.markdown("""
        ### About This Dashboard
        
        This dashboard provides comprehensive analysis of trading spreads across multiple cryptocurrency exchanges. 
        
        ### Key Concepts:
        
        - **Spread**: The difference between the buy and sell price, representing the cost of trading.
        
        - **Trading Sizes**: Different order sizes for analysis:
          - **Major tokens** (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K
          - **Altcoin tokens**: 20K, 50K, 100K, 200K
        
        - **Fee Columns**: 
          - `fee1`, `fee2`, `fee3`, `fee4` correspond to spreads at different trading sizes
        
        - **Exchange Comparison**: The dashboard highlights when SurfFuture has tighter spreads (lower fees) than the average of other exchanges.
        
        ### Interpreting the Data:
        
        - Lower spread values indicate better pricing for traders
        - Color coding highlights the relative spread values (blue/green for lower spreads, red/orange for higher spreads)
        - The "Better" percentage shows how much lower SurfFuture's spread is compared to other exchanges (positive values mean SurfFuture is better)
        - The scaling factor is applied to make small decimal values more readable
        
        ### Data Source:
        
        Data is fetched from the `oracle_exchange_fee` table, with 10-minute interval data points that represent the average of previous 10 one-minute points.
        """)
    
    # Execute the app
    if __name__ == '__main__':
        pass  # The app is already running


def render_tab_3():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    refresh = st.button("Refresh Data", key="refresh_button_3")
    if refresh:
        st.cache_data.clear()
        st.experimental_rerun()
    with st.container():
    # Save this as pages/05_Daily_Volatility_Table.py in your Streamlit app folder
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz
    
    
    # --- DB CONFIG ---
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()
    
    # --- UI Setup ---
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Daily Volatility Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")
    
    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    rolling_window = 20  # Window size for volatility calculation
    expected_points = 48  # Expected data points per pair over 24 hours
    singapore_timezone = pytz.timezone('Asia/Singapore')
    
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set extreme volatility threshold
    extreme_vol_threshold = 1.0  # 100% annualized volatility
    
    # Fetch all available tokens from DB
    @st.cache_data(show_spinner="Fetching tokens...")
    def fetch_all_tokens():
        query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                st.error("No tokens found in the database.")
                return []
            return df['pair_name'].tolist()
        except Exception as e:
            st.error(f"Error fetching tokens: {e}")
            return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback
    
    all_tokens = fetch_all_tokens()
    
    # UI Controls
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
    
    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    if not selected_tokens:
        st.warning("Please select at least one token")
        st.stop()
    
    # Function to calculate various volatility metrics
    def calculate_volatility_metrics(price_series):
        if price_series is None or len(price_series) < 2:
            return {
                'realized_vol': np.nan,
                'parkinson_vol': np.nan,
                'gk_vol': np.nan,
                'rs_vol': np.nan
            }
        
        try:
            # Calculate log returns
            log_returns = np.diff(np.log(price_series))
            
            # 1. Standard deviation of returns (realized volatility)
            realized_vol = np.std(log_returns) * np.sqrt(252 * 48)  # Annualized volatility (30min bars)
            
            # For other volatility metrics, need OHLC data
            # For simplicity, we'll focus on realized volatility for now
            # But the structure allows adding more volatility metrics
            
            return {
                'realized_vol': realized_vol,
                'parkinson_vol': np.nan,  # Placeholder for Parkinson volatility
                'gk_vol': np.nan,         # Placeholder for Garman-Klass volatility
                'rs_vol': np.nan          # Placeholder for Rogers-Satchell volatility
            }
        except Exception as e:
            print(f"Error in volatility calculation: {e}")
            return {
                'realized_vol': np.nan,
                'parkinson_vol': np.nan,
                'gk_vol': np.nan,
                'rs_vol': np.nan
            }
    
    # Volatility classification function
    def classify_volatility(vol):
        if pd.isna(vol):
            return ("UNKNOWN", 0, "Insufficient data")
        elif vol < 0.30:  # 30% annualized volatility threshold for low volatility
            return ("LOW", 1, "Low volatility")
        elif vol < 0.60:  # 60% annualized volatility threshold for medium volatility
            return ("MEDIUM", 2, "Medium volatility")
        elif vol < 1.00:  # 100% annualized volatility threshold for high volatility
            return ("HIGH", 3, "High volatility")
        else:
            return ("EXTREME", 4, "Extreme volatility")
    
    # Function to generate aligned 30-minute time blocks for the past 24 hours
    def generate_aligned_time_blocks(current_time):
        """
        Generate fixed 30-minute time blocks for past 24 hours,
        aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
        """
        # Round down to the nearest 30-minute mark
        if current_time.minute < 30:
            # Round down to XX:00
            latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
        else:
            # Round down to XX:30
            latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
        
        # Generate block labels for display
        blocks = []
        for i in range(48):  # 24 hours of 30-minute blocks
            block_end = latest_complete_block_end - timedelta(minutes=i*30)
            block_start = block_end - timedelta(minutes=30)
            block_label = f"{block_start.strftime('%H:%M')}"
            blocks.append((block_start, block_end, block_label))
        
        return blocks
    
    # Generate aligned time blocks
    aligned_time_blocks = generate_aligned_time_blocks(now_sg)
    time_block_labels = [block[2] for block in aligned_time_blocks]
    
    # Fetch and calculate volatility for a token with 30min timeframe
    @st.cache_data(ttl=600, show_spinner="Calculating volatility metrics...")
    def fetch_and_calculate_volatility(token):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
    
        query = f"""
        SELECT 
            created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
            final_price, 
            pair_name
        FROM public.oracle_price_log
        WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{token}';
        """
        try:
            print(f"[{token}] Executing query: {query}")
            df = pd.read_sql(query, engine)
            print(f"[{token}] Query executed. DataFrame shape: {df.shape}")
    
            if df.empty:
                print(f"[{token}] No data found.")
                return None
    
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Create 1-minute OHLC data
            one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
            if one_min_ohlc.empty:
                print(f"[{token}] No OHLC data after resampling.")
                return None
                
            # Calculate rolling volatility on 1-minute data
            one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(
                lambda x: calculate_volatility_metrics(x)['realized_vol']
            
            # Resample to exactly 30min intervals aligned with clock
            thirty_min_vol = one_min_ohlc['realized_vol'].resample('30min', closed='left', label='left').mean().dropna()
            
            if thirty_min_vol.empty:
                print(f"[{token}] No 30-min volatility data.")
                return None
                
            # Get last 24 hours (48 30-minute bars)
            last_24h_vol = thirty_min_vol.tail(48)  # Get up to last 48 periods (24 hours)
            last_24h_vol = last_24h_vol.to_frame()
            
            # Store original datetime index for reference
            last_24h_vol['original_datetime'] = last_24h_vol.index
            
            # Format time label to match our aligned blocks (HH:MM format)
            last_24h_vol['time_label'] = last_24h_vol.index.strftime('%H:%M')
            
            # Calculate 24-hour average volatility
            last_24h_vol['avg_24h_vol'] = last_24h_vol['realized_vol'].mean()
            
            # Classify volatility
            last_24h_vol['vol_info'] = last_24h_vol['realized_vol'].apply(classify_volatility)
            last_24h_vol['vol_regime'] = last_24h_vol['vol_info'].apply(lambda x: x[0])
            last_24h_vol['vol_desc'] = last_24h_vol['vol_info'].apply(lambda x: x[2])
            
            # Also classify the 24-hour average
            last_24h_vol['avg_vol_info'] = last_24h_vol['avg_24h_vol'].apply(classify_volatility)
            last_24h_vol['avg_vol_regime'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[0])
            last_24h_vol['avg_vol_desc'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[2])
            
            # Flag extreme volatility events
            last_24h_vol['is_extreme'] = last_24h_vol['realized_vol'] >= extreme_vol_threshold
            
            print(f"[{token}] Successful Volatility Calculation")
            return last_24h_vol
        except Exception as e:
            st.error(f"Error processing {token}: {e}")
            print(f"[{token}] Error processing: {e}")
            return None
    
    # Show the blocks we're analyzing
    with st.expander("View Time Blocks Being Analyzed"):
        time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                      for b in aligned_time_blocks], 
                                     columns=['Start Time', 'End Time', 'Block Label'])
        st.dataframe(time_blocks_df)
    
    # Show progress bar while calculating
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate volatility for each token
    token_results = {}
    for i, token in enumerate(selected_tokens):
        try:
            progress_bar.progress((i) / len(selected_tokens))
            status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
            result = fetch_and_calculate_volatility(token)
            if result is not None:
                token_results[token] = result
        except Exception as e:
            st.error(f"Error processing token {token}: {e}")
            print(f"Error processing token {token} in main loop: {e}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")
    
    # Create table for display
    if token_results:
        # Create table data
        table_data = {}
        for token, df in token_results.items():
            vol_series = df.set_index('time_label')['realized_vol']
            table_data[token] = vol_series
        
        # Create DataFrame with all tokens
        vol_table = pd.DataFrame(table_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(vol_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
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
        st.markdown("## Volatility Table (30min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
        st.markdown("Values shown as annualized volatility percentage")
        st.dataframe(styled_table, height=700, use_container_width=True)
        
        # Create ranking table based on average volatility
        st.subheader("Volatility Ranking (24-Hour Average, Descending Order)")
        
        ranking_data = []
        for token, df in token_results.items():
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
            # Sort by average volatility (high to low)
            ranking_df = ranking_df.sort_values(by='Avg Vol (%)', ascending=False)
            # Add rank column
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
            
            # Reset the index to remove it
            ranking_df = ranking_df.reset_index(drop=True)
            
            # Format ranking table with colors
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
        else:
            st.warning("No ranking data available.")
        
        # Identify and display extreme volatility events
        st.subheader("Extreme Volatility Events (>= 100% Annualized)")
        
        extreme_events = []
        for token, df in token_results.items():
            if not df.empty and 'is_extreme' in df.columns:
                extreme_periods = df[df['is_extreme']]
                for idx, row in extreme_periods.iterrows():
                    # Safely access values with explicit casting to avoid attribute errors
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
            # Sort by volatility (highest first)
            extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)
            
            # Reset the index to remove it
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
        
        # 24-Hour Average Volatility Distribution
        st.subheader("24-Hour Average Volatility Overview (Singapore Time)")
        avg_values = {}
        for token, df in token_results.items():
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
            
            labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
            values = [low_vol, medium_vol, high_vol, extreme_vol]
            colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
            fig.update_layout(
                title="24-Hour Average Volatility Distribution (Singapore Time)",
                height=400,
                font=dict(color="#000000", size=12),
            st.plotly_chart(fig, use_container_width=True)
            
            # Create columns for each volatility category
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                st.markdown("### Low Average Volatility Tokens")
                lv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v < 0.3]
                lv_tokens.sort(key=lambda x: x[1])
                if lv_tokens:
                    for token, value, regime in lv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")
            
            with col2:
                st.markdown("### Medium Average Volatility Tokens")
                mv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.3 <= v < 0.6]
                mv_tokens.sort(key=lambda x: x[1])
                if mv_tokens:
                    for token, value, regime in mv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")
            
            with col3:
                st.markdown("### High Average Volatility Tokens")
                hv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.6 <= v < 1.0]
                hv_tokens.sort(key=lambda x: x[1])
                if hv_tokens:
                    for token, value, regime in hv_tokens:
                        st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")
            
            with col4:
                st.markdown("### Extreme Average Volatility Tokens")
                ev_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v >= 1.0]
                ev_tokens.sort(key=lambda x: x[1], reverse=True)
                if ev_tokens:
                    for token, value, regime in ev_tokens:
                        st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
                else:
                    st.markdown("*No tokens in this category*")
        else:
            st.warning("No average volatility data available for the selected tokens.")
    
    with st.expander("Understanding the Volatility Table"):
        st.markdown("""
        ### How to Read This Table
        This table shows annualized volatility values for all selected tokens over the last 24 hours using 30-minute bars.
        Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
        
        **Color coding:**
        - **Green** (< 30%): Low volatility
        - **Yellow** (30-60%): Medium volatility
        - **Orange** (60-100%): High volatility
        - **Red** (> 100%): Extreme volatility
        
        **The intensity of the color indicates the strength of the volatility:**
        - Darker green = Lower volatility
        - Darker red = Higher volatility
        
        **Ranking Table:**
        The ranking table sorts tokens by their 24-hour average volatility from highest to lowest.
        
        **Extreme Volatility Events:**
        These are specific 30-minute periods where a token's annualized volatility exceeded 100%.
        
        **Technical details:**
        - Volatility is calculated as the standard deviation of log returns, annualized to represent the expected price variation over a year
        - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
        - The calculation uses a rolling window of 20 one-minute price points
        - The 24-hour average section shows the mean volatility across all 48 30-minute periods
        - Missing values (light gray cells) indicate insufficient data for calculation
        """)


def render_tab_4():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    refresh = st.button("Refresh Data", key="refresh_button_4")
    if refresh:
        st.cache_data.clear()
        st.experimental_rerun()
    with st.container():
    # Save this as pages/04_Daily_Hurst_Table.py in your Streamlit app folder
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    import pytz
    
    
    # --- DB CONFIG ---
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()
    
    # --- UI Setup ---
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Daily Hurst Table (30min)")
    st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")
    
    # Define parameters for the 30-minute timeframe
    timeframe = "30min"
    lookback_days = 1  # 24 hours
    rolling_window = 20  # Window size for Hurst calculation
    expected_points = 48  # Expected data points per pair over 24 hours
    singapore_timezone = pytz.timezone('Asia/Singapore')
    
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch all available tokens from DB
    @st.cache_data(show_spinner="Fetching tokens...")
    def fetch_all_tokens():
        query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                st.error("No tokens found in the database.")
                return []
            return df['pair_name'].tolist()
        except Exception as e:
            st.error(f"Error fetching tokens: {e}")
            return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback
    
    all_tokens = fetch_all_tokens()
    
    # UI Controls
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
    
    with col2:
        # Add a refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    if not selected_tokens:
        st.warning("Please select at least one token")
        st.stop()
    
    # Universal Hurst calculation function
    def universal_hurst(ts):
        print(f"universal_hurst called with ts: {ts}")
        print(f"Type of ts: {type(ts)}")
        if isinstance(ts, (list, np.ndarray, pd.Series)) and len(ts) > 0:
            print(f"First few values of ts: {ts[:5]}")
        
        if ts is None:
            print("ts is None")
            return np.nan
        
        if isinstance(ts, pd.Series) and ts.empty:
            print("ts is empty series")
            return np.nan
            
        if not isinstance(ts, (list, np.ndarray, pd.Series)):
            print(f"ts is not a list, NumPy array, or Series. Type: {type(ts)}")
            return np.nan
    
        try:
            ts = np.array(ts, dtype=float)
        except Exception as e:
            print(f"ts cannot be converted to float: {e}")
            return np.nan
    
        if len(ts) < 10 or np.any(~np.isfinite(ts)):
            print(f"ts length < 10 or non-finite values: {ts}")
            return np.nan
    
        # Convert to returns - using log returns handles any scale of asset
        epsilon = 1e-10
        adjusted_ts = ts + epsilon
        log_returns = np.diff(np.log(adjusted_ts))
        
        # If all returns are exactly zero (completely flat price), return 0.5
        if np.all(log_returns == 0):
            return 0.5
        
        # Use multiple methods and average for robustness
        hurst_estimates = []
        
        # Method 1: Rescaled Range (R/S) Analysis
        try:
            max_lag = min(len(log_returns) // 4, 40)
            lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
            rs_values = []
            for lag in lags:
                segments = len(log_returns) // lag
                if segments < 1:
                    continue
                rs_by_segment = []
                for i in range(segments):
                    segment = log_returns[i*lag:(i+1)*lag]
                    if len(segment) < lag // 2:
                        continue
                    mean_return = np.mean(segment)
                    std_return = np.std(segment)
                    if std_return == 0:
                        continue
                    cumdev = np.cumsum(segment - mean_return)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = std_return
                    rs_by_segment.append(r / s)
                if rs_by_segment:
                    rs_values.append((lag, np.mean(rs_by_segment)))
            if len(rs_values) >= 4:
                lags_log = np.log10([x[0] for x in rs_values])
                rs_log = np.log10([x[1] for x in rs_values])
                poly = np.polyfit(lags_log, rs_log, 1)
                h_rs = poly[0]
                hurst_estimates.append(h_rs)
        except Exception as e:
            print(f"Error in R/S calculation: {e}")
            pass
        
        # Method 2: Variance Method
        try:
            max_lag = min(len(log_returns) // 4, 40)
            lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
            var_values = []
            for lag in lags:
                if lag >= len(log_returns):
                    continue
                lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
                if len(lagged_returns) < 2:
                    continue
                var = np.var(lagged_returns)
                if var > 0:
                    var_values.append((lag, var))
            if len(var_values) >= 4:
                lags_log = np.log10([x[0] for x in var_values])
                var_log = np.log10([x[1] for x in var_values])
                poly = np.polyfit(lags_log, var_log, 1)
                h_var = (poly[0] + 1) / 2
                hurst_estimates.append(h_var)
        except Exception as e:
            print(f"Error in Variance calculation: {e}")
            pass
        
        # Fallback to autocorrelation method if other methods fail
        if not hurst_estimates and len(log_returns) > 1:
            try:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
                hurst_estimates.append(h_acf)
            except Exception as e:
                print(f"Error in Autocorrelation calculation: {e}")
                pass
        
        # If we have estimates, aggregate them and constrain to 0-1 range
        if hurst_estimates:
            valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
            if not valid_estimates and hurst_estimates:
                valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
            if valid_estimates:
                return np.median(valid_estimates)
        
        return 0.5
    
    # Detailed regime classification function
    def detailed_regime_classification(hurst):
        if pd.isna(hurst):
            return ("UNKNOWN", 0, "Insufficient data")
        elif hurst < 0.2:
            return ("MEAN-REVERT", 3, "Strong mean-reversion")
        elif hurst < 0.3:
            return ("MEAN-REVERT", 2, "Moderate mean-reversion")
        elif hurst < 0.4:
            return ("MEAN-REVERT", 1, "Mild mean-reversion")
        elif hurst < 0.45:
            return ("NOISE", 1, "Slight mean-reversion bias")
        elif hurst <= 0.55:
            return ("NOISE", 0, "Pure random walk")
        elif hurst < 0.6:
            return ("NOISE", 1, "Slight trending bias")
        elif hurst < 0.7:
            return ("TREND", 1, "Mild trending")
        elif hurst < 0.8:
            return ("TREND", 2, "Moderate trending")
        else:
            return ("TREND", 3, "Strong trending")
    
    # Function to generate proper 30-minute time blocks for the past 24 hours
    def generate_aligned_time_blocks(current_time):
        """
        Generate fixed 30-minute time blocks for past 24 hours,
        aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
        """
        # Round down to the nearest 30-minute mark
        if current_time.minute < 30:
            # Round down to XX:00
            latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
        else:
            # Round down to XX:30
            latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
        
        # Generate block labels for display
        blocks = []
        for i in range(48):  # 24 hours of 30-minute blocks
            block_end = latest_complete_block_end - timedelta(minutes=i*30)
            block_start = block_end - timedelta(minutes=30)
            block_label = f"{block_start.strftime('%H:%M')}"
            blocks.append(block_label)
        
        return blocks
    
    # Generate aligned time blocks
    aligned_time_blocks = generate_aligned_time_blocks(now_sg)
    
    # Fetch and calculate Hurst for a token with 30min timeframe
    @st.cache_data(ttl=600, show_spinner="Calculating Hurst exponents...")
    def fetch_and_calculate_hurst(token):
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
    
        query = f"""
        SELECT 
            created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
            final_price, 
            pair_name
        FROM public.oracle_price_log
        WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{token}';
        """
        try:
            print(f"[{token}] Executing query: {query}")
            df = pd.read_sql(query, engine)
            print(f"[{token}] Query executed. DataFrame shape: {df.shape}")
    
            if df.empty:
                print(f"[{token}] No data found.")
                return None
    
            print(f"[{token}] First few rows:\n{df.head()}")
            print(f"[{token}] DataFrame columns and types:\n{df.info()}")
    
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
            if one_min_ohlc.empty:
                print(f"[{token}] No OHLC data after resampling.")
                return None
                
            print(f"[{token}] one_min_ohlc head:\n{one_min_ohlc.head()}")
            print(f"[{token}] one_min_ohlc info:\n{one_min_ohlc.info()}")
    
            # Apply universal_hurst to the 'close' prices directly
            one_min_ohlc['Hurst'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(universal_hurst)
    
            # Resample to exactly 30min intervals aligned with clock
            thirty_min_hurst = one_min_ohlc['Hurst'].resample('30min', closed='left', label='left').mean().dropna()
            
            if thirty_min_hurst.empty:
                print(f"[{token}] No 30-min Hurst data.")
                return None
                
            last_24h_hurst = thirty_min_hurst.tail(48)  # Get up to last 48 periods (24 hours)
            last_24h_hurst = last_24h_hurst.to_frame()
            
            # Store original datetime index for reference
            last_24h_hurst['original_datetime'] = last_24h_hurst.index
            
            # Format time label to match our aligned blocks (HH:MM format)
            last_24h_hurst['time_label'] = last_24h_hurst.index.strftime('%H:%M')
            
            # Calculate regime information
            last_24h_hurst['regime_info'] = last_24h_hurst['Hurst'].apply(detailed_regime_classification)
            last_24h_hurst['regime'] = last_24h_hurst['regime_info'].apply(lambda x: x[0])
            last_24h_hurst['regime_desc'] = last_24h_hurst['regime_info'].apply(lambda x: x[2])
            
            print(f"[{token}] Successful Calculation")
            return last_24h_hurst
        except Exception as e:
            st.error(f"Error processing {token}: {e}")
            print(f"[{token}] Error processing: {e}")
            return None
    
    # Show progress bar while calculating
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate Hurst for each token
    token_results = {}
    for i, token in enumerate(selected_tokens):
        try:  # Added try-except around token processing
            progress_bar.progress((i) / len(selected_tokens))
            status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
            result = fetch_and_calculate_hurst(token)
            if result is not None:
                token_results[token] = result
        except Exception as e:
            st.error(f"Error processing token {token}: {e}")
            print(f"Error processing token {token} in main loop: {e}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")
    
    # Create table for display
    if token_results:
        # Create table data
        table_data = {}
        for token, df in token_results.items():
            hurst_series = df.set_index('time_label')['Hurst']
            table_data[token] = hurst_series
        
        # Create DataFrame with all tokens
        hurst_table = pd.DataFrame(table_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(hurst_table.index)
        ordered_times = [t for t in aligned_time_blocks if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
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
        
        st.subheader("Current Market Overview (Singapore Time)")
        
        # Use the first aligned time block that has data for each token
        latest_values = {}
        for token, df in token_results.items():
            if not df.empty and not df['Hurst'].isna().all():
                # Try to find the most recent time block in our aligned blocks
                for block_time in aligned_time_blocks[:5]:  # Check the 5 most recent blocks
                    latest_data = df[df['time_label'] == block_time]
                    if not latest_data.empty:
                        latest = latest_data['Hurst'].iloc[0]
                        regime = latest_data['regime_desc'].iloc[0]
                        latest_values[token] = (latest, regime)
                        break
                
                # If no match in aligned blocks, use the most recent data point
                if token not in latest_values:
                    latest = df['Hurst'].iloc[-1]
                    regime = df['regime_desc'].iloc[-1]
                    latest_values[token] = (latest, regime)
        
        if latest_values:
            mean_reverting = sum(1 for v, r in latest_values.values() if v < 0.4)
            random_walk = sum(1 for v, r in latest_values.values() if 0.4 <= v <= 0.6)
            trending = sum(1 for v, r in latest_values.values() if v > 0.6)
            total = mean_reverting + random_walk + trending
            
            if total > 0:  # Avoid division by zero
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean-Reverting", f"{mean_reverting} ({mean_reverting/total*100:.1f}%)", delta=f"{mean_reverting/total*100:.1f}%")
                col2.metric("Random Walk", f"{random_walk} ({random_walk/total*100:.1f}%)", delta=f"{random_walk/total*100:.1f}%")
                col3.metric("Trending", f"{trending} ({trending/total*100:.1f}%)", delta=f"{trending/total*100:.1f}%")
                
                labels = ['Mean-Reverting', 'Random Walk', 'Trending']
                values = [mean_reverting, random_walk, trending]
                colors = ['rgba(255,100,100,0.8)', 'rgba(200,200,200,0.8)', 'rgba(100,255,100,0.8)'] # Slightly more opaque
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)]) # Added black borders
                fig.update_layout(
                    title="Current Market Regime Distribution (Singapore Time)",
                    height=400,
                    font=dict(color="#000000", size=12),  # Set default font color and size
                st.plotly_chart(fig, use_container_width=True)
                
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
            else:
                st.warning("No valid data found for analysis.")
        else:
            st.warning("No data available for the selected tokens.")
    
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
    
    # Add an expandable section to view the exact time blocks being analyzed
    with st.expander("View Time Blocks Being Analyzed"):
        time_blocks_df = pd.DataFrame(aligned_time_blocks, columns=['Block Start Time'])
        st.dataframe(time_blocks_df)



import streamlit as st

st.set_page_config(page_title="ALL-IN-ONE Dashboard", layout="wide")

st.title("🧠 ALL-IN-ONE DASHBOARD")
st.markdown("Select a tab to view your dashboards. Each one can be refreshed independently.")

# Define Tabs
tab_labels = ['Cumulative pnl matrix', 'Daily pnl and trades matrix 30 mins', 'Spread matrix', 'Volatility matrix', 'Hurst matrix']
tab_functions = { "Cumulative pnl matrix": render_tab_0, "Daily pnl and trades matrix 30 mins": render_tab_1, "Spread matrix": render_tab_2, "Volatility matrix": render_tab_3, "Hurst matrix": render_tab_4 }

selected_tab = st.selectbox("Choose a Dashboard View", tab_labels)

# Render selected tab
tab_functions[selected_tab]()