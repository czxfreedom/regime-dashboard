# The main query and data processing parts remain the same,
# Add this code after calculating token_daily_avgs but before displaying individual token tables

# Create a consolidated summary table at the top
if token_daily_avgs:
    st.markdown("## Overall Daily Average Spreads by Token and Exchange")
    
    # Create a DataFrame to hold all daily averages
    all_averages_data = []
    
    for token, averages in token_daily_avgs.items():
        row_data = {'Token': token}
        
        # Add each exchange's average
        for exchange, avg in averages.items():
            if exchange != 'Average':  # Skip the overall average for now
                row_data[exchange] = avg
        
        # Calculate and add the non-surf average (average excluding 'SurfFuture')
        exchange_avgs = [v for k, v in averages.items() 
                         if k != 'Average' and k != 'SurfFuture' and not pd.isna(v)]
        if exchange_avgs:
            row_data['Non-Surf Avg'] = sum(exchange_avgs) / len(exchange_avgs)
        else:
            row_data['Non-Surf Avg'] = float('nan')
            
        all_averages_data.append(row_data)
    
    # Create DataFrame and sort by token name
    all_averages_df = pd.DataFrame(all_averages_data)
    all_averages_df = all_averages_df.sort_values(by='Token')
    
    # Ensure all exchange columns are present, fill with NaN if missing
    exchange_cols = [exchanges_display[ex] for ex in exchanges if exchanges_display[ex] != 'SurfFuture']
    exchange_cols.append('Non-Surf Avg')
    
    for col in exchange_cols:
        if col not in all_averages_df.columns:
            all_averages_df[col] = float('nan')
    
    # Reorder columns to put Token first, then exchanges
    cols_order = ['Token'] + exchange_cols
    all_averages_df = all_averages_df[cols_order]
    
    # Style the summary table with special colors for the daily average
    def color_daily_avg(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val < 0.1:  # Excellent
            return 'background-color: rgba(75, 0, 130, 0.7); color: white'  # Purple
        elif val < 0.5:  # Very good
            return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
        elif val < 1.0:  # Good
            return 'background-color: rgba(0, 128, 128, 0.7); color: white'  # Teal
        elif val < 2.0:  # Average
            return 'background-color: rgba(210, 105, 30, 0.7); color: white'  # Brown
        else:  # Poor
            return 'background-color: rgba(128, 0, 0, 0.7); color: white'  # Maroon
    
    # Format numbers and apply styling
    formatted_averages = all_averages_df.copy()
    for col in exchange_cols:
        formatted_averages[col] = formatted_averages[col].apply(
            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
        )
    
    styled_averages = all_averages_df.style.applymap(
        color_daily_avg, 
        subset=exchange_cols
    )
    
    # Display the summary table
    st.dataframe(styled_averages, height=min(600, 80 + 35 * len(all_averages_df)), use_container_width=True)
    
    # Calculate and display the best exchange based on average spreads
    exchange_overall_avgs = {}
    for col in exchange_cols:
        if col != 'Non-Surf Avg':
            avg_value = all_averages_df[col].mean()
            if not pd.isna(avg_value):
                exchange_overall_avgs[col] = avg_value
    
    if exchange_overall_avgs:
        best_exchange = min(exchange_overall_avgs.items(), key=lambda x: x[1])
        st.info(f"Best exchange overall based on average spreads: **{best_exchange[0]}** (Average spread: {best_exchange[1]:.4f})")

    # Add some explanation
    st.markdown("""
    This table shows the daily average spread for each token across different exchanges. 
    The 'Non-Surf Avg' column shows the average spread excluding SurfFuture.
    
    **Color coding**:
    - **Purple** (< 0.1): Excellent
    - **Blue** (0.1-0.5): Very good
    - **Teal** (0.5-1.0): Good
    - **Brown** (1.0-2.0): Average
    - **Maroon** (> 2.0): Poor
    """)
    
    st.markdown("---")  # Add a separator before individual token tables
    st.markdown("## Detailed Analysis by Token")
    st.markdown("Each token has its own table showing spreads at 10-minute intervals.")

# Then continue with the original code for individual token tables...