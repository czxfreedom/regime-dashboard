# In your tab2 section of the code, modify the beta table display section:

# TAB 2: BETA ANALYSIS
with tab2:
    st.header("Beta Analysis")
    st.write(f"Analyzing how much each token moves per 1% move in {reference_token} (accounting for correlation)")
    
    if token_results and reference_token in token_results:
        reference_returns = token_results[reference_token]['returns']
        
        # Debug - Show reference token returns
        if enable_debug:
            st.write(f"Reference token ({reference_token}) returns stats:")
            st.write({
                'min': reference_returns.min(),
                'max': reference_returns.max(),
                'mean': reference_returns.mean(),
                'std': reference_returns.std(),
                'zero_returns_pct': (abs(reference_returns) < epsilon).mean() * 100
            })
        
        # Create table data for betas
        beta_table_data = {}
        beta_values = {}
        
        for token, df in token_results.items():
            if token != reference_token:  # Skip reference token as we're comparing others to it
                token_returns = df['returns']
                
                # Debug - Show token returns
                if enable_debug and token == next(iter([t for t in token_results if t != reference_token]), None):
                    st.write(f"Sample returns for {token}:")
                    st.write({
                        'min': token_returns.min(),
                        'max': token_returns.max(),
                        'mean': token_returns.mean(),
                        'std': token_returns.std(),
                        'zero_returns_pct': (abs(token_returns) < epsilon).mean() * 100
                    })
                
                # Filter out extreme return values for more stable calculations
                max_return = 50  # Cap at 50% for single period return
                filtered_token_returns = token_returns.clip(lower=-max_return, upper=max_return)
                filtered_ref_returns = reference_returns.clip(lower=-max_return, upper=max_return)
                
                # Calculate overall beta for the entire period
                overall_beta, overall_alpha, overall_r_squared = calculate_beta(
                    filtered_token_returns, filtered_ref_returns, min_data_points=5
                )
                
                # Calculate correlation
                clean_data = pd.concat([filtered_token_returns, filtered_ref_returns], axis=1).dropna()
                overall_corr = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1]) if len(clean_data) > 1 else np.nan
                
                # Calculate rolling betas for each time period
                beta_by_time = {}
                
                # Group data by time label (hour:minute)
                time_groups = df.groupby('time_label')
                
                for time_label, group in time_groups:
                    # Get reference data for the same time period
                    ref_group = token_results[reference_token][token_results[reference_token]['time_label'] == time_label]
                    
                    if not group.empty and not ref_group.empty:
                        # Get returns and filter extremes
                        period_token_returns = group['returns'].clip(lower=-max_return, upper=max_return)
                        period_ref_returns = ref_group['returns'].clip(lower=-max_return, upper=max_return)
                        
                        # Calculate beta for this time period
                        period_beta, _, _ = calculate_beta(
                            period_token_returns, period_ref_returns, min_data_points=3
                        )
                        beta_by_time[time_label] = period_beta
                
                # Convert to series
                beta_series = pd.Series(beta_by_time)
                beta_table_data[token] = beta_series
                
                # Store overall metrics
                beta_values[token] = {
                    'beta': overall_beta,
                    'alpha': overall_alpha,
                    'r_squared': overall_r_squared,
                    'correlation': overall_corr
                }
        
        # Create DataFrame with all token betas
        beta_table = pd.DataFrame(beta_table_data)
        
        # Debug - Show raw beta table
        if enable_debug:
            st.write("Raw beta table:")
            st.dataframe(beta_table)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(beta_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # Reindex with the ordered times if they exist
        if ordered_times:
            beta_table = beta_table.reindex(ordered_times)
        
        # Function to color cells based on beta value
        def color_beta_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0:  # Negative beta (moves opposite to reference)
                return 'background-color: rgba(255, 0, 255, 0.7); color: white'  # Purple
            elif val < 0.5:  # Low beta
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Moderate beta
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to reference
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # High beta
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Very high beta
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        # CRITICAL FIX: Clean the beta table before styling by replacing inf, -inf with NaN
        # and clipping any extremely high/low values
        beta_table_clean = beta_table.replace([np.inf, -np.inf], np.nan)
        
        # Optional: Clip extreme values
        max_beta_display = 10  # Limit to +/- 10 for display purposes
        beta_table_clean = beta_table_clean.clip(lower=-max_beta_display, upper=max_beta_display)
        
        # Now style the cleaned table
        try:
            styled_beta_table = beta_table_clean.style.applymap(color_beta_cells).format("{:.4f}")
            st.markdown(f"## Beta Coefficient Table ({timeframe} intervals, Last {lookback_days} day(s), Singapore Time)")
            st.markdown(f"### Reference Token: {reference_token}")
            st.markdown("### Color Legend: <span style='color:purple'>Negative Beta</span>, <span style='color:blue'>Low Beta</span>, <span style='color:lightblue'>Moderate Beta</span>, <span style='color:black'>Similar to Reference</span>, <span style='color:orange'>High Beta</span>, <span style='color:red'>Very High Beta</span>", unsafe_allow_html=True)
            st.markdown(f"Values shown as Beta coefficient (how much token moves per 1% move in {reference_token})")
            st.dataframe(styled_beta_table, height=700, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying styled beta table: {str(e)}")
            st.write("Displaying unstyled beta table instead:")
            st.dataframe(beta_table_clean, height=700, use_container_width=True)
            
        # Rest of your tab2 code continues below
        # Create ranking table based on overall beta
        # ...