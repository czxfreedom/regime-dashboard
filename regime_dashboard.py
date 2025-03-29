# --- After fetching data ---
# Validate the data
data_diagnostics = validate_price_data(df)

# Display diagnostics in an expander
with st.expander("Data Diagnostics"):
    st.json(data_diagnostics)
    
    # Plot histogram of price changes if we have valid data
    if 'change_stats' in data_diagnostics:
        pct_changes = df['final_price'].pct_change().dropna() * 100
        
        fig_hist = px.histogram(
            pct_changes, 
            nbins=50,
            title="Distribution of Price Changes (%)",
            labels={'value': 'Price Change (%)'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- Update Hurst calculation ---
# Replace the existing Hurst calculation with:
ohlc['Hurst'] = ohlc['close'].rolling(rolling_window).apply(universal_hurst)

# --- Add additional visualizations to debug calculation issues ---
st.subheader("Calculation Diagnostics")
hurst_valid = ohlc['Hurst'].notna().sum()
hurst_invalid = ohlc['Hurst'].isna().sum()
hurst_validity_pct = hurst_valid / (hurst_valid + hurst_invalid) * 100 if (hurst_valid + hurst_invalid) > 0 else 0

st.metric("Valid Hurst Values", f"{hurst_valid} ({hurst_validity_pct:.1f}%)")

if hurst_valid == 0:
    st.error("No valid Hurst values calculated. Performing deep diagnostics...")
    
    # Sample calculation on a fixed window
    sample_size = min(rolling_window, len(ohlc))
    if sample_size > 10:
        # Get sample window
        sample_window = ohlc['close'].iloc[:sample_size].values
        
        st.write(f"Attempting manual Hurst calculation on first {sample_size} values...")
        try:
            manual_hurst = universal_hurst(sample_window)
            st.write(f"Manual Hurst result: {manual_hurst}")
            
            # Show the sample data
            st.write("Sample price data:")
            sample_df = pd.DataFrame({
                'index': range(len(sample_window)),
                'price': sample_window
            })
            st.dataframe(sample_df)
            
            # Plot the sample data
            fig = px.line(sample_df, x='index', y='price', title='Sample Price Window')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate returns
            returns = np.diff(np.log(sample_window))
            returns_df = pd.DataFrame({
                'index': range(len(returns)),
                'log_return': returns
            })
            
            st.write("Log returns from sample:")
            st.dataframe(returns_df)
            
            # Plot returns
            fig = px.line(returns_df, x='index', y='log_return', title='Log Returns')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate autocorrelation
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                st.write(f"Lag-1 Autocorrelation: {autocorr:.4f}")
        except Exception as e:
            st.error(f"Error in manual calculation: {str(e)}")