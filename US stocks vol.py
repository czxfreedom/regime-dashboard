import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math

# Set page configuration
st.set_page_config(page_title="US Stocks Volatility Dashboard", layout="wide")

# App title and description
st.title("US Stocks Volatility Dashboard")
st.markdown("""
This dashboard analyzes realized volatility for major US stocks across different timeframes.
* **Tab 1**: 5-minute sampling volatility averaged over 30 minutes (last 24 hours)
* **Tab 2**: 30-minute sampling volatility averaged over 1 day (last 7 days)
* **Tab 3**: Trading strategies and volatility analytics with 2-year backtest
""")

# Sidebar for stock selection
st.sidebar.header("Settings")
ticker_options = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD"]
selected_ticker = st.sidebar.selectbox("Select stock:", ticker_options)

# Function to calculate realized volatility
def calculate_realized_volatility(returns, window, sampling_rate):
    """
    Calculate realized volatility from returns
    window: in sampling rate units
    sampling_rate: in minutes
    """
    # Annualization factor: sqrt(minutes per year / sampling_rate)
    annualization_factor = math.sqrt(525600 / sampling_rate)
    
    # Calculate rolling standard deviation of returns
    realized_vol = returns.rolling(window=window).std() * annualization_factor * 100  # Convert to percentage
    
    return realized_vol

# Function to fetch and prepare data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        st.error(f"No data available for {ticker} with interval {interval} for period {period}.")
        return None
    
    # Calculate returns
    data['returns'] = data['Close'].pct_change()
    
    return data

# Function to compute backtest metrics including profit factor
def compute_backtest_metrics(returns):
    """
    Compute comprehensive backtest metrics
    """
    if len(returns) == 0:
        return {}
    
    # Total return
    total_return = (1 + returns.fillna(0)).cumprod().iloc[-1] - 1
    
    # Separate winning and losing trades
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    
    # Profit factor (sum of winners / sum of losers in absolute terms)
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
    
    # Win rate
    win_rate = len(winning_trades) / len(returns.dropna()) if len(returns.dropna()) > 0 else 0
    
    # Annualized return
    days = (returns.index[-1] - returns.index[0]).days
    annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
    
    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    risk_free_rate = 0.03  # Assuming 3% risk-free rate
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
    # Maximum drawdown
    wealth_index = (1 + returns.fillna(0)).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdown.min()
    
    # Average gain/loss
    avg_gain = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    # Gain-to-loss ratio
    gain_to_loss = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')
    
    # Monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    positive_months = len(monthly_returns[monthly_returns > 0])
    total_months = len(monthly_returns)
    pct_positive_months = positive_months / total_months if total_months > 0 else 0
    
    # Calculate Calmar ratio (annualized return / max drawdown)
    calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate Sortino ratio (focusing on downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
    sortino = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    # Consecutive wins/losses
    returns_sign = np.sign(returns.fillna(0))
    pos_streak = (returns_sign * (returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).cumcount() + 1))
    max_consecutive_wins = pos_streak[pos_streak > 0].max() if len(pos_streak[pos_streak > 0]) > 0 else 0
    max_consecutive_losses = abs(pos_streak[pos_streak < 0].min()) if len(pos_streak[pos_streak < 0]) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'gain_to_loss_ratio': gain_to_loss,
        'percent_profitable_months': pct_positive_months,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }

# Create tabs
tab1, tab2, tab3 = st.tabs(["5-Min Vol (24h)", "30-Min Vol (7d)", "Trading Strategies"])

# Tab 1: 5-minute sampling volatility averaged over 30 minutes (last 24 hours)
with tab1:
    st.header(f"5-Minute Sampling Volatility for {selected_ticker}")
    st.subheader("Averaged over 30 minutes (Last 24 hours)")
    
    # Get 5-minute data for the last 2 days (to ensure we have a full 24 hours)
    data_5min = get_stock_data(selected_ticker, "2d", "5m")
    
    if data_5min is not None:
        # Keep only last 24 hours of data
        last_24h = datetime.now() - timedelta(days=1)
        data_5min = data_5min[data_5min.index > last_24h]
        
        if len(data_5min) > 0:
            # Calculate 5-min realized volatility (6 periods = 30 minutes)
            data_5min['realized_vol'] = calculate_realized_volatility(data_5min['returns'], window=6, sampling_rate=5)
            
            # Create plots
            fig1 = go.Figure()
            
            # Add price line
            fig1.add_trace(go.Scatter(
                x=data_5min.index,
                y=data_5min['Close'],
                name='Price',
                line=dict(color='blue')
            ))
            
            # Create a secondary y-axis for volatility
            fig1.update_layout(
                yaxis=dict(title='Price'),
                yaxis2=dict(title='Volatility (%)', overlaying='y', side='right')
            )
            
            # Add volatility line
            fig1.add_trace(go.Scatter(
                x=data_5min.index,
                y=data_5min['realized_vol'],
                name='30-min Avg Volatility',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig1.update_layout(
                title='Price vs. Realized Volatility (5-min sampling)',
                height=600,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                current_vol = data_5min['realized_vol'].dropna().iloc[-1]
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            
            with col2:
                avg_vol = data_5min['realized_vol'].dropna().mean()
                st.metric("Average 24h Volatility", f"{avg_vol:.2f}%")
            
            with col3:
                max_vol = data_5min['realized_vol'].dropna().max()
                st.metric("Max 24h Volatility", f"{max_vol:.2f}%")
            
            # Hourly breakdown table
            st.subheader("Hourly Volatility Breakdown")
            data_5min['hour'] = data_5min.index.hour
            hourly_vol = data_5min.groupby('hour')['realized_vol'].mean().reset_index()
            hourly_vol.columns = ['Hour', 'Average Volatility (%)']
            hourly_vol['Hour'] = hourly_vol['Hour'].apply(lambda x: f"{x:02d}:00")
            hourly_vol['Average Volatility (%)'] = hourly_vol['Average Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(hourly_vol, use_container_width=True)
        else:
            st.warning("Not enough 5-minute data available for the last 24 hours.")
    else:
        st.error("Failed to fetch 5-minute data for selected ticker.")

# Tab 2: 30-minute sampling volatility averaged over 1 day (last 7 days)
with tab2:
    st.header(f"30-Minute Sampling Volatility for {selected_ticker}")
    st.subheader("Averaged over 1 day (Last 7 days)")
    
    # Get 30-minute data for the last 7 days
    data_30min = get_stock_data(selected_ticker, "10d", "30m")
    
    if data_30min is not None:
        # Keep only last 7 days of data
        last_7d = datetime.now() - timedelta(days=7)
        data_30min = data_30min[data_30min.index > last_7d]
        
        if len(data_30min) > 0:
            # Calculate 30-min realized volatility (48 periods = 1 day (24 hours))
            data_30min['realized_vol'] = calculate_realized_volatility(data_30min['returns'], window=16, sampling_rate=30)
            
            # Create plots
            fig2 = go.Figure()
            
            # Add price line
            fig2.add_trace(go.Scatter(
                x=data_30min.index,
                y=data_30min['Close'],
                name='Price',
                line=dict(color='blue')
            ))
            
            # Create a secondary y-axis for volatility
            fig2.update_layout(
                yaxis=dict(title='Price'),
                yaxis2=dict(title='Volatility (%)', overlaying='y', side='right')
            )
            
            # Add volatility line
            fig2.add_trace(go.Scatter(
                x=data_30min.index,
                y=data_30min['realized_vol'],
                name='1-day Avg Volatility',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig2.update_layout(
                title='Price vs. Realized Volatility (30-min sampling)',
                height=600,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                current_vol = data_30min['realized_vol'].dropna().iloc[-1]
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            
            with col2:
                avg_vol = data_30min['realized_vol'].dropna().mean()
                st.metric("Average 7d Volatility", f"{avg_vol:.2f}%")
            
            with col3:
                max_vol = data_30min['realized_vol'].dropna().max()
                st.metric("Max 7d Volatility", f"{max_vol:.2f}%")
            
            # Daily breakdown table
            st.subheader("Daily Volatility Breakdown")
            data_30min['date'] = data_30min.index.date
            daily_vol = data_30min.groupby('date')['realized_vol'].mean().reset_index()
            daily_vol.columns = ['Date', 'Average Volatility (%)']
            daily_vol['Average Volatility (%)'] = daily_vol['Average Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(daily_vol, use_container_width=True)
            
            # Volatility distribution
            st.subheader("Volatility Distribution")
            fig_hist = px.histogram(
                data_30min.dropna(), 
                x='realized_vol',
                nbins=20,
                labels={'realized_vol': 'Realized Volatility (%)'},
                title='Distribution of 30-Min Volatility over Last 7 Days'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Not enough 30-minute data available for the last 7 days.")
    else:
        st.error("Failed to fetch 30-minute data for selected ticker.")

# Tab 3: Trading strategies and analytics
with tab3:
    st.header("Volatility-Based Trading Strategies")
    
    # Fetch daily data for a 2-year historical period for thorough backtest
    historical_data = get_stock_data(selected_ticker, "2y", "1d")
    
    if historical_data is not None:
        # Calculate different volatility measures
        historical_data['10d_vol'] = calculate_realized_volatility(historical_data['returns'], window=10, sampling_rate=1440)
        historical_data['20d_vol'] = calculate_realized_volatility(historical_data['returns'], window=20, sampling_rate=1440)
        historical_data['vol_ratio'] = historical_data['10d_vol'] / historical_data['20d_vol']
        
        # Add some additional volatility metrics for more sophisticated strategies
        historical_data['5d_vol'] = calculate_realized_volatility(historical_data['returns'], window=5, sampling_rate=1440)
        historical_data['30d_vol'] = calculate_realized_volatility(historical_data['returns'], window=30, sampling_rate=1440)
        historical_data['60d_vol'] = calculate_realized_volatility(historical_data['returns'], window=60, sampling_rate=1440)
        historical_data['vol_zscore'] = (historical_data['10d_vol'] - historical_data['10d_vol'].rolling(60).mean()) / historical_data['10d_vol'].rolling(60).std()
        
        st.subheader("Volatility Term Structure")
        
        # Volatility term structure visualization
        fig_term = go.Figure()
        
        fig_term.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['10d_vol'],
            name='10-Day Volatility',
            line=dict(color='blue')
        ))
        
        fig_term.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['20d_vol'],
            name='20-Day Volatility',
            line=dict(color='green')
        ))
        
        fig_term.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['vol_ratio'] * 10,  # Scale for visibility
            name='Vol Ratio (10d/20d) x10',
            line=dict(color='red', dash='dash'),
            yaxis='y2'
        ))
        
        fig_term.update_layout(
            title='Volatility Term Structure',
            height=500,
            yaxis=dict(title='Volatility (%)'),
            yaxis2=dict(title='Ratio (scaled)', overlaying='y', side='right'),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_term, use_container_width=True)
        
        # Strategy selector and backtest
        st.subheader("2-Year Backtested Volatility Trading Strategies")
        
        strategy_option = st.selectbox(
            "Select a volatility-based trading strategy:",
            ["Mean Reversion on High Volatility", "Volatility Breakout Momentum", "Volatility Regime Switching", 
             "Volatility Term Structure", "Volatility Range Breakout"]
        )
        
        # Create placeholder for all strategies' backtests
        all_strategies_results = {}
        
        # Strategy 1: Mean Reversion on High Volatility
        historical_data['vol_z_score'] = (historical_data['10d_vol'] - historical_data['10d_vol'].rolling(60).mean()) / historical_data['10d_vol'].rolling(60).std()
        historical_data['mean_reversion_signal'] = np.where(historical_data['vol_z_score'] > 1.5, -1, 0)
        historical_data['mean_reversion_signal'] = np.where(historical_data['vol_z_score'] < 0, 1, historical_data['mean_reversion_signal'])
        historical_data['mean_reversion_signal'] = historical_data['mean_reversion_signal'].shift(1)
        historical_data['mean_reversion_return'] = historical_data['mean_reversion_signal'] * historical_data['returns']
        all_strategies_results['Mean Reversion'] = compute_backtest_metrics(historical_data['mean_reversion_return'].dropna())
        
        # Strategy 2: Volatility Breakout Momentum
        historical_data['vol_change_pct'] = historical_data['10d_vol'].pct_change() * 100
        historical_data['price_direction'] = np.sign(historical_data['returns'])
        historical_data['breakout_signal'] = np.where(historical_data['vol_change_pct'] > 30, historical_data['price_direction'], 0)
        historical_data['breakout_signal'] = historical_data['breakout_signal'].shift(1)
        historical_data['breakout_return'] = historical_data['breakout_signal'] * historical_data['returns']
        all_strategies_results['Volatility Breakout'] = compute_backtest_metrics(historical_data['breakout_return'].dropna())
        
        # Strategy 3: Volatility Regime Switching
        low_vol = historical_data['10d_vol'].rolling(60).quantile(0.25)
        high_vol = historical_data['10d_vol'].rolling(60).quantile(0.75)
        
        historical_data['regime'] = np.where(historical_data['10d_vol'] <= low_vol, 'low', 
                                     np.where(historical_data['10d_vol'] >= high_vol, 'high', 'medium'))
        
        historical_data['regime_signal'] = np.where(historical_data['regime'] == 'low', 
                                     np.sign(historical_data['returns'].rolling(5).mean()), 
                                     np.where(historical_data['regime'] == 'high', 
                                             -np.sign(historical_data['returns']), 0))
        
        historical_data['regime_signal'] = historical_data['regime_signal'].shift(1)
        historical_data['regime_return'] = historical_data['regime_signal'] * historical_data['returns']
        all_strategies_results['Regime Switching'] = compute_backtest_metrics(historical_data['regime_return'].dropna())
        
        # Strategy 4: Volatility Term Structure
        historical_data['term_signal'] = np.where(historical_data['vol_ratio'] > 1.1, 1, 
                                        np.where(historical_data['vol_ratio'] < 0.9, -1, 0))
        historical_data['term_signal'] = historical_data['term_signal'].shift(1)
        historical_data['term_return'] = historical_data['term_signal'] * historical_data['returns']
        all_strategies_results['Term Structure'] = compute_backtest_metrics(historical_data['term_return'].dropna())
        
        # Strategy 5: Volatility Range Breakout
        historical_data['vol_upper'] = historical_data['10d_vol'].rolling(20).mean() + 2 * historical_data['10d_vol'].rolling(20).std()
        historical_data['vol_lower'] = historical_data['10d_vol'].rolling(20).mean() - 2 * historical_data['10d_vol'].rolling(20).std()
        
        historical_data['range_signal'] = np.where(historical_data['10d_vol'] > historical_data['vol_upper'], -1,
                                         np.where(historical_data['10d_vol'] < historical_data['vol_lower'], 1, 0))
        historical_data['range_signal'] = historical_data['range_signal'].shift(1)
        historical_data['range_return'] = historical_data['range_signal'] * historical_data['returns']
        all_strategies_results['Range Breakout'] = compute_backtest_metrics(historical_data['range_return'].dropna())
        
        # Determine active strategy based on selection
        if strategy_option == "Mean Reversion on High Volatility":
            active_strategy = "mean_reversion"
            strategy_signal = historical_data['mean_reversion_signal']
            strategy_returns = historical_data['mean_reversion_return']
            strategy_metrics = all_strategies_results['Mean Reversion']
            
            # Show strategy description
            st.markdown("""
            ### Mean Reversion Strategy
            
            This strategy assumes that periods of high volatility tend to revert to the mean. It:
            - Goes short when volatility is significantly higher than its historical average (Z-score > 1.5)
            - Goes long when volatility returns to normal levels (Z-score < 0)
            - Remains neutral in between these thresholds
            
            The strategy is based on the tendency of volatility to cluster but eventually revert to its long-term average.
            """)
            
            # Strategy parameters
            vol_threshold = st.slider("Volatility Threshold (Z-score)", 1.0, 3.0, 1.5, 0.1)
            st.info(f"The strategy goes short when volatility Z-score exceeds {vol_threshold} and goes long when it falls below 0.")
            
        elif strategy_option == "Volatility Breakout Momentum":
            active_strategy = "breakout"
            strategy_signal = historical_data['breakout_signal']
            strategy_returns = historical_data['breakout_return']
            strategy_metrics = all_strategies_results['Volatility Breakout']
            
            st.markdown("""
            ### Volatility Breakout Strategy
            
            This strategy capitalizes on momentum following volatility breakouts:
            - When volatility increases significantly (>30% day-over-day), it follows the prevailing price direction
            - Takes no position when volatility is stable or declining
            
            The strategy is based on the observation that sharp volatility increases often signal the start of a strong directional move.
            """)
            
            # Parameters
            breakout_threshold = st.slider("Volatility Breakout Threshold (%)", 10, 100, 30, 5)
            st.info(f"The strategy takes a position in the direction of the price move when volatility increases by more than {breakout_threshold}%.")
            
        elif strategy_option == "Volatility Regime Switching":
            active_strategy = "regime"
            strategy_signal = historical_data['regime_signal']
            strategy_returns = historical_data['regime_return']
            strategy_metrics = all_strategies_results['Regime Switching']
            
            st.markdown("""
            ### Volatility Regime Switching Strategy
            
            This strategy adapts to different volatility regimes:
            - In low volatility regimes (bottom 25%), it follows trends (5-day momentum)
            - In high volatility regimes (top 25%), it takes counter-trend positions
            - In medium volatility regimes, it stays neutral
            
            The strategy is based on research showing that different trading approaches work better in different volatility environments.
            """)
            
            # Parameters
            low_vol_threshold = st.slider("Low Volatility Threshold (percentile)", 0, 50, 25, 5)
            high_vol_threshold = st.slider("High Volatility Threshold (percentile)", 50, 100, 75, 5)
            st.info(f"The strategy uses trend-following in low volatility (below {low_vol_threshold}th percentile) and counter-trend in high volatility (above {high_vol_threshold}th percentile).")
            
        elif strategy_option == "Volatility Term Structure":
            active_strategy = "term"
            strategy_signal = historical_data['term_signal']
            strategy_returns = historical_data['term_return']
            strategy_metrics = all_strategies_results['Term Structure']
            
            st.markdown("""
            ### Volatility Term Structure Strategy
            
            This strategy trades based on the relationship between short and long-term volatility:
            - Goes long when short-term volatility (10-day) is significantly higher than long-term volatility (20-day)
            - Goes short when short-term volatility is significantly lower than long-term volatility
            - Stays neutral when the volatility term structure is flat
            
            The strategy is based on the forward-looking expectations implied by the volatility term structure.
            """)
            
            # Parameters
            term_high = st.slider("Upper Term Structure Threshold", 1.0, 1.5, 1.1, 0.05)
            term_low = st.slider("Lower Term Structure Threshold", 0.5, 1.0, 0.9, 0.05)
            st.info(f"The strategy goes long when 10d/20d vol ratio > {term_high} and short when 10d/20d vol ratio < {term_low}.")
            
        elif strategy_option == "Volatility Range Breakout":
            active_strategy = "range"
            strategy_signal = historical_data['range_signal']
            strategy_returns = historical_data['range_return']
            strategy_metrics = all_strategies_results['Range Breakout']
            
            st.markdown("""
            ### Volatility Range Breakout Strategy
            
            This strategy trades based on significant deviations from the recent volatility range:
            - Goes short when volatility breaks above its upper range (mean + 2 std)
            - Goes long when volatility breaks below its lower range (mean - 2 std)
            - Stays neutral when volatility is within its normal range
            
            The strategy is based on statistical mean reversion principles applied to volatility.
            """)
            
            # Parameters
            std_range = st.slider("Standard Deviation Range", 1.0, 3.0, 2.0, 0.1)
            st.info(f"The strategy uses a range of mean ± {std_range} standard deviations to determine breakouts.")
        
        # Calculate buy & hold returns for comparison
        historical_data['buy_hold_cumret'] = (1 + historical_data['returns'].fillna(0)).cumprod() - 1
        historical_data[f'{active_strategy}_cumret'] = (1 + historical_data[f'{active_strategy}_return'].fillna(0)).cumprod() - 1
        
        # Compare strategy performance with buy & hold
        st.subheader("Strategy Performance")
        
        # Plot cumulative returns
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[f'{active_strategy}_cumret'] * 100,
            name='Strategy Returns (%)',
            line=dict(color='green')
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['buy_hold_cumret'] * 100,
            name='Buy & Hold Returns (%)',
            line=dict(color='gray')
        ))
        
        fig_perf.update_layout(
            title='Strategy Performance vs. Buy & Hold (2-Year Backtest)',
            height=500,
            yaxis=dict(title='Cumulative Return (%)'),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Display key performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{strategy_metrics['total_return']*100:.2f}%")
            st.metric("Annualized Return", f"{strategy_metrics['annualized_return']*100:.2f}%")
            st.metric("Win Rate", f"{strategy_metrics['win_rate']*100:.2f}%")
            st.metric("Max Drawdown", f"{strategy_metrics['max_drawdown']*100:.2f}%")
            
        with col2:
            st.metric("Profit Factor", f"{strategy_metrics['profit_factor']:.2f}")
            st.metric("Sharpe Ratio", f"{strategy_metrics['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{strategy_metrics['sortino_ratio']:.2f}")
            st.metric("Calmar Ratio", f"{strategy_metrics['calmar_ratio']:.2f}")
            
        with col3:
            st.metric("Annualized Volatility", f"{strategy_metrics['annualized_volatility']*100:.2f}%")
            st.metric("Gain/Loss Ratio", f"{strategy_metrics['gain_to_loss_ratio']:.2f}")
            st.metric("% Profitable Months", f"{strategy_metrics['percent_profitable_months']*100:.2f}%")
            st.metric("Max Consecutive Wins", f"{int(strategy_metrics['max_consecutive_wins'])}")
        
        # Strategy comparison - display a comprehensive backtest comparison
        st.subheader("Strategy Comparison")
        
        # Prepare comparison dataframe
        comparison_metrics = ['total_return', 'annualized_return', 'profit_factor', 'win_rate', 'max_drawdown', 'sharpe_ratio']
        comparison_labels = ['Total Return', 'Ann. Return', 'Profit Factor', 'Win Rate', 'Max Drawdown', 'Sharpe Ratio']
        
        comparison_data = {}
        for strategy_name, metrics in all_strategies_results.items():
            comparison_data[strategy_name] = [
                f"{metrics['total_return']*100:.2f}%",
                f"{metrics['annualized_return']*100:.2f}%",
                f"{metrics['profit_factor']:.2f}",
                f"{metrics['win_rate']*100:.2f}%",
                f"{metrics['max_drawdown']*100:.2f}%",
                f"{metrics['sharpe_ratio']:.2f}"
            ]
        
        comparison_df = pd.DataFrame(comparison_data, index=comparison_labels)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show strategy equity curve for all strategies
        st.subheader("Equity Curves for All Strategies")
        
        # Calculate equity curves for all strategies
        equity_fig = go.Figure()
        
        for strategy, returns_col in {
            'Mean Reversion': 'mean_reversion_return',
            'Volatility Breakout': 'breakout_return',
            'Regime Switching': 'regime_return',
            'Term Structure': 'term_return',
            'Range Breakout': 'range_return'
        }.items():
            equity = (1 + historical_data[returns_col].fillna(0)).cumprod() - 1
            equity_fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=equity * 100,
                name=f"{strategy} ({all_strategies_results[strategy]['profit_factor']:.2f} PF)",
                mode='lines'
            ))
        
        # Add buy & hold for reference
        equity_fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['buy_hold_cumret'] * 100,
            name='Buy & Hold',
            line=dict(color='gray')
        ))
        
        equity_fig.update_layout(
            title='Equity Curves Comparison (2-Year Backtest)',
            height=600,
            yaxis=dict(title='Cumulative Return (%)'),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(equity_fig, use_container_width=True)
        
        # Drawdown analysis
        st.subheader("Drawdown Analysis")
        
        def calculate_drawdown(returns):
            wealth_index = (1 + returns.fillna(0)).cumprod()
            previous_peaks = wealth_index.cummax()
            drawdown = (wealth_index - previous_peaks) / previous_peaks
            return drawdown
        
        historical_data[f'{active_strategy}_dd'] = calculate_drawdown(historical_data[f'{active_strategy}_return'])
        historical_data['buy_hold_dd'] = calculate_drawdown(historical_data['returns'])
        
        fig_dd = go.Figure()
        
        fig_dd.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[f'{active_strategy}_dd'] * 100,
            name='Strategy Drawdown',
            line=dict(color='red')
        ))
        
        fig_dd.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['buy_hold_dd'] * 100,
            name='Buy & Hold Drawdown',
            line=dict(color='gray')
        ))
        
        fig_dd.update_layout(
            title='Drawdown Comparison',
            height=400,
            yaxis=dict(title='Drawdown (%)', autorange="reversed"),  # Reversed y-axis for drawdown
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Monthly returns heatmap
        st.subheader("Monthly Returns Heatmap")
        
        # Resample to monthly returns
        monthly_returns = historical_data[f'{active_strategy}_return'].fillna(0).resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = pd.DataFrame(monthly_returns)
        monthly_returns_df.index = monthly_returns_df.index.to_period('M')
        
        # Pivot table for heatmap
        monthly_returns_pivot = monthly_returns_df.copy()
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        monthly_returns_pivot = monthly_returns_pivot.pivot_table(
            index='year',
            columns='month',
            values=f'{active_strategy}_return'
        )
        
        # Get month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot.columns = [month_names[i-1] for i in monthly_returns_pivot.columns]
        
        # Format values as percentages
        formatted_pivot = monthly_returns_pivot.copy()
        for col in formatted_pivot.columns:
            formatted_pivot[col] = formatted_pivot[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
        
        st.dataframe(formatted_pivot, use_container_width=True)
        
        # Strategy insights
        st.subheader("Strategy Insights and Recommendations")
        
        # Determine the best strategy
        best_strategy = max(all_strategies_results.items(), key=lambda x: x[1]['profit_factor'])[0]
        best_profit_factor = max(all_strategies_results.items(), key=lambda x: x[1]['profit_factor'])[1]['profit_factor']
        best_sharpe = max(all_strategies_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        
        st.markdown(f"""
        ### Key Insights:
        
        * **Best Overall Strategy**: {best_strategy} with a profit factor of {best_profit_factor:.2f}
        * **Best Risk-Adjusted Strategy**: {best_sharpe} with the highest Sharpe ratio
        * **Current Strategy Performance**: The {strategy_option.split()[0]} strategy has a profit factor of {strategy_metrics['profit_factor']:.2f} and Sharpe ratio of {strategy_metrics['sharpe_ratio']:.2f}
        
        ### Strategy Recommendations:
        
        1. **Optimal Allocation**: Based on the backtests, consider allocating:
           - 40% to {best_strategy} Strategy
           - 30% to {best_sharpe} Strategy  
           - 30% to a complementary strategy with low correlation to these two
        
        2. **Position Sizing**: Scale position sizes inversely with current volatility level:
           - Current 10-day volatility: {historical_data['10d_vol'].iloc[-1]:.2f}%
           - Long-term average volatility: {historical_data['10d_vol'].rolling(252).mean().iloc[-1]:.2f}%
           - Suggested position size scaling: {(historical_data['10d_vol'].rolling(252).mean().iloc[-1] / historical_data['10d_vol'].iloc[-1]):.2f}x baseline
        
        3. **Risk Management**:
           - Set stop-loss at {max(1.5, historical_data['10d_vol'].iloc[-1] / 10):.1f}x daily ATR
           - Use dynamic trailing stops of {max(2.0, historical_data['10d_vol'].iloc[-1] / 8):.1f}x daily ATR on profitable trades
           - Implement strict position limits of 5% maximum portfolio allocation per trade
        """)
        
        # Volatility Analytics
        st.header("Advanced Volatility Analytics")
        
        # Volatility seasonality
        st.subheader("Volatility Seasonality")
        
        # Calculate daily volatility for each day of the week
        historical_data['day_of_week'] = historical_data.index.dayofweek
        day_of_week_vol = historical_data.groupby('day_of_week')['10d_vol'].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_of_week_vol.index = [day_names[i] for i in day_of_week_vol.index]
        
        # Plot day of week volatility
        fig_dow = px.bar(
            day_of_week_vol,
            labels={'index': 'Day of Week', 'value': 'Average Volatility (%)'},
            title='Average Volatility by Day of Week'
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        
        # Get month-of-year seasonality
        historical_data['month'] = historical_data.index.month
        month_vol = historical_data.groupby('month')['10d_vol'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_vol.index = [month_names[i-1] for i in month_vol.index]
        
        fig_month = px.bar(
            month_vol,
            labels={'index': 'Month', 'value': 'Average Volatility (%)'},
            title='Average Volatility by Month'
        )
        st.plotly_chart(fig_month, use_container_width=True)
        
        # Volatility regimes detection
        st.subheader("Volatility Regime Detection")
        
        # Implement a simple Hidden Markov Model (emulated with thresholds for simplicity)
        low_percentile = historical_data['10d_vol'].rolling(252).quantile(0.3)
        high_percentile = historical_data['10d_vol'].rolling(252).quantile(0.7)
        
        # Determine regimes
        historical_data['vol_regime'] = np.where(historical_data['10d_vol'] <= low_percentile, 'Low', 
                                        np.where(historical_data['10d_vol'] >= high_percentile, 'High', 'Normal'))
        
        # Calculate regime duration and transitions
        historical_data['regime_change'] = historical_data['vol_regime'] != historical_data['vol_regime'].shift(1)
        regime_changes = historical_data[historical_data['regime_change']].copy()
        
        # Calculate duration of each regime
        regime_durations = []
        current_regime = None
        regime_start = None
        
        for date, row in historical_data.iterrows():
            if row['vol_regime'] != current_regime:
                if current_regime is not None:
                    duration = (date - regime_start).days
                    regime_durations.append({
                        'regime': current_regime,
                        'start': regime_start,
                        'end': date,
                        'duration_days': duration
                    })
                current_regime = row['vol_regime']
                regime_start = date
        
        # Add the last regime if it exists
        if current_regime is not None and regime_start is not None:
            duration = (historical_data.index[-1] - regime_start).days
            regime_durations.append({
                'regime': current_regime,
                'start': regime_start,
                'end': historical_data.index[-1],
                'duration_days': duration
            })
        
        # Create a dataframe of regime durations
        if regime_durations:
            regime_df = pd.DataFrame(regime_durations)
            
            # Calculate average duration by regime
            avg_duration = regime_df.groupby('regime')['duration_days'].mean()
            
            st.write("Average Duration of Volatility Regimes (days):")
            st.dataframe(avg_duration)
            
            # Show current regime
            current_regime = historical_data['vol_regime'].iloc[-1]
            days_in_current = (historical_data.index[-1] - regime_durations[-1]['start']).days
            avg_duration_current = avg_duration[current_regime]
            
            st.info(f"Current Volatility Regime: **{current_regime}** (Duration: {days_in_current} days, Average: {avg_duration_current:.1f} days)")
            
            # Create a regime timeline
            fig_regime = go.Figure()
            
            # Add volatility line
            fig_regime.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['10d_vol'],
                name='10-Day Volatility',
                line=dict(color='blue')
            ))
            
            # Add regime backgrounds
            colors = {'Low': 'rgba(0, 255, 0, 0.1)', 'Normal': 'rgba(255, 255, 0, 0.1)', 'High': 'rgba(255, 0, 0, 0.1)'}
            
            for regime in regime_durations:
                fig_regime.add_vrect(
                    x0=regime['start'],
                    x1=regime['end'],
                    fillcolor=colors[regime['regime']],
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
            
            # Add regime labels at midpoints
            for regime in regime_durations:
                if (regime['end'] - regime['start']).days > 30:  # Only add labels for longer regimes
                    midpoint = regime['start'] + (regime['end'] - regime['start']) / 2
                    fig_regime.add_annotation(
                        x=midpoint,
                        y=historical_data['10d_vol'].max() * 0.95,
                        text=regime['regime'],
                        showarrow=False,
                        font=dict(color="black", size=14)
                    )
            
            fig_regime.update_layout(
                title='Volatility Regimes Timeline',
                height=500,
                yaxis=dict(title='Volatility (%)'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_regime, use_container_width=True)
            
            # Regime transition probabilities
            st.subheader("Regime Transition Probabilities")
            
            # Calculate transition matrix
            transitions = historical_data['vol_regime'].shift(-1).dropna()
            transition_matrix = pd.crosstab(
                historical_data['vol_regime'].iloc[:-1], 
                transitions,
                normalize='index'
            )
            
            # Display transition probabilities
            st.write("Probability of transitioning from current regime (rows) to next regime (columns):")
            transition_formatted = transition_matrix.copy()
            for col in transition_formatted.columns:
                transition_formatted[col] = transition_formatted[col].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(transition_formatted)
            
            # Trading recommendations based on current regime
            st.subheader("Regime-Based Trading Recommendations")
            
            if current_regime == 'Low':
                st.markdown("""
                ### Low Volatility Regime Recommendations:
                
                1. **Strategy Approach**:
                   - Focus on trend-following strategies
                   - Look for breakouts with volume confirmation
                   - Consider longer holding periods
                
                2. **Position Sizing**:
                   - Increase position sizes by 20-30%
                   - Spread capital across more positions
                
                3. **Options Strategies**:
                   - Sell options premium (credit spreads, iron condors)
                   - Consider calendar spreads
                   - Target lower delta positions (0.20-0.30)
                
                4. **Risk Management**:
                   - Widen stop-loss levels
                   - Use time-based exits
                   - Set profit targets at key resistance levels
                """)
            
            elif current_regime == 'Normal':
                st.markdown("""
                ### Normal Volatility Regime Recommendations:
                
                1. **Strategy Approach**:
                   - Balance between trend and counter-trend strategies
                   - Focus on sectors showing relative strength
                   - Use standard technical analysis setups
                
                2. **Position Sizing**:
                   - Standard position sizing (1-2% risk per trade)
                   - Moderate diversification
                
                3. **Options Strategies**:
                   - Balance between debit and credit spreads
                   - Vertical spreads offer good risk/reward
                   - Target ATM to slight OTM options
                
                4. **Risk Management**:
                   - Standard 1-2 ATR stop-losses
                   - Trailing stops on profitable trades
                   - Partial profit taking at targets
                """)
            
            else:  # High volatility
                st.markdown("""
                ### High Volatility Regime Recommendations:
                
                1. **Strategy Approach**:
                   - Focus on mean-reversion strategies
                   - Look for oversold/overbought conditions
                   - Reduce holding periods
                
                2. **Position Sizing**:
                   - Reduce position sizes by 30-50%
                   - Concentrate on fewer, high-conviction trades
                
                3. **Options Strategies**:
                   - Long options (puts, calls, straddles) to benefit from large moves
                   - Avoid naked short options positions
                   - Consider put spreads for protection
                
                4. **Risk Management**:
                   - Tighten stop-losses (0.5-1 ATR)
                   - Use time-based exits (1-3 days maximum)
                   - Take profits quickly
                """)
        
        # Additional analytics: Correlation between volatility and returns
        st.subheader("Volatility-Return Correlation Analysis")
        
        # Calculate correlation between returns and volatility
        returns_vs_vol = historical_data[['returns', '10d_vol']].copy()
        returns_vs_vol['abs_returns'] = returns_vs_vol['returns'].abs()
        returns_vs_vol['next_day_return'] = returns_vs_vol['returns'].shift(-1)
        returns_vs_vol['vol_change'] = returns_vs_vol['10d_vol'].pct_change()
        
        # Correlation coefficient
        corr_vol_ret = returns_vs_vol['10d_vol'].corr(returns_vs_vol['returns'])
        corr_vol_abs_ret = returns_vs_vol['10d_vol'].corr(returns_vs_vol['abs_returns'])
        corr_vol_next_ret = returns_vs_vol['10d_vol'].corr(returns_vs_vol['next_day_return'])
        corr_vol_change_ret = returns_vs_vol['vol_change'].corr(returns_vs_vol['returns'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Vol vs. Return Correlation", f"{corr_vol_ret:.3f}")
            st.metric("Vol vs. Absolute Return", f"{corr_vol_abs_ret:.3f}")
            
        with col2:
            st.metric("Vol vs. Next Day Return", f"{corr_vol_next_ret:.3f}")
            st.metric("Vol Change vs. Return", f"{corr_vol_change_ret:.3f}")
        
        # Scatter plot of volatility vs returns
        fig_corr = px.scatter(
            returns_vs_vol.dropna(),
            x='10d_vol',
            y='returns',
            trendline='ols',
            labels={'10d_vol': '10-Day Volatility (%)', 'returns': 'Daily Return (%)'},
            title='Relationship Between Volatility and Returns'
        )
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Volatility clustering analysis
        st.subheader("Volatility Clustering Analysis")
        
        # Calculate autocorrelation of volatility
        autocorr_lags = 10
        autocorr = [historical_data['10d_vol'].autocorr(lag=i) for i in range(1, autocorr_lags + 1)]
        
        fig_autocorr = px.bar(
            x=list(range(1, autocorr_lags + 1)),
            y=autocorr,
            labels={'x': 'Lag (Days)', 'y': 'Autocorrelation'},
            title='Volatility Autocorrelation'
        )
        
        st.plotly_chart(fig_autocorr, use_container_width=True)
        
        # Check if volatility shows significant autocorrelation
        if max(autocorr) > 0.3:
            st.info("""
            **Volatility Clustering Detected**: The data shows significant autocorrelation in volatility, confirming the presence of volatility clustering.
            
            This suggests that:
            1. High volatility periods tend to be followed by more high volatility
            2. Low volatility periods tend to be followed by more low volatility
            3. Volatility forecasts based on recent volatility have predictive value
            """)
        else:
            st.info("""
            **Limited Volatility Clustering**: The data shows relatively weak autocorrelation in volatility.
            
            This suggests that:
            1. Volatility changes may be more random for this asset
            2. Forecasts based solely on recent volatility might be less reliable
            3. Consider incorporating other factors in volatility predictions
            """)
    else:
        st.error("Failed to fetch historical data for strategy analysis.")

# Footer
st.markdown("---")
st.markdown("**US Stocks Volatility Dashboard** | Data source: Yahoo Finance")