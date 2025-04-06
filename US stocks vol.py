import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import pytz
from ib_insync import *

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

# Function to connect to Interactive Brokers
@st.cache_resource
def connect_to_ib():
    ib = IB()
    try:
        # Try to connect to TWS
        ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 for TWS paper trading, 7496 for TWS real, 4002 for IB Gateway paper
        st.sidebar.success("Connected to Interactive Brokers")
        return ib
    except:
        try:
            # Try to connect to IB Gateway
            ib.connect('127.0.0.1', 4001, clientId=1)  # 4001 for IB Gateway real
            st.sidebar.success("Connected to Interactive Brokers Gateway")
            return ib
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Interactive Brokers: {str(e)}")
            st.sidebar.info("Make sure TWS or IB Gateway is running and API connections are enabled")
            return None

# Sidebar for stock selection
st.sidebar.header("Settings")
ticker_options = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD"]
selected_ticker = st.sidebar.selectbox("Select stock:", ticker_options)
    
# Connect to IB when the app starts
ib = connect_to_ib()

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

# Function to fetch historical data from Interactive Brokers
def get_ib_historical_data(ib, ticker, duration, bar_size):
    """
    Fetch historical data from Interactive Brokers
    
    Parameters:
    ib: IB connection
    ticker: Stock symbol
    duration: e.g., '2 D', '1 W', '2 Y'
    bar_size: e.g., '5 mins', '30 mins', '1 day'
    
    Returns:
    pandas DataFrame with historical data
    """
    if ib is None:
        return None
    
    # Create contract
    contract = Stock(ticker, 'SMART', 'USD')
    
    try:
        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',  # '' for latest data
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Convert to DataFrame
        if bars:
            df = util.df(bars)
            # Convert time to datetime with Eastern Time Zone
            df['date'] = pd.to_datetime(df['date'])
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            return df
        else:
            st.error(f"No data returned for {ticker}")
            return None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

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

# Check if IB connection is available
if ib is None:
    st.error("Interactive Brokers connection is required. Please make sure TWS or IB Gateway is running.")
else:
    # Tab 1: 5-minute sampling volatility averaged over 30 minutes (last 24 hours)
    with tab1:
        st.header(f"5-Minute Sampling Volatility for {selected_ticker}")
        st.subheader("Averaged over 30 minutes (Last 24 hours)")
        
        # Get 5-minute data for the last 2 days
        data_5min = get_ib_historical_data(ib, selected_ticker, "2 D", "5 mins")
        
        if data_5min is not None and not data_5min.empty:
            # Set index to date column
            data_5min = data_5min.set_index('date')
            
            # Keep only last 24 hours of data
            last_24h = datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=1)
            data_5min = data_5min[data_5min.index > last_24h]
            
            if len(data_5min) > 0:
                # Calculate 5-min realized volatility (6 periods = 30 minutes)
                data_5min['realized_vol'] = calculate_realized_volatility(data_5min['returns'], window=6, sampling_rate=5)
                
                # Create plots
                fig1 = go.Figure()
                
                # Add price line
                fig1.add_trace(go.Scatter(
                    x=data_5min.index,
                    y=data_5min['close'],
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
        
        # Get 30-minute data for the last 10 days (to ensure we have a full 7 days)
        data_30min = get_ib_historical_data(ib, selected_ticker, "10 D", "30 mins")
        
        if data_30min is not None and not data_30min.empty:
            # Set index to date column
            data_30min = data_30min.set_index('date')
            
            # Keep only last 7 days of data
            last_7d = datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=7)
            data_30min = data_30min[data_30min.index > last_7d]
            
            if len(data_30min) > 0:
                # Calculate 30-min realized volatility (16 periods ~= 1 trading day of 8 hours)
                data_30min['realized_vol'] = calculate_realized_volatility(data_30min['returns'], window=16, sampling_rate=30)
                
                # Create plots
                fig2 = go.Figure()
                
                # Add price line
                fig2.add_trace(go.Scatter(
                    x=data_30min.index,
                    y=data_30min['close'],
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
        historical_data = get_ib_historical_data(ib, selected_ticker, "2 Y", "1 day")
        
        if historical_data is not None and not historical_data.empty:
            # Set index to date column
            historical_data = historical_data.set_index('date')
            
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
                
                This strategy trades based on the relationship between short-term and long-term volatility:
                - Goes long when short-term volatility (10-day) is significantly higher than long-term volatility (20-day), ratio > 1.1
                - Goes short when short-term volatility is significantly lower than long-term volatility, ratio < 0.9
                - Stays neutral when the volatility term structure is flat (ratio between 0.9 and 1.1)
                
                The strategy capitalizes on the mean-reverting nature of volatility term structure.
                """)
                
                # Parameters
                upper_ratio = st.slider("Upper Ratio Threshold", 1.01, 1.5, 1.1, 0.01)
                lower_ratio = st.slider("Lower Ratio Threshold", 0.5, 0.99, 0.9, 0.01)
                st.info(f"The strategy goes long when 10d/20d volatility ratio exceeds {upper_ratio} and short when it falls below {lower_ratio}.")
                
            elif strategy_option == "Volatility Range Breakout":
                active_strategy = "range"
                strategy_signal = historical_data['range_signal']
                strategy_returns = historical_data['range_return']
                strategy_metrics = all_strategies_results['Range Breakout']
                
                st.markdown("""
                ### Volatility Range Breakout Strategy
                
                This strategy identifies statistical extremes in volatility:
                - Goes short when volatility breaks above its 2-standard deviation upper band
                - Goes long when volatility breaks below its 2-standard deviation lower band
                - Stays neutral when volatility is within its normal range
                
                The strategy is based on the principle that extreme volatility levels tend to revert to the mean.
                """)
                
                # Parameters
                std_dev_multiplier = st.slider("Standard Deviation Multiplier", 1.0, 3.0, 2.0, 0.1)
                lookback_period = st.slider("Lookback Period (days)", 10, 60, 20, 5)
                st.info(f"The strategy identifies volatility extremes using a {std_dev_multiplier}-standard deviation band calculated over a {lookback_period}-day period.")
            
            # Display equity curve for selected strategy
            st.subheader("Strategy Performance")
            
            # Determine which strategy's returns to use based on selection
            if strategy_option == "Mean Reversion on High Volatility":
                strategy_returns = historical_data['mean_reversion_return']
                strategy_metrics = all_strategies_results['Mean Reversion']
            elif strategy_option == "Volatility Breakout Momentum":
                strategy_returns = historical_data['breakout_return']
                strategy_metrics = all_strategies_results['Volatility Breakout']
            elif strategy_option == "Volatility Regime Switching":
                strategy_returns = historical_data['regime_return']
                strategy_metrics = all_strategies_results['Regime Switching']
            elif strategy_option == "Volatility Term Structure":
                strategy_returns = historical_data['term_return']
                strategy_metrics = all_strategies_results['Term Structure']
            elif strategy_option == "Volatility Range Breakout":
                strategy_returns = historical_data['range_return']
                strategy_metrics = all_strategies_results['Range Breakout']
            
            # Calculate equity curve
            strategy_equity = (1 + strategy_returns.fillna(0)).cumprod()
            buy_hold_equity = (1 + historical_data['returns'].fillna(0)).cumprod()
            
            # Plot equity curves
            fig_equity = go.Figure()
            
            fig_equity.add_trace(go.Scatter(
                x=strategy_equity.index,
                y=strategy_equity,
                name='Strategy Equity',
                line=dict(color='green', width=2)
            ))
            
            fig_equity.add_trace(go.Scatter(
                x=buy_hold_equity.index,
                y=buy_hold_equity,
                name='Buy & Hold Equity',
                line=dict(color='gray', width=1.5, dash='dot')
            ))
            
            fig_equity.update_layout(
                title='Strategy Equity Curve vs. Buy & Hold',
                height=500,
                yaxis=dict(title='Growth of $1 Invested'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Display strategy metrics
            st.subheader("Strategy Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{strategy_metrics['total_return']:.2%}")
                st.metric("Annualized Return", f"{strategy_metrics['annualized_return']:.2%}")
                st.metric("Annualized Volatility", f"{strategy_metrics['annualized_volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{strategy_metrics['sharpe_ratio']:.2f}")
            
            with col2:
                st.metric("Max Drawdown", f"{strategy_metrics['max_drawdown']:.2%}")
                st.metric("Sortino Ratio", f"{strategy_metrics['sortino_ratio']:.2f}")
                st.metric("Calmar Ratio", f"{strategy_metrics['calmar_ratio']:.2f}")
                st.metric("Win Rate", f"{strategy_metrics['win_rate']:.2%}")
            
            with col3:
                st.metric("Profit Factor", f"{strategy_metrics['profit_factor']:.2f}")
                st.metric("Gain/Loss Ratio", f"{strategy_metrics['gain_to_loss_ratio']:.2f}")
                st.metric("% Profitable Months", f"{strategy_metrics['percent_profitable_months']:.2%}")
                st.metric("Max Consecutive Wins", f"{strategy_metrics['max_consecutive_wins']:.0f}")
            
            # Strategy signal distribution
            st.subheader("Signal Distribution")
            
            # Determine which strategy's signals to analyze
            if strategy_option == "Mean Reversion on High Volatility":
                signal_col = 'mean_reversion_signal'
            elif strategy_option == "Volatility Breakout Momentum":
                signal_col = 'breakout_signal'
            elif strategy_option == "Volatility Regime Switching":
                signal_col = 'regime_signal'
            elif strategy_option == "Volatility Term Structure":
                signal_col = 'term_signal'
            elif strategy_option == "Volatility Range Breakout":
                signal_col = 'range_signal'
            
            # Create signal distribution chart
            signal_counts = historical_data[signal_col].value_counts()
            
            fig_signal = px.pie(
                values=signal_counts.values,
                names=signal_counts.index.map({1: 'Long', -1: 'Short', 0: 'Neutral'}),
                title='Strategy Signal Distribution',
                color=signal_counts.index.map({1: 'Long', -1: 'Short', 0: 'Neutral'}),
                color_discrete_map={'Long': 'green', 'Short': 'red', 'Neutral': 'gray'}
            )
            
            st.plotly_chart(fig_signal, use_container_width=True)
            
            # Show monthly returns heatmap
            st.subheader("Monthly Returns Heatmap")
            
            # Resample returns to month
            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create a DataFrame for heatmap
            monthly_returns_df = pd.DataFrame(monthly_returns)
            monthly_returns_df['year'] = monthly_returns_df.index.year
            monthly_returns_df['month'] = monthly_returns_df.index.month
            
            # Pivot the DataFrame
            heatmap_data = monthly_returns_df.pivot_table(
                index='year',
                columns='month',
                values=0,
                aggfunc='sum'
            ).fillna(0)
            
            # Rename columns to month names
            heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Convert to percentages
            heatmap_data = heatmap_data * 100
            
            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_data,
                text_auto='.2f',
                aspect="auto",
                title="Monthly Returns (%)",
                labels=dict(x="Month", y="Year", color="Return (%)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
                zmin=-max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max())),
                zmax=max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Display drawdown chart
            st.subheader("Drawdown Analysis")
            
            # Calculate drawdowns
            equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            
            # Create drawdown chart
            fig_drawdown = go.Figure()
            
            fig_drawdown.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ))
            
            fig_drawdown.update_layout(
                title='Drawdown Over Time',
                height=400,
                yaxis=dict(title='Drawdown (%)', tickformat='.0%'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_drawdown, use_container_width=True)
            
            # Display a table of worst drawdowns
            def find_drawdown_periods(drawdown_series):
                drawdown_start = None
                drawdown_end = None
                drawdown_periods = []
                
                for date, value in drawdown_series.items():
                    if drawdown_start is None and value < 0:
                        drawdown_start = date
                    elif drawdown_start is not None and value == 0:
                        drawdown_end = date
                        max_drawdown = drawdown_series[drawdown_start:drawdown_end].min()
                        max_drawdown_date = drawdown_series[drawdown_start:drawdown_end].idxmin()
                        recovery_days = (drawdown_end - max_drawdown_date).days
                        duration = (drawdown_end - drawdown_start).days
                        
                        drawdown_periods.append({
                            'Start': drawdown_start,
                            'Bottom': max_drawdown_date,
                            'End': drawdown_end,
                            'Depth': max_drawdown,
                            'Length (days)': duration,
                            'Recovery (days)': recovery_days
                        })
                        
                        drawdown_start = None
                
                # Handle ongoing drawdown
                if drawdown_start is not None:
                    drawdown_end = drawdown_series.index[-1]
                    max_drawdown = drawdown_series[drawdown_start:drawdown_end].min()
                    max_drawdown_date = drawdown_series[drawdown_start:drawdown_end].idxmin()
                    duration = (drawdown_end - drawdown_start).days
                    
                    drawdown_periods.append({
                        'Start': drawdown_start,
                        'Bottom': max_drawdown_date,
                        'End': 'Ongoing',
                        'Depth': max_drawdown,
                        'Length (days)': duration,
                        'Recovery (days)': float('nan')
                    })
                
                return pd.DataFrame(drawdown_periods).sort_values('Depth')
            
            drawdown_periods = find_drawdown_periods(drawdown)
            
            if not drawdown_periods.empty:
                # Keep only the 5 worst drawdowns
                worst_drawdowns = drawdown_periods.head(5)
                
                # Format the data for display
                display_drawdowns = worst_drawdowns.copy()
                display_drawdowns['Depth'] = display_drawdowns['Depth'].apply(lambda x: f"{x:.2%}")
                
                st.subheader("Worst Drawdowns")
                st.dataframe(display_drawdowns, use_container_width=True)
            
            # Strategy settings and explanation
            st.subheader("Strategy Settings")
            
            # Create expander for detailed strategy explanation
            with st.expander("Strategy Details and Implementation"):
                # Display Python code for implementing the selected strategy
                if strategy_option == "Mean Reversion on High Volatility":
                    st.code("""
def mean_reversion_strategy(data, z_score_threshold=1.5):
    # Calculate 10-day volatility
    data['10d_vol'] = calculate_realized_volatility(data['returns'], window=10, sampling_rate=1440)
    
    # Calculate volatility Z-score using 60-day lookback
    data['vol_z_score'] = (data['10d_vol'] - data['10d_vol'].rolling(60).mean()) / data['10d_vol'].rolling(60).std()
    
    # Generate signals
    data['signal'] = 0  # Initialize with neutral position
    data.loc[data['vol_z_score'] > z_score_threshold, 'signal'] = -1  # Short when volatility is high
    data.loc[data['vol_z_score'] < 0, 'signal'] = 1  # Long when volatility is below average
    
    # Shift signals for next-day execution
    data['signal'] = data['signal'].shift(1)
    
    # Calculate strategy returns
    data['strategy_return'] = data['signal'] * data['returns']
    
    return data
""", language="python")
                
                elif strategy_option == "Volatility Breakout Momentum":
                    st.code("""
def volatility_breakout_strategy(data, volatility_change_threshold=30):
    # Calculate 10-day volatility
    data['10d_vol'] = calculate_realized_volatility(data['returns'], window=10, sampling_rate=1440)
    
    # Calculate day-over-day volatility change percentage
    data['vol_change_pct'] = data['10d_vol'].pct_change() * 100
    
    # Determine price direction
    data['price_direction'] = np.sign(data['returns'])
    
    # Generate signals - go with price direction when volatility spikes
    data['signal'] = 0  # Initialize with neutral position
    data.loc[data['vol_change_pct'] > volatility_change_threshold, 'signal'] = data['price_direction']
    
    # Shift signals for next-day execution
    data['signal'] = data['signal'].shift(1)
    
    # Calculate strategy returns
    data['strategy_return'] = data['signal'] * data['returns']
    
    return data
""", language="python")
                
                elif strategy_option == "Volatility Regime Switching":
                    st.code("""
def regime_switching_strategy(data, low_percentile=25, high_percentile=75):
    # Calculate 10-day volatility
    data['10d_vol'] = calculate_realized_volatility(data['returns'], window=10, sampling_rate=1440)
    
    # Determine volatility regimes
    low_vol = data['10d_vol'].rolling(60).quantile(low_percentile/100)
    high_vol = data['10d_vol'].rolling(60).quantile(high_percentile/100)
    
    data['regime'] = 'medium'  # Default regime
    data.loc[data['10d_vol'] <= low_vol, 'regime'] = 'low'
    data.loc[data['10d_vol'] >= high_vol, 'regime'] = 'high'
    
    # Generate signals based on regime
    data['signal'] = 0  # Initialize with neutral position (medium volatility)
    
    # In low volatility - trend following (5-day momentum)
    data.loc[data['regime'] == 'low', 'signal'] = np.sign(data['returns'].rolling(5).mean())
    
    # In high volatility - counter-trend
    data.loc[data['regime'] == 'high', 'signal'] = -np.sign(data['returns'])
    
    # Shift signals for next-day execution
    data['signal'] = data['signal'].shift(1)
    
    # Calculate strategy returns
    data['strategy_return'] = data['signal'] * data['returns']
    
    return data
""", language="python")
                
                elif strategy_option == "Volatility Term Structure":
                    st.code("""
def term_structure_strategy(data, upper_threshold=1.1, lower_threshold=0.9):
    # Calculate volatility for different time periods
    data['10d_vol'] = calculate_realized_volatility(data['returns'], window=10, sampling_rate=1440)
    data['20d_vol'] = calculate_realized_volatility(data['returns'], window=20, sampling_rate=1440)
    
    # Calculate the ratio of short-term to longer-term volatility
    data['vol_ratio'] = data['10d_vol'] / data['20d_vol']
    
    # Generate signals based on the term structure
    data['signal'] = 0  # Initialize with neutral position
    data.loc[data['vol_ratio'] > upper_threshold, 'signal'] = 1  # Long when short-term vol is higher
    data.loc[data['vol_ratio'] < lower_threshold, 'signal'] = -1  # Short when short-term vol is lower
    
    # Shift signals for next-day execution
    data['signal'] = data['signal'].shift(1)
    
    # Calculate strategy returns
    data['strategy_return'] = data['signal'] * data['returns']
    
    return data
""", language="python")
                
                elif strategy_option == "Volatility Range Breakout":
                    st.code("""
def range_breakout_strategy(data, std_dev_multiplier=2.0, lookback_period=20):
    # Calculate 10-day volatility
    data['10d_vol'] = calculate_realized_volatility(data['returns'], window=10, sampling_rate=1440)
    
    # Calculate the upper and lower bands
    data['vol_ma'] = data['10d_vol'].rolling(lookback_period).mean()
    data['vol_std'] = data['10d_vol'].rolling(lookback_period).std()
    data['vol_upper'] = data['vol_ma'] + std_dev_multiplier * data['vol_std']
    data['vol_lower'] = data['vol_ma'] - std_dev_multiplier * data['vol_std']
    
    # Generate signals
    data['signal'] = 0  # Initialize with neutral position
    data.loc[data['10d_vol'] > data['vol_upper'], 'signal'] = -1  # Short when volatility breaks upper band
    data.loc[data['10d_vol'] < data['vol_lower'], 'signal'] = 1  # Long when volatility breaks lower band
    
    # Shift signals for next-day execution
    data['signal'] = data['signal'].shift(1)
    
    # Calculate strategy returns
    data['strategy_return'] = data['signal'] * data['returns']
    
    return data
""", language="python")
                
                st.markdown("""
                #### Implementation Notes:
                - All strategies use realized volatility calculated from daily returns
                - Signals are shifted by 1 day to avoid look-ahead bias
                - Position sizing is simplified (equal size for all positions)
                - No transaction costs or slippage are considered in the backtest
                
                #### Potential Enhancements:
                - Add position sizing based on volatility (risk parity approach)
                - Implement stop-loss mechanisms
                - Add transaction costs and slippage
                - Combine multiple volatility strategies for diversification
                - Add volatility targeting to control overall risk
                """)
            
            # Add export functionality
            st.subheader("Export Results")
            
            export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"], horizontal=True)
            export_button = st.button("Generate Export")
            
            if export_button:
                # Create a DataFrame with the signals, returns, and equity curve
                export_data = pd.DataFrame({
                    'Date': historical_data.index,
                    'Price': historical_data['close'],
                    'Volatility': historical_data['10d_vol'],
                    'Signal': strategy_signal,
                    'Strategy_Return': strategy_returns,
                    'Equity_Curve': strategy_equity
                })
                
                if export_format == "CSV":
                    csv = export_data.to_csv(index=False)
                    st.download_button("Download CSV", csv, f"{selected_ticker}_{active_strategy}_results.csv", "text/csv")
                elif export_format == "Excel":
                    st.warning("Excel export is not directly supported in Streamlit. Please use CSV or JSON format.")
                elif export_format == "JSON":
                    json_data = export_data.to_json(orient="records", date_format="iso")
                    st.download_button("Download JSON", json_data, f"{selected_ticker}_{active_strategy}_results.json", "application/json")
        else:
            st.error("Failed to fetch historical daily data for selected ticker.")

# Clean up IB connection when app is closed
def disconnect_ib():
    if 'ib' in locals() and ib is not None:
        ib.disconnect()

# Register the cleanup function to run at app shutdown
import atexit
atexit.register(disconnect_ib)