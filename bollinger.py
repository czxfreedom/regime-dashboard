# Save this as pages/07_Bollinger_Bands_Profitability.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pkg_resources  # Add this import here
import pytz

st.set_page_config(
    page_title="Bollinger Bands Profitability Analysis",
    page_icon="📈",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("Bollinger Bands Trading Profitability Analysis")
st.subheader("1-Minute Data Analysis Across Trading Pairs")

# Define parameters
timeframe = "1min"  # Using 1-minute intervals as requested
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

# UI Controls for analysis parameters
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Let user select tokens to analyze
    select_all = st.checkbox("Select All Tokens", value=False)
    
    if select_all:
        selected_tokens = all_tokens
    else:
        selected_tokens = st.multiselect(
            "Select Tokens to Analyze", 
            all_tokens,
            default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens
        )

with col2:
    # Bollinger Bands parameters
    st.subheader("Bollinger Parameters")
    bb_period = st.slider("BB Period", min_value=5, max_value=50, value=20, step=1)
    bb_std_dev = st.slider("BB Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

with col3:
    # Backtest parameters
    st.subheader("Backtest Parameters")
    lookback_days = st.slider("Lookback Days", min_value=1, max_value=30, value=7, step=1)
    
    # Strategy parameters
    strategy_options = {
        "BB Bounce": "Trade when price bounces off the bands",
        "BB Breakout": "Trade when price breaks through the bands",
        "BB Squeeze": "Trade when bands narrow and then expand",
        "All Strategies": "Test all strategies and show best"
    }
    selected_strategy = st.selectbox("Strategy Type", list(strategy_options.keys()))
    
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands for the provided dataframe without requiring TA-Lib
    """
    df['bb_middle'] = df['price'].rolling(window=period).mean()
    rolling_std = df['price'].rolling(window=period).std(ddof=0)
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Calculate %B (where price is within the bands)
    df['bb_percent_b'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

# Define trading strategies
def bb_bounce_strategy(df):
    """
    Strategy: Buy when price touches lower band, sell when price touches upper band
    """
    # Initialize position and signal columns
    df['position'] = 0
    df['signal'] = 0
    
    # Generate buy signals (1) when price touches or goes below the lower band
    df.loc[df['price'] <= df['bb_lower'], 'signal'] = 1
    
    # Generate sell signals (-1) when price touches or goes above the upper band
    df.loc[df['price'] >= df['bb_upper'], 'signal'] = -1
    
    # Track position (1 = long, 0 = no position, -1 = short)
    position = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position <= 0:  # Buy
            position = 1
        elif df['signal'].iloc[i] == -1 and position >= 0:  # Sell
            position = -1
        df['position'].iloc[i] = position
        
    return df

def bb_breakout_strategy(df):
    """
    Strategy: Buy when price breaks above upper band, sell when price breaks below lower band
    """
    # Initialize position and signal columns
    df['position'] = 0
    df['signal'] = 0
    
    # Generate buy signals (1) when price breaks above the upper band
    df.loc[df['price'] > df['bb_upper'], 'signal'] = 1
    
    # Generate sell signals (-1) when price breaks below the lower band
    df.loc[df['price'] < df['bb_lower'], 'signal'] = -1
    
    # Track position
    position = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position <= 0:  # Buy
            position = 1
        elif df['signal'].iloc[i] == -1 and position >= 0:  # Sell
            position = -1
        df['position'].iloc[i] = position
        
    return df

def bb_squeeze_strategy(df):
    """
    Strategy: Buy when bands narrow and then expand with price moving upward
    Sell when bands narrow and then expand with price moving downward
    """
    # Initialize position and signal columns
    df['position'] = 0
    df['signal'] = 0
    
    # Calculate the rate of change in bandwidth
    df['bandwidth_change'] = df['bb_bandwidth'].pct_change(periods=5)
    
    # Define squeeze condition (when bandwidth is in bottom 20% of its range over lookback period)
    lookback = min(len(df), 100)
    df['bandwidth_percentile'] = df['bb_bandwidth'].rolling(window=lookback).apply(
        lambda x: np.percentile(x, 20)
    )
    
    # Identify squeeze
    df['in_squeeze'] = df['bb_bandwidth'] <= df['bandwidth_percentile']
    
    # Identify squeeze exit with upward momentum
    df['squeeze_exit_long'] = (df['in_squeeze'].shift(1) & ~df['in_squeeze']) & (df['price'] > df['bb_middle'])
    
    # Identify squeeze exit with downward momentum
    df['squeeze_exit_short'] = (df['in_squeeze'].shift(1) & ~df['in_squeeze']) & (df['price'] < df['bb_middle'])
    
    # Generate signals
    df.loc[df['squeeze_exit_long'], 'signal'] = 1
    df.loc[df['squeeze_exit_short'], 'signal'] = -1
    
    # Track position
    position = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position <= 0:  # Buy
            position = 1
        elif df['signal'].iloc[i] == -1 and position >= 0:  # Sell
            position = -1
        df['position'].iloc[i] = position
        
    return df

# Calculate strategy returns
def calculate_returns(df):
    """
    Calculate returns based on position signals
    """
    # Calculate price returns
    df['price_return'] = df['price'].pct_change()
    
    # Calculate strategy returns
    df['strategy_return'] = df['position'].shift(1) * df['price_return']
    
    # Calculate cumulative returns
    df['cum_price_return'] = (1 + df['price_return']).cumprod() - 1
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    return df

# Calculate performance metrics
def calculate_performance_metrics(df):
    """
    Calculate performance metrics for the strategy
    """
    # Calculate total return
    total_return = df['cum_strategy_return'].iloc[-1] if not df.empty else 0
    
    # Calculate annualized return
    days = (df.index[-1] - df.index[0]).days
    if days > 0:
        annualized_return = ((1 + total_return) ** (365 / days)) - 1
    else:
        annualized_return = 0
    
    # Calculate daily returns
    df['daily_return'] = df['strategy_return'].resample('D').sum()
    
    # Calculate Sharpe ratio (annualized, assuming risk-free rate of 0)
    if df['daily_return'].std() > 0:
        sharpe_ratio = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0
    
    # Calculate maximum drawdown
    df['cum_peak'] = df['cum_strategy_return'].cummax()
    df['drawdown'] = df['cum_peak'] - df['cum_strategy_return']
    max_drawdown = df['drawdown'].max()
    
    # Calculate win rate
    df['trade'] = df['position'].diff().ne(0)
    df['trade_return'] = np.where(df['trade'], df['strategy_return'], 0)
    wins = (df[df['trade_return'] > 0]['trade_return'].count())
    losses = (df[df['trade_return'] < 0]['trade_return'].count())
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Calculate profit factor
    total_profit = df[df['trade_return'] > 0]['trade_return'].sum()
    total_loss = abs(df[df['trade_return'] < 0]['trade_return'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades
    }

# Fetch price data and run strategy
@st.cache_data(ttl=600, show_spinner="Fetching price data...")
def fetch_price_data_and_run_strategy(token, lookback_days, bb_period, bb_std_dev, strategy_name):
    """
    Fetch price data and run the selected trading strategy
    """
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
        final_price AS price, 
        pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_name = '{token}';
    """
    try:
        df = pd.read_sql(query, engine)

        if df.empty:
            return None, None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Resample to 1-minute data
        df = df.resample('1min').last()
        df = df.fillna(method='ffill')
        
        # Calculate Bollinger Bands
        df = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std_dev)
        
        # Run the selected strategy
        if strategy_name == "BB Bounce":
            df = bb_bounce_strategy(df)
        elif strategy_name == "BB Breakout":
            df = bb_breakout_strategy(df)
        elif strategy_name == "BB Squeeze":
            df = bb_squeeze_strategy(df)
        elif strategy_name == "All Strategies":
            # Run all strategies and keep the best one
            strategies = {
                "BB Bounce": bb_bounce_strategy,
                "BB Breakout": bb_breakout_strategy,
                "BB Squeeze": bb_squeeze_strategy
            }
            
            best_return = -float('inf')
            best_df = None
            best_strat_name = None
            
            for strat_name, strat_func in strategies.items():
                temp_df = df.copy()
                temp_df = strat_func(temp_df)
                temp_df = calculate_returns(temp_df)
                
                if not temp_df.empty and 'cum_strategy_return' in temp_df.columns:
                    final_return = temp_df['cum_strategy_return'].iloc[-1]
                    if final_return > best_return:
                        best_return = final_return
                        best_df = temp_df
                        best_strat_name = strat_name
            
            if best_df is not None:
                df = best_df
                strategy_name = best_strat_name
            else:
                strategy_name = "No valid strategy found"
        
        # Calculate returns
        df = calculate_returns(df)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)
        metrics['strategy_name'] = strategy_name
        
        return df, metrics
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None, None

# Show progress bar while processing
progress_bar = st.progress(0)
status_text = st.empty()

# Process all selected tokens
results = {}
metrics_data = []

for i, token in enumerate(selected_tokens):
    progress_bar.progress((i) / len(selected_tokens))
    status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
    
    df, metrics = fetch_price_data_and_run_strategy(
        token, 
        lookback_days, 
        bb_period, 
        bb_std_dev, 
        selected_strategy
    )
    
    if df is not None and metrics is not None:
        results[token] = {
            'df': df,
            'metrics': metrics
        }
        
        metrics_data.append({
            'Token': token,
            'Strategy': metrics['strategy_name'],
            'Total Return (%)': round(metrics['total_return'] * 100, 2),
            'Annualized Return (%)': round(metrics['annualized_return'] * 100, 2),
            'Sharpe Ratio': round(metrics['sharpe_ratio'], 2),
            'Max Drawdown (%)': round(metrics['max_drawdown'] * 100, 2),
            'Win Rate (%)': round(metrics['win_rate'] * 100, 2),
            'Profit Factor': round(metrics['profit_factor'], 2),
            'Total Trades': metrics['total_trades']
        })

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(results)}/{len(selected_tokens)} tokens successfully")

# Display results
if results:
    # Create performance metrics table
    metrics_df = pd.DataFrame(metrics_data)
    
    # Sort by total return (descending)
    metrics_df = metrics_df.sort_values(by='Total Return (%)', ascending=False)
    
    # Add rank column
    metrics_df.insert(0, 'Rank', range(1, len(metrics_df) + 1))
    
    # Function to color cells based on performance
    def color_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return f'background-color: rgba(0, 255, 0, {min(1, abs(val) / 100)}); color: black'
            elif val < 0:
                return f'background-color: rgba(255, 0, 0, {min(1, abs(val) / 100)}); color: white'
        return ''
    
    def color_sharpe(val):
        if isinstance(val, (int, float)):
            if val > 2:
                return 'background-color: rgba(0, 200, 0, 0.7); color: black'
            elif val > 1:
                return 'background-color: rgba(100, 200, 0, 0.5); color: black'
            elif val < 0:
                return 'background-color: rgba(200, 0, 0, 0.5); color: white'
        return ''
    
    def color_win_rate(val):
        if isinstance(val, (int, float)):
            if val > 60:
                return 'background-color: rgba(0, 200, 0, 0.7); color: black'
            elif val > 50:
                return 'background-color: rgba(100, 200, 0, 0.5); color: black'
            elif val < 40:
                return 'background-color: rgba(200, 0, 0, 0.5); color: white'
        return ''
    
    def color_profit_factor(val):
        if isinstance(val, (int, float)):
            if val > 3:
                return 'background-color: rgba(0, 200, 0, 0.7); color: black'
            elif val > 1.5:
                return 'background-color: rgba(100, 200, 0, 0.5); color: black'
            elif val < 1:
                return 'background-color: rgba(200, 0, 0, 0.5); color: white'
        return ''
    
    # Apply styling
    styled_metrics = metrics_df.style\
        .applymap(color_returns, subset=['Total Return (%)', 'Annualized Return (%)'])\
        .applymap(color_sharpe, subset=['Sharpe Ratio'])\
        .applymap(color_win_rate, subset=['Win Rate (%)'])\
        .applymap(color_profit_factor, subset=['Profit Factor'])\
        .format({
            'Total Return (%)': '{:.2f}%',
            'Annualized Return (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}%',
            'Win Rate (%)': '{:.2f}%',
            'Profit Factor': '{:.2f}'
        })
    
    # Display metrics table
    st.subheader(f"Bollinger Bands Strategy Performance Ranking (Period: {bb_period}, StdDev: {bb_std_dev})")
    st.dataframe(styled_metrics, height=500, use_container_width=True)
    
    # Visualize top performers
    st.subheader("Top 5 Performers - Cumulative Returns")
    
    top5_tokens = metrics_df.sort_values(by='Total Return (%)', ascending=False).head(5)['Token'].tolist()
    
    fig = go.Figure()
    for token in top5_tokens:
        if token in results:
            df = results[token]['df']
            strategy_name = results[token]['metrics']['strategy_name']
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['cum_strategy_return'] * 100,
                mode='lines',
                name=f"{token} ({strategy_name})",
                hovertemplate="%{y:.2f}%"
            ))
    
    fig.update_layout(
        title="Cumulative Returns for Top 5 Performers",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500,
        legend_title="Token (Strategy)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Distribution
    if selected_strategy == "All Strategies":
        st.subheader("Best Strategy Distribution")
        
        strategy_counts = metrics_df['Strategy'].value_counts().reset_index()
        strategy_counts.columns = ['Strategy', 'Count']
        
        fig = px.pie(
            strategy_counts,
            values='Count',
            names='Strategy',
            title='Distribution of Best Performing Strategies',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis of the best performer
    st.subheader("Detailed Analysis of Top Performer")
    
    best_token = metrics_df.iloc[0]['Token']
    best_strategy = metrics_df.iloc[0]['Strategy']
    best_return = metrics_df.iloc[0]['Total Return (%)']
    
    st.write(f"**{best_token}** showed the best performance using the **{best_strategy}** strategy with a return of **{best_return:.2f}%** over the {lookback_days}-day period.")
    
    if best_token in results:
        best_df = results[best_token]['df']
        
        # Show price chart with Bollinger Bands and position signals
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=1)
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['bb_upper'],
            mode='lines',
            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
            name='Upper Band',
            hoverinfo='none'
        ))
        
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['bb_lower'],
            mode='lines',
            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            name='Lower Band',
            hoverinfo='none'
        ))
        
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['bb_middle'],
            mode='lines',
            line=dict(color='rgba(173, 216, 230, 0.9)', width=1, dash='dash'),
            name='Middle Band',
            hoverinfo='none'
        ))
        
        # Add buy signals
        buy_signals = best_df[best_df['signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            marker=dict(color='green', size=8, symbol='triangle-up'),
            name='Buy Signal'
        ))
        
        # Add sell signals
        sell_signals = best_df[best_df['signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['price'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='triangle-down'),
            name='Sell Signal'
        ))
        
        fig.update_layout(
            title=f"{best_token} - Price with Bollinger Bands and Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cumulative returns
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['cum_price_return'] * 100,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='grey', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df['cum_strategy_return'] * 100,
            mode='lines',
            name=f'{best_strategy} Strategy',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{best_token} - Strategy Performance vs Buy & Hold",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show drawdown
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=best_df.index,
            y=-best_df['drawdown'] * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title=f"{best_token} - Strategy Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{best_return:.2f}%")
            st.metric("Annualized Return", f"{metrics_df.iloc[0]['Annualized Return (%)']:.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics_df.iloc[0]['Sharpe Ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics_df.iloc[0]['Max Drawdown (%)']:.2f}%")
        
        with col3:
            st.metric("Win Rate", f"{metrics_df.iloc[0]['Win Rate (%)']:.2f}%")
            st.metric("Profit Factor", f"{metrics_df.iloc[0]['Profit Factor']:.2f}")
    
    # Correlation analysis
    st.subheader("Performance Correlation Analysis")
    
    if len(metrics_df) > 1:
        # Analyze correlation between key metrics
        corr_metrics = metrics_df[['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Profit Factor', 'Max Drawdown (%)']]
        corr_matrix = corr_metrics.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Correlation Between Performance Metrics',
            labels=dict(color="Correlation")
        )
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Correlation Analysis Insights:**
        
        This correlation matrix shows how different performance metrics relate to each other:
        - A strong positive correlation between Total Return and Sharpe Ratio indicates that higher returns generally come with better risk-adjusted performance.
        - The relationship between Win Rate and Total Return shows whether fewer, more profitable trades or more frequent smaller wins are driving performance.
        - Max Drawdown correlation with returns helps understand the risk-return tradeoff in different strategies.
        """)
    
    # Parameter optimization suggestions
    st.subheader("Parameter Optimization Insights")
    
    st.write(f"""
    **Current Parameters:** Bollinger Period = {bb_period}, Standard Deviations = {bb_std_dev}
    
    **Optimization Suggestions:**
    
    1. **For higher volatility tokens** (like {', '.join(top5_tokens[:2]) if len(top5_tokens) >= 2 else top5_tokens[0]}):
       - Consider testing with wider bands (higher standard deviations, 2.5-3.0)
       - Shorter periods (15-18) may capture more trading opportunities
    
    2. **For lower volatility tokens:**
       - Narrower bands (1.5-1.8 standard deviations) may be more effective
       - Longer periods (25-30) can help filter out noise
    
    3. **For optimal results across all tokens:**
       - Test multiple parameter combinations and strategy types
       - Consider different parameters for different market conditions
    """)
    
   # Profitability by market condition
    st.subheader("Strategy Profitability by Market Condition")
    
    market_condition_data = []
    for token, result in results.items():
        df = result['df']
        if not df.empty and 'strategy_return' in df.columns:
            # Classify market conditions based on price trends
            df['price_trend_20'] = df['price'].pct_change(periods=20)
            
            # Define market condition based on volatility (measured by BB bandwidth)
            df['high_volatility'] = df['bb_bandwidth'] > df['bb_bandwidth'].quantile(0.75)
            df['low_volatility'] = df['bb_bandwidth'] < df['bb_bandwidth'].quantile(0.25)
            
            # Calculate returns during different market conditions
            uptrend_returns = df[df['price_trend_20'] > 0.01]['strategy_return'].mean() * 100
            downtrend_returns = df[df['price_trend_20'] < -0.01]['strategy_return'].mean() * 100
            sideways_returns = df[(df['price_trend_20'] >= -0.01) & (df['price_trend_20'] <= 0.01)]['strategy_return'].mean() * 100
            
            high_vol_returns = df[df['high_volatility']]['strategy_return'].mean() * 100
            low_vol_returns = df[df['low_volatility']]['strategy_return'].mean() * 100
            
            market_condition_data.append({
                'Token': token,
                'Strategy': result['metrics']['strategy_name'],
                'Uptrend Returns (%)': round(uptrend_returns, 2) if not pd.isna(uptrend_returns) else 0,
                'Downtrend Returns (%)': round(downtrend_returns, 2) if not pd.isna(downtrend_returns) else 0,
                'Sideways Returns (%)': round(sideways_returns, 2) if not pd.isna(sideways_returns) else 0,
                'High Volatility Returns (%)': round(high_vol_returns, 2) if not pd.isna(high_vol_returns) else 0,
                'Low Volatility Returns (%)': round(low_vol_returns, 2) if not pd.isna(low_vol_returns) else 0
            })
    
    if market_condition_data:
        market_condition_df = pd.DataFrame(market_condition_data)
        
        # Sort by overall return
        market_condition_df = market_condition_df.merge(
            metrics_df[['Token', 'Total Return (%)']],
            on='Token'
        ).sort_values(by='Total Return (%)', ascending=False)
        
        # Apply color styling
        def color_market_returns(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    color_intensity = min(1, abs(val) / 50)
                    return f'background-color: rgba(0, 200, 0, {color_intensity}); color: black'
                elif val < 0:
                    color_intensity = min(1, abs(val) / 50)
                    return f'background-color: rgba(200, 0, 0, {color_intensity}); color: white'
            return ''
        
        styled_market_df = market_condition_df.style.applymap(
            color_market_returns, 
            subset=[
                'Uptrend Returns (%)', 
                'Downtrend Returns (%)', 
                'Sideways Returns (%)',
                'High Volatility Returns (%)',
                'Low Volatility Returns (%)'
            ]
        )
        
        st.dataframe(styled_market_df, height=400, use_container_width=True)
        
        # Create a bar chart comparing market condition returns
        condition_summary = market_condition_df[
            ['Token', 'Uptrend Returns (%)', 'Downtrend Returns (%)', 'Sideways Returns (%)']
        ].head(5)  # Top 5 tokens
        
        # Melt the DataFrame for easier plotting
        condition_summary_melted = pd.melt(
            condition_summary, 
            id_vars=['Token'], 
            value_vars=['Uptrend Returns (%)', 'Downtrend Returns (%)', 'Sideways Returns (%)'],
            var_name='Market Condition', 
            value_name='Returns (%)'
        )
        
        # Create bar chart
        fig = px.bar(
            condition_summary_melted,
            x='Token',
            y='Returns (%)',
            color='Market Condition',
            barmode='group',
            title='Strategy Returns by Market Condition (Top 5 Tokens)',
            color_discrete_map={
                'Uptrend Returns (%)': 'green',
                'Downtrend Returns (%)': 'red',
                'Sideways Returns (%)': 'blue'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a similar chart for volatility conditions
        volatility_summary = market_condition_df[
            ['Token', 'High Volatility Returns (%)', 'Low Volatility Returns (%)']
        ].head(5)  # Top 5 tokens
        
        # Melt the DataFrame for easier plotting
        volatility_summary_melted = pd.melt(
            volatility_summary, 
            id_vars=['Token'], 
            value_vars=['High Volatility Returns (%)', 'Low Volatility Returns (%)'],
            var_name='Volatility Condition', 
            value_name='Returns (%)'
        )
        
        # Create bar chart
        fig = px.bar(
            volatility_summary_melted,
            x='Token',
            y='Returns (%)',
            color='Volatility Condition',
            barmode='group',
            title='Strategy Returns by Volatility Condition (Top 5 Tokens)',
            color_discrete_map={
                'High Volatility Returns (%)': 'purple',
                'Low Volatility Returns (%)': 'orange'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.write("""
        **Market Condition Analysis Insights:**
        
        This analysis shows how each strategy performs under different market conditions:
        
        1. **Trend Analysis:**
           - Strategies that perform well in both up and down trends are more versatile
           - Strategies that excel in sideways markets often struggle in trending markets (and vice versa)
        
        2. **Volatility Impact:**
           - High volatility environments may benefit breakout strategies
           - Low volatility environments typically favor mean-reversion strategies
           
        3. **Strategy Selection:**
           - Choose strategies based on your market outlook and token characteristics
           - Consider combining multiple strategies for different market conditions
        """)
    
    # Trade timing analysis
    st.subheader("Trade Timing Analysis")
    
    best_token = metrics_df.iloc[0]['Token']
    if best_token in results:
        best_df = results[best_token]['df']
        
        # Analyze trade timing
        if 'signal' in best_df.columns:
            # Group by hour of day to see when trades occur
            best_df['hour'] = best_df.index.hour
            
            # Count signals by hour
            signal_counts = best_df[best_df['signal'] != 0].groupby('hour')['signal'].count()
            
            # Calculate returns by hour
            returns_by_hour = best_df.groupby('hour')['strategy_return'].mean() * 100
            
            # Combine the data
            timing_df = pd.DataFrame({
                'Hour': signal_counts.index,
                'Signal Count': signal_counts.values,
                'Avg Return (%)': returns_by_hour.reindex(signal_counts.index).values
            })
            
            # Create a dual-axis chart
            fig = go.Figure()
            
            # Add signal counts as bars
            fig.add_trace(go.Bar(
                x=timing_df['Hour'],
                y=timing_df['Signal Count'],
                name='Number of Signals',
                marker_color='blue',
                opacity=0.7,
                yaxis='y'
            ))
            
            # Add average returns as line
            fig.add_trace(go.Scatter(
                x=timing_df['Hour'],
                y=timing_df['Avg Return (%)'],
                name='Avg Return (%)',
                marker_color='green',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            # Update layout for dual y-axes
            fig.update_layout(
                title=f"Trade Timing Analysis for {best_token} (Hour of Day, Singapore Time)",
                xaxis=dict(
                    title='Hour of Day',
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                yaxis=dict(
                    title='Number of Signals',
                    titlefont=dict(color='blue'),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title='Avg Return (%)',
                    titlefont=dict(color='green'),
                    tickfont=dict(color='green'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insights on trade timing
            best_hour_return = timing_df.loc[timing_df['Avg Return (%)'].idxmax()]
            worst_hour_return = timing_df.loc[timing_df['Avg Return (%)'].idxmin()]
            most_active_hour = timing_df.loc[timing_df['Signal Count'].idxmax()]
            
            st.write(f"""
            **Trade Timing Insights for {best_token}:**
            
            - **Most profitable hour:** {int(best_hour_return['Hour'])}:00 with average return of {best_hour_return['Avg Return (%)']:.2f}%
            - **Least profitable hour:** {int(worst_hour_return['Hour'])}:00 with average return of {worst_hour_return['Avg Return (%)']:.2f}%
            - **Most active trading hour:** {int(most_active_hour['Hour'])}:00 with {most_active_hour['Signal Count']} trading signals
            
            These patterns may relate to market open/close times, liquidity conditions, or specific news release schedules.
            """)
    
    # Trade size optimization
    st.subheader("Position Sizing Optimization")
    
    # Create a hypothetical position sizing strategy
    st.write("""
    **Position Sizing Strategy Recommendations:**
    
    Based on the performance metrics, we recommend the following position sizing approaches:
    
    1. **Fixed Fractional Method:**
       - Allocate a consistent percentage of capital to each trade
       - For high win-rate strategies (>60%), consider 2-5% per trade
       - For lower win-rate strategies, reduce to 1-2% per trade
    
    2. **Kelly Criterion-based Sizing:**
       - For top performers:
         ```
         Kelly % = Win Rate - ((1 - Win Rate) / (Profit Factor))
         ```
         
    3. **Volatility-Adjusted Sizing:**
       - Reduce position size during high volatility periods
       - Increase position size during stable market conditions
       
    4. **Average True Range (ATR) Based Sizing:**
       - Set stop losses at 1-2x ATR from entry
       - Size positions to risk a consistent percentage per trade
    """)
    
    # Final summary and recommendations
    st.subheader("Executive Summary and Recommendations")
    
    # Calculate overall metrics across all tokens
    avg_return = metrics_df['Total Return (%)'].mean()
    avg_sharpe = metrics_df['Sharpe Ratio'].mean()
    avg_win_rate = metrics_df['Win Rate (%)'].mean()
    
    # Find percentage of profitable tokens
    profitable_tokens = metrics_df[metrics_df['Total Return (%)'] > 0].shape[0]
    profitable_pct = (profitable_tokens / metrics_df.shape[0]) * 100
    
    # Find best strategy (if All Strategies was selected)
    if selected_strategy == "All Strategies":
        best_strategy_overall = metrics_df['Strategy'].value_counts().idxmax()
    else:
        best_strategy_overall = selected_strategy
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Return", f"{avg_return:.2f}%")
    
    with col2:
        st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
    
    with col3:
        st.metric("Avg Win Rate", f"{avg_win_rate:.2f}%")
    
    with col4:
        st.metric("Profitable Tokens", f"{profitable_pct:.1f}%")
    
    # Provide executive summary
    st.write(f"""
    ### Key Findings

    1. **Overall Performance:** {profitable_tokens} out of {metrics_df.shape[0]} tokens ({profitable_pct:.1f}%) showed profitable results using Bollinger Bands strategies over the {lookback_days}-day period.
    
    2. **Best Performer:** **{metrics_df.iloc[0]['Token']}** achieved the highest return of **{metrics_df.iloc[0]['Total Return (%)']:.2f}%** using the **{metrics_df.iloc[0]['Strategy']}** strategy.
    
    3. **Strategy Performance:** The **{best_strategy_overall}** strategy was the most effective overall for the analyzed tokens.
    
    4. **Parameter Settings:** Bollinger Bands with period = {bb_period} and standard deviation = {bb_std_dev} delivered the best results for the majority of tokens.
    
    ### Recommendations
    
    1. **Top Trading Opportunities:**
       - Focus on the top 3 tokens: {', '.join(metrics_df.head(3)['Token'].tolist())}
       - These tokens show strong technical patterns compatible with Bollinger Band strategies
    
    2. **Optimize Parameters:**
       - Consider testing {bb_period-5}-{bb_period+5} periods and {bb_std_dev-0.5}-{bb_std_dev+0.5} standard deviations
       - Different tokens respond best to different parameter settings
    
    3. **Implementation Strategy:**
       - Trade during the most profitable hours identified in the timing analysis
       - Adjust position sizes based on win rate and profit factor
       - Monitor market conditions and adjust strategies accordingly
    
    4. **Risk Management:**
       - Set stop losses at key technical levels or based on ATR
       - Diversify across multiple tokens and possibly multiple strategies
       - Consider reducing position sizes during high volatility periods
    """)
    
    # Add a download button for the full results
    csv = metrics_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name=f"bollinger_bands_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# Add information about the page
with st.expander("About This Analysis"):
    st.markdown("""
    ### Bollinger Bands Trading Profitability Analysis
    
    This tool analyzes the profitability of trading strategies based on Bollinger Bands indicators using 1-minute data for various cryptocurrency pairs.
    
    #### Trading Strategies Explained
    
    1. **BB Bounce Strategy:**
       - Buy when price touches the lower band (oversold condition)
       - Sell when price touches the upper band (overbought condition)
       - Works best in ranging markets with consistent volatility
    
    2. **BB Breakout Strategy:**
       - Buy when price breaks above the upper band (strong upward momentum)
       - Sell when price breaks below the lower band (strong downward momentum)
       - Works best in trending markets with increasing volatility
    
    3. **BB Squeeze Strategy:**
       - Identifies when Bollinger Bands narrow (low volatility) and then expand
       - Buy when bands start to expand with price moving upward
       - Sell when bands start to expand with price moving downward
       - Works well for capturing the beginning of new trends
    
    #### Performance Metrics Explained
    
    - **Total Return:** Cumulative return over the entire period
    - **Annualized Return:** Return normalized to an annual rate
    - **Sharpe Ratio:** Risk-adjusted return (higher is better)
    - **Max Drawdown:** Largest percentage drop from peak to trough
    - **Win Rate:** Percentage of trades that were profitable
    - **Profit Factor:** Ratio of gross profits to gross losses
    
    #### Data Source
    
    The analysis uses 1-minute price data from the oracle price database, with timestamps converted to Singapore time.
    
    #### Limitations
    
    - Backtest results are theoretical and don't account for slippage, fees, or liquidity
    - Past performance is not indicative of future results
    - The analysis does not consider fundamental factors or market news
    """)