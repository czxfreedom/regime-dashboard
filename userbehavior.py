import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# --- User Behavior Analysis System (Weekly Analysis) ---

def fetch_user_behavior_data(engine, start_time, end_time):
    """
    Fetch user trading behavior data for weekly analysis.
    
    Returns DataFrame with user metrics by pair including:
    - user_id: Unique identifier for the user
    - pair_id: Trading pair identifier
    - pair_name: Trading pair name (e.g., BTC/USDT)
    - total_trades: Number of trades in the period
    - winning_trades: Number of profitable trades
    - losing_trades: Number of unprofitable trades
    - winning_amount: Total profit from winning trades
    - losing_amount: Total loss from losing trades
    - profit_factor: winning_amount / abs(losing_amount)
    - total_volume: Total trading volume in USD
    - platform_revenue: How much the platform made from this user-pair
    - avg_position_time: Average time positions were held
    - volatility_rank: Rank of pair by volatility (1=highest)
    - max_drawdown: Maximum drawdown in the period
    """
    
    query = """
    WITH user_pair_trades AS (
        -- Get all user trades by pair in the period
        SELECT
            t."user_id",
            t."pair_id",
            p."pair_name",
            t."way", -- 1 for long, 2 for short
            t."size",
            t."price",
            t."taker_fee",
            t."collateral_price",
            t."created_at",
            t."closed_at",
            t."pnl",
            EXTRACT(EPOCH FROM (t."closed_at" - t."created_at"))/3600 AS "position_hours",
            CASE WHEN t."pnl" > 0 THEN 1 ELSE 0 END AS "is_winning_trade",
            CASE WHEN t."pnl" <= 0 THEN 1 ELSE 0 END AS "is_losing_trade",
            CASE WHEN t."pnl" > 0 THEN t."pnl" * t."collateral_price" ELSE 0 END AS "winning_amount",
            CASE WHEN t."pnl" <= 0 THEN ABS(t."pnl" * t."collateral_price") ELSE 0 END AS "losing_amount"
        FROM
            "public"."trade_fill_fresh" t
        JOIN
            "public"."trade_pool_pairs" p ON t."pair_id" = p."pair_id"
        WHERE
            t."created_at" BETWEEN :start_time AND :end_time
            AND t."taker_way" IN (1, 2, 3, 4) -- Real trades, not funding
    ),
    
    user_pair_metrics AS (
        -- Calculate key metrics per user-pair
        SELECT
            "user_id",
            "pair_id",
            "pair_name",
            COUNT(*) AS "total_trades",
            SUM("is_winning_trade") AS "winning_trades",
            SUM("is_losing_trade") AS "losing_trades",
            SUM("winning_amount") AS "winning_amount",
            SUM("losing_amount") AS "losing_amount",
            SUM("size" * "price" * "collateral_price") AS "total_volume",
            SUM("taker_fee" * "collateral_price") AS "platform_revenue",
            AVG("position_hours") AS "avg_position_hours",
            SUM("pnl" * "collateral_price") AS "net_pnl"
        FROM
            user_pair_trades
        GROUP BY
            "user_id", "pair_id", "pair_name"
    ),
    
    -- Calculate profit factor with safeguards against division by zero
    user_profit_factors AS (
        SELECT
            *,
            CASE 
                WHEN "losing_amount" > 0 THEN "winning_amount" / "losing_amount"
                WHEN "losing_amount" = 0 AND "winning_amount" > 0 THEN 999 -- Very high value for all wins
                ELSE 0 -- No winning trades
            END AS "profit_factor"
        FROM 
            user_pair_metrics
    ),
    
    -- Get pair volatility rankings
    pair_volatility AS (
        SELECT
            "pair_id",
            "pair_name",
            STDDEV("price_change_24h_percent") AS "volatility_score",
            RANK() OVER (ORDER BY STDDEV("price_change_24h_percent") DESC) AS "volatility_rank"
        FROM
            "public"."trade_pool_pairs"
        GROUP BY
            "pair_id", "pair_name"
    ),
    
    -- Calculate drawdown series for each user-pair
    user_pair_drawdowns AS (
        SELECT
            t."user_id",
            t."pair_id",
            t."pair_name",
            t."created_at",
            SUM(t."pnl" * t."collateral_price") OVER (
                PARTITION BY t."user_id", t."pair_id" 
                ORDER BY t."created_at" 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS "cumulative_pnl",
            MAX(SUM(t."pnl" * t."collateral_price")) OVER (
                PARTITION BY t."user_id", t."pair_id"
                ORDER BY t."created_at" 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS "max_cumulative_pnl"
        FROM
            user_pair_trades t
    ),
    
    -- Get maximum drawdown for each user-pair
    max_drawdowns AS (
        SELECT
            "user_id",
            "pair_id",
            "pair_name",
            ABS(MIN("max_cumulative_pnl" - "cumulative_pnl")) AS "max_drawdown"
        FROM
            user_pair_drawdowns
        GROUP BY
            "user_id", "pair_id", "pair_name"
    ),
    
    -- Calculate pair FDV (Fully Diluted Value) where available
    pair_fdv AS (
        SELECT
            "pair_id",
            "pair_name",
            "market_cap" * ("total_supply" / NULLIF("circulating_supply", 0)) AS "estimated_fdv"
        FROM
            "public"."trade_pool_pairs"
    )
    
    -- Combine all metrics
    SELECT
        upf."user_id",
        upf."pair_id",
        upf."pair_name",
        upf."total_trades",
        upf."winning_trades",
        upf."losing_trades",
        upf."winning_amount",
        upf."losing_amount",
        upf."profit_factor",
        upf."total_volume",
        upf."platform_revenue",
        upf."avg_position_hours",
        upf."net_pnl",
        pv."volatility_rank",
        pv."volatility_score",
        md."max_drawdown",
        -- Add FDV where available
        pf."estimated_fdv",
        -- Calculate PNL metrics
        upf."net_pnl" / NULLIF(upf."total_trades", 0) AS "pnl_per_trade",
        upf."net_pnl" / NULLIF(upf."avg_position_hours", 0) AS "pnl_per_hour",
        -- Calculate platform metrics
        upf."platform_revenue" / NULLIF(upf."total_volume", 0) AS "revenue_per_volume",
        upf."platform_revenue" / NULLIF(upf."total_trades", 0) AS "revenue_per_trade",
        -- Calculate win rate
        upf."winning_trades"::FLOAT / NULLIF(upf."total_trades", 0) AS "win_rate"
    FROM
        user_profit_factors upf
    LEFT JOIN
        pair_volatility pv ON upf."pair_id" = pv."pair_id"
    LEFT JOIN
        max_drawdowns md ON upf."user_id" = md."user_id" AND upf."pair_id" = md."pair_id"
    LEFT JOIN
        pair_fdv pf ON upf."pair_id" = pf."pair_id"
    WHERE
        upf."total_trades" >= 5 -- Minimum threshold for meaningful analysis
    """
    
    try:
        df = pd.read_sql(
            query, 
            engine,
            params={"start_time": start_time, "end_time": end_time}
        )
        return df
    except Exception as e:
        print(f"Error fetching user behavior data: {e}")
        # Return sample data for testing
        return generate_sample_user_data(100)

def generate_sample_user_data(num_users=100):
    """Generate sample user-pair data for testing"""
    np.random.seed(42)
    
    user_ids = [f"user_{i}" for i in range(1, num_users+1)]
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT", 
             "AVAX/USDT", "BNB/USDT", "XRP/USDT", "LINK/USDT", "DOT/USDT"]
    
    # Generate more rows by creating multiple pair entries per user
    user_pair_rows = []
    for user_id in user_ids:
        # Each user trades 1-5 random pairs
        num_pairs = np.random.randint(1, 6)
        user_pairs = np.random.choice(pairs, size=num_pairs, replace=False)
        
        for pair in user_pairs:
            # More trades for BTC/USDT and ETH/USDT
            trade_multiplier = 3 if pair in ["BTC/USDT", "ETH/USDT"] else 1
            
            # Generate metrics for this user-pair
            total_trades = np.random.randint(5, 100 * trade_multiplier)
            win_rate = np.random.beta(5, 5)  # Beta distribution centered around 0.5
            winning_trades = int(total_trades * win_rate)
            losing_trades = total_trades - winning_trades
            
            # Make some users consistently profitable, others not
            user_profitable = np.random.random() < 0.3  # 30% of users are profitable
            
            if user_profitable:
                # Profitable users have higher winning amounts
                avg_win = np.random.uniform(50, 200)
                avg_loss = np.random.uniform(30, 100)
            else:
                # Unprofitable users have lower winning amounts
                avg_win = np.random.uniform(30, 100)
                avg_loss = np.random.uniform(50, 200)
            
            winning_amount = winning_trades * avg_win
            losing_amount = losing_trades * avg_loss
            
            # Calculate profit factor
            profit_factor = winning_amount / losing_amount if losing_amount > 0 else (999 if winning_amount > 0 else 0)
            
            # Add volatility based on the pair
            if pair in ["PEPE/USDT", "DOGE/USDT"]:
                volatility_rank = np.random.randint(1, 4)  # Higher volatility
                volatility_score = np.random.uniform(8, 15)
            elif pair in ["SOL/USDT", "AVAX/USDT"]:
                volatility_rank = np.random.randint(3, 6)  # Medium volatility
                volatility_score = np.random.uniform(5, 8)
            else:
                volatility_rank = np.random.randint(5, 11)  # Lower volatility
                volatility_score = np.random.uniform(2, 5)
            
            # Position time varies by pair
            if pair in ["BTC/USDT", "ETH/USDT"]:
                avg_position_hours = np.random.exponential(24)  # Longer holds for blue chips
            else:
                avg_position_hours = np.random.exponential(6)  # Shorter holds for alts
            
            # More platform revenue from higher volume pairs
            total_volume = np.random.uniform(1000, 100000) * trade_multiplier
            platform_revenue = total_volume * np.random.uniform(0.001, 0.005)
            
            # Create the row
            user_pair_rows.append({
                'user_id': user_id,
                'pair_id': hash(pair) % 10000,  # Create a dummy pair_id
                'pair_name': pair,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'winning_amount': winning_amount,
                'losing_amount': losing_amount,
                'profit_factor': profit_factor,
                'total_volume': total_volume,
                'platform_revenue': platform_revenue,
                'avg_position_hours': avg_position_hours,
                'net_pnl': winning_amount - losing_amount,
                'volatility_rank': volatility_rank,
                'volatility_score': volatility_score,
                'max_drawdown': losing_amount * np.random.uniform(0.5, 2.0),
                'estimated_fdv': np.random.uniform(1e6, 1e10) if np.random.random() > 0.2 else None,
                'win_rate': win_rate
            })
    
    # Create DataFrame from rows
    df = pd.DataFrame(user_pair_rows)
    
    # Calculate derived metrics
    df['pnl_per_trade'] = df['net_pnl'] / df['total_trades']
    df['pnl_per_hour'] = df['net_pnl'] / df['avg_position_hours']
    df['revenue_per_volume'] = df['platform_revenue'] / df['total_volume']
    df['revenue_per_trade'] = df['platform_revenue'] / df['total_trades']
    
    return df

def classify_users(user_pair_df):
    """
    Classify user-pair combinations according to the behavior matrix.
    
    Classification dimensions:
    1. Profit Factor (High/Low) - most important metric
    2. Win Rate (High/Low)
    3. Trading Volume (High/Low)
    4. PnL per Time Held (High/Low)
    5. Pair Volatility (High/Low)
    """
    # Make a copy to avoid modifying the original
    df = user_pair_df.copy()
    
    # Define thresholds
    PF_THRESHOLD = 1.0  # Profit Factor threshold (< 1 means user is losing)
    HIGH_PF_THRESHOLD = 1.5  # High profit factor
    WIN_RATE_THRESHOLD = 0.6  # High win rate
    EXTREMELY_HIGH_WIN_RATE = 0.9  # Suspiciously high win rate
    
    # Calculate quantiles for relative metrics
    volume_threshold = df['total_volume'].quantile(0.7)  # Top 30% is high volume
    pnl_time_threshold = df['pnl_per_hour'].quantile(0.6)  # Top 40% is high PnL/time
    
    # Flag high volatility pairs (top 3 ranks)
    df['high_volatility'] = df['volatility_rank'] <= 3
    
    # Calculate reward-to-risk ratio
    df['reward_risk_ratio'] = df['platform_revenue'] / df['max_drawdown'].clip(0.01)
    reward_risk_threshold = df['reward_risk_ratio'].quantile(0.6)  # Top 40% is high reward/risk
    
    # Initialize classification columns
    df['pf_category'] = np.where(df['profit_factor'] < PF_THRESHOLD, 'Low', 
                              np.where(df['profit_factor'] >= HIGH_PF_THRESHOLD, 'High', 'Medium'))
                              
    df['volume_category'] = np.where(df['total_volume'] > volume_threshold, 'High', 'Low')
    df['pnl_time_category'] = np.where(df['pnl_per_hour'] > pnl_time_threshold, 'High', 'Low')
    df['win_rate_category'] = np.where(df['win_rate'] > WIN_RATE_THRESHOLD, 'High', 'Low')
    df['reward_risk_category'] = np.where(df['reward_risk_ratio'] > reward_risk_threshold, 'High', 'Low')
    
    # Apply the decision matrix logic
    def apply_user_behavior_matrix(row):
        # Low PF is what we want (users losing money)
        if row['profit_factor'] < PF_THRESHOLD:
            return "Marketing Giveaway"
        
        # High PF (users making money) need careful analysis
        if row['profit_factor'] >= PF_THRESHOLD:
            # Check reward-to-risk ratio for platform
            if row['reward_risk_category'] == 'High':
                # Check trading volume
                if row['volume_category'] == 'High':
                    return "Increase Bust Buffer"
                else:  # Low volume
                    # Check PnL/time efficiency
                    if row['pnl_time_category'] == 'High':
                        # Fast earners on low volume - check if sniper
                        if row['avg_position_hours'] < 2:  # Position held less than 2 hours
                            return "Sniper"
                        # Check for negative variance exposure
                        elif row['max_drawdown'] > row['total_volume'] * 0.3:
                            return "High Negative Variance Exposure"
                        # Check if pair could be vulnerable
                        elif row['high_volatility'] and row['estimated_fdv'] is not None and row['estimated_fdv'] < 1e8:
                            return "Delist"
                        else:
                            return "Monitor"
                    else:  # Low PnL/time
                        return "Smart Money / Good Trend Trader"
            
            # Check win rate separately
            if row['win_rate'] > EXTREMELY_HIGH_WIN_RATE:
                return "Likely Manipulation"
            elif row['win_rate'] > WIN_RATE_THRESHOLD:
                return "Study Behavior"
        
        return "Uncategorized"
    
    # Apply matrix logic
    df['user_category'] = df.apply(apply_user_behavior_matrix, axis=1)
    
    # Define actions based on category
    def determine_action(row):
        category = row['user_category']
        
        actions = {
            'Marketing Giveaway': "Encourage user with promotions and rewards",
            'Increase Bust Buffer': "Apply stricter risk controls for high volume activity",
            'Sniper': "Monitor for market timing exploitation, potential insider knowledge",
            'High Negative Variance Exposure': "Watch for large drawdowns, implement additional risk limits",
            'Delist': f"Consider removing {row['pair_name']} if high volatility, low FDV, negative trimmed mean PnL",
            'Smart Money / Good Trend Trader': "Increase Base Rate + Increase Position Multiplier + Consider Following",
            'Study Behavior': "Analyze patterns for platform intelligence",
            'Likely Manipulation': "Investigate for potential market manipulation",
            'Monitor': "Regular monitoring, no special action",
            'Uncategorized': "No specific action required"
        }
        
        return actions.get(category, "No specific action")
    
    df['recommended_action'] = df.apply(determine_action, axis=1)
    
    return df

def aggregate_user_data(classified_df):
    """Aggregate user data across all pairs"""
    # Group by user and get weighted categories
    user_aggregated = classified_df.groupby('user_id').apply(
        lambda x: pd.Series({
            'total_volume': x['total_volume'].sum(),
            'platform_revenue': x['platform_revenue'].sum(),
            'total_trades': x['total_trades'].sum(),
            'net_pnl': x['net_pnl'].sum(),
            'winning_amount': x['winning_amount'].sum(),
            'losing_amount': x['losing_amount'].sum(),
            'avg_profit_factor': (x['profit_factor'] * x['total_volume']).sum() / x['total_volume'].sum(),
            'dominant_category': x.loc[x['total_volume'].idxmax(), 'user_category'],
            'pairs_traded': len(x),
            'win_rate': x['winning_trades'].sum() / x['total_trades'].sum(),
            'high_vol_pairs': x['high_volatility'].sum(),
            'primary_pair': x.loc[x['total_volume'].idxmax(), 'pair_name']
        })
    ).reset_index()
    
    return user_aggregated

def analyze_pairs(classified_df):
    """Analyze trading pairs across all users"""
    # Group by pair
    pair_analysis = classified_df.groupby('pair_name').agg({
        'user_id': 'count',
        'total_volume': 'sum',
        'platform_revenue': 'sum',
        'net_pnl': 'sum',
        'profit_factor': 'mean',
        'volatility_rank': 'first',
        'estimated_fdv': 'first',
        'user_category': lambda x: x.value_counts().index[0]  # Most common category
    }).reset_index()
    
    pair_analysis.columns = [
        'pair_name', 'user_count', 'total_volume', 'platform_revenue', 
        'net_pnl', 'avg_profit_factor', 'volatility_rank', 'estimated_fdv',
        'dominant_category'
    ]
    
    # Calculate profitability metrics
    pair_analysis['profitable_for_platform'] = pair_analysis['platform_revenue'] > 0
    pair_analysis['revenue_per_volume'] = pair_analysis['platform_revenue'] / pair_analysis['total_volume']
    
    # Calculate risk metrics
    pair_analysis['is_high_volatility'] = pair_analysis['volatility_rank'] <= 3
    pair_analysis['is_low_fdv'] = pair_analysis['estimated_fdv'] < 1e8
    
    # Identify pairs potentially needing delisting
    risk_conditions = (
        pair_analysis['is_high_volatility'] & 
        pair_analysis['is_low_fdv'] & 
        (pair_analysis['net_pnl'] > 0)  # Users are making money on this pair
    )
    
    pair_analysis['delist_candidate'] = risk_conditions
    
    return pair_analysis

def visualize_user_matrix(classified_df, user_agg_df, pair_analysis_df):
    """Create visualizations for the user behavior matrix"""
    
    # 1. User category distribution
    category_counts = classified_df['user_category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'User-Pair Count']
    
    fig1 = px.bar(
        category_counts, 
        x='Category', 
        y='User-Pair Count',
        title='User-Pair Distribution by Category',
        color='Category',
        labels={'Category': 'User Behavior Category', 'User-Pair Count': 'Number of User-Pair Combinations'}
    )
    
    # 2. Profit Factor vs Platform Revenue scatter plot
    fig2 = px.scatter(
        classified_df,
        x='profit_factor',
        y='platform_revenue',
        color='user_category',
        size='total_volume',
        hover_data=['user_id', 'pair_name', 'total_trades', 'win_rate'],
        title='User-Pair Classification by Profit Factor and Platform Revenue',
        labels={
            'profit_factor': 'Profit Factor',
            'platform_revenue': 'Platform Revenue (USD)',
            'user_category': 'User Category'
        }
    )
    
    # Add a vertical reference line at PF=1
    fig2.add_vline(x=1, line_dash="dash", line_color="red")
    
    # 3. Platform revenue by category
    category_revenue = classified_df.groupby('user_category').agg({
        'platform_revenue': 'sum',
        'user_id': 'count'
    }).reset_index()
    
    category_revenue.columns = ['Category', 'Platform Revenue', 'User-Pair Count']
    category_revenue['Avg Revenue Per User-Pair'] = category_revenue['Platform Revenue'] / category_revenue['User-Pair Count']
    
    fig3 = px.bar(
        category_revenue,
        x='Category',
        y='Platform Revenue',
        title='Total Platform Revenue by User Category',
        color='Category',
        labels={
            'Category': 'User Behavior Category', 
            'Platform Revenue': 'Platform Revenue (USD)'
        }
    )
    
    # 4. Pair analysis visualization
    fig4 = px.scatter(
        pair_analysis_df,
        x='avg_profit_factor',
        y='revenue_per_volume',
        size='total_volume',
        color='dominant_category',
        hover_data=['pair_name', 'user_count', 'platform_revenue', 'is_high_volatility', 'delist_candidate'],
        title='Trading Pair Analysis',
        labels={
            'avg_profit_factor': 'Average Profit Factor',
            'revenue_per_volume': 'Platform Revenue per Volume',
            'dominant_category': 'Dominant User Category'
        }
    )
    
    # Highlight delist candidates
    delist_candidates = pair_analysis_df[pair_analysis_df['delist_candidate']]
    if not delist_candidates.empty:
        fig4.add_trace(
            go.Scatter(
                x=delist_candidates['avg_profit_factor'],
                y=delist_candidates['revenue_per_volume'],
                mode='markers',
                marker=dict(
                    size=15,
                    line=dict(width=2, color='red'),
                    symbol='circle-open'
                ),
                name='Potential Delist Candidates',
                hoverinfo='text',
                hovertext=delist_candidates['pair_name']
            )
        )
    
    return fig1, fig2, fig3, fig4

def implement_user_behavior_matrix(engine, time_period=7):
    """
    Main function to implement the user behavior matrix.
    
    Args:
        engine: SQLAlchemy engine for database connection
        time_period: Number of days to analyze (default: 7 days for weekly analysis)
    
    Returns:
        Classified user DataFrame, aggregated user data, pair analysis, and visualizations
    """
    # Get time boundaries
    now_utc = datetime.now(pytz.utc)
    start_time = now_utc - timedelta(days=time_period)
    
    # Fetch user-pair data
    user_pair_df = fetch_user_behavior_data(engine, start_time, now_utc)
    
    # Classify users
    classified_df = classify_users(user_pair_df)
    
    # Aggregate user data across pairs
    user_aggregated = aggregate_user_data(classified_df)
    
    # Analyze trading pairs
    pair_analysis = analyze_pairs(classified_df)
    
    # Create visualizations
    visualizations = visualize_user_matrix(classified_df, user_aggregated, pair_analysis)
    
    return classified_df, user_aggregated, pair_analysis, visualizations

# Example Streamlit integration
def add_user_behavior_matrix_to_dashboard():
    st.title("User Behavior Decision Matrix")
    st.markdown("""
    This dashboard analyzes user trading behavior on a weekly basis to identify:
    - Users who are profitable vs. unprofitable (Profit Factor analysis)
    - Trading patterns and behaviors requiring attention
    - Pairs that may need risk adjustments or delisting
    """)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["User-Pair Matrix", "User Analysis", "Pair Analysis"])
    
    # Get the data
    engine = create_engine("postgresql://user:password@host:port/database")  # Replace with your connection
    classified_df, user_agg, pair_analysis, visualizations = implement_user_behavior_matrix(engine)
    
    with tab1:
        # Display the matrix visualization
        st.plotly_chart(visualizations[0], use_container_width=True)
        st.plotly_chart(visualizations[1], use_container_width=True)
        
        # Show the detailed classification table
        st.subheader("User-Pair Classification Details")
        st.dataframe(
            classified_df[['user_id', 'pair_name', 'profit_factor', 'total_volume', 
                          'platform_revenue', 'user_category', 'recommended_action']]
        )
    
    with tab2:
        # User aggregated view
        st.subheader("User Aggregated Analysis")
        st.plotly_chart(visualizations[2], use_container_width=True)
        
        # Display the user aggregated data
        st.dataframe(
            user_agg[['user_id', 'total_volume', 'platform_revenue', 'avg_profit_factor', 
                      'net_pnl', 'dominant_category', 'pairs_traded', 'primary_pair']]
        )
        
        # Show top profitable and unprofitable users
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Profitable Users (High PF)")
            profitable_users = user_agg.sort_values('avg_profit_factor', ascending=False).head(10)
            st.dataframe(
                profitable_users[['user_id', 'avg_profit_factor', 'platform_revenue', 'net_pnl']]
            )
        
        with col2:
            st.subheader("Best Platform Revenue Users (Low PF)")
            revenue_users = user_agg[user_agg['avg_profit_factor'] < 1].sort_values('platform_revenue', ascending=False).head(10)
            st.dataframe(
                revenue_users[['user_id', 'avg_profit_factor', 'platform_revenue', 'net_pnl']]
            )
    
    with tab3:
        # Pair analysis
        st.subheader("Trading Pair Analysis")
        st.plotly_chart(visualizations[3], use_container_width=True)
        
        # Highlight potential delist candidates
        st.subheader("Potential Delist Candidates")
        delist_pairs = pair_analysis[pair_analysis['delist_candidate']].sort_values('total_volume', ascending=False)
        
        if len(delist_pairs) > 0:
            st.dataframe(
                delist_pairs[['pair_name', 'avg_profit_factor', 'volatility_rank', 
                             'estimated_fdv', 'net_pnl', 'platform_revenue']]
            )
        else:
            st.info("No pairs currently meet the criteria for delisting.")
        
        # Show highest volume pairs
        st.subheader("Top Pairs by Trading Volume")
        volume_pairs[['pair_name', 'total_volume', 'platform_revenue', 
                         'avg_profit_factor', 'user_count', 'dominant_category']]
        )
        
        # Show most profitable pairs for users
        st.subheader("Most Profitable Pairs for Users")
        user_profitable_pairs = pair_analysis.sort_values('avg_profit_factor', ascending=False).head(10)
        st.dataframe(
            user_profitable_pairs[['pair_name', 'avg_profit_factor', 'net_pnl', 
                                  'platform_revenue', 'user_count']]
        )
        
        # Show most profitable pairs for platform
        st.subheader("Most Profitable Pairs for Platform")
        platform_profitable_pairs = pair_analysis.sort_values('platform_revenue', ascending=False).head(10)
        st.dataframe(
            platform_profitable_pairs[['pair_name', 'platform_revenue', 'total_volume', 
                                     'avg_profit_factor', 'user_count']]
        )

def generate_weekly_user_behavior_report(engine, email_recipients=None):
    """
    Generate and distribute a weekly user behavior report
    
    Args:
        engine: SQLAlchemy engine for database connection
        email_recipients: List of email addresses to send the report to
    """
    # Get the data
    classified_df, user_agg, pair_analysis, _ = implement_user_behavior_matrix(engine)
    
    # Build report sections
    
    # 1. Executive Summary
    total_users = user_agg['user_id'].nunique()
    total_pairs = pair_analysis['pair_name'].nunique()
    total_volume = user_agg['total_volume'].sum()
    total_revenue = user_agg['platform_revenue'].sum()
    
    profitable_users = user_agg[user_agg['avg_profit_factor'] >= 1]['user_id'].nunique()
    unprofitable_users = total_users - profitable_users
    
    platform_profit_pct = unprofitable_users / total_users * 100 if total_users > 0 else 0
    
    # 2. User Categories Breakdown
    category_counts = classified_df['user_category'].value_counts()
    
    # 3. Action Items
    high_risk_users = user_agg[
        (user_agg['avg_profit_factor'] > 1.5) & 
        (user_agg['total_volume'] > user_agg['total_volume'].quantile(0.7))
    ]
    
    delist_candidates = pair_analysis[pair_analysis['delist_candidate']]
    
    # 4. User Behavior Trends
    # (Would need historical data for full implementation)
    
    # Create the report
    report = f"""
    ## Weekly User Behavior Analysis Report
    
    ### Executive Summary
    - Total Users Analyzed: {total_users}
    - Trading Pairs Analyzed: {total_pairs}
    - Total Trading Volume: ${total_volume:,.2f}
    - Total Platform Revenue: ${total_revenue:,.2f}
    - Profitable Users: {profitable_users} ({profitable_users/total_users*100:.1f}%)
    - Unprofitable Users: {unprofitable_users} ({unprofitable_users/total_users*100:.1f}%)
    - Platform Profit Percentage: {platform_profit_pct:.1f}%
    
    ### User Categories Breakdown
    {category_counts.to_string()}
    
    ### Action Items
    
    #### High Risk Users (High PF, High Volume)
    {high_risk_users[['user_id', 'avg_profit_factor', 'total_volume', 'platform_revenue']].to_string(index=False) if not high_risk_users.empty else "None identified"}
    
    #### Delist Candidates
    {delist_candidates[['pair_name', 'avg_profit_factor', 'volatility_rank', 'estimated_fdv']].to_string(index=False) if not delist_candidates.empty else "None identified"}
    
    ### Top Revenue Users (Low PF)
    {user_agg[user_agg['avg_profit_factor'] < 1].sort_values('platform_revenue', ascending=False).head(5)[['user_id', 'platform_revenue', 'avg_profit_factor']].to_string(index=False)}
    
    ### Top Volume Pairs
    {pair_analysis.sort_values('total_volume', ascending=False).head(5)[['pair_name', 'total_volume', 'platform_revenue']].to_string(index=False)}
    """
    
    print(report)
    
    # Send email if recipients are provided
    if email_recipients:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            
            # Email setup would go here
            pass
        except Exception as e:
            print(f"Error sending email: {e}")
    
    return report

# For standalone testing
if __name__ == "__main__":
    # Create sample engine
    engine = create_engine("sqlite:///:memory:")
    
    # Run the analysis
    classified_df, user_agg, pair_analysis, visualizations = implement_user_behavior_matrix(engine)
    
    # Print summary
    print("User Categories:")
    print(classified_df['user_category'].value_counts())
    
    print("\nTop 5 Pairs by Volume:")
    print(pair_analysis.sort_values('total_volume', ascending=False).head(5)[['pair_name', 'total_volume', 'platform_revenue']])
    
    # Generate report
    generate_weekly_user_behavior_report(engine)