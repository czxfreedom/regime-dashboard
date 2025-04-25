import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import os
import io
import base64

# Page configuration - absolute minimum for speed
st.set_page_config(
    page_title="Depth Tier Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for speed
)

# Ultra-minimal CSS for maximum speed with prominent tabs
st.markdown("""
<style>
    .block-container {padding: 0 !important;}
    .main .block-container {max-width: 98% !important;}
    h1, h2, h3 {margin: 0 !important; padding: 0 !important;}
    .stButton > button {width: 100%;}
    div.stProgress > div > div {height: 5px !important;}
    div.row-widget.stRadio > div {flex-direction: row;}
    div.stDataFrame {margin: 0 !important; padding: 0 !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1oe6o3n {padding-top: 0 !important;}
    .css-18e3th9 {padding-top: 0 !important;}
    
    /* Make tabs much bigger and more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        color: #000000;
        font-size: 18px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Connection pool for better performance and reliability
@st.cache_resource
def get_connection_pool():
    try:
        pool = ThreadedConnectionPool(
            5, 20,  # min_conn, max_conn
            host="aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com",
            port=5432,
            database="report_dev",
            user="public_rw",
            password="aTJ92^kl04hllk"
        )
        return pool
    except Exception as e:
        st.error(f"Error creating connection pool: {e}")
        return None

# Get a connection from the pool
def get_conn():
    pool = get_connection_pool()
    if pool:
        return pool.getconn()
    return None

# Return a connection to the pool
def release_conn(conn):
    pool = get_connection_pool()
    if pool and conn:
        pool.putconn(conn)

# Pre-defined pairs as a fast fallback
PREDEFINED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "AVAX/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "DOT/USDT"
]

# Fast version of the depth tier analyzer
class FastDepthTierAnalyzer:
    def __init__(self):
        self.point_counts = [500, 5000, 10000, 50000]
        
        # Define depth tiers
        self.depth_tier_columns = [
            'price_1', 'price_2', 'price_3', 'price_4', 'price_5', 
            'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
            'price_11', 'price_12', 'price_13', 'price_14', 'price_15'
        ]
        
        # Map column names to actual depth values
        self.depth_tier_values = {
            'price_1': '10k',
            'price_2': '50k',
            'price_3': '100k',
            'price_4': '200k',
            'price_5': '300k',
            'price_6': '400k',
            'price_7': '500k',
            'price_8': '600k',
            'price_9': '700k',
            'price_10': '800k',
            'price_11': '900k',
            'price_12': '1000k',
            'price_13': '2000k',
            'price_14': '3000k',
            'price_15': '4000k'
        }
        
        # Metrics to calculate
        self.metrics = [
            'direction_changes',   # Frequency of price direction reversals (%)
            'choppiness',          # Measures price oscillation within a range
            'tick_atr_pct',        # ATR % 
            'trend_strength'       # Measures directional strength
        ]
        
        # Display names for metrics
        self.metric_display_names = {
            'direction_changes': 'Direction Changes (%)',
            'choppiness': 'Choppiness',
            'tick_atr_pct': 'Tick ATR %',
            'trend_strength': 'Trend Strength'
        }
        
        # What makes a depth tier "better" for each metric
        self.metric_desired_direction = {
            'direction_changes': 'higher',  # Higher direction changes is better
            'choppiness': 'higher',         # Higher choppiness is better
            'tick_atr_pct': 'higher',       # Higher ATR % is better
            'trend_strength': 'lower'       # Lower trend strength is better
        }
        
        # Weights for overall score
        self.metric_weights = {
            'direction_changes': 0.25,
            'choppiness': 0.25,
            'tick_atr_pct': 0.25,
            'trend_strength': 0.25
        }
        
        # Store results
        self.results = {point: None for point in self.point_counts}
    
    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None):
        """Optimized fetch and analyze for maximum speed"""
        try:
            # First, get all data at once with a single query
            conn = get_conn()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get current day's partition table (most likely to have data)
            table_date = datetime.now().strftime("%Y%m%d")
            table_name = f"oracle_order_book_level_price_data_partition_v3_{table_date}"
            
            try:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table_name,))
                
                if not cursor.fetchone()[0]:
                    # Try yesterday if today doesn't exist
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                    table_name = f"oracle_order_book_level_price_data_partition_v3_{yesterday}"
                
                # Fetch all data at once with a single query for the max point count
                max_points = max(self.point_counts)
                
                # Format time strings for query
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                
                query = f"""
                    SELECT
                        pair_name,
                        {', '.join(self.depth_tier_columns)}
                    FROM
                        public.{table_name}
                    WHERE
                        pair_name = %s
                        AND created_at >= %s
                    ORDER BY created_at DESC
                    LIMIT {max_points + 1000}
                """
                
                # Execute query with parameters to prevent SQL injection
                cursor.execute(query, (pair_name, start_str))
                
                # Fetch all rows and create DataFrame
                columns = ['pair_name'] + self.depth_tier_columns
                all_data = cursor.fetchall()
                
                if not all_data or len(all_data) < min(self.point_counts):
                    cursor.close()
                    release_conn(conn)
                    return False
                
                # Convert to DataFrame for faster processing
                all_df = pd.DataFrame(all_data, columns=columns)
                
                # Close cursor and connection
                cursor.close()
                release_conn(conn)
                
                # Process each point count using the pre-fetched data
                # This avoids multiple database connections
                for i, point_count in enumerate(self.point_counts):
                    if progress_bar:
                        progress_bar.progress((i / len(self.point_counts)) * 0.9 + 0.1, 
                                          text=f"Processing {point_count} points...")
                    
                    if len(all_df) >= point_count:
                        df = all_df.iloc[:point_count].copy()
                        
                        # Process each depth tier in parallel
                        tier_results = {}
                        
                        # Process in a batch for speed
                        for column in self.depth_tier_columns:
                            metrics = self._calculate_metrics(df, column, point_count)
                            if metrics:
                                tier = self.depth_tier_values[column]
                                tier_results[tier] = metrics
                        
                        # Calculate scores and ranking
                        self.results[point_count] = self._calculate_scores(tier_results)
                
                if progress_bar:
                    progress_bar.progress(1.0, text="Analysis complete!")
                    
                # Check if we got any results
                has_results = False
                for pc in self.point_counts:
                    if self.results[pc] is not None:
                        has_results = True
                        break
                        
                return has_results
                
            except Exception as e:
                st.error(f"Database query error: {e}")
                if cursor:
                    cursor.close()
                release_conn(conn)
                return False
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return False
    
    def _calculate_metrics(self, df, price_col, point_count):
        """Optimized calculation of metrics"""
        try:
            # Convert to numeric 
            prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
            
            if len(prices) < point_count * 0.8:  # Allow some flexibility for missing data
                return None
                
            # Calculate metrics with vectorized operations for speed
            mean_price = prices.mean()
            
            # Direction changes (vectorized)
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0
            
            # Smaller window for faster calculation
            window = min(10, max(2, point_count // 50))
            
            # Choppiness index (vectorized)
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
            choppiness = (100 * sum_abs_changes / (price_range + 1e-10)).mean()
            
            # Tick ATR (vectorized)
            tick_atr = price_changes.abs().mean()
            tick_atr_pct = (tick_atr / mean_price) * 100
            
            # Trend strength (vectorized)
            net_change = (prices - prices.shift(window)).abs()
            trend_strength = (net_change / (sum_abs_changes + 1e-10)).dropna().mean()
            
            return {
                'direction_changes': direction_change_pct,
                'choppiness': min(choppiness, 1000),  # Cap extreme values
                'tick_atr_pct': tick_atr_pct,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            return None
    
    def _calculate_scores(self, tier_results):
        """Calculate scores and rankings for all tiers (optimized)"""
        if not tier_results:
            return None
            
        # Create DataFrame directly (faster)
        data = []
        for tier, metrics in tier_results.items():
            row = {'Tier': tier}
            row.update(metrics)
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Vectorized scoring for all metrics at once
        for metric in self.metrics:
            display_name = self.metric_display_names[metric]
            
            if metric in df.columns and not df[metric].isna().all():
                min_val = df[metric].min()
                max_val = df[metric].max()
                
                if min_val == max_val:
                    df[f'{display_name} Score'] = 100
                    continue
                
                if self.metric_desired_direction[metric] == 'higher':
                    df[f'{display_name} Score'] = ((df[metric] - min_val) / (max_val - min_val)) * 100
                else:
                    df[f'{display_name} Score'] = ((max_val - df[metric]) / (max_val - min_val)) * 100
        
        # Calculate overall score (vectorized)
        score_columns = [f'{self.metric_display_names[m]} Score' for m in self.metrics 
                        if f'{self.metric_display_names[m]} Score' in df.columns]
        
        if score_columns:
            weighted_sum = pd.Series(0, index=df.index)
            total_weight = 0
            
            for metric in self.metrics:
                display_name = self.metric_display_names[metric]
                score_col = f'{display_name} Score'
                
                if score_col in df.columns:
                    weight = self.metric_weights[metric]
                    weighted_sum += df[score_col] * weight
                    total_weight += weight
            
            if total_weight > 0:
                df['Overall Score'] = weighted_sum / total_weight
            
            # Sort and rank
            df = df.sort_values('Overall Score', ascending=False)
            df.insert(0, 'Rank', range(1, len(df) + 1))
            
        return df

# Balanced table and chart creation
def create_point_count_analysis(analyzer, point_count):
    """Creates tables and charts with balanced sizes"""
    if analyzer.results[point_count] is None:
        st.info(f"No data available for {point_count} points analysis.")
        return
    
    df = analyzer.results[point_count]
    
    # Create two columns with balanced sizes
    col1, col2 = st.columns([2, 1])  # Give the table more space
    
    with col1:  # Table column - now wider
        # Show full rankings table
        st.markdown(f"### Rankings Table")
        
        # Full results with all metrics
        display_df = df.copy()
        # Format numeric columns for display
        for col in display_df.columns:
            if col not in ['Rank', 'Tier']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
                )
        
        # Use bigger text for the table
        st.markdown("""
        <style>
        .dataframe {
            font-size: 16px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Show all tiers with formatted numbers
        st.dataframe(display_df, use_container_width=True, height=400)
    
    with col2:  # Chart column
        # Create compact chart
        fig, ax = plt.subplots(figsize=(5, 7))  # Taller, narrower chart
        
        # Extract data - show more tiers
        max_tiers = min(10, len(df))
        tiers = df['Tier'].iloc[:max_tiers]
        scores = df['Overall Score'].iloc[:max_tiers]
        
        # Plot horizontal bars with clearer colors
        y_pos = range(len(tiers))
        
        # Define colors based on score
        colors = []
        for score in scores:
            if score >= 90:
                colors.append('#2ecc71')  # Green
            elif score >= 75:
                colors.append('#f1c40f')  # Yellow
            elif score >= 60:
                colors.append('#e67e22')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        bars = ax.barh(y_pos, scores, color=colors)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{scores.iloc[i]:.1f}', va='center', fontsize=12)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tiers, fontsize=14)  # Larger font for tier labels
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(f'Top Depth Tiers', fontsize=16)
        ax.set_xlim(0, 105)
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
    
    # Add recommendations below
    st.markdown(f"### Recommendation")
    st.markdown(f"**Best depth tier for {point_count} points**: **{df.iloc[0]['Tier']}** with score {df.iloc[0]['Overall Score']:.1f}")

def main():
    # Main layout - super streamlined
    st.markdown("<h1 style='text-align: center; font-size:24px;'>Liquidity Depth Tier Analyzer</h1>", unsafe_allow_html=True)
    
    # Main selection area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        selected_pair = st.selectbox(
            "Select Pair",
            PREDEFINED_PAIRS,
            index=0
        )
    
    with col2:
        run_analysis = st.button("ANALYZE", use_container_width=True)
    
    # Main content
    if run_analysis and selected_pair:
        # Set up tabs for results - removed summary tab, made tabs bigger
        tabs = st.tabs(["500 POINTS", "5,000 POINTS", "10,000 POINTS", "50,000 POINTS"])
        
        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Initialize analyzer and run analysis
        analyzer = FastDepthTierAnalyzer()
        success = analyzer.fetch_and_analyze(selected_pair, 24, progress_bar)
        
        if success:
            # Display detailed results for each point count
            with tabs[0]:
                create_point_count_analysis(analyzer, 500)
            
            with tabs[1]:
                create_point_count_analysis(analyzer, 5000)
            
            with tabs[2]:
                create_point_count_analysis(analyzer, 10000)
                
            with tabs[3]:
                create_point_count_analysis(analyzer, 50000)
                
        else:
            progress_bar.empty()
            st.error(f"Failed to analyze {selected_pair}. Please try another pair.")
            
    else:
        # Minimal welcome message
        st.info("Select a pair and click ANALYZE to find the optimal depth tier.")

if __name__ == "__main__":
    main()