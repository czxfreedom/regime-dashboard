import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import psycopg2
import os
import io
import base64
from concurrent.futures import ThreadPoolExecutor

# Page configuration with minimal elements
st.set_page_config(
    page_title="Depth Tier Analyzer",
    page_icon="📊",
    layout="wide"
)

# Apply minimal CSS
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .main .block-container {max-width: 95%;}
    h1, h2, h3 {margin-top: 0.2rem; margin-bottom: 0.2rem;}
    .stButton > button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(
            host="aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com",
            port=5432,
            database="report_dev",
            user="public_rw",
            password="aTJ92^kl04hllk"
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Get pairs from database (cached)
@st.cache_data(ttl=3600)
def fetch_pairs():
    try:
        conn = init_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT pair_name
            FROM public.trade_pool_pairs
            WHERE status = 1
            ORDER BY pair_name;
        """)
        
        pairs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        return pairs
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return []

class FastDepthTierAnalyzer:
    """Simplified and faster version of the depth tier analyzer"""
    
    def __init__(self):
        self.point_counts = [500, 5000, 10000]
        
        # Define depth tiers by their column names and actual depth values
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
            'tick_atr_pct',        # ATR % (Average True Range as percentage of mean price)
            'trend_strength'       # Measures directional strength
        ]
        
        # Display names for metrics (for printing)
        self.metric_display_names = {
            'direction_changes': 'Direction Changes (%)',
            'choppiness': 'Choppiness',
            'tick_atr_pct': 'Tick ATR %',
            'trend_strength': 'Trend Strength'
        }
        
        # Define what makes a depth tier "better" for each metric
        self.metric_desired_direction = {
            'direction_changes': 'higher',  # Higher direction changes is better
            'choppiness': 'higher',         # Higher choppiness is better
            'tick_atr_pct': 'higher',       # Higher ATR % is better
            'trend_strength': 'lower'       # Lower trend strength is better
        }
        
        # Weights for combining metrics into an overall score
        self.metric_weights = {
            'direction_changes': 0.25,
            'choppiness': 0.25,
            'tick_atr_pct': 0.25,
            'trend_strength': 0.25
        }
        
        # Store results
        self.results = {point: None for point in self.point_counts}
    
    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None):
        """Fast version of fetch and analyze that only gets essential data"""
        try:
            conn = init_connection()
            if not conn:
                return False
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Convert to formatted strings
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get list of partition tables to query
            cursor = conn.cursor()
            
            # Only check for tables in the relevant date range (faster)
            current_date = start_time.date()
            end_date = end_time.date()
            table_dates = []
            
            while current_date <= end_date:
                table_dates.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)
            
            # Get existing tables more efficiently
            table_name_list = ", ".join([f"'oracle_order_book_level_price_data_partition_v3_{date}'" for date in table_dates])
            
            cursor.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ({table_name_list})
            """)
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if not existing_tables:
                return False
            
            if progress_bar:
                progress_bar.progress(0.1, text="Found data tables, fetching price data...")
            
            # Build query to fetch data
            query_parts = []
            
            for table in existing_tables:
                query_parts.append(f"""
                    SELECT
                        pair_name,
                        (created_at + INTERVAL '8 hour') AS utc8,
                        {', '.join(self.depth_tier_columns)}
                    FROM
                        public.{table}
                    WHERE
                        pair_name = '{pair_name}'
                        AND created_at >= '{start_str}'::timestamp
                        AND created_at <= '{end_str}'::timestamp
                """)
            
            # Combine queries and add LIMIT for faster results
            query = " UNION ALL ".join(query_parts)
            query += " ORDER BY utc8 DESC"
            
            # Process for each point count with different limits to speed up
            with ThreadPoolExecutor() as executor:
                futures = {}
                
                for point_count in self.point_counts:
                    # Create point-specific query with appropriate limit
                    point_query = f"{query} LIMIT {point_count * 2}"  # Get double to account for NAs
                    
                    # Start asynchronous processing for each point count
                    futures[point_count] = executor.submit(
                        self._process_point_count, conn, point_query, point_count, pair_name
                    )
                
                # Update progress as each future completes
                completed = 0
                total = len(futures)
                
                for point_count, future in futures.items():
                    self.results[point_count] = future.result()
                    completed += 1
                    if progress_bar:
                        progress_bar.progress(0.1 + (0.9 * completed / total), 
                                             text=f"Processed {point_count} points ({completed}/{total})")
            
            cursor.close()
            return True
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return False
    
    def _process_point_count(self, conn, query, point_count, pair_name):
        """Process a specific point count (for parallel execution)"""
        try:
            # Fetch data for this point count
            df = pd.read_sql_query(query, conn)
            
            if len(df) < point_count:
                return None
            
            # Process each depth tier
            tier_results = {}
            
            for column in self.depth_tier_columns:
                if column in df.columns and not df[column].isna().all():
                    # Calculate metrics for this tier
                    metrics = self._calculate_metrics(df, column, point_count)
                    if metrics:
                        tier = self.depth_tier_values[column]
                        tier_results[tier] = metrics
            
            # Calculate scores and ranking
            return self._calculate_scores(tier_results)
        
        except Exception as e:
            st.error(f"Error processing {point_count} points: {e}")
            return None
    
    def _calculate_metrics(self, df, price_col, point_count):
        """Calculate all metrics for a specific tier and point count"""
        try:
            # Convert to numeric and get only the needed points
            prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
            
            if len(prices) < point_count:
                return None
                
            # Take the last n points for analysis
            prices = prices.iloc[-point_count:]
            
            # Calculate metrics more efficiently
            mean_price = prices.mean()
            
            # Direction changes
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0
            
            # Choppiness - use smaller window for faster calculation
            window = min(20, point_count // 10)
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
            choppiness = (100 * sum_abs_changes / (price_range + 1e-10)).mean()
            
            # Tick ATR
            tick_atr = price_changes.abs().mean()
            tick_atr_pct = (tick_atr / mean_price) * 100
            
            # Trend strength
            net_change = (prices - prices.shift(window)).abs()
            trend_strength = (net_change / (sum_abs_changes + 1e-10)).dropna().mean()
            
            return {
                'direction_changes': direction_change_pct,
                'choppiness': min(choppiness, 1000),  # Cap extreme values
                'tick_atr_pct': tick_atr_pct,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            return None
    
    def _calculate_scores(self, tier_results):
        """Calculate scores and rankings for all tiers"""
        if not tier_results:
            return None
            
        # Create a DataFrame for easier processing
        data = []
        
        for tier, metrics in tier_results.items():
            row = {'Tier': tier}
            row.update(metrics)
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # For each metric, calculate min, max, and normalized scores
        for metric in self.metrics:
            display_name = self.metric_display_names[metric]
            
            if metric in df.columns and not df[metric].isna().all():
                # Get min and max values
                min_val = df[metric].min()
                max_val = df[metric].max()
                
                # Skip if min equals max (no variation)
                if min_val == max_val:
                    df[f'{display_name} Score'] = 100  # Give full score to all
                    continue
                
                # Calculate normalized score (0-100)
                if self.metric_desired_direction[metric] == 'higher':
                    # Higher is better: normalize to 0-100 where 100 is best
                    df[f'{display_name} Score'] = ((df[metric] - min_val) / 
                                                (max_val - min_val)) * 100
                else:
                    # Lower is better: normalize to 0-100 where 100 is best
                    df[f'{display_name} Score'] = ((max_val - df[metric]) / 
                                                (max_val - min_val)) * 100
        
        # Calculate overall score based on weighted average of metric scores
        score_columns = [f'{self.metric_display_names[m]} Score' for m in self.metrics 
                        if f'{self.metric_display_names[m]} Score' in df.columns]
        
        if score_columns:
            # Calculate weighted average of all available score columns
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
            
            # Sort by overall score (descending) and add rank
            df = df.sort_values('Overall Score', ascending=False)
            df.insert(0, 'Rank', range(1, len(df) + 1))
            
        return df

# Functions to create downloadable content
def get_image_download_link(fig, filename):
    """Creates a download link for a matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Chart</a>'
    return href

def create_tables_for_point_count(analyzer, point_count):
    """Creates tables and charts for a specific point count"""
    if analyzer.results[point_count] is None:
        st.info(f"No data available for {point_count} points analysis.")
        return
    
    df = analyzer.results[point_count]
    
    # Create summary table
    summary_df = df.copy()
    # Format numeric columns for display
    for col in summary_df.columns:
        if col not in ['Rank', 'Tier']:
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
    
    # Show top 5 tiers
    st.dataframe(summary_df.head(5), use_container_width=True)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract data for chart
    tiers = df['Tier'].iloc[:10]  # Show top 10 for clarity
    scores = df['Overall Score'].iloc[:10]
    
    # Plot horizontal bars
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
                f'{scores.iloc[i]:.1f}', va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tiers)
    ax.set_xlabel('Overall Score (Higher is Better)')
    ax.set_title(f'Top Depth Tiers - {point_count} Points')
    ax.set_xlim(0, 105)  # Leave space for score labels
    ax.grid(axis='x', alpha=0.3)
    
    # Display the chart
    st.pyplot(fig)
    
    # Add download link
    st.markdown(get_image_download_link(fig, f"depth_tier_{point_count}pts.png"), unsafe_allow_html=True)

def create_combined_results_chart(analyzer):
    """Creates a chart showing the top tiers across different point counts"""
    # Check if we have results
    has_results = False
    for point_count in analyzer.point_counts:
        if analyzer.results[point_count] is not None:
            has_results = True
            break
    
    if not has_results:
        return
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25  # Width of bars
    indices = np.arange(3)  # Three categories: 500, 5000, 10000 points
    
    # Get top 3 tiers for each point count
    top_tiers = {}
    
    for i, point_count in enumerate(analyzer.point_counts):
        if analyzer.results[point_count] is not None:
            df = analyzer.results[point_count]
            top_tiers[point_count] = []
            
            for j in range(min(3, len(df))):
                if j < len(df):
                    tier = df.iloc[j]['Tier']
                    score = df.iloc[j]['Overall Score']
                    top_tiers[point_count].append((tier, score))
                else:
                    top_tiers[point_count].append(("N/A", 0))
    
    # Plot bars for each rank
    for rank in range(3):
        scores = []
        labels = []
        
        for point_count in analyzer.point_counts:
            if point_count in top_tiers and rank < len(top_tiers[point_count]):
                tier, score = top_tiers[point_count][rank]
                scores.append(score)
                labels.append(tier)
            else:
                scores.append(0)
                labels.append("N/A")
        
        bars = ax.bar(indices + (width * rank), scores, width, 
                     label=f'Rank {rank+1}')
        
        # Add tier labels on top of bars
        for i, bar in enumerate(bars):
            if scores[i] > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        labels[i], ha='center', va='bottom', rotation=0,
                        fontsize=9)
    
    # Set axis labels and title
    ax.set_ylabel('Score')
    ax.set_title('Top 3 Depth Tiers by Point Count')
    ax.set_xticks(indices + width)
    ax.set_xticklabels([f'{pc} Points' for pc in analyzer.point_counts])
    ax.set_ylim(0, 105)  # Leave space for tier labels
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def main():
    st.title("Liquidity Depth Tier Analyzer")
    
    # Sidebar settings - keep minimal for speed
    with st.sidebar:
        st.header("Settings")
        
        # Get pair options
        pairs = fetch_pairs()
        major_pairs = [p for p in pairs if any(m in p for m in ["BTC", "ETH", "SOL", "BNB", "XRP"])]
        
        if major_pairs:
            # Default to BTC/USDT if available
            default_idx = major_pairs.index("BTC/USDT") if "BTC/USDT" in major_pairs else 0
            pair_to_analyze = st.selectbox(
                "Select Pair",
                major_pairs,
                index=min(default_idx, len(major_pairs)-1)
            )
        else:
            pair_to_analyze = st.text_input("Enter Pair (e.g., BTC/USDT)")
        
        # Simple time option
        hours = st.select_slider(
            "Analysis Timeframe",
            options=[1, 2, 6, 12, 24, 48, 72],
            value=24
        )
        
        run_analysis = st.button("Run Analysis", use_container_width=True)
    
    # Main content
    if run_analysis and pair_to_analyze:
        # Set up tabs for results
        tabs = st.tabs(["Summary", "500 Points", "5000 Points", "10000 Points"])
        
        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Initialize analyzer and run analysis
        analyzer = FastDepthTierAnalyzer()
        success = analyzer.fetch_and_analyze(pair_to_analyze, hours, progress_bar)
        
        if success:
            progress_bar.progress(1.0, text="Analysis complete!")
            
            # Show summary in first tab
            with tabs[0]:
                st.header(f"Analysis Summary: {pair_to_analyze}")
                st.markdown(f"**Timeframe:** Last {hours} hours")
                
                # Create side-by-side metrics for the top tier in each point count
                col1, col2, col3 = st.columns(3)
                
                # Get best tiers for each point count
                best_tiers = {}
                for point_count in analyzer.point_counts:
                    if analyzer.results[point_count] is not None and not analyzer.results[point_count].empty:
                        best_tier = analyzer.results[point_count].iloc[0]['Tier']
                        best_score = analyzer.results[point_count].iloc[0]['Overall Score']
                        best_tiers[point_count] = (best_tier, best_score)
                
                # Display metrics
                with col1:
                    if 500 in best_tiers:
                        tier, score = best_tiers[500]
                        st.metric("Best Tier (500 pts)", tier, f"Score: {score:.1f}")
                    else:
                        st.metric("Best Tier (500 pts)", "N/A", "No data")
                
                with col2:
                    if 5000 in best_tiers:
                        tier, score = best_tiers[5000]
                        st.metric("Best Tier (5000 pts)", tier, f"Score: {score:.1f}")
                    else:
                        st.metric("Best Tier (5000 pts)", "N/A", "No data")
                
                with col3:
                    if 10000 in best_tiers:
                        tier, score = best_tiers[10000]
                        st.metric("Best Tier (10000 pts)", tier, f"Score: {score:.1f}")
                    else:
                        st.metric("Best Tier (10000 pts)", "N/A", "No data")
                
                # Show combined chart
                st.subheader("Comparison Across Time Frames")
                combined_fig = create_combined_results_chart(analyzer)
                if combined_fig:
                    st.pyplot(combined_fig)
                    
                # Add key insights
                st.subheader("Key Insights")
                
                # Generate some insights based on the results
                insights = []
                
                # Check if the best tier is consistent across time frames
                if len(best_tiers) > 1:
                    best_tier_values = [t[0] for t in best_tiers.values()]
                    if len(set(best_tier_values)) == 1:
                        insights.append(f"The {best_tier_values[0]} depth tier is consistently optimal across all analyzed time frames.")
                    else:
                        # Look for patterns like lower is better for short term
                        try:
                            if 500 in best_tiers and 10000 in best_tiers:
                                short_tier = int(best_tiers[500][0].replace('k', ''))
                                long_tier = int(best_tiers[10000][0].replace('k', ''))
                                
                                if short_tier < long_tier:
                                    insights.append(f"Lower depths ({short_tier}k) perform better for short-term trading, while higher depths ({long_tier}k) are better for longer-term trading.")
                                elif short_tier > long_tier:
                                    insights.append(f"Higher depths ({short_tier}k) perform better for short-term trading, while lower depths ({long_tier}k) are better for longer-term trading.")
                        except:
                            pass
                
                # Check if there's a generally optimal tier
                if best_tiers:
                    avg_scores = {}
                    for point_count, (tier, score) in best_tiers.items():
                        if tier not in avg_scores:
                            avg_scores[tier] = []
                        avg_scores[tier].append(score)
                    
                    avg_scores = {tier: sum(scores)/len(scores) for tier, scores in avg_scores.items()}
                    best_overall = max(avg_scores.items(), key=lambda x: x[1])
                    
                    insights.append(f"The {best_overall[0]} depth tier has the best overall performance with an average score of {best_overall[1]:.1f}.")
                
                # If no insights, add default message
                if not insights:
                    insights.append("Analysis complete. See detailed results in the other tabs.")
                
                # Display insights
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                # Add recommendations
                st.subheader("Recommendations")
                
                if best_tiers:
                    if 500 in best_tiers:
                        st.markdown(f"• For short-term trading, use the **{best_tiers[500][0]}** depth tier.")
                    
                    if 5000 in best_tiers:
                        st.markdown(f"• For medium-term trading, use the **{best_tiers[5000][0]}** depth tier.")
                    
                    if 10000 in best_tiers:
                        st.markdown(f"• For long-term trading, use the **{best_tiers[10000][0]}** depth tier.")
                else:
                    st.markdown("No specific recommendations available. Try analyzing a different pair or time period.")
            
            # Display detailed results for each point count
            with tabs[1]:
                st.header(f"500 Points Analysis (Short-Term)")
                create_tables_for_point_count(analyzer, 500)
            
            with tabs[2]:
                st.header(f"5000 Points Analysis (Medium-Term)")
                create_tables_for_point_count(analyzer, 5000)
            
            with tabs[3]:
                st.header(f"10000 Points Analysis (Long-Term)")
                create_tables_for_point_count(analyzer, 10000)
                
        else:
            progress_bar.empty()
            st.error(f"Failed to analyze {pair_to_analyze}. Please check if data exists for this pair.")
            
    else:
        # Simple welcome message - keep it minimal for speed
        st.info("""
        ### Quick Start
        1. Select a cryptocurrency pair from the sidebar
        2. Set the analysis timeframe
        3. Click "Run Analysis"
        
        The tool will analyze the best liquidity depth tiers at 500, 5000, and 10000 data points.
        """)

if __name__ == "__main__":
    main()