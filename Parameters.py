import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz
from scipy import stats
import math
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Configure database
def init_db_connection():
    # DB parameters - these should be stored in Streamlit secrets in production
    db_params = {
        'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
        'port': 5432,
        'database': 'report_dev',
        'user': 'public_rw',
        'password': 'aTJ92^kl04hllk'
    }
    
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        return conn, db_params
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, db_params

# Initialize connection
conn, db_params = init_db_connection()

# Main title
st.title("Crypto Analytics Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Exchange Comparison", "Mean Reversion Analysis"])

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)

#################################################################
# TAB 1: EXCHANGE COMPARISON
#################################################################
with tab1:
    st.header("Exchange Comparison Dashboard")
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exchange comparison class
    class ExchangeAnalyzer:
        """Specialized analyzer for comparing metrics between any two exchanges and ranking coins."""
        
        def __init__(self):
            self.exchange_data = {}  # Will store data from different exchanges
            self.all_exchanges = ['rollbit', 'uat', 'prod', 'sit']
            self.report_data = {}  # Will store all data needed for the report
            
            # Metrics to calculate and compare
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
            
            # Short names for metrics (for tables to avoid overflow)
            self.metric_short_names = {
                'direction_changes': 'Dir Chg',
                'choppiness': 'Chop',
                'tick_atr_pct': 'ATR%',
                'trend_strength': 'Trend'
            }
            
            # Point counts to analyze
            self.point_counts = [500, 2000, 5000, 10000, 50000]
            
            # The desired direction for each metric (whether higher or lower is better)
            self.metric_desired_direction = {
                'direction_changes': 'lower',  # Lower direction changes is generally better (less choppy)
                'choppiness': 'lower',         # Lower choppiness is generally better
                'tick_atr_pct': 'lower',       # Lower ATR % is generally more stable
                'trend_strength': 'lower'      # Lower trend strength is now considered better
            }
            
            # Initialize exchange_data structure
            for metric in self.metrics:
                self.exchange_data[metric] = {point: {} for point in self.point_counts}

        def _get_partition_tables(self, conn, start_date, end_date):
            """
            Get list of partition tables that need to be queried based on date range.
            Returns a list of table names (oracle_price_log_partition_YYYYMMDD)
            """
            # Convert to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str) and end_date:
                end_date = pd.to_datetime(end_date)
            elif end_date is None:
                end_date = datetime.now()
                
            # Ensure timezone is removed
            start_date = start_date.replace(tzinfo=None)
            end_date = end_date.replace(tzinfo=None)
                
            # Generate list of dates between start and end
            current_date = start_date
            dates = []
            
            while current_date <= end_date:
                dates.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)
            
            # Create table names from dates
            table_names = [f"oracle_price_log_partition_{date}" for date in dates]
            
            # Verify which tables actually exist in the database
            cursor = conn.cursor()
            existing_tables = []
            
            for table in table_names:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                if cursor.fetchone()[0]:
                    existing_tables.append(table)
            
            cursor.close()
            
            if not existing_tables:
                st.warning(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
            
            return existing_tables

        def _build_query_for_partition_tables(self, tables, source_type, pair_name, start_time, end_time, is_prod=False, is_sit=False, is_uat=False):
            """
            Build a complete UNION query for multiple partition tables.
            This creates a complete, valid SQL query with correct WHERE clauses.
            """
            if not tables:
                return ""
                
            union_parts = []
            
            for table in tables:
                # For PROD data
                if is_prod:
                    query = f"""
                    SELECT 
                        REPLACE(pair_name, 'PROD', '') AS pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND LOWER(pair_name) LIKE '%prod%'
                        AND REPLACE(pair_name, 'PROD', '') = '{pair_name}'
                    """
                # For SIT data
                elif is_sit:
                    query = f"""
                    SELECT 
                        REPLACE(pair_name, 'SIT', '') AS pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND LOWER(pair_name) LIKE '%sit%'
                        AND REPLACE(pair_name, 'SIT', '') = '{pair_name}'
                    """
                # For UAT data
                elif is_uat:
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 0
                        AND pair_name = '{pair_name}'
                    """
                else:
                    # For Rollbit data
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 1
                        AND pair_name = '{pair_name}'
                    """
                
                union_parts.append(query)
            
            # Join with UNION and add ORDER BY at the end
            complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
            return complete_query

        def fetch_and_analyze(self, conn, pairs_to_analyze, exchanges_to_compare, hours=24, start_time=None, end_time=None):
            """
            Fetch data for specified exchanges, analyze their convergence, and rank coins.
            
            Args:
                conn: Database connection
                pairs_to_analyze: List of coin pairs to analyze
                exchanges_to_compare: List of 2 exchanges to compare
                hours: Number of hours to analyze if start/end time not provided
                start_time: Analysis start time (string)
                end_time: Analysis end time (string)
            """
            if len(exchanges_to_compare) != 2:
                st.error("Error: Exactly 2 exchanges must be specified for comparison.")
                return
                
            # Validate exchanges
            for exchange in exchanges_to_compare:
                if exchange.lower() not in self.all_exchanges:
                    st.error(f"Error: Unknown exchange '{exchange}'. Valid options are: {', '.join(self.all_exchanges)}")
                    return
                    
            primary_exchange = exchanges_to_compare[0].lower()
            secondary_exchange = exchanges_to_compare[1].lower()
            
            # Display time window information
            if start_time and end_time:
                start_dt = pd.to_datetime(start_time)
                end_dt = pd.to_datetime(end_time)
                duration = end_dt - start_dt
                hours_diff = duration.total_seconds() / 3600
                st.info(f"Time window: {hours_diff:.2f} hours ({start_time} to {end_time})")
            else:
                st.info(f"Time window: {hours} hours")
            
            # Calculate start and end times if not provided
            if not start_time:
                start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            if not end_time:
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            try:
                # Get relevant partition tables for this time range
                partition_tables = self._get_partition_tables(conn, start_time, end_time)
                
                if not partition_tables:
                    st.error("No data tables available for the selected time range.")
                    return
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each pair for both exchanges
                for i, pair in enumerate(pairs_to_analyze):
                    progress_bar.progress((i) / len(pairs_to_analyze))
                    status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                    
                    # Process each exchange
                    for exchange in exchanges_to_compare:
                        exchange = exchange.lower()
                        
                        # Build appropriate query for this exchange
                        if exchange == 'rollbit':
                            query = self._build_query_for_partition_tables(
                                partition_tables,
                                source_type=1,
                                pair_name=pair,
                                start_time=start_time,
                                end_time=end_time
                            )
                        elif exchange == 'uat':
                            query = self._build_query_for_partition_tables(
                                partition_tables,
                                source_type=None,
                                pair_name=pair,
                                start_time=start_time,
                                end_time=end_time,
                                is_uat=True
                            )
                        elif exchange == 'prod':
                            query = self._build_query_for_partition_tables(
                                partition_tables,
                                source_type=None,
                                pair_name=pair,
                                start_time=start_time,
                                end_time=end_time,
                                is_prod=True
                            )
                        elif exchange == 'sit':
                            query = self._build_query_for_partition_tables(
                                partition_tables,
                                source_type=None,
                                pair_name=pair,
                                start_time=start_time,
                                end_time=end_time,
                                is_sit=True
                            )
                        
                        # Fetch data
                        if query:
                            # Add timeout to avoid hanging on large queries
                            try:
                                df = pd.read_sql_query(query, conn)
                                if len(df) > 0:
                                    # Store the pair name in a way that's easier to reference later
                                    coin_key = pair.replace('/', '_')
                                    self._process_price_data(df, 'timestamp', 'price', coin_key, exchange)
                                else:
                                    st.warning(f"No data found for {exchange.upper()}_{pair}")
                            except Exception as e:
                                st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
                
                # Final progress update
                progress_bar.progress(1.0)
                status_text.text(f"Processing complete!")
                
                # After processing all pairs, analyze the performance
                return self._analyze_performance(primary_exchange, secondary_exchange)
                
            except Exception as e:
                st.error(f"Error fetching and processing data: {e}")
        
        def _process_price_data(self, data, timestamp_col, price_col, coin_key, exchange):
            """Process price data for a cryptocurrency and calculate metrics for specified point counts."""
            try:
                # Extract price data
                filtered_df = data.copy()
                prices = pd.to_numeric(filtered_df[price_col], errors='coerce')
                prices = prices.dropna()
                
                if len(prices) < 100:  # Minimum threshold for meaningful analysis
                    return
                
                # Calculate metrics for each point count
                for point_count in self.point_counts:
                    if len(prices) >= point_count:
                        # Use the most recent N points
                        sample = prices.iloc[-point_count:]
                        
                        # Calculate mean price for ATR percentage calculation
                        mean_price = sample.mean()
                        
                        # Calculate each metric with improved error handling
                        direction_changes = self._calculate_direction_changes(sample)
                        choppiness = self._calculate_choppiness(sample, min(20, point_count // 10))
                        
                        # Calculate tick ATR
                        true_ranges = sample.diff().abs().dropna()
                        tick_atr = true_ranges.mean()
                        tick_atr_pct = (tick_atr / mean_price) * 100  # Convert to percentage of mean price
                        
                        # Calculate trend strength
                        trend_strength = self._calculate_trend_strength(sample, min(20, point_count // 10))
                        
                        # Store results in the metrics dictionary
                        if coin_key not in self.exchange_data['direction_changes'][point_count]:
                            self.exchange_data['direction_changes'][point_count][coin_key] = {}
                        if coin_key not in self.exchange_data['choppiness'][point_count]:
                            self.exchange_data['choppiness'][point_count][coin_key] = {}
                        if coin_key not in self.exchange_data['tick_atr_pct'][point_count]:
                            self.exchange_data['tick_atr_pct'][point_count][coin_key] = {}
                        if coin_key not in self.exchange_data['trend_strength'][point_count]:
                            self.exchange_data['trend_strength'][point_count][coin_key] = {}
                        
                        self.exchange_data['direction_changes'][point_count][coin_key][exchange] = direction_changes
                        self.exchange_data['choppiness'][point_count][coin_key][exchange] = choppiness
                        self.exchange_data['tick_atr_pct'][point_count][coin_key][exchange] = tick_atr_pct
                        self.exchange_data['trend_strength'][point_count][coin_key][exchange] = trend_strength
            except Exception as e:
                st.error(f"Error processing {coin_key}: {e}")
        
        def _calculate_direction_changes(self, prices):
            """Calculate the percentage of times the price direction changes."""
            try:
                price_changes = prices.diff().dropna()
                signs = np.sign(price_changes)
                direction_changes = (signs.shift(1) != signs).sum()
                
                total_periods = len(signs) - 1
                if total_periods > 0:
                    direction_change_pct = (direction_changes / total_periods) * 100
                else:
                    direction_change_pct = 0
                
                return direction_change_pct
            except Exception as e:
                return 50.0  # Return a reasonable default instead of zero
        
        def _calculate_choppiness(self, prices, window):
            """Calculate average Choppiness Index with improved error handling."""
            try:
                diff = prices.diff().abs()
                sum_abs_changes = diff.rolling(window, min_periods=1).sum()
                price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
                
                # Check for zero price range
                if (price_range == 0).any():
                    # Replace zeros with a small value to avoid division by zero
                    price_range = price_range.replace(0, 1e-10)
                
                # Avoid division by zero
                epsilon = 1e-10
                choppiness = 100 * sum_abs_changes / (price_range + epsilon)
                
                # Cap extreme values and handle NaN
                choppiness = np.minimum(choppiness, 1000)
                choppiness = choppiness.fillna(200)  # Replace NaN with a reasonable default
                
                return choppiness.mean()
            except Exception as e:
                return 200.0  # Return a reasonable default value
        
        def _calculate_trend_strength(self, prices, window):
            """Calculate average Trend Strength with improved error handling."""
            try:
                diff = prices.diff().abs()
                sum_abs_changes = diff.rolling(window, min_periods=1).sum()
                net_change = (prices - prices.shift(window)).abs()
                
                # Avoid division by zero
                epsilon = 1e-10
                
                # Check if sum_abs_changes is close to zero
                trend_strength = np.where(
                    sum_abs_changes > epsilon,
                    net_change / (sum_abs_changes + epsilon),
                    0.5  # Default value when there's no change
                )
                
                # Convert to pandas Series if it's a numpy array
                if isinstance(trend_strength, np.ndarray):
                    trend_strength = pd.Series(trend_strength, index=net_change.index)
                
                # Handle NaN values
                trend_strength = pd.Series(trend_strength).fillna(0.5)
                
                return trend_strength.mean()
            except Exception as e:
                return 0.5  # Return a reasonable default value
        
        def _analyze_performance(self, primary_exchange, secondary_exchange):
            """Analyze the performance of secondary_exchange relative to primary_exchange."""
            # Check if we have data for both exchanges
            has_data = False
            for metric in self.metrics:
                for point_count in self.point_counts:
                    for coin_data in self.exchange_data[metric][point_count].values():
                        if primary_exchange in coin_data and secondary_exchange in coin_data:
                            has_data = True
                            break
                    if has_data:
                        break
                if has_data:
                    break
            
            if not has_data:
                st.error(f"Insufficient data for both {primary_exchange.upper()} and {secondary_exchange.upper()} to perform analysis.")
                return None
            
            # Results for all point counts
            results = {}
            
            # For each point count, create a performance table
            for point_count in self.point_counts:
                # Check if we have data for this point count
                has_data_for_point = False
                for metric in self.metrics:
                    for coin_data in self.exchange_data[metric][point_count].values():
                        if primary_exchange in coin_data and secondary_exchange in coin_data:
                            has_data_for_point = True
                            break
                    if has_data_for_point:
                        break
                
                if not has_data_for_point:
                    continue
                
                # Create a DataFrame for the comparison table
                comparison_data = []
                
                # Get all coins that have data for both exchanges
                all_coins = set()
                for metric in self.metrics:
                    for coin, exchanges in self.exchange_data[metric][point_count].items():
                        if primary_exchange in exchanges and secondary_exchange in exchanges:
                            all_coins.add(coin)
                
                # For each coin, calculate relative performance for each metric
                for coin in all_coins:
                    row = {'Coin': coin.replace('_', '/')}
                    relative_scores = []
                    
                    for metric in self.metrics:
                        # Check if we have data for both exchanges for this metric
                        if (coin in self.exchange_data[metric][point_count] and
                            primary_exchange in self.exchange_data[metric][point_count][coin] and
                            secondary_exchange in self.exchange_data[metric][point_count][coin]):
                            
                            primary_value = self.exchange_data[metric][point_count][coin][primary_exchange]
                            secondary_value = self.exchange_data[metric][point_count][coin][secondary_exchange]
                            
                            # Calculate absolute and percentage difference
                            abs_diff = secondary_value - primary_value
                            pct_diff = (abs_diff / primary_value * 100) if primary_value != 0 else 0
                            
                            # Calculate relative performance score (100 means equal to primary, >100 means better)
                            if metric == 'trend_strength':
                                # For trend_strength, lower is better, so inverse the ratio
                                if secondary_value == 0:
                                    # Edge case
                                    relative_score = 100
                                else:
                                    relative_score = (primary_value / secondary_value) * 100
                            else:
                                # For all other metrics, higher is better
                                if primary_value == 0:
                                    # Edge case
                                    relative_score = 100 if secondary_value == 0 else 200  # Arbitrary high value
                                else:
                                    relative_score = (secondary_value / primary_value) * 100
                            
                            relative_scores.append(relative_score)
                            
                            # Add to the row
                            row[f'{self.metric_short_names[metric]} {primary_exchange.upper()}'] = primary_value
                            row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()}'] = secondary_value
                            row[f'{self.metric_short_names[metric]} Diff'] = abs_diff
                            row[f'{self.metric_short_names[metric]} Diff %'] = pct_diff
                            row[f'{self.metric_short_names[metric]} Score'] = relative_score
                    
                    # Calculate overall relative score (average of individual scores)
                    if relative_scores:
                        row['Overall Score'] = sum(relative_scores) / len(relative_scores)
                    else:
                        row['Overall Score'] = 100  # Default to "equal" if no metrics
                    
                    comparison_data.append(row)
                
                # Create DataFrame and sort by relative score (highest first)
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
                    
                    # Store for results
                    results[point_count] = comparison_df
            
            return results
        
        def _create_individual_ranking(self, exchange, point_count):
            """Create a separate ranking of coins for a single exchange based on each metric."""
            rankings = {}
            
            # For each metric, create a ranking
            for metric in self.metrics:
                # Get all coins that have data for this exchange and metric
                coin_data = {}
                for coin, exchanges in self.exchange_data[metric][point_count].items():
                    if exchange in exchanges:
                        coin_data[coin] = exchanges[exchange]
                
                if not coin_data:
                    continue
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Coin': [coin.replace('_', '/') for coin in coin_data.keys()],
                    'Value': list(coin_data.values())
                })
                
                # Sort based on metric (ascending or descending)
                # For trend_strength, lower is better
                # For all other metrics, higher is now better
                ascending = True if metric == 'trend_strength' else False
                
                df = df.sort_values('Value', ascending=ascending)
                
                # Add rank column
                df.insert(0, 'Rank', range(1, len(df) + 1))
                
                # Store ranking
                rankings[metric] = df
                
            return rankings
    
    # Setup form for parameters
    with st.form("exchange_comparison_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Exchange selection
            primary_exchange = st.selectbox(
                "Primary Exchange",
                ["rollbit", "prod", "uat", "sit"],
                index=0
            )
            
            secondary_exchange = st.selectbox(
                "Secondary Exchange",
                ["prod", "rollbit", "uat", "sit"],
                index=0
            )
            
        with col2:
            # Time window selection
            time_option = st.radio(
                "Time Window",
                ["Last N Hours", "Custom Range"]
            )
            
            if time_option == "Last N Hours":
                hours = st.number_input("Hours", min_value=1, max_value=168, value=24)
                start_time = None
                end_time = None
            else:
                hours = None
                start_time = st.date_input("Start Date", value=datetime.now() - timedelta(days=1))
                end_time = st.date_input("End Date", value=datetime.now())
                
                # Convert to datetime
                start_time = datetime.combine(start_time, datetime.min.time())
                end_time = datetime.combine(end_time, datetime.max.time())
                
                # Format as strings
                start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
                
        with col3:
            # Default list of pairs
            default_pairs = """PEPE/USDT
PAXG/USDT
DOGE/USDT
BTC/USDT
EOS/USDT
BNB/USDT
MERL/USDT
FHE/USDT
IP/USDT
ORCA/USDT
TRUMP/USDT
LIBRA/USDT
AI16Z/USDT
OM/USDT
TRX/USDT
S/USDT
PI/USDT
JUP/USDT
BABY/USDT
PARTI/USDT
ADA/USDT
HYPE/USDT
VIRTUAL/USDT
SUI/USDT
SATS/USDT
XRP/USDT
ORDI/USDT
WIF/USDT
VANA/USDT
PENGU/USDT
VINE/USDT
GRIFFAIN/USDT
MEW/USDT
POPCAT/USDT
FARTCOIN/USDT
TON/USDT
MELANIA/USDT
SOL/USDT
PNUT/USDT
CAKE/USDT
TST/USDT
ETH/USDT"""
            
            # Pair selection (textarea for multiple pairs)
            pair_input = st.text_area("Pairs to Analyze (one per line)", 
                                        default_pairs, 
                                        height=300)
            
            # Parse pairs
            pairs = [p.strip() for p in pair_input.split("\n") if p.strip()]
        
        # Submit button
        submit_button = st.form_submit_button("Analyze Exchanges")
    
    # When form is submitted
    if submit_button:
        if not conn:
            st.error("Database connection not available.")
        elif primary_exchange == secondary_exchange:
            st.error("Please select two different exchanges to compare.")
        elif not pairs:
            st.error("Please enter at least one pair to analyze.")
        else:
            # Initialize analyzer
            analyzer = ExchangeAnalyzer()
            
            # Run analysis
            st.subheader(f"Comparing {primary_exchange.upper()} vs {secondary_exchange.upper()}")
            
            with st.spinner("Fetching and analyzing data..."):
                results = analyzer.fetch_and_analyze(
                    conn=conn,
                    pairs_to_analyze=pairs,
                    exchanges_to_compare=[primary_exchange, secondary_exchange],
                    hours=hours,
                    start_time=start_time,
                    end_time=end_time
                )
            
            if results:
                # Create tabs for each point count
                point_count_tabs = st.tabs([f"{count} Points" for count in analyzer.point_counts if count in results])
                
                for i, point_count in enumerate([pc for pc in analyzer.point_counts if pc in results]):
                    with point_count_tabs[i]:
                        df = results[point_count]
                        
                        # Style the DataFrame
                        def highlight_scores(val):
                            if 'Score' in str(val.name):
                                if val > 130:
                                    return 'background-color: #60b33c; color: white; font-weight: bold'
                                elif val > 110:
                                    return 'background-color: #a0d995; color: black'
                                elif val > 90:
                                    return 'background-color: #f1f1aa; color: black'
                                elif val > 70:
                                    return 'background-color: #ffc299; color: black'
                                else:
                                    return 'background-color: #ff8080; color: black; font-weight: bold'
                            return ''
                        
                        # Display the data
                        st.dataframe(
                            df.style.applymap(highlight_scores),
                            height=600,
                            use_container_width=True
                        )
                        
                        # Create visualization
                        st.subheader(f"Relative Performance Visualization ({point_count} Points)")
                        
                        # Top 10 and bottom 10 coins
                        top_10 = df.nlargest(10, 'Overall Score')
                        bottom_10 = df.nsmallest(10, 'Overall Score')
                        
                        # Combined visualization
                        fig = go.Figure()
                        
                        # Add top 10
                        fig.add_trace(go.Bar(
                            x=top_10['Coin'],
                            y=top_10['Overall Score'],
                            name='Top Performers',
                            marker_color='green'
                        ))
                        
                        # Add bottom 10
                        fig.add_trace(go.Bar(
                            x=bottom_10['Coin'],
                            y=bottom_10['Overall Score'],
                            name='Bottom Performers',
                            marker_color='red'
                        ))
                        
                        # Add reference line at 100
                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            y0=100,
                            x1=len(top_10) + len(bottom_10) - 0.5,
                            y1=100,
                            line=dict(
                                color="black",
                                width=2,
                                dash="dash",
                            )
                        )
                        
                        fig.update_layout(
                            title=f"{secondary_exchange.upper()} Performance Relative to {primary_exchange.upper()} (100 = Equal)",
                            xaxis_title="Coin",
                            yaxis_title="Relative Performance Score",
                            barmode='group',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed metrics breakdown
                        st.subheader("Metric-by-Metric Comparison")
                        
                        # Create tabs for each metric
                        metric_tabs = st.tabs([analyzer.metric_display_names[metric] for metric in analyzer.metrics])
                        
                        for j, metric in enumerate(analyzer.metrics):
                            with metric_tabs[j]:
                                # Get relevant columns
                                short_name = analyzer.metric_short_names[metric]
                                primary_col = f'{short_name} {primary_exchange.upper()}'
                                secondary_col = f'{short_name} {secondary_exchange.upper()}'
                                score_col = f'{short_name} Score'
                                
                                if primary_col in df.columns and secondary_col in df.columns:
                                    # Sort by score
                                    sorted_df = df.sort_values(score_col, ascending=False)
                                    
                                    # Display the top 20 coins
                                    metric_df = sorted_df[['Coin', primary_col, secondary_col, score_col]].head(20)
                                    
                                    # Plot comparison
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=metric_df['Coin'],
                                        y=metric_df[primary_col],
                                        name=f"{primary_exchange.upper()}",
                                        marker_color='blue'
                                    ))
                                    
                                    fig.add_trace(go.Bar(
                                        x=metric_df['Coin'],
                                        y=metric_df[secondary_col],
                                        name=f"{secondary_exchange.upper()}",
                                        marker_color='orange'
                                    ))
                                    
                                    # Update layout with better alignment
                                    fig.update_layout(
                                        title=f"{analyzer.metric_display_names[metric]} Comparison - Top 20 by Score",
                                        xaxis_title="Coin",
                                        yaxis_title="Value",
                                        barmode='group',
                                        height=500,
                                        xaxis=dict(
                                            tickangle=45,
                                            tickmode='array',
                                            tickvals=list(range(len(metric_df))),
                                            ticktext=metric_df['Coin']
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display table with all coins
                                    st.dataframe(
                                        sorted_df[['Coin', primary_col, secondary_col, score_col]].style.applymap(highlight_scores),
                                        height=400,
                                        use_container_width=True
                                    )
                                    
                                    # Data distribution
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Histogram for primary exchange
                                        fig1 = px.histogram(
                                            sorted_df, x=primary_col,
                                            title=f"{primary_exchange.upper()} {analyzer.metric_display_names[metric]} Distribution",
                                            nbins=30, 
                                            color_discrete_sequence=['blue']
                                        )
                                        st.plotly_chart(fig1, use_container_width=True)
                                        
                                    with col2:
                                        # Histogram for secondary exchange
                                        fig2 = px.histogram(
                                            sorted_df, x=secondary_col,
                                            title=f"{secondary_exchange.upper()} {analyzer.metric_display_names[metric]} Distribution",
                                            nbins=30,
                                            color_discrete_sequence=['orange']
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                else:
                                    st.warning(f"No data available for {analyzer.metric_display_names[metric]}.")

#################################################################
# TAB 2: MEAN REVERSION ANALYSIS
#################################################################
with tab2:
    st.header("Mean Reversion Analysis")
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Helper class for mean reversion analysis
    class MeanReversionAnalyzer:
        def __init__(self):
            self.pair_data = {}
            
        def _get_partition_tables(self, conn, start_date, end_date):
            """Get list of partition tables that need to be queried based on date range."""
            # Convert to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str) and end_date:
                end_date = pd.to_datetime(end_date)
            elif end_date is None:
                end_date = datetime.now()
                
            # Ensure timezone is removed
            start_date = start_date.replace(tzinfo=None)
            end_date = end_date.replace(tzinfo=None)
                
            # Generate list of dates between start and end
            current_date = start_date
            dates = []
            
            while current_date <= end_date:
                dates.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)
            
            # Create table names from dates
            table_names = [f"oracle_price_log_partition_{date}" for date in dates]
            
            # Verify which tables actually exist in the database
            cursor = conn.cursor()
            existing_tables = []
            
            for table in table_names:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                if cursor.fetchone()[0]:
                    existing_tables.append(table)
            
            cursor.close()
            
            if not existing_tables:
                st.warning(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
            
            return existing_tables
        
        def _build_query_for_partition_tables(self, tables, pair_name, start_time, end_time, exchange):
            """Build a complete UNION query for multiple partition tables."""
            if not tables:
                return ""
                
            union_parts = []
            
            for table in tables:
                # Modify query based on exchange
                if exchange == 'prod':
                    query = f"""
                    SELECT 
                        REPLACE(pair_name, 'PROD', '') AS pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND LOWER(pair_name) LIKE '%prod%'
                        AND REPLACE(pair_name, 'PROD', '') = '{pair_name}'
                    """
                elif exchange == 'sit':
                    query = f"""
                    SELECT 
                        REPLACE(pair_name, 'SIT', '') AS pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND LOWER(pair_name) LIKE '%sit%'
                        AND REPLACE(pair_name, 'SIT', '') = '{pair_name}'
                    """
                elif exchange == 'uat':
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 0
                        AND pair_name = '{pair_name}'
                    """
                else:  # rollbit is default
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp,
                        final_price AS price
                    FROM 
                        public.{table}
                    WHERE 
                        created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 1
                        AND pair_name = '{pair_name}'
                    """
                
                union_parts.append(query)
            
            # Join with UNION and add ORDER BY at the end
            complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
            return complete_query
        
        def fetch_price_data(self, conn, pair_name, hours=24, start_time=None, end_time=None, exchange='rollbit'):
            """Fetch price data for a specific pair and calculate mean reversion metrics."""
            # Calculate start and end times if not provided
            if not start_time:
                start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            if not end_time:
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # Get relevant partition tables for this time range
                partition_tables = self._get_partition_tables(conn, start_time, end_time)
                
                if not partition_tables:
                    st.error("No data tables available for the selected time range.")
                    return None
                
                # Build query
                query = self._build_query_for_partition_tables(
                    partition_tables,
                    pair_name,
                    start_time,
                    end_time,
                    exchange
                )
                
                if not query:
                    st.error("Failed to build query.")
                    return None
                
                # Fetch data
                df = pd.read_sql_query(query, conn)
                
                if len(df) < 100:
                    st.warning(f"Insufficient data points for {pair_name} on {exchange} (minimum 100 required).")
                    return None
                
                st.success(f"Successfully fetched {len(df)} data points for {pair_name} on {exchange}.")
                
                # Process the data
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Ensure price is numeric
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df = df.dropna(subset=['price'])
                
                return df
                
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return None
        
        def calculate_metrics(self, df, window_minutes=30):
            """Calculate various mean reversion metrics for the price data."""
            if df is None or len(df) < 100:
                return None
            
            try:
                # Create a copy of the dataframe
                data = df.copy()
                
                # Resample to evenly spaced intervals (1-minute)
                data = data.resample('1T').mean().interpolate(method='linear')
                
                # Calculate returns
                data['returns'] = data['price'].pct_change()
                
                # For sliding window analysis
                window_size = window_minutes  # in minutes
                
                # Initialize results dataframe
                results = []
                
                # Ensure we have enough data
                if len(data) <= window_size:
                    st.warning(f"Not enough data points for {window_size}-minute window analysis.")
                    return None
                
                # Calculate metrics for each window
                for i in range(0, len(data) - window_size, window_size):
                    window_data = data.iloc[i:i+window_size].copy()
                    
                    if len(window_data) < window_size * 0.9:  # Require at least 90% of expected data points
                        continue
                    
                    window_start = window_data.index[0]
                    window_end = window_data.index[-1]
                    
                    # Calculate direction changes
                    price_changes = window_data['price'].diff().dropna()
                    signs = np.sign(price_changes)
                    direction_changes = (signs.shift(1) != signs).sum()
                    
                    total_periods = len(signs) - 1
                    if total_periods > 0:
                        direction_change_pct = (direction_changes / total_periods) * 100
                    else:
                        direction_change_pct = 0
                    
                    # Calculate price range
                    price_high = window_data['price'].max()
                    price_low = window_data['price'].min()
                    price_range_pct = ((price_high - price_low) / price_low) * 100
                    
                    # Calculate A/B ratio (direction changes to price range ratio)
                    if price_range_pct > 0:
                        a_b_ratio = direction_change_pct / price_range_pct
                    else:
                        a_b_ratio = 0
                    
                    # Calculate Hurst exponent
                    try:
                        prices = window_data['price'].values
                        hurst = self.calculate_hurst_exponent(prices)
                    except:
                        hurst = 0.5  # Default to 0.5 (random walk) if calculation fails
                    
                    # Store results
                    results.append({
                        'window_start': window_start,
                        'window_end': window_end,
                        'direction_changes':float(direction_change_pct),
                        'price_range_pct': float(price_range_pct),
                        'a_b_ratio': float(a_b_ratio),
                        'hurst': float(hurst),
                        'start_price': float(window_data['price'].iloc[0]),
                        'end_price': float(window_data['price'].iloc[-1]),
                        'mean_price': float(window_data['price'].mean())
                    })
                
                if not results:
                    st.warning("No valid analysis windows found.")
                    return None
                
                # Convert to DataFrame
                results_df = pd.DataFrame(results)
                # Ensure all columns except timestamps are numeric
                for col in results_df.columns:
                  if col not in ['window_start', 'window_end']:
                    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

                results_df.set_index('window_start', inplace=True)
                
                return results_df
                
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
                return None
        
        def calculate_hurst_exponent(self, prices, max_lag=20):
            """Calculate the Hurst exponent for a price series."""
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Need sufficient data
            if len(returns) < max_lag * 2:
                max_lag = len(returns) // 4
            
            # Calculate variance of returns for different lags
            lags = range(2, max_lag)
            tau = []
            
            for lag in lags:
                # Take difference with lag
                pp = np.subtract(returns[lag:], returns[:-lag])
                
                # Calculate variance
                tau.append(np.sqrt(np.std(pp)))
            
            # Avoid log(0)
            tau = [max(t, 1e-10) for t in tau]
            
            # Calculate Hurst as the slope of the log-log plot
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] / 2.0
            
            # Bound between 0 and 1
            return max(0, min(1, hurst))
    
    # Setup form for parameters
    with st.form("mean_reversion_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Pair and exchange selection
            pair = st.text_input("Trading Pair", "BTC/USDT")
            exchange = st.selectbox(
                "Exchange",
                ["rollbit", "prod", "uat", "sit"],
                index=0
            )
            
            # Window size for analysis
            window_minutes = st.slider(
                "Analysis Window (minutes)",
                min_value=5,
                max_value=60,
                value=30,
                step=5
            )
            
        with col2:
            # Time range selection
            time_option = st.radio(
                "Time Range",
                ["Last N Hours", "Custom Range"]
            )
            
            if time_option == "Last N Hours":
                hours = st.number_input("Hours to Analyze", min_value=1, max_value=72, value=24)
                start_time = None
                end_time = None
            else:
                hours = None
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=1))
                end_date = st.date_input("End Date", value=datetime.now())
                
                # Convert to datetime
                start_time = datetime.combine(start_date, datetime.min.time())
                end_time = datetime.combine(end_date, datetime.max.time())
                
                # Format as strings
                start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Submit button
        analyze_button = st.form_submit_button("Analyze Mean Reversion")
    
    # When form is submitted
    if analyze_button:
        if not conn:
            st.error("Database connection not available.")
        else:
            # Initialize analyzer
            analyzer = MeanReversionAnalyzer()
            
            # Display analysis info
            st.subheader(f"Mean Reversion Analysis: {pair} on {exchange.upper()}")
            
            if hours:
                st.info(f"Analyzing data from the last {hours} hours using {window_minutes}-minute windows")
            else:
                st.info(f"Analyzing data from {start_time} to {end_time} using {window_minutes}-minute windows")
            
            # Fetch and process data
            with st.spinner("Fetching price data..."):
                price_df = analyzer.fetch_price_data(
                    conn=conn,
                    pair_name=pair,
                    hours=hours,
                    start_time=start_time,
                    end_time=end_time,
                    exchange=exchange
                )
            
            if price_df is not None:
                # Display raw price chart
                st.subheader("Price Chart")
                
                fig = px.line(
                    price_df, 
                    y='price',
                    title=f"{pair} Price ({exchange.upper()})",
                    labels={'price': 'Price', 'timestamp': 'Time'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics
                with st.spinner("Calculating mean reversion metrics..."):
                    metrics_df = analyzer.calculate_metrics(price_df, window_minutes)
                
                if metrics_df is not None:
                    # Display summary statistics
                    st.subheader("Mean Reversion Metrics Summary")
                    
                    summary = {
                        'Metric': [
                            'Avg. Direction Changes (%)',
                            'Avg. Price Range (%)',
                            'Avg. A/B Ratio',
                            'Avg. Hurst Exponent',
                            'Latest Direction Changes (%)',
                            'Latest Price Range (%)',
                            'Latest A/B Ratio',
                            'Latest Hurst Exponent'
                        ],
                        'Value': [
                            f"{float(metrics_df['direction_changes'].mean()):.2f}%",
                            f"{float(metrics_df['price_range_pct'].mean()):.4f}%",
                            f"{float(metrics_df['a_b_ratio'].mean()):.4f}",
                            f"{float(metrics_df['hurst'].mean()):.4f}",
                            f"{float(metrics_df['direction_changes'].iloc[-1]):.2f}%",
                            f"{float(metrics_df['price_range_pct'].iloc[-1]):.4f}%",
                            f"{float(metrics_df['a_b_ratio'].iloc[-1]):.4f}",
                            f"{float(metrics_df['hurst'].iloc[-1]):.4f}"
                        ],
                        'Interpretation': [
                            'Higher = More direction changes',
                            'Higher = Larger price movements',
                            'Higher = More mean-reverting behavior',
                            '<0.5 = Mean-reverting, 0.5 = Random, >0.5 = Trending',
                            'Latest window direction changes',
                            'Latest window price range',
                            'Latest window A/B ratio',
                            'Latest window Hurst exponent'
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Key Metrics Over Time", "Mean Reversion Indicators", "Raw Data"])
                    
                    with viz_tabs[0]:
                        # Time series of key metrics
                        st.subheader("Key Metrics Over Time")
                        
                        # Plot direction changes and price range
                        fig1 = go.Figure()
                        
                        # Direction changes
                        fig1.add_trace(go.Scatter(
                            x=metrics_df.index,
                            y=metrics_df['direction_changes'],
                            name='Direction Changes (%)',
                            line=dict(color='blue')
                        ))
                        
                        # Price range
                        fig1.add_trace(go.Scatter(
                            x=metrics_df.index,
                            y=metrics_df['price_range_pct'],
                            name='Price Range (%)',
                            line=dict(color='red'),
                            yaxis='y2'
                        ))
                        
                        # Update layout with two y-axes
                        fig1.update_layout(
                            title="Direction Changes vs Price Range",
                            xaxis=dict(title="Time"),
                            yaxis=dict(
                                title="Direction Changes (%)",
                                titlefont=dict(color="blue"),
                                tickfont=dict(color="blue")
                            ),
                            yaxis2=dict(
                                title="Price Range (%)",
                                titlefont=dict(color="red"),
                                tickfont=dict(color="red"),
                                overlaying="y",
                                side="right"
                            ),
                            height=400
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Plot A/B ratio and Hurst exponent
                        fig2 = go.Figure()
                        
                        # A/B ratio
                        fig2.add_trace(go.Scatter(
                            x=metrics_df.index,
                            y=metrics_df['a_b_ratio'],
                            name='A/B Ratio',
                            line=dict(color='green')
                        ))
                        
                        # Hurst exponent
                        fig2.add_trace(go.Scatter(
                            x=metrics_df.index,
                            y=metrics_df['hurst'],
                            name='Hurst Exponent',
                            line=dict(color='purple'),
                            yaxis='y2'
                        ))
                        
                        # Add reference line at Hurst = 0.5
                        fig2.add_shape(
                            type="line",
                            x0=metrics_df.index[0],
                            y0=0.5,
                            x1=metrics_df.index[-1],
                            y1=0.5,
                            line=dict(
                                color="purple",
                                width=1,
                                dash="dash",
                            ),
                            yref='y2'
                        )
                        
                        # Update layout with two y-axes
                        fig2.update_layout(
                            title="A/B Ratio vs Hurst Exponent",
                            xaxis=dict(title="Time"),
                            yaxis=dict(
                                title="A/B Ratio",
                                titlefont=dict(color="green"),
                                tickfont=dict(color="green")
                            ),
                            yaxis2=dict(
                                title="Hurst Exponent",
                                titlefont=dict(color="purple"),
                                tickfont=dict(color="purple"),
                                overlaying="y",
                                side="right",
                                range=[0, 1]
                            ),
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with viz_tabs[1]:
                        # Mean reversion indicators
                        st.subheader("Mean Reversion Indicators")
                        
                        # Create a combined visualization of price and indicators
                        fig3 = go.Figure()
                        
                        # Add price
                        fig3.add_trace(go.Scatter(
                            x=metrics_df.index,
                            y=metrics_df['mean_price'],
                            name='Average Price',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Highlight regions based on mean reversion indicators
                        # Green for strong mean reversion, red for trend
                        for i in range(len(metrics_df)):
                            if i > 0:
                                start_time = metrics_df.index[i-1]
                                end_time = metrics_df.index[i]
                                
                                # Mean reversion indicators
                                hurst = metrics_df['hurst'].iloc[i]
                                a_b_ratio = metrics_df['a_b_ratio'].iloc[i]
                                
                                # Determine color based on indicators
                                if hurst < 0.4 and a_b_ratio > 1.5:
                                    # Strong mean reversion
                                    color = 'rgba(0, 255, 0, 0.2)'  # Green
                                elif hurst > 0.6 and a_b_ratio < 0.5:
                                    # Strong trend
                                    color = 'rgba(255, 0, 0, 0.2)'  # Red
                                else:
                                    # Neutral/random
                                    color = 'rgba(255, 255, 0, 0.1)'  # Yellow
                                
                                # Add shaded region
                                fig3.add_shape(
                                    type="rect",
                                    x0=start_time,
                                    x1=end_time,
                                    y0=metrics_df['mean_price'].min() * 0.95,
                                    y1=metrics_df['mean_price'].max() * 1.05,
                                    fillcolor=color,
                                    opacity=0.5,
                                    layer="below",
                                    line_width=0
                                )
                        
                        # Update layout
                        fig3.update_layout(
                            title="Price with Mean Reversion Indicators",
                            xaxis=dict(title="Time"),
                            yaxis=dict(title="Price"),
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Add explanation of colors
                        st.markdown("""
                        **Color Legend:**
                        - 🟢 **Green regions**: Strong mean-reverting behavior (Hurst < 0.4, A/B Ratio > 1.5)
                        - 🔴 **Red regions**: Strong trending behavior (Hurst > 0.6, A/B Ratio < 0.5)
                        - 🟡 **Yellow regions**: Neutral/random behavior
                        """)
                        
                        # Create scatter plot of Hurst vs A/B ratio
                        fig4 = px.scatter(
                            metrics_df,
                            x='hurst',
                            y='a_b_ratio',
                            title="Hurst Exponent vs A/B Ratio",
                            color='direction_changes',
                            size='price_range_pct',
                            hover_data=['window_end', 'direction_changes', 'price_range_pct'],
                            labels={
                                'hurst': 'Hurst Exponent',
                                'a_b_ratio': 'A/B Ratio',
                                'direction_changes': 'Direction Changes (%)',
                                'price_range_pct': 'Price Range (%)',
                                'window_end': 'Window End Time'
                            }
                        )
                        
                        # Add reference lines
                        fig4.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="A/B = 1.0")
                        fig4.add_vline(x=0.5, line_dash="dash", line_color="purple", annotation_text="Hurst = 0.5")
                        
                        # Add quadrant labels
                        fig4.add_annotation(x=0.25, y=2.0, text="Mean-Reverting", showarrow=False, font=dict(color="green"))
                        fig4.add_annotation(x=0.75, y=0.25, text="Trending", showarrow=False, font=dict(color="red"))
                        
                        fig4.update_layout(height=500)
                        st.plotly_chart(fig4, use_container_width=True)
                    
                    with viz_tabs[2]:
                        # Raw data tab
                        st.subheader("Raw Metrics Data")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Download link for the data
                        csv = metrics_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{pair.replace('/', '_')}_{exchange}_metrics.csv",
                            mime="text/csv"
                        )
                        
                        # Display explanation of metrics
                        with st.expander("Metrics Explanation"):
                            st.markdown("""
                            ### Mean Reversion Metrics Explained

                            #### Direction Changes (%)
                            The percentage of price movements that change direction. Higher values indicate more oscillating behavior.

                            #### Price Range (%)
                            The percentage difference between the highest and lowest price in the window. Higher values indicate greater volatility.

                            #### A/B Ratio (Direction Changes / Price Range)
                            A ratio comparing the frequency of direction changes to the price range. 
                            - **Higher values (>1)**: More mean-reverting behavior (many direction changes with relatively small price range)
                            - **Lower values (<1)**: More trending behavior (few direction changes with relatively large price range)

                            #### Hurst Exponent
                            A measure of long-term memory in a time series:
                            - **H < 0.5**: Mean-reverting behavior (negative autocorrelation)
                            - **H = 0.5**: Random walk/Brownian motion (no memory)
                            - **H > 0.5**: Trending behavior (positive autocorrelation)

                            The combination of a high A/B ratio and low Hurst exponent is the strongest indication of mean-reverting behavior.
                            """)
                else:
                    st.error("Failed to calculate metrics. Please try with a different time window or pair.")
            else:
                st.error("Failed to fetch price data. Please try a different pair or time window.")

# Add explanation for running this app
st.sidebar.title("About This Dashboard")
st.sidebar.write("""
## Crypto Analytics Dashboard

This dashboard provides tools for analyzing cryptocurrency price data across different exchanges:

1. **Exchange Comparison**: Compare metrics between two exchanges across multiple coins
2. **Mean Reversion Analysis**: Analyze the mean-reverting behavior of a specific cryptocurrency pair

### How to Run Locally

1. Save this code to a file named `crypto_analytics.py`
2. Install the required packages:
   ```
   pip install streamlit pandas numpy plotly psycopg2-binary pytz scipy matplotlib
   ```
3. Run the app:
   ```
   streamlit run crypto_analytics.py
   ```

### Deployment to GitHub

1. Create a new GitHub repository
2. Add the `crypto_analytics.py` file to the repository
3. Create a `requirements.txt` file with all the dependencies
4. Connect the repository to Streamlit Cloud for deployment

### Notes

- The database connection parameters should be stored securely in production
- For large scale deployment, consider optimizing the database queries
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")