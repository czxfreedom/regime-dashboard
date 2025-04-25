import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Exchange Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Configure database
def init_db_connection():
    # DB parameters - these should be stored in Streamlit secrets in production
    db_params = {
        'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
        'port': 5432,
        'database': 'replication_report',
        'user': 'public_replication',
        'password': '866^FKC4hllk'
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
st.title("Crypto Exchange Analysis Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Parameter Comparison", "Rankings & Analysis"])

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Add session state for last analysis time if not exists
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = None

# Helper function to ensure datetime is timezone aware
def ensure_timezone_aware(dt, timezone=pytz.UTC):
    """Ensure a datetime object is timezone aware"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return timezone.localize(dt)
    return dt

# Exchange comparison class
class ExchangeAnalyzer:
    """Specialized analyzer for comparing metrics between Surf and Rollbit"""
    
    def __init__(self):
        self.exchange_data = {}  # Will store data from different exchanges
        self.all_exchanges = ['rollbit', 'surf']  # Only Rollbit and Surf
        
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
        self.point_counts = [500, 5000, 50000]
        
        # The desired direction for each metric (whether higher or lower is better)
        self.metric_desired_direction = {
            'direction_changes': 'lower',  
            'choppiness': 'lower',        
            'tick_atr_pct': 'lower',       
            'trend_strength': 'lower'     
        }
        
        # Initialize exchange_data structure
        for metric in self.metrics:
            self.exchange_data[metric] = {point: {} for point in self.point_counts}
        
        # Add timestamp structure to track when data was last updated
        self.data_timestamps = {point: {} for point in self.point_counts}
        
        # Add structure to track the newest data timestamp from the database
        self.newest_data_timestamps = {exchange: {} for exchange in self.all_exchanges}

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

    def _build_query_for_partition_tables(self, tables, pair_name, start_time, end_time, exchange):
        """
        Build a complete UNION query for multiple partition tables.
        This creates a complete, valid SQL query with correct WHERE clauses.
        """
        if not tables:
            return ""
            
        union_parts = []
        
        for table in tables:
            # For Surf data (production)
            if exchange == 'surf':
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

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """
        Fetch data for Surf and Rollbit, analyze metrics, and calculate rankings.
        
        Args:
            conn: Database connection
            pairs_to_analyze: List of coin pairs to analyze
            hours: Hours to look back for data retrieval
        """
        # Always compare rollbit and surf
        exchanges_to_compare = ['rollbit', 'surf']
        primary_exchange = 'rollbit'
        secondary_exchange = 'surf'
        
        # Calculate times
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Set the current analysis time
        self.analysis_time = ensure_timezone_aware(datetime.now())
        st.session_state.last_analysis_time = self.analysis_time
        
        st.info(f"Retrieving data from the last {hours} hours")
        
        try:
            # Get relevant partition tables for this time range
            partition_tables = self._get_partition_tables(conn, start_time, end_time)
            
            if not partition_tables:
                st.error("No data tables available for the selected time range.")
                return None
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each pair for both exchanges
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress((i) / len(pairs_to_analyze))
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                # Process each exchange
                for exchange in exchanges_to_compare:
                    # Build appropriate query for this exchange
                    query = self._build_query_for_partition_tables(
                        partition_tables,
                        pair_name=pair,
                        start_time=start_time,
                        end_time=end_time,
                        exchange=exchange
                    )
                    
                    # Fetch data
                    if query:
                        try:
                            df = pd.read_sql_query(query, conn)
                            if len(df) > 0:
                                # Store the pair name in a way that's easier to reference later
                                coin_key = pair.replace('/', '_')
                                
                                # Store newest timestamp from database for this pair and exchange
                                newest_timestamp = df['timestamp'].max()
                                # Ensure timestamp is timezone aware
                                newest_timestamp = ensure_timezone_aware(newest_timestamp)
                                
                                if coin_key not in self.newest_data_timestamps[exchange]:
                                    self.newest_data_timestamps[exchange][coin_key] = newest_timestamp
                                
                                self._process_price_data(df, 'timestamp', 'price', coin_key, exchange)
                            else:
                                st.warning(f"No data found for {exchange.upper()}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            # Create comparison results
            comparison_results = self._create_comparison_results(primary_exchange, secondary_exchange)
            
            # Create individual rankings
            individual_rankings = {}
            for exchange in exchanges_to_compare:
                individual_rankings[exchange] = self._create_individual_rankings(exchange)
            
            return {
                'comparison_results': comparison_results,
                'individual_rankings': individual_rankings,
                'raw_data': self.exchange_data,
                'newest_data_timestamps': self.newest_data_timestamps,
                'analysis_time': self.analysis_time
            }
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
                    
                    # Get the timestamps for these points to know their time range
                    sample_timestamps = filtered_df[timestamp_col].iloc[-point_count:]
                    newest_timestamp = sample_timestamps.max()
                    oldest_timestamp = sample_timestamps.min()
                    
                    # Ensure timestamps are timezone aware
                    newest_timestamp = ensure_timezone_aware(newest_timestamp)
                    oldest_timestamp = ensure_timezone_aware(oldest_timestamp)
                    
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
                    
                    # Store timestamp information for this data point
                    if coin_key not in self.data_timestamps[point_count]:
                        self.data_timestamps[point_count][coin_key] = {}
                    
                    # Store both the calculation time and the time range of the data points
                    self.data_timestamps[point_count][coin_key][exchange] = {
                        'calculation_time': ensure_timezone_aware(datetime.now()),
                        'newest_data_time': newest_timestamp,
                        'oldest_data_time': oldest_timestamp,
                        'time_span_seconds': (newest_timestamp - oldest_timestamp).total_seconds()
                    }
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
    
    def _create_comparison_results(self, primary_exchange, secondary_exchange):
        """Create comparison results between the two exchanges for all point counts."""
        comparison_results = {}
        
        for point_count in self.point_counts:
            comparison_data = []
            
            # Get all coins that have data for both exchanges for any metric
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
                                relative_score = 100 if secondary_value == 0 else 200
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
                
                # Add timestamp information for this coin
                if coin in self.data_timestamps[point_count]:
                    # Get the newest data point timestamps for both exchanges if available
                    newest_times = []
                    time_spans = []
                    
                    if primary_exchange in self.data_timestamps[point_count][coin]:
                        primary_newest = self.data_timestamps[point_count][coin][primary_exchange]['newest_data_time']
                        primary_span = self.data_timestamps[point_count][coin][primary_exchange]['time_span_seconds']
                        newest_times.append(primary_newest)
                        time_spans.append(primary_span)
                    
                    if secondary_exchange in self.data_timestamps[point_count][coin]:
                        secondary_newest = self.data_timestamps[point_count][coin][secondary_exchange]['newest_data_time']
                        secondary_span = self.data_timestamps[point_count][coin][secondary_exchange]['time_span_seconds']
                        newest_times.append(secondary_newest)
                        time_spans.append(secondary_span)
                    
                    if newest_times:
                        # Use the most recent timestamp from either exchange
                        row['Data Timestamp'] = max(newest_times)
                        # Use the average time span
                        if time_spans:
                            row['Data Span (minutes)'] = sum(time_spans) / len(time_spans) / 60
                
                comparison_data.append(row)
            
            # Create DataFrame and sort by relative score (highest first)
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
                comparison_results[point_count] = comparison_df
            else:
                comparison_results[point_count] = None
        
        return comparison_results
    
    def _create_individual_rankings(self, exchange):
        """Create rankings for a specific exchange across all metrics and point counts."""
        rankings = {}
        
        for point_count in self.point_counts:
            point_rankings = {}
            
            for metric in self.metrics:
                # Get all coins that have data for this exchange and metric
                coin_data = {}
                timestamp_data = {}
                
                for coin, exchanges in self.exchange_data[metric][point_count].items():
                    if exchange in exchanges:
                        coin_data[coin] = exchanges[exchange]
                        # Get timestamp info if available
                        if coin in self.data_timestamps[point_count] and exchange in self.data_timestamps[point_count][coin]:
                            timestamp_data[coin] = self.data_timestamps[point_count][coin][exchange]['newest_data_time']
                
                if coin_data:
                    # Create DataFrame
                    df_data = {
                        'Coin': [coin.replace('_', '/') for coin in coin_data.keys()],
                        'Value': list(coin_data.values())
                    }
                    
                    # Add timestamps if available
                    if timestamp_data:
                        df_data['Data Timestamp'] = [timestamp_data.get(coin, None) for coin in coin_data.keys()]
                        current_time = ensure_timezone_aware(datetime.now())
                        df_data['Data Age (minutes)'] = [(current_time - timestamp_data.get(coin, current_time)).total_seconds() / 60 
                                                     for coin in coin_data.keys()]
                    
                    df = pd.DataFrame(df_data)
                    
                    # Sort based on metric (ascending or descending)
                    # For trend_strength, lower is better
                    # For all other metrics, higher is now better
                    ascending = True if metric == 'trend_strength' else False
                    
                    df = df.sort_values('Value', ascending=ascending)
                    
                    # Add rank column
                    df.insert(0, 'Rank', range(1, len(df) + 1))
                    
                    point_rankings[metric] = df
            
            rankings[point_count] = point_rankings
        
        return rankings
    
    def create_parameter_comparison_table(self):
        """
        Create a comprehensive table with all metrics for all pairs across all point counts.
        This is for tab 1 to display a huge table of parameters.
        """
        # Always use rollbit as primary and surf as secondary
        primary_exchange = 'rollbit'
        secondary_exchange = 'surf'
        
        # First, collect all coins that have data for any metric at any point count
        all_coins = set()
        for metric in self.metrics:
            for point_count in self.point_counts:
                for coin in self.exchange_data[metric][point_count].keys():
                    all_coins.add(coin)
        
        # Create a huge dataframe with all metrics
        rows = []
        for coin in sorted(all_coins):
            row = {'Coin': coin.replace('_', '/')}
            
            # Add metrics for each point count
            for point_count in self.point_counts:
                for metric in self.metrics:
                    # Check if we have primary exchange data
                    if (coin in self.exchange_data[metric][point_count] and 
                        primary_exchange in self.exchange_data[metric][point_count][coin]):
                        row[f'{self.metric_short_names[metric]} {primary_exchange.upper()} ({point_count})'] = self.exchange_data[metric][point_count][coin][primary_exchange]
                    else:
                        row[f'{self.metric_short_names[metric]} {primary_exchange.upper()} ({point_count})'] = None
                    
                    # Check if we have secondary exchange data
                    if (coin in self.exchange_data[metric][point_count] and 
                        secondary_exchange in self.exchange_data[metric][point_count][coin]):
                        row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()} ({point_count})'] = self.exchange_data[metric][point_count][coin][secondary_exchange]
                    else:
                        row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()} ({point_count})'] = None
                
                # Add timestamp information if available
                if coin in self.data_timestamps[point_count]:
                    if primary_exchange in self.data_timestamps[point_count][coin]:
                        row[f'Last Data {primary_exchange.upper()} ({point_count})'] = self.data_timestamps[point_count][coin][primary_exchange]['newest_data_time']
                    else:
                        row[f'Last Data {primary_exchange.upper()} ({point_count})'] = None
                        
                    if secondary_exchange in self.data_timestamps[point_count][coin]:
                        row[f'Last Data {secondary_exchange.upper()} ({point_count})'] = self.data_timestamps[point_count][coin][secondary_exchange]['newest_data_time']
                    else:
                        row[f'Last Data {secondary_exchange.upper()} ({point_count})'] = None
            
            rows.append(row)
        
        # Create the dataframe
        if rows:
            comparison_df = pd.DataFrame(rows)
            return comparison_df
        else:
            return None

    def get_data_freshness_info(self):
        """Get information about data freshness for display"""
        # Create a dictionary to hold freshness information
        freshness_info = {
            'newest_timestamps': {},
            'oldest_timestamps': {},
            'time_spans': {}
        }
        
        # Calculate for each point count
        for point_count in self.point_counts:
            newest_timestamps = {}
            oldest_timestamps = {}
            time_spans = {}
            
            for coin, exchanges in self.data_timestamps[point_count].items():
                coin_name = coin.replace('_', '/')
                newest_timestamps[coin_name] = {}
                oldest_timestamps[coin_name] = {}
                time_spans[coin_name] = {}
                
                for exchange, timestamps in exchanges.items():
                    newest_timestamps[coin_name][exchange] = timestamps['newest_data_time']
                    oldest_timestamps[coin_name][exchange] = timestamps['oldest_data_time']
                    time_spans[coin_name][exchange] = timestamps['time_span_seconds'] / 60  # Convert to minutes
            
            freshness_info['newest_timestamps'][point_count] = newest_timestamps
            freshness_info['oldest_timestamps'][point_count] = oldest_timestamps
            freshness_info['time_spans'][point_count] = time_spans
        
        return freshness_info


# Setup sidebar with simplified options
# Replace the quick selection buttons with form-compatible elements
# Here's what you need to change in the form section:

with st.sidebar:
    st.header("Analysis Parameters")
    all_pairs = [
            "PEPE/USDT", "PAXG/USDT", "DOGE/USDT", "BTC/USDT", "EOS/USDT",
            "BNB/USDT", "MERL/USDT", "FHE/USDT", "IP/USDT", "ORCA/USDT",
            "TRUMP/USDT", "LIBRA/USDT", "AI16Z/USDT", "OM/USDT", "TRX/USDT",
            "S/USDT", "PI/USDT", "JUP/USDT", "BABY/USDT", "PARTI/USDT",
            "ADA/USDT", "HYPE/USDT", "VIRTUAL/USDT", "SUI/USDT", "SATS/USDT",
            "XRP/USDT", "ORDI/USDT", "WIF/USDT", "VANA/USDT", "PENGU/USDT",
            "VINE/USDT", "GRIFFAIN/USDT", "MEW/USDT", "POPCAT/USDT", "FARTCOIN/USDT",
            "TON/USDT", "MELANIA/USDT", "SOL/USDT", "PNUT/USDT", "CAKE/USDT",
            "TST/USDT", "ETH/USDT"
        ]
    
    # Initialize session state for selections if not present
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = ["ETH/USDT", "BTC/USDT"]  # Default selection
    
    # Create buttons OUTSIDE the form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Select Major Coins"):
            st.session_state.selected_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
            st.rerun()
            
    with col2:
        if st.button("Select All"):
            st.session_state.selected_pairs = all_pairs
            st.rerun()
            
    with col3:
        if st.button("Clear Selection"):
            st.session_state.selected_pairs = []
            st.rerun()
    
    # Then create the form without the buttons inside
    with st.form("exchange_comparison_form"):
        # Data retrieval window
        hours = st.number_input(
            "Hours to Look Back (for data retrieval)",
            min_value=1,
            max_value=168,
            value=8,
            help="How many hours of historical data to retrieve. This ensures enough data for point-based analysis."
        )
        
        st.info("Analysis will be performed on the most recent data points: 500, 5000, and 50000 points regardless of time span.")
        
        # Create multiselect for pairs
        selected_pairs = st.multiselect(
            "Select Pairs to Analyze",
            options=all_pairs,
            default=st.session_state.selected_pairs,
            help="Select one or more cryptocurrency pairs to analyze"
        )
        
        # Update session state
        st.session_state.selected_pairs = selected_pairs
        
        # Set the pairs variable for the analyzer
        pairs = selected_pairs
        
        # Show a warning if no pairs are selected
        if not pairs:
            st.warning("Please select at least one pair to analyze.")
        
        # Submit button - this should be indented at the same level as other elements in the form
        submit_button = st.form_submit_button("Analyze Exchanges")

# Display the last analysis time if available
if st.session_state.last_analysis_time:
    last_analysis_time_sg = st.session_state.last_analysis_time.astimezone(singapore_timezone)
    st.info(f"Last Analysis Time: {last_analysis_time_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)")

# When form is submitted
if submit_button:
    if not conn:
        st.error("Database connection not available.")
    elif not pairs:
        st.error("Please enter at least one pair to analyze.")
    else:
        # Initialize analyzer
        analyzer = ExchangeAnalyzer()
        
        # Run analysis
        st.header("Comparing ROLLBIT vs SURF")
        
        with st.spinner("Fetching and analyzing data..."):
            results = analyzer.fetch_and_analyze(
                conn=conn,
                pairs_to_analyze=pairs,
                hours=hours
            )
        
        if results:
            # Display data freshness information
            with st.expander("📊 Data Freshness Information", expanded=True):
                st.subheader("Data Freshness")
                
                # Create a DataFrame with the timestamp information for SOL/USDT
                sol_key = "SOL/USDT" if "SOL/USDT" in pairs else None
                
                if sol_key:
                    sol_freshness_data = []
                    for point_count in analyzer.point_counts:
                        sol_timestamps = analyzer.data_timestamps[point_count].get('SOL_USDT', {})
                        
                        for exchange in ['rollbit', 'surf']:
                            if exchange in sol_timestamps:
                                timestamps = sol_timestamps[exchange]
                                newest_time = timestamps['newest_data_time']
                                oldest_time = timestamps['oldest_data_time']
                                time_span_minutes = timestamps['time_span_seconds'] / 60
                                
                                # Calculate age of the data in minutes
                                # Ensure both datetimes are timezone-aware for comparison
                                current_time = ensure_timezone_aware(datetime.now())
                                data_age_minutes = (current_time - newest_time).total_seconds() / 60
                                
                                sol_freshness_data.append({
                                    'Exchange': exchange.upper(),
                                    'Point Count': point_count,
                                    'Newest Data Time': newest_time,
                                    'Oldest Data Time': oldest_time,
                                    'Time Span (minutes)': time_span_minutes,
                                    'Data Age (minutes)': data_age_minutes
                                })
                    
                    if sol_freshness_data:
                        sol_freshness_df = pd.DataFrame(sol_freshness_data)
                        
                        # Style the dataframe with colors based on data age
                        def highlight_freshness(s):
                            is_age_column = s.name == 'Data Age (minutes)'
                            return ['background-color: #a0d995' if is_age_column and v < 1 else
                                   'background-color: #f1f1aa' if is_age_column and v < 5 else
                                   'background-color: #ffc299' if is_age_column else
                                   '' for v in s]
                        
                        st.write("SOL/USDT Data Freshness")
                        st.dataframe(sol_freshness_df.style.apply(highlight_freshness), use_container_width=True)
                    else:
                        st.write("No SOL/USDT data available")
                
                # Show a summary for all other pairs
                freshness_summary = []
                current_time = ensure_timezone_aware(datetime.now())
                
                for pair in pairs:
                    pair_key = pair.replace('/', '_')
                    for point_count in analyzer.point_counts:
                        if pair_key in analyzer.data_timestamps[point_count]:
                            for exchange, timestamps in analyzer.data_timestamps[point_count][pair_key].items():
                                newest_time = timestamps['newest_data_time']
                                data_age_minutes = (current_time - newest_time).total_seconds() / 60
                                time_span_minutes = timestamps['time_span_seconds'] / 60
                                
                                freshness_summary.append({
                                    'Pair': pair,
                                    'Exchange': exchange.upper(),
                                    'Points': point_count,
                                    'Last Update': newest_time,
                                    'Data Age (minutes)': data_age_minutes,
                                    'Time Span (minutes)': time_span_minutes
                                })
                
                if freshness_summary:
                    freshness_df = pd.DataFrame(freshness_summary)
                    
                    # Define styling function for the dataframe
                    def color_age(s):
                        is_age_column = s.name == 'Data Age (minutes)'
                        return ['background-color: #a0d995' if is_age_column and v < 1 else
                               'background-color: #f1f1aa' if is_age_column and v < 5 else
                               'background-color: #ffc299' if is_age_column else
                               '' for v in s]
                    
                    st.write("All Pairs Data Freshness")
                    st.dataframe(
                        freshness_df.sort_values(['Pair', 'Points']).style.apply(color_age),
                        height=300,
                        use_container_width=True
                    )
                    
                    # Create a download button for the freshness data
                    csv = freshness_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Freshness Data CSV",
                        data=csv,
                        file_name="data_freshness.csv",
                        mime="text/csv"
                    )
            
            # Create parameter comparison table for Tab 1
            with tab1:
                st.header("Parameter Comparison Table")
                st.write("This table shows all metrics for all pairs across different point counts.")
                
                comparison_table = analyzer.create_parameter_comparison_table()
                
                if comparison_table is not None:
                    # Style the table
                    def style_comparison_table(val):
                        """Style cells, highlight differences."""
                        if pd.isna(val):
                            return 'background-color: #f2f2f2'  # Light gray for missing values
                        return ''
                    
                    # Display the table with horizontal scrolling
                    st.dataframe(
                        comparison_table.style.applymap(style_comparison_table),
                        height=600,
                        use_container_width=True,
                    )
                    
                    # Add download button for the table
                    csv = comparison_table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"parameter_comparison_rollbit_vs_surf.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No comparison data available.")
            
            # Rankings and analysis for Tab 2
            with tab2:
                st.header("Rankings & Analysis")
                
                if results['comparison_results']:
                    # Create subtabs for different point counts
                    point_count_tabs = st.tabs([f"{count} Points" for count in analyzer.point_counts if count in results['comparison_results']])
                    
                    for i, point_count in enumerate([pc for pc in analyzer.point_counts if pc in results['comparison_results']]):
                        with point_count_tabs[i]:
                            df = results['comparison_results'][point_count]
                            
                            if df is not None and not df.empty:
                                # Style the DataFrame for relative scores
                                def highlight_scores(s):
                                    is_score_column = isinstance(s.name, str) and 'Score' in s.name
                                    return ['background-color: #60b33c; color: white; font-weight: bold' if is_score_column and v > 130 else
                                           'background-color: #a0d995; color: black' if is_score_column and v > 110 else
                                           'background-color: #f1f1aa; color: black' if is_score_column and v > 90 else
                                           'background-color: #ffc299; color: black' if is_score_column and v > 70 else
                                           'background-color: #ff8080; color: black; font-weight: bold' if is_score_column else
                                           '' for v in s]
                                
                                # Display the data
                                st.subheader(f"Relative Performance: {point_count} Points")
                                
                                # Add column with data age if available
                                if 'Data Timestamp' in df.columns:
                                    current_time = ensure_timezone_aware(datetime.now())
                                    df['Data Age (minutes)'] = [(current_time - ts).total_seconds() / 60 
                                                           if not pd.isna(ts) else None 
                                                           for ts in df['Data Timestamp']]
                                
                                st.dataframe(
                                    df.style.apply(highlight_scores),
                                    height=400,
                                    use_container_width=True
                                )
                                
                                # Create visualization
                                st.subheader(f"Top and Bottom Performers")
                                
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
                                    title=f"SURF Performance Relative to ROLLBIT (100 = Equal)",
                                    xaxis_title="Coin",
                                    yaxis_title="Relative Performance Score",
                                    barmode='group',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add metric-by-metric analysis
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Individual rankings for Rollbit
                                    st.subheader("ROLLBIT Rankings")
                                    
                                    if point_count in results['individual_rankings']['rollbit']:
                                        # Create subtabs for each metric
                                        metric_tabs_primary = st.tabs([analyzer.metric_display_names[m] for m in analyzer.metrics 
                                                              if m in results['individual_rankings']['rollbit'][point_count]])
                                        
                                        for j, metric in enumerate([m for m in analyzer.metrics if m in results['individual_rankings']['rollbit'][point_count]]):
                                            with metric_tabs_primary[j]:
                                                metric_df = results['individual_rankings']['rollbit'][point_count][metric]
                                                if not metric_df.empty:
                                                    # Style by data age if available
                                                    def color_data_age(s):
                                                        is_age_column = s.name == 'Data Age (minutes)'
                                                        return ['background-color: #a0d995' if is_age_column and v < 1 else
                                                               'background-color: #f1f1aa' if is_age_column and v < 5 else
                                                               'background-color: #ffc299' if is_age_column else
                                                               '' for v in s]
                                                    
                                                    if 'Data Age (minutes)' in metric_df.columns:
                                                        st.dataframe(metric_df.style.apply(color_data_age), height=300, use_container_width=True)
                                                    else:
                                                        st.dataframe(metric_df, height=300, use_container_width=True)
                                                    
                                                    # Bar chart of top 10
                                                    top_10_metric = metric_df.head(10)
                                                    fig = px.bar(
                                                        top_10_metric, 
                                                        x='Coin', 
                                                        y='Value',
                                                        title=f"Top 10 by {analyzer.metric_display_names[metric]}",
                                                        color='Rank',
                                                        color_continuous_scale='Viridis_r'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"No ranking data available for ROLLBIT at {point_count} points")
                                
                                with col2:
                                    # Individual rankings for Surf
                                    st.subheader("SURF Rankings")
                                    
                                    if point_count in results['individual_rankings']['surf']:
                                        # Create subtabs for each metric
                                        metric_tabs_secondary = st.tabs([analyzer.metric_display_names[m] for m in analyzer.metrics 
                                                              if m in results['individual_rankings']['surf'][point_count]])
                                        
                                        for j, metric in enumerate([m for m in analyzer.metrics if m in results['individual_rankings']['surf'][point_count]]):
                                            with metric_tabs_secondary[j]:
                                                metric_df = results['individual_rankings']['surf'][point_count][metric]
                                                if not metric_df.empty:
                                                    # Style by data age if available
                                                    def color_data_age(s):
                                                        is_age_column = s.name == 'Data Age (minutes)'
                                                        return ['background-color: #a0d995' if is_age_column and v < 1 else
                                                               'background-color: #f1f1aa' if is_age_column and v < 5 else
                                                               'background-color: #ffc299' if is_age_column else
                                                               '' for v in s]
                                                    
                                                    if 'Data Age (minutes)' in metric_df.columns:
                                                        st.dataframe(metric_df.style.apply(color_data_age), height=300, use_container_width=True)
                                                    else:
                                                        st.dataframe(metric_df, height=300, use_container_width=True)
                                                    
                                                    # Bar chart of top 10
                                                    top_10_metric = metric_df.head(10)
                                                    fig = px.bar(
                                                        top_10_metric, 
                                                        x='Coin', 
                                                        y='Value',
                                                        title=f"Top 10 by {analyzer.metric_display_names[metric]}",
                                                        color='Rank',
                                                        color_continuous_scale='Viridis_r'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"No ranking data available for SURF at {point_count} points")
                            else:
                                st.warning(f"No data available for {point_count} points")
                else:
                    st.warning("No comparison results available.")
        else:
            st.error("Failed to analyze data. Please try again with different parameters.")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This dashboard analyzes cryptocurrency prices between Rollbit and Surf exchanges and calculates various metrics:

- **Direction Changes (%)**: Frequency of price reversals
- **Choppiness**: Measures price oscillation within a range
- **Tick ATR %**: Average True Range as percentage of mean price
- **Trend Strength**: Measures directional price strength

The dashboard compares these metrics and provides rankings and visualizations for various point counts (500, 5000, and 50000).
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")