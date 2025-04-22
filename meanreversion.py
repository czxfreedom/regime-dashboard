import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz
from scipy import stats
import statsmodels.api as sm  # Added missing import
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Mean Reversion Monitor",
    page_icon="📊",
    layout="wide"
)

# Create session state variables for caching if not already present
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'last_selected_pairs' not in st.session_state:
    st.session_state.last_selected_pairs = []
if 'last_hours' not in st.session_state:
    st.session_state.last_hours = 0
if 'selected_historical_exchange' not in st.session_state:
    st.session_state.selected_historical_exchange = None
if 'selected_historical_pair' not in st.session_state:
    st.session_state.selected_historical_pair = None
if 'pnl_data_cache' not in st.session_state:
    st.session_state.pnl_data_cache = {}

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
st.title("Crypto Mean Reversion Monitor")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Debug current time
st.write(f"UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Current Status", "Historical Trends", "PNL Correlation"])

class MeanReversionAnalyzer:
    """Analyzer for mean reversion behaviors in cryptocurrency prices"""
    
    def __init__(self):
        self.data = {}  # Main data storage
        self.time_series_data = {}  # For tracking metrics over time
        self.exchanges = ['rollbit', 'surf']  # Exchanges to analyze
        self.lookback_hours = 24  # Hours to look back for historical trends
        
        # Metrics for mean reversion analysis
        self.metrics = [
            'direction_changes_30min',  # A: Number of direction changes in last 30 min
            'absolute_range_pct',       # B: (local high - local low) / local low
            'dc_range_ratio',           # C: A/B ratio
            'hurst_exponent'            # D: Hurst exponent
        ]
        
        # Display names for metrics
        self.metric_display_names = {
            'direction_changes_30min': 'Direction Changes (30min)',
            'absolute_range_pct': 'Range %',
            'dc_range_ratio': 'Dir Changes/Range Ratio',
            'hurst_exponent': 'Hurst Exponent'
        }
        
        # Short names for metrics
        self.metric_short_names = {
            'direction_changes_30min': 'Dir Chg 30m',
            'absolute_range_pct': 'Range %',
            'dc_range_ratio': 'DC/Range',
            'hurst_exponent': 'Hurst'
        }
        
        # Optimal values (for highlighting)
        self.optimal_values = {
            'direction_changes_30min': 'higher',  # More changes = more mean reversion
            'absolute_range_pct': 'lower',        # Smaller range = more stable
            'dc_range_ratio': 'higher',           # Higher ratio = more mean reversion
            'hurst_exponent': 'lower'             # <0.5 indicates mean reversion
        }
    
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
                    created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
                    final_price AS price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_time}'::timestamp
                    AND created_at <= '{end_time}'::timestamp
                    AND source_type = 0
                    AND pair_name = '{pair_name}'
                """
            else:
                # For Rollbit data
                query = f"""
                SELECT 
                    pair_name,
                    created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
                    final_price AS price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_time}'::timestamp
                    AND created_at <= '{end_time}'::timestamp
                    AND source_type = 1
                    AND pair_name = '{pair_name}'
                """
            
            union_parts.append(query)
        
        # Join with UNION and add ORDER BY at the end
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
        return complete_query

    def calculate_hurst_exponent(self, prices, min_k=5, max_k=None):
        """
        Calculate the Hurst exponent, which indicates:
        H < 0.5: Mean-reverting series
        H = 0.5: Random walk
        H > 0.5: Trending series
        
        Lowered min_k to 5 (from 10) to allow calculation with smaller data samples
        """
        try:
            # Convert to numpy array and ensure we have enough data
            prices = np.array(prices)
            if len(prices) < 50:  # Reduced minimum requirement from 100 to 50
                return 0.5  # Return neutral value if not enough data
                
            # Calculate returns
            returns = np.log(prices[1:] / prices[:-1])
            
            # Set default max_k if not provided
            if max_k is None:
                max_k = min(int(len(returns) / 4), 120)  # Use at most 120 or 1/4 of series length (more aggressive)
            
            if max_k <= min_k:
                max_k = min_k + 5  # Smaller window size
                
            # Create a range of k values
            k_values = list(range(min_k, max_k))
            
            # Calculate rescaled range for each k value
            rs_values = []
            
            for k in k_values:
                # Split returns into segments of length k
                segments = len(returns) // k
                if segments == 0:
                    continue
                
                # Calculate R/S for each segment and take average
                rs_array = []
                
                for i in range(segments):
                    segment = returns[i*k:(i+1)*k]
                    mean = np.mean(segment)
                    
                    # Calculate cumulative deviation
                    deviation = np.cumsum(segment - mean)
                    
                    # Calculate range and standard deviation
                    r = max(deviation) - min(deviation)
                    s = np.std(segment)
                    
                    # Calculate R/S value (avoid division by zero)
                    if s > 0:
                        rs = r / s
                        rs_array.append(rs)
                
                if rs_array:
                    # Append average R/S value for this k
                    rs_values.append(np.mean(rs_array))
                    
            # If we don't have enough values, return default
            if len(rs_values) < 3:  # Reduced from 5 to 3
                return 0.5
                
            # Perform regression to estimate Hurst exponent
            log_k = np.log10(k_values[:len(rs_values)])
            log_rs = np.log10(rs_values)
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_k, log_rs)
            
            # Return Hurst exponent (the slope of the regression line)
            hurst = slope
            
            # Cap at reasonable values
            if hurst < 0:
                hurst = 0
            elif hurst > 1:
                hurst = 1
                
            return hurst
        except Exception as e:
            st.error(f"Error calculating Hurst exponent: {e}")
            return 0.5  # Return neutral value on error

    def calculate_direction_changes(self, prices):
        """Calculate the number of times the price direction changes."""
        try:
            price_changes = np.diff(prices)
            signs = np.sign(price_changes)
            direction_changes = np.sum(signs[1:] != signs[:-1])
            return int(direction_changes)
        except Exception as e:
            return 0

    def calculate_range_percentage(self, prices):
        """Calculate (local high - local low) / local low as percentage."""
        try:
            local_high = np.max(prices)
            local_low = np.min(prices)
            if local_low > 0:  # Avoid division by zero
                range_pct = ((local_high - local_low) / local_low) * 100
                return range_pct
            return 0
        except Exception as e:
            return 0

    def process_30min_data(self, df, pair_name, exchange):
        """Process the last 30 minutes of data for mean reversion metrics."""
        try:
            # Ensure df is sorted by timestamp
            df = df.sort_values('timestamp')
            
            # Get the latest 30 minutes of data
            end_time = df['timestamp'].max()
            start_time = end_time - pd.Timedelta(minutes=30)
            df_30min = df[df['timestamp'] >= start_time]
            
            if len(df_30min) < 10:  # Need at least 10 data points for meaningful analysis
                return None
                
            # Extract prices
            prices = pd.to_numeric(df_30min['price'], errors='coerce').dropna().values
            
            # Calculate metrics
            direction_changes = self.calculate_direction_changes(prices)
            range_pct = self.calculate_range_percentage(prices)
            
            # Calculate A/B ratio (direction changes / range %)
            # Use a small epsilon to avoid division by zero
            dc_range_ratio = direction_changes / (range_pct + 1e-10)
            
            # Calculate Hurst exponent on a longer timeframe for better accuracy
            # Use at least the last 1000 data points if available
            all_prices = pd.to_numeric(df['price'], errors='coerce').dropna().values
            hurst_window = min(1000, len(all_prices))
            hurst = self.calculate_hurst_exponent(all_prices[-hurst_window:])
            
            # Store results
            if pair_name not in self.data:
                self.data[pair_name] = {}
            if exchange not in self.data[pair_name]:
                self.data[pair_name][exchange] = {}
                
            # Store current metrics
            self.data[pair_name][exchange]['direction_changes_30min'] = direction_changes
            self.data[pair_name][exchange]['absolute_range_pct'] = range_pct
            self.data[pair_name][exchange]['dc_range_ratio'] = dc_range_ratio
            self.data[pair_name][exchange]['hurst_exponent'] = hurst
            
            # Create timestamp for this measurement
            timestamp = end_time
            
            # Initialize time series if needed
            if pair_name not in self.time_series_data:
                self.time_series_data[pair_name] = {}
            if exchange not in self.time_series_data[pair_name]:
                self.time_series_data[pair_name][exchange] = []
                
            # Add current metrics to time series
            self.time_series_data[pair_name][exchange].append({
                'timestamp': timestamp,
                'direction_changes_30min': direction_changes,
                'absolute_range_pct': range_pct,
                'dc_range_ratio': dc_range_ratio,
                'hurst_exponent': hurst
            })
            
            return {
                'pair': pair_name,
                'exchange': exchange,
                'direction_changes_30min': direction_changes,
                'absolute_range_pct': range_pct,
                'dc_range_ratio': dc_range_ratio,
                'hurst_exponent': hurst
            }
        except Exception as e:
            st.error(f"Error processing 30min data for {pair_name} on {exchange}: {e}")
            return None

    def process_historical_data(self, df, pair_name, exchange):
        """Process historical data to generate 30-minute interval metrics for the last 24 hours."""
        try:
            # Ensure df is sorted by timestamp
            df = df.sort_values('timestamp')
            
            # Define the start and end times
            end_time = df['timestamp'].max()
            start_time = end_time - pd.Timedelta(hours=self.lookback_hours)
            
            # Filter data to the lookback period
            df_lookback = df[df['timestamp'] >= start_time]
            
            if len(df_lookback) < 50:  # Reduced from 100 to 50
                st.warning(f"Not enough historical data for {pair_name} on {exchange}: {len(df_lookback)} points")
                return
                
            # Create 30-minute intervals
            intervals = pd.date_range(start=start_time, end=end_time, freq='30min')
            
            # Initialize time series if needed
            if pair_name not in self.time_series_data:
                self.time_series_data[pair_name] = {}
            if exchange not in self.time_series_data[pair_name]:
                self.time_series_data[pair_name][exchange] = []
                
            # Process each interval
            for i in range(len(intervals) - 1):
                interval_start = intervals[i]
                interval_end = intervals[i+1]
                
                # Get data for this interval
                df_interval = df_lookback[(df_lookback['timestamp'] >= interval_start) & 
                                         (df_lookback['timestamp'] < interval_end)]
                
                if len(df_interval) < 5:  # Reduced from 10 to 5
                    continue
                    
                # Extract prices
                prices = pd.to_numeric(df_interval['price'], errors='coerce').dropna().values
                
                # Skip if no valid prices
                if len(prices) < 5:
                    continue
                
                # Calculate metrics
                direction_changes = self.calculate_direction_changes(prices)
                range_pct = self.calculate_range_percentage(prices)
                
                # Calculate ratio
                dc_range_ratio = direction_changes / (range_pct + 1e-10)
                
                # Calculate Hurst exponent
                # For historical data, use the available data to calculate Hurst
                hurst = self.calculate_hurst_exponent(prices, min_k=3, max_k=min(len(prices)//3, 20))
                
                # Add to time series
                self.time_series_data[pair_name][exchange].append({
                    'timestamp': interval_end,
                    'direction_changes_30min': direction_changes,
                    'absolute_range_pct': range_pct,
                    'dc_range_ratio': dc_range_ratio,
                    'hurst_exponent': hurst
                })
        except Exception as e:
            st.error(f"Error processing historical data for {pair_name} on {exchange}: {e}")

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """
        Fetch data for specified pairs and exchanges, analyze mean reversion metrics.
        
        Args:
            conn: Database connection
            pairs_to_analyze: List of coin pairs to analyze
            hours: Hours to look back for data retrieval
        """
        # Calculate times - using exact current time to get the most recent data
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        st.info(f"Retrieving data from {start_time} to {end_time} ({hours} hours)")
        
        try:
            # Get relevant partition tables for this time range
            partition_tables = self._get_partition_tables(conn, start_time, end_time)
            
            if not partition_tables:
                st.error("No data tables available for the selected time range.")
                return None
            
            st.write(f"Found {len(partition_tables)} partition tables: {', '.join(partition_tables)}")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processed results
            results = []
            
            # Process each pair for both exchanges
            for i, pair in enumerate(pairs_to_analyze):
                progress_percentage = (i) / len(pairs_to_analyze)
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                # Process each exchange
                for exchange in self.exchanges:
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
                            
                            # Debug timestamp range
                            if len(df) > 0:
                                min_time = df['timestamp'].min()
                                max_time = df['timestamp'].max()
                                st.write(f"{exchange.upper()} {pair} data range: {min_time} to {max_time} ({len(df)} points)")
                            
                            if len(df) > 0:
                                # Process current 30-min data
                                result = self.process_30min_data(df, pair, exchange)
                                if result:
                                    results.append(result)
                                
                                # Process historical data for time series
                                self.process_historical_data(df, pair, exchange)
                            else:
                                st.warning(f"No data found for {exchange.upper()}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            return results
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_current_status_table(self, results, exchange_filter=None):
        """Create a table showing current mean reversion metrics for all pairs."""
        if not results:
            return None
            
        # Filter by exchange if specified
        if exchange_filter:
            filtered_results = [r for r in results if r['exchange'] == exchange_filter]
        else:
            filtered_results = results
            
        # Convert to DataFrame
        df = pd.DataFrame(filtered_results)
        
        # Add a mean reversion score (composite metric)
        # Lower Hurst is better, higher dc_range_ratio is better
        df['mean_reversion_score'] = (
            (1 - df['hurst_exponent']) * 50 +  # Invert Hurst (lower is better for mean reversion)
            np.log1p(df['dc_range_ratio']) * 25  # Log transform to handle outliers
        )
        
        # Sort by mean reversion score (descending)
        df = df.sort_values('mean_reversion_score', ascending=False)
        
        return df

# Function to fetch PNL data for a specific pair
def fetch_platform_pnl_for_pair(pair_name, hours=24):
    """Fetch platform PNL data for a specific pair over the past hours"""
    # Check if already cached
    cache_key = f"{pair_name}_{hours}"
    if cache_key in st.session_state.pnl_data_cache:
        return st.session_state.pnl_data_cache[cache_key]
    
    # Set up time range
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(hours=hours)
    
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Query for platform PNL data at 30-minute intervals
    query = f"""
    WITH time_intervals AS (
      -- Generate 30-minute intervals for the past hours
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 30),
          '{end_time_utc}'::timestamp,
          INTERVAL '30 minutes'
        ) AS "UTC+8"
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" IN (0, 1, 2, 3, 4)
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    fee_data AS (
      -- Calculate user fee payments
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    funding_pnl AS (
      -- Calculate platform funding fee PNL
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" = 0
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    ),
    
    rebate_data AS (
      -- Calculate platform rebate payments
      SELECT
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30) AS "timestamp",
        COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
      FROM
        "public"."user_cashbooks"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "remark" = '给邀请人返佣'
      GROUP BY
        date_trunc('hour', "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 30)
    )
    
    -- Final query: combine all data sources
    SELECT
      t."UTC+8" AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp",
      COALESCE(o."platform_order_pnl", 0) +
      COALESCE(f."user_fee_payments", 0) +
      COALESCE(ff."platform_funding_pnl", 0) +
      COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
    FROM
      time_intervals t
    LEFT JOIN
      order_pnl o ON t."UTC+8" = o."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      fee_data f ON t."UTC+8" = f."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      funding_pnl ff ON t."UTC+8" = ff."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      rebate_data r ON t."UTC+8" = r."timestamp" AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      t."UTC+8" ASC
    """
    
    try:
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            st.warning(f"No PNL data found for {pair_name}")
            return None
        
        # Convert timestamp to pandas datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Cache the result
        st.session_state.pnl_data_cache[cache_key] = df
        
        return df
    except Exception as e:
        st.error(f"Error fetching PNL data for {pair_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Setup sidebar with simplified options
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
    
    # Create the form
    with st.form("mean_reversion_form"):
        # Data retrieval window
        hours = st.number_input(
            "Hours to Look Back",
            min_value=1,
            max_value=168,
            value=24,
            help="How many hours of historical data to retrieve for analysis."
        )
        
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
        
        # Exchange selection
        exchange_filter = st.radio(
            "Select Exchange",
            ["Both", "rollbit", "surf"],
            index=0,
            help="Filter results by exchange"
        )
        
        # Show a warning if no pairs are selected
        if not pairs:
            st.warning("Please select at least one pair to analyze.")
        
        # Submit button
        submit_button = st.form_submit_button("Analyze Mean Reversion")

# Check if we need to run analysis
should_run_analysis = (
    submit_button and 
    conn is not None and 
    len(pairs) > 0 and 
    (not st.session_state.data_processed or 
     st.session_state.last_selected_pairs != pairs or 
     st.session_state.last_hours != hours)
)

# Run analysis if needed
if should_run_analysis:
    # Initialize analyzer
    analyzer = MeanReversionAnalyzer()
    
    # Store analyzer in session state
    st.session_state.analyzer = analyzer
    
    # Run analysis
    with st.spinner("Fetching and analyzing data..."):
        results = analyzer.fetch_and_analyze(
            conn=conn,
            pairs_to_analyze=pairs,
            hours=hours
        )
        
        # Store analysis results and state
        if results:
            st.session_state.analysis_results = results
            st.session_state.data_processed = True
            st.session_state.last_selected_pairs = pairs.copy()
            st.session_state.last_hours = hours
            
            # Clear PNL data cache when new analysis is run
            st.session_state.pnl_data_cache = {}
        else:
            st.error("No results returned from analysis.")
            st.session_state.data_processed = False

# Only proceed if we have processed data
if st.session_state.data_processed and st.session_state.analysis_results:
    # Get the saved analyzer and results
    analyzer = st.session_state.analyzer
    results = st.session_state.analysis_results
    
    # Tab 1: Current Status
    with tab1:
        st.header("Current Mean Reversion Status")
        
        # Convert exchange filter for data filtering
        filter_value = None if exchange_filter == "Both" else exchange_filter
        
        # Create current status table
        status_df = analyzer.create_current_status_table(results, filter_value)
        
        if status_df is not None:
            # Style the table to highlight mean reversion conditions
            def style_mean_reversion_table(df):
                # Create a DataFrame of styles
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                
                # Apply styles for hurst_exponent
                if 'hurst_exponent' in df.columns:
                    styles.loc[df['hurst_exponent'] < 0.4, 'hurst_exponent'] = 'background-color: #a0d995; color: black'  # Strong mean reversion (green)
                    styles.loc[(df['hurst_exponent'] >= 0.4) & (df['hurst_exponent'] < 0.5), 'hurst_exponent'] = 'background-color: #f1f1aa; color: black'  # Mild mean reversion (yellow)
                    styles.loc[df['hurst_exponent'] >= 0.5, 'hurst_exponent'] = 'background-color: #ffc299; color: black'  # No mean reversion (orange)
                
                # Apply styles for dc_range_ratio
                if 'dc_range_ratio' in df.columns:
                    styles.loc[df['dc_range_ratio'] > 5, 'dc_range_ratio'] = 'background-color: #a0d995; color: black'  # High ratio (green)
                    styles.loc[(df['dc_range_ratio'] > 2) & (df['dc_range_ratio'] <= 5), 'dc_range_ratio'] = 'background-color: #f1f1aa; color: black'  # Medium ratio (yellow)
                    styles.loc[df['dc_range_ratio'] <= 2, 'dc_range_ratio'] = 'background-color: #ffc299; color: black'  # Low ratio (orange)
                
                # Apply styles for mean_reversion_score
                if 'mean_reversion_score' in df.columns:
                    styles.loc[df['mean_reversion_score'] > 50, 'mean_reversion_score'] = 'background-color: #60b33c; color: white; font-weight: bold'  # Strong (green)
                    styles.loc[(df['mean_reversion_score'] > 40) & (df['mean_reversion_score'] <= 50), 'mean_reversion_score'] = 'background-color: #a0d995; color: black'  # Good (light green)
                    styles.loc[(df['mean_reversion_score'] > 30) & (df['mean_reversion_score'] <= 40), 'mean_reversion_score'] = 'background-color: #f1f1aa; color: black'  # Moderate (yellow)
                    styles.loc[df['mean_reversion_score'] <= 30, 'mean_reversion_score'] = 'background-color: #ffc299; color: black'  # Weak (orange)
                
                return styles
            
            # Define columns to display
            display_cols = ['pair', 'exchange', 'direction_changes_30min', 'absolute_range_pct', 
                           'dc_range_ratio', 'hurst_exponent', 'mean_reversion_score']
            
            # Display the table with styling
            st.dataframe(
                status_df[display_cols].style.apply(style_mean_reversion_table, axis=None),
                height=600,
                use_container_width=True,
            )
            
            # Display summary of top mean reverting pairs
            st.subheader("Top Mean Reverting Pairs")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create bar chart of top 10 by mean reversion score
                top_10 = status_df.head(10)
                fig = px.bar(
                    top_10, 
                    x='pair', 
                    y='mean_reversion_score',
                    color='exchange',
                    title="Top 10 Pairs by Mean Reversion Score",
                    labels={'mean_reversion_score': 'Mean Reversion Score', 'pair': 'Pair'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Create scatter plot of Hurst vs DC/Range Ratio
                fig = px.scatter(
                    status_df, 
                    x='hurst_exponent', 
                    y='dc_range_ratio',
                    color='mean_reversion_score',
                    size='direction_changes_30min',
                    hover_name='pair',
                    title="Hurst Exponent vs. DC/Range Ratio",
                    labels={
                        'hurst_exponent': 'Hurst Exponent', 
                        'dc_range_ratio': 'Direction Changes / Range Ratio'
                    },
                    color_continuous_scale='Viridis'
                )
                
                # Add vertical line at Hurst = 0.5
                fig.add_shape(
                    type="line",
                    x0=0.5,
                    y0=0,
                    x1=0.5,
                    y1=status_df['dc_range_ratio'].max() * 1.1,
                    line=dict(
                        color="red",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Add annotation for Hurst reference
                fig.add_annotation(
                    x=0.5,
                    y=0,
                    text="Random Walk (H=0.5)",
                    showarrow=False,
                    xshift=5,
                    yshift=-20
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the table
            csv = status_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"mean_reversion_status_{now_sg.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No mean reversion data available.")
    
    # Tab 2: Historical Trends - completely rewritten for better user experience
    with tab2:
        st.header("Historical Mean Reversion Trends")
        
        # Get the unique pairs from the results
        unique_pairs = sorted(pairs)  # Use all selected pairs

        
        if unique_pairs:
            # Create a dictionary to track which exchanges have data for each pair
            pair_exchanges = {}
            for r in results:
                if r['pair'] not in pair_exchanges:
                    pair_exchanges[r['pair']] = []
                if r['exchange'] not in pair_exchanges[r['pair']]:
                    pair_exchanges[r['pair']].append(r['exchange'])
            
            # Use or update selected_historical_pair in session state
            if st.session_state.selected_historical_pair not in unique_pairs:
                st.session_state.selected_historical_pair = unique_pairs[0]
                
            # Select a pair for detailed analysis
            selected_pair = st.selectbox(
                "Select Pair for Historical Analysis", 
                unique_pairs,
                index=unique_pairs.index(st.session_state.selected_historical_pair),
                key="pair_selector"
            )
            
            # Update session state
            st.session_state.selected_historical_pair = selected_pair
            
            # Get exchanges with data for the selected pair
            if selected_pair in pair_exchanges:
                exchanges_with_data = sorted(pair_exchanges[selected_pair])
                
                if exchanges_with_data:
                    # Initialize selected exchange if needed
                    if st.session_state.selected_historical_exchange not in exchanges_with_data:
                        st.session_state.selected_historical_exchange = exchanges_with_data[0]
                    
                    # Select exchange with persistent state
                    selected_exchange = st.radio(
                        "Select Exchange", 
                        exchanges_with_data,
                        index=exchanges_with_data.index(st.session_state.selected_historical_exchange),
                        horizontal=True,
                        key="exchange_selector"
                    )
                    
                    # Update session state
                    st.session_state.selected_historical_exchange = selected_exchange
                    
                    # Add debug information
                    with st.expander("Debug Information"):
                        st.write(f"Selected pair: {selected_pair}")
                        st.write(f"Selected exchange: {selected_exchange}")
                        
                        # Check if we have time series data
                        if selected_pair in analyzer.time_series_data:
                            st.write(f"Exchanges with time series data for {selected_pair}: {list(analyzer.time_series_data[selected_pair].keys())}")
                            
                            if selected_exchange in analyzer.time_series_data[selected_pair]:
                                data_points = len(analyzer.time_series_data[selected_pair][selected_exchange])
                                st.write(f"Number of time series data points for {selected_exchange}: {data_points}")
                                
                                if data_points > 0:
                                    ts_data = analyzer.time_series_data[selected_pair][selected_exchange]
                                    ts_df = pd.DataFrame(ts_data)
                                    
                                    # Show the first and last timestamp
                                    if not ts_df.empty:
                                        st.write(f"First timestamp: {ts_df['timestamp'].min()}")
                                        st.write(f"Last timestamp: {ts_df['timestamp'].max()}")
                                        
                                        # Show data completeness for each metric
                                        for metric in ['direction_changes_30min', 'absolute_range_pct', 'dc_range_ratio', 'hurst_exponent']:
                                            missing = ts_df[metric].isna().sum()
                                            if missing > 0:
                                                st.write(f"Missing values in {metric}: {missing} out of {len(ts_df)}")
                            else:
                                st.write(f"No time series data for {selected_exchange}")
                        else:
                            st.write(f"No time series data for {selected_pair}")
                    
                    # Safely check if we have time series data for this pair/exchange
                    has_data = (
                        selected_pair in analyzer.time_series_data and 
                        selected_exchange in analyzer.time_series_data[selected_pair] and 
                        len(analyzer.time_series_data[selected_pair][selected_exchange]) > 0
                    )
                    
                    if has_data:
                        # Get time series data and convert to DataFrame
                        ts_data = analyzer.time_series_data[selected_pair][selected_exchange]
                        ts_df = pd.DataFrame(ts_data)
                        
                        # Get the time range for all charts
                        if not ts_df.empty:
                            min_time = ts_df['timestamp'].min()
                            max_time = ts_df['timestamp'].max()
                            
                            # Sort by timestamp
                            ts_df = ts_df.sort_values('timestamp')
                            
                            # Display charts
                            st.subheader(f"30-Minute Interval Metrics for {selected_pair} on {selected_exchange.upper()}")
                            
                            # Define metrics to plot with their display names
                            metrics_to_plot = [
                                ('dc_range_ratio', 'Dir Changes/Range Ratio'),
                                ('hurst_exponent', 'Hurst Exponent'),
                                ('direction_changes_30min', 'Direction Changes (30min)'),
                                ('absolute_range_pct', 'Range %')
                            ]
                            
                            # Create and display each chart
                            for metric, title in metrics_to_plot:
                                if metric in ts_df.columns:
                                    # Filter out NaN values
                                    metric_df = ts_df[['timestamp', metric]].dropna()
                                    
                                    if len(metric_df) > 0:
                                        # Create basic line chart
                                        fig = px.line(
                                            metric_df, 
                                            x='timestamp', 
                                            y=metric,
                                            title=f"{title} for {selected_pair} ({selected_exchange.upper()})"
                                        )
                                        
                                        # Ensure all charts share the same x-axis range
                                        fig.update_xaxes(range=[min_time, max_time])
                                        
                                        # Calculate mean value
                                        mean_value = metric_df[metric].mean()
                                        
                                        # Add mean line to all charts
                                        fig.add_shape(
                                            type="line",
                                            x0=min_time,
                                            y0=mean_value,
                                            x1=max_time,
                                            y1=mean_value,
                                            line=dict(
                                                color="green",
                                                width=1,
                                                dash="dot",
                                            )
                                        )
                                        
                                        # Add annotation for mean line
                                        fig.add_annotation(
                                            x=min_time,
                                            y=mean_value,
                                            text=f"Mean: {mean_value:.2f}",
                                            showarrow=False,
                                            yshift=10,
                                            xshift=50
                                        )
                                        
                                        # Add reference line for Hurst exponent
                                        if metric == 'hurst_exponent':
                                            fig.add_shape(
                                                type="line",
                                                x0=min_time,
                                                y0=0.5,
                                                x1=max_time,
                                                y1=0.5,
                                                line=dict(
                                                    color="red",
                                                    width=1,
                                                    dash="dash",
                                                )
                                            )
                                            
                                            # Add annotation for Hurst reference
                                            fig.add_annotation(
                                                x=max_time,
                                                y=0.5,
                                                text="Random Walk (H=0.5)",
                                                showarrow=False,
                                                yshift=10
                                            )
                                            
                                            # Add trend line for Hurst
                                            if len(metric_df) > 1:
                                                try:
                                                    # Calculate trend line
                                                    x_numeric = np.arange(len(metric_df))
                                                    y = metric_df[metric].values
                                                    slope, intercept, _, _, _ = stats.linregress(x_numeric, y)
                                                    trend_y = intercept + slope * x_numeric
                                                    
                                                    # Add trend line to figure
                                                    fig.add_trace(go.Scatter(
                                                        x=metric_df['timestamp'],
                                                        y=trend_y,
                                                        mode='lines+markers',
                                                        line=dict(color='blue', width=1, dash='solid'),
                                                        name='Trend'
                                                    ))
                                                except Exception as e:
                                                    st.warning(f"Could not calculate trend line: {e}")
                                        
                                        # Ensure timestamps display correctly
                                        fig.update_xaxes(
                                            title_text="Singapore Time",
                                            tickformat="%H:%M\n%b %d"  # Format: Hours:Minutes and date below
                                        )
                                        
                                        # Update layout
                                        fig.update_layout(
                                            xaxis_title="Time (Singapore)",
                                            yaxis_title=title,
                                            height=400
                                        )
                                        
                                        # Display the chart
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"No valid data for {title} metric")
                                else:
                                    st.warning(f"Metric {title} not found in data")
                    else:
                        st.warning(f"No historical data available for {selected_pair} on {selected_exchange}.")
                        st.info("Try increasing the 'Hours to Look Back' parameter to collect more data.")
                else:
                    st.warning(f"No exchanges with data for {selected_pair}")
            else:
                st.warning(f"No exchange data available for {selected_pair}")
        else:
            st.warning("No pairs available for historical analysis")
    
    # Tab 3: PNL Correlation Analysis
    with tab3:
        st.header("PNL Correlation Analysis")
        st.subheader("Understanding How Mean Reversion Metrics Impact Platform PNL")
        
        # Get the unique pairs from the results
        unique_pairs = sorted(pairs)
        
        if unique_pairs:
            # Create a dictionary to track which exchanges have data for each pair
            pair_exchanges = {}
            for r in results:
                if r['pair'] not in pair_exchanges:
                    pair_exchanges[r['pair']] = []
                if r['exchange'] not in pair_exchanges[r['pair']]:
                    pair_exchanges[r['pair']].append(r['exchange'])
            
            # Use or update selected pair in session state for PNL analysis
            if 'selected_pnl_pair' not in st.session_state:
                st.session_state.selected_pnl_pair = unique_pairs[0]
            elif st.session_state.selected_pnl_pair not in unique_pairs:
                st.session_state.selected_pnl_pair = unique_pairs[0]
            
            # Select a pair for detailed PNL analysis
            selected_pair = st.selectbox(
                "Select Pair for PNL Correlation Analysis", 
                unique_pairs,
                index=unique_pairs.index(st.session_state.selected_pnl_pair),
                key="pnl_pair_selector"
            )
            
            # Update session state
            st.session_state.selected_pnl_pair = selected_pair
            
            # Get exchanges with data for the selected pair
            if selected_pair in pair_exchanges:
                exchanges_with_data = sorted(pair_exchanges[selected_pair])
                
                if exchanges_with_data:
                    # Use or update selected exchange in session state for PNL analysis
                    if 'selected_pnl_exchange' not in st.session_state:
                        st.session_state.selected_pnl_exchange = exchanges_with_data[0]
                    elif st.session_state.selected_pnl_exchange not in exchanges_with_data:
                        st.session_state.selected_pnl_exchange = exchanges_with_data[0]
                    
                    # Select exchange with persistent state
                    selected_exchange = st.radio(
                        "Select Exchange", 
                        exchanges_with_data,
                        index=exchanges_with_data.index(st.session_state.selected_pnl_exchange),
                        horizontal=True,
                        key="pnl_exchange_selector"
                    )
                    
                    # Update session state
                    st.session_state.selected_pnl_exchange = selected_exchange
                    
                    # Fetch PNL data for this pair
                    with st.spinner(f"Fetching PNL data for {selected_pair}..."):
                        pnl_data = fetch_platform_pnl_for_pair(selected_pair, hours=hours)
                    
                    # Check if we have time series data for mean reversion metrics
                    has_metrics_data = (
                        selected_pair in analyzer.time_series_data and 
                        selected_exchange in analyzer.time_series_data[selected_pair] and 
                        len(analyzer.time_series_data[selected_pair][selected_exchange]) > 0
                    )
                    
                    if has_metrics_data and pnl_data is not None and not pnl_data.empty:
                        # Get time series data for mean reversion metrics
                        metrics_data = analyzer.time_series_data[selected_pair][selected_exchange]
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Show raw data sample for debugging PNL values
                        with st.expander("Debug PNL Data"):
                            st.write("### Raw PNL Data Sample (first 10 rows)")
                            st.dataframe(pnl_data.head(10))
                            
                            # Show summary statistics
                            st.write("### PNL Data Summary")
                            st.write(f"Total PNL entries: {len(pnl_data)}")
                            st.write(f"Min PNL value: ${pnl_data['platform_total_pnl'].min():.2f}")
                            st.write(f"Max PNL value: ${pnl_data['platform_total_pnl'].max():.2f}")
                            st.write(f"Total PNL (sum): ${pnl_data['platform_total_pnl'].sum():.2f}")
                            st.write(f"Average PNL per interval: ${pnl_data['platform_total_pnl'].mean():.2f}")
                        
                        # Ensure timestamps are datetime objects
                        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                        pnl_data['timestamp'] = pd.to_datetime(pnl_data['timestamp'])
                        
                        # Remove timezone info if present (make both naive)
                        metrics_df['timestamp'] = metrics_df['timestamp'].dt.tz_localize(None)
                        pnl_data['timestamp'] = pnl_data['timestamp'].dt.tz_localize(None)
                        
                        # Sort both dataframes by timestamp
                        metrics_df = metrics_df.sort_values('timestamp')
                        pnl_data = pnl_data.sort_values('timestamp')
                        
                        # Display raw data indicator
                        st.write(f"Raw platform_total_pnl range: ${pnl_data['platform_total_pnl'].min():.2f} to ${pnl_data['platform_total_pnl'].max():.2f}")
                        
                        # Calculate proper cumulative PNL
                        running_sum = 0
                        cumulative_values = []
                        
                        for pnl_value in pnl_data['platform_total_pnl']:
                            running_sum += pnl_value
                            cumulative_values.append(running_sum)
                        
                        pnl_data['cumulative_pnl'] = cumulative_values
                        
                        # Merge PNL data with metrics data based on closest timestamp
                        # First, create a column to join on by rounding timestamp to nearest 30 min
                        metrics_df['rounded_timestamp'] = metrics_df['timestamp'].dt.floor('30min')
                        pnl_data['rounded_timestamp'] = pnl_data['timestamp'].dt.floor('30min')
                        
                        # Merge the dataframes
                        merged_df = pd.merge_asof(
                            pnl_data,
                            metrics_df,
                            on='timestamp',
                            direction='nearest',
                            tolerance=pd.Timedelta('15 minutes')
                        )
                        
                        # Filter out rows with missing data
                        merged_df = merged_df.dropna(subset=['direction_changes_30min', 'absolute_range_pct', 'dc_range_ratio', 'hurst_exponent'])
                        
                        if merged_df.empty:
                            st.warning("Unable to align mean reversion metrics with PNL data. Time ranges may not overlap.")
                        else:
                            # Calculate correlations
                            correlation_metrics = [
                                'direction_changes_30min', 
                                'absolute_range_pct', 
                                'dc_range_ratio', 
                                'hurst_exponent'
                            ]
                            
                            correlation_results = {}
                            for metric in correlation_metrics:
                                correlation = merged_df['cumulative_pnl'].corr(merged_df[metric])
                                correlation_results[metric] = correlation
                            
                            # Show correlation summary
                            st.subheader("Correlation Between Mean Reversion Metrics and PNL")
                            
                            # Create a correlation table
                            corr_df = pd.DataFrame({
                                'Metric': [analyzer.metric_display_names[m] for m in correlation_metrics],
                                'Correlation with PNL': [correlation_results[m] for m in correlation_metrics]
                            })
                            
                            # Style the correlation table
                            def style_correlation(val):
                                if abs(val) > 0.7:
                                    return 'background-color: #60b33c; color: white; font-weight: bold'  # Strong correlation
                                elif abs(val) > 0.4:
                                    return 'background-color: #a0d995; color: black'  # Moderate correlation
                                elif abs(val) > 0.2:
                                    return 'background-color: #f1f1aa; color: black'  # Weak correlation
                                else:
                                    return 'background-color: #ffc299; color: black'  # No significant correlation
                            
                            # Display styled correlation table
                            st.dataframe(
                                corr_df.style.format({
                                    'Correlation with PNL': '{:.3f}'
                                }).applymap(style_correlation, subset=['Correlation with PNL']),
                                height=200,
                                use_container_width=True
                            )
                            
                            # Interpretation of correlation
                            st.markdown("### Interpretation")
                            st.markdown("""
                            - **Positive correlation (> 0)**: As the metric increases, PNL tends to increase
                            - **Negative correlation (< 0)**: As the metric increases, PNL tends to decrease
                            - **Strong correlation (> 0.7 or < -0.7)**: Very strong relationship
                            - **Moderate correlation (0.4 to 0.7 or -0.4 to -0.7)**: Noticeable relationship
                            - **Weak correlation (0.2 to 0.4 or -0.2 to -0.4)**: Slight relationship
                            - **No correlation (-0.2 to 0.2)**: No significant relationship
                            """)
                            
                            # Create visualization of PNL vs. metrics over time
                            st.subheader("PNL vs. Mean Reversion Metrics Over Time")
                            
                            # Define metrics to plot with their display names
                            metrics_to_plot = [
                                ('dc_range_ratio', 'Dir Changes/Range Ratio'),
                                ('hurst_exponent', 'Hurst Exponent'),
                                ('direction_changes_30min', 'Direction Changes (30min)'),
                                ('absolute_range_pct', 'Range %')
                            ]
                            
                            # Plot each metric against cumulative PNL
                            for metric, title in metrics_to_plot:
                                # Create figure with secondary y-axis
                                fig = go.Figure()
                                
                                # Add PNL line
                                fig.add_trace(go.Scatter(
                                    x=merged_df['timestamp'],
                                    y=merged_df['cumulative_pnl'],
                                    name='Cumulative PNL (USD)',
                                    line=dict(color='green', width=3)
                                ))
                                
                                # Add metric line on secondary axis
                                fig.add_trace(go.Scatter(
                                    x=merged_df['timestamp'],
                                    y=merged_df[metric],
                                    name=title,
                                    line=dict(color='blue', width=2),
                                    yaxis='y2'
                                ))
                                
                                # Update layout with two y-axes
                                fig.update_layout(
                                    title=f"Cumulative PNL vs. {title} Over Time",
                                    xaxis=dict(title="Time (Singapore)"),
                                    yaxis=dict(
                                        title="Cumulative PNL (USD)",
                                        titlefont=dict(color="green"),
                                        tickfont=dict(color="green"),
                                        side="left"
                                    ),
                                    yaxis2=dict(
                                        title=title,
                                        titlefont=dict(color="blue"),
                                        tickfont=dict(color="blue"),
                                        anchor="x",
                                        overlaying="y",
                                        side="right"
                                    ),
                                    legend=dict(x=0.01, y=0.99),
                                    height=500
                                )
                                
                                # Ensure timestamps display correctly
                                fig.update_xaxes(
                                    tickformat="%H:%M\n%b %d"  # Format: Hours:Minutes and date below
                                )
                                
                                # Display the chart
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add correlation annotation
                                corr_value = correlation_results[metric]
                                corr_text = "Strong " if abs(corr_value) > 0.7 else "Moderate " if abs(corr_value) > 0.4 else "Weak " if abs(corr_value) > 0.2 else "No "
                                corr_text += "positive correlation" if corr_value > 0 else "negative correlation"
                                
                                st.markdown(f"**Correlation Analysis:** {corr_text} between {title} and PNL (r = {corr_value:.3f}).")
                                
                                # Add specific insights based on the metric
                                if metric == 'hurst_exponent':
                                    if corr_value < -0.2:
                                        st.markdown("**Insight:** Lower Hurst exponents (more mean-reverting behavior) appear to be associated with higher PNL. "
                                                  "This suggests that mean reversion trading strategies may be more profitable in this market.")
                                    elif corr_value > 0.2:
                                        st.markdown("**Insight:** Higher Hurst exponents (more trending behavior) appear to be associated with higher PNL. "
                                                  "This suggests that trend-following strategies may be more profitable in this market.")
                                    else:
                                        st.markdown("**Insight:** No clear relationship between Hurst exponent and PNL. "
                                                  "This suggests that other factors may be more important for profitability.")
                                
                                elif metric == 'dc_range_ratio':
                                    if corr_value > 0.2:
                                        st.markdown("**Insight:** Higher direction changes to range ratio appears to be associated with higher PNL. "
                                                  "This suggests that markets with more direction changes relative to their range are more profitable.")
                                    elif corr_value < -0.2:
                                        st.markdown("**Insight:** Lower direction changes to range ratio appears to be associated with higher PNL. "
                                                  "This suggests that markets with fewer direction changes relative to their range are more profitable.")
                                    else:
                                        st.markdown("**Insight:** No clear relationship between direction changes to range ratio and PNL. "
                                                  "This suggests that other factors may be more important for profitability.")
                            
                            # Add scatter plots for each metric vs PNL change (not cumulative)
                            st.subheader("Mean Reversion Metrics vs. PNL Changes")
                            
                            # Calculate PNL changes between intervals
                            merged_df['pnl_change'] = merged_df['platform_total_pnl'].diff().fillna(0)
                            
                            # Create 2x2 grid of scatter plots
                            col1, col2 = st.columns(2)
                            
                            for i, (metric, title) in enumerate(metrics_to_plot):
                                fig = px.scatter(
                                    merged_df, 
                                    x=metric, 
                                    y='pnl_change',
                                    color='pnl_change',
                                    color_continuous_scale='RdYlGn',
                                    title=f"{title} vs. PNL Change per Interval",
                                    labels={metric: title, 'pnl_change': 'PNL Change (USD)'},
                                    hover_data=['timestamp']
                                )
                                
                                # Add trendline
                                fig.update_layout(height=400)
                                
                                # Add to appropriate column
                                if i % 2 == 0:
                                    col1.plotly_chart(fig, use_container_width=True)
                                else:
                                    col2.plotly_chart(fig, use_container_width=True)
                            
                            # Multiple regression analysis
                            st.subheader("Multiple Regression Analysis")
                            
                            try:
                                # Prepare data for regression
                                X = merged_df[correlation_metrics]
                                y = merged_df['pnl_change']
                                
                                # Add constant for intercept
                                X = sm.add_constant(X)
                                
                                # Fit regression model
                                model = sm.OLS(y, X).fit()
                                
                                # Display results
                                st.write("### Regression Results")
                                
                                # Create a summary table
                                results_df = pd.DataFrame({
                                    'Variable': model.params.index,
                                    'Coefficient': model.params.values,
                                    'P-Value': model.pvalues.values,
                                    'Significant': model.pvalues < 0.05
                                })
                                
                                # Format the table
                                st.dataframe(
                                    results_df.style.format({
                                        'Coefficient': '{:.4f}',
                                        'P-Value': '{:.4f}'
                                    }),
                                    height=200,
                                    use_container_width=True
                                )
                                
                                # Model summary statistics
                                st.write(f"**R-squared:** {model.rsquared:.4f} (Higher values indicate better fit)")
                                st.write(f"**Adjusted R-squared:** {model.rsquared_adj:.4f}")
                                st.write(f"**F-statistic:** {model.fvalue:.4f} (P-value: {model.f_pvalue:.4f})")
                                
                                # Interpretation
                                if model.f_pvalue < 0.05:
                                    st.success("The model is statistically significant (p < 0.05). The mean reversion metrics together have a meaningful relationship with PNL changes.")
                                    
                                    # Identify significant variables
                                    sig_vars = results_df[results_df['Significant'] == True]
                                    if not sig_vars.empty:
                                        st.write("### Significant Factors:")
                                        for _, row in sig_vars.iterrows():
                                            if row['Variable'] != 'const':
                                                var_name = analyzer.metric_display_names.get(row['Variable'], row['Variable'])
                                                direction = "increases" if row['Coefficient'] > 0 else "decreases"
                                                st.write(f"- As {var_name} increases by 1 unit, PNL {direction} by ${abs(row['Coefficient']):.2f} on average")
                                else:
                                    st.warning("The model is not statistically significant (p >= 0.05). The mean reversion metrics together do not show a clear relationship with PNL changes.")
                            except Exception as e:
                                st.error(f"Error performing regression analysis: {e}")
                                st.write("This typically occurs when there is insufficient data or high multicollinearity between variables.")
                            
                            # Intraday Analysis
                            st.subheader("Intraday PNL Patterns")
                            
                            # Group by hour of day
                            merged_df['hour'] = merged_df['timestamp'].dt.hour
                            hourly_analysis = merged_df.groupby('hour').agg({
                                'pnl_change': ['mean', 'sum', 'std'],
                                'direction_changes_30min': 'mean',
                                'absolute_range_pct': 'mean',
                                'dc_range_ratio': 'mean',
                                'hurst_exponent': 'mean'
                            })
                            
                            hourly_analysis.columns = ['_'.join(col).strip() for col in hourly_analysis.columns.values]
                            hourly_analysis = hourly_analysis.reset_index()
                            
                            # Create hourly PNL chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=hourly_analysis['hour'],
                                y=hourly_analysis['pnl_change_sum'],
                                name='Total PNL Change',
                                marker_color='green'
                            ))
                            
                            fig.update_layout(
                                title="Total PNL Change by Hour of Day (Singapore Time)",
                                xaxis=dict(title="Hour of Day", tickmode='linear', tick0=0, dtick=1),
                                yaxis=dict(title="Total PNL Change (USD)"),
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show hourly metrics table
                            st.write("### Mean Reversion Metrics by Hour of Day")
                            
                            # Format the table
                            hourly_display = hourly_analysis.rename(columns={
                                'hour': 'Hour',
                                'pnl_change_mean': 'Avg PNL Change',
                                'pnl_change_sum': 'Total PNL Change',
                                'pnl_change_std': 'PNL Volatility',
                                'direction_changes_30min_mean': 'Avg Dir Changes',
                                'absolute_range_pct_mean': 'Avg Range %',
                                'dc_range_ratio_mean': 'Avg DC/Range Ratio',
                                'hurst_exponent_mean': 'Avg Hurst Exponent'
                            })
                            
                            st.dataframe(
                                hourly_display.style.format({
                                    'Avg PNL Change': '${:.2f}',
                                    'Total PNL Change': '${:.2f}',
                                    'PNL Volatility': '${:.2f}',
                                    'Avg Dir Changes': '{:.2f}',
                                    'Avg Range %': '{:.2f}%',
                                    'Avg DC/Range Ratio': '{:.3f}',
                                    'Avg Hurst Exponent': '{:.3f}'
                                }),
                                height=500,
                                use_container_width=True
                            )
                            
                            # Thresholds and Decision Boundaries
                            st.subheader("Finding Optimal Thresholds for Trading Decisions")
                            
                            # Choose a metric to analyze thresholds
                            threshold_metric = st.selectbox(
                                "Select Metric for Threshold Analysis",
                                [('hurst_exponent', 'Hurst Exponent'),
                                 ('dc_range_ratio', 'DC/Range Ratio'),
                                 ('direction_changes_30min', 'Direction Changes (30min)'),
                                 ('absolute_range_pct', 'Range %')],
                                format_func=lambda x: x[1]
                            )
                            
                            selected_metric, metric_display = threshold_metric
                            
                            # Create threshold ranges
                            min_val = merged_df[selected_metric].min()
                            max_val = merged_df[selected_metric].max()
                            
                            # Create 10 threshold points
                            thresholds = np.linspace(min_val, max_val, 10)
                            
                            # Calculate PNL performance at different thresholds
                            threshold_results = []
                            
                            for threshold in thresholds:
                                # For metrics where lower is better (Hurst)
                                if selected_metric == 'hurst_exponent':
                                    filtered_df = merged_df[merged_df[selected_metric] <= threshold]
                                    comparison = "<="
                                # For metrics where higher is better
                                else:
                                    filtered_df = merged_df[merged_df[selected_metric] >= threshold]
                                    comparison = ">="
                                
                                if len(filtered_df) > 0:
                                    total_pnl = filtered_df['pnl_change'].sum()
                                    avg_pnl = filtered_df['pnl_change'].mean()
                                    count = len(filtered_df)
                                    
                                    threshold_results.append({
                                        'Threshold': threshold,
                                        'Comparison': comparison,
                                        'Total PNL': total_pnl,
                                        'Average PNL': avg_pnl,
                                        'Count': count,
                                        'Percent of Total': count / len(merged_df) * 100
                                    })
                            
                            # Convert to DataFrame
                            threshold_df = pd.DataFrame(threshold_results)
                            
                            # Plot threshold analysis
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=threshold_df['Threshold'],
                                y=threshold_df['Total PNL'],
                                name='Total PNL',
                                marker_color='green'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=threshold_df['Threshold'],
                                y=threshold_df['Count'],
                                name='Number of Intervals',
                                yaxis='y2',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Update layout with two y-axes
                            fig.update_layout(
                                title=f"PNL Performance vs {metric_display} Thresholds",
                                xaxis=dict(title=f"{metric_display} Threshold"),
                                yaxis=dict(
                                    title="Total PNL (USD)",
                                    titlefont=dict(color="green"),
                                    tickfont=dict(color="green")
                                ),
                                yaxis2=dict(
                                    title="Number of Intervals",
                                    titlefont=dict(color="blue"),
                                    tickfont=dict(color="blue"),
                                    anchor="x",
                                    overlaying="y",
                                    side="right"
                                ),
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display threshold analysis table
                            st.write("### Threshold Analysis Results")
                            
                            # Format the table
                            st.dataframe(
                                threshold_df.style.format({
                                    'Threshold': '{:.3f}',
                                    'Total PNL': '${:.2f}',
                                    'Average PNL': '${:.2f}',
                                    'Percent of Total': '{:.1f}%'
                                }),
                                height=300,
                                use_container_width=True
                            )
                            
                            # Find optimal threshold
                            if not threshold_df.empty:
                                optimal_row = threshold_df.loc[threshold_df['Total PNL'].idxmax()]
                                
                                st.success(f"**Optimal Threshold:** {optimal_row['Threshold']:.3f} " +
                                           f"({optimal_row['Comparison']} {optimal_row['Threshold']:.3f})")
                                
                                st.write(f"**Total PNL at optimal threshold:** ${optimal_row['Total PNL']:.2f}")
                                st.write(f"**Average PNL per interval:** ${optimal_row['Average PNL']:.2f}")
                                st.write(f"**Number of trading intervals:** {optimal_row['Count']} " +
                                       f"({optimal_row['Percent of Total']:.1f}% of all intervals)")
                                
                                # Trading strategy conclusion
                                st.subheader("Trading Strategy Recommendation")
                                
                                strategy = f"Based on this analysis, an optimal strategy would be to trade {selected_pair} " + \
                                          f"when the {metric_display} is {optimal_row['Comparison']} {optimal_row['Threshold']:.3f}. "
                                
                                # Add specific guidelines based on metric
                                if selected_metric == 'hurst_exponent':
                                    if optimal_row['Threshold'] < 0.5:
                                        strategy += "This confirms that mean-reverting market conditions " + \
                                                   "are more profitable for this pair. "
                                    else:
                                        strategy += "Interestingly, this suggests trending or random market conditions " + \
                                                   "are more profitable for this pair, contrary to mean reversion theory. "
                                elif selected_metric == 'dc_range_ratio':
                                    strategy += "This confirms that markets with high direction changes relative to their range " + \
                                               "create more profitable trading conditions. "
                                elif selected_metric == 'direction_changes_30min':
                                    strategy += "This suggests that markets with more price reversals " + \
                                               "create more profitable trading opportunities. "
                                elif selected_metric == 'absolute_range_pct':
                                    if optimal_row['Comparison'] == ">=":
                                        strategy += "This suggests that more volatile markets with wider ranges " + \
                                                   "create more profitable trading opportunities. "
                                    else:
                                        strategy += "This suggests that less volatile markets with narrower ranges " + \
                                                   "create more profitable trading opportunities. "
                                
                                # Add combined insight
                                strategy += "\n\nFor optimal results, consider combining metrics. The multiple regression " + \
                                           "analysis shows which combination of factors has the strongest relationship with PNL."
                                
                                st.write(strategy)
                            else:
                                st.warning("Unable to fetch PNL data or time series data for the selected pair and exchange. Please ensure both datasets are available.")
                else:
                    st.warning(f"No exchanges with data for {selected_pair}")
            else:
                st.warning(f"No exchange data available for {selected_pair}")
        else:
            st.warning("No pairs available for PNL correlation analysis")
else:
    if not st.session_state.data_processed and submit_button:
        st.warning("Please click 'Analyze Mean Reversion' to fetch and process data.")
    elif not submit_button:
        st.info("Select pairs and click 'Analyze Mean Reversion' to begin analysis.")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This dashboard analyzes cryptocurrency price data to detect mean reversion patterns:

- **Direction Changes (30min)**: Number of price direction reversals in last 30 minutes
- **Range %**: (Local high - local low) / local low × 100
- **DC/Range Ratio**: Direction changes divided by range percentage
- **Hurst Exponent**: Values < 0.5 indicate mean reversion behavior

The "Mean Reversion Score" is a composite metric where higher values indicate stronger mean-reverting tendencies.
""")

# Add explanation of mean reversion
st.sidebar.subheader("What is Mean Reversion?")
st.sidebar.markdown("""
Mean reversion trading strategies are based on the theory that prices and returns eventually move back toward the mean or average.

**Key indicators of mean reversion:**
- High number of direction changes relative to price range
- Hurst exponent < 0.5
- Frequent oscillation within a price band

These patterns can suggest potential trading opportunities for strategies that exploit price movements back toward the average.
""")

# Add explanation of the mean reversion score
st.sidebar.subheader("Mean Reversion Score")
st.sidebar.markdown("""
The mean reversion score is calculated as:

**Score = (1 - Hurst) × 50 + log(1 + DC/Range) × 25**

- The Hurst component contributes up to 50 points (lower Hurst = higher score)
- The DC/Range ratio is log-transformed to handle extreme values
- Higher scores indicate stronger mean-reverting tendencies
""")

# Add information about PNL correlation
st.sidebar.subheader("PNL Correlation Analysis")
st.sidebar.markdown("""
The PNL Correlation tab helps you understand:

- How mean reversion metrics impact trading profitability
- Which market conditions generate the most profit
- How to establish optimal thresholds for trading decisions
- When to apply mean reversion trading strategies

Use this analysis to develop data-driven trading strategies based on quantifiable market behaviors.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")