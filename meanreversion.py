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
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Mean Reversion vs PNL Monitor",
    page_icon="📊",
    layout="wide"
)

# --- DB CONFIG ---
def init_db_connection():
    # DB parameters
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
st.title("Crypto Mean Reversion vs Cumulative PNL Monitor")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Create session state variables for caching if not already present
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = []
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'last_selected_pairs' not in st.session_state:
    st.session_state.last_selected_pairs = []
if 'last_hours' not in st.session_state:
    st.session_state.last_hours = 0
if 'pnl_data_cache' not in st.session_state:
    st.session_state.pnl_data_cache = {}

class MeanReversionAnalyzer:
    """Analyzer for mean reversion behaviors in cryptocurrency prices"""
    
    def __init__(self):
        self.data = {}  # Main data storage
        self.time_series_data = {}  # For tracking metrics over time
        self.exchanges = ['rollbit', 'surf']  # Exchanges to analyze
        
        # Metrics for mean reversion analysis
        self.metrics = [
            'direction_changes_30min',  # Number of direction changes in last 30 min
            'absolute_range_pct',       # (local high - local low) / local low
            'dc_range_ratio',           # Direction changes / range percentage ratio
            'hurst_exponent'            # Hurst exponent
        ]
        
        # Display names for metrics
        self.metric_display_names = {
            'direction_changes_30min': 'Direction Changes (30min)',
            'absolute_range_pct': 'Range %',
            'dc_range_ratio': 'Dir Changes/Range Ratio',
            'hurst_exponent': 'Hurst Exponent'
        }
    
    def _get_partition_tables(self, conn, start_date, end_date):
        """Get partition tables for date range, ensuring we cover all needed dates"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str) and end_date:
            end_date = pd.to_datetime(end_date)
        elif end_date is None:
            end_date = datetime.now()
            
        # Ensure no timezone info for the database query
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)
            
        # Generate list of dates needed (year-month-day format)
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        dates = []
        
        # Ensure we include the entire days needed for the analysis
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        # If we're doing 24-hour analysis, make sure we include today AND yesterday
        # to handle cases where the current time is early in the day
        if (end_date - start_date).total_seconds() <= 86400:  # 24 hours in seconds
            yesterday = (end_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_str = yesterday.strftime("%Y%m%d")
            if yesterday_str not in dates:
                dates.append(yesterday_str)
        
        # Generate table names from dates
        table_names = [f"oracle_price_log_partition_{date}" for date in dates]
        
        # Check which tables actually exist in the database
        cursor = conn.cursor()
        existing_tables = []
        
        for table in table_names:
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
        """Build query for partition tables, ensuring we handle cross-midnight data correctly"""
        if not tables:
            return ""
            
        union_parts = []
        
        for table in tables:
            # For Surf data
            if exchange == 'surf':
                query = f"""
                SELECT 
                    pair_name,
                    created_at AT TIME ZONE 'UTC' AS timestamp,
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
                    created_at AT TIME ZONE 'UTC' AS timestamp,
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
        
        # Combine all queries with UNION and sort by timestamp
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
        return complete_query

    def calculate_hurst_exponent(self, prices, min_k=5, max_k=None):
        """Calculate Hurst exponent"""
        try:
            prices = np.array(prices)
            if len(prices) < 50:
                return 0.5
                
            returns = np.log(prices[1:] / prices[:-1])
            
            if max_k is None:
                max_k = min(int(len(returns) / 4), 120)
            
            if max_k <= min_k:
                max_k = min_k + 5
                
            k_values = list(range(min_k, max_k))
            rs_values = []
            
            for k in k_values:
                segments = len(returns) // k
                if segments == 0:
                    continue
                
                rs_array = []
                
                for i in range(segments):
                    segment = returns[i*k:(i+1)*k]
                    mean = np.mean(segment)
                    deviation = np.cumsum(segment - mean)
                    r = max(deviation) - min(deviation)
                    s = np.std(segment)
                    
                    if s > 0:
                        rs = r / s
                        rs_array.append(rs)
                
                if rs_array:
                    rs_values.append(np.mean(rs_array))
                    
            if len(rs_values) < 3:
                return 0.5
                
            log_k = np.log10(k_values[:len(rs_values)])
            log_rs = np.log10(rs_values)
            
            slope, _, _, _, _ = stats.linregress(log_k, log_rs)
            
            hurst = slope
            
            if hurst < 0:
                hurst = 0
            elif hurst > 1:
                hurst = 1
                
            return hurst
        except Exception as e:
            return 0.5

    def calculate_direction_changes(self, prices):
        """Calculate direction changes"""
        try:
            price_changes = np.diff(prices)
            signs = np.sign(price_changes)
            direction_changes = np.sum(signs[1:] != signs[:-1])
            return int(direction_changes)
        except Exception as e:
            return 0

    def calculate_range_percentage(self, prices):
        """Calculate range percentage"""
        try:
            local_high = np.max(prices)
            local_low = np.min(prices)
            if local_low > 0:
                range_pct = ((local_high - local_low) / local_low) * 100
                return range_pct
            return 0
        except Exception as e:
            return 0

    def generate_30min_intervals(self, start_time, end_time):
        """Generate 30-minute intervals from start time to end time"""
        intervals = []
        current_time = start_time
        
        while current_time < end_time:
            next_time = current_time + timedelta(minutes=30)
            intervals.append((current_time, next_time))
            current_time = next_time
            
        return intervals

    def process_historical_data(self, df, pair_name, exchange, lookback_hours=24):
        """Process historical data to generate 30-minute interval metrics"""
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Extract the time range
            end_time = df['timestamp'].max()
            start_time = end_time - pd.Timedelta(hours=lookback_hours)
            
            # Filter data to lookback period
            df_lookback = df[df['timestamp'] >= start_time]
            
            if len(df_lookback) < 20:
                # Not enough data points
                st.warning(f"Not enough data points for {pair_name} on {exchange}: {len(df_lookback)} points")
                return
            
            # Generate 30-minute intervals
            intervals = self.generate_30min_intervals(start_time, end_time)
            
            # Initialize data structure if needed
            if pair_name not in self.time_series_data:
                self.time_series_data[pair_name] = {}
            if exchange not in self.time_series_data[pair_name]:
                self.time_series_data[pair_name][exchange] = []
            
            # Process each interval
            for interval_start, interval_end in intervals:
                # Filter data for this interval
                df_interval = df_lookback[(df_lookback['timestamp'] >= interval_start) & 
                                         (df_lookback['timestamp'] < interval_end)]
                
                # Skip if not enough data points
                if len(df_interval) < 5:
                    continue
                
                # Get prices for this interval
                prices = pd.to_numeric(df_interval['price'], errors='coerce').dropna().values
                
                # Skip if not enough valid prices
                if len(prices) < 5:
                    continue
                
                # Calculate metrics
                direction_changes = self.calculate_direction_changes(prices)
                range_pct = self.calculate_range_percentage(prices)
                dc_range_ratio = direction_changes / (range_pct + 1e-10)  # Avoid division by zero
                hurst = self.calculate_hurst_exponent(prices, min_k=3, max_k=min(len(prices)//3, 20))
                
                # Store calculations with both timestamp and Singapore time for reference
                sg_time = interval_end.replace(tzinfo=pytz.utc).astimezone(singapore_timezone)
                
                self.time_series_data[pair_name][exchange].append({
                    'timestamp': interval_end,
                    'timestamp_sg': sg_time,
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'data_points': len(prices),
                    'direction_changes_30min': direction_changes,
                    'absolute_range_pct': range_pct,
                    'dc_range_ratio': dc_range_ratio,
                    'hurst_exponent': hurst
                })
            
        except Exception as e:
            st.error(f"Error processing historical data for {pair_name} on {exchange}: {e}")

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """Fetch data for specified pairs and analyze mean reversion metrics"""
        # Calculate time range in UTC to ensure consistent time handling
        end_time_utc = datetime.now(pytz.utc)
        start_time_utc = end_time_utc - timedelta(hours=hours)
        
        # Convert to strings for database queries (removing timezone info)
        end_time_str = end_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        start_time_str = start_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        
        st.info(f"Retrieving data from {start_time_str} to {end_time_str} (UTC) - {hours} hours")
        
        try:
            # Get partition tables for the time range
            partition_tables = self._get_partition_tables(conn, start_time_utc, end_time_utc)
            
            if not partition_tables:
                st.error("No data tables available for the selected time range.")
                return None
            
            st.write(f"Found {len(partition_tables)} partition tables: {', '.join(partition_tables)}")
            
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # Process each pair
            for i, pair in enumerate(pairs_to_analyze):
                progress_percentage = (i) / len(pairs_to_analyze)
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                # Process for each exchange
                for exchange in self.exchanges:
                    # Build query
                    query = self._build_query_for_partition_tables(
                        partition_tables,
                        pair_name=pair,
                        start_time=start_time_str,
                        end_time=end_time_str,
                        exchange=exchange
                    )
                    
                    if query:
                        try:
                            # Execute query
                            df = pd.read_sql_query(query, conn)
                            
                            if len(df) > 0:
                                # Convert timestamp to datetime if needed
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                
                                # Debug timestamp range
                                min_time = df['timestamp'].min()
                                max_time = df['timestamp'].max()
                                time_span = max_time - min_time
                                
                                # Log the actual data timespan to help with debugging
                                st.write(f"{exchange}_{pair}: {len(df)} data points spanning {time_span}")
                                
                                # Process historical data for time series analysis
                                self.process_historical_data(df, pair, exchange, lookback_hours=hours)
                                
                                # Add to results
                                results.append({
                                    'pair': pair,
                                    'exchange': exchange,
                                    'data_points': len(df),
                                    'time_range': f"{min_time} to {max_time}"
                                })
                            else:
                                st.warning(f"No data found for {exchange.upper()}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            return results
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            return None

# Function to fetch 30-minute PNL data for a specific pair and specific time period
def fetch_platform_pnl_for_pair(conn, pair_name, hours=24):
    """Fetch platform PNL data for a specific pair"""
    # Check if already cached
    cache_key = f"{pair_name}_{hours}"
    if cache_key in st.session_state.pnl_data_cache:
        return st.session_state.pnl_data_cache[cache_key]
    
    # Set up time range in UTC to match the metrics data
    now_utc = datetime.now(pytz.utc)
    start_time_utc = now_utc - timedelta(hours=hours)
    
    # Construct the query for 30-minute intervals
    query = f"""
    WITH time_intervals AS (
      -- Generate 30-minute intervals for the past hours
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 30),
          '{now_utc}'::timestamp,
          INTERVAL '30 minutes'
        ) AS interval_time
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp",
        COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{now_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" IN (0, 1, 2, 3, 4)
      GROUP BY
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30)
    ),
    
    fee_data AS (
      -- Calculate user fee payments
      SELECT
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp",
        COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{now_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
      GROUP BY
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30)
    ),
    
    funding_pnl AS (
      -- Calculate platform funding fee PNL
      SELECT
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp",
        COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{now_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_way" = 0
      GROUP BY
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30)
    ),
    
    rebate_data AS (
      -- Calculate platform rebate payments
      SELECT
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp",
        COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
      FROM
        "public"."user_cashbooks"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{now_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "remark" = '给邀请人返佣'
      GROUP BY
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30)
    )
    
    -- Final query
    SELECT
      t.interval_time AS "timestamp",
      t.interval_time AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp_sg",
      COALESCE(o."platform_order_pnl", 0) +
      COALESCE(f."user_fee_payments", 0) +
      COALESCE(ff."platform_funding_pnl", 0) +
      COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
    FROM
      time_intervals t
    LEFT JOIN
      order_pnl o ON t.interval_time = o."timestamp"
    LEFT JOIN
      fee_data f ON t.interval_time = f."timestamp"
    LEFT JOIN
      funding_pnl ff ON t.interval_time = ff."timestamp"
    LEFT JOIN
      rebate_data r ON t.interval_time = r."timestamp"
    ORDER BY
      t.interval_time ASC
    """
    
    try:
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            st.warning(f"No PNL data found for {pair_name}")
            return None
        
        # Convert timestamp columns to pandas datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate cumulative PNL
        df['cumulative_pnl'] = df['platform_total_pnl'].cumsum()
        
        # Cache the result
        st.session_state.pnl_data_cache[cache_key] = df
        
        return df
    except Exception as e:
        st.error(f"Error fetching PNL data for {pair_name}: {e}")
        return None

# Function to create visualization of mean reversion metrics vs PNL
def create_metric_vs_pnl_chart(metrics_df, pnl_df, metric_name, metric_display_name, pair_name, exchange, use_sg_time=True):
    """Create visualization comparing a mean reversion metric with cumulative PNL"""
    
    # Check if data is available
    if metrics_df is None or metrics_df.empty or pnl_df is None or pnl_df.empty:
        return None
    
    try:
        # Choose which timestamp column to use (UTC or Singapore time)
        x_column = 'timestamp_sg' if use_sg_time and 'timestamp_sg' in metrics_df.columns and 'timestamp_sg' in pnl_df.columns else 'timestamp'
        
        # Convert to pandas datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(metrics_df[x_column]):
            metrics_df[x_column] = pd.to_datetime(metrics_df[x_column])
        
        if not pd.api.types.is_datetime64_any_dtype(pnl_df[x_column]):
            pnl_df[x_column] = pd.to_datetime(pnl_df[x_column])
        
        # Sort by timestamp
        metrics_df = metrics_df.sort_values(x_column)
        pnl_df = pnl_df.sort_values(x_column)
        
        # Create the figure
        fig = go.Figure()
        
        # Add cumulative PNL line
        fig.add_trace(go.Scatter(
            x=pnl_df[x_column],
            y=pnl_df['cumulative_pnl'],
            name='Cumulative PNL (USD)',
            line=dict(color='green', width=3)
        ))
        
        # Add metric line on second y-axis
        fig.add_trace(go.Scatter(
            x=metrics_df[x_column],
            y=metrics_df[metric_name],
            name=metric_display_name,
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        # Configure layout
        fig.update_layout(
            title=f"{metric_display_name} vs Cumulative PNL: {pair_name}",
            xaxis_title="Time (Singapore)" if use_sg_time else "Time (UTC)",
            yaxis=dict(
                title="Cumulative PNL (USD)",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                side="left"
            ),
            yaxis2=dict(
                title=metric_display_name,
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(x=0.01, y=0.99),
            height=500
        )
        
        # Add reference line for Hurst Exponent
        if metric_name == 'hurst_exponent':
            fig.add_shape(
                type="line",
                x0=metrics_df[x_column].min(),
                x1=metrics_df[x_column].max(),
                y0=0.5,
                y1=0.5,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                ),
                yref='y2'
            )
            
            # Add annotation for Hurst exponent
            fig.add_annotation(
                x=metrics_df[x_column].max(),
                y=0.5,
                text="Random Walk (H=0.5)",
                showarrow=False,
                yshift=10,
                yref='y2'
            )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# Function to calculate correlation between a metric and PNL
def calculate_metric_pnl_correlation(metrics_df, pnl_df, metric_name):
    """Calculate correlation between a mean reversion metric and PNL"""
    
    # Check if data is available
    if metrics_df is None or metrics_df.empty or pnl_df is None or pnl_df.empty:
        return None
    
    try:
        # Create common timeline with 15-minute intervals
        start_time = min(metrics_df['timestamp'].min(), pnl_df['timestamp'].min())
        end_time = max(metrics_df['timestamp'].max(), pnl_df['timestamp'].max())
        
        # Create resampled dataframes
        common_index = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        # Create Series from the metrics data
        metric_series = pd.Series(index=metrics_df['timestamp'], data=metrics_df[metric_name].values)
        metric_resampled = metric_series.reindex(common_index, method='nearest')
        
        # Create Series from the PNL data
        pnl_series = pd.Series(index=pnl_df['timestamp'], data=pnl_df['cumulative_pnl'].values)
        pnl_resampled = pnl_series.reindex(common_index, method='nearest')
        
        # Remove NaN values
        valid_data = pd.DataFrame({
            'metric': metric_resampled,
            'pnl': pnl_resampled
        }).dropna()
        
        # Calculate correlation if we have enough data points
        if len(valid_data) > 5:
            correlation = valid_data['metric'].corr(valid_data['pnl'])
            return correlation
        else:
            return None
    
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
        return None

# Setup sidebar with options
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
        st.session_state.selected_pairs = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]  # Default selection
    
    # Create buttons outside the form for quick selections
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
        # Data retrieval window - fixed to 24 hours for proper cumulative PNL analysis
        hours = 24
        st.write("Analysis Window: 24 Hours (fixed for proper cumulative PNL analysis)")
        
        # Create multiselect for pairs
        selected_pairs = st.multiselect(
            "Select Pairs to Analyze",
            options=all_pairs,
            default=st.session_state.selected_pairs,
            help="Select cryptocurrency pairs to analyze"
        )
        
        # Update session state
        st.session_state.selected_pairs = selected_pairs
        
        # Set the pairs variable for the analyzer
        pairs = selected_pairs
        
        # Exchange selection
        exchange_filter = st.radio(
            "Select Exchange",
            ["surf", "rollbit"],
            index=0,
            help="Select exchange for analysis"
        )
        
        # Time display option
        use_sg_time = st.checkbox("Display Singapore Time", value=True, 
                                  help="Display times in Singapore timezone (SGT) instead of UTC")
        
        # Show a warning if no pairs are selected
        if not pairs:
            st.warning("Please select at least one pair to analyze.")
        
        # Submit button
        submit_button = st.form_submit_button("Analyze Mean Reversion vs PNL")

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
    
    # Run analysis
    with st.spinner("Fetching and analyzing data..."):
        results = analyzer.fetch_and_analyze(
            conn=conn,
            pairs_to_analyze=pairs,
            hours=hours
        )
        
        # Store analysis results and state
        if results:
            st.session_state.analyzed_data = results
            st.session_state.analyzer = analyzer
            st.session_state.data_processed = True
            st.session_state.last_selected_pairs = pairs.copy()
            st.session_state.last_hours = hours
            
            # Clear PNL data cache when new analysis is run
            st.session_state.pnl_data_cache = {}
        else:
            st.error("No results returned from analysis.")
            st.session_state.data_processed = False

# Only proceed if we have processed data
if st.session_state.data_processed and 'analyzer' in st.session_state:
    # Get the saved analyzer
    analyzer = st.session_state.analyzer
    
    # Create metrics visualization with cumulative PNL
    st.header("Mean Reversion Metrics vs Cumulative PNL (24 Hours)")
    
    # Get pairs that were analyzed
    analyzed_pairs = []
    for pair in pairs:
        if pair in analyzer.time_series_data and exchange_filter in analyzer.time_series_data[pair]:
            analyzed_pairs.append(pair)
    
    if analyzed_pairs:
        # Select a pair for detailed analysis
        selected_pair = st.selectbox(
            "Select Pair for Analysis", 
            analyzed_pairs,
            index=0
        )
        
        # Get mean reversion metrics for the selected pair
        if selected_pair in analyzer.time_series_data and exchange_filter in analyzer.time_series_data[selected_pair]:
            metrics_data = analyzer.time_series_data[selected_pair][exchange_filter]
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            if not metrics_df.empty:
                # Get PNL data for the selected pair
                pnl_df = fetch_platform_pnl_for_pair(conn, selected_pair, hours=hours)
                
                if pnl_df is not None and not pnl_df.empty:
                    # Create tabs for different metrics
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Hurst Exponent vs PNL", 
                        "Direction Changes/Range vs PNL", 
                        "Direction Changes vs PNL", 
                        "Range % vs PNL"
                    ])
                    
                    # Show data ranges to help with debugging
                    st.write(f"Metrics data: {len(metrics_df)} points from {metrics_df['timestamp'].min()} to {metrics_df['timestamp'].max()}")
                    st.write(f"PNL data: {len(pnl_df)} points from {pnl_df['timestamp'].min()} to {pnl_df['timestamp'].max()}")
                    
                    with tab1:
                        # Create chart for Hurst Exponent
                        fig = create_metric_vs_pnl_chart(
                            metrics_df,
                            pnl_df,
                            'hurst_exponent',
                            'Hurst Exponent',
                            selected_pair,
                            exchange_filter,
                            use_sg_time=use_sg_time
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate correlation
                            correlation = calculate_metric_pnl_correlation(metrics_df, pnl_df, 'hurst_exponent')
                            
                            if correlation is not None:
                                st.write(f"**Correlation:** {correlation:.3f}")
                                
                                if correlation < -0.3:
                                    st.success("Negative correlation suggests mean reversion (Hurst < 0.5) is associated with higher PNL")
                                elif correlation > 0.3:
                                    st.info("Positive correlation suggests trending (Hurst > 0.5) is associated with higher PNL")
                                else:
                                    st.warning("No strong correlation between Hurst exponent and PNL")
                        else:
                            st.warning("Could not create chart for Hurst Exponent")
                    
                    with tab2:
                        # Create chart for Direction Changes/Range Ratio
                        fig = create_metric_vs_pnl_chart(
                            metrics_df,
                            pnl_df,
                            'dc_range_ratio',
                            'Direction Changes/Range Ratio',
                            selected_pair,
                            exchange_filter,
                            use_sg_time=use_sg_time
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate correlation
                            correlation = calculate_metric_pnl_correlation(metrics_df, pnl_df, 'dc_range_ratio')
                            
                            if correlation is not None:
                                st.write(f"**Correlation:** {correlation:.3f}")
                                
                                if correlation > 0.3:
                                    st.success("Positive correlation suggests higher direction changes relative to range are associated with higher PNL")
                                elif correlation < -0.3:
                                    st.info("Negative correlation suggests lower direction changes relative to range are associated with higher PNL")
                                else:
                                    st.warning("No strong correlation between DC/Range ratio and PNL")
                        else:
                            st.warning("Could not create chart for Direction Changes/Range Ratio")
                            
                    with tab3:
                        # Create chart for Direction Changes
                        fig = create_metric_vs_pnl_chart(
                            metrics_df,
                            pnl_df,
                            'direction_changes_30min',
                            'Direction Changes (30min)',
                            selected_pair,
                            exchange_filter,
                            use_sg_time=use_sg_time
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate correlation
                            correlation = calculate_metric_pnl_correlation(metrics_df, pnl_df, 'direction_changes_30min')
                            
                            if correlation is not None:
                                st.write(f"**Correlation:** {correlation:.3f}")
                                
                                if correlation > 0.3:
                                    st.success("Positive correlation suggests higher direction changes are associated with higher PNL")
                                elif correlation < -0.3:
                                    st.info("Negative correlation suggests lower direction changes are associated with higher PNL")
                                else:
                                    st.warning("No strong correlation between direction changes and PNL")
                        else:
                            st.warning("Could not create chart for Direction Changes")
                            
                    with tab4:
                        # Create chart for Range Percentage
                        fig = create_metric_vs_pnl_chart(
                            metrics_df,
                            pnl_df,
                            'absolute_range_pct',
                            'Range %',
                            selected_pair,
                            exchange_filter,
                            use_sg_time=use_sg_time
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate correlation
                            correlation = calculate_metric_pnl_correlation(metrics_df, pnl_df, 'absolute_range_pct')
                            
                            if correlation is not None:
                                st.write(f"**Correlation:** {correlation:.3f}")
                                
                                if correlation > 0.3:
                                    st.info("Positive correlation suggests higher price range is associated with higher PNL")
                                elif correlation < -0.3:
                                    st.success("Negative correlation suggests lower price range is associated with higher PNL")
                                else:
                                    st.warning("No strong correlation between price range and PNL")
                        else:
                            st.warning("Could not create chart for Range Percentage")
                            
                    # Show multi-pair correlation analysis
                    st.header("Multi-Pair Correlation Analysis")
                    
                    # Analyze all pairs
                    correlation_data = []
                    
                    with st.spinner("Calculating correlations across all pairs..."):
                        for pair in analyzed_pairs:
                            try:
                                # Get metrics data
                                pair_metrics_df = pd.DataFrame(analyzer.time_series_data[pair][exchange_filter])
                                
                                # Get PNL data
                                pair_pnl_df = fetch_platform_pnl_for_pair(conn, pair, hours=hours)
                                
                                if not pair_metrics_df.empty and pair_pnl_df is not None and not pair_pnl_df.empty:
                                    # Calculate correlations
                                    hurst_corr = calculate_metric_pnl_correlation(pair_metrics_df, pair_pnl_df, 'hurst_exponent')
                                    dc_range_corr = calculate_metric_pnl_correlation(pair_metrics_df, pair_pnl_df, 'dc_range_ratio')
                                    dc_corr = calculate_metric_pnl_correlation(pair_metrics_df, pair_pnl_df, 'direction_changes_30min')
                                    range_corr = calculate_metric_pnl_correlation(pair_metrics_df, pair_pnl_df, 'absolute_range_pct')
                                    
                                    # Get final PNL
                                    final_pnl = pair_pnl_df['cumulative_pnl'].iloc[-1] if len(pair_pnl_df) > 0 else 0
                                    
                                    # Calculate average metric values
                                    avg_hurst = pair_metrics_df['hurst_exponent'].mean() if 'hurst_exponent' in pair_metrics_df.columns else None
                                    avg_dc_range = pair_metrics_df['dc_range_ratio'].mean() if 'dc_range_ratio' in pair_metrics_df.columns else None
                                    
                                    # Add to correlation data
                                    correlation_data.append({
                                        'Pair': pair,
                                        'Hurst-PNL Corr': hurst_corr if hurst_corr is not None else float('nan'),
                                        'DC/Range-PNL Corr': dc_range_corr if dc_range_corr is not None else float('nan'),
                                        'DC-PNL Corr': dc_corr if dc_corr is not None else float('nan'),
                                        'Range-PNL Corr': range_corr if range_corr is not None else float('nan'),
                                        'Final PNL': final_pnl,
                                        'Avg Hurst': avg_hurst if avg_hurst is not None else float('nan'),
                                        'Avg DC/Range': avg_dc_range if avg_dc_range is not None else float('nan')
                                    })
                            except Exception as e:
                                st.error(f"Error processing correlation for {pair}: {e}")
                        
                    # Display correlation table
                    if correlation_data:
                        # Convert to DataFrame
                        correlation_df = pd.DataFrame(correlation_data)
                        
                        # Sort by Final PNL (highest first)
                        correlation_df = correlation_df.sort_values('Final PNL', ascending=False)
                        
                        # Function to color correlations
                        def color_correlation(val):
                            if pd.isna(val):
                                return ''
                            elif val > 0.5:
                                return 'background-color: rgba(0, 200, 0, 0.5); color: black'
                            elif val > 0.3:
                                return 'background-color: rgba(150, 200, 150, 0.5); color: black'
                            elif val < -0.5:
                                return 'background-color: rgba(200, 0, 0, 0.5); color: black'
                            elif val < -0.3:
                                return 'background-color: rgba(200, 150, 150, 0.5); color: black'
                            else:
                                return 'background-color: rgba(200, 200, 200, 0.5); color: black'
                        
                        # Function to color PNL
                        def color_pnl(val):
                            if pd.isna(val):
                                return ''
                            elif val > 5000:
                                return 'background-color: rgba(0, 200, 0, 0.8); color: black'
                            elif val > 1000:
                                return 'background-color: rgba(100, 200, 100, 0.5); color: black'
                            elif val > 0:
                                return 'background-color: rgba(200, 255, 200, 0.5); color: black'
                            elif val > -1000:
                                return 'background-color: rgba(255, 200, 200, 0.5); color: black'
                            else:
                                return 'background-color: rgba(255, 100, 100, 0.5); color: black'
                        
                        # Style the DataFrame
                        styled_df = correlation_df.style.format({
                            'Hurst-PNL Corr': '{:.3f}',
                            'DC/Range-PNL Corr': '{:.3f}',
                            'DC-PNL Corr': '{:.3f}',
                            'Range-PNL Corr': '{:.3f}',
                            'Final PNL': '${:.2f}',
                            'Avg Hurst': '{:.3f}',
                            'Avg DC/Range': '{:.3f}'
                        })
                        
                        # Apply color styling
                        styled_df = styled_df.applymap(color_correlation, subset=['Hurst-PNL Corr', 'DC/Range-PNL Corr', 'DC-PNL Corr', 'Range-PNL Corr'])
                        styled_df = styled_df.applymap(color_pnl, subset=['Final PNL'])
                        
                        # Show table
                        st.subheader("Correlation Between Mean Reversion Metrics and PNL by Pair")
                        st.dataframe(styled_df, height=400, use_container_width=True)
                        
                        # Create scatter plot
                        try:
                            # Drop rows with NaN values
                            plot_df = correlation_df.dropna(subset=['Avg Hurst', 'Final PNL'])
                            
                            if not plot_df.empty:
                                fig = px.scatter(
                                    plot_df,
                                    x='Avg Hurst',
                                    y='Final PNL',
                                    color='Avg DC/Range',
                                    size=abs(plot_df['Hurst-PNL Corr']).fillna(0.1) * 10 + 5,
                                    hover_name='Pair',
                                    title='Average Hurst Exponent vs Final PNL (Color = Avg DC/Range)',
                                    labels={
                                        'Avg Hurst': 'Average Hurst Exponent', 
                                        'Final PNL': 'Final PNL (USD)',
                                        'Avg DC/Range': 'Avg Direction Changes / Range'
                                    },
                                    color_continuous_scale='Viridis'
                                )
                                
                                # Add reference line at Hurst = 0.5
                                if len(plot_df) > 0:
                                    fig.add_shape(
                                        type="line",
                                        x0=0.5,
                                        y0=plot_df['Final PNL'].min(),
                                        x1=0.5,
                                        y1=plot_df['Final PNL'].max(),
                                        line=dict(
                                            color="red",
                                            width=1,
                                            dash="dash",
                                        )
                                    )
                                    
                                    # Add annotation
                                    fig.add_annotation(
                                        x=0.5,
                                        y=plot_df['Final PNL'].min(),
                                        text="Random Walk (H=0.5)",
                                        showarrow=False,
                                        yshift=-20
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Not enough data points with valid values for scatter plot")
                        
                        except Exception as e:
                            st.error(f"Error creating scatter plot: {e}")
                        
                        # Calculate summary statistics
                        st.subheader("Summary Insights")
                        
                        try:
                            # Calculate average correlations
                            avg_hurst_corr = correlation_df['Hurst-PNL Corr'].mean()
                            avg_dc_range_corr = correlation_df['DC/Range-PNL Corr'].mean()
                            avg_dc_corr = correlation_df['DC-PNL Corr'].mean()
                            avg_range_corr = correlation_df['Range-PNL Corr'].mean()
                            
                            # Count profitable pairs
                            profitable_pairs = len(correlation_df[correlation_df['Final PNL'] > 0])
                            unprofitable_pairs = len(correlation_df[correlation_df['Final PNL'] <= 0])
                            
                            # Calculate average Hurst for profitable vs. unprofitable pairs
                            avg_hurst_profitable = correlation_df[correlation_df['Final PNL'] > 0]['Avg Hurst'].mean()
                            avg_hurst_unprofitable = correlation_df[correlation_df['Final PNL'] <= 0]['Avg Hurst'].mean() if unprofitable_pairs > 0 else None
                            
                            # Display summary
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Profitable Pairs:** {profitable_pairs} out of {len(correlation_df)}")
                                st.write(f"**Average Correlations:**")
                                st.write(f"- Hurst-PNL: {avg_hurst_corr:.3f}")
                                st.write(f"- DC/Range-PNL: {avg_dc_range_corr:.3f}")
                                st.write(f"- Direction Changes-PNL: {avg_dc_corr:.3f}")
                                st.write(f"- Range-PNL: {avg_range_corr:.3f}")
                            
                            with col2:
                                st.write(f"**Average Hurst for Profitable Pairs:** {avg_hurst_profitable:.3f}")
                                if avg_hurst_unprofitable is not None:
                                    st.write(f"**Average Hurst for Unprofitable Pairs:** {avg_hurst_unprofitable:.3f}")
                                
                                # Insight about mean reversion
                                if avg_hurst_profitable < 0.5:
                                    st.success("**Mean reversion (Hurst < 0.5) appears beneficial for profitability across pairs**")
                                elif avg_hurst_profitable > 0.5:
                                    st.info("**Trending behavior (Hurst > 0.5) appears beneficial for profitability across pairs**")
                                else:
                                    st.warning("**No clear pattern between Hurst exponent and profitability across pairs**")
                        
                        except Exception as e:
                            st.error(f"Error calculating summary statistics: {e}")
                    
                    else:
                        st.warning("No correlation data available for multi-pair analysis")
                else:
                    st.error(f"No PNL data available for {selected_pair}")
            else:
                st.error(f"No mean reversion metrics available for {selected_pair}")
        else:
            st.error(f"No data available for {selected_pair} on {exchange_filter}")
    else:
        st.error("No pairs were successfully analyzed. Please try different pairs or parameters.")
else:
    st.info("Select pairs and click 'Analyze Mean Reversion vs PNL' to begin analysis")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This dashboard analyzes cryptocurrency price data to detect mean reversion patterns and compares them with cumulative PNL:

- **Hurst Exponent**: Values < 0.5 indicate mean reversion behavior
- **Direction Changes/Range Ratio**: Higher values suggest more mean reversion
- **Cumulative PNL**: Platform profit reset to zero at the start of the analysis period

The analysis helps identify which mean reversion characteristics correlate with higher platform profitability.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")