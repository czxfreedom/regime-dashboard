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
        """Get partition tables for date range"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str) and end_date:
            end_date = pd.to_datetime(end_date)
        elif end_date is None:
            end_date = datetime.now()
            
        # Ensure no timezone info for the database query
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)
            
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        table_names = [f"oracle_price_log_partition_{date}" for date in dates]
        
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
        """Build query for partition tables"""
        if not tables:
            return ""
            
        union_parts = []
        
        for table in tables:
            # For Surf data (production)
            if exchange == 'surf':
                query = f"""
                SELECT 
                    pair_name,
                    created_at AT TIME ZONE 'UTC' AS timestamp_utc,
                    created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
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
                    created_at AT TIME ZONE 'UTC' AS timestamp_utc,
                    created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
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
        
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp_utc"
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

    def process_30min_data(self, df, pair_name, exchange):
        """Process 30min data for mean reversion metrics"""
        try:
            # Both timestamp_utc and timestamp_sg should be in the dataframe
            # Sort by UTC timestamp for consistent processing
            df = df.sort_values('timestamp_utc')
            
            end_time = df['timestamp_utc'].max()
            start_time = end_time - pd.Timedelta(minutes=30)
            df_30min = df[df['timestamp_utc'] >= start_time]
            
            if len(df_30min) < 10:
                return None
                
            prices = pd.to_numeric(df_30min['price'], errors='coerce').dropna().values
            
            direction_changes = self.calculate_direction_changes(prices)
            range_pct = self.calculate_range_percentage(prices)
            dc_range_ratio = direction_changes / (range_pct + 1e-10)
            
            all_prices = pd.to_numeric(df['price'], errors='coerce').dropna().values
            hurst_window = min(1000, len(all_prices))
            hurst = self.calculate_hurst_exponent(all_prices[-hurst_window:])
            
            if pair_name not in self.data:
                self.data[pair_name] = {}
            if exchange not in self.data[pair_name]:
                self.data[pair_name][exchange] = {}
                
            self.data[pair_name][exchange]['direction_changes_30min'] = direction_changes
            self.data[pair_name][exchange]['absolute_range_pct'] = range_pct
            self.data[pair_name][exchange]['dc_range_ratio'] = dc_range_ratio
            self.data[pair_name][exchange]['hurst_exponent'] = hurst
            
            # Store both UTC and SG timestamps for the end time
            timestamp_utc = df_30min['timestamp_utc'].max()
            timestamp_sg = df_30min['timestamp_sg'].max()
            
            if pair_name not in self.time_series_data:
                self.time_series_data[pair_name] = {}
            if exchange not in self.time_series_data[pair_name]:
                self.time_series_data[pair_name][exchange] = []
                
            self.time_series_data[pair_name][exchange].append({
                'timestamp_utc': timestamp_utc,
                'timestamp_sg': timestamp_sg,
                'direction_changes_30min': direction_changes,
                'absolute_range_pct': range_pct,
                'dc_range_ratio': dc_range_ratio,
                'hurst_exponent': hurst
            })
            
            return {
                'pair': pair_name,
                'exchange': exchange,
                'timestamp_utc': timestamp_utc,
                'timestamp_sg': timestamp_sg,
                'direction_changes_30min': direction_changes,
                'absolute_range_pct': range_pct,
                'dc_range_ratio': dc_range_ratio,
                'hurst_exponent': hurst
            }
        except Exception as e:
            st.error(f"Error processing 30min data for {pair_name} on {exchange}: {e}")
            return None

    def process_historical_data(self, df, pair_name, exchange):
        """Process historical data to generate 30-minute interval metrics"""
        try:
            # Sort by UTC timestamp
            df = df.sort_values('timestamp_utc')
            
            end_time_utc = df['timestamp_utc'].max()
            start_time_utc = end_time_utc - pd.Timedelta(hours=24)  # Always use 24 hours for analysis
            
            df_lookback = df[df['timestamp_utc'] >= start_time_utc]
            
            if len(df_lookback) < 50:
                st.warning(f"Not enough historical data for {pair_name} on {exchange}: {len(df_lookback)} points")
                return
                
            # Generate 30-minute intervals in UTC
            intervals_utc = pd.date_range(start=start_time_utc, end=end_time_utc, freq='30min')
            
            if pair_name not in self.time_series_data:
                self.time_series_data[pair_name] = {}
            if exchange not in self.time_series_data[pair_name]:
                self.time_series_data[pair_name][exchange] = []
                
            for i in range(len(intervals_utc) - 1):
                interval_start_utc = intervals_utc[i]
                interval_end_utc = intervals_utc[i+1]
                
                # Filter data for this interval using UTC timestamps
                df_interval = df_lookback[(df_lookback['timestamp_utc'] >= interval_start_utc) & 
                                         (df_lookback['timestamp_utc'] < interval_end_utc)]
                
                if len(df_interval) < 5:
                    continue
                    
                prices = pd.to_numeric(df_interval['price'], errors='coerce').dropna().values
                
                if len(prices) < 5:
                    continue
                
                direction_changes = self.calculate_direction_changes(prices)
                range_pct = self.calculate_range_percentage(prices)
                dc_range_ratio = direction_changes / (range_pct + 1e-10)
                hurst = self.calculate_hurst_exponent(prices, min_k=3, max_k=min(len(prices)//3, 20))
                
                # Get corresponding SG timestamp
                if not df_interval.empty:
                    timestamp_sg = df_interval['timestamp_sg'].max()
                else:
                    # Convert UTC to SG if no data available
                    timestamp_sg = interval_end_utc.replace(tzinfo=pytz.utc).astimezone(singapore_timezone)
                
                self.time_series_data[pair_name][exchange].append({
                    'timestamp_utc': interval_end_utc,
                    'timestamp_sg': timestamp_sg,
                    'direction_changes_30min': direction_changes,
                    'absolute_range_pct': range_pct,
                    'dc_range_ratio': dc_range_ratio,
                    'hurst_exponent': hurst
                })
        except Exception as e:
            st.error(f"Error processing historical data for {pair_name} on {exchange}: {e}")

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """Fetch and analyze data for specified pairs"""
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        st.info(f"Retrieving data from {start_time} to {end_time} ({hours} hours)")
        
        try:
            partition_tables = self._get_partition_tables(conn, start_time, end_time)
            
            if not partition_tables:
                st.error("No data tables available for the selected time range.")
                return None
            
            st.write(f"Found {len(partition_tables)} partition tables")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, pair in enumerate(pairs_to_analyze):
                progress_percentage = (i) / len(pairs_to_analyze)
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                for exchange in self.exchanges:
                    query = self._build_query_for_partition_tables(
                        partition_tables,
                        pair_name=pair,
                        start_time=start_time,
                        end_time=end_time,
                        exchange=exchange
                    )
                    
                    if query:
                        try:
                            df = pd.read_sql_query(query, conn)
                            
                            if len(df) > 0:
                                result = self.process_30min_data(df, pair, exchange)
                                if result:
                                    results.append(result)
                                
                                self.process_historical_data(df, pair, exchange)
                            else:
                                st.warning(f"No data found for {exchange.upper()}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
            
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            return results
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            return None

# Function to fetch 30-minute PNL data for a specific pair over a specific time period
def fetch_platform_pnl_for_pair(conn, pair_name, hours=24):
    """Fetch platform PNL data for a specific pair"""
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
        ) AS interval_time_utc
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp_utc",
        COALESCE(SUM(-1 * "taker_pnl" * "collateral_price"), 0) AS "platform_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
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
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp_utc",
        COALESCE(SUM("taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
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
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp_utc",
        COALESCE(SUM(-1 * "funding_fee" * "collateral_price"), 0) AS "platform_funding_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
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
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30) AS "timestamp_utc",
        COALESCE(SUM(-1 * "amount" * "coin_price"), 0) AS "platform_rebate_payments"
      FROM
        "public"."user_cashbooks"
      WHERE
        "created_at" BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "remark" = '给邀请人返佣'
      GROUP BY
        date_trunc('hour', "created_at") + 
        INTERVAL '30 min' * floor(EXTRACT(MINUTE FROM "created_at") / 30)
    )
    
    -- Final query: combine all data sources
    SELECT
      t.interval_time_utc AS "timestamp_utc",
      t.interval_time_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS "timestamp_sg",
      COALESCE(o."platform_order_pnl", 0) +
      COALESCE(f."user_fee_payments", 0) +
      COALESCE(ff."platform_funding_pnl", 0) +
      COALESCE(r."platform_rebate_payments", 0) AS "platform_total_pnl"
    FROM
      time_intervals t
    LEFT JOIN
      order_pnl o ON t.interval_time_utc = o."timestamp_utc"
    LEFT JOIN
      fee_data f ON t.interval_time_utc = f."timestamp_utc"
    LEFT JOIN
      funding_pnl ff ON t.interval_time_utc = ff."timestamp_utc"
    LEFT JOIN
      rebate_data r ON t.interval_time_utc = r."timestamp_utc"
    ORDER BY
      t.interval_time_utc ASC
    """
    
    try:
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            st.warning(f"No PNL data found for {pair_name}")
            return None
        
        # Convert timestamp columns to pandas datetime
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        
        # Sort by timestamp_utc
        df = df.sort_values('timestamp_utc')
        
        # Calculate proper cumulative PNL (restarting from 0 at beginning of period)
        df['cumulative_pnl'] = df['platform_total_pnl'].cumsum()
        
        # Cache the result
        st.session_state.pnl_data_cache[cache_key] = df
        
        return df
    except Exception as e:
        st.error(f"Error fetching PNL data for {pair_name}: {e}")
        return None

# Function to align metrics and PNL data on common timestamps
def align_data_on_common_timestamps(metrics_df, pnl_df):
    """Align metrics and PNL data on common timestamps for plotting and correlation analysis"""
    if metrics_df.empty or pnl_df is None or pnl_df.empty:
        return None
    
    try:
        # Create copies to avoid modifying the original dataframes
        metrics = metrics_df.copy()
        pnl = pnl_df.copy()
        
        # Ensure timestamp columns exist in both dataframes
        if 'timestamp_utc' not in metrics.columns or 'timestamp_utc' not in pnl.columns:
            return None
        
        # Set timestamp_utc as index for both dataframes
        metrics.set_index('timestamp_utc', inplace=True)
        pnl.set_index('timestamp_utc', inplace=True)
        
        # Create a common date range
        all_timestamps = sorted(set(metrics.index) | set(pnl.index))
        common_index = pd.DatetimeIndex(all_timestamps)
        
        # Reindex both dataframes to the common index
        metrics_reindexed = metrics.reindex(common_index)
        pnl_reindexed = pnl.reindex(common_index)
        
        # Forward fill NaN values (from one timestamp to the next)
        metrics_reindexed = metrics_reindexed.fillna(method='ffill')
        pnl_reindexed = pnl_reindexed.fillna(method='ffill')
        
        # Combine into a single dataframe
        combined_df = pd.concat([metrics_reindexed, pnl_reindexed], axis=1)
        
        # Reset index to get timestamp_utc back as a column
        combined_df.reset_index(inplace=True)
        combined_df.rename(columns={'index': 'timestamp_utc'}, inplace=True)
        
        # Add timestamp_sg if it's missing
        if 'timestamp_sg' not in combined_df.columns and 'timestamp_sg_x' in combined_df.columns:
            combined_df['timestamp_sg'] = combined_df['timestamp_sg_x'].combine_first(combined_df['timestamp_sg_y'])
        elif 'timestamp_sg' not in combined_df.columns:
            # Convert UTC to SG timezone
            combined_df['timestamp_sg'] = combined_df['timestamp_utc'].apply(
                lambda x: x.replace(tzinfo=pytz.utc).astimezone(singapore_timezone) if pd.notnull(x) else None
            )
        
        # Return the aligned dataframe
        return combined_df
        
    except Exception as e:
        st.error(f"Error aligning data on common timestamps: {e}")
        return None

# Function to create plots with aligned data
def plot_metric_vs_pnl(combined_df, metric_name, metric_display_name, selected_pair, exchange_filter):
    """Create plot for a specific metric vs PNL using aligned data"""
    try:
        if combined_df is None or combined_df.empty:
            return None
        
        # Check if required columns exist
        required_columns = [metric_name, 'cumulative_pnl', 'timestamp_sg']
        if not all(col in combined_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in combined_df.columns]
            st.error(f"Missing columns in combined dataframe: {missing}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add PNL line
        fig.add_trace(go.Scatter(
            x=combined_df['timestamp_sg'],
            y=combined_df['cumulative_pnl'],
            name='Cumulative PNL (USD)',
            line=dict(color='green', width=3)
        ))
        
        # Add metric line on secondary axis
        fig.add_trace(go.Scatter(
            x=combined_df['timestamp_sg'],
            y=combined_df[metric_name],
            name=metric_display_name,
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        # Update layout with two y-axes
        fig.update_layout(
            title=f"{metric_display_name} vs Cumulative PNL for {selected_pair} ({exchange_filter})",
            xaxis=dict(title="Time (Singapore)"),
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
        
        # Add horizontal reference line for Hurst Exponent
        if metric_name == 'hurst_exponent':
            fig.add_shape(
                type="line",
                x0=combined_df['timestamp_sg'].min(),
                x1=combined_df['timestamp_sg'].max(),
                y0=0.5,
                y1=0.5,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                ),
                yref='y2'
            )
            
            # Add annotation for Hurst reference
            fig.add_annotation(
                x=combined_df['timestamp_sg'].max(),
                y=0.5,
                text="Random Walk (H=0.5)",
                showarrow=False,
                yshift=10,
                yref='y2'
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot for {metric_name}: {e}")
        return None

# Function to calculate correlation with aligned data
def calculate_correlation(combined_df, metric_name):
    """Calculate correlation between a metric and PNL using aligned data"""
    try:
        if combined_df is None or combined_df.empty:
            return None
        
        # Check if required columns exist
        if metric_name not in combined_df.columns or 'cumulative_pnl' not in combined_df.columns:
            return None
            
        # Drop rows with NaN values
        valid_data = combined_df.dropna(subset=[metric_name, 'cumulative_pnl'])
        
        if len(valid_data) < 5:  # Need enough points for meaningful correlation
            return None
            
        # Calculate Pearson correlation
        correlation = valid_data[metric_name].corr(valid_data['cumulative_pnl'])
        
        return correlation
    except Exception as e:
        st.error(f"Error calculating correlation for {metric_name}: {e}")
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
    
    # Get pairs list - fixed the error by checking for list type
    if isinstance(st.session_state.analyzed_data, list) and len(st.session_state.analyzed_data) > 0:
        analyzed_pairs = list(set([r['pair'] for r in st.session_state.analyzed_data]))
    else:
        analyzed_pairs = pairs  # Fallback to selected pairs if no analyzed data
    
    if analyzed_pairs:
        # Select a pair for detailed analysis
        selected_pair = st.selectbox(
            "Select Pair for Analysis", 
            analyzed_pairs,
            index=0
        )
        
        # Get PNL data for the selected pair
        pnl_data = fetch_platform_pnl_for_pair(conn, selected_pair, hours=24)
        
        # Get mean reversion metrics for the selected pair and exchange
        if selected_pair in analyzer.time_series_data and exchange_filter in analyzer.time_series_data[selected_pair]:
            metrics_data = analyzer.time_series_data[selected_pair][exchange_filter]
            metrics_df = pd.DataFrame(metrics_data)
            
            if not metrics_df.empty and metrics_df.shape[0] > 0:
                # Convert timestamps to pandas datetime if needed
                for ts_col in ['timestamp_utc', 'timestamp_sg']:
                    if ts_col in metrics_df.columns:
                        metrics_df[ts_col] = pd.to_datetime(metrics_df[ts_col])
                
                # Sort by timestamp_utc
                if 'timestamp_utc' in metrics_df.columns:
                    metrics_df = metrics_df.sort_values('timestamp_utc')
                
                # Check if we have PNL data
                if pnl_data is not None and not pnl_data.empty:
                    # Align metrics and PNL data
                    combined_df = align_data_on_common_timestamps(metrics_df, pnl_data)
                    
                    if combined_df is not None and not combined_df.empty:
                        # Create tab visualization
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Hurst Exponent vs PNL", 
                            "Direction Changes/Range vs PNL", 
                            "Direction Changes vs PNL", 
                            "Range % vs PNL"
                        ])
                        
                        with tab1:
                            st.subheader(f"Hurst Exponent vs Cumulative PNL: {selected_pair}")
                            
                            # Create plot with aligned data
                            fig = plot_metric_vs_pnl(
                                combined_df, 
                                'hurst_exponent', 
                                'Hurst Exponent',
                                selected_pair,
                                exchange_filter
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate correlation with aligned data
                                corr = calculate_correlation(combined_df, 'hurst_exponent')
                                
                                if corr is not None:
                                    st.write(f"**Correlation:** {corr:.3f}")
                                    
                                    if corr < -0.3:
                                        st.success("Negative correlation suggests mean reversion (Hurst < 0.5) is associated with higher PNL")
                                    elif corr > 0.3:
                                        st.info("Positive correlation suggests trending (Hurst > 0.5) is associated with higher PNL")
                                    else:
                                        st.warning("No strong correlation between Hurst exponent and PNL")
                            else:
                                st.error("Could not create plot for Hurst Exponent")
                        
                        with tab2:
                            st.subheader(f"Direction Changes/Range Ratio vs Cumulative PNL: {selected_pair}")
                            
                            # Create plot with aligned data
                            fig = plot_metric_vs_pnl(
                                combined_df, 
                                'dc_range_ratio', 
                                'DC/Range Ratio',
                                selected_pair,
                                exchange_filter
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate correlation with aligned data
                                corr = calculate_correlation(combined_df, 'dc_range_ratio')
                                
                                if corr is not None:
                                    st.write(f"**Correlation:** {corr:.3f}")
                                    
                                    if corr > 0.3:
                                        st.success("Positive correlation suggests higher direction changes relative to range are associated with higher PNL")
                                    elif corr < -0.3:
                                        st.info("Negative correlation suggests lower direction changes relative to range are associated with higher PNL")
                                    else:
                                        st.warning("No strong correlation between DC/Range ratio and PNL")
                            else:
                                st.error("Could not create plot for DC/Range Ratio")
                        
                        with tab3:
                            st.subheader(f"Direction Changes vs Cumulative PNL: {selected_pair}")
                            
                            # Create plot with aligned data
                            fig = plot_metric_vs_pnl(
                                combined_df, 
                                'direction_changes_30min', 
                                'Direction Changes (30min)',
                                selected_pair,
                                exchange_filter
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate correlation with aligned data
                                corr = calculate_correlation(combined_df, 'direction_changes_30min')
                                
                                if corr is not None:
                                    st.write(f"**Correlation:** {corr:.3f}")
                                    
                                    if corr > 0.3:
                                        st.success("Positive correlation suggests higher direction changes are associated with higher PNL")
                                    elif corr < -0.3:
                                        st.info("Negative correlation suggests lower direction changes are associated with higher PNL")
                                    else:
                                        st.warning("No strong correlation between direction changes and PNL")
                            else:
                                st.error("Could not create plot for Direction Changes")
                        
                        with tab4:
                            st.subheader(f"Range % vs Cumulative PNL: {selected_pair}")
                            
                            # Create plot with aligned data
                            fig = plot_metric_vs_pnl(
                                combined_df, 
                                'absolute_range_pct', 
                                'Range %',
                                selected_pair,
                                exchange_filter
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate correlation with aligned data
                                corr = calculate_correlation(combined_df, 'absolute_range_pct')
                                
                                if corr is not None:
                                    st.write(f"**Correlation:** {corr:.3f}")
                                    
                                    if corr > 0.3:
                                        st.info("Positive correlation suggests higher price range is associated with higher PNL")
                                    elif corr < -0.3:
                                        st.success("Negative correlation suggests lower price range is associated with higher PNL")
                                    else:
                                        st.warning("No strong correlation between price range and PNL")
                            else:
                                st.error("Could not create plot for Range %")
                        
                        # Show multi-pair correlation analysis
                        st.header("Multi-Pair Correlation Analysis")
                        
                        # Get correlation data for all analyzed pairs
                        correlation_data = []
                        
                        with st.spinner("Calculating correlations across all pairs..."):
                            for pair in analyzed_pairs:
                                try:
                                    pair_pnl = fetch_platform_pnl_for_pair(conn, pair, hours=24)
                                    
                                    if pair in analyzer.time_series_data and exchange_filter in analyzer.time_series_data[pair]:
                                        pair_metrics_data = analyzer.time_series_data[pair][exchange_filter]
                                        pair_metrics_df = pd.DataFrame(pair_metrics_data)
                                        
                                        if not pair_metrics_df.empty and pair_pnl is not None and not pair_pnl.empty:
                                            # Ensure timestamp columns are datetime
                                            for ts_col in ['timestamp_utc', 'timestamp_sg']:
                                                if ts_col in pair_metrics_df.columns:
                                                    pair_metrics_df[ts_col] = pd.to_datetime(pair_metrics_df[ts_col])
                                            
                                            # Sort by timestamp_utc
                                            if 'timestamp_utc' in pair_metrics_df.columns:
                                                pair_metrics_df = pair_metrics_df.sort_values('timestamp_utc')
                                            
                                            # Align metrics and PNL data
                                            pair_combined_df = align_data_on_common_timestamps(pair_metrics_df, pair_pnl)
                                            
                                            if pair_combined_df is not None and not pair_combined_df.empty:
                                                # Calculate correlations with aligned data
                                                hurst_corr = calculate_correlation(pair_combined_df, 'hurst_exponent')
                                                dc_range_corr = calculate_correlation(pair_combined_df, 'dc_range_ratio')
                                                dc_corr = calculate_correlation(pair_combined_df, 'direction_changes_30min')
                                                range_corr = calculate_correlation(pair_combined_df, 'absolute_range_pct')
                                                
                                                # Get final PNL
                                                final_pnl = pair_pnl['cumulative_pnl'].iloc[-1] if not pair_pnl.empty else None
                                                
                                                # Only add if we have valid correlations or PNL
                                                if (hurst_corr is not None or dc_range_corr is not None or 
                                                    dc_corr is not None or range_corr is not None or final_pnl is not None):
                                                    
                                                    # Calculate average values
                                                    avg_hurst = pair_combined_df['hurst_exponent'].mean() if 'hurst_exponent' in pair_combined_df.columns else None
                                                    avg_dc_range = pair_combined_df['dc_range_ratio'].mean() if 'dc_range_ratio' in pair_combined_df.columns else None
                                                    
                                                    correlation_data.append({
                                                        'Pair': pair,
                                                        'Hurst-PNL Corr': hurst_corr if hurst_corr is not None else float('nan'),
                                                        'DC/Range-PNL Corr': dc_range_corr if dc_range_corr is not None else float('nan'),
                                                        'DC-PNL Corr': dc_corr if dc_corr is not None else float('nan'),
                                                        'Range-PNL Corr': range_corr if range_corr is not None else float('nan'),
                                                        'Final PNL': final_pnl if final_pnl is not None else float('nan'),
                                                        'Avg Hurst': avg_hurst if avg_hurst is not None else float('nan'),
                                                        'Avg DC/Range': avg_dc_range if avg_dc_range is not None else float('nan')
                                                    })
                                except Exception as e:
                                    st.error(f"Error processing multi-pair correlation for {pair}: {e}")
                        
                        if correlation_data:
                            # Convert to DataFrame
                            correlation_df = pd.DataFrame(correlation_data)
                            
                            # Sort by Final PNL (descending)
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
                            
                            # Style DataFrame
                            styled_corr = correlation_df.style.format({
                                'Hurst-PNL Corr': '{:.3f}',
                                'DC/Range-PNL Corr': '{:.3f}',
                                'DC-PNL Corr': '{:.3f}',
                                'Range-PNL Corr': '{:.3f}',
                                'Final PNL': '${:.2f}',
                                'Avg Hurst': '{:.3f}',
                                'Avg DC/Range': '{:.3f}'
                            })
                            
                            # Apply coloring
                            styled_corr = styled_corr.applymap(color_correlation, subset=['Hurst-PNL Corr', 'DC/Range-PNL Corr', 'DC-PNL Corr', 'Range-PNL Corr'])
                            styled_corr = styled_corr.applymap(color_pnl, subset=['Final PNL'])
                            
                            # Show DataFrame
                            st.subheader("Correlation Between Mean Reversion Metrics and PNL by Pair")
                            st.dataframe(styled_corr, height=400, use_container_width=True)
                            
                            # Create scatter plot of Hurst vs PNL
                            try:
                                # Filter out rows with NaN values for the plot
                                plot_df = correlation_df.dropna(subset=['Avg Hurst', 'Final PNL'])
                                
                                if not plot_df.empty and len(plot_df) >= 3:  # Need at least 3 points for a meaningful plot
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
                                    
                                    # Add vertical line at Hurst = 0.5
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
                                        
                                        # Add annotation for Hurst=0.5
                                        fig.add_annotation(
                                            x=0.5,
                                            y=plot_df['Final PNL'].min(),
                                            text="Random Walk (H=0.5)",
                                            showarrow=False,
                                            yshift=-20
                                        )
                                    
                                    # Show plot
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Not enough valid data points to create scatter plot")
                            except Exception as e:
                                st.error(f"Error creating scatter plot: {e}")
                                
                            # Summary insights
                            st.subheader("Summary Insights")
                            
                            try:
                                # Calculate average correlations (excluding NaN values)
                                avg_hurst_corr = correlation_df['Hurst-PNL Corr'].mean()
                                avg_dc_range_corr = correlation_df['DC/Range-PNL Corr'].mean()
                                avg_dc_corr = correlation_df['DC-PNL Corr'].mean()
                                avg_range_corr = correlation_df['Range-PNL Corr'].mean()
                                
                                # Count profitable pairs
                                profitable_pairs = len(correlation_df[correlation_df['Final PNL'] > 0])
                                unprofitable_pairs = len(correlation_df[correlation_df['Final PNL'] <= 0])
                                
                                # Average Hurst for profitable pairs
                                profitable_df = correlation_df[correlation_df['Final PNL'] > 0]
                                unprofitable_df = correlation_df[correlation_df['Final PNL'] <= 0]
                                
                                avg_hurst_profitable = profitable_df['Avg Hurst'].mean() if len(profitable_df) > 0 else None
                                avg_hurst_unprofitable = unprofitable_df['Avg Hurst'].mean() if len(unprofitable_df) > 0 else None
                                
                                # Display insights
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Profitable Pairs:** {profitable_pairs} out of {len(correlation_df)}")
                                    st.write(f"**Average Correlations:**")
                                    st.write(f"- Hurst-PNL: {avg_hurst_corr:.3f}")
                                    st.write(f"- DC/Range-PNL: {avg_dc_range_corr:.3f}")
                                    st.write(f"- Direction Changes-PNL: {avg_dc_corr:.3f}")
                                    st.write(f"- Range-PNL: {avg_range_corr:.3f}")
                                
                                with col2:
                                    if avg_hurst_profitable is not None:
                                        st.write(f"**Average Hurst for Profitable Pairs:** {avg_hurst_profitable:.3f}")
                                    if avg_hurst_unprofitable is not None:
                                        st.write(f"**Average Hurst for Unprofitable Pairs:** {avg_hurst_unprofitable:.3f}")
                                    
                                    # Determine if mean reversion is beneficial overall
                                    if avg_hurst_profitable is not None and avg_hurst_profitable < 0.5:
                                        st.success("**Mean reversion (Hurst < 0.5) appears beneficial for profitability across pairs**")
                                    elif avg_hurst_profitable is not None and avg_hurst_profitable > 0.5:
                                        st.info("**Trending behavior (Hurst > 0.5) appears beneficial for profitability across pairs**")
                                    else:
                                        st.warning("**No clear pattern between Hurst exponent and profitability across pairs**")
                            except Exception as e:
                                st.error(f"Error calculating summary insights: {e}")
                        else:
                            st.warning("No correlation data available for multi-pair analysis")
                    else:
                        st.error("Could not align metrics with PNL data for visualization")
                else:
                    st.error(f"No PNL data available for {selected_pair}")
            else:
                st.error(f"No mean reversion metrics available for {selected_pair}")
        else:
            st.error(f"No time series data available for {selected_pair} on {exchange_filter}")
    else:
        st.error("No analyzed pairs found. Please run the analysis first.")
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