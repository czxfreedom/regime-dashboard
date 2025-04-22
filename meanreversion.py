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
    page_title="Crypto Mean Reversion Monitor",
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
st.title("Crypto Mean Reversion Monitor")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2 = st.tabs(["Current Status", "Historical Trends"])

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

    def calculate_hurst_exponent(self, prices, min_k=10, max_k=None):
        """
        Calculate the Hurst exponent, which indicates:
        H < 0.5: Mean-reverting series
        H = 0.5: Random walk
        H > 0.5: Trending series
        """
        try:
            # Convert to numpy array and ensure we have enough data
            prices = np.array(prices)
            if len(prices) < 100:
                return 0.5  # Return neutral value if not enough data
                
            # Calculate returns
            returns = np.log(prices[1:] / prices[:-1])
            
            # Set default max_k if not provided
            if max_k is None:
                max_k = min(int(len(returns) / 10), 120)  # Use at most 120 or 1/10 of series length
            
            if max_k <= min_k:
                max_k = min_k + 10
                
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
            if len(rs_values) < 5:
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
            
            if len(df_lookback) < 100:  # Need at least 100 data points for time series
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
                
                if len(df_interval) < 10:  # Skip intervals with insufficient data
                    continue
                    
                # Extract prices
                prices = pd.to_numeric(df_interval['price'], errors='coerce').dropna().values
                
                # Calculate metrics
                direction_changes = self.calculate_direction_changes(prices)
                range_pct = self.calculate_range_percentage(prices)
                
                # Calculate ratio
                dc_range_ratio = direction_changes / (range_pct + 1e-10)
                
                # Calculate Hurst exponent
                # For historical data, use only the interval data to see how it changes
                hurst = self.calculate_hurst_exponent(prices, min_k=5, max_k=min(len(prices)//4, 20))
                
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
        # Calculate times
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
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
            
            # Processed results
            results = []
            
            # Process each pair for both exchanges
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress((i) / len(pairs_to_analyze))
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

    def create_time_series_chart(self, pair, exchange, metric):
        """Create a time series chart for a specific pair, exchange, and metric."""
        if pair not in self.time_series_data or exchange not in self.time_series_data[pair]:
            return None
            
        # Get time series data
        ts_data = self.time_series_data[pair][exchange]
        
        # Convert to DataFrame
        df = pd.DataFrame(ts_data)
        
        # Create time series plot
        fig = px.line(
            df, 
            x='timestamp', 
            y=metric,
            title=f"{self.metric_display_names[metric]} for {pair} ({exchange.upper()})"
        )
        
        # Add reference lines for Hurst exponent
        if metric == 'hurst_exponent':
            fig.add_shape(
                type="line",
                x0=df['timestamp'].min(),
                y0=0.5,
                x1=df['timestamp'].max(),
                y1=0.5,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                )
            )
            
            # Add annotation for Hurst reference
            fig.add_annotation(
                x=df['timestamp'].max(),
                y=0.5,
                text="Random Walk (H=0.5)",
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=self.metric_display_names[metric],
            height=400
        )
        
        return fig


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

# Process when form is submitted
if submit_button:
    if not conn:
        st.error("Database connection not available.")
    elif not pairs:
        st.error("Please enter at least one pair to analyze.")
    else:
        # Initialize analyzer
        analyzer = MeanReversionAnalyzer()
        
        # Run analysis
        with st.spinner("Fetching and analyzing data..."):
            results = analyzer.fetch_and_analyze(
                conn=conn,
                pairs_to_analyze=pairs,
                hours=hours
            )
        
        if results:
            # Tab 1: Current Status
            with tab1:
                st.header("Current Mean Reversion Status")
                
                # Convert exchange filter for data filtering
                filter_value = None if exchange_filter == "Both" else exchange_filter
                
                # Create current status table
                status_df = analyzer.create_current_status_table(results, filter_value)
                
                if status_df is not None:
                    # Style the table to highlight mean reversion conditions
                    def style_mean_reversion_table(val, col_name):
                        """Highlight mean reversion indicators."""
                        if col_name == 'hurst_exponent':
                            if val < 0.4:
                                return 'background-color: #a0d995; color: black'  # Strong mean reversion (green)
                            elif val < 0.5:
                                return 'background-color: #f1f1aa; color: black'  # Mild mean reversion (yellow)
                            else:
                                return 'background-color: #ffc299; color: black'  # No mean reversion (orange)
                        elif col_name == 'dc_range_ratio':
                            if val > 5:
                                return 'background-color: #a0d995; color: black'  # High ratio (green)
                            elif val > 2:
                                return 'background-color: #f1f1aa; color: black'  # Medium ratio (yellow)
                            else:
                                return 'background-color: #ffc299; color: black'  # Low ratio (orange)
                        elif col_name == 'mean_reversion_score':
                            if val > 50:
                                return 'background-color: #60b33c; color: white; font-weight: bold'  # Strong (green)
                            elif val > 40:
                                return 'background-color: #a0d995; color: black'  # Good (light green)
                            elif val > 30:
                                return 'background-color: #f1f1aa; color: black'  # Moderate (yellow)
                            else:
                                return 'background-color: #ffc299; color: black'  # Weak (orange)
                        return ''
                    
                    # Define columns to display
                    display_cols = ['pair', 'exchange', 'direction_changes_30min', 'absolute_range_pct', 
                                    'dc_range_ratio', 'hurst_exponent', 'mean_reversion_score']
                    
                    # Apply styling
                    styled_df = status_df[display_cols].style.applymap(
                        lambda x: style_mean_reversion_table(x, status_df.columns[status_df.columns.get_loc(
                            next((col for col in display_cols if pd.Series(x).name == col), None)
                        )]),
                        subset=['hurst_exponent', 'dc_range_ratio', 'mean_reversion_score']
                    )
                    
                    # Display the table
                    st.dataframe(
                        styled_df,
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
            
            # Tab 2: Historical Trends
            with tab2:
                st.header("Historical Mean Reversion Trends")
                
                # Select a pair for detailed analysis
                if results:
                    unique_pairs = sorted(list(set([r['pair'] for r in results])))
                    selected_pair = st.selectbox("Select Pair for Historical Analysis", unique_pairs)
                    
                    # Select exchange
                    exchanges_with_data = [r['exchange'] for r in results if r['pair'] == selected_pair]
                    if exchanges_with_data:
                        selected_exchange = st.radio(
                            "Select Exchange", 
                            exchanges_with_data,
                            horizontal=True
                        )
                        
                        # Create time series charts
                        st.subheader(f"30-Minute Interval Metrics for {selected_pair} on {selected_exchange.upper()}")
                        
                        # DC/Range Ratio chart
                        dc_range_fig = analyzer.create_time_series_chart(
                            selected_pair,
                            selected_exchange,
                            'dc_range_ratio'
                        )
                        if dc_range_fig:
                            st.plotly_chart(dc_range_fig, use_container_width=True)
                        else:
                            st.warning(f"No historical DC/Range Ratio data for {selected_pair} on {selected_exchange}")
                        
                        # Hurst Exponent chart
                        hurst_fig = analyzer.create_time_series_chart(
                            selected_pair,
                            selected_exchange,
                            'hurst_exponent'
                        )
                        if hurst_fig:
                            st.plotly_chart(hurst_fig, use_container_width=True)
                        else:
                            st.warning(f"No historical Hurst Exponent data for {selected_pair} on {selected_exchange}")
                        
                        # Direction Changes chart
                        dir_changes_fig = analyzer.create_time_series_chart(
                            selected_pair,
                            selected_exchange,
                            'direction_changes_30min'
                        )
                        if dir_changes_fig:
                            st.plotly_chart(dir_changes_fig, use_container_width=True)
                        else:
                            st.warning(f"No historical Direction Changes data for {selected_pair} on {selected_exchange}")
                            
                        # Range Percentage chart
                        range_fig = analyzer.create_time_series_chart(
                            selected_pair,
                            selected_exchange,
                            'absolute_range_pct'
                        )
                        if range_fig:
                            st.plotly_chart(range_fig, use_container_width=True)
                        else:
                            st.warning(f"No historical Range % data for {selected_pair} on {selected_exchange}")
                    else:
                        st.warning(f"No exchanges with data for {selected_pair}")
                else:
                    st.warning("No pairs available for historical analysis")
        else:
            st.error("Failed to analyze data. Please try again with different parameters.")

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

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")