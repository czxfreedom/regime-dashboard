import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, UTC

st.set_page_config(
    page_title="Daily Hurst Table (Multi-Window Analysis)",
    page_icon="📊",
    layout="wide"
)

# Database configuration
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

def calculate_multi_window_hurst(prices, num_windows=5, window_overlap=0.5):
    if len(prices) < 20:
        return 0.5
    
    total_length = len(prices)
    window_size = total_length // num_windows
    overlap_size = int(window_size * window_overlap)
    
    hurst_estimates = []
    
    for i in range(num_windows):
        start = max(0, i * (window_size - overlap_size))
        end = min(total_length, start + window_size)
        
        window_prices = prices[start:end]
        
        if len(window_prices) < 10:
            continue
        
        try:
            log_returns = np.diff(np.log(window_prices))
            
            hurst_methods = []
            
            try:
                autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
                hurst_methods.append(0.5 + (np.sign(autocorr) * min(abs(autocorr), 0.4)))
            except:
                pass
            
            try:
                lags = [1, 2, 4]
                var_ratios = []
                for lag in lags:
                    var_ratio = np.var(log_returns[lag:]) / np.var(log_returns)
                    var_ratios.append(var_ratio)
                
                hurst_var = 0.5 + np.mean(var_ratios) / 2
                hurst_methods.append(hurst_var)
            except:
                pass
            
            try:
                cumulative_returns = np.cumsum(log_returns)
                cum_corr = np.corrcoef(cumulative_returns[:-1], cumulative_returns[1:])[0, 1]
                hurst_methods.append(0.5 + (np.sign(cum_corr) * min(abs(cum_corr), 0.4)))
            except:
                pass
            
            if hurst_methods:
                hurst_estimates.append(np.median(hurst_methods))
        
        except Exception as e:
            continue
    
    return np.median(hurst_estimates) if hurst_estimates else 0.5

def calculate_comprehensive_hurst(token):
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=1)
    
    # Convert timestamps to strings in the format PostgreSQL expects
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    query = text(f"""
    WITH time_series AS (
        SELECT 
            generate_series(
                date_trunc('day', '{start_time_str}'::timestamp) + INTERVAL '8 hours',
                date_trunc('day', '{start_time_str}'::timestamp) + INTERVAL '32 hours',
                INTERVAL '30 minutes'
            ) AS block_time
    ),
    token_data AS (
        SELECT 
            time_series.block_time,
            final_price
        FROM time_series
        LEFT JOIN public.oracle_price_log ON 
            pair_name = '{token}' AND
            created_at >= time_series.block_time AND
            created_at < time_series.block_time + INTERVAL '30 minutes'
    )
    SELECT 
        block_time,
        final_price
    FROM token_data
    WHERE final_price IS NOT NULL
    ORDER BY block_time;
    """)
    
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        if df.empty:
            return None
        
        # Group by time blocks and calculate Hurst
        grouped_df = df.groupby('block_time').apply(
            lambda x: calculate_multi_window_hurst(x['final_price'].values)
        ).reset_index()
        grouped_df.columns = ['time_block', 'Hurst']
        
        grouped_df['time_label'] = grouped_df['time_block'].dt.strftime('%H:%M')
        
        def classify_regime(hurst):
            if pd.isna(hurst):
                return ("UNKNOWN", 0, "Insufficient data")
            elif hurst < 0.2:
                return ("MEAN-REVERT", 3, "Strong mean-reversion")
            elif hurst < 0.4:
                return ("MEAN-REVERT", 2, "Moderate mean-reversion")
            elif hurst <= 0.6:
                return ("NOISE", 0, "Random walk")
            elif hurst < 0.8:
                return ("TREND", 2, "Moderate trending")
            else:
                return ("TREND", 3, "Strong trending")
        
        grouped_df['regime_info'] = grouped_df['Hurst'].apply(classify_regime)
        grouped_df['regime'] = grouped_df['regime_info'].apply(lambda x: x[0])
        grouped_df['regime_desc'] = grouped_df['regime_info'].apply(lambda x: x[2])
        
        standard_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]]
        
        result_df = grouped_df.set_index('time_label')[['Hurst', 'regime', 'regime_desc']]
        result_df = result_df.reindex(standard_labels)
        
        return result_df
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        return None

def check_data_availability():
    try:
        with engine.connect() as connection:
            query = text("""
                SELECT 
                    pair_name, 
                    COUNT(*) as record_count
                FROM public.oracle_price_log
                GROUP BY pair_name
                ORDER BY record_count DESC
                LIMIT 100;
            """)
            
            result = connection.execute(query)
            data = result.fetchall()
            
            return [row[0] for row in data]
    
    except Exception as e:
        st.error(f"Error checking data availability: {e}")
        return []

# (Rest of the code remains the same as in the previous artifact)