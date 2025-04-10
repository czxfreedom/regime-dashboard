# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
from datetime import datetime

from oracle.sweet_spot_oracle import MultiPairIndexManager
from hyperliquid.utils.exchange_config import get_base_url
from hyperliquid.info import Info
from hyperliquid.websocket_manager import WebsocketManager

# Page config
st.set_page_config(
    page_title="Crypto Sweet Spot Oracle",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}
if 'sweet_spots' not in st.session_state:
    st.session_state.sweet_spots = {}
if 'weights' not in st.session_state:
    st.session_state.weights = {}
if 'oracle_manager' not in st.session_state:
    st.session_state.oracle_manager = None
if 'available_pairs' not in st.session_state:
    st.session_state.available_pairs = []
if 'selected_pairs' not in st.session_state:
    st.session_state.selected_pairs = []
if 'run_status' not in st.session_state:
    st.session_state.run_status = False

# Fetch available pairs
@st.cache_data(ttl=3600)
def get_available_pairs():
    info = Info("mainnet")
    meta = info.meta()
    return [coin["name"] for coin in meta["universe"]]

# Header
st.title("Crypto Sweet Spot Oracle")
st.markdown("### Dynamic Order Book Analysis Tool")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Fetch available pairs if not already done
    if not st.session_state.available_pairs:
        with st.spinner("Fetching available pairs..."):
            try:
                st.session_state.available_pairs = get_available_pairs()
            except Exception as e:
                st.error(f"Error fetching pairs: {str(e)}")
                st.session_state.available_pairs = ["BTC", "ETH", "SOL", "DOGE"]  # Fallback
    
    # Pair selection
    selected_pairs = st.multiselect(
        "Select Trading Pairs",
        options=st.session_state.available_pairs,
        default=["BTC", "ETH", "SOL"] if not st.session_state.selected_pairs else st.session_state.selected_pairs
    )
    
    # Oracle parameters
    st.subheader("Oracle Parameters")
    depths = st.text_input(
        "Order Book Depths (comma-separated)",
        value="5000,10000,25000,50000,100000,150000,200000,300000"
    )
    depths = [float(d.strip()) for d in depths.split(",")]
    
    update_interval = st.slider(
        "Update Interval (seconds)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1
    )
    
    sweet_spot_interval = st.slider(
        "Sweet Spot Recalculation (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10
    )
    
    # Control buttons
    start_button = st.button("Start Oracle", type="primary", disabled=st.session_state.run_status)
    stop_button = st.button("Stop Oracle", type="secondary", disabled=not st.session_state.run_status)
    
    # Status indicator
    st.subheader("Status")
    status = st.empty()
    if st.session_state.run_status:
        status.success("Oracle is running")
    else:
        status.warning("Oracle is stopped")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Price Charts")
    price_chart = st.empty()
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["Price History", "Sweet Spots", "Weight Distribution"])
    
    with tab1:
        price_history_chart = st.empty()
    
    with tab2:
        sweet_spot_chart = st.empty()
    
    with tab3:
        weight_chart = st.empty()

with col2:
    st.header("Current Data")
    current_data = st.empty()
    
    st.subheader("Sweet Spot Analysis")
    sweet_spot_analysis = st.empty()

# Callback for price updates
def on_price_update(symbol, price):
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if symbol not in st.session_state.price_history:
        st.session_state.price_history[symbol] = []
    
    # Keep last 100 price points
    if len(st.session_state.price_history[symbol]) >= 100:
        st.session_state.price_history[symbol].pop(0)
    
    st.session_state.price_history[symbol].append({
        'timestamp': timestamp,
        'price': price
    })

# Function to start the oracle
def start_oracle(pairs, depths, update_interval, sweet_spot_interval):
    # Stop if already running
    if st.session_state.oracle_manager is not None:
        stop_oracle()
    
    try:
        # Initialize WebSocketManager
        base_url = get_base_url("mainnet")
        ws_manager = WebsocketManager(base_url)
        ws_manager.start()
        
        # Initialize Oracle Manager
        oracle_manager = MultiPairIndexManager(
            ws_manager, 
            base_url,
            update_interval=update_interval
        )
        oracle_manager.oracle_master = SweetSpotOracleMaster(
            depths=depths,
            update_interval=sweet_spot_interval
        )
        
        # Set callback
        oracle_manager.set_global_price_callback(on_price_update)
        
        # Start the oracle
        oracle_manager.start(pairs=pairs)
        
        # Update session state
        st.session_state.oracle_manager = oracle_manager
        st.session_state.selected_pairs = pairs
        st.session_state.run_status = True
        
        return True
    except Exception as e:
        st.error(f"Error starting oracle: {str(e)}")
        return False

# Function to stop the oracle
def stop_oracle():
    if st.session_state.oracle_manager is not None:
        try:
            st.session_state.oracle_manager.stop()
            st.session_state.oracle_manager = None
            st.session_state.run_status = False
            return True
        except Exception as e:
            st.error(f"Error stopping oracle: {str(e)}")
            return False
    return True

# Handle button actions
if start_button:
    start_oracle(selected_pairs, depths, update_interval, sweet_spot_interval)

if stop_button:
    stop_oracle()

# Update UI with current data
def update_ui():
    while True:
        if st.session_state.run_status and st.session_state.oracle_manager is not None:
            # Update sweet spot data
            for symbol in st.session_state.selected_pairs:
                sweet_spot_info = st.session_state.oracle_manager.get_sweet_spot_info(symbol)
                if sweet_spot_info and symbol in sweet_spot_info:
                    st.session_state.sweet_spots[symbol] = sweet_spot_info[symbol]['sweet_spots']
                    st.session_state.weights[symbol] = sweet_spot_info[symbol]['weights']
        
        # Update current data display
        if st.session_state.run_status:
            # Create current price data
            if st.session_state.oracle_manager:
                prices = st.session_state.oracle_manager.get_all_prices()
                df_current = pd.DataFrame({
                    'Symbol': list(prices.keys()),
                    'Price': list(prices.values())
                })
                current_data.dataframe(df_current, use_container_width=True)
            
            # Update price history chart
            if st.session_state.price_history:
                # Create a figure with a secondary y-axis for each symbol
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                
                for i, symbol in enumerate(st.session_state.price_history.keys()):
                    if not st.session_state.price_history[symbol]:
                        continue
                        
                    df = pd.DataFrame(st.session_state.price_history[symbol])
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)])
                    ))
                
                fig.update_layout(
                    title="Price History",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                price_history_chart.plotly_chart(fig, use_container_width=True)
            
            # Update sweet spot chart
            if st.session_state.sweet_spots:
                df_sweet_spots = pd.DataFrame({
                    'Symbol': list(st.session_state.sweet_spots.keys()),
                    'Sweet Spot': [next(iter(spots.values())) for spots in st.session_state.sweet_spots.values()]
                })
                
                fig = px.bar(
                    df_sweet_spots,
                    x='Symbol',
                    y='Sweet Spot',
                    title="Current Sweet Spots",
                    color='Symbol'
                )
                sweet_spot_chart.plotly_chart(fig, use_container_width=True)
            
            # Update weight distribution chart
            if st.session_state.weights:
                # Prepare data for weight chart
                weight_data = []
                
                for symbol, weights_dict in st.session_state.weights.items():
                    for exchange, exchange_weights in weights_dict.items():
                        for depth_str, weight in exchange_weights.items():
                            weight_data.append({
                                'Symbol': symbol,
                                'Exchange': exchange,
                                'Depth': float(depth_str),
                                'Weight': weight
                            })
                
                if weight_data:
                    df_weights = pd.DataFrame(weight_data)
                    
                    # Create weight distribution chart
                    fig = px.line(
                        df_weights,
                        x='Depth',
                        y='Weight',
                        color='Symbol',
                        title="Weight Distribution by Depth",
                        markers=True
                    )
                    fig.update_layout(xaxis_type="log")  # Log scale for depth
                    weight_chart.plotly_chart(fig, use_container_width=True)
            
            # Update sweet spot analysis
            if st.session_state.sweet_spots and st.session_state.weights:
                sweet_spot_analysis.markdown(f"""
                ### Current Sweet Spots
                {', '.join([f"{symbol}: {next(iter(spots.values()))}" for symbol, spots in st.session_state.sweet_spots.items()])}
                
                *Sweet spots represent the optimal depth in the order book where price movements are most informative.*
                
                ### Weight Distribution
                The algorithm dynamically assigns higher weights to depths near the sweet spot,
                which makes the index price more responsive to changes at those levels.
                """)
        
        time.sleep(0.5)

# Start UI update thread
ui_thread = threading.Thread(target=update_ui)
ui_thread.daemon = True
ui_thread.start()