# Sweetspot.py - Complete self-contained implementation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import json
import websocket
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Callable, Optional, NamedTuple

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== Order Book Analysis Classes =====

class OrderBookSnapshot:
    """Store a snapshot of the order book at a specific time"""
    def __init__(self, timestamp: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        self.timestamp = timestamp
        self.bids = bids  # List of (price, size) tuples
        self.asks = asks  # List of (price, size) tuples
    
    def mid_price(self) -> float:
        """Calculate the mid price from top of book"""
        if not self.bids or not self.asks:
            return 0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    def price_at_depth(self, depth: float) -> Tuple[float, float]:
        """Calculate price at a specific depth"""
        bid_depth = 0
        ask_depth = 0
        
        bid_price = self.bids[0][0] if self.bids else 0
        ask_price = self.asks[0][0] if self.asks else 0
        
        # Accumulate bid depth
        for price, size in self.bids:
            bid_depth += price * size
            if bid_depth >= depth:
                bid_price = price
                break
                
        # Accumulate ask depth
        for price, size in self.asks:
            ask_depth += price * size
            if ask_depth >= depth:
                ask_price = price
                break
                
        return bid_price, ask_price

class ExchangeData:
    """Store and analyze order book data for a specific exchange"""
    def __init__(self, exchange_name: str, max_history: int = 300):
        self.exchange_name = exchange_name
        self.order_book_history = deque(maxlen=max_history)  # Store last 5 minutes @ 1 snapshot/second
        self.last_update_time = 0
        self.current_bids = []
        self.current_asks = []
        
    def update_order_book(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Update the current order book and add to history"""
        self.current_bids = bids
        self.current_asks = asks
        
        # Create a snapshot and add to history
        timestamp = time.time()
        if timestamp - self.last_update_time >= 1.0:  # Store max 1 snapshot per second
            snapshot = OrderBookSnapshot(timestamp, bids.copy(), asks.copy())
            self.order_book_history.append(snapshot)
            self.last_update_time = timestamp
    
    def get_current_snapshot(self) -> OrderBookSnapshot:
        """Get the current order book snapshot"""
        return OrderBookSnapshot(time.time(), self.current_bids, self.current_asks)
    
    def calculate_volatility_at_depth(self, depth: float, lookback: int = 30) -> float:
        """Calculate price volatility at a specific depth"""
        if len(self.order_book_history) < lookback:
            return 0
            
        # Get prices at depth for the lookback period
        prices = []
        for i in range(min(lookback, len(self.order_book_history))):
            snapshot = self.order_book_history[-(i+1)]
            bid, ask = snapshot.price_at_depth(depth)
            mid = (bid + ask) / 2
            prices.append(mid)
            
        if not prices:
            return 0
            
        # Calculate returns and volatility
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * 100  # As percentage
    
    def calculate_sweet_spot(self, depths: List[float], lookback: int = 30) -> float:
        """Find the depth with optimal volatility and stability"""
        if len(self.order_book_history) < lookback:
            return depths[len(depths) // 2]  # Default to middle depth if not enough data
            
        volatilities = {}
        price_changes = {}
        
        # Calculate volatility for each depth
        for depth in depths:
            volatilities[depth] = self.calculate_volatility_at_depth(depth, lookback)
            
            # Calculate how much prices at this depth change
            price_changes_at_depth = []
            for i in range(min(lookback-1, len(self.order_book_history)-1)):
                snapshot1 = self.order_book_history[-(i+1)]
                snapshot2 = self.order_book_history[-(i+2)]
                
                bid1, ask1 = snapshot1.price_at_depth(depth)
                bid2, ask2 = snapshot2.price_at_depth(depth)
                
                mid1 = (bid1 + ask1) / 2
                mid2 = (bid2 + ask2) / 2
                
                if mid2 != 0:
                    change = abs(mid1 - mid2) / mid2
                    price_changes_at_depth.append(change)
            
            if price_changes_at_depth:
                price_changes[depth] = np.mean(price_changes_at_depth) * 100  # As percentage
            else:
                price_changes[depth] = 0
        
        # Calculate composite score (balance of volatility and responsiveness)
        scores = {}
        for depth in depths:
            # We want moderate volatility - not too high, not too low
            volatility_score = volatilities[depth]
            if volatility_score > 0.5:  # Cap at 0.5% to prevent extreme values dominating
                volatility_score = 0.5
                
            # We want responsive prices (higher price changes)
            change_score = price_changes[depth]
            
            # Composite score - higher is better
            scores[depth] = (volatility_score * 0.5) + (change_score * 0.5)
        
        # Return depth with highest score
        if not scores:
            return depths[len(depths) // 2]
        return max(scores, key=scores.get)


class SweetSpotOracle:
    """Oracle that identifies sweet spots in order book depth and calculates index prices"""
    def __init__(self, depths: List[float] = None, update_interval: int = 60):
        self.exchanges = {}  # Exchange name -> ExchangeData
        self.depths = depths or [5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000]
        self.sweet_spots = {}  # Exchange name -> sweet spot depth
        self.weights = {}  # Exchange name -> {depth -> weight}
        self.last_sweet_spot_update = 0
        self.update_interval = update_interval  # How often to update sweet spots (seconds)
        self.lock = threading.Lock()
        
    def register_exchange(self, exchange_name: str):
        """Register a new exchange to track"""
        with self.lock:
            if exchange_name not in self.exchanges:
                self.exchanges[exchange_name] = ExchangeData(exchange_name)
                # Initialize with default sweet spot (middle of depths)
                self.sweet_spots[exchange_name] = self.depths[len(self.depths) // 2]
                # Initialize with flat weights
                self.weights[exchange_name] = {depth: 1.0 / len(self.depths) for depth in self.depths}
    
    def update_order_book(self, exchange_name: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Update order book data for an exchange"""
        with self.lock:
            if exchange_name not in self.exchanges:
                self.register_exchange(exchange_name)
                
            self.exchanges[exchange_name].update_order_book(bids, asks)
            
            # Update sweet spots periodically
            current_time = time.time()
            if current_time - self.last_sweet_spot_update >= self.update_interval:
                self._update_sweet_spots()
                self._update_weights()
                self.last_sweet_spot_update = current_time
    
    def _update_sweet_spots(self):
        """Update sweet spots for all exchanges"""
        for exchange_name, exchange_data in self.exchanges.items():
            sweet_spot = exchange_data.calculate_sweet_spot(self.depths)
            self.sweet_spots[exchange_name] = sweet_spot
            logging.info(f"Updated sweet spot for {exchange_name}: {sweet_spot}")
    
    def _update_weights(self):
        """Update weights based on sweet spots"""
        for exchange_name, sweet_spot in self.sweet_spots.items():
            # Create exponential weights centered around the sweet spot
            weights = {}
            for depth in self.depths:
                # Calculate distance from sweet spot (normalized)
                max_distance = max(self.depths) - min(self.depths)
                distance = abs(depth - sweet_spot) / max_distance
                
                # Inverted exponential weight - higher at sweet spot, lower further away
                alpha = 3  # Amplification factor
                L = 1.0 / max_distance
                weights[depth] = L * np.exp(-alpha * L * distance)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for depth in weights:
                    weights[depth] /= total_weight
                    
            self.weights[exchange_name] = weights
    
    def calculate_price(self, exchange_name: str) -> float:
        """Calculate price for a specific exchange using sweet spot weights"""
        with self.lock:
            if exchange_name not in self.exchanges:
                return 0
                
            exchange_data = self.exchanges[exchange_name]
            current_snapshot = exchange_data.get_current_snapshot()
            
            weighted_price = 0
            total_weight = 0
            
            for depth, weight in self.weights[exchange_name].items():
                bid, ask = current_snapshot.price_at_depth(depth)
                if bid == 0 or ask == 0:
                    continue
                    
                mid_price = (bid + ask) / 2
                weighted_price += mid_price * weight
                total_weight += weight
            
            if total_weight == 0:
                return current_snapshot.mid_price()  # Fallback to top of book
                
            return weighted_price / total_weight
    
    def get_sweet_spot_info(self) -> Dict:
        """Get current sweet spot info for all exchanges"""
        with self.lock:
            info = {
                'sweet_spots': self.sweet_spots.copy(),
                'weights': {
                    exch: {str(depth): weight for depth, weight in weights.items()}
                    for exch, weights in self.weights.items()
                }
            }
            return info

class SweetSpotOracleMaster:
    """Master oracle that manages multiple symbol-specific oracles"""
    def __init__(self, depths: List[float] = None, update_interval: int = 60):
        self.depths = depths or [5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000]
        self.update_interval = update_interval
        self.oracles = {}  # symbol -> SweetSpotOracle
        self.lock = threading.Lock()
    
    def get_or_create_oracle(self, symbol: str) -> SweetSpotOracle:
        """Get or create an oracle for a specific symbol"""
        with self.lock:
            if symbol not in self.oracles:
                self.oracles[symbol] = SweetSpotOracle(
                    depths=self.depths, 
                    update_interval=self.update_interval
                )
            return self.oracles[symbol]
    
    def update_order_book(self, exchange_name: str, symbol: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Update order book for a specific exchange and symbol"""
        oracle = self.get_or_create_oracle(symbol)
        oracle.update_order_book(exchange_name, bids, asks)
    
    def calculate_price(self, symbol: str) -> float:
        """Calculate index price for a specific symbol"""
        if symbol not in self.oracles:
            return 0
        return self.oracles[symbol].calculate_index_price()
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get index prices for all symbols"""
        with self.lock:
            return {symbol: oracle.calculate_index_price() for symbol, oracle in self.oracles.items()}
    
    def get_sweet_spot_info(self, symbol: str = None) -> Dict:
        """Get sweet spot info for one or all symbols"""
        with self.lock:
            if symbol:
                if symbol in self.oracles:
                    return {symbol: self.oracles[symbol].get_sweet_spot_info()}
                return {}
            else:
                return {symbol: oracle.get_sweet_spot_info() for symbol, oracle in self.oracles.items()}

# ===== Mock Exchange Connector for Testing =====

class MockExchangeConnector:
    """Simulates exchange data for testing"""
    def __init__(self, symbol="BTC", update_interval=0.5):
        self.symbol = symbol
        self.update_interval = update_interval
        self.base_price = 50000  # Starting price for BTC
        self.volatility = 0.001  # Price volatility
        self.running = False
        self.thread = None
        self.oracle_master = None
        self.callbacks = []
    
    def set_oracle_master(self, oracle_master):
        self.oracle_master = oracle_master
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._simulate_market_data)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _simulate_market_data(self):
        """Generate simulated market data"""
        while self.running:
            # Simulate price change
            price_change = np.random.normal(0, self.volatility)
            self.base_price *= (1 + price_change)
            
            # Generate order book
            bids = self._generate_order_book(self.base_price, "bid")
            asks = self._generate_order_book(self.base_price, "ask")
            
            # Update oracle with order book data
            if self.oracle_master:
                self.oracle_master.update_order_book("simulation", self.symbol, bids, asks)
            
            # Notify callbacks
            for callback in self.callbacks:
                callback(self.symbol, self.base_price)
            
            time.sleep(self.update_interval)
    
    def _generate_order_book(self, price, side):
        """Generate a simulated order book"""
        result = []
        levels = 20  # Number of price levels to generate
        
        if side == "bid":
            # Generate bids (descending prices)
            for i in range(levels):
                level_price = price * (1 - i * 0.001)  # Each level is 0.1% lower
                size = np.random.uniform(0.1, 10)  # Random size between 0.1 and 10 BTC
                result.append((level_price, size))
            return sorted(result, key=lambda x: x[0], reverse=True)  # Sort descending
        else:
            # Generate asks (ascending prices)
            for i in range(levels):
                level_price = price * (1 + i * 0.001)  # Each level is 0.1% higher
                size = np.random.uniform(0.1, 10)  # Random size between 0.1 and 10 BTC
                result.append((level_price, size))
            return sorted(result, key=lambda x: x[0])  # Sort ascending

# ===== Streamlit Application =====

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
if 'oracle_master' not in st.session_state:
    st.session_state.oracle_master = None
if 'connector' not in st.session_state:
    st.session_state.connector = None
if 'available_pairs' not in st.session_state:
    st.session_state.available_pairs = ["BTC", "ETH", "SOL", "DOGE"]
if 'selected_pairs' not in st.session_state:
    st.session_state.selected_pairs = []
if 'run_status' not in st.session_state:
    st.session_state.run_status = False

# Header
st.title("Crypto Sweet Spot Oracle")
st.markdown("### Dynamic Order Book Analysis Tool")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Pair selection
    selected_pairs = st.multiselect(
        "Select Trading Pairs",
        options=st.session_state.available_pairs,
        default=["BTC"] if not st.session_state.selected_pairs else st.session_state.selected_pairs
    )
    
    # Oracle parameters
    st.subheader("Oracle Parameters")
    depths_input = st.text_input(
        "Order Book Depths (comma-separated)",
        value="5000,10000,25000,50000,100000,150000,200000,300000"
    )
    depths = [float(d.strip()) for d in depths_input.split(",")]
    
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
    if st.session_state.oracle_master is not None:
        stop_oracle()
    
    try:
        # Initialize Oracle Master
        oracle_master = SweetSpotOracleMaster(
            depths=depths,
            update_interval=sweet_spot_interval
        )
        
        # Initialize mock connector for each selected pair
        connectors = {}
        for symbol in pairs:
            connector = MockExchangeConnector(symbol=symbol, update_interval=update_interval)
            connector.set_oracle_master(oracle_master)
            connector.add_callback(on_price_update)
            connector.start()
            connectors[symbol] = connector
        
        # Update session state
        st.session_state.oracle_master = oracle_master
        st.session_state.connectors = connectors
        st.session_state.selected_pairs = pairs
        st.session_state.run_status = True
        
        return True
    except Exception as e:
        st.error(f"Error starting oracle: {str(e)}")
        return False

# Function to stop the oracle
def stop_oracle():
    if st.session_state.connectors:
        try:
            for symbol, connector in st.session_state.connectors.items():
                connector.stop()
            st.session_state.connectors = {}
            st.session_state.oracle_master = None
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
        if st.session_state.run_status and st.session_state.oracle_master is not None:
            # Update sweet spot data
            for symbol in st.session_state.selected_pairs:
                sweet_spot_info = st.session_state.oracle_master.get_sweet_spot_info(symbol)
                if sweet_spot_info and symbol in sweet_spot_info:
                    st.session_state.sweet_spots[symbol] = sweet_spot_info[symbol].get('sweet_spots', {})
                    st.session_state.weights[symbol] = sweet_spot_info[symbol].get('weights', {})
        
        # Update current data display
        if st.session_state.run_status and hasattr(st.session_state, 'price_history'):
            # Create current price data
            prices = {}
            for symbol in st.session_state.price_history:
                if st.session_state.price_history[symbol]:
                    prices[symbol] = st.session_state.price_history[symbol][-1]['price']
            
            if prices:
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
                sweet_spot_data = []
                for symbol, spots in st.session_state.sweet_spots.items():
                    for exchange, depth in spots.items():
                        sweet_spot_data.append({
                            'Symbol': symbol,
                            'Exchange': exchange,
                            'Sweet Spot': depth
                        })
                
                if sweet_spot_data:
                    df_sweet_spots = pd.DataFrame(sweet_spot_data)
                    
                    fig = px.bar(
                        df_sweet_spots,
                        x='Symbol',
                        y='Sweet Spot',
                        title="Current Sweet Spots",
                        color='Exchange'
                    )
                    sweet_spot_chart.plotly_chart(fig, use_container_width=True)
            
            # Update weight distribution chart
            if st.session_state.weights:
                # Prepare data for weight chart
                weight_data = []
                
                for symbol, exchanges_weights in st.session_state.weights.items():
                    for exchange, exchange_weights in exchanges_weights.items():
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
            if st.session_state.sweet_spots:
                sweet_spot_text = []
                for symbol, spots in st.session_state.sweet_spots.items():
                    spot_values = [f"{exchange}: {depth}" for exchange, depth in spots.items()]
                    sweet_spot_text.append(f"**{symbol}**: {', '.join(spot_values)}")
                
                sweet_spot_analysis.markdown(f"""
                ### Current Sweet Spots
                {'  \n'.join(sweet_spot_text)}
                
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

# Display instructions and explanation
st.markdown("""
## How the Sweet Spot Oracle Works

1. **Order Book Analysis**: The oracle analyzes the order book at different depth levels.

2. **Sweet Spot Detection**: It identifies the "sweet spot" - the depth level where price movements are most informative.

3. **Dynamic Weighting**: It assigns weights to different depth levels, with higher weights near the sweet spot.

4. **Price Calculation**: The final price is calculated as a weighted average of prices at different depths.

This approach dynamically adapts to market conditions, making the price more responsive and less manipulable.

## Implementation Details

- The current implementation uses simulated data for demonstration purposes.
- In a production environment, it would connect to real exchange APIs.
- The sweet spot detection algorithm balances volatility and responsiveness.
- The inverted exponential weighting formula emphasizes depths near the sweet spot.
""")