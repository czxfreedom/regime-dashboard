import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import json
import logging
import pkg_resources  # Add this import here
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Callable, Optional, NamedTuple

try:
    from hyperliquid.info import Info
    from hyperliquid.utils.exchange_config import get_base_url
    from hyperliquid.websocket_manager import WebsocketManager
except ImportError as e:
    st.error(f"Error importing Hyperliquid: {str(e)}")
    st.error(f"Installed packages: {', '.join(sorted([pkg.key for pkg in pkg_resources.working_set]))}")
    st.stop()

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
    
    def calculate_index_price(self) -> float:
        """Calculate index price across all exchanges"""
        with self.lock:
            prices = {}
            for exchange_name in self.exchanges:
                price = self.calculate_price(exchange_name)
                if price > 0:
                    prices[exchange_name] = price
            
            if not prices:
                return 0
                
            # Equal weight for all exchanges for now
            # This could be enhanced with exchange-specific weighting
            return sum(prices.values()) / len(prices)
    
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

# ===== Real Exchange Connector =====

class HyperliquidConnector:
    """Connects to Hyperliquid exchange and processes real-time data"""
    def __init__(self, network="mainnet"):
        self.network = network
        self.base_url = get_base_url(network)
        self.ws_manager = None
        self.oracle_master = None
        self.running = False
        self.subscriptions = {}  # symbol -> subscription_id
        self.callbacks = {}  # symbol -> callback
        self.current_prices = {}  # symbol -> price
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
    
    def set_oracle_master(self, oracle_master):
        self.oracle_master = oracle_master
    
    def add_callback(self, symbol, callback):
        self.callbacks[symbol] = callback
    
    def start(self, symbols=None):
        if self.running:
            return
        
        try:
            # Create websocket manager
            self.ws_manager = WebsocketManager(self.base_url)
            self.ws_manager.start()
            
            # Get available symbols if none provided
            if not symbols:
                symbols = self._get_available_symbols()
                
            self.running = True
            
            # Subscribe to order book for each symbol
            for symbol in symbols:
                self._subscribe_to_order_book(symbol)
                
            logging.info(f"Successfully connected to Hyperliquid and subscribed to {len(symbols)} symbols")
            self._reconnect_attempts = 0  # Reset reconnect counter on successful connection
            
        except Exception as e:
            self._reconnect_attempts += 1
            logging.error(f"Error starting Hyperliquid connector: {str(e)}")
            
            if self._reconnect_attempts < self._max_reconnect_attempts:
                logging.info(f"Reconnect attempt {self._reconnect_attempts} of {self._max_reconnect_attempts}...")
                time.sleep(2)  # Wait before reconnecting
                self.start(symbols)
            else:
                logging.error("Max reconnect attempts reached. Please check your connection and try again.")
                raise e
    
    def stop(self):
        self.running = False
        if self.ws_manager:
            try:
                # Unsubscribe from all
                for symbol, sub_id in self.subscriptions.items():
                    self._unsubscribe_from_order_book(symbol, sub_id)
                self.ws_manager.stop()
                logging.info("Hyperliquid connector stopped")
            except Exception as e:
                logging.error(f"Error stopping Hyperliquid connector: {str(e)}")
    
    def _get_available_symbols(self):
        """Get available symbols from Hyperliquid"""
        try:
            info = Info(self.network)
            meta = info.meta()
            return [coin["name"] for coin in meta["universe"]]
        except Exception as e:
            logging.error(f"Error fetching symbols: {str(e)}")
            # Fallback to common symbols
            return ["BTC", "ETH", "SOL"]
    
    def _subscribe_to_order_book(self, symbol):
        """Subscribe to order book updates for a symbol"""
        try:
            subscription = {
                "type": "l2Book",
                "coin": symbol
            }
            sub_id = self.ws_manager.subscribe(
                subscription, 
                lambda msg: self._handle_order_book_message(msg, symbol)
            )
            self.subscriptions[symbol] = sub_id
            logging.info(f"Subscribed to {symbol} order book with subscription ID {sub_id}")
        except Exception as e:
            logging.error(f"Error subscribing to {symbol} order book: {str(e)}")
    
    def _unsubscribe_from_order_book(self, symbol, subscription_id):
        """Unsubscribe from order book updates"""
        try:
            subscription = {
                "type": "l2Book",
                "coin": symbol
            }
            self.ws_manager.unsubscribe(subscription, subscription_id)
            logging.info(f"Unsubscribed from {symbol} order book")
        except Exception as e:
            logging.error(f"Error unsubscribing from {symbol} order book: {str(e)}")
    
    def _handle_order_book_message(self, msg, symbol):
        """Process order book update messages"""
        try:
            if msg and "channel" in msg and msg["channel"] == "l2Book" and "data" in msg and "coin" in msg["data"] and msg["data"]["coin"].upper() == symbol.upper():
                # Extract bid and ask data
                data = msg["data"]
                
                # Check if bids and asks exist
                if "bids" not in data or "asks" not in data:
                    logging.warning(f"Incomplete order book data for {symbol}: missing bids or asks")
                    return
                
                bids = [(float(bid["px"]), float(bid["sz"])) for bid in data.get("bids", [])]
                asks = [(float(ask["px"]), float(ask["sz"])) for ask in data.get("asks", [])]
                
                # Calculate current mid price
                if bids and asks:
                    mid_price = (bids[0][0] + asks[0][0]) / 2
                    self.current_prices[symbol] = mid_price
                    
                    # Call price update callback if registered
                    if symbol in self.callbacks and callable(self.callbacks[symbol]):
                        self.callbacks[symbol](symbol, mid_price)
                
                # Update oracle with order book data
                if self.oracle_master:
                    self.oracle_master.update_order_book("hyperliquid", symbol, bids, asks)
                    
                logging.debug(f"Processed {symbol} order book update: {len(bids)} bids, {len(asks)} asks")
        except Exception as e:
            logging.error(f"Error processing {symbol} order book update: {str(e)}")
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        return self.current_prices.get(symbol, 0)

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
    st.session_state.available_pairs = []
if 'selected_pairs' not in st.session_state:
    st.session_state.selected_pairs = []
if 'run_status' not in st.session_state:
    st.session_state.run_status = False
if 'connection_error' not in st.session_state:
    st.session_state.connection_error = None

# Fetch available pairs
@st.cache_data(ttl=3600)
def get_available_pairs():
    try:
        info = Info("mainnet")
        meta = info.meta()
        return [coin["name"] for coin in meta["universe"]]
    except Exception as e:
        error_msg = f"Error fetching pairs: {str(e)}"
        st.session_state.connection_error = error_msg
        logging.error(error_msg)
        return ["BTC", "ETH", "SOL"]  # Fallback

# Header
st.title("Crypto Sweet Spot Oracle")
st.markdown("### Dynamic Order Book Analysis Tool")

# Display any connection errors
if st.session_state.connection_error:
    st.error(st.session_state.connection_error)
    if st.button("Clear Error"):
        st.session_state.connection_error = None
        st.experimental_rerun()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Fetch available pairs if not already done
    if not st.session_state.available_pairs:
        with st.spinner("Fetching available pairs..."):
            st.session_state.available_pairs = get_available_pairs()
    
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
    
    try:
        depths = [float(d.strip()) for d in depths_input.split(",")]
    except ValueError:
        st.error("Invalid depth values. Please enter comma-separated numbers.")
        depths = [5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000]
    
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
    
    # Network selection
    network = st.radio("Network", ["mainnet", "testnet"], index=0)
    
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
def start_oracle(pairs, depths, update_interval, sweet_spot_interval, network="mainnet"):
    # Stop if already running
    if st.session_state.oracle_master is not None:
        stop_oracle()
    
    try:
        # Clear any previous connection errors
        st.session_state.connection_error = None
        
        # Initialize Oracle Master
        oracle_master = SweetSpotOracleMaster(
            depths=depths,
            update_interval=sweet_spot_interval
        )
        
        # Initialize Hyperliquid connector
        connector = HyperliquidConnector(network=network)
        connector.set_oracle_master(oracle_master)
        
        # Register callbacks for each pair
        for symbol in pairs:
            connector.add_callback(symbol, on_price_update)
        
        # Start the connector with selected pairs
        connector.start(symbols=pairs)
        
        # Update session state
        st.session_state.oracle_master = oracle_master
        st.session_state.connector = connector
        st.session_state.selected_pairs = pairs
        st.session_state.run_status = True
        
        return True
    except Exception as e:
        error_msg = f"Error starting oracle: {str(e)}"
        st.session_state.connection_error = error_msg
        logging.error(error_msg)
        st.session_state.run_status = False
        return False

# Function to stop the oracle
def stop_oracle():
    if st.session_state.connector:
        try:
            st.session_state.connector.stop()
            st.session_state.connector = None
            st.session_state.oracle_master = None
            st.session_state.run_status = False
            return True
        except Exception as e:
            error_msg = f"Error stopping oracle: {str(e)}"
            st.session_state.connection_error = error_msg
            logging.error(error_msg)
            return False
    return True

# Handle button actions
if start_button:
    start_oracle(selected_pairs, depths, update_interval, sweet_spot_interval, network)

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

1. **Order Book Analysis**: The oracle analyzes the Hyperliquid order book at different depth levels.

2. **Sweet Spot Detection**: It identifies the "sweet spot" - the depth level where price movements are most informative.

3. **Dynamic Weighting**: It assigns weights to different depth levels, with higher weights near the sweet spot.

4. **Price Calculation**: The final price is calculated as a weighted average of prices at different depths.

This approach dynamically adapts to market conditions, making the price more responsive and less manipulable.

## Implementation Details

- The implementation connects to Hyperliquid exchange via WebSockets
- Order book data is analyzed in real-time to identify sweet spots
- The sweet spot detection algorithm balances volatility and responsiveness
- The inverted exponential weighting formula emphasizes depths near the sweet spot
- Parameters can be adjusted to optimize for different market conditions
""")

# Display real-time metrics at the bottom
st.header("Execution Metrics")
metrics_container = st.container()

# Create columns for different metrics
metric_col1, metric_col2, metric_col3, metric_col4 = metrics_container.columns(4)

# Define metric placeholders
order_book_updates_count = 0
sweet_spot_updates_count = 0
price_updates_count = 0
last_update_time = "N/A"

# Function to update metrics
def update_metrics():
    global order_book_updates_count, sweet_spot_updates_count, price_updates_count, last_update_time
    
    while True:
        if st.session_state.run_status:
            # Update the metrics
            order_book_updates_count += 1
            if order_book_updates_count % 20 == 0:  # Every 20 order book updates
                sweet_spot_updates_count += 1
            price_updates_count += 1
            last_update_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Display the metrics
            with metric_col1:
                st.metric("Order Book Updates", order_book_updates_count)
            
            with metric_col2:
                st.metric("Sweet Spot Recalculations", sweet_spot_updates_count)
                
            with metric_col3:
                st.metric("Price Updates", price_updates_count)
                
            with metric_col4:
                st.metric("Last Update", last_update_time)
                
        time.sleep(1)

# Start metrics update thread
metrics_thread = threading.Thread(target=update_metrics)
metrics_thread.daemon = True
metrics_thread.start()

# Status monitoring thread - checks for WebSocket health
def monitor_connection():
    while True:
        if st.session_state.run_status and st.session_state.connector:
            # Check if we're receiving data
            active_symbols = list(st.session_state.price_history.keys())
            data_received = False
            
            for symbol in active_symbols:
                # Check if we have recent price updates (within last 10 seconds)
                if (symbol in st.session_state.price_history and 
                    st.session_state.price_history[symbol] and 
                    len(st.session_state.price_history[symbol]) > 1):
                    data_received = True
                    break
            
            if not data_received and st.session_state.run_status:
                # No data received, connection might be dead
                logging.warning("No data received recently. Connection may be lost.")
                
                # Try to reconnect automatically
                try:
                    st.session_state.connector.stop()
                    time.sleep(2)
                    
                    # Recreate connector and restart
                    connector = HyperliquidConnector(network="mainnet")
                    connector.set_oracle_master(st.session_state.oracle_master)
                    
                    for symbol in st.session_state.selected_pairs:
                        connector.add_callback(symbol, on_price_update)
                    
                    connector.start(symbols=st.session_state.selected_pairs)
                    st.session_state.connector = connector
                    
                    logging.info("Automatically reconnected to Hyperliquid")
                except Exception as e:
                    logging.error(f"Failed to auto-reconnect: {str(e)}")
        
        time.sleep(10)  # Check every 10 seconds

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_connection)
monitor_thread.daemon = True
monitor_thread.start()

# Display developer info
st.sidebar.markdown("---")
st.sidebar.info("""
**Developer Info**

This Sweet Spot Oracle was created to optimize price feeds by dynamically 
identifying the most informative parts of the order book.

For implementation details or questions, contact the development team.
""")

# Debugging section (collapsible)
with st.expander("Debug Information"):
    st.subheader("WebSocket Connection Status")
    
    if st.session_state.connector:
        if st.session_state.run_status:
            st.success("WebSocket connected")
            
            # Show subscription info
            if hasattr(st.session_state.connector, 'subscriptions'):
                st.write("Active Subscriptions:")
                for symbol, sub_id in st.session_state.connector.subscriptions.items():
                    st.write(f"- {symbol}: ID {sub_id}")
        else:
            st.warning("WebSocket disconnected")
    else:
        st.info("WebSocket not initialized")
    
    # Manual reconnect button
    if st.button("Force Reconnect"):
        if st.session_state.run_status:
            stop_oracle()
            time.sleep(1)
            start_oracle(
                st.session_state.selected_pairs, 
                depths,
                update_interval, 
                sweet_spot_interval
            )
            st.success("Reconnection initiated")

# Add download button for order book data
if st.session_state.run_status and hasattr(st.session_state, 'price_history') and len(st.session_state.price_history) > 0:
    # Prepare download data
    download_data = {}
    for symbol in st.session_state.price_history:
        if not st.session_state.price_history[symbol]:
            continue
        df = pd.DataFrame(st.session_state.price_history[symbol])
        download_data[symbol] = df
    
    # Combine all data
    if download_data:
        all_data = pd.concat([df.assign(Symbol=symbol) for symbol, df in download_data.items()])
        
        # Create download button
        st.sidebar.download_button(
            label="Download Price History CSV",
            data=all_data.to_csv(index=False),
            file_name="sweet_spot_price_history.csv",
            mime="text/csv"
        )
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.sidebar.download_button(
            label="Download Full Analysis",
            data=json.dumps({
                "price_history": {k: v for k, v in st.session_state.price_history.items()},
                "sweet_spots": {k: v for k, v in st.session_state.sweet_spots.items()},
                "weights": {k: v for k, v in st.session_state.weights.items()}
            }, indent=2),
            file_name=f"sweet_spot_analysis_{timestamp}.json",
            mime="application/json"
        )