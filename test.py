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
from typing import Dict, List, Tuple, Any, Callable, Optional

# Use direct WebSocket connections instead of hyperliquid library
HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"

# Rest of the order book analysis code remains the same
# ...

# Replace the HyperliquidConnector class with a direct WebSocket implementation:
class DirectExchangeConnector:
    """Connects directly to exchange WebSocket API"""
    def __init__(self, ws_url=HYPERLIQUID_WS_URL):
        self.ws_url = ws_url
        self.oracle_master = None
        self.running = False
        self.callbacks = {}  # symbol -> callback
        self.current_prices = {}  # symbol -> price
        self.ws = None
        self.ws_thread = None
    
    def set_oracle_master(self, oracle_master):
        self.oracle_master = oracle_master
    
    def add_callback(self, symbol, callback):
        self.callbacks[symbol] = callback
    
    def start(self, symbols=None):
        if self.running:
            return
        
        self.symbols = symbols or ["BTC", "ETH"]
        self.running = True
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
    
    def _run_websocket(self):
        """Run the WebSocket connection"""
        def on_message(ws, message):
            try:
                msg = json.loads(message)
                self._handle_message(msg)
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
        
        def on_error(ws, error):
            logging.error(f"WebSocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            logging.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            if self.running:
                time.sleep(5)  # Wait before reconnecting
                self._run_websocket()  # Reconnect
        
        def on_open(ws):
            logging.info("WebSocket connection opened")
            # Subscribe to order books for all symbols
            for symbol in self.symbols:
                subscription = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "l2Book",
                        "coin": symbol
                    }
                }
                ws.send(json.dumps(subscription))
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run WebSocket in a loop (with reconnection)
        while self.running:
            try:
                self.ws.run_forever()
                if not self.running:
                    break
                time.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logging.error(f"WebSocket error: {str(e)}")
                time.sleep(5)
    
    def _handle_message(self, msg):
        """Process incoming WebSocket message"""
        try:
            if msg.get("channel") == "l2Book":
                data = msg.get("data", {})
                symbol = data.get("coin", "").upper()
                
                if symbol in self.symbols:
                    # Extract bid and ask data
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
                        self.oracle_master.update_order_book("direct", symbol, bids, asks)
        except Exception as e:
            logging.error(f"Error handling message: {str(e)}")
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        return self.current_prices.get(symbol, 0)