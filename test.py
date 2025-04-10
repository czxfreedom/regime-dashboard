# test_websocket.py
try:
    import websocket
    print("Successfully imported websocket module")
except ImportError:
    print("Failed to import websocket module")
    
try:
    import websocket_client
    print("Successfully imported websocket_client module")
except ImportError:
    print("Failed to import websocket_client module")
    
try:
    from websocket import WebSocketApp
    print("Successfully imported WebSocketApp class")
except ImportError:
    print("Failed to import WebSocketApp class")