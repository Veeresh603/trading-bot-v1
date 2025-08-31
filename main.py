# main.py
import time
import threading
import sys
import argparse
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import csv

# --- Broker-Specific Imports ---
# This example assumes usage of the smartapi-python library for Angel One
# You may need to adjust imports based on your chosen broker library
try:
    from smartapi import SmartConnect, WebSocket
except ImportError:
    print("Please install smartapi-python: pip install smartapi-python")
    sys.exit(1)

from config import Config
from core.strategy import AIStrategy
from core.data import HistoricalDataHandler # We use this to get the scaler
from core.utils import logger, telegram

class LiveBrokerManager:
    """
    Handles all communication with the live trading broker.
    This is a concrete implementation for Angel One.
    """
    def __init__(self):
        self.api_client = SmartConnect(api_key=Config.API_KEY)
        self.websocket = None
        self.on_tick_callback = None
        self.on_connect_callback = None

    def login(self) -> bool:
        """Logs into the broker using credentials from the config."""
        try:
            logger.info("Logging into broker...")
            data = self.api_client.generateSession(Config.CLIENT_ID, Config.PASSWORD, Config.TOTP)
            if data['status'] and data['data']['jwtToken']:
                logger.info("✅ Broker login successful.")
                return True
            else:
                logger.error(f"Broker login failed: {data['message']}")
                return False
        except Exception as e:
            logger.error(f"An exception occurred during broker login: {e}")
            return False

    def place_order(self, symbol, quantity, direction):
        """Places an order with the broker."""
        # This is a simplified example. A real implementation needs more details.
        # (e.g., product type, order type, price, etc.)
        # You must map your internal symbol format to the broker's format.
        logger.info(f"PLACING LIVE ORDER: {direction} {quantity} of {symbol}")
        telegram.send_message(f"🚀 *LIVE ORDER*: {direction} {quantity} {symbol}")
        # Add actual order placement logic here using self.api_client.placeOrder(...)

    def setup_websockets(self, on_tick_callback, on_connect_callback):
        """Initializes and connects the WebSocket for live data."""
        self.on_tick_callback = on_tick_callback
        self.on_connect_callback = on_connect_callback
        
        auth_token = self.api_client.access_token
        api_key = Config.API_KEY
        client_code = Config.CLIENT_ID
        
        self.websocket = WebSocket(auth_token, api_key, client_code)
        self.websocket.on_ticks = self._on_ticks
        self.websocket.on_connect = self._on_connect
        
        logger.info("Connecting to WebSocket for live data...")
        self.websocket.connect()

    def _on_ticks(self, ws, ticks):
        if self.on_tick_callback:
            self.on_tick_callback(ticks)
    
    def _on_connect(self, ws, response):
        if self.on_connect_callback:
            self.on_connect_callback(response)
            
    def close_connection(self):
        if self.websocket:
            self.websocket.close()


class LiveDataHandler:
    """
    Handles incoming live tick data, aggregates it into bars,
    calculates features, and scales them.
    """
    def __init__(self, symbols, timeframe_minutes, scaler, on_bar_callback):
        self.symbols = symbols
        self.timeframe = timedelta(minutes=timeframe_minutes)
        self.scaler = scaler
        self.on_bar_callback = on_bar_callback
        
        self.tick_buffers = {symbol: [] for symbol in symbols}
        self.current_bars = {symbol: None for symbol in symbols}
        self.last_bar_timestamp = None

    def on_new_tick(self, tick):
        """Processes a single incoming tick."""
        # This function needs to be adapted to your broker's tick format
        # and symbol/token mapping. This is a conceptual example.
        symbol = self._map_token_to_symbol(tick.get('instrument_token'))
        if symbol not in self.symbols:
            return

        price = tick.get('last_price')
        volume = tick.get('volume_traded')
        
        if price is None or volume is None:
            return
            
        now = datetime.now()
        
        # Check if it's time to form a new bar
        if self.last_bar_timestamp is None or now >= self.last_bar_timestamp + self.timeframe:
            # First, process the completed previous bar
            if self.last_bar_timestamp is not None:
                for s in self.symbols:
                    if self.tick_buffers[s]:
                        self._finalize_and_process_bar(s)

            # Then, start a new bar
            self.last_bar_timestamp = self._get_bar_start_time(now)
            for s in self.symbols:
                self.tick_buffers[s] = []

        # Append tick to the current bar's buffer
        self.tick_buffers[symbol].append({'price': price, 'volume': volume})

    def _finalize_and_process_bar(self, symbol):
        """Creates a final OHLCV bar from the tick buffer and processes it."""
        ticks = self.tick_buffers[symbol]
        if not ticks: return

        df_ticks = pd.DataFrame(ticks)
        
        bar = {
            'open': df_ticks['price'].iloc[0],
            'high': df_ticks['price'].max(),
            'low': df_ticks['price'].min(),
            'close': df_ticks['price'].iloc[-1],
            'volume': df_ticks['volume'].sum()
        }

        # Use the same feature calculation as the backtester for consistency
        features = HistoricalDataHandler._calculate_features(self, pd.DataFrame([bar]))
        
        # Scale features using the pre-fitted scaler
        scaled_features = self.scaler.transform(features).flatten()
        
        logger.info(f"New 15m Bar for {symbol} | Close: {bar['close']:.2f}")
        self.on_bar_callback(symbol, bar, scaled_features)

    def _get_bar_start_time(self, dt):
        """Calculates the start time of the current bar."""
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)

    def _map_token_to_symbol(self, token):
        # Create a reverse mapping from token to symbol
        token_map = {v: k for k, v in Config.INSTRUMENT_TOKENS.items()}
        return token_map.get(str(token)) # Ensure token is a string for lookup

class TradingBot:
    """
    The main orchestrator for the live/paper trading bot.
    """
    def __init__(self, mode: str):
        self.mode = mode
        self.is_live = mode == 'live'
        self.shutdown_event = threading.Event()
        
        logger.info(f"--- Initializing Trading Bot in {self.mode.upper()} mode ---")
        telegram.send_message(f"🚀 *Bot is starting up in {self.mode.upper()} mode...*")
        
        # Initialize core components
        self.broker = LiveBrokerManager()
        self.scaler = self._get_fitted_scaler()
        self.data_handler = LiveDataHandler(
            symbols=Config.SYMBOLS_TO_TRADE,
            timeframe_minutes=15,
            scaler=self.scaler,
            on_bar_callback=self.on_new_bar
        )
        self.strategy = AIStrategy(data_handler=None, sequence_length=Config.SEQUENCE_LENGTH) # data_handler not needed for live
        self.strategy.data_sequences = {s: deque(maxlen=Config.SEQUENCE_LENGTH) for s in Config.SYMBOLS_TO_TRADE}

        self.current_positions = {symbol: 0 for symbol in Config.SYMBOLS_TO_TRADE}
        if not self.is_live:
            self._setup_paper_trading_log()

    def _get_fitted_scaler(self):
        """
        Initializes a HistoricalDataHandler just to get a scaler
        fitted on historical data. This ensures live data is scaled
        identically to training/backtesting data.
        """
        logger.info("Fitting scaler on historical data...")
        # A bit inefficient, but ensures consistency.
        # A better approach would be to save the fitted scaler during training.
        hist_data = HistoricalDataHandler(Config.SYMBOLS_TO_TRADE, "2020-01-01", "2024-01-01", Config.HISTORICAL_DATA_TIMEFRAME)
        logger.info("✅ Scaler ready.")
        return hist_data.scaler
        
    def initialize(self) -> bool:
        """Connects to the broker."""
        return self.broker.login()

    def on_new_bar(self, symbol, bar, scaled_features):
        """Callback triggered by LiveDataHandler when a new bar is ready."""
        # 1. Append new data to the strategy's sequence
        self.strategy.data_sequences[symbol].append(scaled_features)
        
        # 2. Check if we have enough data to generate a signal
        if len(self.strategy.data_sequences[symbol]) < Config.SEQUENCE_LENGTH:
            return

        # 3. Generate signals using the AI strategy
        sequence = np.array(self.strategy.data_sequences[symbol])
        action, confidence = self.strategy.model.decide_action(sequence)
        
        # 4. Process the signal
        self._process_signal(symbol, action, confidence, bar['close'])

    def _process_signal(self, symbol, action, confidence, price):
        """Acts on a trading signal from the strategy."""
        position = self.current_positions[symbol]
        
        # Map model output to signal direction: 1=Buy, 0=Sell/Close
        direction = "NONE"
        if action == 1 and confidence >= 0.75: # Hardcoded confidence threshold
            direction = "LONG"
        elif action == 0:
            direction = "EXIT"

        logger.info(f"Signal for {symbol}: {direction} with {confidence:.2f} confidence.")

        # Execution Logic
        if direction == 'LONG' and position == 0:
            logger.info(f"Signal to OPEN LONG for {symbol}")
            quantity = 10 # Simplified position sizing. Replace with a real function.
            self.execute_trade(symbol, quantity, "BUY", price)
            self.current_positions[symbol] = quantity

        elif direction == 'EXIT' and position > 0:
            logger.info(f"Signal to CLOSE LONG for {symbol}")
            quantity = self.current_positions[symbol]
            self.execute_trade(symbol, quantity, "SELL", price)
            self.current_positions[symbol] = 0

    def execute_trade(self, symbol, quantity, direction, price):
        """Executes a trade in either live or paper mode."""
        if self.is_live:
            self.broker.place_order(symbol, quantity, direction)
        else:
            self._log_paper_trade(symbol, direction, price, quantity)
            
    def _setup_paper_trading_log(self):
        with open("paper_trades.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Symbol", "Direction", "Price", "Quantity"])

    def _log_paper_trade(self, symbol, direction, price, quantity):
        telegram.send_message(f"📝 *PAPER TRADE*: {direction} {quantity} {symbol} @ {price:.2f}")
        with open("paper_trades.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), symbol, direction, price, quantity])

    def run(self):
        """Main run loop of the bot."""
        if not self.initialize():
            self.shutdown()
            return

        self.broker.setup_websockets(
            on_tick_callback=lambda ticks: [self.data_handler.on_new_tick(t) for t in ticks],
            on_connect_callback=lambda resp: telegram.send_message("✅ *Live Data Feed Connected*")
        )
        
        # Subscribe to instrument tokens
        tokens_to_subscribe = list(Config.INSTRUMENT_TOKENS.values())
        # Add subscription logic here via the broker's websocket object if required by the library

        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shuts down the bot."""
        logger.info("--- Shutting Down Bot ---")
        telegram.send_message("👋 *Bot is shutting down...*")
        self.shutdown_event.set()
        self.broker.close_connection()
        logger.info("--- Bot Shutdown Complete ---")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot - Live/Paper Trading Engine")
    parser.add_argument("--mode", type=str, choices=['paper', 'live'], default='paper', 
                        help="Set the trading mode: 'paper' for simulated trading or 'live' for real trading.")
    args = parser.parse_args()
    
    bot = TradingBot(mode=args.mode)
    bot.run()
