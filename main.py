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

try:
    from smartapi import SmartConnect, WebSocket
except ImportError:
    print("Please install smartapi-python: pip install smartapi-python")
    sys.exit(1)

from config import Config
from core.strategy import AIStrategy
from core.data import HistoricalDataHandler
from core.utils import logger, telegram
from core.execution import BacktestExecutionHandler

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
        self.access_token = None

    def login(self) -> bool:
        """Logs into the broker using credentials from the config."""
        try:
            logger.info("Logging into broker...")
            data = self.api_client.generateSession(Config.CLIENT_ID, Config.PASSWORD, Config.TOTP)
            if data['status'] and data['data']['jwtToken']:
                logger.info("✅ Broker login successful.")
                self.api_client.setAccessToken(data['data']['jwtToken'])
                self.access_token = data['data']['jwtToken']
                return True
            else:
                logger.error(f"Broker login failed: {data['message']}")
                return False
        except Exception as e:
            logger.error(f"An exception occurred during broker login: {e}")
            return False

    def place_order(self, order_params):
        """Places an order with the broker."""
        try:
            order_id = self.api_client.placeOrder(order_params)
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            telegram.send_message(f"Order for {order_params['tradingsymbol']} placed successfully with ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
            
    def get_order_status(self, order_id):
        """Checks the status of a specific order."""
        try:
            order_history = self.api_client.getOrderHistory()
            for order in order_history:
                if order['orderid'] == order_id:
                    return order['status']
            return "NOT_FOUND"
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return "ERROR"

    def setup_websockets(self, on_tick_callback, on_connect_callback):
        """Initializes and connects the WebSocket for live data."""
        self.on_tick_callback = on_tick_callback
        self.on_connect_callback = on_connect_callback
        
        auth_token = self.access_token
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
        
        tokens_to_subscribe = [str(token) for token in Config.INSTRUMENT_TOKENS.values()]
        self.websocket.subscribe(tokens_to_subscribe)
        logger.info(f"Subscribed to {len(tokens_to_subscribe)} instrument tokens.")
            
    def close_connection(self):
        if self.websocket:
            self.websocket.close()

class LiveExecutionHandler:
    """
    Handles the entire lifecycle of an order in a live trading environment,
    including placing orders, checking status, and managing fills.
    """
    def __init__(self, broker: LiveBrokerManager):
        self.broker = broker
        self.pending_orders = {}
        self.filled_orders = []
        self.order_monitor_thread = None
        self.stop_monitoring = threading.Event()

    def place_order(self, order_data: dict):
        """
        Places a new order with the broker and adds it to the pending order book.
        """
        symbol = order_data['symbol']
        direction = order_data['direction']
        quantity = order_data['quantity']
        order_type = order_data.get('order_type', 'MARKET')
        price = order_data.get('price', 0)
        
        logger.info(f"Placing new {order_type} order for {direction} {quantity} of {symbol} at price {price}")
        
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": Config.INSTRUMENT_TOKENS.get(symbol),
            "transactiontype": "BUY" if direction == "BUY" else "SELL",
            "exchange": "NSE" if ".NS" in symbol else "BSE",
            "ordertype": order_type,
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": price,
            "quantity": quantity
        }
        
        broker_order_id = self.broker.place_order(order_params)
        
        if broker_order_id:
            self.pending_orders[broker_order_id] = {
                'id': broker_order_id,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'status': 'PENDING'
            }
            logger.info(f"Order {broker_order_id} added to pending list.")

    def start_order_monitor(self):
        """Starts a background thread to monitor pending orders."""
        self.order_monitor_thread = threading.Thread(target=self._monitor_orders, daemon=True)
        self.order_monitor_thread.start()
        logger.info("Order monitoring thread started.")
    
    def _monitor_orders(self):
        """Continuously checks the status of pending orders."""
        while not self.stop_monitoring.is_set():
            if not self.pending_orders:
                time.sleep(1)
                continue
                
            orders_to_remove = []
            for order_id, order in self.pending_orders.items():
                status = self.broker.get_order_status(order_id)
                if status == "COMPLETE":
                    logger.info(f"Order {order_id} for {order['symbol']} is COMPLETE.")
                    self.filled_orders.append(order)
                    orders_to_remove.append(order_id)
                elif status in ["REJECTED", "CANCELLED"]:
                    logger.warning(f"Order {order_id} for {order['symbol']} was {status}.")
                    orders_to_remove.append(order_id)
            
            for order_id in orders_to_remove:
                del self.pending_orders[order_id]
                
            time.sleep(10)

    def stop_monitor(self):
        self.stop_monitoring.set()

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
        self.last_bar_timestamp = None
        self.last_tick_time = datetime.now()
        self.last_prices = {symbol: None for symbol in symbols}

    def on_new_tick(self, ticks):
        """Processes a single incoming tick."""
        self.last_tick_time = datetime.now()
        for tick in ticks:
            symbol = self._map_token_to_symbol(tick.get('instrument_token'))
            if symbol not in self.symbols:
                continue

            price = tick.get('last_price')
            volume = tick.get('volume_traded')
            
            if price is None or volume is None:
                continue
            
            self.last_prices[symbol] = price
            now = datetime.now()
            
            if self.last_bar_timestamp is None:
                self.last_bar_timestamp = self._get_bar_start_time(now)
            
            if now >= self.last_bar_timestamp + self.timeframe:
                for s in self.symbols:
                    self._finalize_and_process_bar(s)
                self.last_bar_timestamp = self._get_bar_start_time(now)
                for s in self.symbols:
                    self.tick_buffers[s] = []

            self.tick_buffers[symbol].append({'price': price, 'volume': volume})

    def _finalize_and_process_bar(self, symbol):
        """
        Creates a final OHLCV bar from the tick buffer and processes it.
        Handles market gaps by using the last known price.
        """
        ticks = self.tick_buffers[symbol]
        
        if not ticks and self.last_prices[symbol] is not None:
            # Handle market gaps: no ticks received for this symbol in this bar
            last_price = self.last_prices[symbol]
            bar = {'open': last_price, 'high': last_price, 'low': last_price, 'close': last_price, 'volume': 0}
            logger.warning(f"Market gap detected for {symbol}. Creating placeholder bar with last price {last_price}.")
        elif not ticks:
            return # No data available at all for this symbol
        else:
            df_ticks = pd.DataFrame(ticks)
            bar = {
                'open': df_ticks['price'].iloc[0],
                'high': df_ticks['price'].max(),
                'low': df_ticks['price'].min(),
                'close': df_ticks['price'].iloc[-1],
                'volume': df_ticks['volume'].sum()
            }

        temp_df = pd.DataFrame([bar])
        features = HistoricalDataHandler._calculate_features(self, temp_df)
        
        feature_cols_to_scale = [col for col in features.columns if col in self.scaler.feature_names_in_]
        
        scaled_features = self.scaler.transform(features[feature_cols_to_scale]).flatten()
        
        logger.info(f"New 15m Bar for {symbol} | Close: {bar['close']:.2f}")
        self.on_bar_callback(symbol, bar, scaled_features)

    def _get_bar_start_time(self, dt):
        """Calculates the start time of the current bar."""
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)

    def _map_token_to_symbol(self, token):
        token_map = {v: k for k, v in Config.INSTRUMENT_TOKENS.items()}
        return token_map.get(str(token))

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
        
        self.broker = LiveBrokerManager()
        
        hist_data = HistoricalDataHandler(
            symbols=Config.SYMBOLS_TO_TRADE,
            start_date=BacktestConfig.START_DATE,
            end_date=BacktestConfig.END_DATE,
            timeframe=Config.HISTORICAL_DATA_TIMEFRAME
        )
        self.scaler = hist_data.scaler

        self.data_handler = LiveDataHandler(
            symbols=Config.SYMBOLS_TO_TRADE,
            timeframe_minutes=15,
            scaler=self.scaler,
            on_bar_callback=self.on_new_bar
        )
        self.strategy = AIStrategy(data_handler=hist_data, sequence_length=Config.SEQUENCE_LENGTH)
        self.strategy.data_sequences = {s: deque(maxlen=Config.SEQUENCE_LENGTH) for s in Config.SYMBOLS_TO_TRADE}

        self.current_positions = {symbol: 0 for symbol in Config.SYMBOLS_TO_TRADE}
        if not self.is_live:
            self._setup_paper_trading_log()
        else:
            self.live_execution_handler = LiveExecutionHandler(self.broker)

    def _get_fitted_scaler(self):
        """
        Initializes a HistoricalDataHandler just to get a scaler
        fitted on historical data. This ensures live data is scaled
        identically to training/backtesting data.
        """
        logger.info("Fitting scaler on historical data...")
        hist_data = HistoricalDataHandler(Config.SYMBOLS_TO_TRADE, "2020-01-01", "2024-01-01", Config.HISTORICAL_DATA_TIMEFRAME)
        logger.info("✅ Scaler ready.")
        return hist_data.scaler
        
    def initialize(self) -> bool:
        """Connects to the broker."""
        if not self.is_live:
            return True
        return self.broker.login()

    def on_new_bar(self, symbol, bar, scaled_features):
        """Callback triggered by LiveDataHandler when a new bar is ready."""
        self.strategy.data_sequences[symbol].append(scaled_features)
        
        if len(self.strategy.data_sequences[symbol]) < Config.SEQUENCE_LENGTH:
            return

        sequence = np.array(self.strategy.data_sequences[symbol])
        action, confidence = self.strategy.model.decide_action(sequence)
        
        self._process_signal(symbol, action, confidence, bar['close'])

    def _process_signal(self, symbol, action, confidence, price):
        """Acts on a trading signal from the strategy."""
        position = self.current_positions[symbol]
        
        direction = "NONE"
        if action == 1 and confidence >= Config.MIN_CONFIDENCE_TO_TRADE:
            direction = "LONG"
        elif action == 0 and position > 0:
            direction = "EXIT"

        logger.info(f"Signal for {symbol}: {direction} with {confidence:.2f} confidence.")

        if direction == 'LONG' and position == 0:
            logger.info(f"Signal to OPEN LONG for {symbol}")
            quantity = 10
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
            self.live_execution_handler.place_order({
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'order_type': 'MARKET',
                'price': price
            })
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

        if self.is_live:
            self.broker.setup_websockets(
                on_tick_callback=self.data_handler.on_new_tick,
                on_connect_callback=lambda resp: telegram.send_message("✅ *Live Data Feed Connected*")
            )
            self.live_execution_handler.start_order_monitor()
        
        try:
            # Watchdog timer to monitor data feed reliability
            WATCHDOG_TIMEOUT = 120 # Seconds
            while not self.shutdown_event.is_set():
                if self.is_live and (datetime.now() - self.data_handler.last_tick_time).total_seconds() > WATCHDOG_TIMEOUT:
                    logger.error("🚨 Watchdog timer expired! No ticks received. Shutting down to prevent bad trades.")
                    telegram.send_message("🚨 Watchdog timer expired. No ticks received for 2 minutes. Shutting down!")
                    break
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
        if self.is_live:
            self.live_execution_handler.stop_monitor()
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