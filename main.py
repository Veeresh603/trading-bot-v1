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
import os
import talib
from kiteconnect import KiteConnect, WebSocket
from config import Config, BacktestConfig, get_current_options_contracts
from core.options_data import OptionsDataHandler
from core.strategy import AIStrategy
from core.utils import logger, telegram

class KiteBrokerManager:
    """
    Handles all communication with the live trading broker (Zerodha Kite).
    This version is adapted for options trading.
    """
    def __init__(self):
        self.api_client = KiteConnect(api_key=Config.KITE_API_KEY)
        self.websocket = None
        self.access_token = Config.KITE_ACCESS_TOKEN
        self.on_tick_callback = None

    def login(self) -> bool:
        """Sets the access token for the KiteConnect API."""
        try:
            self.api_client.set_access_token(self.access_token)
            logger.info("✅ KiteConnect access token set successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to set KiteConnect access token: {e}")
            return False

    def setup_websockets(self, on_tick_callback, contracts):
        """Initializes and connects the WebSocket for live data."""
        self.on_tick_callback = on_tick_callback
        
        self.websocket = WebSocket(Config.KITE_API_KEY, self.access_token)
        self.websocket.on_ticks = self._on_ticks
        self.websocket.on_connect = self._on_connect
        
        logger.info("Connecting to WebSocket for live options data...")
        self.websocket.connect(threaded=True)
        
        instrument_tokens = [c['instrument_token'] for c in contracts]
        self.websocket.subscribe(instrument_tokens)
        logger.info(f"Subscribed to {len(instrument_tokens)} instrument tokens.")

    def _on_ticks(self, ws, ticks):
        if self.on_tick_callback:
            self.on_tick_callback(ws, ticks)
    
    def _on_connect(self, ws, response):
        logger.info("✅ Live Data Feed Connected via WebSocket.")
        telegram.send_message("✅ *Live Data Feed Connected*")
            
    def close_connection(self):
        if self.websocket:
            self.websocket.close()

class LiveDataHandler:
    """
    Handles incoming live tick data for options contracts.
    """
    def __init__(self, contracts, timeframe_minutes, scaler, on_bar_callback):
        self.contracts = contracts
        self.timeframe = timedelta(minutes=timeframe_minutes)
        self.scaler = scaler
        self.on_bar_callback = on_bar_callback
        self.tick_buffers = {c['trading_symbol']: [] for c in self.contracts}
        self.last_bar_timestamp = None
        self.last_tick_time = datetime.now()
        self.last_prices = {c['trading_symbol']: None for c in self.contracts}
        
        self.tick_file_paths = {c['trading_symbol']: os.path.join(Config.LOCAL_DATA_DIR, f"{c['trading_symbol']}_ticks.csv") for c in self.contracts}
        self._setup_tick_files()
        
        self.token_to_symbol_map = {c['instrument_token']: c['trading_symbol'] for c in self.contracts}

    def _setup_tick_files(self):
        """Sets up CSV files for writing raw tick data."""
        os.makedirs(Config.LOCAL_DATA_DIR, exist_ok=True)
        for symbol, filepath in self.tick_file_paths.items():
            if not os.path.exists(filepath):
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'symbol', 'price', 'volume'])

    def on_new_tick(self, ws, ticks):
        """Processes a single incoming tick and persists it."""
        self.last_tick_time = datetime.now()
        for tick in ticks:
            token = tick.get('instrument_token')
            symbol = self.token_to_symbol_map.get(token)
            if symbol is None:
                continue

            price = tick.get('last_price')
            volume = tick.get('last_quantity') 
            timestamp = datetime.now()
            
            if price is None or volume is None:
                continue
            
            self._persist_to_local_file(timestamp, symbol, price, volume)

            self.last_prices[symbol] = price
            now = datetime.now()
            
            if self.last_bar_timestamp is None:
                self.last_bar_timestamp = self._get_bar_start_time(now)
            
            if now >= self.last_bar_timestamp + self.timeframe:
                for s in self.token_to_symbol_map.values():
                    self._finalize_and_process_bar(s)
                self.last_bar_timestamp = self._get_bar_start_time(now)
                for s in self.token_to_symbol_map.values():
                    self.tick_buffers[s] = []

            self.tick_buffers[symbol].append({'price': price, 'volume': volume})

    def _persist_to_local_file(self, timestamp, symbol, price, volume):
        """Appends a new tick to the local CSV file."""
        filepath = self.tick_file_paths[symbol]
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, symbol, price, volume])

    def _finalize_and_process_bar(self, symbol):
        """
        Creates a final OHLCV bar from the tick buffer and processes it.
        """
        ticks = self.tick_buffers[symbol]
        
        if not ticks and self.last_prices[symbol] is not None:
            last_price = self.last_prices[symbol]
            bar = {'open': last_price, 'high': last_price, 'low': last_price, 'close': last_price, 'volume': 0}
            logger.warning(f"Market gap detected for {symbol}. Creating placeholder bar with last price {last_price}.")
        elif not ticks:
            return 
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
        features = self._calculate_features(temp_df)
        
        scaled_features = self.scaler.transform(features).flatten()
        
        logger.info(f"New 15m Bar for {symbol} | Close: {bar['close']:.2f}")
        self.on_bar_callback(symbol, bar, scaled_features)

    def _get_bar_start_time(self, dt):
        """Calculates the start time of the current bar."""
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)

    def _calculate_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values.astype('float64')
        features['RSI'] = talib.RSI(close, timeperiod=14)
        features['MACD'], _, _ = talib.MACD(close)
        features['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features.fillna(0, inplace=True)
        return features

class TradingBot:
    """
    The main orchestrator for the live/paper options trading bot.
    """
    def __init__(self, mode: str):
        self.mode = mode
        self.is_live = mode == 'live'
        self.shutdown_event = threading.Event()
        
        logger.info(f"--- Initializing Options Trading Bot in {self.mode.upper()} mode ---")
        telegram.send_message(f"🚀 *Options Bot is starting up in {self.mode.upper()} mode...*")
        
        self.broker = KiteBrokerManager()
        
        kite_client = self.broker.api_client
        contracts = get_current_options_contracts(kite_client)

        if not contracts:
            logger.error("Could not fetch contracts to trade. Exiting.")
            sys.exit(1)
            
        hist_data = OptionsDataHandler(
            kite_client=kite_client,
            contracts=contracts,
            start_date=BacktestConfig.START_DATE,
            end_date=BacktestConfig.END_DATE,
            timeframe=Config.HISTORICAL_DATA_TIMEFRAME
        )
        self.scaler = hist_data.scaler

        self.data_handler = LiveDataHandler(
            contracts=contracts,
            timeframe_minutes=15,
            scaler=self.scaler,
            on_bar_callback=self.on_new_bar
        )
        
        self.strategy = AIStrategy(
            data_handler=hist_data,
            contracts=contracts,
            sequence_length=Config.SEQUENCE_LENGTH
        )
        self.strategy.data_sequences = {c['trading_symbol']: deque(maxlen=Config.SEQUENCE_LENGTH) for c in contracts}
        
        self.current_positions = {c['trading_symbol']: 0 for c in contracts}

    def initialize(self) -> bool:
        """Connects to the broker."""
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
        
        if self.is_live:
            if action == 1:
                telegram.send_message(f"🚨 **LIVE ALERT (PAPER TRADE)**: BUY signal for {symbol} @ {price:.2f} with {confidence:.2f} confidence.")
            elif action == 0:
                telegram.send_message(f"🚨 **LIVE ALERT (PAPER TRADE)**: SELL signal for {symbol} @ {price:.2f} with {confidence:.2f} confidence.")
            return

        position = self.current_positions.get(symbol, 0)
        
        direction = "NONE"
        if action == 1 and confidence >= Config.MIN_CONFIDENCE_TO_TRADE:
            direction = "BUY"
        elif action == 0 and position > 0:
            direction = "SELL"
        
        logger.info(f"Signal for {symbol}: {direction} with {confidence:.2f} confidence.")

        if direction == 'BUY' and position == 0:
            logger.info(f"Signal to OPEN LONG for {symbol}")
            quantity = 10
            self._log_paper_trade(symbol, quantity, direction, price)
            self.current_positions[symbol] = quantity

        elif direction == 'SELL' and position > 0:
            logger.info(f"Signal to CLOSE LONG for {symbol}")
            quantity = self.current_positions[symbol]
            self._log_paper_trade(symbol, quantity, direction, price)
            self.current_positions[symbol] = 0
            
    def _setup_paper_trading_log(self):
        with open("paper_trades.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Symbol", "Direction", "Price", "Quantity"])

    def _log_paper_trade(self, symbol, quantity, direction, price):
        telegram.send_message(f"📝 *PAPER TRADE*: {direction} {quantity} {symbol} @ {price:.2f}")
        with open("paper_trades.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), symbol, direction, price, quantity])

    def run(self):
        """Main run loop of the bot."""
        if not self.initialize():
            self.shutdown()
            return

        contracts = get_current_options_contracts(self.broker.api_client)
        if not contracts:
            logger.error("Could not fetch contracts to trade in live mode. Exiting.")
            self.shutdown()
            return
            
        self.data_handler.contracts = contracts
        self.data_handler.token_to_symbol_map = {c['instrument_token']: c['trading_symbol'] for c in contracts}

        if self.is_live:
            self.broker.setup_websockets(on_tick_callback=self.data_handler.on_new_tick, contracts=contracts)
        
        try:
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
        logger.info("--- Shutting Down Options Bot ---")
        telegram.send_message("👋 *Options Bot is shutting down...*")
        self.shutdown_event.set()
        if self.is_live:
            self.broker.close_connection()
        logger.info("--- Bot Shutdown Complete ---")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot - Live/Paper Options Trading Engine")
    parser.add_argument("--mode", type=str, choices=['paper', 'live'], default='paper', 
                        help="Set the trading mode: 'paper' for simulated trading or 'live' for real trading.")
    args = parser.parse_args()
    
    bot = TradingBot(mode=args.mode)
    bot.run()