# main.py
import time
import threading
import sys
import argparse
import csv
from datetime import datetime
from collections import deque
import numpy as np
import torch

from config import Config
from core.broker import BrokerManager
from core.data_manager import DataManager
from core.model import LSTMBrain
from core.utils import logger, telegram

class TradingBot:
    def __init__(self, mode: str):
        self.mode = mode
        if self.mode == 'live':
            Config.PAPER_TRADING = False
        else:
            Config.PAPER_TRADING = True

        self.shutdown_event = threading.Event()
        self.broker = BrokerManager()
        self.data_manager = DataManager(Config.SYMBOLS_TO_TRADE)
        self.model = LSTMBrain()

        # State management for live data aggregation
        self.tick_buffers = {token: [] for token in Config.INSTRUMENT_TOKENS.values()}
        self.live_data_sequences = {
            symbol: deque(maxlen=Config.SEQUENCE_LENGTH)
            for symbol in Config.SYMBOLS_TO_TRADE
        }
        self.current_positions = {} # symbol -> { 'quantity': int, 'entry_price': float }

    def initialize(self) -> bool:
        """Initializes all components of the bot."""
        logger.info(f"--- Initializing Trading Bot in {self.mode.upper()} mode ---")
        telegram.send_message(f"🚀 *Bot is starting up in {self.mode.upper()} mode...*")

        if not self.broker.login():
            return False

        self.data_manager.fetch_and_prepare_historical_data()

        try:
            self.model.load_state_dict(torch.load(Config.WEIGHTS_FILENAME))
            self.model.eval()
            logger.info("✅ AI model weights loaded successfully.")
        except FileNotFoundError:
            logger.critical(f"AI model weights '{Config.WEIGHTS_FILENAME}' not found. Please train the model first.")
            return False
        
        self.setup_paper_trading_log()
        return True

    def on_ticks(self, ws, ticks):
        """Callback for WebSocket tick data."""
        for tick in ticks:
            token = tick['instrument_token']
            self.tick_buffers[token].append(tick)
            
            # Aggregate ticks into 1-minute bars
            now = datetime.now()
            if now.second == 0:
                self.process_bar(token)

    def on_connect(self, ws, response):
        """Callback for successful WebSocket connection."""
        logger.info("✅ WebSocket connected.")
        telegram.send_message("✅ *Live Data Feed Connected*")
        tokens_to_subscribe = list(Config.INSTRUMENT_TOKENS.values())
        ws.subscribe(tokens_to_subscribe)
        ws.set_mode(ws.MODE_FULL, tokens_to_subscribe)

    def process_bar(self, token):
        """Processes an aggregated bar of tick data."""
        ticks = self.tick_buffers[token]
        if not ticks: return
        
        df_ticks = pd.DataFrame(ticks)
        symbol = next(s for s, t in Config.INSTRUMENT_TOKENS.items() if t == token)
        
        # Create OHLC bar
        bar = {
            'Open': df_ticks['last_price'].iloc[0],
            'High': df_ticks['last_price'].max(),
            'Low': df_ticks['last_price'].min(),
            'Close': df_ticks['last_price'].iloc[-1],
            'Volume': df_ticks['volume_traded'].sum()
        }
        self.tick_buffers[token] = [] # Clear buffer for next bar

        # Calculate features for the new bar
        features_df = self.data_manager._calculate_features(pd.DataFrame([bar]))
        scaled_features = self.data_manager.scaler.transform(features_df).flatten()
        
        self.live_data_sequences[symbol].append(scaled_features)
        
        if len(self.live_data_sequences[symbol]) == Config.SEQUENCE_LENGTH:
            self.make_trading_decision(symbol, bar['Close'])

    def make_trading_decision(self, symbol, current_price):
        """Uses the AI to make a trade decision and executes it."""
        sequence = np.array(self.live_data_sequences[symbol])
        action, confidence = self.model.decide_action(sequence)

        if confidence < Config.MIN_CONFIDENCE_TO_TRADE: return

        if symbol not in self.current_positions and action == 1: # Buy Signal
            position_size = 1 # Simplified position sizing
            logger.info(f"AI signal to OPEN LONG for {symbol} with confidence {confidence:.2f}")
            
            if Config.PAPER_TRADING:
                self.log_paper_trade(symbol, "BUY", current_price, position_size)
            else:
                self.broker.place_order(symbol, position_size, "LONG")
            self.current_positions[symbol] = {'quantity': position_size, 'entry_price': current_price}

        elif symbol in self.current_positions and action == 0: # Sell Signal
            position_size = self.current_positions[symbol]['quantity']
            logger.info(f"AI signal to CLOSE LONG for {symbol} with confidence {confidence:.2f}")
            
            if Config.PAPER_TRADING:
                self.log_paper_trade(symbol, "SELL", current_price, position_size)
            else:
                self.broker.place_order(symbol, position_size, "SELL")
            del self.current_positions[symbol]

    def setup_paper_trading_log(self):
        if Config.PAPER_TRADING:
            with open("paper_trades.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Symbol", "Direction", "Price", "Quantity"])

    def log_paper_trade(self, symbol, direction, price, quantity):
        """Logs a paper trade to a CSV file."""
        telegram.send_message(f"📝 *PAPER TRADE*: {direction} {quantity} {symbol} @ {price}")
        with open("paper_trades.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), symbol, direction, price, quantity])

    def run(self):
        """Main run loop of the bot."""
        if not self.initialize():
            self.shutdown()
            return

        self.broker.setup_websockets(on_tick_callback=self.on_ticks, on_connect_callback=self.on_connect)

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
        if self.broker.kws:
            self.broker.kws.close()
        logger.info("--- Bot Shutdown Complete ---")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--mode", type=str, choices=['paper', 'live'], default='paper', help="Set the trading mode: 'paper' or 'live'")
    args = parser.parse_args()
    
    bot = TradingBot(mode=args.mode)
    bot.run()