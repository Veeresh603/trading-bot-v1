# main.py
import time
import threading
import sys
import argparse
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import asyncio
import os
import csv
import talib
import numpy as np

from kiteconnect import KiteConnect, WebSocket
from config import Config, BacktestConfig, get_current_options_contracts
from core.infrastructure.dependency_injection import get_container, register_services
from core.infrastructure.event_bus import IEventBus, Event, EventType
from core.domain.services.market_data_service import MarketDataService
from core.utils import logger, telegram

class LiveDataHandler:
    """
    Handles incoming live tick data for options contracts, creates bars,
    and publishes them to the event bus. This is the bridge between the WebSocket
    and the domain services.
    """
    def __init__(self, contracts: list, timeframe_minutes: int, scaler, event_bus: IEventBus):
        self.contracts = contracts
        self.timeframe = timedelta(minutes=timeframe_minutes)
        self.scaler = scaler
        self.event_bus = event_bus
        
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
        """Processes incoming ticks, forms bars, and publishes events."""
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
            
            # Check if it's time to form a new bar
            if now >= self.last_bar_timestamp + self.timeframe:
                for s in self.token_to_symbol_map.values():
                    self._finalize_and_publish_bar(s, self.last_bar_timestamp)
                self.last_bar_timestamp = self._get_bar_start_time(now)
                for s in self.token_to_symbol_map.values():
                    self.tick_buffers[s] = []

            self.tick_buffers[symbol].append({'price': price, 'volume': volume})

    def _finalize_and_publish_bar(self, symbol: str, bar_timestamp: datetime):
        """Creates a final OHLCV bar and publishes a MARKET_DATA event."""
        ticks = self.tick_buffers[symbol]
        
        if not ticks and self.last_prices[symbol] is not None:
            last_price = self.last_prices[symbol]
            bar = {'open': last_price, 'high': last_price, 'low': last_price, 'close': last_price, 'volume': 0}
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

        # Publish the raw bar data. The MarketDataService will handle feature calculation.
        market_data_event = Event(
            event_type=EventType.MARKET_DATA,
            timestamp=bar_timestamp,
            payload={'symbol': symbol, 'bar': bar},
            correlation_id=f"market_data_{bar_timestamp.timestamp()}",
            source="LiveDataHandler"
        )
        # Use asyncio to run the async publish method from a synchronous context
        asyncio.run(self.event_bus.publish(market_data_event))
        logger.info(f"Published 15m Bar for {symbol} | Close: {bar['close']:.2f}")

    def _persist_to_local_file(self, timestamp, symbol, price, volume):
        """Appends a new tick to the local CSV file."""
        filepath = self.tick_file_paths[symbol]
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, symbol, price, volume])

    def _get_bar_start_time(self, dt: datetime) -> datetime:
        """Calculates the start time of the current bar."""
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)

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

        self.container = get_container()
        register_services(self.container)

        self.event_bus = self.container.resolve(IEventBus)
        self.market_data_service = self.container.resolve(MarketDataService)
        self.broker = KiteBrokerManager()
        self.live_data_handler = None

    def initialize(self) -> bool:
        """Connects to the broker."""
        return self.broker.login()

    def run(self):
        """Main run loop of the bot."""
        if not self.initialize():
            self.shutdown()
            return

        contracts_dict = get_current_options_contracts(self.broker.api_client)
        if not contracts_dict or not contracts_dict.get('all'):
            logger.error("Could not fetch contracts to trade in live mode. Exiting.")
            self.shutdown()
            return
        
        contracts = contracts_dict['all']

        # Start the event bus processor
        if hasattr(self.event_bus, 'start'):
            asyncio.run(self.event_bus.start())

        # Initialize the MarketDataService and LiveDataHandler
        self.market_data_service.initialize(contracts, self.broker.api_client)
        self.live_data_handler = LiveDataHandler(
            contracts=contracts,
            timeframe_minutes=15,
            scaler=self.market_data_service.scaler,
            event_bus=self.event_bus
        )

        if self.is_live:
            self.broker.setup_websockets(on_tick_callback=self.live_data_handler.on_new_tick, contracts=contracts)
        
        try:
            WATCHDOG_TIMEOUT = 120 # Seconds
            while not self.shutdown_event.is_set():
                if self.is_live and (datetime.now() - self.live_data_handler.last_tick_time).total_seconds() > WATCHDOG_TIMEOUT:
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
        if hasattr(self.event_bus, 'stop'):
            asyncio.run(self.event_bus.stop())
        logger.info("--- Bot Shutdown Complete ---")
        sys.exit(0)

class KiteBrokerManager:
    """Handles all communication with the live trading broker (Zerodha Kite)."""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot - Live/Paper Options Trading Engine")
    parser.add_argument("--mode", type=str, choices=['paper', 'live'], default='paper', 
                        help="Set the trading mode: 'paper' for simulated trading or 'live' for real trading.")
    args = parser.parse_args()
    
    bot = TradingBot(mode=args.mode)
    bot.run()