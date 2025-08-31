# core/data_manager.py
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable

from config import Config
from .utils import logger

class DataManager:
    """
    Handles fetching, processing, and managing both historical and live market data.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.scaler = StandardScaler()

    def fetch_and_prepare_historical_data(self):
        """Fetches historical data, calculates features, and fits the scaler."""
        logger.info("Fetching and preparing historical data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.HISTORICAL_DATA_YEARS * 365)

        all_features_df = pd.DataFrame()

        for symbol in self.symbols:
            try:
                # Use a stock symbol format yfinance understands, e.g., 'RELIANCE.NS'
                yf_symbol = f"{symbol.split('-')[0]}.NS"
                df = yf.download(yf_symbol, start=start_date, end=end_date, interval=Config.HISTORICAL_DATA_TIMEFRAME.replace('minute','m'))
                if df.empty:
                    raise ValueError("No data returned from yfinance")

                features = self._calculate_features(df)
                self.data_cache[symbol] = pd.concat([df, features], axis=1)
                all_features_df = pd.concat([all_features_df, features])
                logger.info(f"Loaded {len(df)} historical records for {symbol}")
            except Exception as e:
                logger.error(f"Could not fetch data for {symbol}: {e}")

        if not all_features_df.empty:
            all_features_df.dropna(inplace=True)
            self.scaler.fit(all_features_df)
            logger.info("✅ StandardScaler fitted on all historical data.")

    def get_scaled_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Returns the fully processed and scaled data for training or backtesting."""
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol].copy()
        features_df = df.iloc[:, 6:] # Assuming first 6 cols are OHLCV, Adj Close
        features_df.dropna(inplace=True)

        scaled_features = self.scaler.transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, index=features_df.index, columns=features_df.columns)

        # Re-combine with the original OHLCV data
        result = pd.concat([df.iloc[:, :6], scaled_features_df], axis=1)
        result.dropna(inplace=True)
        return result

    @staticmethod
    def _calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a wide range of technical indicators."""
        features = pd.DataFrame(index=df.index)
        close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

        # Momentum Indicators
        features['RSI'] = talib.RSI(close)
        features['MACD'], features['MACD_SIGNAL'], _ = talib.MACD(close)
        features['ADX'] = talib.ADX(high, low, close)
        features['CCI'] = talib.CCI(high, low, close)
        
        # Volatility Indicators
        features['ATR'] = talib.ATR(high, low, close)
        bb_upper, _, bb_lower = talib.BBANDS(close)
        features['BB_WIDTH'] = (bb_upper - bb_lower) / close
        
        # Trend Indicators
        features['EMA_20'] = talib.EMA(close, timeperiod=20)
        features['SMA_50'] = talib.SMA(close, timeperiod=50)

        # Fill dummy features to meet the required input size
        # In a real system, these would be meaningful features.
        num_existing_features = len(features.columns)
        for i in range(Config.INPUT_SIZE - num_existing_features):
             features[f'DUMMY_{i}'] = 0.0

        return features.iloc[:, :Config.INPUT_SIZE]

    # --- Live Data Handling (WebSocket) ---
    def setup_websockets(self, smart_api_instance, on_tick_callback: Callable):
        """Sets up the WebSocket connection for live data."""
        logger.info("Setting up WebSocket for live data...")
        # This part is highly specific to the broker's library (SmartAPI)
        # You would typically have methods to handle connection, disconnection,
        # and the arrival of new ticks. This is a conceptual implementation.
        # token_list = [{"exchangeType": 1, "tokens": list(Config.INSTRUMENT_TOKENS.values())}]
        # smart_api_instance.subscribe(token_list)
        # smart_api_instance.on_ticks = on_tick_callback
        # smart_api_instance.connect()
        logger.warning("Live WebSocket setup is conceptual. You need to implement the actual connection logic.")