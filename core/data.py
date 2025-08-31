# core/data.py
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import talib
from config import Config
from core.utils import logger

class HistoricalDataHandler:
    """
    Handles fetching, processing, and providing historical market data
    for backtesting.
    """
    def __init__(self, symbols, start_date, end_date, timeframe):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.scaler = StandardScaler()
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        
        self._fetch_and_prepare_data()
        self.data_iterator = self._create_combined_data_iterator()
    
    def _fetch_and_prepare_data(self):
        """Fetches data, calculates features, scales them, and stores it."""
        logger.info("Fetching and preparing historical data...")
        combined_data = {}
        all_features_df = pd.DataFrame()

        for symbol in self.symbols:
            # Download historical data
            df = yf.download(symbol, start=self.start_date, end=self.end_date, interval=self.timeframe)
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            
            # Calculate features
            features = self._calculate_features(df)
            
            # Combine raw data and features
            combined = pd.concat([df, features], axis=1).dropna()
            combined_data[symbol] = combined
            
            # Append features to a master DataFrame for fitting the scaler
            all_features_df = pd.concat([all_features_df, combined[features.columns]])

        # Fit the scaler on all features from all symbols together
        self.scaler.fit(all_features_df)
        
        # Scale the features for each symbol
        for symbol in self.symbols:
            feature_cols = [col for col in combined_data[symbol].columns if col not in ['open', 'high', 'low', 'close', 'volume', 'Adj Close']]
            scaled_features = self.scaler.transform(combined_data[symbol][feature_cols])
            scaled_features_df = pd.DataFrame(scaled_features, index=combined_data[symbol].index, columns=feature_cols)
            
            # Replace original features with scaled ones
            self.symbol_data[symbol] = pd.concat([combined_data[symbol][['open', 'high', 'low', 'close', 'volume']], scaled_features_df], axis=1)
        
        logger.info("✅ Historical data preparation complete.")

    def _calculate_features(self, df):
        """Calculates a wide range of technical indicators."""
        features = pd.DataFrame(index=df.index)
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume'].astype(float)
        
        # Momentum Indicators
        features['RSI'] = talib.RSI(close)
        features['MACD'], features['MACD_signal'], features['MACD_hist'] = talib.MACD(close)
        features['ADX'] = talib.ADX(high, low, close)
        features['CCI'] = talib.CCI(high, low, close)
        
        # Volatility Indicators
        features['ATR'] = talib.ATR(high, low, close)
        bollinger_upper, _, bollinger_lower = talib.BBANDS(close)
        features['BB_width'] = (bollinger_upper - bollinger_lower) / close
        
        # Volume Indicators
        features['OBV'] = talib.OBV(close, volume)
        
        # Trend Indicators
        features['SMA_20'] = talib.SMA(close, timeperiod=20)
        features['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # Normalize price-based features
        features['SMA_20'] = (close - features['SMA_20']) / close
        features['EMA_50'] = (close - features['EMA_50']) / close

        return features.dropna()

    def _create_combined_data_iterator(self):
        """Creates an iterator that yields data for each timestamp."""
        # Align all data to a common index, forward-filling missing values
        combined_df = pd.DataFrame()
        for symbol, data in self.symbol_data.items():
            renamed_data = data.add_prefix(f"{symbol}_")
            if combined_df.empty:
                combined_df = renamed_data
            else:
                combined_df = combined_df.join(renamed_data, how='outer')
        
        combined_df.ffill(inplace=True)
        combined_df.dropna(inplace=True) # Drop any remaining NaNs (usually at the start)
        return combined_df.iterrows()

    def update_bars(self):
        """Advances the data by one bar for all symbols."""
        try:
            _, latest_data_row = next(self.data_iterator)
            for symbol in self.symbols:
                # Extract data for the current symbol from the combined row
                symbol_prefix = f"{symbol}_"
                symbol_cols = {col.replace(symbol_prefix, ''): val for col, val in latest_data_row.items() if col.startswith(symbol_prefix)}
                if symbol_cols:
                    self.latest_symbol_data[symbol] = pd.Series(symbol_cols)
        except StopIteration:
            self.continue_backtest = False

    def get_latest_bar_value(self, symbol, val_type):
        """Returns the specified value (e.g., 'close') for the latest bar."""
        return self.latest_symbol_data.get(symbol, {}).get(val_type.lower())

    def get_latest_scaled_features(self, symbol):
        """Returns the scaled features for the latest bar as a numpy array."""
        data = self.latest_symbol_data.get(symbol)
        if data is None: return None
        
        feature_cols = [col for col in data.index if col not in ['open', 'high', 'low', 'close', 'volume']]
        return data[feature_cols].values

    def get_latest_bar_datetime(self):
        """Returns the timestamp of the latest bar."""
        # All symbols share the same index in the iterator
        if self.latest_symbol_data:
            return next(iter(self.latest_symbol_data.values())).name
        return None
