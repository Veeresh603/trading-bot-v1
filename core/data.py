# core/data.py
import pandas as pd
import yfinance as yf
import talib
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import logger
from config import Config
import traceback
import os
from datetime import datetime, timedelta

try:
    from kiteconnect import KiteConnect, exceptions
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("KiteConnect library not found. Yahoo Finance will be the only data source.")


class HistoricalDataHandler:
    """
    Handles fetching, processing, and storing historical market data.
    It prioritizes Zerodha Kite and uses yfinance as a fallback.
    """
    def __init__(self, symbols, start_date, end_date, timeframe):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.symbol_data = {}
        self.scaler = StandardScaler()
        self.kite = None
        self._setup_kite_client()
        self._fetch_and_prepare_data()

    def _setup_kite_client(self):
        """Initializes the KiteConnect client if credentials are provided."""
        if KITE_AVAILABLE and Config.KITE_API_KEY and Config.KITE_ACCESS_TOKEN:
            try:
                self.kite = KiteConnect(api_key=Config.KITE_API_KEY)
                self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                logger.info("✅ KiteConnect client initialized for historical data.")
            except Exception as e:
                logger.error(f"Failed to initialize Kite client: {e}")
                self.kite = None
        else:
            logger.warning("KiteConnect credentials not configured. Will rely solely on yfinance.")

    def _get_instrument_token(self, symbol):
        """Gets the instrument token for a given symbol from the NSE segment."""
        try:
            instruments = self.kite.instruments("NSE")
            df = pd.DataFrame(instruments)
            token_row = df[df['tradingsymbol'] == symbol]
            if not token_row.empty:
                return token_row.iloc[0]['instrument_token']
        except Exception as e:
            logger.error(f"Failed to get instrument token for {symbol}: {e}")
        return None

    def _fetch_data_kite(self, symbol):
        """Fetches historical data for a symbol using KiteConnect in chunks."""
        token = self._get_instrument_token(symbol)
        if not token:
            logger.warning(f"Could not find instrument token for {symbol}. Cannot fetch from Kite.")
            return None
        
        try:
            logger.info(f"Fetching historical data for {symbol} (Token: {token}) from Kite...")
            start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
            end = datetime.strptime(self.end_date, '%Y-%m-%d').date()
            
            chunk_size = timedelta(days=199)
            all_chunks = []
            current_start = start

            while current_start <= end:
                current_end = min(current_start + chunk_size, end)
                logger.info(f"  Fetching chunk from {current_start} to {current_end}")
                chunk = self.kite.historical_data(token, current_start, current_end, self.timeframe)
                all_chunks.extend(chunk)
                current_start += chunk_size + timedelta(days=1)
            
            if not all_chunks:
                return None

            df = pd.DataFrame(all_chunks)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.columns = [col.lower() for col in df.columns]
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data from Kite for {symbol}: {e}")
            return None

    def _fetch_data_yfinance(self, symbol):
        """Fetches historical data using yfinance as a fallback."""
        logger.info(f"Fetching historical data for {symbol} using yfinance as fallback...")
        try:
            yf_symbol = '^NSEI' if symbol == 'NIFTY 50' else symbol
            yf_interval = self.timeframe.replace('minute', 'm')

            df = yf.download(yf_symbol, start=self.start_date, end=self.end_date, interval=yf_interval, progress=False, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} ({yf_symbol}) from yfinance.")
                return None
            
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data from yfinance for {symbol}: {e}")
            return None

    def _fetch_and_prepare_data(self):
        """Orchestrator to fetch, process, and scale data for all symbols."""
        all_feature_dfs = []
        os.makedirs(Config.LOCAL_DATA_DIR, exist_ok=True)

        for symbol in self.symbols:
            filename_symbol = symbol.replace(' ', '').replace(':', '')
            filepath = os.path.join(Config.LOCAL_DATA_DIR, f"{filename_symbol}_{self.timeframe}.csv")
            df = pd.DataFrame()

            if os.path.exists(filepath):
                logger.info(f"Loading data for {symbol} from local file: {filepath}")
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                except Exception:
                    df = pd.DataFrame()
            
            if df.empty:
                if self.kite:
                    df = self._fetch_data_kite(symbol)
                
                if df is None or df.empty:
                    df = self._fetch_data_yfinance(symbol)

                if df is None or df.empty:
                    logger.error(f"Failed to fetch data for {symbol} from all sources.")
                    continue
                
                logger.info(f"Saving fetched data for {symbol} to '{filepath}'...")
                df.to_csv(filepath)

            df = df.loc[self.start_date:self.end_date]
            if df.empty:
                logger.warning(f"No data for {symbol} within the specified date range.")
                continue

            logger.info(f"Calculating features for {symbol}...")
            features = self._calculate_features(df)
            combined_df = pd.concat([df, features], axis=1)
            
            combined_df.dropna(inplace=True)
            
            if combined_df.empty:
                logger.warning(f"DataFrame for {symbol} is empty after feature calculation. Skipping.")
                continue
            
            feature_cols = [col for col in features.columns if col in combined_df.columns]
            all_feature_dfs.append(combined_df[feature_cols])
            self.symbol_data[symbol] = combined_df

        if not all_feature_dfs:
            logger.error("No data could be processed for any symbol.")
            return

        logger.info("Fitting scaler on all available historical data...")
        full_feature_set = pd.concat(all_feature_dfs)
        feature_cols_to_scale = [col for col in full_feature_set.columns if col not in ['open', 'high', 'low', 'close', 'adj close', 'volume']]
        
        if feature_cols_to_scale:
            self.scaler.fit(full_feature_set[feature_cols_to_scale])

            for symbol, data in self.symbol_data.items():
                scalable_cols = [col for col in feature_cols_to_scale if col in data.columns]
                if scalable_cols:
                    data.loc[:, scalable_cols] = self.scaler.transform(data[scalable_cols])
                    self.symbol_data[symbol] = data

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a comprehensive set of technical indicators and features."""
        features = pd.DataFrame(index=df.index)
        
        open_p = df['open'].values.astype('float64')
        close = df['close'].values.astype('float64')
        high = df['high'].values.astype('float64')
        low = df['low'].values.astype('float64')
        volume = df['volume'].values.astype('float64')

        # Momentum Indicators
        features['RSI'] = talib.RSI(close)
        features['MACD'], features['MACD_signal'], features['MACD_hist'] = talib.MACD(close)
        features['ADX'] = talib.ADX(high, low, close)
        features['CCI'] = talib.CCI(high, low, close)
        features['MOM'] = talib.MOM(close)
        features['STOCH_k'], features['STOCH_d'] = talib.STOCH(high, low, close)
        features['WILLR'] = talib.WILLR(high, low, close)
        features['TRIX'] = talib.TRIX(close)

        # Volatility Indicators
        features['ATR'] = talib.ATR(high, low, close)
        features['BB_upper'], features['BB_middle'], features['BB_lower'] = talib.BBANDS(close)
        
        # Volume Indicator
        features['OBV'] = talib.OBV(close, volume)
        
        # Trend Indicators
        features['SMA_20'] = talib.SMA(close, timeperiod=20)
        features['EMA_20'] = talib.EMA(close, timeperiod=20)
        features['SMA_50'] = talib.SMA(close, timeperiod=50)
        features['EMA_50'] = talib.EMA(close, timeperiod=50)

        # Price & Volume Change Features
        features['price_change_1d'] = df['close'].pct_change(1)
        features['volume_change_1d'] = df['volume'].pct_change(1)
        
        # Time-based Features
        features['day_of_week'] = df.index.dayofweek
        features['month_of_year'] = df.index.month

        # Candlestick Pattern Recognition (Examples)
        features['CDL2CROWS'] = talib.CDL2CROWS(open_p, high, low, close)
        features['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open_p, high, low, close)
        
        final_features = features.copy()
        final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_features.dropna(axis=1, how='all', inplace=True)
        final_features.fillna(0, inplace=True)

        while len(final_features.columns) < Config.INPUT_SIZE:
            col_name = f'placeholder_{len(final_features.columns)}'
            final_features[col_name] = 0.0

        if len(final_features.columns) > Config.INPUT_SIZE:
            final_features = final_features.iloc[:, :Config.INPUT_SIZE]
            
        return final_features

    def get_atr(self, symbol, current_time):
        """Returns the ATR value for a given symbol at a specific time."""
        if symbol in self.symbol_data and current_time in self.symbol_data[symbol].index:
            return self.symbol_data[symbol].loc[current_time, 'ATR']
        return None