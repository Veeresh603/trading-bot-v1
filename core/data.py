# core/data.py
import pandas as pd
import yfinance as yf
import talib
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import logger
from config import Config

class HistoricalDataHandler:
    """
    Handles fetching, processing, and storing historical market data.
    This includes calculating technical indicators and scaling features.
    """
    def __init__(self, symbols, start_date, end_date, timeframe):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.symbol_data = {}  # Stores the final processed DataFrame for each symbol
        self.scaler = StandardScaler()
        self._fetch_and_prepare_data()

    def _fetch_and_prepare_data(self):
        """Orchestrator to fetch, process, and scale data for all symbols."""
        all_feature_dfs = []

        for symbol in self.symbols:
            logger.info(f"Fetching historical data for {symbol}...")
            try:
                df = yf.download(symbol, start=self.start_date, end=self.end_date, interval=self.timeframe, progress=False)
                if df.empty:
                    logger.warning(f"No data returned for {symbol} for the given period.")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                df.columns = [col.lower() for col in df.columns]

                logger.info(f"Calculating features for {symbol}...")
                features = self._calculate_features(df)
                
                combined_df = pd.concat([df, features], axis=1)
                combined_df.dropna(inplace=True) 
                
                # Check for feature columns before appending
                feature_cols = [col for col in features.columns if col in combined_df.columns]
                if not feature_cols:
                    logger.warning(f"No valid features generated for {symbol}. Skipping.")
                    continue

                all_feature_dfs.append(combined_df[feature_cols])
                self.symbol_data[symbol] = combined_df

            except Exception as e:
                logger.error(f"Failed to process data for {symbol}: {e}")
                import traceback
                traceback.print_exc()


        if not all_feature_dfs:
            logger.error("No data could be fetched or processed for any symbol. Exiting.")
            return

        logger.info("Fitting scaler on all available historical data...")
        full_feature_set = pd.concat(all_feature_dfs)
        feature_cols_to_scale = [col for col in full_feature_set.columns if col not in ['open', 'high', 'low', 'close', 'adj close', 'volume']]
        
        if not feature_cols_to_scale:
             logger.warning("No feature columns found to scale. Skipping scaling.")
             return
        
        # Replace inf/-inf with NaN, then drop rows with NaN in the feature set to be scaled
        full_feature_set.replace([np.inf, -np.inf], np.nan, inplace=True)
        full_feature_set.dropna(subset=feature_cols_to_scale, inplace=True)

        self.scaler.fit(full_feature_set[feature_cols_to_scale])

        logger.info("Applying scaler to each symbol's data...")
        for symbol, data in self.symbol_data.items():
            feature_cols = [col for col in feature_cols_to_scale if col in data.columns]
            if feature_cols:
                # Ensure data to be transformed is clean
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data.dropna(subset=feature_cols, inplace=True)
                if data.empty:
                    logger.warning(f"Data for {symbol} became empty after cleaning inf/nan. Skipping scaling.")
                    del self.symbol_data[symbol] # Remove symbol if it has no valid data
                    continue
                
                data[feature_cols] = self.scaler.transform(data[feature_cols])
                self.symbol_data[symbol] = data

    def _calculate_features(self, df):
        """Calculates all technical indicators and features."""
        features = pd.DataFrame(index=df.index)
        
        # --- THE FIX IS APPLIED HERE ---
        # Explicitly cast all data for TA-Lib to float64 to ensure compatibility.
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
        
        # Volatility Indicators
        features['ATR'] = talib.ATR(high, low, close)
        features['BB_upper'], features['BB_middle'], features['BB_lower'] = talib.BBANDS(close)
        
        # Volume Indicators
        features['OBV'] = talib.OBV(close, volume)
        
        # Trend Indicators
        features['SMA_20'] = talib.SMA(close, timeperiod=20)
        features['SMA_50'] = talib.SMA(close, timeperiod=50)
        features['EMA_20'] = talib.EMA(close, timeperiod=20)
        features['EMA_50'] = talib.EMA(close, timeperiod=50)

        # Price & Time based features
        features['price_change_1d'] = df['close'].pct_change(1)
        features['price_change_5d'] = df['close'].pct_change(5)
        features['day_of_week'] = df.index.dayofweek
        features['month_of_year'] = df.index.month

        # More standard indicators
        features['MOM'] = talib.MOM(close)
        features['STOCH_k'], features['STOCH_d'] = talib.STOCH(high, low, close)
        features['WILLR'] = talib.WILLR(high, low, close)
        features['TSF'] = talib.TSF(close)
        features['TRIX'] = talib.TRIX(close)
        
        # Pattern Recognition
        features['CDL2CROWS'] = talib.CDL2CROWS(open_p, high, low, close)
        features['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open_p, high, low, close)

        # Add remaining features to match INPUT_SIZE = 55
        features['SMA_100'] = talib.SMA(close, timeperiod=100)
        features['SMA_200'] = talib.SMA(close, timeperiod=200)
        features['EMA_100'] = talib.EMA(close, timeperiod=100)
        features['EMA_200'] = talib.EMA(close, timeperiod=200)
        features['RSI_10'] = talib.RSI(close, timeperiod=10)
        features['RSI_20'] = talib.RSI(close, timeperiod=20)
        features['ATR_10'] = talib.ATR(high, low, close, timeperiod=10)
        features['ATR_20'] = talib.ATR(high, low, close, timeperiod=20)
        features['CCI_10'] = talib.CCI(high, low, close, timeperiod=10)
        features['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)
        features['MACD_fast10_slow30'], _, _ = talib.MACD(close, fastperiod=10, slowperiod=30)
        features['MACD_fast15_slow40'], _, _ = talib.MACD(close, fastperiod=15, slowperiod=40)
        features['MOM_10'] = talib.MOM(close, timeperiod=10)
        features['MOM_20'] = talib.MOM(close, timeperiod=20)
        features['ADX_10'] = talib.ADX(high, low, close, timeperiod=10)
        features['ADX_20'] = talib.ADX(high, low, close, timeperiod=20)
        features['WILLR_10'] = talib.WILLR(high, low, close, timeperiod=10)
        features['WILLR_20'] = talib.WILLR(high, low, close, timeperiod=20)
        features['BB_upper_10'], _, features['BB_lower_10'] = talib.BBANDS(close, timeperiod=10)
        features['BB_upper_30'], _, features['BB_lower_30'] = talib.BBANDS(close, timeperiod=30)
        
        ema_50_safe = features['EMA_50'].replace(0, 1e-9)
        features['price_to_ema50_ratio'] = df['close'] / ema_50_safe
        
        features['volume_change_1d'] = df['volume'].pct_change(1)
        features['volume_change_5d'] = df['volume'].pct_change(5)
        
        final_features = features.copy()
        while len(final_features.columns) < 55:
            col_name = f'placeholder_{len(final_features.columns)}'
            final_features[col_name] = 0.0

        if len(final_features.columns) > 55:
            final_features = final_features.iloc[:, :55]

        return final_features

    def get_latest_bar(self, symbol):
        if symbol in self.symbol_data:
            return self.symbol_data[symbol].iloc[-right]
        return None

    def get_sequence(self, symbol, end_date):
        if symbol in self.symbol_data:
            data = self.symbol_data[symbol]
            end_idx = data.index.get_loc(end_date)
            start_idx = max(0, end_idx - Config.SEQUENCE_LENGTH + 1)
            return data.iloc[start_idx:end_idx + 1]
        return None

