# core/options_data.py
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from kiteconnect import KiteConnect, exceptions
from config import Config
from .utils import logger
import os
from datetime import datetime, timedelta

class OptionsDataHandler:
    def __init__(self, kite_client: KiteConnect, contracts: list, start_date: str, end_date: str, timeframe: str):
        self.kite = kite_client
        self.contracts = contracts
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data = {}
        self.scaler = StandardScaler()
        if self.contracts:
            self._fetch_and_prepare_data()

    def _fetch_and_prepare_data(self):
        """Fetches and prepares options data for all contracts."""
        all_features_dfs = []
        os.makedirs(Config.LOCAL_DATA_DIR, exist_ok=True)

        for contract in self.contracts:
            # --- FIX: Ensure the item is a dictionary before proceeding ---
            if not isinstance(contract, dict):
                logger.warning(f"Skipping invalid contract object: {contract}")
                continue

            token = contract.get('instrument_token')
            symbol = contract.get('trading_symbol')

            if token is None or symbol is None:
                logger.warning(f"Skipping contract with missing token or symbol: {contract}")
                continue
                
            filepath = os.path.join(Config.LOCAL_DATA_DIR, f"{symbol}_{self.timeframe}.csv")
            df = pd.DataFrame()

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    df = df.loc[self.start_date:self.end_date]
                except Exception as e:
                    logger.error(f"Failed to load local data for {symbol}: {e}")
                    df = pd.DataFrame()

            if df.empty:
                logger.info(f"Fetching historical data for {symbol} (Token: {token})...")
                try:
                    start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
                    end = datetime.strptime(self.end_date, '%Y-%m-%d').date()
                    
                    chunk_size = timedelta(days=200)
                    current_start = start
                    all_chunks = []

                    while current_start <= end:
                        current_end = min(current_start + chunk_size, end)
                        logger.info(f"  Fetching chunk from {current_start} to {current_end}")
                        
                        chunk = self.kite.historical_data(
                            token,
                            current_start,
                            current_end,
                            self.timeframe
                        )
                        all_chunks.extend(chunk)
                        current_start += chunk_size + timedelta(days=1)
                    
                    df = pd.DataFrame(all_chunks)
                    if df.empty:
                        logger.warning(f"No data returned for {symbol} for the given period.")
                        continue
                    
                    df.set_index('date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    df.columns = [col.lower() for col in df.columns]

                    logger.info(f"Saving data for {symbol} to '{filepath}'...")
                    df.to_csv(filepath)
                except (exceptions.InputException, exceptions.TokenException) as e:
                    logger.error(f"Failed to fetch data from Kite for {symbol}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"An unexpected error occurred while fetching data for {symbol}: {e}")
                    continue

            if not df.empty:
                logger.info(f"Calculating features for {symbol}...")
                features = self._calculate_features(df)
                
                combined_df = pd.concat([df, features], axis=1)
                combined_df.dropna(inplace=True)
                
                all_features_dfs.append(combined_df)
                self.data[symbol] = combined_df

        if not all_features_dfs:
            logger.error("No data could be fetched or processed for any contract. Exiting.")
            return

        logger.info("Fitting scaler on all available historical data...")
        full_feature_set = pd.concat(all_features_dfs)
        feature_cols_to_scale = [col for col in full_feature_set.columns if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'date']]
        
        if not feature_cols_to_scale:
             logger.warning("No feature columns found to scale. Skipping scaling.")
             return
        
        full_feature_set.replace([np.inf, -np.inf], np.nan, inplace=True)
        full_feature_set.dropna(subset=feature_cols_to_scale, inplace=True)

        self.scaler.fit(full_feature_set[feature_cols_to_scale])

        logger.info("Applying scaler to each contract's data...")
        for symbol, data in self.data.items():
            feature_cols = [col for col in feature_cols_to_scale if col in data.columns]
            if feature_cols:
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data.dropna(subset=feature_cols, inplace=True)
                if data.empty:
                    logger.warning(f"Data for {symbol} became empty after cleaning inf/nan. Skipping scaling.")
                    del self.data[symbol]
                    continue
                
                data[feature_cols] = self.scaler.transform(data[feature_cols])
                self.data[symbol] = data

    def _calculate_features(self, df):
        """Calculates features for options data using technical indicators."""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values.astype('float64')

        features['RSI'] = talib.RSI(close, timeperiod=14)
        features['MACD'], _, _ = talib.MACD(close)
        features['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        features['price_change'] = df['close'].pct_change()
        features['volume_change'] = df['volume'].pct_change()

        features.fillna(0, inplace=True)

        final_features = features.copy()
        while len(final_features.columns) < Config.INPUT_SIZE:
            col_name = f'placeholder_{len(final_features.columns)}'
            final_features[col_name] = 0.0

        if len(final_features.columns) > Config.INPUT_SIZE:
            final_features = final_features.iloc[:, :Config.INPUT_SIZE]

        return final_features

    def get_latest_bar(self, symbol):
        if symbol in self.data:
            return self.data[symbol].iloc[-1]
        return None

    def get_sequence(self, symbol, end_date):
        if symbol in self.data:
            data = self.data[symbol]
            end_idx = data.index.get_loc(end_date)
            start_idx = max(0, end_idx - Config.SEQUENCE_LENGTH + 1)
            return data.iloc[start_idx:end_idx + 1]
        return None
    
    def get_atr(self, symbol, current_time):
        if symbol in self.data and current_time in self.data[symbol].index:
            return self.data[symbol].loc[current_time, 'ATR']
        return None