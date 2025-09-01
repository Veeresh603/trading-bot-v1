# train_offline.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys
from datetime import datetime, timedelta
import talib

from config import Config, BacktestConfig
from core.model import LSTMBrain
from core.utils import logger
from kiteconnect import KiteConnect, exceptions

# --- Data Fetching Functions ---

def get_index_token(kite_client: KiteConnect, tradingsymbol: str) -> int:
    """Gets the instrument token for the underlying index."""
    nse = pd.DataFrame(kite_client.instruments("NSE"))
    row = nse.loc[nse["tradingsymbol"] == tradingsymbol]
    if row.empty:
        raise RuntimeError(f"Index '{tradingsymbol}' not found in NSE instruments.")
    return int(row.iloc[0]["instrument_token"])

def fetch_candles(kite_client: KiteConnect, instrument_token: int,
                  start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Pulls historical candles in chunks to respect the API's 200-day limit.
    Returns a clean DataFrame with a tz-naive DatetimeIndex.
    """
    logger.info(f"Fetching historical data for token {instrument_token} from {start_date} to {end_date}...")
    
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    chunk_size = timedelta(days=199) # Stay safely within the 200-day limit
    all_chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + chunk_size, end)
        logger.info(f"  Fetching chunk from {current_start} to {current_end}")
        
        try:
            chunk = kite_client.historical_data(
                instrument_token,
                current_start,
                current_end,
                interval
            )
            all_chunks.extend(chunk)
            current_start += chunk_size + timedelta(days=1)
        except (exceptions.InputException, exceptions.TokenException) as e:
            logger.error(f"Kite API error while fetching data chunk: {e}")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred during data fetching: {e}")
            break
            
    if not all_chunks:
        raise RuntimeError("No candles returned. Check the date range, interval, and token.")

    df = pd.DataFrame(all_chunks)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Remove duplicate timestamps that can occur at chunk boundaries
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    return df


# --- Professional Feature Engineering ---

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical indicators and features.
    """
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
    
    logger.info(f"Generated {len(features.columns)} raw features.")
    
    # Ensure we have exactly INPUT_SIZE features
    final_features = features.copy()
    final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns that are all NaN first
    final_features.dropna(axis=1, how='all', inplace=True)
    
    # Then fill remaining NaNs
    final_features.fillna(0, inplace=True)

    while len(final_features.columns) < Config.INPUT_SIZE:
        col_name = f'placeholder_{len(final_features.columns)}'
        final_features[col_name] = 0.0

    if len(final_features.columns) > Config.INPUT_SIZE:
        final_features = final_features.iloc[:, :Config.INPUT_SIZE]
        
    logger.info(f"Final feature count matches INPUT_SIZE: {len(final_features.columns)}")
    
    return final_features

def build_training_frame(kite_client: KiteConnect):
    """Fetches candle data and applies feature engineering."""
    token = get_index_token(kite_client, Config.UNDERLYING_SYMBOL)
    
    df_raw = fetch_candles(kite_client, token, BacktestConfig.START_DATE, BacktestConfig.END_DATE, Config.HISTORICAL_DATA_TIMEFRAME)
    logger.info(f"Fetched {len(df_raw)} raw candles for {Config.UNDERLYING_SYMBOL}.")
    
    df_features = calculate_features(df_raw)
    
    df_combined = pd.concat([df_raw, df_features], axis=1)
    df_combined.dropna(inplace=True)
    
    return df_combined, df_features.columns.tolist()


# --- Training functions ---

def get_target_labels_triple_barrier(
    close_prices: pd.Series,
    profit_take_pct: float = 0.015,
    stop_loss_pct: float = 0.008,
    time_limit_bars: int = 10
) -> pd.Series:
    """Implements the Triple Barrier Method for labeling."""
    labels = pd.Series(2, index=close_prices.index)
    
    for i in range(len(close_prices) - time_limit_bars):
        entry_price = close_prices.iloc[i]
        profit_target = entry_price * (1 + profit_take_pct)
        stop_loss_target = entry_price * (1 - stop_loss_pct)

        for j in range(1, time_limit_bars + 1):
            future_price = close_prices.iloc[i + j]
            if future_price >= profit_target:
                labels.iloc[i] = 1
                break
            elif future_price <= stop_loss_target:
                labels.iloc[i] = 0
                break
    return labels

def create_sequences(input_data: pd.DataFrame, target_data: pd.Series, seq_length: int):
    """Creates sequences and corresponding labels for the LSTM."""
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data.iloc[i:(i + seq_length)].values
        y = target_data.iloc[i + seq_length]
        if y in [0, 1]:
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    logger.info("--- Starting Offline AI Model Training with Enhanced Features ---")

    kite_client = KiteConnect(api_key=Config.KITE_API_KEY)
    try:
        kite_client.set_access_token(Config.KITE_ACCESS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to set access token: {e}")
        sys.exit(1)

    try:
        training_df, feature_cols = build_training_frame(kite_client)
    except Exception as e:
        logger.error(f"Failed to build training data: {e}")
        sys.exit(1)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(training_df[feature_cols])
    features = pd.DataFrame(features_scaled, index=training_df.index, columns=feature_cols)

    # --- Labeling and Sequence Creation ---
    labels = get_target_labels_triple_barrier(training_df['close'])
    
    combined = pd.concat([features, labels.rename('target')], axis=1).dropna()
    features_final, labels_final = combined.drop('target', axis=1), combined['target']

    X, y = create_sequences(features_final, labels_final, Config.SEQUENCE_LENGTH)

    if len(X) == 0:
        logger.error("No training sequences could be generated after processing. Exiting.")
        return

    logger.info(f"Total training sequences created: {len(X)}")
    
    # --- FIX: Use stratified splitting to handle class imbalance ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Model Training ---
    model = LSTMBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor, y_train_tensor = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).long().to(device)
    X_val_tensor, y_val_tensor = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long().to(device)
    model.to(device)

    epochs, batch_size = 25, 64
    logger.info(f"Starting training for {epochs} epochs on device: {device.type.upper()}...")
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch, y_batch = X_train_tensor[i:i+batch_size], y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), Config.WEIGHTS_FILENAME)
    logger.info(f"✅ Model training complete. Weights saved to {Config.WEIGHTS_FILENAME}")

    try:
        model.eval()
        example_input = torch.randn(1, Config.SEQUENCE_LENGTH, len(feature_cols)).to(device)
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(Config.TORCHSCRIPT_FILENAME)
        logger.info(f"✅ Model compiled and saved to {Config.TORCHSCRIPT_FILENAME} for high-performance inference.")
    except Exception as e:
        logger.error(f"Could not compile model to TorchScript: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()