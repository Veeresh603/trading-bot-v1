# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

from config import Config
from core.model import LSTMBrain
from core.data_manager import DataManager
from core.utils import logger

def get_target_labels_triple_barrier(
    close_prices: pd.Series,
    profit_take_pct: float = 0.015,  # 1.5%
    stop_loss_pct: float = 0.008,   # 0.8%
    time_limit_bars: int = 10
) -> pd.Series:
    """
    Implements the Triple Barrier Method for labeling.
    A professional method that teaches the AI about risk-to-reward.

    - Profit Take: Upper barrier
    - Stop Loss: Lower barrier
    - Time Limit: Vertical barrier

    Returns:
        pd.Series: Labels (1 for Buy/Profit, 0 for Sell/Loss, 2 for Hold/Timeout)
    """
    labels = pd.Series(2, index=close_prices.index)  # Default to Hold (2)
    
    for i in range(len(close_prices) - time_limit_bars):
        entry_price = close_prices.iloc[i]
        profit_target = entry_price * (1 + profit_take_pct)
        stop_loss_target = entry_price * (1 - stop_loss_pct)

        for j in range(1, time_limit_bars + 1):
            future_price = close_prices.iloc[i + j]
            if future_price >= profit_target:
                labels.iloc[i] = 1  # Profit target hit first
                break
            elif future_price <= stop_loss_target:
                labels.iloc[i] = 0  # Stop loss hit first
                break
    return labels

def create_sequences(input_data: pd.DataFrame, target_data: pd.Series, seq_length: int):
    """Creates sequences and corresponding labels for the LSTM."""
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data.iloc[i:(i + seq_length)].values
        y = target_data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """Loads and prepares data from a single CSV file."""
    logger.info(f"Loading data from CSV: {filepath}")
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV {filepath}: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="AI Model Training Script")
    parser.add_argument("--csv", type=str, help="Path to a single CSV file for training.")
    args = parser.parse_args()

    data_manager = DataManager(Config.SYMBOLS_TO_TRADE)
    all_features, all_labels = [], []

    if args.csv:
        logger.info(f"--- Starting AI Model Training from CSV: {args.csv} ---")
        df_raw = load_data_from_csv(args.csv)
        if df_raw.empty: return
        
        symbol_name = os.path.basename(args.csv).split('.')[0]
        feature_df = data_manager._calculate_features(df_raw)
        data_manager.data_cache[symbol_name] = pd.concat([df_raw, feature_df], axis=1)
        
        # Fit scaler on this single CSV's data
        feature_df_cleaned = feature_df.dropna()
        data_manager.scaler.fit(feature_df_cleaned)
        
        symbols_to_process = [symbol_name]

    else:
        logger.info("--- Starting AI Model Training using yfinance data ---")
        data_manager.fetch_and_prepare_historical_data()
        symbols_to_process = Config.SYMBOLS_TO_TRADE

    for symbol in symbols_to_process:
        processed_data = data_manager.get_scaled_data(symbol)
        if processed_data is None: continue
        
        feature_columns = [col for col in processed_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        features = processed_data[feature_columns]
        labels = get_target_labels_triple_barrier(processed_data['Close'])

        combined = pd.concat([features, labels.rename('target')], axis=1).dropna()
        features, labels = combined.drop('target', axis=1), combined['target']

        X, y = create_sequences(features, labels, Config.SEQUENCE_LENGTH)
        all_features.append(X)
        all_labels.append(y)

    if not all_features:
        logger.error("No data available for training. Exiting.")
        return

    X_all, y_all = np.concatenate(all_features), np.concatenate(all_labels)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    model = LSTMBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor, y_train_tensor = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    X_val_tensor, y_val_tensor = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()

    epochs, batch_size = 25, 64
    logger.info(f"Starting training for {epochs} epochs...")
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
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), Config.WEIGHTS_FILENAME)
    logger.info(f"✅ Model training complete. Weights saved to {Config.WEIGHTS_FILENAME}")

    try:
        model.eval()
        example_input = torch.randn(1, Config.SEQUENCE_LENGTH, Config.INPUT_SIZE)
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(Config.TORCHSCRIPT_FILENAME)
        logger.info(f"✅ Model compiled and saved to {Config.TORCHSCRIPT_FILENAME} for C++ engine.")
    except Exception as e:
        logger.error(f"Could not compile model to TorchScript: {e}")

if __name__ == "__main__":
    main()