# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

from config import Config, BacktestConfig, get_current_options_contracts
from core.model import LSTMBrain
from core.options_data import OptionsDataHandler
from core.utils import logger
from kiteconnect import KiteConnect

def get_target_labels_triple_barrier(
    close_prices: pd.Series,
    profit_take_pct: float = 0.015,
    stop_loss_pct: float = 0.008,
    time_limit_bars: int = 10
) -> pd.Series:
    """
    Implements the Triple Barrier Method for labeling.
    """
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
    logger.info("--- Starting Options AI Model Training ---")

    kite_client = KiteConnect(api_key=Config.KITE_API_KEY)
    try:
        kite_client.set_access_token(Config.KITE_ACCESS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to set access token during training setup: {e}")
        logger.error("Please run login_kite.py to get a new access token and update your .env file.")
        sys.exit(1)

    # --- FIX: Dynamically fetch contracts and check if the list is empty ---
    contracts = get_current_options_contracts(kite_client)
    if not contracts:
        logger.error("Could not fetch contracts to backtest. Exiting.")
        sys.exit(1)

    data_handler = OptionsDataHandler(
        kite_client=kite_client,
        contracts=contracts,
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )

    all_features, all_labels = [], []

    # --- FIX: Iterate over the dynamically fetched contracts list ---
    for contract in contracts:
        symbol = contract['trading_symbol']
        processed_data = data_handler.data.get(symbol)
        if processed_data is None or processed_data.empty:
            logger.warning(f"No data found for {symbol}, skipping training for this contract.")
            continue
        
        logger.info(f"Processing data for {symbol}...")
        
        feature_cols_for_training = [col for col in data_handler.scaler.feature_names_in_ if col in processed_data.columns]

        if not feature_cols_for_training:
             logger.warning(f"No valid features found for {symbol} to train on. Skipping.")
             continue
        
        features = processed_data[feature_cols_for_training]
        
        labels = get_target_labels_triple_barrier(processed_data['close'])

        combined = pd.concat([features, labels.rename('target')], axis=1).dropna()
        features, labels = combined.drop('target', axis=1), combined['target']

        X, y = create_sequences(features, labels, Config.SEQUENCE_LENGTH)
        
        if len(X) > 0:
            all_features.append(X)
            all_labels.append(y)
        else:
            logger.warning(f"Not enough valid sequences generated for {symbol}.")

    if not all_features:
        logger.error("No data available for training after processing. Exiting.")
        return

    X_all, y_all = np.concatenate(all_features), np.concatenate(all_labels)
    logger.info(f"Total training sequences created: {len(X_all)}")

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

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
        example_input = torch.randn(1, Config.SEQUENCE_LENGTH, Config.INPUT_SIZE).to(device)
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(Config.TORCHSCRIPT_FILENAME)
        logger.info(f"✅ Model compiled and saved to {Config.TORCHSCRIPT_FILENAME} for high-performance inference.")
    except Exception as e:
        logger.error(f"Could not compile model to TorchScript: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()