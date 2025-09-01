# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from config import Config, BacktestConfig
from core.model import LSTMBrain
from core.data import HistoricalDataHandler
from core.utils import logger

def get_target_labels_triple_barrier(
    close_prices: pd.Series,
    profit_take_pct: float = 0.015,  # 1.5%
    stop_loss_pct: float = 0.008,   # 0.8%
    time_limit_bars: int = 10
) -> pd.Series:
    """
    Implements the Triple Barrier Method for labeling.
    This teaches the AI about risk-to-reward, not just price direction.

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
                labels.iloc[i] = 1  # Profit target hit
                break
            elif future_price <= stop_loss_target:
                labels.iloc[i] = 0  # Stop loss hit
                break
    return labels

def create_sequences(input_data: pd.DataFrame, target_data: pd.Series, seq_length: int):
    """Creates sequences and corresponding labels for the LSTM."""
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data.iloc[i:(i + seq_length)].values
        y = target_data.iloc[i + seq_length]
        # We only want to train on clear buy/sell signals, not 'hold'
        if y in [0, 1]:
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    logger.info("--- Starting AI Model Training ---")

    # Use the new, robust HistoricalDataHandler to get consistently processed data
    # We use the backtest config dates to ensure our training data is from the same period
    data_handler = HistoricalDataHandler(
        symbols=Config.SYMBOLS_TO_TRADE,
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )

    all_features, all_labels = [], []

    for symbol in Config.SYMBOLS_TO_TRADE:
        # Get the fully processed (features calculated + scaled) data from the handler
        processed_data = data_handler.symbol_data.get(symbol)
        if processed_data is None or processed_data.empty:
            logger.warning(f"No data found for {symbol}, skipping.")
            continue
        
        logger.info(f"Processing data for {symbol}...")
        
        feature_columns = [col for col in processed_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        features = processed_data[feature_columns]
        
        # Generate labels using the Triple Barrier Method
        labels = get_target_labels_triple_barrier(processed_data['close'])

        # Combine and drop any rows with missing data
        combined = pd.concat([features, labels.rename('target')], axis=1).dropna()
        features, labels = combined.drop('target', axis=1), combined['target']

        # Create sequences for the LSTM
        X, y = create_sequences(features, labels, Config.SEQUENCE_LENGTH)
        
        if len(X) > 0:
            all_features.append(X)
            all_labels.append(y)
        else:
            logger.warning(f"Not enough valid sequences generated for {symbol}.")


    if not all_features:
        logger.error("No data available for training after processing. Exiting.")
        return

    # Combine data from all symbols into one large dataset
    X_all, y_all = np.concatenate(all_features), np.concatenate(all_labels)
    logger.info(f"Total training sequences created: {len(X_all)}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    # Initialize model, optimizer, and loss function
    model = LSTMBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Convert data to PyTorch tensors
    X_train_tensor, y_train_tensor = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    X_val_tensor, y_val_tensor = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()

    # --- Training Loop ---
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
        
        # Validation at the end of each epoch
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Accuracy: {accuracy:.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), Config.WEIGHTS_FILENAME)
    logger.info(f"✅ Model training complete. Weights saved to {Config.WEIGHTS_FILENAME}")

    # Optional: Save a TorchScript version for the C++ engine
    try:
        model.eval()
        example_input = torch.randn(1, Config.SEQUENCE_LENGTH, Config.INPUT_SIZE)
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(Config.TORCHSCRIPT_FILENAME)
        logger.info(f"✅ Model compiled and saved to {Config.TORCHSCRIPT_FILENAME} for high-performance inference.")
    except Exception as e:
        logger.error(f"Could not compile model to TorchScript: {e}")

if __name__ == "__main__":
    main()
