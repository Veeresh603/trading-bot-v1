# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import sys

from config import Config, BacktestConfig
from core.model import LSTMBrain
from core.data import HistoricalDataHandler
from core.utils import logger
from kiteconnect import KiteConnect

def get_target_labels_future_direction(
    price_data: pd.DataFrame,
    look_forward_bars: int = 10,
    atr_threshold_multiplier: float = 0.5
) -> pd.Series:
    """
    Creates labels based on the future price direction relative to ATR.
    - 1 (BUY): If future price > entry_price + (ATR * multiplier)
    - 0 (SELL): If future price < entry_price - (ATR * multiplier)
    - 2 (HOLD): If price stays within the neutral zone.
    """
    labels = pd.Series(2, index=price_data.index) # Default to HOLD
    
    # Calculate future price `look_forward_bars` ahead
    future_close = price_data['close'].shift(-look_forward_bars)
    
    # Calculate the dynamic threshold using ATR
    atr_threshold = price_data['ATR'] * atr_threshold_multiplier
    
    # Conditions for BUY and SELL signals
    buy_condition = future_close > price_data['close'] + atr_threshold
    sell_condition = future_close < price_data['close'] - atr_threshold
    
    labels[buy_condition] = 1
    labels[sell_condition] = 0
    
    # The rest will remain as HOLD (2)
    return labels

def create_sequences(input_data: pd.DataFrame, target_data: pd.Series, seq_length: int):
    """Creates sequences and corresponding labels for the LSTM."""
    xs, ys = [], []
    # Adjust loop to ensure target_data slicing is always valid
    for i in range(len(input_data) - seq_length - 1):
        x = input_data.iloc[i:(i + seq_length)].values
        # Target is the label corresponding to the *end* of the sequence
        y = target_data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    logger.info("--- Starting AI Model Training on Underlying Asset ---")
    
    data_handler = HistoricalDataHandler(
        symbols=[Config.UNDERLYING_SYMBOL],
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )

    if not data_handler.symbol_data:
        logger.error("No data available for training after processing. Exiting.")
        return

    processed_data = data_handler.symbol_data[Config.UNDERLYING_SYMBOL]
    feature_cols = [col for col in processed_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    features = processed_data[feature_cols]
    
    # Use the new, more robust labeling function
    labels = get_target_labels_future_direction(processed_data[['close', 'ATR']])

    # Drop NaNs that may have been created by the label shifting
    combined = pd.concat([features, labels.rename('target')], axis=1).dropna()
    features_final = combined.drop('target', axis=1)
    labels_final = combined['target']

    X, y = create_sequences(features_final, labels_final, Config.SEQUENCE_LENGTH)
    
    if len(X) == 0:
        logger.error("Not enough valid sequences generated. Try a larger date range.")
        return
        
    logger.info(f"Total training sequences created: {len(X)}")
    
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")

    if len(unique) < 3:
        logger.warning("The dataset does not contain all three classes (BUY, SELL, HOLD). This may be okay but indicates low volatility or strong trends.")

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Calculated class weights: {class_weights}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LSTMBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)
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
        
        scheduler.step()

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