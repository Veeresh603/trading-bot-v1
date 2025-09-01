# core/strategy.py
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from .model import LSTMBrain
from .utils import logger

class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures that any new strategy implements the necessary methods.
    """
    @abstractmethod
    def generate_signals(self, symbol: str, current_time: pd.Timestamp):
        """
        The core logic of the strategy.
        This method is called for each new data bar.
        It should analyze data and generate trade signals ('BUY', 'SELL', 'HOLD').
        """
        raise NotImplementedError("Should implement generate_signals()")


class AIStrategy(Strategy):
    """
    An AI-driven trading strategy using an LSTM model.
    """
    def __init__(self, data_handler, sequence_length: int):
        self.data_handler = data_handler
        self.sequence_length = sequence_length
        self.model = LSTMBrain()
        self.model.load_weights() # Assumes weights are loaded on init
        self.model.eval()

    def generate_signals(self, symbol: str, current_time: pd.Timestamp):
        """
        Uses the trained LSTM model to generate a trading signal.
        """
        # 1. Get the last 'sequence_length' data points up to the current time
        full_data = self.data_handler.symbol_data.get(symbol)
        if full_data is None:
            return None

        try:
            # Get the index location for the current timestamp
            end_idx = full_data.index.get_loc(current_time)
        except KeyError:
            # This can happen if the timestamp doesn't align perfectly, skip this bar
            return None
        
        # Ensure we have enough historical data to form a full sequence
        if end_idx < self.sequence_length - 1:
            return None # Not enough data yet to make a prediction

        start_idx = end_idx - self.sequence_length + 1
        
        # 2. Extract the feature columns for the model
        sequence_df = full_data.iloc[start_idx : end_idx + 1]
        
        # Use the same feature columns the scaler was trained on
        feature_cols = [col for col in sequence_df.columns if col in self.data_handler.scaler.feature_names_in_]
        
        if len(feature_cols) != len(self.data_handler.scaler.feature_names_in_):
             # This check prevents errors if a column is missing for some reason
             return None

        sequence_features = sequence_df[feature_cols].values
        
        # 3. Use the model to predict the action
        with torch.no_grad():
            # --- FIX APPLIED HERE ---
            # First, determine which device the model is currently living on (CPU or GPU).
            device = next(self.model.parameters()).device
            
            # Create the input tensor and IMMEDIATELY move it to the same device as the model.
            sequence_tensor = torch.from_numpy(sequence_features).float().unsqueeze(0).to(device)

            prediction = self.model(sequence_tensor)
            action_index = torch.argmax(prediction, dim=1).item()

        # 4. Convert the prediction index to a signal
        # Based on train.py: 1=BUY, 0=SELL, 2=HOLD
        if action_index == 1:
            return 'BUY'
        elif action_index == 0:
            return 'SELL'
        else: # action_index == 2
            return 'HOLD'

