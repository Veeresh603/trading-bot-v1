# core/strategy.py
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from .model import LSTMBrain
from .utils import logger
from config import Config
from collections import deque
# Import both data handlers to allow for type hinting and flexibility
from .data import HistoricalDataHandler
from .options_data import OptionsDataHandler

class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures that any new strategy implements the necessary methods.
    """
    def __init__(self, data_handler, contracts):
        self.data_handler = data_handler
        self.contracts = contracts

    @abstractmethod
    def generate_signals(self, contract: dict, current_time: pd.Timestamp):
        """
        The core logic of the strategy.
        This method is called for each new data bar.
        It should analyze data and generate trade signals ('BUY', 'SELL', 'HOLD').
        """
        raise NotImplementedError("Should implement generate_signals()")

class AIStrategy(Strategy):
    """
    An AI-driven trading strategy using an LSTM model, adapted for options.
    """
    def __init__(self, data_handler: (HistoricalDataHandler, OptionsDataHandler), contracts: list, sequence_length: int):
        super().__init__(data_handler, contracts)
        self.sequence_length = sequence_length
        self.model = LSTMBrain()
        self.model.load_weights()
        self.model.eval()
        # This will now work for both data handlers
        self.data_sequences = {c['trading_symbol']: deque(maxlen=self.sequence_length) for c in self.contracts}

    def generate_signals(self, contract: dict, current_time: pd.Timestamp):
        """
        Uses the trained LSTM model to generate a trading signal.
        """
        symbol = contract['trading_symbol']
        
        # --- FIX: Access the correct data attribute ---
        # The HistoricalDataHandler uses 'symbol_data', while OptionsDataHandler uses 'data'
        if hasattr(self.data_handler, 'symbol_data'):
            full_data = self.data_handler.symbol_data.get(symbol)
        elif hasattr(self.data_handler, 'data'):
            full_data = self.data_handler.data.get(symbol)
        else:
            logger.error("Data handler has no recognizable data attribute ('symbol_data' or 'data').")
            return None

        if full_data is None or full_data.empty:
            return None

        try:
            end_idx = full_data.index.get_loc(current_time)
        except KeyError:
            # This can happen if the current timestamp is not in the data (e.g., market holidays)
            return None
        
        if end_idx < self.sequence_length - 1:
            return None

        start_idx = end_idx - self.sequence_length + 1
        sequence_df = full_data.iloc[start_idx : end_idx + 1]
        
        # Use the scaler from the data handler for consistency
        feature_cols = [col for col in sequence_df.columns if col in self.data_handler.scaler.feature_names_in_]
        
        if len(feature_cols) != len(self.data_handler.scaler.feature_names_in_):
             logger.warning(f"Feature mismatch for {symbol} at {current_time}. Skipping signal generation.")
             return None

        sequence_features = sequence_df[feature_cols].values

        with torch.no_grad():
            device = next(self.model.parameters()).device
            sequence_tensor = torch.from_numpy(sequence_features).float().unsqueeze(0).to(device)
            prediction = self.model(sequence_tensor)
            
            confidence_scores = torch.softmax(prediction, dim=1)[0]
            confidence = confidence_scores.max().item()
            action_index = torch.argmax(prediction, dim=1).item()
            
        signal_payload = {'symbol': symbol, 'confidence': confidence}
        
        # --- Simplified Signal Logic ---
        # In a real backtest, you might have a MIN_CONFIDENCE_TO_TRADE from config
        if action_index == 1: # Assuming 1 is BUY
            signal_payload['direction'] = 'BUY'
        elif action_index == 0: # Assuming 0 is SELL
            signal_payload['direction'] = 'SELL'
        else: # Assuming 2 is HOLD
            signal_payload['direction'] = 'HOLD'
            
        return signal_payload