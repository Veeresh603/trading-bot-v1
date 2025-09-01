# core/strategy.py
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from .model import LSTMBrain
from .utils import logger
from config import Config
from collections import deque
from core.options_data import OptionsDataHandler

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
    def __init__(self, data_handler: OptionsDataHandler, contracts: list, sequence_length: int):
        super().__init__(data_handler, contracts)
        self.sequence_length = sequence_length
        self.model = LSTMBrain()
        self.model.load_weights()
        self.model.eval()
        self.data_sequences = {c['trading_symbol']: deque(maxlen=self.sequence_length) for c in self.contracts}

    def generate_signals(self, contract: dict, current_time: pd.Timestamp):
        """
        Uses the trained LSTM model to generate an options trading signal.
        """
        symbol = contract['trading_symbol']
        full_data = self.data_handler.data.get(symbol)
        if full_data is None or full_data.empty:
            return None

        try:
            end_idx = full_data.index.get_loc(current_time)
        except KeyError:
            return None
        
        if end_idx < self.sequence_length - 1:
            return None

        start_idx = end_idx - self.sequence_length + 1
        
        sequence_df = full_data.iloc[start_idx : end_idx + 1]
        
        feature_cols = [col for col in sequence_df.columns if col in self.data_handler.scaler.feature_names_in_]
        
        if len(feature_cols) != len(self.data_handler.scaler.feature_names_in_):
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
        
        if action_index == 1 and confidence >= Config.MIN_CONFIDENCE_TO_TRADE:
            signal_payload['direction'] = 'BUY'
        elif action_index == 0:
            signal_payload['direction'] = 'SELL'
        else:
            signal_payload['direction'] = 'HOLD'
            
        return signal_payload