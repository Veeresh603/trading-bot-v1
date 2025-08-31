# core/strategy.py
from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd

from core.utils import logger
from core.model import LSTMBrain
from config import Config

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Forces a common interface for generating signals.
    """
    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Should implement generate_signals()")


class AIStrategy(Strategy):
    """
    The AI-driven strategy that uses the LSTM model to generate signals.
    """
    def __init__(self, data_handler, sequence_length):
        self.data_handler = data_handler
        self.symbols = self.data_handler.symbols
        self.sequence_length = sequence_length
        self.model = self._load_model()
        
        # Initialize data sequences for each symbol
        self.data_sequences = {symbol: [] for symbol in self.symbols}

    def _load_model(self):
        """Loads the pre-trained PyTorch model."""
        try:
            model = LSTMBrain()
            model.load_state_dict(torch.load(Config.WEIGHTS_FILENAME))
            model.eval()
            logger.info("✅ AI model weights loaded successfully for strategy.")
            return model
        except FileNotFoundError:
            logger.critical(f"AI model weights '{Config.WEIGHTS_FILENAME}' not found. Please train the model first.")
            raise
    
    def generate_signals(self):
        """
        Main logic for the AI strategy. It checks the latest data for each symbol
        and generates a signal if a full sequence is available.
        """
        signals = {}
        current_bar_time = self.data_handler.get_latest_bar_datetime()

        for symbol in self.symbols:
            # Get the latest scaled features for the symbol
            latest_features = self.data_handler.get_latest_scaled_features(symbol)
            
            if latest_features is None:
                continue

            # Append features to the sequence
            self.data_sequences[symbol].append(latest_features)

            # Ensure we have enough data to make a prediction
            if len(self.data_sequences[symbol]) >= self.sequence_length:
                # Trim sequence to the required length
                sequence = np.array(self.data_sequences[symbol][-self.sequence_length:])
                
                # Get prediction from the AI model
                action, confidence = self.model.decide_action(sequence)
                
                # Map model output to signal direction
                # Model: 1=Buy, 0=Sell/Close, 2=Hold
                direction = "NONE"
                if action == 1:
                    direction = "LONG"
                elif action == 0:
                    direction = "EXIT" # Signal to close any open position

                if direction != "NONE":
                    signals[symbol] = {
                        "timestamp": current_bar_time,
                        "direction": direction,
                        "confidence": confidence
                    }
                
        return signals
