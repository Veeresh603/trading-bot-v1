# core/model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from config import Config
from .utils import logger

# Import the C++ engine
try:
    from ai_core_wrapper import PyFastLSTMEngine
except ImportError:
    logger.critical("Could not import C++ engine 'ai_core_wrapper'. The bot will fall back to slower PyTorch inference.")
    PyFastLSTMEngine = None


class LSTMBrain(nn.Module):
    """
    Unified LSTM model for both training (PyTorch) and high-speed inference (C++).
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            Config.INPUT_SIZE,
            Config.HIDDEN_SIZE,
            Config.NUM_LAYERS,
            batch_first=True,
            dropout=Config.DROPOUT
        )
        self.fc = nn.Linear(Config.HIDDEN_SIZE, 3)  # 0: Sell, 1: Buy, 2: Hold

        self.brain_cpp = None
        self.load_cpp_engine()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        # Softmax provides probability distribution over actions
        return F.softmax(out, dim=-1)

    def load_cpp_engine(self):
        """Loads the compiled C++ inference engine if available."""
        if PyFastLSTMEngine and os.path.exists(Config.TORCHSCRIPT_FILENAME):
            try:
                self.brain_cpp = PyFastLSTMEngine(Config.TORCHSCRIPT_FILENAME)
                logger.info("✅ Successfully loaded high-performance C++ AI engine.")
            except Exception as e:
                logger.error(f"❌ Failed to load C++ AI engine: {e}")
                self.brain_cpp = None
        else:
            logger.warning("C++ engine not found or TorchScript model not compiled.")

    def decide_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Makes a trading decision. Uses the C++ engine if available, otherwise
        falls back to the slower PyTorch model for inference.

        Args:
            state (np.ndarray): The input features for the current time step.

        Returns:
            Tuple[int, float]: The decided action (0, 1, or 2) and the confidence score.
        """
        if self.brain_cpp:
            # Use high-performance C++ engine
            action_probs = self.brain_cpp.forward(state.astype(np.float32))
        else:
            # Fallback to PyTorch model (slower)
            self.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Reshape state to (batch_size, sequence_length, input_size)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
                action_probs_tensor = self(state_tensor)
                action_probs = action_probs_tensor.squeeze().cpu().numpy()

        action = np.argmax(action_probs).item()
        confidence = action_probs[action].item()
        return action, confidence