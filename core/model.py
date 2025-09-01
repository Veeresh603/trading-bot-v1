# core/model.py
import torch
import torch.nn as nn
import os
from config import Config
from .utils import logger

# --- C++ Engine (Optional, for performance) ---
try:
    # This will try to import the compiled C++ extension
    from ai_core_wrapper import PyFastLSTMEngine
    logger.info("✅ Successfully imported high-performance C++ AI engine module.")
except ImportError:
    PyFastLSTMEngine = None
    logger.warning("C++ engine module not found or not compiled. Falling back to Python engine.")


class LSTMBrain(nn.Module):
    """
    Defines the core LSTM neural network architecture for the trading AI.
    """
    def __init__(self):
        super(LSTMBrain, self).__init__()
        self.input_size = Config.INPUT_SIZE
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS
        self.output_size = 3  # Buy, Sell, Hold

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=Config.DROPOUT
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        self.cpp_engine = None
        self.load_cpp_engine()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def load_cpp_engine(self):
        """Loads the compiled C++ TorchScript model if available."""
        if PyFastLSTMEngine and os.path.exists(Config.TORCHSCRIPT_FILENAME):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.cpp_engine = PyFastLSTMEngine(Config.TORCHSCRIPT_FILENAME, device)
                logger.info(f"✅ Successfully loaded high-performance C++ AI engine on {device.upper()}.")
            except Exception as e:
                logger.error(f"Failed to load C++ engine: {e}")
                self.cpp_engine = None
        else:
             logger.warning("C++ engine not found or TorchScript model not compiled.")

    # --- FIX APPLIED HERE ---
    # This method was missing. It provides the necessary logic to load the
    # saved model weights from the .pt file created during training.
    def load_weights(self):
        """
        Loads the trained model weights from the file specified in the config.
        """
        weights_file = Config.WEIGHTS_FILENAME
        if not os.path.exists(weights_file):
            logger.error(f"Model weights file not found at '{weights_file}'. Please train the model first.")
            raise FileNotFoundError(f"Model weights file not found: {weights_file}")
            
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.load_state_dict(torch.load(weights_file, map_location=device))
            self.to(device) # Ensure model is on the correct device
            logger.info(f"✅ Successfully loaded model weights from '{weights_file}' to {device.upper()}.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

    def decide_action(self, sequence_data):
        """
        Uses either the C++ engine or the Python engine to make a prediction.
        """
        if self.cpp_engine:
            # Use high-performance C++ engine
            return self.cpp_engine.decide_action(sequence_data)
        else:
            # Fallback to slower Python engine
            with torch.no_grad():
                device = next(self.parameters()).device
                sequence_tensor = torch.from_numpy(sequence_data).float().unsqueeze(0).to(device)
                prediction = self(sequence_tensor)
                confidence = torch.softmax(prediction, dim=1).max().item()
                action_index = torch.argmax(prediction, dim=1).item()
                return action_index, confidence
