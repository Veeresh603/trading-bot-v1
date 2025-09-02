# core/model.py
import torch
import torch.nn as nn
import os
from config import Config
from .utils import logger
import numpy as np
import math

try:
    from ai_core_wrapper import PyFastLSTMEngine
    logger.info("✅ Successfully imported high-performance C++ AI engine module.")
except ImportError:
    PyFastLSTMEngine = None
    logger.warning("C++ engine module not found or not compiled. Falling back to Python engine.")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class LSTMBrain(nn.Module):
    """
    Defines the core LSTM neural network architecture with Attention.
    """
    def __init__(self):
        super(LSTMBrain, self).__init__()
        self.input_size = Config.INPUT_SIZE
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 3
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            # FIX: Increased dropout for better regularization
            dropout=0.4 
        )
        self.attention = Attention(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.cpp_engine = None
        self.load_cpp_engine()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        context_vector = self.attention(out)
        out = self.fc(context_vector)
        return out

    def load_cpp_engine(self):
        """Loads the compiled C++ TorchScript model if available."""
        if PyFastLSTMEngine and os.path.exists(Config.TORCHSCRIPT_FILENAME):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.cpp_engine = PyFastLSTMEngine(Config.TORCHSCRIPT_FILENAME)
                logger.info(f"✅ Successfully loaded high-performance C++ AI engine on {device.upper()}.")
            except Exception as e:
                logger.error(f"Failed to load C++ engine: {e}")
                self.cpp_engine = None
        else:
             logger.warning("C++ engine not found or TorchScript model not compiled.")

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
            self.to(device)
            logger.info(f"✅ Successfully loaded model weights from '{weights_file}' to {device.upper()}.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

    def decide_action(self, sequence_data):
        """
        Uses either the C++ engine or the Python engine to make a prediction.
        """
        if self.cpp_engine:
            results = self.cpp_engine.forward(sequence_data.flatten())
            prediction = torch.from_numpy(np.array(results)).unsqueeze(0)
            confidence = torch.softmax(prediction, dim=1).max().item()
            action_index = torch.argmax(prediction, dim=1).item()
            return action_index, confidence
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                sequence_tensor = torch.from_numpy(sequence_data).float().unsqueeze(0).to(device)
                prediction = self(sequence_tensor)
                confidence = torch.softmax(prediction, dim=1).max().item()
                action_index = torch.argmax(prediction, dim=1).item()
                return action_index, confidence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBrain(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, nhead=4):
        super(TransformerBrain, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layers = nn.TransformerEncoderLayer(input_size, nhead, hidden_size, 0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(input_size, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[:, -1, :]