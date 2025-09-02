# core/strategy.py
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from .model import LSTMBrain
from .utils import logger
from config import Config
from collections import deque
from datetime import datetime, timedelta
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

class MarketRegimeDetector:
    """Detects market regimes to filter trading signals"""
    
    def detect_regime(self, data, current_time):
        """Detect if market is in high volatility or trending regime"""
        try:
            recent_data = data.loc[:current_time].tail(20)
            if len(recent_data) < 10:
                return "INSUFFICIENT_DATA"
                
            recent_volatility = recent_data['close'].pct_change().std() * np.sqrt(252)
            
            # Calculate trend strength using moving averages
            if 'SMA_20' in recent_data.columns and 'SMA_50' in recent_data.columns:
                ma_20 = recent_data['SMA_20'].iloc[-1]
                ma_50 = recent_data['SMA_50'].iloc[-1]
                trend_strength = abs(ma_20 - ma_50) / ma_50 if ma_50 > 0 else 0
            else:
                trend_strength = 0
            
            if recent_volatility > 0.3:  # 30% annualized volatility
                return "HIGH_VOLATILITY"
            elif recent_volatility < 0.15:
                if trend_strength > 0.05:  # 5% difference in MAs
                    return "LOW_VOL_TRENDING"
                else:
                    return "LOW_VOLATILITY"
            else:
                return "NORMAL"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "UNKNOWN"

class SignalFilter:
    """Filters signals to prevent overtrading and improve quality"""
    
    def __init__(self):
        self.last_signal_time = {}
        self.signal_cooldown = timedelta(hours=1)  # Prevent overtrading
        self.consecutive_losses = {}
        self.max_consecutive_losses = 3
    
    def validate_signal(self, symbol, action_index, current_time, confidence):
        """Filter signals to prevent overtrading and poor quality signals"""
        
        # Confidence check
        if confidence < Config.MIN_CONFIDENCE_TO_TRADE:
            return False
        
        # Cooldown check to prevent overtrading
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
                return False
        
        # Consecutive losses check
        if symbol in self.consecutive_losses:
            if self.consecutive_losses[symbol] >= self.max_consecutive_losses:
                logger.warning(f"Too many consecutive losses for {symbol}, skipping signal")
                return False
        
        self.last_signal_time[symbol] = current_time
        return True
    
    def update_trade_result(self, symbol, profit_loss):
        """Update consecutive loss counter based on trade result"""
        if symbol not in self.consecutive_losses:
            self.consecutive_losses[symbol] = 0
            
        if profit_loss < 0:
            self.consecutive_losses[symbol] += 1
        else:
            self.consecutive_losses[symbol] = 0  # Reset on profitable trade

class AIStrategy(Strategy):
    """
    An AI-driven trading strategy using an LSTM model, enhanced with risk management
    """
    def __init__(self, data_handler: (HistoricalDataHandler, OptionsDataHandler), contracts: list, sequence_length: int):
        super().__init__(data_handler, contracts)
        self.sequence_length = sequence_length
        self.model = LSTMBrain()
        self.model.load_weights()
        self.model.eval()
        
        # Enhanced components
        self.regime_detector = MarketRegimeDetector()
        self.signal_filter = SignalFilter()
        
        # This will now work for both data handlers
        self.data_sequences = {c['trading_symbol']: deque(maxlen=self.sequence_length) for c in self.contracts}

    def generate_signals(self, contract: dict, current_time: pd.Timestamp):
        """
        Enhanced signal generation with regime detection and filtering
        """
        symbol = contract['trading_symbol']
        
        # Access the correct data attribute based on handler type
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

        # Market regime check - avoid trading in high volatility periods
        current_regime = self.regime_detector.detect_regime(full_data, current_time)
        if current_regime in ["HIGH_VOLATILITY", "INSUFFICIENT_DATA"]:
            return {'symbol': symbol, 'direction': 'HOLD', 'confidence': 0.0, 'reason': f'Market regime: {current_regime}'}

        start_idx = end_idx - self.sequence_length + 1
        sequence_df = full_data.iloc[start_idx : end_idx + 1]
        
        # Use the scaler from the data handler for consistency
        if hasattr(self.data_handler, 'scaler') and hasattr(self.data_handler.scaler, 'feature_names_in_'):
            feature_cols = [col for col in sequence_df.columns if col in self.data_handler.scaler.feature_names_in_]
        else:
            # Fallback to excluding price/volume columns
            feature_cols = [col for col in sequence_df.columns if col not in ['open', 'high', 'low', 'close', 'adj close', 'volume']]
        
        if len(feature_cols) == 0:
            logger.warning(f"No feature columns found for {symbol} at {current_time}. Skipping signal generation.")
            return None

        # Ensure we have the right number of features
        if len(feature_cols) != Config.INPUT_SIZE:
            # Pad or truncate to match expected input size
            if len(feature_cols) < Config.INPUT_SIZE:
                # Add placeholder columns
                for i in range(Config.INPUT_SIZE - len(feature_cols)):
                    placeholder_col = f'placeholder_{i}'
                    if placeholder_col not in sequence_df.columns:
                        sequence_df[placeholder_col] = 0.0
                        feature_cols.append(placeholder_col)
            else:
                # Truncate to expected size
                feature_cols = feature_cols[:Config.INPUT_SIZE]

        sequence_features = sequence_df[feature_cols].values

        # Generate prediction
        with torch.no_grad():
            device = next(self.model.parameters()).device
            sequence_tensor = torch.from_numpy(sequence_features).float().unsqueeze(0).to(device)
            prediction = self.model(sequence_tensor)
            
            confidence_scores = torch.softmax(prediction, dim=1)[0]
            confidence = confidence_scores.max().item()
            action_index = torch.argmax(prediction, dim=1).item()
            
        signal_payload = {'symbol': symbol, 'confidence': confidence}
        
        # Apply signal filtering
        if not self.signal_filter.validate_signal(symbol, action_index, current_time, confidence):
            signal_payload['direction'] = 'HOLD'
            signal_payload['reason'] = 'Signal filtered (cooldown, confidence, or consecutive losses)'
            return signal_payload
            
        # Map actions properly with enhanced logic
        action_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
        direction = action_map.get(action_index, 'HOLD')
        
        # Additional confidence-based filtering
        if direction != 'HOLD':
            # Higher confidence required for volatile regimes
            min_confidence = Config.MIN_CONFIDENCE_TO_TRADE
            if current_regime == "NORMAL":
                min_confidence *= 1.1  # 10% higher confidence required
                
            if confidence < min_confidence:
                direction = 'HOLD'
                signal_payload['reason'] = f'Confidence {confidence:.3f} below threshold {min_confidence:.3f}'
        
        signal_payload['direction'] = direction
        signal_payload['regime'] = current_regime
        
        return signal_payload
    
    def update_signal_filter_with_trade_result(self, symbol, profit_loss):
        """Update signal filter based on trade results"""
        self.signal_filter.update_trade_result(symbol, profit_loss)