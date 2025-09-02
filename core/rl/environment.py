# core/rl/environment.py
import numpy as np
import pandas as pd
from core.portfolio import Portfolio
from core.data import HistoricalDataHandler

class MarketEnvironment:
    def __init__(self, data_handler: HistoricalDataHandler, portfolio: Portfolio):
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.current_step = 0
        self.symbol = self.data_handler.symbols[0]  # Assuming single symbol for now
        self.data = self.data_handler.symbol_data[self.symbol]
        self.n_steps = len(self.data)

    def reset(self):
        self.current_step = 0
        self.portfolio.__init__(self.portfolio.initial_capital)
        return self._get_state()

    def step(self, action):
        # Action: 0=HOLD, 1=BUY, 2=SELL
        # FIX: Added 'symbol' to the signal dictionary
        signal = {'symbol': self.symbol, 'direction': ['HOLD', 'BUY', 'SELL'][action]}
        
        current_bar = self.data.iloc[self.current_step]
        order = self.portfolio.create_order_from_signal(signal, current_bar, self.data_handler)
        
        if order:
            self.portfolio.execute_trade(order, current_bar['close'], current_bar.name, 0)
        
        self.portfolio.update_market_data(current_bar.name, {self.symbol: current_bar})
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        reward = self._calculate_reward()
        next_state = self._get_state()
        
        return next_state, reward, done

    def _get_state(self):
        # For simplicity, state is the last `SEQUENCE_LENGTH` bars
        if self.current_step < 60:
            return np.zeros((60, 55)) # Assuming INPUT_SIZE=55
        
        state = self.data.iloc[self.current_step - 60 : self.current_step]
        feature_cols = [col for col in state.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        return state[feature_cols].values

    def _calculate_reward(self):
        # Simple reward: change in portfolio value
        if len(self.portfolio.equity_curve) < 2:
            return 0
        return self.portfolio.equity_curve.iloc[-1] - self.portfolio.equity_curve.iloc[-2]