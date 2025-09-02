# core/portfolio.py
import pandas as pd
from .utils import logger
from datetime import datetime
from config import BacktestConfig

class Portfolio:
    """
    Manages the state of our trading account: cash, positions, and equity.
    Also acts as the central risk manager for the backtest.
    """
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.positions = pd.DataFrame(columns=['entry_price', 'quantity', 'market_value'])
        self.holdings = {'cash': initial_capital, 'positions_value': 0.0, 'total': initial_capital}
        self.trades = []
        self.equity_curve = pd.Series(dtype='float64')
        self.trade_count = 0

    def update_market_data(self, timestamp: datetime, market_data: dict):
        """Updates the market value of all open positions based on the latest bar."""
        positions_value = 0.0
        for symbol in self.positions.index:
            if symbol in market_data:
                latest_price = market_data[symbol]['close']
                self.positions.at[symbol, 'market_value'] = self.positions.at[symbol, 'quantity'] * latest_price
                positions_value += self.positions.at[symbol, 'market_value']

        self.holdings['positions_value'] = positions_value
        self.holdings['total'] = self.holdings['cash'] + positions_value
        self.equity_curve[timestamp] = self.holdings['total']

    def create_order_from_signal(self, signal: dict, bar_data: pd.Series, data_handler):
        """Validates a signal and creates an order if risk checks pass."""
        if not isinstance(signal, dict) or signal.get('direction') == 'HOLD':
            return None

        symbol = signal['symbol']
        direction = signal['direction']
        
        if direction == 'BUY' and symbol in self.positions.index:
            return None
        if direction == 'SELL' and symbol not in self.positions.index:
            return None

        price = bar_data['close']
        if direction == 'BUY':
            quantity = self._calculate_position_size(symbol, price, data_handler, bar_data.name)
            if self.holdings['cash'] < quantity * price:
                logger.warning(f"Not enough cash for {symbol}. Order size reduced.")
                quantity = int(self.holdings['cash'] / price)
        else:
            quantity = self.positions.at[symbol, 'quantity']

        if quantity <= 0:
            return None

        return {'symbol': symbol, 'direction': direction, 'quantity': quantity}

    def _calculate_position_size(self, symbol: str, price: float, data_handler, timestamp: datetime) -> int:
        """
        Calculates position size based on volatility (ATR).
        If ATR is not available, it falls back to a fixed fractional size.
        """
        if price <= 0: return 0
        
        atr = data_handler.get_atr(symbol, timestamp)
        if atr is None or atr <= 0:
            logger.warning(f"ATR not available for {symbol} at {timestamp}. Using fixed fractional position size.")
            risk_amount = self.holdings['total'] * BacktestConfig.RISK_PER_TRADE_PCT
            return int(risk_amount / price) if price > 0 else 0
            
        risk_per_share = atr * BacktestConfig.ATR_MULTIPLIER
        if risk_per_share <= 0: return 0

        risk_amount = self.holdings['total'] * BacktestConfig.RISK_PER_TRADE_PCT
        
        quantity = int(risk_amount / risk_per_share)
        return quantity

    def execute_trade(self, order: dict, fill_price: float, timestamp: datetime, commission: float):
        """
        FIX: Corrected signature to accept final trade details from the ExecutionHandler.
        Updates portfolio state after a trade is executed.
        """
        symbol = order['symbol']
        direction = order['direction']
        quantity = order['quantity']

        self.holdings['cash'] -= commission
        self.trade_count += 1
        
        if direction == 'BUY':
            self.positions.loc[symbol] = [fill_price, quantity, quantity * fill_price]
            self.holdings['cash'] -= quantity * fill_price
            action = "BOUGHT"
        elif direction == 'SELL':
            entry_price = self.positions.at[symbol, 'entry_price']
            pnl = (fill_price - entry_price) * quantity - commission
            self.holdings['cash'] += quantity * fill_price
            self.positions.drop(symbol, inplace=True)
            action = "SOLD"
            logger.info(f"Closed {symbol} for P/L: ${pnl:,.2f}")

        logger.info(f"{action} {quantity} {symbol} @ ${fill_price:,.2f}")
        self.trades.append({
            'timestamp': timestamp, 'symbol': symbol, 'direction': direction,
            'quantity': quantity, 'price': fill_price, 'commission': commission
        })