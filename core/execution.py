# core/execution.py
from .utils import logger
import random
from datetime import datetime

class BacktestExecutionHandler:
    """
    Simulates the execution of trades in a backtest environment, 
    including realistic costs like commissions and slippage.
    """
    def __init__(self, commission_bps: int = 0, slippage_bps: int = 0):
        """
        Initializes the execution handler.

        Args:
            commission_bps (int): The commission fee in basis points (e.g., 2.0 bps = 0.02%).
            slippage_bps (int): The estimated slippage in basis points.
        """
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _calculate_costs(self, price: float, quantity: int, direction: str):
        """Calculates commission and simulates slippage."""
        commission = (price * quantity) * (self.commission_bps / 10000.0)
        
        slippage_amount = price * (self.slippage_bps / 10000.0) * random.uniform(0.8, 1.2)
        
        if direction == 'BUY':
            fill_price = price + slippage_amount
        elif direction == 'SELL':
            fill_price = price - slippage_amount
        else:
            fill_price = price

        return fill_price, commission

    def execute_order(self, order: dict, bar_data, portfolio):
        """
        Executes an order, updates the portfolio, and logs the trade.
        """
        if not order:
            return

        symbol = order['symbol']
        direction = order['direction']
        quantity = order['quantity']
        price = bar_data['close']
        timestamp = bar_data.name

        fill_price, commission = self._calculate_costs(price, quantity, direction)

        # Call the portfolio's execute_trade method with the correct arguments
        portfolio.execute_trade(order, fill_price, timestamp, commission)