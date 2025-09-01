# core/execution.py
from .utils import logger
import random

class SimulatedExecutionHandler:
    """
    Simulates the execution of trades, including realistic costs
    like commissions and slippage.
    """
    def __init__(self, commission_bps: int = 0, slippage_bps: int = 0):
        """
        Initializes the execution handler.

        Args:
            commission_bps (int): The commission fee in basis points (e.g., 5 bps = 0.05%).
            slippage_bps (int): The estimated slippage in basis points.
        """
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _calculate_costs(self, price: float, quantity: int, direction: str):
        """Calculates commission and simulates slippage."""
        # --- Commission ---
        commission = (price * quantity) * (self.commission_bps / 10000.0)

        # --- Slippage ---
        slippage_amount = (price * (self.slippage_bps / 10000.0)) * random.choice([-1, 1])
        
        # Slippage hurts you on both buys and sells
        if direction == 'BUY':
            fill_price = price + abs(slippage_amount)
        elif direction == 'SELL':
            fill_price = price - abs(slippage_amount)
        else:
            fill_price = price

        return fill_price, commission

    def execute_order(self, order: dict, bar_data, portfolio):
        """
        Executes an order, updates the portfolio, and logs the trade.
        The portfolio object is now passed here directly.
        """
        if not order:
            return

        symbol = order['symbol']
        direction = order['direction']
        quantity = order['quantity']
        price = bar_data['close'] # Ideal execution price
        timestamp = bar_data.name # The datetime index of the bar

        fill_price, commission = self._calculate_costs(price, quantity, direction)

        # Update the portfolio with the results of the execution
        portfolio.execute_trade(order, fill_price, timestamp, commission)

