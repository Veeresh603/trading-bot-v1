# core/execution.py
import queue
from datetime import datetime

class SimulatedExecutionHandler:
    """
    Simulates the execution of orders, including slippage and commissions.
    This is a crucial component for realistic backtesting.
    """
    def __init__(self, commission_per_trade, slippage_pct):
        self.commission = commission_per_trade
        self.slippage_pct = slippage_pct
        self.fills_queue = queue.Queue()

    def place_order(self, symbol, quantity, direction):
        """
        Simulates placing an order. In a real system, this would
        interact with a broker's API. Here, it simulates the fill.
        """
        # In a backtest, we assume the order is filled on the next bar's open.
        # This is a simplification. More complex models could use VWAP.
        # For simplicity, we'll just create a fill event to be processed.
        # The actual price will be determined by the Portfolio using next bar's data.
        # Here we just queue the intent to trade.
        # A more realistic model would require the data_handler here to get the fill price.
        # For now, the portfolio will handle the fill logic.
        
        # In this simplified model, we will let the portfolio create the fill event
        # since it has access to the next bar's data. This handler is more of a
        # placeholder for where broker interaction logic would go.
        # A better implementation would pass the data_handler to this class.
        pass # The Portfolio will directly create fill events for simplicity.
    
    def calculate_fill_price(self, ideal_price, direction):
        """Calculates the execution price including slippage."""
        if direction == 'BUY':
            return ideal_price * (1 + self.slippage_pct)
        else: # SELL
            return ideal_price * (1 - self.slippage_pct)

    def calculate_commission(self, trade_value):
        """Calculates the commission for a trade."""
        return trade_value * self.commission
        
    def create_fill_event(self, timestamp, symbol, quantity, direction, fill_price):
        """Creates a fill event and puts it in the queue."""
        trade_value = fill_price * quantity
        commission = self.calculate_commission(trade_value)
        fill = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'direction': direction,
            'fill_price': fill_price,
            'commission': commission
        }
        self.fills_queue.put(fill)
