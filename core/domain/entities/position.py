# core/domain/entities/position.py
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"

@dataclass
class PositionUpdate:
    """Represents an update to a position"""
    timestamp: datetime
    quantity_change: Decimal
    price: Decimal
    commission: Decimal
    order_id: str
    update_type: str  # "OPEN", "ADD", "REDUCE", "CLOSE"

@dataclass
class Position:
    position_id: str
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    
    # Status tracking
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    # P&L calculations
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    total_commission: Decimal = Decimal(0)
    
    # Risk metrics
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_distance: Optional[Decimal] = None
    max_position_value: Decimal = Decimal(0)
    max_drawdown: Decimal = Decimal(0)
    
    # Position management
    average_entry_price: Decimal = field(default_factory=lambda: Decimal(0))
    total_quantity_traded: Decimal = field(default_factory=lambda: Decimal(0))
    
    # Strategy tracking
    strategy_id: Optional[str] = None
    signal_ids: List[str] = field(default_factory=list)
    
    # Update history
    updates: List[PositionUpdate] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.average_entry_price = self.entry_price
        self.total_quantity_traded = self.quantity
        self.update_unrealized_pnl()
        self.max_position_value = self.market_value()
    
    def market_value(self) -> Decimal:
        """Calculate current market value of position"""
        return self.quantity * self.current_price
    
    def update_price(self, new_price: Decimal):
        """Update current price and recalculate P&L"""
        self.current_price = new_price
        self.update_unrealized_pnl()
        self.update_max_drawdown()
    
    def update_unrealized_pnl(self):
        """Calculate unrealized P&L"""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (self.current_price - self.average_entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.average_entry_price - self.current_price) * self.quantity
    
    def update_max_drawdown(self):
        """Track maximum drawdown"""
        current_value = self.market_value()
        if current_value > self.max_position_value:
            self.max_position_value = current_value
        
        drawdown = (self.max_position_value - current_value) / self.max_position_value if self.max_position_value > 0 else Decimal(0)
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def add_to_position(self, quantity: Decimal, price: Decimal, commission: Decimal = Decimal(0), order_id: str = None):
        """Add to existing position"""
        # Update average entry price
        total_cost = (self.average_entry_price * self.quantity) + (price * quantity)
        self.quantity += quantity
        self.average_entry_price = total_cost / self.quantity if self.quantity > 0 else Decimal(0)
        
        # Update totals
        self.total_quantity_traded += quantity
        self.total_commission += commission
        
        # Add update record
        update = PositionUpdate(
            timestamp=datetime.now(),
            quantity_change=quantity,
            price=price,
            commission=commission,
            order_id=order_id or "",
            update_type="ADD"
        )
        self.updates.append(update)
        
        # Recalculate P&L
        self.update_unrealized_pnl()
    
    def reduce_position(self, quantity: Decimal, price: Decimal, commission: Decimal = Decimal(0), order_id: str = None) -> Decimal:
        """Reduce position and calculate realized P&L"""
        if quantity > self.quantity:
            quantity = self.quantity
        
        # Calculate realized P&L for this reduction
        if self.side == PositionSide.LONG:
            trade_pnl = (price - self.average_entry_price) * quantity
        else:  # SHORT
            trade_pnl = (self.average_entry_price - price) * quantity
        
        # Update position
        self.quantity -= quantity
        self.realized_pnl += trade_pnl - commission
        self.total_commission += commission
        
        # Update status
        if self.quantity == 0:
            self.status = PositionStatus.CLOSED
            self.closed_at = datetime.now()
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
        
        # Add update record
        update = PositionUpdate(
            timestamp=datetime.now(),
            quantity_change=-quantity,
            price=price,
            commission=commission,
            order_id=order_id or "",
            update_type="REDUCE" if self.quantity > 0 else "CLOSE"
        )
        self.updates.append(update)
        
        # Recalculate unrealized P&L
        self.update_unrealized_pnl()
        
        return trade_pnl
    
    def close_position(self, price: Decimal, commission: Decimal = Decimal(0), order_id: str = None) -> Decimal:
        """Close entire position"""
        return self.reduce_position(self.quantity, price, commission, order_id)
    
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_commission
        return total_pnl > 0
    
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calculate risk/reward ratio if stop loss and take profit are set"""
        if not self.stop_loss or not self.take_profit:
            return None
        
        risk = abs(self.average_entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.average_entry_price)
        
        if risk == 0:
            return None
        
        return reward / risk
    
    def should_close_on_stop_loss(self) -> bool:
        """Check if position should be closed due to stop loss"""
        if not self.stop_loss:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def should_close_on_take_profit(self) -> bool:
        """Check if position should be closed due to take profit"""
        if not self.take_profit:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit
    
    def update_trailing_stop(self):
        """Update trailing stop based on current price"""
        if not self.trailing_stop_distance:
            return
        
        if self.side == PositionSide.LONG:
            new_stop = self.current_price - self.trailing_stop_distance
            if not self.stop_loss or new_stop > self.stop_loss:
                self.stop_loss = new_stop
        else:  # SHORT
            new_stop = self.current_price + self.trailing_stop_distance
            if not self.stop_loss or new_stop < self.stop_loss:
                self.stop_loss = new_stop
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price),
            'current_price': str(self.current_price),
            'average_entry_price': str(self.average_entry_price),
            'unrealized_pnl': str(self.unrealized_pnl),
            'realized_pnl': str(self.realized_pnl),
            'total_commission': str(self.total_commission),
            'status': self.status.value,
            'opened_at': self.opened_at.isoformat(),
            'metadata': self.metadata
        }