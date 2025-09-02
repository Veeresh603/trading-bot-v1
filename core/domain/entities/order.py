# core/domain/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from decimal import Decimal
import uuid

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class OrderFill:
    fill_id: str
    timestamp: datetime
    quantity: Decimal
    price: Decimal
    commission: Decimal
    venue: str

@dataclass
class Order:
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal(0)
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    
    # Tracking fields
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Execution details
    filled_quantity: Decimal = Decimal(0)
    average_fill_price: Optional[Decimal] = None
    fills: list[OrderFill] = field(default_factory=list)
    
    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_distance: Optional[Decimal] = None
    max_slippage: Optional[Decimal] = None
    
    # Strategy metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # Broker specific
    broker_order_id: Optional[str] = None
    exchange: Optional[str] = None
    venue: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    def fill_percentage(self) -> Decimal:
        if self.quantity == 0:
            return Decimal(0)
        return (self.filled_quantity / self.quantity) * Decimal(100)
    
    def add_fill(self, fill: OrderFill):
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        
        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else Decimal(0)
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self):
        if self.is_active():
            self.status = OrderStatus.CANCELLED
            self.cancelled_at = datetime.now()
    
    def reject(self, reason: str = None):
        self.status = OrderStatus.REJECTED
        if reason:
            self.metadata['rejection_reason'] = reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else None,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity),
            'average_fill_price': str(self.average_fill_price) if self.average_fill_price else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class OrderRequest:
    """Request to create a new order"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)