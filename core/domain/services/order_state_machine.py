# core/domain/services/order_state_machine.py
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging
from core.domain.entities.order import Order, OrderStatus
from core.infrastructure.event_bus import Event, EventType, IEventBus

logger = logging.getLogger(__name__)

class OrderEvent(Enum):
    SUBMIT = "SUBMIT"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    CANCEL = "CANCEL"
    REJECT = "REJECT"
    EXPIRE = "EXPIRE"
    MODIFY = "MODIFY"

@dataclass
class StateTransition:
    from_state: OrderStatus
    event: OrderEvent
    to_state: OrderStatus
    condition: Optional[Callable[[Order], bool]] = None
    action: Optional[Callable[[Order], None]] = None

class OrderStateMachine:
    """State machine for order lifecycle management"""
    
    def __init__(self, event_bus: IEventBus = None):
        self.event_bus = event_bus
        self.transitions: Dict[tuple[OrderStatus, OrderEvent], StateTransition] = {}
        self.state_handlers: Dict[OrderStatus, List[Callable]] = {}
        self._define_transitions()
    
    def _define_transitions(self):
        """Define all valid state transitions"""
        transitions = [
            # From PENDING
            StateTransition(OrderStatus.PENDING, OrderEvent.SUBMIT, OrderStatus.SUBMITTED),
            StateTransition(OrderStatus.PENDING, OrderEvent.CANCEL, OrderStatus.CANCELLED),
            StateTransition(OrderStatus.PENDING, OrderEvent.REJECT, OrderStatus.REJECTED),
            
            # From SUBMITTED
            StateTransition(OrderStatus.SUBMITTED, OrderEvent.ACKNOWLEDGE, OrderStatus.ACKNOWLEDGED),
            StateTransition(OrderStatus.SUBMITTED, OrderEvent.REJECT, OrderStatus.REJECTED),
            StateTransition(OrderStatus.SUBMITTED, OrderEvent.CANCEL, OrderStatus.CANCELLED),
            StateTransition(OrderStatus.SUBMITTED, OrderEvent.FILL, OrderStatus.FILLED),
            
            # From ACKNOWLEDGED
            StateTransition(OrderStatus.ACKNOWLEDGED, OrderEvent.FILL, OrderStatus.FILLED),
            StateTransition(OrderStatus.ACKNOWLEDGED, OrderEvent.PARTIAL_FILL, OrderStatus.PARTIALLY_FILLED),
            StateTransition(OrderStatus.ACKNOWLEDGED, OrderEvent.CANCEL, OrderStatus.CANCELLED),
            StateTransition(OrderStatus.ACKNOWLEDGED, OrderEvent.EXPIRE, OrderStatus.EXPIRED),
            StateTransition(OrderStatus.ACKNOWLEDGED, OrderEvent.REJECT, OrderStatus.REJECTED),
            
            # From PARTIALLY_FILLED
            StateTransition(OrderStatus.PARTIALLY_FILLED, OrderEvent.FILL, OrderStatus.FILLED),
            StateTransition(OrderStatus.PARTIALLY_FILLED, OrderEvent.PARTIAL_FILL, OrderStatus.PARTIALLY_FILLED),
            StateTransition(OrderStatus.PARTIALLY_FILLED, OrderEvent.CANCEL, OrderStatus.CANCELLED),
            StateTransition(OrderStatus.PARTIALLY_FILLED, OrderEvent.EXPIRE, OrderStatus.EXPIRED),
        ]
        
        for transition in transitions:
            key = (transition.from_state, transition.event)
            self.transitions[key] = transition
    
    def can_transition(self, order: Order, event: OrderEvent) -> bool:
        """Check if a transition is valid"""
        key = (order.status, event)
        if key not in self.transitions:
            return False
        
        transition = self.transitions[key]
        if transition.condition:
            return transition.condition(order)
        
        return True
    
    async def process_event(self, order: Order, event: OrderEvent, **kwargs) -> bool:
        """Process an event and transition the order state"""
        key = (order.status, event)
        
        if key not in self.transitions:
            logger.warning(f"Invalid transition: {order.status} -> {event} for order {order.order_id}")
            return False
        
        transition = self.transitions[key]
        
        # Check condition if exists
        if transition.condition and not transition.condition(order):
            logger.warning(f"Transition condition failed: {order.status} -> {event}")
            return False
        
        # Store old state
        old_status = order.status
        
        # Update order state
        order.status = transition.to_state
        self._update_timestamps(order, event)
        
        # Execute action if exists
        if transition.action:
            transition.action(order)
        
        # Trigger state handlers
        await self._trigger_state_handlers(order, old_status)
        
        # Publish event
        if self.event_bus:
            await self._publish_state_change(order, old_status, event, kwargs)
        
        logger.info(f"Order {order.order_id} transitioned: {old_status} -> {order.status} (event: {event})")
        
        return True
    
    def _update_timestamps(self, order: Order, event: OrderEvent):
        """Update order timestamps based on event"""
        now = datetime.now()
        
        if event == OrderEvent.SUBMIT:
            order.submitted_at = now
        elif event == OrderEvent.ACKNOWLEDGE:
            order.acknowledged_at = now
        elif event in [OrderEvent.FILL, OrderEvent.PARTIAL_FILL]:
            if order.status == OrderStatus.FILLED:
                order.filled_at = now
        elif event == OrderEvent.CANCEL:
            order.cancelled_at = now
    
    async def _trigger_state_handlers(self, order: Order, old_status: OrderStatus):
        """Trigger registered handlers for state change"""
        if order.status in self.state_handlers:
            for handler in self.state_handlers[order.status]:
                try:
                    await handler(order, old_status)
                except Exception as e:
                    logger.error(f"Error in state handler: {e}")
    
    async def _publish_state_change(self, order: Order, old_status: OrderStatus, event: OrderEvent, data: dict):
        """Publish state change event to event bus"""
        event_payload = Event(
            event_type=EventType.ORDER_UPDATED if order.status != OrderStatus.FILLED else EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            payload={
                'order': order.to_dict(),
                'old_status': old_status.value,
                'new_status': order.status.value,
                'transition_event': event.value,
                'additional_data': data
            },
            correlation_id=order.order_id,
            source="OrderStateMachine"
        )
        
        await self.event_bus.publish(event_payload)
    
    def register_state_handler(self, status: OrderStatus, handler: Callable):
        """Register a handler for when an order enters a specific state"""
        if status not in self.state_handlers:
            self.state_handlers[status] = []
        self.state_handlers[status].append(handler)
    
    def get_valid_events(self, order: Order) -> List[OrderEvent]:
        """Get list of valid events for current order state"""
        valid_events = []
        for (from_state, event), transition in self.transitions.items():
            if from_state == order.status:
                if not transition.condition or transition.condition(order):
                    valid_events.append(event)
        return valid_events
    
    def visualize_state_machine(self) -> str:
        """Generate a text representation of the state machine"""
        lines = ["Order State Machine:"]
        lines.append("-" * 50)
        
        for status in OrderStatus:
            valid_transitions = []
            for (from_state, event), transition in self.transitions.items():
                if from_state == status:
                    valid_transitions.append(f"{event.value} -> {transition.to_state.value}")
            
            if valid_transitions:
                lines.append(f"{status.value}:")
                for transition in valid_transitions:
                    lines.append(f"  - {transition}")
        
        return "\n".join(lines)

class OrderLifecycleManager:
    """Manages the complete lifecycle of orders"""
    
    def __init__(self, state_machine: OrderStateMachine, event_bus: IEventBus):
        self.state_machine = state_machine
        self.event_bus = event_bus
        self.orders: Dict[str, Order] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register state handlers"""
        self.state_machine.register_state_handler(OrderStatus.FILLED, self._handle_filled_order)
        self.state_machine.register_state_handler(OrderStatus.CANCELLED, self._handle_cancelled_order)
        self.state_machine.register_state_handler(OrderStatus.REJECTED, self._handle_rejected_order)
    
    async def _handle_filled_order(self, order: Order, old_status: OrderStatus):
        """Handle order fill completion"""
        logger.info(f"Order {order.order_id} filled: {order.filled_quantity} @ {order.average_fill_price}")
        
        # Publish fill event
        event = Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            payload={'order': order.to_dict()},
            correlation_id=order.order_id,
            source="OrderLifecycleManager"
        )
        await self.event_bus.publish(event)
    
    async def _handle_cancelled_order(self, order: Order, old_status: OrderStatus):
        """Handle order cancellation"""
        logger.info(f"Order {order.order_id} cancelled")
        
        event = Event(
            event_type=EventType.ORDER_CANCELLED,
            timestamp=datetime.now(),
            payload={'order': order.to_dict()},
            correlation_id=order.order_id,
            source="OrderLifecycleManager"
        )
        await self.event_bus.publish(event)
    
    async def _handle_rejected_order(self, order: Order, old_status: OrderStatus):
        """Handle order rejection"""
        reason = order.metadata.get('rejection_reason', 'Unknown')
        logger.warning(f"Order {order.order_id} rejected: {reason}")
        
        event = Event(
            event_type=EventType.RISK_ALERT,
            timestamp=datetime.now(),
            payload={
                'order': order.to_dict(),
                'alert_type': 'ORDER_REJECTED',
                'reason': reason
            },
            correlation_id=order.order_id,
            source="OrderLifecycleManager"
        )
        await self.event_bus.publish(event)
    
    async def submit_order(self, order: Order) -> bool:
        """Submit a new order"""
        self.orders[order.order_id] = order
        return await self.state_machine.process_event(order, OrderEvent.SUBMIT)
    
    async def acknowledge_order(self, order_id: str, broker_order_id: str) -> bool:
        """Acknowledge order from broker"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.broker_order_id = broker_order_id
        return await self.state_machine.process_event(order, OrderEvent.ACKNOWLEDGE)
    
    async def fill_order(self, order_id: str, fill_data: dict) -> bool:
        """Process order fill"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Determine if partial or complete fill
        event = OrderEvent.PARTIAL_FILL if fill_data.get('is_partial') else OrderEvent.FILL
        
        return await self.state_machine.process_event(order, event, **fill_data)
    
    async def cancel_order(self, order_id: str, reason: str = None) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if reason:
            order.metadata['cancellation_reason'] = reason
        
        return await self.state_machine.process_event(order, OrderEvent.CANCEL)
    
    async def reject_order(self, order_id: str, reason: str) -> bool:
        """Reject an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.metadata['rejection_reason'] = reason
        
        return await self.state_machine.process_event(order, OrderEvent.REJECT)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.orders.values() if order.is_active()]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for a specific symbol"""
        return [order for order in self.orders.values() if order.symbol == symbol]