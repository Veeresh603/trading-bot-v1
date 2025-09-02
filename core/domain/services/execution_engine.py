# core/domain/services/execution_engine.py
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np
import logging
from core.domain.entities.order import Order, OrderType, OrderSide, TimeInForce
from core.infrastructure.event_bus import IEventBus, Event, EventType

logger = logging.getLogger(__name__)

class ExecutionAlgorithm(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"  # Percentage of Volume
    ICEBERG = "ICEBERG"
    SNIPER = "SNIPER"
    PAIRS = "PAIRS"
    SMART = "SMART"

@dataclass
class ExecutionParams:
    algorithm: ExecutionAlgorithm
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    participation_rate: Decimal = Decimal("0.1")  # 10% of volume
    urgency: int = 5  # 1-10 scale
    max_spread: Optional[Decimal] = None
    slice_size: Optional[int] = None
    min_fill_size: Optional[int] = None
    price_improvement: bool = True
    dark_pool_enabled: bool = False
    
@dataclass
class MarketMicrostructure:
    bid: Decimal
    ask: Decimal
    spread: Decimal
    mid_price: Decimal
    bid_size: int
    ask_size: int
    order_imbalance: Decimal
    volatility: Decimal
    average_trade_size: int
    tick_size: Decimal
    
@dataclass
class ExecutionReport:
    order_id: str
    algorithm: ExecutionAlgorithm
    slices_sent: int
    slices_filled: int
    average_fill_price: Decimal
    vwap: Decimal
    implementation_shortfall: Decimal
    market_impact: Decimal
    timing_cost: Decimal
    total_commission: Decimal
    fill_rate: Decimal
    adverse_selection: Decimal

class SmartOrderRouter:
    """Routes orders to best execution venue"""
    
    def __init__(self):
        self.venues = ["NSE", "BSE", "DARK_POOL"]
        self.venue_latencies = {"NSE": 1, "BSE": 2, "DARK_POOL": 5}  # ms
        self.venue_fees = {"NSE": Decimal("0.0001"), "BSE": Decimal("0.00012"), "DARK_POOL": Decimal("0.00008")}
        
    def route_order(self, order: Order, microstructure: MarketMicrostructure) -> str:
        """Determine best venue for order execution"""
        best_venue = "NSE"  # Default
        
        # Large orders may benefit from dark pools
        if order.quantity > 10000 and "DARK_POOL" in self.venues:
            best_venue = "DARK_POOL"
        # For latency-sensitive orders
        elif order.metadata.get("urgency", 5) > 8:
            best_venue = min(self.venue_latencies, key=self.venue_latencies.get)
        # For cost-sensitive orders
        else:
            best_venue = min(self.venue_fees, key=self.venue_fees.get)
            
        return best_venue

class TWAPExecutor:
    """Time-Weighted Average Price execution"""
    
    def __init__(self, event_bus: IEventBus):
        self.event_bus = event_bus
        
    async def execute(self, order: Order, params: ExecutionParams, market_data_provider: Callable):
        """Execute order using TWAP algorithm"""
        if not params.start_time or not params.end_time:
            params.start_time = datetime.now()
            params.end_time = datetime.now() + timedelta(hours=1)
            
        duration = (params.end_time - params.start_time).total_seconds()
        num_slices = max(10, int(duration / 60))  # One slice per minute, minimum 10
        slice_size = order.quantity / num_slices
        slice_interval = duration / num_slices
        
        filled_quantity = Decimal(0)
        total_value = Decimal(0)
        
        for i in range(num_slices):
            if filled_quantity >= order.quantity:
                break
                
            # Get current market data
            market_data = await market_data_provider(order.symbol)
            
            # Create child order
            child_order = Order(
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=min(slice_size, order.remaining_quantity()),
                price=market_data.mid_price,
                time_in_force=TimeInForce.IOC
            )
            
            # Simulate execution (in production, send to broker)
            fill_price = await self._simulate_fill(child_order, market_data)
            if fill_price:
                filled_quantity += child_order.quantity
                total_value += child_order.quantity * fill_price
                
                # Publish execution event
                await self._publish_slice_execution(order, child_order, fill_price)
            
            # Wait for next slice
            await asyncio.sleep(slice_interval)
        
        # Update order with final execution details
        if filled_quantity > 0:
            order.filled_quantity = filled_quantity
            order.average_fill_price = total_value / filled_quantity
    
    async def _simulate_fill(self, order: Order, market_data: MarketMicrostructure) -> Optional[Decimal]:
        """Simulate order fill (replace with actual broker execution)"""
        # Simple simulation - fill at mid price with small slippage
        slippage = market_data.spread * Decimal("0.1") * (1 if order.side == OrderSide.BUY else -1)
        fill_price = market_data.mid_price + slippage
        return fill_price
    
    async def _publish_slice_execution(self, parent_order: Order, child_order: Order, fill_price: Decimal):
        """Publish slice execution event"""
        event = Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            payload={
                'parent_order_id': parent_order.order_id,
                'child_order_id': child_order.order_id,
                'symbol': child_order.symbol,
                'quantity': str(child_order.quantity),
                'fill_price': str(fill_price),
                'algorithm': 'TWAP'
            },
            correlation_id=parent_order.order_id,
            source="TWAPExecutor"
        )
        await self.event_bus.publish(event)

class VWAPExecutor:
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, event_bus: IEventBus):
        self.event_bus = event_bus
        self.historical_volume_profile = {}
        
    async def execute(self, order: Order, params: ExecutionParams, market_data_provider: Callable):
        """Execute order using VWAP algorithm"""
        # Get historical volume profile
        volume_profile = await self._get_volume_profile(order.symbol)
        
        total_expected_volume = sum(volume_profile.values())
        filled_quantity = Decimal(0)
        total_value = Decimal(0)
        
        for time_bucket, volume_pct in volume_profile.items():
            if filled_quantity >= order.quantity:
                break
            
            # Calculate slice size based on volume profile
            slice_size = order.quantity * Decimal(str(volume_pct))
            
            # Get current market data
            market_data = await market_data_provider(order.symbol)
            
            # Adjust for participation rate
            max_participation = market_data.average_trade_size * params.participation_rate
            slice_size = min(slice_size, max_participation)
            
            # Create and execute child order
            child_order = Order(
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=min(slice_size, order.remaining_quantity()),
                price=self._calculate_limit_price(market_data, order.side),
                time_in_force=TimeInForce.IOC
            )
            
            fill_price = await self._execute_slice(child_order, market_data)
            if fill_price:
                filled_quantity += child_order.quantity
                total_value += child_order.quantity * fill_price
                
                await self._publish_slice_execution(order, child_order, fill_price)
            
            # Wait for next time bucket
            await asyncio.sleep(60)  # 1 minute buckets
        
        if filled_quantity > 0:
            order.filled_quantity = filled_quantity
            order.average_fill_price = total_value / filled_quantity
    
    async def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get historical intraday volume profile"""
        # Simplified - in production, use actual historical data
        # Returns hour -> volume percentage mapping
        profile = {}
        total = 0
        
        # Typical U-shaped volume profile
        hourly_volumes = [0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.06, 0.08, 0.12, 0.15]
        
        for hour, volume in enumerate(hourly_volumes, start=9):
            profile[hour] = volume
            
        return profile
    
    def _calculate_limit_price(self, market_data: MarketMicrostructure, side: OrderSide) -> Decimal:
        """Calculate aggressive limit price"""
        if side == OrderSide.BUY:
            # Price at or slightly above ask for aggressive execution
            return market_data.ask + market_data.tick_size
        else:
            # Price at or slightly below bid for aggressive execution
            return market_data.bid - market_data.tick_size
    
    async def _execute_slice(self, order: Order, market_data: MarketMicrostructure) -> Optional[Decimal]:
        """Execute a slice of the order"""
        # Simulate execution with market impact
        impact = self._calculate_market_impact(order.quantity, market_data)
        
        if order.side == OrderSide.BUY:
            fill_price = market_data.ask * (Decimal(1) + impact)
        else:
            fill_price = market_data.bid * (Decimal(1) - impact)
            
        return fill_price
    
    def _calculate_market_impact(self, quantity: Decimal, market_data: MarketMicrostructure) -> Decimal:
        """Calculate expected market impact"""
        # Simplified square-root model
        avg_daily_volume = Decimal("1000000")  # Example
        participation = quantity / avg_daily_volume
        
        # Impact = volatility * sqrt(participation)
        impact = market_data.volatility * Decimal(str(np.sqrt(float(participation))))
        
        return min(impact, Decimal("0.005"))  # Cap at 0.5%
    
    async def _publish_slice_execution(self, parent_order: Order, child_order: Order, fill_price: Decimal):
        """Publish slice execution event"""
        event = Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            payload={
                'parent_order_id': parent_order.order_id,
                'child_order_id': child_order.order_id,
                'symbol': child_order.symbol,
                'quantity': str(child_order.quantity),
                'fill_price': str(fill_price),
                'algorithm': 'VWAP'
            },
            correlation_id=parent_order.order_id,
            source="VWAPExecutor"
        )
        await self.event_bus.publish(event)

class AdverseSelectionDetector:
    """Detects adverse selection in order execution"""
    
    def __init__(self):
        self.fill_history: List[Dict] = []
        self.detection_window = 100  # Number of fills to analyze
        
    def analyze_fill(self, order: Order, fill_price: Decimal, market_price_after: Decimal) -> Decimal:
        """Analyze fill for adverse selection"""
        # Calculate price movement after fill
        if order.side == OrderSide.BUY:
            adverse_move = (fill_price - market_price_after) / fill_price
        else:
            adverse_move = (market_price_after - fill_price) / fill_price
            
        # Store fill history
        self.fill_history.append({
            'timestamp': datetime.now(),
            'side': order.side,
            'fill_price': fill_price,
            'market_price_after': market_price_after,
            'adverse_move': adverse_move
        })
        
        # Keep only recent history
        if len(self.fill_history) > self.detection_window:
            self.fill_history = self.fill_history[-self.detection_window:]
        
        return adverse_move
    
    def get_adverse_selection_score(self) -> Decimal:
        """Calculate overall adverse selection score"""
        if not self.fill_history:
            return Decimal(0)
            
        total_adverse = sum(f['adverse_move'] for f in self.fill_history)
        return Decimal(str(total_adverse / len(self.fill_history)))

class ExecutionEngine:
    """Main execution engine coordinating all execution algorithms"""
    
    def __init__(self, event_bus: IEventBus):
        self.event_bus = event_bus
        self.smart_router = SmartOrderRouter()
        self.twap_executor = TWAPExecutor(event_bus)
        self.vwap_executor = VWAPExecutor(event_bus)
        self.adverse_detector = AdverseSelectionDetector()
        self.active_executions: Dict[str, asyncio.Task] = {}
        
    async def execute_order(self, order: Order, params: ExecutionParams, market_data_provider: Callable):
        """Execute order using specified algorithm"""
        logger.info(f"Executing order {order.order_id} using {params.algorithm.value}")
        
        # Route order to best venue
        market_data = await market_data_provider(order.symbol)
        microstructure = self._create_microstructure(market_data)
        venue = self.smart_router.route_order(order, microstructure)
        order.venue = venue
        
        # Execute based on algorithm
        if params.algorithm == ExecutionAlgorithm.TWAP:
            task = asyncio.create_task(
                self.twap_executor.execute(order, params, market_data_provider)
            )
        elif params.algorithm == ExecutionAlgorithm.VWAP:
            task = asyncio.create_task(
                self.vwap_executor.execute(order, params, market_data_provider)
            )
        elif params.algorithm == ExecutionAlgorithm.MARKET:
            task = asyncio.create_task(
                self._execute_market_order(order, market_data_provider)
            )
        else:
            # Default to limit order
            task = asyncio.create_task(
                self._execute_limit_order(order, market_data_provider)
            )
        
        self.active_executions[order.order_id] = task
        
        # Wait for execution to complete
        await task
        
        # Analyze execution quality
        report = await self._generate_execution_report(order, params)
        await self._publish_execution_report(report)
        
        return report
    
    async def _execute_market_order(self, order: Order, market_data_provider: Callable):
        """Execute market order immediately"""
        market_data = await market_data_provider(order.symbol)
        microstructure = self._create_microstructure(market_data)
        
        # Simulate immediate fill at market price
        if order.side == OrderSide.BUY:
            fill_price = microstructure.ask
        else:
            fill_price = microstructure.bid
            
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        
        # Check for adverse selection
        await asyncio.sleep(1)  # Wait briefly
        market_data_after = await market_data_provider(order.symbol)
        adverse_move = self.adverse_detector.analyze_fill(
            order, fill_price, market_data_after['close']
        )
        
        logger.info(f"Market order {order.order_id} filled at {fill_price}, adverse move: {adverse_move:.4f}")
    
    async def _execute_limit_order(self, order: Order, market_data_provider: Callable):
        """Execute limit order"""
        max_wait = 300  # 5 minutes max wait
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait:
            market_data = await market_data_provider(order.symbol)
            microstructure = self._create_microstructure(market_data)
            
            # Check if limit price is marketable
            if order.side == OrderSide.BUY and order.price >= microstructure.ask:
                order.filled_quantity = order.quantity
                order.average_fill_price = microstructure.ask
                break
            elif order.side == OrderSide.SELL and order.price <= microstructure.bid:
                order.filled_quantity = order.quantity
                order.average_fill_price = microstructure.bid
                break
                
            await asyncio.sleep(1)
    
    def _create_microstructure(self, market_data: dict) -> MarketMicrostructure:
        """Create market microstructure from market data"""
        # Simplified - in production, get from order book
        close_price = Decimal(str(market_data.get('close', 100)))
        spread = close_price * Decimal("0.001")  # 0.1% spread
        
        return MarketMicrostructure(
            bid=close_price - spread/2,
            ask=close_price + spread/2,
            spread=spread,
            mid_price=close_price,
            bid_size=1000,
            ask_size=1000,
            order_imbalance=Decimal(0),
            volatility=Decimal("0.02"),
            average_trade_size=100,
            tick_size=Decimal("0.01")
        )
    
    async def _generate_execution_report(self, order: Order, params: ExecutionParams) -> ExecutionReport:
        """Generate execution quality report"""
        # Calculate execution metrics
        vwap = order.average_fill_price  # Simplified
        implementation_shortfall = Decimal(0)  # Requires arrival price
        market_impact = Decimal("0.001")  # Simplified
        
        return ExecutionReport(
            order_id=order.order_id,
            algorithm=params.algorithm,
            slices_sent=1,
            slices_filled=1,
            average_fill_price=order.average_fill_price,
            vwap=vwap,
            implementation_shortfall=implementation_shortfall,
            market_impact=market_impact,
            timing_cost=Decimal(0),
            total_commission=Decimal("10"),  # Example
            fill_rate=order.filled_quantity / order.quantity if order.quantity > 0 else Decimal(0),
            adverse_selection=self.adverse_detector.get_adverse_selection_score()
        )
    
    async def _publish_execution_report(self, report: ExecutionReport):
        """Publish execution report"""
        event = Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            payload={
                'order_id': report.order_id,
                'algorithm': report.algorithm.value,
                'average_fill_price': str(report.average_fill_price),
                'fill_rate': str(report.fill_rate),
                'market_impact': str(report.market_impact),
                'adverse_selection': str(report.adverse_selection)
            },
            correlation_id=report.order_id,
            source="ExecutionEngine"
        )
        await self.event_bus.publish(event)
    
    def cancel_execution(self, order_id: str):
        """Cancel ongoing execution"""
        if order_id in self.active_executions:
            self.active_executions[order_id].cancel()
            del self.active_executions[order_id]
            logger.info(f"Cancelled execution for order {order_id}")