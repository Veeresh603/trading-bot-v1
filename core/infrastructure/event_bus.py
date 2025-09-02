# core/infrastructure/event_bus.py
import asyncio
import json
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import aio_pika
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    RISK_ALERT = "risk_alert"
    POSITION_UPDATED = "position_updated"
    STRATEGY_ALERT = "strategy_alert"
    SYSTEM_HEALTH = "system_health"
    MODEL_PREDICTION = "model_prediction"

@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    payload: Dict[str, Any]
    correlation_id: str
    source: str
    priority: int = 5

    def to_json(self) -> str:
        event_dict = asdict(self)
        event_dict['event_type'] = self.event_type.value
        event_dict['timestamp'] = self.timestamp.isoformat()
        return json.dumps(event_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        data = json.loads(json_str)
        data['event_type'] = EventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class IEventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None:
        pass

    @abstractmethod
    async def subscribe(self, event_type: EventType, handler: Callable) -> None:
        pass

    @abstractmethod
    async def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        pass

class RabbitMQEventBus(IEventBus):
    def __init__(self, connection_url: str = "amqp://guest:guest@localhost/"):
        self.connection_url = connection_url
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.queues: Dict[str, aio_pika.Queue] = {}

    async def connect(self):
        self.connection = await aio_pika.connect_robust(self.connection_url)
        self.channel = await self.connection.channel()
        self.exchange = await self.channel.declare_exchange(
            'trading_events', 
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        logger.info("Connected to RabbitMQ event bus")

    async def disconnect(self):
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ event bus")

    async def publish(self, event: Event) -> None:
        if not self.exchange:
            await self.connect()
        
        routing_key = f"trading.{event.event_type.value}"
        message = aio_pika.Message(
            body=event.to_json().encode(),
            priority=event.priority,
            correlation_id=event.correlation_id,
            timestamp=event.timestamp
        )
        
        await self.exchange.publish(message, routing_key=routing_key)
        logger.debug(f"Published event: {event.event_type.value}")

    async def subscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
        self.subscribers[event_type].append(handler)
        
        queue_name = f"queue_{event_type.value}_{handler.__name__}"
        if queue_name not in self.queues:
            queue = await self.channel.declare_queue(queue_name, durable=True)
            await queue.bind(self.exchange, routing_key=f"trading.{event_type.value}")
            self.queues[queue_name] = queue
            
            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    event = Event.from_json(message.body.decode())
                    for subscriber_handler in self.subscribers.get(event_type, []):
                        try:
                            await subscriber_handler(event)
                        except Exception as e:
                            logger.error(f"Error in event handler {subscriber_handler.__name__}: {e}")
            
            await queue.consume(process_message)
        
        logger.info(f"Subscribed {handler.__name__} to {event_type.value}")

    async def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Unsubscribed {handler.__name__} from {event_type.value}")

class InMemoryEventBus(IEventBus):
    """Lightweight in-memory event bus for testing and single-process scenarios"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._process_events())

    async def stop(self):
        self.running = False

    async def _process_events(self):
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue

    async def _dispatch_event(self, event: Event):
        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")

    async def publish(self, event: Event) -> None:
        await self.event_queue.put(event)
        logger.debug(f"Published event: {event.event_type.value}")

    async def subscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed {handler.__name__} to {event_type.value}")

    async def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.info(f"Unsubscribed {handler.__name__} from {event_type.value}")

class EventBusFactory:
    @staticmethod
    def create(bus_type: str = "inmemory", **kwargs) -> IEventBus:
        if bus_type == "rabbitmq":
            return RabbitMQEventBus(**kwargs)
        elif bus_type == "inmemory":
            return InMemoryEventBus(**kwargs)
        else:
            raise ValueError(f"Unknown event bus type: {bus_type}")