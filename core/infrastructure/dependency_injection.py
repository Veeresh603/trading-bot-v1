# core/infrastructure/dependency_injection.py
from typing import Dict, Type, Any, Callable, Optional, List
from abc import ABC
import inspect
from functools import wraps
import asyncio
from dataclasses import dataclass
from enum import Enum

class Scope(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceDescriptor:
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: Scope = Scope.SINGLETON
    dependencies: List[Type] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class ServiceContainer:
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        
    def register_singleton(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """Register a singleton service"""
        self._register(service_type, implementation, factory, Scope.SINGLETON)
        
    def register_transient(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """Register a transient service (new instance each time)"""
        self._register(service_type, implementation, factory, Scope.TRANSIENT)
        
    def register_scoped(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """Register a scoped service (one instance per scope)"""
        self._register(service_type, implementation, factory, Scope.SCOPED)
        
    def _register(self, service_type: Type, implementation: Type, factory: Callable, scope: Scope):
        if implementation is None and factory is None:
            implementation = service_type
            
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            scope=scope
        )
        
        # Extract dependencies from constructor
        if implementation:
            sig = inspect.signature(implementation.__init__)
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    descriptor.dependencies.append(param.annotation)
                    
        self._services[service_type] = descriptor
        
    def resolve(self, service_type: Type) -> Any:
        """Resolve a service and its dependencies"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")
            
        return self._resolve_service(service_type)
        
    def _resolve_service(self, service_type: Type) -> Any:
        descriptor = self._services[service_type]
        
        # Check for existing instance based on scope
        if descriptor.scope == Scope.SINGLETON and descriptor.instance:
            return descriptor.instance
        elif descriptor.scope == Scope.SCOPED and service_type in self._scoped_instances:
            return self._scoped_instances[service_type]
            
        # Resolve dependencies
        dependencies = {}
        for dep_type in descriptor.dependencies:
            dependencies[dep_type.__name__.lower()] = self.resolve(dep_type)
            
        # Create instance
        if descriptor.factory:
            instance = descriptor.factory(**dependencies)
        elif descriptor.implementation:
            instance = descriptor.implementation(**dependencies)
        else:
            raise ValueError(f"No implementation or factory for {service_type}")
            
        # Store instance based on scope
        if descriptor.scope == Scope.SINGLETON:
            descriptor.instance = instance
        elif descriptor.scope == Scope.SCOPED:
            self._scoped_instances[service_type] = instance
            
        return instance
        
    def create_scope(self):
        """Create a new scope context"""
        return ServiceScope(self)
        
    def clear_scope(self):
        """Clear scoped instances"""
        self._scoped_instances.clear()

class ServiceScope:
    def __init__(self, container: ServiceContainer):
        self.container = container
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.clear_scope()

def inject(container: ServiceContainer):
    """Decorator for dependency injection"""
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Get the signature of the original __init__
            sig = inspect.signature(original_init)
            
            # Inject dependencies
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    if param_name not in kwargs:
                        try:
                            kwargs[param_name] = container.resolve(param.annotation)
                        except ValueError:
                            pass  # Dependency not registered, use default or provided value
                            
            original_init(self, *args, **kwargs)
            
        cls.__init__ = new_init
        return cls
        
    return decorator

class IService(ABC):
    """Base interface for services"""
    pass

# Global container instance
_container = ServiceContainer()

def get_container() -> ServiceContainer:
    """Get the global service container"""
    return _container

def register_services(container: ServiceContainer):
    """Register all services in the container"""
    from core.infrastructure.event_bus import IEventBus, EventBusFactory
    from core.domain.services.risk_manager import RiskManager
    from core.domain.services.position_manager import PositionManager
    from core.domain.services.order_manager import OrderManager
    from core.domain.services.market_data_service import MarketDataService
    from core.domain.services.strategy_engine import StrategyEngine
    from core.domain.services.execution_engine import ExecutionEngine
    from core.infrastructure.monitoring import MonitoringService
    from core.infrastructure.cache import CacheService
    
    # Register infrastructure services
    container.register_singleton(
        IEventBus,
        factory=lambda: EventBusFactory.create("inmemory")
    )
    
    container.register_singleton(CacheService)
    container.register_singleton(MonitoringService)
    
    # Register domain services
    container.register_singleton(RiskManager)
    container.register_singleton(PositionManager)
    container.register_singleton(OrderManager)
    container.register_singleton(MarketDataService)
    container.register_singleton(StrategyEngine)
    container.register_singleton(ExecutionEngine)