# core/domain/services/risk_manager.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import logging
from core.domain.entities.order import Order, OrderRequest
from core.domain.entities.position import Position
from core.infrastructure.event_bus import IEventBus, Event, EventType
import asyncio

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    portfolio_value: Decimal
    total_exposure: Decimal
    var_95: Decimal  # Value at Risk at 95% confidence
    cvar_95: Decimal  # Conditional VaR
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    position_concentration: Dict[str, Decimal]
    correlation_matrix: Optional[np.ndarray] = None
    kelly_fraction: Decimal = Decimal(0)
    portfolio_heat: Decimal = Decimal(0)  # Percentage of capital at risk
    
@dataclass
class RiskLimits:
    max_position_size: Decimal = Decimal("0.1")  # 10% of portfolio
    max_portfolio_heat: Decimal = Decimal("0.06")  # 6% total risk
    max_correlation: Decimal = Decimal("0.7")  # Max correlation between positions
    max_daily_loss: Decimal = Decimal("0.02")  # 2% daily loss limit
    max_drawdown: Decimal = Decimal("0.15")  # 15% max drawdown
    max_leverage: Decimal = Decimal("2.0")  # 2x leverage max
    min_liquidity_ratio: Decimal = Decimal("0.3")  # 30% cash minimum
    position_limit_per_symbol: int = 3  # Max positions per symbol
    daily_trade_limit: int = 50  # Max trades per day
    
@dataclass
class RiskCheck:
    passed: bool
    risk_level: RiskLevel
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class RiskManager:
    def __init__(self, event_bus: IEventBus = None, limits: RiskLimits = None):
        self.event_bus = event_bus
        self.limits = limits or RiskLimits()
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: Decimal = Decimal(0)
        self.daily_trades: int = 0
        self.last_reset: datetime = datetime.now()
        self.historical_returns: List[Decimal] = []
        self.portfolio_value: Decimal = Decimal("100000")  # Default
        self.high_water_mark: Decimal = self.portfolio_value
        self.circuit_breaker_active: bool = False
        
    def check_order_risk(self, order_request: OrderRequest) -> RiskCheck:
        """Comprehensive risk check for new order"""
        violations = []
        warnings = []
        suggestions = []
        
        # Position size check
        position_value = order_request.quantity * (order_request.price or Decimal(1))
        position_pct = position_value / self.portfolio_value
        
        if position_pct > self.limits.max_position_size:
            violations.append(f"Position size {position_pct:.2%} exceeds limit {self.limits.max_position_size:.2%}")
        elif position_pct > self.limits.max_position_size * Decimal("0.8"):
            warnings.append(f"Position size {position_pct:.2%} approaching limit")
        
        # Portfolio heat check
        current_heat = self._calculate_portfolio_heat()
        if current_heat > self.limits.max_portfolio_heat:
            violations.append(f"Portfolio heat {current_heat:.2%} exceeds limit {self.limits.max_portfolio_heat:.2%}")
        
        # Correlation check
        if order_request.symbol in self.positions:
            correlated_positions = self._find_correlated_positions(order_request.symbol)
            if correlated_positions:
                warnings.append(f"High correlation with existing positions: {correlated_positions}")
        
        # Daily loss limit check
        if self.daily_pnl < -self.limits.max_daily_loss * self.portfolio_value:
            violations.append(f"Daily loss limit reached: {self.daily_pnl}")
            self.circuit_breaker_active = True
        
        # Circuit breaker check
        if self.circuit_breaker_active:
            violations.append("Circuit breaker active - trading halted")
        
        # Trade frequency check
        if self.daily_trades >= self.limits.daily_trade_limit:
            violations.append(f"Daily trade limit reached: {self.daily_trades}")
        
        # Drawdown check
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.limits.max_drawdown:
            violations.append(f"Max drawdown exceeded: {current_drawdown:.2%}")
        elif current_drawdown > self.limits.max_drawdown * Decimal("0.8"):
            warnings.append(f"Approaching max drawdown: {current_drawdown:.2%}")
        
        # Leverage check
        total_exposure = self._calculate_total_exposure()
        leverage = total_exposure / self.portfolio_value
        if leverage > self.limits.max_leverage:
            violations.append(f"Leverage {leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x")
        
        # Liquidity check
        cash_ratio = self._calculate_liquidity_ratio()
        if cash_ratio < self.limits.min_liquidity_ratio:
            warnings.append(f"Low liquidity: {cash_ratio:.2%}")
        
        # Position concentration
        if len([p for p in self.positions.values() if p.symbol == order_request.symbol]) >= self.limits.position_limit_per_symbol:
            violations.append(f"Position limit reached for {order_request.symbol}")
        
        # Kelly Criterion suggestion
        kelly_size = self._calculate_kelly_position_size(order_request.symbol)
        if kelly_size > Decimal(0):
            suggestions.append(f"Kelly optimal size: {kelly_size:.0f} units")
        
        # Determine risk level
        if violations:
            risk_level = RiskLevel.CRITICAL
        elif len(warnings) > 2:
            risk_level = RiskLevel.HIGH
        elif warnings:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        passed = len(violations) == 0
        
        # Publish risk event if failed
        if not passed and self.event_bus:
            asyncio.create_task(self._publish_risk_alert(order_request, violations))
        
        return RiskCheck(
            passed=passed,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def calculate_position_size(self, symbol: str, entry_price: Decimal, stop_loss: Decimal) -> Decimal:
        """Calculate position size using risk-based sizing"""
        risk_per_trade = self.portfolio_value * Decimal("0.01")  # 1% risk per trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return Decimal(0)
        
        # Basic position size
        position_size = risk_per_trade / risk_per_share
        
        # Apply Kelly Criterion adjustment
        kelly_fraction = self._calculate_kelly_fraction(symbol)
        if kelly_fraction > 0:
            position_size *= min(kelly_fraction, Decimal("0.25"))  # Cap at 25% Kelly
        
        # Apply volatility adjustment
        volatility_adj = self._calculate_volatility_adjustment(symbol)
        position_size *= volatility_adj
        
        # Apply correlation adjustment
        correlation_adj = self._calculate_correlation_adjustment(symbol)
        position_size *= correlation_adj
        
        # Round to nearest lot size
        return Decimal(int(position_size))
    
    def calculate_var(self, confidence_level: Decimal = Decimal("0.95")) -> Decimal:
        """Calculate Value at Risk"""
        if not self.historical_returns:
            return Decimal(0)
        
        returns = np.array([float(r) for r in self.historical_returns])
        var_percentile = float((1 - confidence_level) * 100)
        var = np.percentile(returns, var_percentile)
        
        return Decimal(str(var)) * self.portfolio_value
    
    def calculate_cvar(self, confidence_level: Decimal = Decimal("0.95")) -> Decimal:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(confidence_level)
        
        if not self.historical_returns:
            return Decimal(0)
        
        returns = np.array([float(r) for r in self.historical_returns])
        var_threshold = float(var / self.portfolio_value)
        
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return var
        
        cvar = np.mean(tail_returns)
        return Decimal(str(cvar)) * self.portfolio_value
    
    def update_position(self, position: Position):
        """Update position tracking"""
        self.positions[position.position_id] = position
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        # Check for risk breaches
        self._check_risk_breaches()
    
    def _calculate_portfolio_heat(self) -> Decimal:
        """Calculate total portfolio risk exposure"""
        total_risk = Decimal(0)
        
        for position in self.positions.values():
            if position.stop_loss:
                position_risk = abs(position.current_price - position.stop_loss) * position.quantity
                total_risk += position_risk
        
        return total_risk / self.portfolio_value if self.portfolio_value > 0 else Decimal(0)
    
    def _calculate_total_exposure(self) -> Decimal:
        """Calculate total market exposure"""
        return sum(p.market_value() for p in self.positions.values())
    
    def _calculate_current_drawdown(self) -> Decimal:
        """Calculate current drawdown from high water mark"""
        if self.high_water_mark == 0:
            return Decimal(0)
        
        return (self.high_water_mark - self.portfolio_value) / self.high_water_mark
    
    def _calculate_liquidity_ratio(self) -> Decimal:
        """Calculate available liquidity ratio"""
        total_exposure = self._calculate_total_exposure()
        available_cash = self.portfolio_value - total_exposure
        
        return available_cash / self.portfolio_value if self.portfolio_value > 0 else Decimal(0)
    
    def _find_correlated_positions(self, symbol: str) -> List[str]:
        """Find positions highly correlated with given symbol"""
        # Simplified correlation check - in production, use actual correlation matrix
        correlated = []
        
        # Check for same sector/similar symbols
        for position in self.positions.values():
            if position.symbol != symbol:
                # Simple heuristic - same prefix means correlated
                if position.symbol[:3] == symbol[:3]:
                    correlated.append(position.symbol)
        
        return correlated
    
    def _calculate_kelly_fraction(self, symbol: str) -> Decimal:
        """Calculate Kelly Criterion position sizing"""
        # Simplified Kelly - in production, use actual win rate and payoff ratio
        win_rate = Decimal("0.55")  # Example: 55% win rate
        avg_win = Decimal("1.5")  # Example: 1.5R average win
        avg_loss = Decimal("1.0")  # Example: 1R average loss
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = payoff ratio
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (p * b - q) / b
        
        # Apply Kelly fraction with safety factor
        return max(Decimal(0), kelly * Decimal("0.25"))  # Use 25% Kelly for safety
    
    def _calculate_kelly_position_size(self, symbol: str) -> Decimal:
        """Calculate position size using Kelly Criterion"""
        kelly_fraction = self._calculate_kelly_fraction(symbol)
        return self.portfolio_value * kelly_fraction / Decimal("100")  # Assume $100 per unit
    
    def _calculate_volatility_adjustment(self, symbol: str) -> Decimal:
        """Adjust position size based on volatility"""
        # Simplified - in production, calculate actual volatility
        base_volatility = Decimal("0.02")  # 2% daily volatility baseline
        current_volatility = Decimal("0.03")  # Example current volatility
        
        return base_volatility / current_volatility
    
    def _calculate_correlation_adjustment(self, symbol: str) -> Decimal:
        """Adjust position size based on portfolio correlation"""
        correlated_positions = self._find_correlated_positions(symbol)
        
        if not correlated_positions:
            return Decimal("1.0")
        
        # Reduce size based on number of correlated positions
        reduction_factor = Decimal("0.8") ** len(correlated_positions)
        return max(Decimal("0.5"), reduction_factor)  # At least 50% size
    
    def _update_portfolio_metrics(self):
        """Update portfolio-wide risk metrics"""
        # Update portfolio value
        total_value = sum(p.market_value() for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        self.portfolio_value = self.portfolio_value + unrealized_pnl
        
        # Update high water mark
        if self.portfolio_value > self.high_water_mark:
            self.high_water_mark = self.portfolio_value
        
        # Reset daily metrics if needed
        if datetime.now().date() > self.last_reset.date():
            self.daily_pnl = Decimal(0)
            self.daily_trades = 0
            self.last_reset = datetime.now()
            self.circuit_breaker_active = False
    
    def _check_risk_breaches(self):
        """Check for risk limit breaches and trigger alerts"""
        breaches = []
        
        # Check drawdown
        if self._calculate_current_drawdown() > self.limits.max_drawdown:
            breaches.append("Maximum drawdown exceeded")
        
        # Check portfolio heat
        if self._calculate_portfolio_heat() > self.limits.max_portfolio_heat:
            breaches.append("Portfolio heat limit exceeded")
        
        # Check leverage
        leverage = self._calculate_total_exposure() / self.portfolio_value
        if leverage > self.limits.max_leverage:
            breaches.append(f"Leverage limit exceeded: {leverage:.2f}x")
        
        if breaches and self.event_bus:
            asyncio.create_task(self._publish_risk_breach(breaches))
    
    async def _publish_risk_alert(self, order_request: OrderRequest, violations: List[str]):
        """Publish risk alert event"""
        event = Event(
            event_type=EventType.RISK_ALERT,
            timestamp=datetime.now(),
            payload={
                'alert_type': 'ORDER_RISK_CHECK_FAILED',
                'order_request': {
                    'symbol': order_request.symbol,
                    'side': order_request.side.value,
                    'quantity': str(order_request.quantity)
                },
                'violations': violations
            },
            correlation_id=f"risk_check_{datetime.now().timestamp()}",
            source="RiskManager",
            priority=8
        )
        await self.event_bus.publish(event)
    
    async def _publish_risk_breach(self, breaches: List[str]):
        """Publish risk breach event"""
        event = Event(
            event_type=EventType.RISK_ALERT,
            timestamp=datetime.now(),
            payload={
                'alert_type': 'RISK_LIMIT_BREACH',
                'breaches': breaches,
                'portfolio_value': str(self.portfolio_value),
                'current_drawdown': str(self._calculate_current_drawdown())
            },
            correlation_id=f"risk_breach_{datetime.now().timestamp()}",
            source="RiskManager",
            priority=9
        )
        await self.event_bus.publish(event)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        position_concentration = {}
        for position in self.positions.values():
            if position.symbol not in position_concentration:
                position_concentration[position.symbol] = Decimal(0)
            position_concentration[position.symbol] += position.market_value() / self.portfolio_value
        
        return RiskMetrics(
            portfolio_value=self.portfolio_value,
            total_exposure=self._calculate_total_exposure(),
            var_95=self.calculate_var(),
            cvar_95=self.calculate_cvar(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            max_drawdown=self.limits.max_drawdown,
            current_drawdown=self._calculate_current_drawdown(),
            position_concentration=position_concentration,
            portfolio_heat=self._calculate_portfolio_heat()
        )
    
    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio"""
        if len(self.historical_returns) < 30:
            return Decimal(0)
        
        returns = np.array([float(r) for r in self.historical_returns])
        risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
        
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return Decimal(0)
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return Decimal(str(sharpe))