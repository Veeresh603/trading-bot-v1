# backtest.py
import argparse
import pandas as pd
from datetime import datetime
import asyncio

from config import Config, BacktestConfig
from core.data import HistoricalDataHandler
from core.infrastructure.dependency_injection import get_container, register_services
from core.infrastructure.event_bus import IEventBus, Event, EventType
from core.domain.services.portfolio_manager import PortfolioManager
from core.utils import logger, telegram
import reports
from decimal import Decimal

def run_backtest(strategy_name: str, symbol: str):
    """
    Main function to run the event-driven backtest, fully utilizing the domain services.
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {symbol} ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")

    # --- Dependency Injection and Service Setup ---
    container = get_container()
    register_services(container)

    event_bus = container.resolve(IEventBus)
    portfolio_manager = container.resolve(PortfolioManager)
    # The other services (StrategyEngine, RiskManager, etc.) are also initialized
    # and will subscribe to events automatically.

    # --- Data Loading ---
    data_handler = HistoricalDataHandler(
        symbols=[symbol],
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )
    
    if not data_handler.symbol_data:
        logger.error(f"No historical data could be fetched for {symbol}.")
        telegram.send_message(f"❌ *Backtest Failed*\n\nNo historical data found for {symbol}.")
        return

    all_dates = sorted(data_handler.symbol_data[symbol].index)
    
    logger.info(f"Backtesting over a total of {len(all_dates)} market timestamps.")

    # --- Event Loop Simulation ---
    async def event_loop():
        # Start the in-memory event bus processor
        if hasattr(event_bus, 'start'):
            await event_bus.start()

        equity_curve = {}
        for bar_datetime in all_dates:
            current_bar = data_handler.symbol_data[symbol].loc[bar_datetime]
            
            # 1. Publish Market Data Event
            market_data_event = Event(
                event_type=EventType.MARKET_DATA,
                timestamp=bar_datetime.to_pydatetime(),
                payload={'symbol': symbol, 'bar': current_bar},
                correlation_id=f"market_data_{bar_datetime.timestamp()}",
                source="Backtester"
            )
            await event_bus.publish(market_data_event)
            
            # In an in-memory setup, the event is processed immediately by subscribers.
            # In a real distributed system (like RabbitMQ), this would be asynchronous.
            
            # Update portfolio equity at the end of each bar for reporting
            portfolio_manager.update_equity()
            equity_curve[bar_datetime] = portfolio_manager.equity

        # Stop the event bus
        if hasattr(event_bus, 'stop'):
            await event_bus.stop()
            
        return pd.Series(equity_curve)

    equity_curve = asyncio.run(event_loop())

    logger.info("--- Backtest Complete. Generating Performance Report... ---")

    final_equity = portfolio_manager.equity
    total_return = (final_equity / Decimal(BacktestConfig.INITIAL_CAPITAL) - 1) * 100
    
    logger.info(f"Final Portfolio Equity: ${final_equity:,.2f}")
    logger.info(f"Total Return: {float(total_return):.2f}%")

    # The DataRecorder service has logged all trades to its file.
    # You can enhance the reports.py module to read from this log file.

    summary_message = (
        f"✅ *Backtest Complete*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Final Equity:* `${final_equity:,.2f}`\n"
        f"*Total Return:* `{float(total_return):.2f}%`\n"
    )
    telegram.send_message(summary_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Event-Driven Backtester")
    parser.add_argument("--strategy", type=str, default="AIStrategy", help="The strategy to backtest.")
    parser.add_argument("--symbol", type=str, default=Config.UNDERLYING_SYMBOL, help="The symbol to backtest (e.g., 'NIFTY 50').")
    args = parser.parse_args()
        
    run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol
    )