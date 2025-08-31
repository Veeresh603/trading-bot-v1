# backtest.py
import pandas as pd
import quantstats as qs

from config import Config, BacktestConfig
from core.data import HistoricalDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from core.utils import logger
import reports

def run_backtest():
    """
    Main function to run the event-driven backtest.
    """
    logger.info(f"--- Starting Backtest: {BacktestConfig.RUN_NAME} ---")
    logger.info(f"Configuration | Start: {BacktestConfig.START_DATE}, End: {BacktestConfig.END_DATE}, Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")

    # 1. Initialize Components
    data_handler = HistoricalDataHandler(
        symbols=Config.SYMBOLS_TO_TRADE,
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )

    strategy = AIStrategy(
        data_handler=data_handler,
        sequence_length=Config.SEQUENCE_LENGTH
    )

    execution_handler = SimulatedExecutionHandler(
        commission_per_trade=BacktestConfig.COMMISSION,
        slippage_pct=BacktestConfig.SLIPPAGE
    )

    portfolio = Portfolio(
        data_handler=data_handler,
        execution_handler=execution_handler,
        initial_capital=BacktestConfig.INITIAL_CAPITAL
    )

    # 2. Main Event Loop
    # This loop simulates the passage of time, bar by bar.
    while data_handler.continue_backtest:
        # A) Update all systems with the latest market data
        data_handler.update_bars()
        if not data_handler.continue_backtest:
            break

        portfolio.update_timeindex()

        # B) Let the strategy generate new trading signals
        signals = strategy.generate_signals()

        # C) The portfolio processes the signals and generates orders
        portfolio.process_signals(signals)

        # D) The portfolio processes any newly filled orders
        portfolio.process_fills()

    # 3. Post-Backtest Analysis & Reporting
    logger.info("--- Backtest Complete ---")
    results = portfolio.create_results_dataframe()
    
    # Save raw results and trades log to CSV
    results_path = f"backtest_results_{BacktestConfig.RUN_NAME}.csv"
    trades_path = f"trades_log_{BacktestConfig.RUN_NAME}.csv"
    results.to_csv(results_path)
    portfolio.trades_df.to_csv(trades_path)
    logger.info(f"Detailed backtest results saved to {results_path}")
    logger.info(f"Trade log saved to {trades_path}")
    
    # Generate and show the professional tearsheet report
    reports.generate_html_report(results['total_value'], trades_path, BacktestConfig.RUN_NAME)

    # Display key performance metrics
    reports.print_performance_summary(results['returns'])


if __name__ == "__main__":
    run_backtest()
