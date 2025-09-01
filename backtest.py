# backtest.py
import argparse
from config import Config, BacktestConfig
from core.data import HistoricalDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from core.utils import logger, telegram
import reports

def run_backtest(strategy_name: str, symbol: str):
    """
    Main function to run the event-driven backtest.
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {symbol} ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")
    
    # --- Send Start Notification ---
    telegram.send_message(
        f"🚀 *Starting Backtest*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Period:* `{BacktestConfig.START_DATE}` to `{BacktestConfig.END_DATE}`"
    )

    # --- Initialization ---
    data_handler = HistoricalDataHandler(
        symbols=[symbol],
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe='1d' # Backtesting on daily data for long periods
    )
    
    historical_data = data_handler.symbol_data.get(symbol)
    if historical_data is None or historical_data.empty:
        logger.error(f"No historical data found for {symbol} in the given date range.")
        telegram.send_message(f"❌ *Backtest Failed*\n\nNo historical data found for `{symbol}`.")
        return

    portfolio = Portfolio(
        initial_capital=BacktestConfig.INITIAL_CAPITAL
    )
    
    execution_handler = SimulatedExecutionHandler(
        commission_bps=BacktestConfig.COMMISSION_BPS,
        slippage_bps=BacktestConfig.SLIPPAGE_BPS
    )

    strategy = AIStrategy(
        data_handler=data_handler,
        sequence_length=Config.SEQUENCE_LENGTH
    )
    
    logger.info(f"Loaded {len(historical_data)} data points for backtesting.")

    # --- Event Loop ---
    for bar_datetime, bar_data in historical_data.iterrows():
        # 1. Update portfolio with the latest market price
        portfolio.update_market_data(bar_datetime, {symbol: bar_data})

        # 2. Strategy generates a signal based on the new data
        # --- FIX APPLIED HERE ---
        # Changed 'generate_signal' to 'generate_signals' to match the method name in the AIStrategy class.
        signal = strategy.generate_signals(symbol, bar_datetime)

        # 3. If a signal is generated, the portfolio creates an order
        if signal:
            order = portfolio.create_order_from_signal(signal, bar_data)
            # 4. The execution handler fills the order
            if order:
                execution_handler.execute_order(order, bar_data, portfolio)

    # --- Post-Backtest Analysis ---
    logger.info("--- Backtest Complete. Generating Performance Report... ---")
    
    # Final portfolio stats
    final_equity = portfolio.holdings['total']
    total_return = (final_equity / BacktestConfig.INITIAL_CAPITAL - 1) * 100
    total_trades = portfolio.trade_count
    
    logger.info(f"Final Portfolio Equity: ${final_equity:,.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades Executed: {total_trades}")

    # Generate and save detailed report using QuantStats
    output_filename = f"report_{strategy_name}_{symbol}.html"
    reports.generate_html_report(portfolio, output_filename)
    logger.info(f"✅ Detailed performance report saved to '{output_filename}'")

    # --- Send Completion Notification ---
    summary_message = (
        f"✅ *Backtest Complete*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Final Equity:* `${final_equity:,.2f}`\n"
        f"*Total Return:* `{total_return:.2f}%`\n"
        f"*Total Trades:* `{total_trades}`\n\n"
        f"📈 Report saved to `{output_filename}`"
    )
    telegram.send_message(summary_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Event-Driven Backtester")
    parser.add_argument("--strategy", type=str, default="lstm", help="The strategy to backtest.")
    parser.add_argument("--symbol", type=str, required=True, help="The trading symbol to backtest (e.g., 'RELIANCE.NS').")
    args = parser.parse_args()

    run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol
    )

