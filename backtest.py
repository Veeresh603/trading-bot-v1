# backtest.py
import argparse
from config import Config, BacktestConfig
from core.data import HistoricalDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from core.utils import logger, telegram
import reports
import pandas as pd
from datetime import datetime

def run_backtest(strategy_name: str, symbol: str):
    """
    Main function to run the event-driven backtest.
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {symbol} ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")
    
    telegram.send_message(
        f"🚀 *Starting Backtest*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Period:* `{BacktestConfig.START_DATE}` to `{BacktestConfig.END_DATE}`"
    )

    data_handler = HistoricalDataHandler(
        symbols=[symbol],
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe='1d'
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

    for bar_datetime, bar_data in historical_data.iterrows():
        portfolio.update_market_data(bar_datetime, {symbol: bar_data})
        
        signal = strategy.generate_signals(symbol, bar_datetime)

        if signal and signal.get('direction') != 'HOLD':
            order = portfolio.create_order_from_signal(signal, bar_data)
            if order:
                execution_handler.execute_order(order, bar_data, portfolio)

    logger.info("--- Backtest Complete. Generating Performance Report... ---")
    
    final_equity = portfolio.holdings['total']
    total_return = (final_equity / BacktestConfig.INITIAL_CAPITAL - 1) * 100
    total_trades = portfolio.trade_count
    
    logger.info(f"Final Portfolio Equity: ${final_equity:,.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades Executed: {total_trades}")
    
    trades_df = pd.DataFrame(portfolio.trades)
    trades_filepath = f"trades_{strategy_name}_{symbol}.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_filepath, index=False)
        returns = portfolio.equity_curve.pct_change().dropna()
        reports.generate_html_report(
            returns=returns,
            trades_filepath=trades_filepath,
            run_name=f"{strategy_name}_{symbol}"
        )
    else:
        logger.warning("No trades executed during backtest. Skipping report generation.")

    summary_message = (
        f"✅ *Backtest Complete*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Final Equity:* `${final_equity:,.2f}`\n"
        f"*Total Return:* `{total_return:.2f}%`\n"
        f"*Total Trades:* `{total_trades}`\n\n"
        f"📈 Report saved to `report_{strategy_name}_{symbol}.html`"
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