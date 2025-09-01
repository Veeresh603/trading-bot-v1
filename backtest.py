# backtest.py
import argparse
import sys
from config import Config, BacktestConfig
# --- FIX: Import the correct data handler for indices ---
from core.data import HistoricalDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import BacktestExecutionHandler
from core.utils import logger, telegram
import reports
import pandas as pd
from datetime import datetime
from kiteconnect import KiteConnect

def run_backtest(strategy_name: str, symbol: str):
    """
    Main function to run the event-driven backtest on the underlying asset (e.g., NIFTY 50).
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {symbol} ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")

    # --- FIX: Use HistoricalDataHandler to fetch data for the underlying index ---
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

    portfolio = Portfolio(
        initial_capital=BacktestConfig.INITIAL_CAPITAL
    )

    execution_handler = BacktestExecutionHandler(
        commission_bps=BacktestConfig.COMMISSION_BPS,
        slippage_bps=BacktestConfig.SLIPPAGE_BPS
    )

    # --- FIX: Adapt strategy initialization for a single symbol ---
    # The 'contracts' parameter is now a simple list with one symbol dict
    strategy_contracts = [{'trading_symbol': symbol}]
    strategy = AIStrategy(
        data_handler=data_handler,
        contracts=strategy_contracts,
        sequence_length=Config.SEQUENCE_LENGTH
    )

    all_dates = sorted(data_handler.symbol_data[symbol].index)
    
    logger.info(f"Backtesting over a total of {len(all_dates)} market timestamps.")

    for bar_datetime in all_dates:
        # --- FIX: Generate signals and execute based on the single underlying symbol ---
        current_bar = data_handler.symbol_data[symbol].loc[bar_datetime]
        
        # The strategy expects a contract dictionary, so we provide one
        signal = strategy.generate_signals(strategy_contracts[0], bar_datetime)

        if signal and signal.get('direction') != 'HOLD':
            order = portfolio.create_order_from_signal(signal, current_bar, data_handler)
            if order:
                execution_handler.execute_order(order, current_bar, portfolio)
                        
    logger.info("--- Backtest Complete. Generating Performance Report... ---")

    final_equity = portfolio.holdings['total']
    total_return = (final_equity / BacktestConfig.INITIAL_CAPITAL - 1) * 100
    total_trades = portfolio.trade_count

    logger.info(f"Final Portfolio Equity: ${final_equity:,.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades Executed: {total_trades}")

    trades_df = pd.DataFrame(portfolio.trades)
    trades_filepath = f"trades_{strategy_name}_{symbol.replace(' ', '')}.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_filepath, index=False)
        returns = portfolio.equity_curve.pct_change().dropna()
        reports.generate_html_report(
            returns=returns,
            trades_filepath=trades_filepath,
            run_name=f"{strategy_name}_{symbol.replace(' ', '')}"
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
        f"📈 Report saved to `report_{strategy_name}_{symbol.replace(' ', '')}.html`"
    )
    telegram.send_message(summary_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Event-Driven Backtester")
    parser.add_argument("--strategy", type=str, default="lstm", help="The strategy to backtest.")
    parser.add_argument("--symbol", type=str, default=Config.UNDERLYING_SYMBOL, help="The symbol to backtest (e.g., 'NIFTY 50').")
    args = parser.parse_args()
        
    run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol
    )