# backtest.py
import argparse
from config import Config, BacktestConfig
from core.data import HistoricalDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import BacktestExecutionHandler
from core.utils import logger, telegram
import reports
import pandas as pd
from datetime import datetime

def run_backtest(strategy_name: str, symbols: list):
    """
    Main function to run the event-driven backtest for multiple symbols.
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {symbols} ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")
    
    telegram.send_message(
        f"🚀 *Starting Backtest*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbols:* `{', '.join(symbols)}`\n"
        f"*Period:* `{BacktestConfig.START_DATE}` to `{BacktestConfig.END_DATE}`"
    )

    data_handler = HistoricalDataHandler(
        symbols=symbols,
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe='1d'
    )
    
    if not data_handler.symbol_data:
        logger.error("No historical data could be fetched for any symbol.")
        telegram.send_message(f"❌ *Backtest Failed*\n\nNo historical data found.")
        return

    portfolio = Portfolio(
        initial_capital=BacktestConfig.INITIAL_CAPITAL
    )
    
    execution_handler = BacktestExecutionHandler(
        commission_bps=BacktestConfig.COMMISSION_BPS,
        slippage_bps=BacktestConfig.SLIPPAGE_BPS
    )

    strategy = AIStrategy(
        data_handler=data_handler,
        sequence_length=Config.SEQUENCE_LENGTH
    )
    
    # Create a master time index from all available data points
    all_dates = sorted(list(set(date for df in data_handler.symbol_data.values() for date in df.index)))
    
    logger.info(f"Backtesting over a total of {len(all_dates)} days.")

    for bar_datetime in all_dates:
        current_market_data = {
            symbol: data_handler.symbol_data[symbol].loc[bar_datetime]
            for symbol in symbols if bar_datetime in data_handler.symbol_data[symbol].index
        }
        
        # This is now the main risk check. If a stop-loss or take-profit is triggered, it will return a signal.
        risk_signal = portfolio.update_market_data(
            bar_datetime, 
            current_market_data
        )

        # Check if a risk signal was generated and act on it immediately
        if risk_signal:
            order = portfolio.create_order_from_signal(risk_signal, current_market_data[risk_signal['symbol']], data_handler)
            if order:
                execution_handler.execute_order(order, current_market_data[risk_signal['symbol']], portfolio)
            continue # Skip AI signal for this bar, as a risk exit has already occurred.
        
        # If no risk exit, check AI signals for new trades
        for symbol in symbols:
            if symbol in current_market_data:
                signal = strategy.generate_signals(symbol, bar_datetime)

                if signal and signal.get('direction') != 'HOLD':
                    order = portfolio.create_order_from_signal(signal, current_market_data[symbol], data_handler)
                    if order:
                        execution_handler.execute_order(order, current_market_data[symbol], portfolio)

    logger.info("--- Backtest Complete. Generating Performance Report... ---")
    
    final_equity = portfolio.holdings['total']
    total_return = (final_equity / BacktestConfig.INITIAL_CAPITAL - 1) * 100
    total_trades = portfolio.trade_count
    
    logger.info(f"Final Portfolio Equity: ${final_equity:,.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades Executed: {total_trades}")
    
    trades_df = pd.DataFrame(portfolio.trades)
    trades_filepath = f"trades_{strategy_name}_{'_'.join(symbols)}.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_filepath, index=False)
        returns = portfolio.equity_curve.pct_change().dropna()
        reports.generate_html_report(
            returns=returns,
            trades_filepath=trades_filepath,
            run_name=f"{strategy_name}_{'_'.join(symbols)}"
        )
    else:
        logger.warning("No trades executed during backtest. Skipping report generation.")

    summary_message = (
        f"✅ *Backtest Complete*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Symbols:* `{', '.join(symbols)}`\n"
        f"*Final Equity:* `${final_equity:,.2f}`\n"
        f"*Total Return:* `{total_return:.2f}%`\n"
        f"*Total Trades:* `{total_trades}`\n\n"
        f"📈 Report saved to `report_{strategy_name}_{'_'.join(symbols)}.html`"
    )
    telegram.send_message(summary_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Event-Driven Backtester")
    parser.add_argument("--strategy", type=str, default="lstm", help="The strategy to backtest.")
    parser.add_argument("--symbol", type=str, nargs='?', help="The trading symbol(s) to backtest (e.g., 'RELIANCE.NS'). Pass 'all' or leave blank to backtest all configured symbols.")
    args = parser.parse_args()

    symbols_to_backtest = Config.SYMBOLS_TO_TRADE
    if args.symbol and args.symbol.lower() != 'all':
        symbols_to_backtest = args.symbol.split(',')

    run_backtest(
        strategy_name=args.strategy,
        symbols=symbols_to_backtest
    )