# backtest.py
import argparse
from config import Config, BacktestConfig, get_current_options_contracts
from core.options_data import OptionsDataHandler
from core.strategy import AIStrategy
from core.portfolio import Portfolio
from core.execution import BacktestExecutionHandler
from core.utils import logger, telegram
import reports
import pandas as pd
from datetime import datetime
from kiteconnect import KiteConnect

def run_backtest(strategy_name: str, contracts: list):
    """
    Main function to run the event-driven backtest for options.
    """
    logger.info(f"--- Starting Backtest for {strategy_name} on {len(contracts)} contracts ---")
    logger.info(f"Period: {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE} | Initial Capital: ${BacktestConfig.INITIAL_CAPITAL:,.2f}")

    kite_client = KiteConnect(api_key=Config.KITE_API_KEY)
    try:
        kite_client.set_access_token(Config.KITE_ACCESS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to set access token during backtest setup: {e}")
        logger.error("Please run login_kite.py to get a new access token and update your .env file.")
        return

    data_handler = OptionsDataHandler(
        kite_client=kite_client,
        contracts=contracts,
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )
    
    if not data_handler.data:
        logger.error("No historical data could be fetched for any contract.")
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
        contracts=contracts,
        sequence_length=Config.SEQUENCE_LENGTH
    )

    all_dates = sorted(list(set(date for df in data_handler.data.values() for date in df.index)))
    
    logger.info(f"Backtesting over a total of {len(all_dates)} days.")

    for bar_datetime in all_dates:
        current_market_data = {
            c['trading_symbol']: data_handler.data[c['trading_symbol']].loc[bar_datetime]
            for c in contracts if c['trading_symbol'] in data_handler.data and bar_datetime in data_handler.data[c['trading_symbol']].index
        }
        
        for contract in contracts:
            symbol = contract['trading_symbol']
            if symbol in current_market_data:
                signal = strategy.generate_signals(contract, bar_datetime)

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
    trades_filepath = f"trades_{strategy_name}_options.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_filepath, index=False)
        returns = portfolio.equity_curve.pct_change().dropna()
        reports.generate_html_report(
            returns=returns,
            trades_filepath=trades_filepath,
            run_name=f"{strategy_name}_options"
        )
    else:
        logger.warning("No trades executed during backtest. Skipping report generation.")

    summary_message = (
        f"✅ *Backtest Complete*\n\n"
        f"*Strategy:* `{strategy_name}`\n"
        f"*Contracts:* `{', '.join(c['trading_symbol'] for c in contracts)}`\n"
        f"*Final Equity:* `${final_equity:,.2f}`\n"
        f"*Total Return:* `{total_return:.2f}%`\n"
        f"*Total Trades:* `{total_trades}`\n\n"
        f"📈 Report saved to `report_{strategy_name}_options.html`"
    )
    telegram.send_message(summary_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Event-Driven Options Backtester")
    parser.add_argument("--strategy", type=str, default="lstm", help="The strategy to backtest.")
    args = parser.parse_args()

    kite_client = KiteConnect(api_key=Config.KITE_API_KEY)
    try:
        kite_client.set_access_token(Config.KITE_ACCESS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to set access token during backtest setup: {e}")
        logger.error("Please run login_kite.py to get a new access token and update your .env file.")
        sys.exit(1)
        
    contracts = get_current_options_contracts(kite_client)
    if not contracts:
        logger.error("Could not fetch contracts to backtest. Exiting.")
        sys.exit(1)

    run_backtest(
        strategy_name=args.strategy,
        contracts=contracts
    )