# reports.py
import pandas as pd
import quantstats as qs
from core.utils import logger

def print_performance_summary(returns: pd.Series):
    """Prints a summary of key performance metrics to the console."""
    logger.info("--- Performance Summary ---")
    if not returns.empty:
        logger.info(f"Sharpe Ratio: {qs.stats.sharpe(returns):.2f}")
        logger.info(f"Sortino Ratio: {qs.stats.sortino(returns):.2f}")
        logger.info(f"Max Drawdown: {qs.stats.max_drawdown(returns):.2%}")
        logger.info(f"CAGR (Compounded Annual Growth Rate): {qs.stats.cagr(returns):.2%}")
        logger.info(f"Total Return: {qs.stats.comp(returns).iloc[-1]:.2%}")
    else:
        logger.info("No returns data to analyze.")
    logger.info("---------------------------")

def generate_html_report(returns: pd.Series, trades_filepath: str, run_name: str):
    """
    Generates a full, professional-grade HTML report using quantstats.
    """
    try:
        output_filename = f"report_{run_name}.html"
        
        # --- FIX: Ensure the trades DataFrame has a proper DatetimeIndex ---
        trades_df = pd.read_csv(trades_filepath, parse_dates=['timestamp'])
        trades_df.set_index('timestamp', inplace=True)

        qs.reports.html(
            returns=returns,
            benchmark=None,
            title=f"{run_name} - Performance Report",
            output=output_filename,
            trades=trades_df
        )
        logger.info(f"✅ Professional HTML performance report saved to '{output_filename}'")
        logger.info("Open this file in your browser to view detailed charts and analytics.")
    except Exception as e:
        logger.error(f"Could not generate HTML report: {e}")