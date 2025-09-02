# reports.py
import pandas as pd
import quantstats as qs
from core.utils import logger
import numpy as np

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

def run_monte_carlo_simulation(returns: pd.Series, simulations=1000, periods=252):
    """
    Runs a Monte Carlo simulation on the strategy returns to assess robustness.
    """
    if returns.empty:
        logger.warning("Cannot run Monte Carlo simulation on empty returns series.")
        return

    logger.info("--- Running Monte Carlo Simulation ---")
    daily_returns = returns.resample('D').sum()
    
    simulation_df = pd.DataFrame()
    for i in range(simulations):
        # Sample with replacement from historical returns
        simulated_returns = np.random.choice(daily_returns, size=periods, replace=True)
        # Create a cumulative return series
        simulated_equity_curve = (1 + simulated_returns).cumprod()
        simulation_df[i] = simulated_equity_curve
    
    logger.info("--- Monte Carlo Simulation Results ---")
    final_returns = simulation_df.iloc[-1]
    logger.info(f"Average Final Return: {final_returns.mean():.2%}")
    logger.info(f"5th Percentile Return: {final_returns.quantile(0.05):.2%}")
    logger.info(f"95th Percentile Return: {final_returns.quantile(0.95):.2%}")
    logger.info(f"Probability of a positive return: {(final_returns > 1).mean():.2%}")
    logger.info("------------------------------------")