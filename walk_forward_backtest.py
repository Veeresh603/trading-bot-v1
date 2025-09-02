# walk_forward_backtest.py
import pandas as pd
from datetime import timedelta
from train import main as run_training
from backtest import run_backtest
from config import BacktestConfig, Config
from core.utils import logger

def perform_walk_forward_analysis():
    """
    Performs a walk-forward analysis by periodically retraining the model.
    """
    start_date = pd.to_datetime(BacktestConfig.START_DATE)
    end_date = pd.to_datetime(BacktestConfig.END_DATE)
    
    train_period = timedelta(days=365)
    test_period = timedelta(days=90)
    
    current_date = start_date
    
    while current_date + train_period + test_period <= end_date:
        train_start = current_date
        train_end = current_date + train_period
        test_start = train_end + timedelta(days=1)
        test_end = test_start + test_period
        
        logger.info("--- Starting Walk-Forward Period ---")
        logger.info(f"Training Period: {train_start.date()} to {train_end.date()}")
        logger.info(f"Testing Period: {test_start.date()} to {test_end.date()}")
        
        # Override config for the current period
        original_start_date, original_end_date = BacktestConfig.START_DATE, BacktestConfig.END_DATE
        BacktestConfig.START_DATE = train_start.strftime('%Y-%m-%d')
        BacktestConfig.END_DATE = train_end.strftime('%Y-%m-%d')
        
        # 1. Train the model on the training period
        logger.info("--- Initiating Training Phase ---")
        run_training()
        
        # 2. Backtest the model on the out-of-sample test period
        logger.info("--- Initiating Backtesting Phase ---")
        BacktestConfig.START_DATE = test_start.strftime('%Y-%m-%d')
        BacktestConfig.END_DATE = test_end.strftime('%Y-%m-%d')
        
        run_backtest(
            strategy_name="AIStrategy_WalkForward",
            symbol=Config.UNDERLYING_SYMBOL
        )
        
        # Restore original config for next iteration if needed
        BacktestConfig.START_DATE, BacktestConfig.END_DATE = original_start_date, original_end_date
        
        # Move to the next period
        current_date += test_period

if __name__ == "__main__":
    perform_walk_forward_analysis()