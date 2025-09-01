# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration for the trading bot, training, and backtesting.
    """
    # --- Credentials ---
    API_KEY = os.getenv("ANGEL_API_KEY")
    CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
    PASSWORD = os.getenv("ANGEL_PASSWORD")
    TOTP = os.getenv("ANGEL_TOTP")

    # --- Telegram ---
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # --- Trading Parameters ---
    # Using .NS tickers for National Stock Exchange of India
    SYMBOLS_TO_TRADE = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS']
    MIN_CONFIDENCE_TO_TRADE = 0.75 # Don't trade if AI confidence is below 75%

    # --- AI Model Parameters ---
    INPUT_SIZE = 55
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    OUTPUT_SIZE = 3  # Buy (1), Sell (0), Hold (2)
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 60 # Use last 60 data points to predict
    WEIGHTS_FILENAME = 'lstm_brain_weights.pt'
    TORCHSCRIPT_FILENAME = 'lstm_brain_model.ts' # For the C++ engine

    # --- Data & Feature Engineering ---
    # NOTE: For multi-year backtests, '1d' is required by yfinance.
    # For live trading or short-term backtests (<60 days), '15m' or other intraday timeframes can be used.
    HISTORICAL_DATA_TIMEFRAME = '1d'

class BacktestConfig:
    """
    Configuration specific to the backtesting engine.
    """
    # --- Backtest Period ---
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"

    # --- Portfolio & Execution ---
    INITIAL_CAPITAL = 100000.0
    COMMISSION_BPS = 2.0  # Commission in basis points (e.g., 2.0 = 0.02%)
    SLIPPAGE_BPS = 1.0    # Slippage in basis points

    # --- NEW: Advanced Risk Management Parameters ---
    # Trailing stop loss percentage (e.g., 2% below the highest price seen)
    TRAILING_STOP_PCT = 0.02
    # Take profit percentage (e.g., 5% above the entry price)
    TAKE_PROFIT_PCT = 0.05
    # Risk percentage of total equity per trade (e.g., 1%)
    RISK_PER_TRADE_PCT = 0.01
    # Multiplier for ATR to set stop-loss (e.g., 2.0 * ATR)
    ATR_MULTIPLIER = 2.0