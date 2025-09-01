# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Central configuration for the trading bot, training, and backtesting.
    """
    API_KEY = os.getenv("ANGEL_API_KEY")
    CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
    PASSWORD = os.getenv("ANGEL_PASSWORD")
    TOTP = os.getenv("ANGEL_TOTP")

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    SYMBOLS_TO_TRADE = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS']
    MIN_CONFIDENCE_TO_TRADE = 0.75

    INPUT_SIZE = 55
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    OUTPUT_SIZE = 3
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 60
    WEIGHTS_FILENAME = 'lstm_brain_weights.pt'
    TORCHSCRIPT_FILENAME = 'lstm_brain_model.ts'

    HISTORICAL_DATA_TIMEFRAME = '1d'

class BacktestConfig:
    """
    Configuration specific to the backtesting engine.
    """
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    INITIAL_CAPITAL = 100000.0
    COMMISSION_BPS = 2.0
    SLIPPAGE_BPS = 1.0

    TRAILING_STOP_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05