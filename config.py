# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration for the trading bot.
    Loads sensitive data from environment variables for security.
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
    SYMBOLS_TO_TRADE = ['RELIANCE-EQ', 'HDFCBANK-EQ', 'INFY-EQ', 'TCS-EQ', 'ICICIBANK-EQ']
    # You will need to get the correct instrument tokens from your broker
    INSTRUMENT_TOKENS = {'RELIANCE-EQ': '2885', 'HDFCBANK-EQ': '1333'} # Example
    TOTAL_CAPITAL = 1000000.0
    MAX_RISK_PER_TRADE = 0.01  # Risk 1% of total capital per trade
    MIN_CONFIDENCE_TO_TRADE = 0.75 # Don't trade if AI confidence is below 75%

    # --- AI Model Parameters ---
    INPUT_SIZE = 55
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 60 # Use last 60 data points to predict
    WEIGHTS_FILENAME = 'lstm_brain_weights.pt'
    TORCHSCRIPT_FILENAME = 'lstm_brain_model.ts'

    # --- Data & Feature Engineering ---
    HISTORICAL_DATA_TIMEFRAME = '15minute'
    HISTORICAL_DATA_YEARS = 5