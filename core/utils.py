# core/utils.py
import logging
import os
import sys
import requests
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This ensures that credentials are loaded once and are available project-wide
load_dotenv()

# --- Centralized Logger Setup ---
def setup_logger():
    """
    Sets up a centralized logger that is robust to unicode characters.
    This logger will be used by all modules in the project.
    """
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)

    # Prevent the logger from propagating to the root logger
    logger.propagate = False

    # Check if handlers are already added to prevent duplicate logs
    if not logger.handlers:
        # --- THE FIX IS HERE ---
        # Use a StreamHandler with UTF-8 encoding to support emojis and other characters
        # This is crucial for running on Windows without UnicodeEncodeError.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler.encoding = 'utf-8' # Explicitly set the encoding
        
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()


# --- Centralized Telegram Messenger ---
class TelegramMessenger:
    """
    Handles all communication with the Telegram API.
    """
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.is_configured = self.token and self.chat_id

        if not self.is_configured:
            logger.warning("Telegram credentials not found in .env file. Notifications will be disabled.")

    def send_message(self, text: str):
        """Sends a message to the configured Telegram chat."""
        if not self.is_configured:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=params, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram message sent successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")

# Create a single, project-wide instance of the messenger
telegram = TelegramMessenger()
