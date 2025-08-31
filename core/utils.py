# core/utils.py
import logging
import requests
from config import Config

def setup_logger():
    """Sets up a dual-output logger."""
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    # File handler
    fh = logging.FileHandler("trading_bot.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

class TelegramManager:
    """Manages sending messages to a Telegram chat."""
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message: str):
        if not self.token or not self.chat_id:
            return

        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(self.base_url, data=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Use the global logger instance
            logger = logging.getLogger("TradingBot")
            logger.error(f"Telegram message failed to send: {e}")

# Initialize logger and telegram manager for global use
logger = setup_logger()
telegram = TelegramManager()