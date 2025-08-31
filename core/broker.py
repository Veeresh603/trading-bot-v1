# core/broker.py
from kiteconnect import KiteConnect, KiteTicker
from config import Config
from .utils import logger, telegram

class BrokerManager:
    """Manages all interactions with the Kite Connect broker API."""
    def __init__(self):
        self.api_key = Config.KITE_API_KEY
        self.access_token = Config.KITE_ACCESS_TOKEN
        self.kite = KiteConnect(api_key=self.api_key)
        self.kws = None # For WebSocket
        self.session_active = False

    def login(self) -> bool:
        """Establishes a session with the broker using the access token."""
        try:
            logger.info("Attempting to connect to Kite...")
            self.kite.set_access_token(self.access_token)
            profile = self.kite.profile()
            if profile and profile.get('user_id'):
                self.session_active = True
                logger.info(f"✅ Kite connection successful. User: {profile['user_id']}")
                telegram.send_message("✅ *Kite Connection Successful*")
                return True
            else:
                raise Exception("Profile not found.")
        except Exception as e:
            logger.critical(f"Kite login failed. Run login_kite.py to get a new access token. Error: {e}")
            telegram.send_message("❌ *CRITICAL: Kite login failed!*")
            return False

    def place_order(self, symbol: str, quantity: int, direction: str):
        """Places an order with Kite."""
        if not self.session_active:
            logger.error("Cannot place order. Broker session is not active.")
            return

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if direction == "LONG" else self.kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS, # Margin Intraday Squareoff
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            logger.info(f"✅ Order placed successfully. Order ID: {order_id}")
            telegram.send_message(f"✅ *Order Placed*: {direction} {quantity} {symbol}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Order placement failed for {symbol}: {e}", exc_info=True)
            telegram.send_message(f"❌ *ORDER FAILED*: {direction} {quantity} {symbol}")

    def setup_websockets(self, on_tick_callback, on_connect_callback):
        """Sets up the KiteTicker WebSocket connection."""
        self.kws = KiteTicker(self.api_key, self.access_token)
        self.kws.on_ticks = on_tick_callback
        self.kws.on_connect = on_connect_callback
        logger.info("Connecting to WebSocket...")
        self.kws.connect(threaded=True)