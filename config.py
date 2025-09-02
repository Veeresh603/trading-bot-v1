# config.py
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from core.utils import logger
import csv
import io

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration for the trading bot, training, and backtesting.
    """
    # --- Credentials ---
    KITE_API_KEY = os.getenv("KITE_API_KEY")
    KITE_API_SECRET = os.getenv("KITE_API_SECRET")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

    # --- Telegram ---
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # --- Trading Parameters (for Options) ---
    UNDERLYING_SYMBOL = 'NIFTY 50'
    OPTIONS_EXCHANGE = 'NFO'
    
    # Filter parameters for options
    CONTRACT_TYPE = ['CE', 'PE']
    MIN_STRIKE_DIFFERENCE = 1000
    MAX_STRIKE_DIFFERENCE = 5000
    
    # --- AI Model Parameters ---
    INPUT_SIZE = 55
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    OUTPUT_SIZE = 3 # Corresponds to BUY, SELL, HOLD
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 60
    WEIGHTS_FILENAME = 'lstm_options_weights.pt'
    TORCHSCRIPT_FILENAME = 'lstm_options_model.ts'
    
    # --- FIX: Added Missing Trading Parameter ---
    MIN_CONFIDENCE_TO_TRADE = 0.75 # Model must be at least 75% confident to place a trade

    # --- Data & Feature Engineering ---
    HISTORICAL_DATA_TIMEFRAME = '15minute'
    LOCAL_DATA_DIR = 'data_options'


class BacktestConfig:
    """
    Configuration specific to the backtesting engine.
    """
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"

    INITIAL_CAPITAL = 100000.0
    COMMISSION_BPS = 2.0
    SLIPPAGE_BPS = 1.0

    TRAILING_STOP_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05
    RISK_PER_TRADE_PCT = 0.01
    ATR_MULTIPLIER = 2.0

def is_market_open():
    """Checks if the Indian market is currently open (9:15 AM to 3:30 PM IST)."""
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return 0 <= now.weekday() <= 4 and market_open_time <= now <= market_close_time

def get_current_options_contracts(kite_client: KiteConnect):
    """
    Dynamically fetches NIFTY options contracts.
    """
    try:
        nse_instruments = kite_client.instruments('NSE')
        nfo_instruments = kite_client.instruments('NFO')
        
        df_nse = pd.DataFrame(nse_instruments)
        df_nfo = pd.DataFrame(nfo_instruments)
        
        df_nse.rename(columns={'tradingsymbol': 'trading_symbol'}, inplace=True)
        df_nfo.rename(columns={'tradingsymbol': 'trading_symbol'}, inplace=True)

        current_nifty_price = 0.0
        nifty_instrument_token = df_nse[df_nse['trading_symbol'] == Config.UNDERLYING_SYMBOL].iloc[0]['instrument_token']

        if is_market_open():
            try:
                ltp_data = kite_client.ltp(f"NSE:{Config.UNDERLYING_SYMBOL}")
                current_nifty_price = ltp_data.get(f"NSE:{Config.UNDERLYING_SYMBOL}", {}).get('last_price', 0.0)
                logger.info(f"Market is OPEN. Fetched live NIFTY 50 LTP: {current_nifty_price}")
            except Exception as e:
                logger.error(f"Failed to fetch live LTP: {e}")
                current_nifty_price = 0.0
        
        if current_nifty_price == 0.0:
            logger.info("Market is CLOSED or live data fetch failed. Using historical data closing price.")
            try:
                today = datetime.now().date()
                from_date = today - timedelta(days=5)
                historical_data = kite_client.historical_data(
                    instrument_token=nifty_instrument_token,
                    from_date=from_date,
                    to_date=today,
                    interval='day'
                )
                if historical_data:
                    current_nifty_price = historical_data[-1]['close']
                    logger.info(f"Fetched historical closing price for NIFTY 50: {current_nifty_price}")
                else:
                    current_nifty_price = 22500.0
            except Exception as e:
                logger.error(f"Failed to fetch historical data: {e}")
                current_nifty_price = 22500.0
                
        if current_nifty_price == 0.0:
            logger.error("Cannot determine a valid NIFTY 50 price. Exiting.")
            return {}
            
        nifty_options = df_nfo[
            (df_nfo['name'] == 'NIFTY') &
            (df_nfo['instrument_type'].isin(Config.CONTRACT_TYPE))
        ].copy()
        
        today = datetime.now().date()
        nifty_options['expiry'] = pd.to_datetime(nifty_options['expiry']).dt.date
        nifty_options = nifty_options[nifty_options['expiry'] >= today]
        
        nifty_options['strike_diff'] = (nifty_options['strike'] - current_nifty_price).abs()
        filtered_options = nifty_options[
            (nifty_options['strike_diff'] >= Config.MIN_STRIKE_DIFFERENCE) &
            (nifty_options['strike_diff'] <= Config.MAX_STRIKE_DIFFERENCE)
        ]

        expiries = sorted(filtered_options['expiry'].unique())
        next_two_expiries = expiries[:2]
        final_contracts = filtered_options[filtered_options['expiry'].isin(next_two_expiries)]
        
        contracts = final_contracts.to_dict('records')
        
        if not contracts:
            logger.warning("No contracts found with the specified filters.")
            return {}

        logger.info(f"Dynamically fetched {len(contracts)} contracts for {Config.UNDERLYING_SYMBOL}.")
        calls = [c for c in contracts if c['instrument_type'] == 'CE']
        puts = [c for c in contracts if c['instrument_type'] == 'PE']
        
        return {
            'all': contracts,
            'calls': calls,
            'puts': puts
        }

    except Exception as e:
        logger.error(f"Failed to fetch and filter options contracts: {e}")
        return {}