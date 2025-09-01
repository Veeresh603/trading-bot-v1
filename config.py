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
    # The underlying index or stock for options trading.
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
    OUTPUT_SIZE = 3
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 60
    WEIGHTS_FILENAME = 'lstm_options_weights.pt'
    TORCHSCRIPT_FILENAME = 'lstm_options_model.ts'

    # --- Data & Feature Engineering ---
    HISTORICAL_DATA_TIMEFRAME = '15minute'
    LOCAL_DATA_DIR = 'data_options'


class BacktestConfig:
    """
    Configuration specific to the backtesting engine.
    """
    START_DATE = "2024-08-01"
    END_DATE = "2025-08-01"

    INITIAL_CAPITAL = 100000.0
    COMMISSION_BPS = 2.0
    SLIPPAGE_BPS = 1.0

    TRAILING_STOP_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05
    RISK_PER_TRADE_PCT = 0.01
    ATR_MULTIPLIER = 2.0


def get_current_options_contracts(kite_client: KiteConnect):
    """
    Dynamically fetches NIFTY options and futures contracts, filters by strike and expiry,
    separates CE and PE, and saves a daily CSV for reference.
    """
    try:
        # Fetch all instruments for NSE and NFO
        nse_instruments = kite_client.instruments('NSE')
        nfo_instruments = kite_client.instruments('NFO')
        
        df_nse = pd.DataFrame(nse_instruments)
        df_nfo = pd.DataFrame(nfo_instruments)
        
        # Always fetch live NIFTY price via LTP
        try:
            ltp_data = kite_client.ltp(f"NSE:{Config.UNDERLYING_SYMBOL}")
            current_nifty_price = ltp_data[f"NSE:{Config.UNDERLYING_SYMBOL}"]['last_price']
            logger.info(f"Fetched last_price via LTP: {current_nifty_price}")
        except Exception as e:
            logger.error(f"Cannot fetch last_price for {Config.UNDERLYING_SYMBOL}: {e}")
            return []

        # Filter NIFTY options from NFO list
        nifty_options = df_nfo[
            (df_nfo['name'] == 'NIFTY') &
            (df_nfo['instrument_type'].isin(Config.CONTRACT_TYPE))
        ].copy()
        
        # Exclude expired contracts
        today = datetime.now().date()
        nifty_options['expiry'] = pd.to_datetime(nifty_options['expiry']).dt.date
        nifty_options = nifty_options[nifty_options['expiry'] >= today]
        
        # Filter by strike difference
        nifty_options['strike_diff'] = (nifty_options['strike'] - current_nifty_price).abs()
        filtered_options = nifty_options[
            (nifty_options['strike_diff'] >= Config.MIN_STRIKE_DIFFERENCE) &
            (nifty_options['strike_diff'] <= Config.MAX_STRIKE_DIFFERENCE)
        ]

        # Keep next two expiries
        expiries = sorted(filtered_options['expiry'].unique())
        next_two_expiries = expiries[:2]
        final_contracts = filtered_options[filtered_options['expiry'].isin(next_two_expiries)]
        
        # Convert to dictionary records
        contracts = final_contracts.to_dict('records')
        
        if contracts:
            logger.info(f"Dynamically fetched {len(contracts)} contracts for {Config.UNDERLYING_SYMBOL}.")
        else:
            logger.warning("No contracts found with the specified filters.")
            return {}

        # Separate CE and PE for convenience
        calls = [c for c in contracts if c['instrument_type'] == 'CE']
        puts = [c for c in contracts if c['instrument_type'] == 'PE']
        
        # Save daily CSV
        df_final = pd.DataFrame(contracts)
        file_name = f"nifty_options_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_final.to_csv(file_name, index=False)
        logger.info(f"Saved {len(df_final)} contracts to {file_name}")

        return {
            'all': contracts,
            'calls': calls,
            'puts': puts
        }

    except Exception as e:
        logger.error(f"Failed to fetch and filter options contracts: {e}")
        return {}
