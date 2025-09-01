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
    # --- FIX: Extend the date range for more historical data ---
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
    
    # Check if it's a weekday (Monday=0, Sunday=6) and within market hours
    return 0 <= now.weekday() <= 4 and market_open_time <= now <= market_close_time

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
        
        # --- FIX: Standardize the column name to 'trading_symbol' ---
        df_nse.rename(columns={'tradingsymbol': 'trading_symbol'}, inplace=True)
        df_nfo.rename(columns={'tradingsymbol': 'trading_symbol'}, inplace=True)

        current_nifty_price = 0.0
        # Get the instrument token for the underlying symbol (NIFTY 50)
        nifty_instrument_token = df_nse[df_nse['trading_symbol'] == Config.UNDERLYING_SYMBOL].iloc[0]['instrument_token']

        if is_market_open():
            # Market is open, try to get live LTP
            try:
                ltp_data = kite_client.ltp(f"NSE:{Config.UNDERLYING_SYMBOL}")
                current_nifty_price = ltp_data.get(f"NSE:{Config.UNDERLYING_SYMBOL}", {}).get('last_price', 0.0)
                logger.info(f"Market is OPEN. Fetched live NIFTY 50 LTP: {current_nifty_price}")
            except Exception as e:
                logger.error(f"Failed to fetch live LTP from Kite for {Config.UNDERLYING_SYMBOL}: {e}")
                current_nifty_price = 0.0
        
        if current_nifty_price == 0.0:
            # Market is closed or live LTP fetch failed, use historical data as fallback
            logger.info("Market is CLOSED or live data fetch failed. Using historical data closing price.")
            try:
                # To get the latest closing price, we need to fetch the last day's data
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
                    logger.error(f"Could not fetch historical closing price for NIFTY 50. Using fallback hardcoded price.")
                    current_nifty_price = 22500.0 # A hardcoded fallback
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {Config.UNDERLYING_SYMBOL}: {e}")
                current_nifty_price = 22500.0 # A hardcoded fallback
                
        if current_nifty_price == 0.0:
            logger.error("Cannot determine a valid NIFTY 50 price. Exiting contract search.")
            return {}
            
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