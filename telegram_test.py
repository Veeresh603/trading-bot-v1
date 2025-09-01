# test_telegram.py
import os
import requests
from dotenv import load_dotenv

# Load variables from your .env file
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Validation ---
if not BOT_TOKEN:
    print("❌ ERROR: TELEGRAM_BOT_TOKEN not found in .env file.")
    exit()
if not CHAT_ID:
    print("❌ ERROR: TELEGRAM_CHAT_ID not found in .env file.")
    exit()

print(f"✅ Bot Token and Chat ID loaded.")
print("Attempting to send a test message...")

message = "Hello from your Trading Bot! This is a test message to confirm notifications are working."
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
params = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

try:
    response = requests.post(url, json=params)
    response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
    
    response_json = response.json()
    if response_json.get("ok"):
        print("✅ SUCCESS! Message sent successfully to Telegram.")
    else:
        print("❌ FAILED: The Telegram API returned an error.")
        print(response_json)

except requests.exceptions.RequestException as e:
    print(f"❌ FAILED: An error occurred with the network request.")
    print(e)
except Exception as e:
    print(f"❌ FAILED: An unexpected error occurred.")
    print(e)
