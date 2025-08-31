# login_kite.py
import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")

kite = KiteConnect(api_key=API_KEY)

# Generate login URL
print("Open this URL in your browser and login:")
print(kite.login_url())

# Get request token from URL
request_token = input("Paste the request_token from the redirected URL: ").strip()

try:
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]
    print(f"✅ Access Token: {access_token}")
    print("\nIMPORTANT: Copy this access token and paste it into your .env file as KITE_ACCESS_TOKEN")
except Exception as e:
    print(f"❌ Failed to generate session: {e}")