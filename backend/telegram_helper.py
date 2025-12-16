import requests
import os

# Telegram Bot Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8212625010:AAEm_dPoadjeR0J-pJIO20mVkvupiIXUU4s")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6187749045")

def send_telegram_alert(product_name, stock):
    """
    Send alert message to admin via Telegram
    """
    message = (
        f"⚠️ Inventory Alert!\n\n"
        f"Product: {product_name}\n"
        f"Current Stock: {stock}\n"
        f"Threshold: 10\n\n"
        f"Please restock this product immediately!"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print(f"✅ Alert sent for product: {product_name}")
        else:
            print(f"❌ Failed to send alert: {response.text}")
    except Exception as e:
        print(f"⚠️ Error sending Telegram alert: {e}")
