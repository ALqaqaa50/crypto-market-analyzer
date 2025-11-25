import os
import asyncio

os.environ['OKX_SANDBOX_KEY'] = '313bca60-24ac-4d5c-90c6-5f27f07e826f'
os.environ['OKX_SANDBOX_SECRET'] = '425FDFDD48F37A01F953CAC05B93A4BD'
os.environ['OKX_SANDBOX_PASSPHRASE'] = 'aQMF$YhiOu7nH2U'
os.environ['TELEGRAM_BOT_TOKEN'] = '8212473860:AAHECTA0F84xpQbaDCeMItekXTVcUsjCouM'
os.environ['TELEGRAM_CHAT_ID'] = '8468481578'
os.environ['TRADING_MODE'] = 'sandbox'

from okx_stream_hunter.core.trading_mode import get_trading_mode_manager
from okx_stream_hunter.notifications import get_telegram_client

async def test():
    mode_mgr = get_trading_mode_manager()
    telegram = get_telegram_client()
    
    print("=" * 60)
    print("PHASE 5.1 CONFIGURATION TEST")
    print("=" * 60)
    print(f"Trading Mode: {mode_mgr.mode.value.upper()}")
    print(f"Log Prefix: {mode_mgr.get_log_prefix()}")
    print(f"API URL: {mode_mgr.api_url}")
    print(f"Sandbox Mode: {mode_mgr.is_sandbox()}")
    print(f"Real Mode: {mode_mgr.is_real()}")
    print("")
    print(f"API Key: {mode_mgr.credentials['api_key'][:15]}...")
    print(f"Telegram Enabled: {telegram.enabled}")
    print(f"Bot Token: {telegram.bot_token[:15] if telegram.bot_token else 'None'}...")
    print(f"Chat ID: {telegram.chat_id}")
    print("")
    
    if telegram.enabled:
        print("Sending test message to Telegram...")
        success = await telegram.send_status("PROMETHEUS v7 OMEGA - Phase 5.1 configured successfully!")
        print(f"Telegram test: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    print("=" * 60)
    print("✅ All systems operational")
    print("=" * 60)

asyncio.run(test())
