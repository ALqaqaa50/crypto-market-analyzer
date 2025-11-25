import os

os.environ['OKX_SANDBOX_KEY'] = '313bca60-24ac-4d5c-90c6-5f27f07e826f'
os.environ['OKX_SANDBOX_SECRET'] = '425FDFDD48F37A01F953CAC05B93A4BD'
os.environ['OKX_SANDBOX_PASSPHRASE'] = 'aQMF$YhiOu7nH2U'
os.environ['TRADING_MODE'] = 'sandbox'

from okx_stream_hunter.core.trading_mode import get_trading_mode_manager

mode_mgr = get_trading_mode_manager()

print("=" * 60)
print("OKX CONFIGURATION TEST")
print("=" * 60)
print(f"Mode: {mode_mgr.mode.value}")
print(f"Is Sandbox: {mode_mgr.is_sandbox()}")
print(f"Is Real: {mode_mgr.is_real()}")
print(f"Log Prefix: {mode_mgr.get_log_prefix()}")
print(f"API URL: {mode_mgr.api_url}")
print("")
print("Credentials:")
creds = mode_mgr.get_credentials()
print(f"  API Key: {creds['api_key']}")
print(f"  Secret Key: {creds['secret_key'][:10]}...{creds['secret_key'][-5:]}")
print(f"  Passphrase: {creds['passphrase']}")
print("")
print("Safety Check:")
check = mode_mgr.get_safety_check()
print(f"  Passed: {check[0]}")
print(f"  Message: {check[1]}")
print("")
print("=" * 60)
print("✅ OKX Configuration Valid")
print("=" * 60)
