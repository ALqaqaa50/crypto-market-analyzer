import os
import logging
import asyncio
from typing import Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, Telegram integration disabled")


class TelegramClient:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        self.enabled = bool(self.bot_token and self.chat_id and HTTPX_AVAILABLE)
        
        if self.enabled:
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            logger.info("Telegram client initialized and enabled")
        else:
            logger.warning("Telegram client disabled (missing credentials or httpx)")
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode
                    }
                )
                
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.status_code}")
                    return False
        
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send_message_sync(self, text: str) -> bool:
        if not self.enabled:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except Exception as e:
            logger.error(f"Telegram sync send error: {e}")
            return False
    
    async def send_trade_alert(self, direction: str, size: float, price: float, 
                              mode: str, symbol: str = "BTC-USDT-SWAP",
                              confidence: float = 0.0, leverage: int = 1) -> bool:
        mode_label = mode.upper()
        
        text = (
            f"<b>[TRADE][{mode_label}]</b> {direction.upper()} {symbol}\n"
            f"<b>Size:</b> {size:.4f}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Leverage:</b> {leverage}x\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_position_open(self, symbol: str, side: str, size: float, 
                                 price: float, confidence: float, mode: str) -> bool:
        mode_label = mode.upper()
        
        text = (
            f"<b>[POSITION OPEN][{mode_label}]</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Size:</b> {size:.4f}\n"
            f"<b>Entry Price:</b> ${price:,.2f}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_position_close(self, symbol: str, side: str, pnl: float, 
                                  confidence: float, mode: str, 
                                  entry_price: float = 0, exit_price: float = 0) -> bool:
        mode_label = mode.upper()
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        
        text = (
            f"<b>[POSITION CLOSE][{mode_label}]</b> {pnl_emoji}\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>P&L:</b> ${pnl:,.2f}\n"
            f"<b>Entry:</b> ${entry_price:,.2f}\n"
            f"<b>Exit:</b> ${exit_price:,.2f}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_error_alert(self, error_text: str, include_trace: bool = True) -> bool:
        if include_trace:
            trace = traceback.format_exc()
            text = (
                f"<b>[ERROR]</b>\n"
                f"<b>Message:</b> {error_text}\n\n"
                f"<b>Traceback:</b>\n<code>{trace[:3000]}</code>"
            )
        else:
            text = f"<b>[ERROR]</b> {error_text}"
        
        return await self.send_message(text)
    
    async def send_status(self, text: str) -> bool:
        formatted = f"<b>[STATUS]</b> {text}"
        return await self.send_message(formatted)
    
    async def send_warning(self, text: str) -> bool:
        formatted = f"<b>⚠️ [WARNING]</b> {text}"
        return await self.send_message(formatted)
    
    async def send_model_update(self, model_type: str, version: str, performance: float) -> bool:
        text = (
            f"<b>[MODEL UPDATE]</b>\n"
            f"<b>Type:</b> {model_type}\n"
            f"<b>Version:</b> {version}\n"
            f"<b>Performance:</b> {performance:.4f}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_rl_update(self, agent_type: str, confidence: float, 
                            action: float, performance: dict) -> bool:
        text = (
            f"<b>[RL UPDATE]</b>\n"
            f"<b>Agent:</b> {agent_type}\n"
            f"<b>Action:</b> {action:.4f}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Avg Performance:</b> {performance.get('avg', 0):.4f}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_safety_alert(self, alert_type: str, details: str) -> bool:
        text = (
            f"<b>🛡️ [SAFETY]</b> {alert_type}\n"
            f"{details}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)
    
    async def send_system_restart(self, reason: str = "Manual") -> bool:
        text = (
            f"<b>🔄 [SYSTEM RESTART]</b>\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send_message(text)


_telegram_instance = None

def get_telegram_client() -> TelegramClient:
    global _telegram_instance
    if _telegram_instance is None:
        _telegram_instance = TelegramClient()
    return _telegram_instance
