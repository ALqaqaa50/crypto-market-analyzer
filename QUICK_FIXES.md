# ⚡ إصلاحات سريعة - الأولوية العالية

## 🔴 الإصلاحات الحرجة (يجب تطبيقها فوراً)

### 1. إصلاح OKXStreamEngine Integration

**الملف:** `okx_stream_hunter/core/autonomous_runtime.py`

**السطر 111-119:**

```python
# ❌ الكود الحالي (خاطئ):
self.stream_engine = OKXStreamEngine(symbol=symbol)

# Subscribe to stream events
self.stream_engine.subscribe('ticker', self._on_ticker)
self.stream_engine.subscribe('trades', self._on_trade)
self.stream_engine.subscribe('orderbook', self._on_orderbook)
```

**✅ الإصلاح:**

```python
# استخدام StreamEngine الصحيح
from okx_stream_hunter.core.stream_manager import StreamEngine

self.stream_engine = StreamEngine(
    symbols=[symbol],
    channels=['tickers', 'trades', 'books5'],
    logger=self.logger.getChild('stream'),
    db_writer=None  # أو db_writer إذا كان متاحاً
)

# لا حاجة لـ subscribe - StreamEngine يعالج callbacks داخلياً
```

---

### 2. إضافة AI Brain للـ Runtime

**الملف:** `okx_stream_hunter/core/autonomous_runtime.py`

**بعد السطر 40:**

```python
# إضافة AI Brain
self.ai_brain: Optional[Any] = None
```

**في `_initialize_components` بعد السطر 117:**

```python
# Initialize AI Brain
from okx_stream_hunter.ai.brain_ultra import get_brain
self.ai_brain = get_brain()
logger.info("✅ AI Brain initialized")
```

**في `_on_ticker` السطر 249:**

```python
async def _on_ticker(self, ticker_data: Dict):
    """Handle ticker updates"""
    try:
        self.market_state.price = ticker_data.get('last', 0.0)
        self.market_state.bid = ticker_data.get('bidPx', 0.0)
        self.market_state.ask = ticker_data.get('askPx', 0.0)
        self.market_state.volume_24h = ticker_data.get('vol24h', 0.0)
        
        # ✅ تغذية AI Brain
        if self.ai_brain:
            self.ai_brain.update_from_ticker(ticker_data)
        
        # Update master loop
        if self.master_loop:
            await self.master_loop.update_market_state(self.market_state)
    except Exception as e:
        logger.error(f"❌ Ticker callback error: {e}")
```

**في `_on_trade` السطر 264:**

```python
async def _on_trade(self, trade_data: Dict):
    """Handle trade updates"""
    try:
        self.stats['total_ticks'] += 1
        
        price = float(trade_data.get('px', 0))
        size = float(trade_data.get('sz', 0))
        timestamp = datetime.now()
        
        # ✅ تغذية AI Brain
        if self.ai_brain:
            self.ai_brain.update_from_trade({
                'price': price,
                'size': size,
                'side': trade_data.get('side', 'buy'),
                'timestamp': timestamp
            })
        
        # Process tick in master loop
        if self.master_loop:
            await self.master_loop.process_tick({
                'price': price,
                'size': size,
                'timestamp': timestamp
            })
        
        # Update market state
        self.market_state.price = price
    except Exception as e:
        logger.error(f"❌ Trade callback error: {e}")
```

---

### 3. إصلاح Stream Health Check

**الملف:** `okx_stream_hunter/core/autonomous_runtime.py`

**السطر 301-306:**

```python
# ❌ الكود الحالي (خاطئ):
async def _check_stream_health(self) -> bool:
    """Check stream engine health"""
    if not self.stream_engine:
        return False
    
    return self.stream_engine.ws_client.is_alive() if hasattr(self.stream_engine, 'ws_client') else True
```

**✅ الإصلاح:**

```python
async def _check_stream_health(self) -> bool:
    """Check stream engine health"""
    if not self.stream_engine:
        return False
    
    # StreamEngine من stream_manager.py
    if hasattr(self.stream_engine, '_running'):
        return self.stream_engine._running
    
    # أو فحص ws_client إذا كان موجوداً
    if hasattr(self.stream_engine, 'ws_client'):
        ws = self.stream_engine.ws_client
        return ws.connected if hasattr(ws, 'connected') else True
    
    return True
```

---

### 4. إضافة Auto-Reconnect

**ملف جديد:** `okx_stream_hunter/utils/reconnect.py`

```python
"""Auto-reconnect utilities"""
import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def run_with_retry(
    func: Callable,
    max_retries: int = 5,
    base_backoff: float = 1.0,
    max_backoff: float = 30.0,
    context: str = "task"
) -> None:
    """
    Run an async function with exponential backoff retry.
    
    Args:
        func: Async function to run
        max_retries: Maximum number of retry attempts
        base_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        context: Description for logging
    """
    retry_count = 0
    backoff = base_backoff
    
    while retry_count < max_retries:
        try:
            logger.info(f"Starting {context} (attempt {retry_count + 1}/{max_retries})")
            await func()
            break  # Success
            
        except asyncio.CancelledError:
            logger.info(f"{context} cancelled")
            raise
            
        except Exception as e:
            retry_count += 1
            logger.error(f"{context} failed: {e}")
            
            if retry_count < max_retries:
                logger.info(f"Retrying {context} in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            else:
                logger.critical(f"{context} failed after {max_retries} attempts")
                raise


class RetryableTask:
    """Wrapper for tasks that need auto-retry"""
    
    def __init__(
        self,
        func: Callable,
        max_retries: int = 5,
        backoff: float = 1.0,
        context: str = "task"
    ):
        self.func = func
        self.max_retries = max_retries
        self.backoff = backoff
        self.context = context
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the retryable task"""
        self.task = asyncio.create_task(
            run_with_retry(
                self.func,
                max_retries=self.max_retries,
                base_backoff=self.backoff,
                context=self.context
            )
        )
        return self.task
    
    def cancel(self):
        """Cancel the task"""
        if self.task and not self.task.done():
            self.task.cancel()
```

**استخدامه في `autonomous_runtime.py`:**

```python
from okx_stream_hunter.utils.reconnect import RetryableTask

async def _start_components(self):
    """Start all components with retry"""
    logger.info("▶️ Starting components...")
    
    # Start Master Loop with retry
    master_task = RetryableTask(
        self.master_loop.start,
        max_retries=3,
        context="Master Loop"
    )
    await master_task.start()
    
    # Start Stream Engine with retry
    stream_task = RetryableTask(
        self.stream_engine.start,
        max_retries=5,
        backoff=2.0,
        context="Stream Engine"
    )
    await stream_task.start()
    
    # Start Watchdog
    asyncio.create_task(self.watchdog.start())
    
    await asyncio.sleep(2)
```

---

### 5. إصلاح Memory Leak في MarketState

**الملف:** `okx_stream_hunter/core/market_state.py`

**البحث عن:**

```python
class MarketState:
    def __init__(self, symbol: str):
        self.recent_trades = []  # ❌ سينمو بلا حدود
```

**✅ الإصلاح:**

```python
from collections import deque

class MarketState:
    MAX_TRADES = 5000
    MAX_CANDLES = 500
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.recent_trades = deque(maxlen=self.MAX_TRADES)  # ✅ محدود
        self.candle_history = deque(maxlen=self.MAX_CANDLES) if hasattr(self, 'candle_history') else deque(maxlen=self.MAX_CANDLES)
        
        # باقي الكود...
```

---

### 6. تحسين Error Handling

**ملف جديد:** `okx_stream_hunter/utils/error_handler.py`

```python
"""Centralized error handling"""
import asyncio
import logging
import traceback
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling with retry logic"""
    
    TRANSIENT_ERRORS = (
        asyncio.TimeoutError,
        ConnectionError,
        ConnectionResetError,
        OSError,
    )
    
    @staticmethod
    def is_transient(error: Exception) -> bool:
        """Check if error is transient (can retry)"""
        return isinstance(error, ErrorHandler.TRANSIENT_ERRORS)
    
    @staticmethod
    def log_error(
        error: Exception,
        context: str,
        include_trace: bool = True,
        level: str = "error"
    ) -> None:
        """Log error with context"""
        msg = f"{context}: {type(error).__name__}: {error}"
        
        if level == "critical":
            logger.critical(msg)
        elif level == "warning":
            logger.warning(msg)
        else:
            logger.error(msg)
        
        if include_trace:
            logger.debug(traceback.format_exc())
    
    @staticmethod
    async def handle_with_recovery(
        error: Exception,
        context: str,
        recovery_func: Optional[Callable] = None
    ) -> bool:
        """
        Handle error with optional recovery.
        
        Returns:
            bool: True if recovered, False if permanent error
        """
        is_transient = ErrorHandler.is_transient(error)
        
        ErrorHandler.log_error(
            error,
            context,
            include_trace=True,
            level="warning" if is_transient else "error"
        )
        
        if is_transient and recovery_func:
            try:
                logger.info(f"Attempting recovery for {context}...")
                if asyncio.iscoroutinefunction(recovery_func):
                    await recovery_func()
                else:
                    recovery_func()
                logger.info(f"✅ Recovery successful for {context}")
                return True
            except Exception as e:
                logger.error(f"❌ Recovery failed for {context}: {e}")
        
        return False


# استخدام في autonomous_runtime.py:
from okx_stream_hunter.utils.error_handler import ErrorHandler

async def _recover_stream(self):
    """Recover stream engine"""
    try:
        if self.stream_engine:
            await self.stream_engine.stop()
            await asyncio.sleep(2)
            asyncio.create_task(self.stream_engine.start())
            logger.info("✅ Stream Engine recovery initiated")
    except Exception as e:
        await ErrorHandler.handle_with_recovery(
            e,
            "Stream recovery",
            recovery_func=None
        )
```

---

## 🟡 الإصلاحات المتوسطة (مستحسنة)

### 7. إضافة PnL Calculation

**الملف:** `main.py`

**إضافة دالة قبل `ai_brain_ultra_loop`:**

```python
def calculate_pnl(position: Dict[str, Any], current_price: Optional[float]) -> float:
    """
    Calculate P&L for current position.
    
    Args:
        position: Position dict with direction, size, entry_price
        current_price: Current market price
    
    Returns:
        float: P&L in quote currency (negative = loss)
    """
    if not position or not current_price:
        return 0.0
    
    if position.get('direction') == 'flat':
        return 0.0
    
    entry = position.get('entry_price')
    size = position.get('size', 0.0)
    
    if not entry or size == 0:
        return 0.0
    
    if position['direction'] == 'long':
        return (current_price - entry) * size
    elif position['direction'] == 'short':
        return (entry - current_price) * size
    
    return 0.0
```

**استخدامه في `ai_brain_ultra_loop`:**

```python
# استبدال السطر:
pnl=0.0,  # TODO: calculate PnL

# بـ:
pnl=calculate_pnl(position, price_for_decision),
```

---

### 8. إضافة Uptime Calculation

**في `main.py`:**

```python
# في بداية main():
system_start_time = datetime.now(timezone.utc)

# في ai_brain_ultra_loop:
# استبدال:
uptime=0.0,  # TODO: calculate uptime

# بـ:
uptime=(datetime.now(timezone.utc) - system_start_time).total_seconds(),
```

---

### 9. تحسين Logging Levels

**في جميع الملفات، استبدال:**

```python
# ❌ من:
logger.info("✅ AI Brain received TICKER: ...")
logger.info("✅ AI Brain received ORDERBOOK: ...")

# ✅ إلى:
logger.debug("Ticker received: price={price}")
logger.debug("Orderbook updated: bid={bid}, ask={ask}")
```

**وفي الإنتاج:**

```python
# في main.py أو run_trading.py:
def setup_logging():
    level = logging.DEBUG if os.getenv("DEBUG", "0") == "1" else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
```

---

## 📋 Checklist للتطبيق

### المرحلة 1: الإصلاحات الحرجة (يجب تطبيقها الآن)
- [ ] 1. إصلاح OKXStreamEngine Integration
- [ ] 2. إضافة AI Brain للـ Runtime
- [ ] 3. إصلاح Stream Health Check
- [ ] 4. إضافة Auto-Reconnect
- [ ] 5. إصلاح Memory Leak
- [ ] 6. تحسين Error Handling

### المرحلة 2: التحسينات المتوسطة (خلال أسبوع)
- [ ] 7. إضافة PnL Calculation
- [ ] 8. إضافة Uptime Calculation
- [ ] 9. تحسين Logging Levels

### اختبار بعد التطبيق:
- [ ] تشغيل `python main.py` - يجب أن يعمل بدون أخطاء
- [ ] تشغيل `python run_trading.py` - يجب أن يعمل بدون أخطاء
- [ ] التحقق من Dashboard على `http://localhost:8000`
- [ ] مراقبة Logs للتأكد من عدم وجود أخطاء
- [ ] اختبار reconnection (قطع الإنترنت ثم إعادته)

---

**ملاحظة مهمة:** قم بعمل backup للمشروع قبل تطبيق أي تعديلات!

```bash
# Backup
cp -r /workspaces/crypto-market-analyzer /workspaces/crypto-market-analyzer-backup-$(date +%Y%m%d)
```
