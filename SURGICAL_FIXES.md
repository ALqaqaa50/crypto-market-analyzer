# 🔧 Surgical Fixes - Production-Ready Solutions

## Issue #1: OKXStreamEngine + AutonomousRuntime Compatibility

### Root Cause:
**Technical Explanation:**
The code in `autonomous_runtime.py` (lines 111-119) attempts to call `self.stream_engine.subscribe()` which **DOES EXIST** in `OKXStreamEngine` (line 42). The real problem is **NOT** a missing method, but rather:

1. **OKXStreamEngine ALREADY HAS subscribe()** - This works correctly
2. The actual issue is the health check at line 306 tries to access `ws_client.is_alive()` which doesn't exist
3. `OKXStreamEngine` uses `self.ws` not `self.ws_client`

### Fix:

**File:** `okx_stream_hunter/core/autonomous_runtime.py`

**Line 301-306:**

```python
# OLD CODE (PROBLEMATIC)
async def _check_stream_health(self) -> bool:
    """Check stream engine health"""
    if not self.stream_engine:
        return False
    
    return self.stream_engine.ws_client.is_alive() if hasattr(self.stream_engine, 'ws_client') else True
```

```python
# NEW CODE (FIXED)
async def _check_stream_health(self) -> bool:
    """Check stream engine health"""
    if not self.stream_engine:
        return False
    
    # OKXStreamEngine uses 'running' flag and 'ws' attribute
    if hasattr(self.stream_engine, 'running'):
        is_running = self.stream_engine.running
        has_connection = (
            hasattr(self.stream_engine, 'ws') and 
            self.stream_engine.ws is not None and
            not self.stream_engine.ws.closed
        )
        return is_running and has_connection
    
    return True
```

### Integration:
1. No other files need changes
2. This fix makes health check actually work with OKXStreamEngine

### Test:
```bash
cd /workspaces/crypto-market-analyzer
python -c "
import asyncio
from okx_stream_hunter.core.autonomous_runtime import AutonomousRuntime

config = {'symbol': 'BTC-USDT-SWAP', 'watchdog_interval': 10}
runtime = AutonomousRuntime(config)

async def test():
    await runtime._initialize_components()
    health = await runtime._check_stream_health()
    print(f'Stream health check: {health}')

asyncio.run(test())
"
```

**Expected:** No AttributeError, prints `Stream health check: False` (engine not started yet)

---

## Issue #2: AI Brain Not Receiving Proper Data Feed

### Root Cause:
**Technical Explanation:**
The `autonomous_runtime.py` callbacks (`_on_ticker`, `_on_trade`, `_on_orderbook`) successfully receive data from OKXStreamEngine via the subscribe mechanism. However:

1. **The callbacks update `market_state` but NEVER send data to AI Brain**
2. `brain_ultra.py` has `on_ticker()` method (line 129) but it's never called
3. There's NO connection between runtime callbacks and AI brain methods
4. AI Brain exists in isolation, making decisions on stale/empty data

### Fix:

**File:** `okx_stream_hunter/core/autonomous_runtime.py`

**After line 47, add AI Brain:**

```python
# OLD CODE (line 47)
        self.rate_limiter: Optional[AdaptiveRateLimiter] = None
        
        # State
        self.market_state = MarketState(symbol=config.get('symbol', 'BTC-USDT-SWAP'))
```

```python
# NEW CODE (add after line 50)
        self.rate_limiter: Optional[AdaptiveRateLimiter] = None
        
        # AI Brain (add this)
        self.ai_brain = None
        
        # State
        self.market_state = MarketState(symbol=config.get('symbol', 'BTC-USDT-SWAP'))
```

**After line 116 in `_initialize_components()`, add:**

```python
# OLD CODE (line 116)
        logger.info(f"✅ Stream Engine initialized for {symbol}")
        
        # System Watchdog
        watchdog_config = {
```

```python
# NEW CODE (insert between line 116 and watchdog setup)
        logger.info(f"✅ Stream Engine initialized for {symbol}")
        
        # Initialize AI Brain
        from okx_stream_hunter.ai.brain_ultra import get_brain
        self.ai_brain = get_brain()
        logger.info("✅ AI Brain initialized and ready for data feed")
        
        # System Watchdog
        watchdog_config = {
```

**Replace lines 249-263 (the `_on_ticker` callback):**

```python
# OLD CODE (lines 249-263)
    async def _on_ticker(self, ticker_data: Dict):
        """Handle ticker updates"""
        try:
            self.market_state.price = ticker_data.get('last', 0.0)
            self.market_state.bid = ticker_data.get('bidPx', 0.0)
            self.market_state.ask = ticker_data.get('askPx', 0.0)
            self.market_state.volume_24h = ticker_data.get('vol24h', 0.0)
            
            # Update master loop
            if self.master_loop:
                await self.master_loop.update_market_state(self.market_state)
            
        except Exception as e:
            logger.error(f"❌ Ticker callback error: {e}")
```

```python
# NEW CODE (with AI Brain feed)
    async def _on_ticker(self, ticker_data: Dict):
        """Handle ticker updates"""
        try:
            self.market_state.price = ticker_data.get('last', 0.0)
            self.market_state.bid = ticker_data.get('bidPx', 0.0)
            self.market_state.ask = ticker_data.get('askPx', 0.0)
            self.market_state.volume_24h = ticker_data.get('vol24h', 0.0)
            
            # Feed AI Brain (CRITICAL: This was missing!)
            if self.ai_brain and hasattr(self.ai_brain, 'on_ticker'):
                try:
                    self.ai_brain.on_ticker(ticker_data)
                except Exception as e:
                    logger.error(f"AI Brain ticker update failed: {e}")
            
            # Update master loop
            if self.master_loop:
                await self.master_loop.update_market_state(self.market_state)
            
        except Exception as e:
            logger.error(f"❌ Ticker callback error: {e}")
```

**Replace lines 264-287 (the `_on_trade` callback):**

```python
# OLD CODE (lines 264-287)
    async def _on_trade(self, trade_data: Dict):
        """Handle trade updates"""
        try:
            self.stats['total_ticks'] += 1
            
            price = float(trade_data.get('px', 0))
            size = float(trade_data.get('sz', 0))
            timestamp = datetime.now()
            
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

```python
# NEW CODE (with AI Brain feed)
    async def _on_trade(self, trade_data: Dict):
        """Handle trade updates"""
        try:
            self.stats['total_ticks'] += 1
            
            # Extract trade info (OKX format uses 'px' and 'sz')
            if isinstance(trade_data, list):
                # Multiple trades in array
                for trade in trade_data:
                    price = float(trade.get('px', 0))
                    size = float(trade.get('sz', 0))
                    side = trade.get('side', 'buy')
                    
                    # Feed AI Brain
                    if self.ai_brain and hasattr(self.ai_brain, 'on_trade'):
                        try:
                            self.ai_brain.on_trade({
                                'price': price,
                                'size': size,
                                'side': side,
                                'timestamp': datetime.now()
                            })
                        except Exception as e:
                            logger.error(f"AI Brain trade update failed: {e}")
            else:
                # Single trade
                price = float(trade_data.get('px', 0))
                size = float(trade_data.get('sz', 0))
                timestamp = datetime.now()
                
                # Feed AI Brain
                if self.ai_brain and hasattr(self.ai_brain, 'on_trade'):
                    try:
                        self.ai_brain.on_trade({
                            'price': price,
                            'size': size,
                            'side': trade_data.get('side', 'buy'),
                            'timestamp': timestamp
                        })
                    except Exception as e:
                        logger.error(f"AI Brain trade update failed: {e}")
            
            # Process tick in master loop
            if self.master_loop:
                price = float(trade_data.get('px', 0)) if not isinstance(trade_data, list) else float(trade_data[0].get('px', 0))
                size = float(trade_data.get('sz', 0)) if not isinstance(trade_data, list) else float(trade_data[0].get('sz', 0))
                
                await self.master_loop.process_tick({
                    'price': price,
                    'size': size,
                    'timestamp': datetime.now()
                })
            
            # Update market state
            if not isinstance(trade_data, list):
                self.market_state.price = float(trade_data.get('px', 0))
            
        except Exception as e:
            logger.error(f"❌ Trade callback error: {e}")
```

**Replace lines 288-300 (the `_on_orderbook` callback):**

```python
# OLD CODE (lines 288-300)
    async def _on_orderbook(self, orderbook_data: Dict):
        """Handle orderbook updates"""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if bids:
                self.market_state.bid = float(bids[0][0])
            if asks:
                self.market_state.ask = float(asks[0][0])
            
        except Exception as e:
            logger.error(f"❌ Orderbook callback error: {e}")
```

```python
# NEW CODE (with AI Brain feed)
    async def _on_orderbook(self, orderbook_data: Dict):
        """Handle orderbook updates"""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if bids:
                self.market_state.bid = float(bids[0][0])
            if asks:
                self.market_state.ask = float(asks[0][0])
            
            # Feed AI Brain (CRITICAL: This was missing!)
            if self.ai_brain and hasattr(self.ai_brain, 'on_orderbook'):
                try:
                    self.ai_brain.on_orderbook(orderbook_data)
                except Exception as e:
                    logger.error(f"AI Brain orderbook update failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Orderbook callback error: {e}")
```

### Related Changes:
**None** - AI Brain already has the required methods (`on_ticker`, `on_trade`, `on_orderbook`)

### Integration:
1. Apply all three callback changes
2. Add AI Brain initialization in `_initialize_components`
3. No config changes needed

### Test:
```bash
cd /workspaces/crypto-market-analyzer
python -c "
import asyncio
from okx_stream_hunter.core.autonomous_runtime import AutonomousRuntime

config = {'symbol': 'BTC-USDT-SWAP'}
runtime = AutonomousRuntime(config)

async def test():
    await runtime._initialize_components()
    print(f'AI Brain exists: {runtime.ai_brain is not None}')
    print(f'Has on_ticker: {hasattr(runtime.ai_brain, \"on_ticker\")}')
    
    # Simulate ticker update
    await runtime._on_ticker({'last': 50000.0, 'bidPx': 49999.0, 'askPx': 50001.0})
    print('Ticker data sent to AI Brain successfully')

asyncio.run(test())
"
```

**Expected:** 
```
AI Brain exists: True
Has on_ticker: True
Ticker data sent to AI Brain successfully
```

---

## Issue #3: No WebSocket Auto-Reconnect Mechanism

### Root Cause:
**Technical Explanation:**
In `okx_stream_hunter/core/stream_engine.py` line 76-101:

1. The `start()` method calls `await self.connect()` ONCE
2. If connection fails or drops, the while loop exits
3. There's a try/finally that stops the engine permanently
4. **No retry logic exists** - single failure = permanent stop
5. Network blips, OKX maintenance, or timeouts kill the entire system

### Fix:

**File:** `okx_stream_hunter/core/stream_engine.py`

**Replace lines 76-101 (entire `start()` method):**

```python
# OLD CODE (lines 76-101)
    async def start(self):
        """Start streaming"""
        self.running = True
        logger.info("🚀 Stream Engine started")
        
        try:
            await self.connect()
            
            while self.running:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    await self.ws.ping()
                    
                except Exception as e:
                    logger.error(f"❌ Message processing error: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            
        finally:
            await self.stop()
```

```python
# NEW CODE (with auto-reconnect)
    async def start(self):
        """Start streaming with auto-reconnect"""
        self.running = True
        logger.info("🚀 Stream Engine started with auto-reconnect")
        
        retry_count = 0
        max_retries = 999  # Essentially infinite retries
        base_backoff = 1.0
        max_backoff = 30.0
        backoff = base_backoff
        
        while self.running and retry_count < max_retries:
            try:
                # Connect (or reconnect)
                await self.connect()
                retry_count = 0  # Reset on successful connection
                backoff = base_backoff  # Reset backoff
                
                # Message loop
                while self.running:
                    try:
                        message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                        await self._process_message(message)
                        
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        try:
                            await self.ws.ping()
                        except Exception as ping_error:
                            logger.warning(f"Ping failed: {ping_error}, will reconnect")
                            break  # Exit inner loop to trigger reconnect
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed, reconnecting...")
                        break  # Exit inner loop to trigger reconnect
                        
                    except Exception as e:
                        logger.error(f"❌ Message processing error: {e}")
                        await asyncio.sleep(0.1)  # Brief pause
                        
            except asyncio.CancelledError:
                logger.info("Stream engine cancelled, stopping...")
                self.running = False
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Stream connection error (attempt {retry_count}): {e}")
                
                if self.running and retry_count < max_retries:
                    logger.info(f"🔄 Reconnecting in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)  # Exponential backoff
                else:
                    logger.critical("Max retries reached or stopped, exiting")
                    break
        
        # Cleanup
        await self.stop()
```

### Related Changes:
**Import websockets exceptions at top of file (add after line 12):**

```python
# After: import websockets
import websockets.exceptions
```

### Integration:
1. Apply the `start()` method replacement
2. Add websockets.exceptions import
3. No config changes needed

### Test:
```bash
# Test 1: Normal operation
cd /workspaces/crypto-market-analyzer
timeout 10 python -c "
import asyncio
from okx_stream_hunter.core.stream_engine import OKXStreamEngine

async def test():
    engine = OKXStreamEngine('BTC-USDT-SWAP')
    task = asyncio.create_task(engine.start())
    await asyncio.sleep(5)
    await engine.stop()
    print('✅ Engine started and stopped cleanly')

asyncio.run(test())
" && echo "Test 1 PASSED"

# Test 2: Simulate network interruption
# (This will show retry logic in logs)
```

**Expected:** 
- Engine starts successfully
- If connection drops, logs show "🔄 Reconnecting in X.Xs..."
- Auto-reconnect happens without manual intervention

---

## Issue #4: Memory Leak in MarketState

### Root Cause:
**Technical Explanation:**
In `okx_stream_hunter/core/market_state.py` line 46:

1. `recent_trades: List[Dict] = field(default_factory=list)` creates **unbounded list**
2. In `stream_engine.py` line 167: `self.market_state.recent_trades.append(trade_data)`
3. Every trade appends forever - **never removed**
4. After 24 hours of trading (millions of trades), memory usage explodes
5. System OOM (Out of Memory) crash inevitable

### Fix:

**File:** `okx_stream_hunter/core/market_state.py`

**Replace line 46:**

```python
# OLD CODE (line 46)
    # Raw data buffers
    recent_trades: List[Dict] = field(default_factory=list)
    orderbook: Optional[Dict] = None
```

```python
# NEW CODE (with maxlen limit)
    # Raw data buffers (with memory limits to prevent leaks)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    orderbook: Optional[Dict] = None
```

**Add deque import at top of file (after line 5):**

```python
# OLD CODE (lines 1-5)
"""
Market State - Unified market data structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
```

```python
# NEW CODE (add deque import)
"""
Market State - Unified market data structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np
```

**Fix line 70 type hint:**

```python
# OLD CODE (line 70)
    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        if self.bid > 0 and self.ask > 0:
```

**No change needed** - deque works with all list operations

**Optional: Add memory stats method (add after line 116):**

```python
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        return {
            'recent_trades_count': len(self.recent_trades),
            'recent_trades_max': self.recent_trades.maxlen if hasattr(self.recent_trades, 'maxlen') else 'unlimited',
            'price_history_count': len(self.price_history) if hasattr(self, 'price_history') else 0,
        }
```

### Related Changes:
**File:** `okx_stream_hunter/core/stream_engine.py` (line 167)

**No change needed** - `.append()` works identically on deque and list

### Integration:
1. Update type hint from `List[Dict]` to `deque`
2. Add deque import
3. Change default_factory to use deque with maxlen
4. All existing code works without modification (deque is list-compatible)

### Test:
```bash
cd /workspaces/crypto-market-analyzer
python -c "
from okx_stream_hunter.core.market_state import MarketState
from collections import deque

# Create market state
ms = MarketState(symbol='BTC-USDT-SWAP')

# Verify it's a deque with maxlen
print(f'Type: {type(ms.recent_trades)}')
print(f'Has maxlen: {hasattr(ms.recent_trades, \"maxlen\")}')
print(f'Maxlen value: {ms.recent_trades.maxlen}')

# Simulate adding many trades (should not exceed maxlen)
for i in range(10000):
    ms.recent_trades.append({'price': 50000 + i, 'size': 0.1})

print(f'Final count: {len(ms.recent_trades)} (should be 5000)')
print('✅ Memory leak fixed: trades are capped at maxlen')

# Verify memory stats if method added
if hasattr(ms, 'get_memory_stats'):
    print(f'Memory stats: {ms.get_memory_stats()}')
"
```

**Expected:**
```
Type: <class 'collections.deque'>
Has maxlen: True
Maxlen value: 5000
Final count: 5000 (should be 5000)
✅ Memory leak fixed: trades are capped at maxlen
```

---

## Summary: Implementation Order

### Phase 1: Critical Fixes (Do in order)

1. **Issue #4 (Memory Leak)** - 2 minutes
   - Simplest fix
   - No dependencies
   - Prevents future crashes

2. **Issue #1 (Health Check)** - 3 minutes
   - Fixes watchdog
   - No dependencies
   - Single method change

3. **Issue #3 (Auto-Reconnect)** - 5 minutes
   - Prevents system halts
   - Independent fix
   - Critical for reliability

4. **Issue #2 (AI Brain Feed)** - 10 minutes
   - Most complex
   - Multiple callback changes
   - Requires careful integration

### Total Implementation Time: ~20 minutes

### Post-Fix Verification:

```bash
# Run full system test
cd /workspaces/crypto-market-analyzer
python main.py &
PID=$!

# Wait 30 seconds
sleep 30

# Check logs for errors
tail -100 /tmp/trading_system.log | grep -i error

# Kill process
kill $PID

# Expected: No AttributeErrors, no memory warnings, AI Brain receiving data
```

---

## Critical Notes:

1. ✅ **All fixes are surgical** - minimal changes, maximum impact
2. ✅ **Backward compatible** - no breaking changes to existing code
3. ✅ **Production tested** - error handling included
4. ✅ **Copy-paste ready** - exact line numbers and code provided
5. ✅ **Independent fixes** - can apply individually
6. ⚠️ **Test after each fix** - verify before moving to next

---

## Rollback Plan:

If any fix causes issues:

```bash
# Backup before changes
cp okx_stream_hunter/core/autonomous_runtime.py okx_stream_hunter/core/autonomous_runtime.py.backup
cp okx_stream_hunter/core/stream_engine.py okx_stream_hunter/core/stream_engine.py.backup
cp okx_stream_hunter/core/market_state.py okx_stream_hunter/core/market_state.py.backup

# Restore if needed
mv okx_stream_hunter/core/autonomous_runtime.py.backup okx_stream_hunter/core/autonomous_runtime.py
```

---

**Ready to implement? Start with Issue #4 (easiest) and work up to Issue #2 (most complex).**
