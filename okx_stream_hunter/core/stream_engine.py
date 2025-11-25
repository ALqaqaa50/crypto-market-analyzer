"""
Stream Engine - Real-time WebSocket streaming from OKX
Handles TICKER, TRADES, ORDERBOOK subscriptions
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional
from collections import deque
import websockets
import websockets.exceptions

from okx_stream_hunter.core.market_state import MarketState

logger = logging.getLogger(__name__)


class OKXStreamEngine:
    """Real-time WebSocket streaming from OKX"""
    
    WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
    
    def __init__(self, symbol: str = "BTC-USDT-SWAP"):
        self.symbol = symbol
        self.ws = None
        self.running = False
        
        self.market_state = MarketState(symbol=symbol)
        self.trade_buffer = deque(maxlen=1000)
        
        self.callbacks = {
            'ticker': [],
            'trades': [],
            'orderbook': [],
            'state_update': []
        }
        
        logger.info(f"🌊 Stream Engine initialized for {symbol}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to stream events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"✅ Subscribed to {event_type}")
    
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(self.WS_URL)
            logger.info(f"✅ Connected to OKX WebSocket")
            
            await self._subscribe_channels()
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    async def _subscribe_channels(self):
        """Subscribe to market data channels"""
        subscriptions = [
            {
                "op": "subscribe",
                "args": [
                    {"channel": "tickers", "instId": self.symbol},
                    {"channel": "trades", "instId": self.symbol},
                    {"channel": "books5", "instId": self.symbol}
                ]
            }
        ]
        
        for sub in subscriptions:
            await self.ws.send(json.dumps(sub))
            logger.info(f"📡 Subscribed to channels for {self.symbol}")
    
    async def start(self):
        """Start streaming with auto-reconnect"""
        self.running = True
        logger.info("🚀 Stream Engine started")
        
        retry_count = 0
        max_retries = 5
        base_delay = 2
        
        while self.running and retry_count < max_retries:
            try:
                await self.connect()
                retry_count = 0  # Reset on successful connection
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                        await self._process_message(message)
                        
                    except asyncio.TimeoutError:
                        await self.ws.ping()
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("⚠️ WebSocket connection closed, reconnecting...")
                        raise  # Re-raise to trigger reconnection in outer try-catch
                        
                    except Exception as e:
                        logger.error(f"❌ Message processing error: {e}")
                        await asyncio.sleep(1)
                        
            except websockets.exceptions.WebSocketException as e:
                retry_count += 1
                delay = min(base_delay * (2 ** retry_count), 60)
                logger.error(f"❌ WebSocket error (attempt {retry_count}/{max_retries}): {e}")
                logger.info(f"🔄 Retrying in {delay}s...")
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"❌ Unexpected stream error: {e}")
                break
        
        if retry_count >= max_retries:
            logger.error("❌ Max reconnection attempts reached. Stopping stream.")
        
        await self.stop()
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'event' in data:
                return
            
            if 'data' not in data:
                return
            
            channel = data.get('arg', {}).get('channel')
            items = data['data']
            
            if channel == 'tickers':
                await self._process_ticker(items[0])
            elif channel == 'trades':
                await self._process_trades(items)
            elif channel == 'books5':
                await self._process_orderbook(items[0])
                
        except Exception as e:
            logger.error(f"❌ Message parsing error: {e}")
    
    async def _process_ticker(self, ticker: Dict):
        """Process ticker update"""
        try:
            self.market_state.price = float(ticker.get('last', 0))
            self.market_state.bid = float(ticker.get('bidPx', 0))
            self.market_state.ask = float(ticker.get('askPx', 0))
            self.market_state.volume_24h = float(ticker.get('vol24h', 0))
            self.market_state.timestamp = datetime.now()
            
            for callback in self.callbacks['ticker']:
                try:
                    await callback(ticker)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
            
            await self._update_state()
            
        except Exception as e:
            logger.error(f"❌ Ticker processing error: {e}")
    
    async def _process_trades(self, trades: List[Dict]):
        """Process trades update"""
        try:
            for trade in trades:
                trade_data = {
                    'price': float(trade.get('px', 0)),
                    'size': float(trade.get('sz', 0)),
                    'side': trade.get('side', 'buy'),
                    'timestamp': datetime.fromtimestamp(int(trade.get('ts', 0)) / 1000)
                }
                
                self.trade_buffer.append(trade_data)
                self.market_state.recent_trades.append(trade_data)
                
                if trade_data['side'] == 'buy':
                    self.market_state.buy_volume += trade_data['size']
                else:
                    self.market_state.sell_volume += trade_data['size']
                
                self.market_state.trade_count += 1
            
            for callback in self.callbacks['trades']:
                try:
                    await callback(trades)
                except Exception as e:
                    logger.error(f"Trades callback error: {e}")
            
            await self._update_state()
            
        except Exception as e:
            logger.error(f"❌ Trades processing error: {e}")
    
    async def _process_orderbook(self, orderbook: Dict):
        """Process orderbook update"""
        try:
            self.market_state.orderbook = {
                'bids': orderbook.get('bids', []),
                'asks': orderbook.get('asks', []),
                'timestamp': datetime.fromtimestamp(int(orderbook.get('ts', 0)) / 1000)
            }
            
            for callback in self.callbacks['orderbook']:
                try:
                    await callback(orderbook)
                except Exception as e:
                    logger.error(f"Orderbook callback error: {e}")
            
            await self._update_state()
            
        except Exception as e:
            logger.error(f"❌ Orderbook processing error: {e}")
    
    async def _update_state(self):
        """Update market state and notify subscribers"""
        try:
            self.market_state.calculate_derived_metrics()
            
            for callback in self.callbacks['state_update']:
                try:
                    await callback(self.market_state)
                except Exception as e:
                    logger.error(f"State update callback error: {e}")
                    
        except Exception as e:
            logger.error(f"❌ State update error: {e}")
    
    def get_latest_market_state(self) -> MarketState:
        """Get current market state"""
        return self.market_state
    
    async def stop(self):
        """Stop streaming"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("⏹️ Stream Engine stopped")
