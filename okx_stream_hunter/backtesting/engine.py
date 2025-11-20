"""
Backtesting Engine
Replay historical data through the system
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timezone

# StreamEngine will be passed as parameter
# from ..core.engine import StreamEngine
from ..utils.logger import get_logger
from .data_loader import HistoricalDataLoader


logger = get_logger(__name__)


class BacktestEngine:
    """
    Backtest engine to replay historical data.
    
    Features:
    - Replay at any speed
    - Skip database writes (optional)
    - Collect performance metrics
    - Generate reports
    """
    
    def __init__(
        self,
        stream_engine,  # StreamEngine type
        data_loader: HistoricalDataLoader,
        speed_multiplier: float = 1.0,
        enable_db_writes: bool = False
    ):
        """
        Args:
            stream_engine: Main stream engine
            data_loader: Historical data loader
            speed_multiplier: Replay speed (1.0 = real-time, 10.0 = 10x)
            enable_db_writes: Whether to write to database during backtest
        """
        self.engine = stream_engine
        self.loader = data_loader
        self.speed = speed_multiplier
        self.enable_db = enable_db_writes
        
        # Metrics
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.candles_processed = 0
        self.trades_processed = 0
        self.patterns_detected = 0
        self.whales_detected = 0
    
    async def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ):
        """
        Run backtest for specified period.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_time: Start of backtest period
            end_time: End of backtest period
        """
        logger.info("="*60)
        logger.info("🔄 STARTING BACKTEST")
        logger.info("="*60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {start_time} to {end_time}")
        logger.info(f"Speed: {self.speed}x")
        logger.info(f"Database writes: {'Enabled' if self.enable_db else 'Disabled'}")
        logger.info("="*60)
        
        self.start_time = datetime.now(timezone.utc)
        
        # Disable database writes if requested
        if not self.enable_db:
            original_write = self.engine.db_writer.write_candle
            self.engine.db_writer.write_candle = lambda *args, **kwargs: None
        
        # Load historical data
        candles = await self.loader.load_from_database(
            symbol, timeframe, start_time, end_time
        )
        
        if not candles:
            logger.error("No historical data found")
            return
        
        # Replay candles
        for i, candle in enumerate(candles, 1):
            # Simulate candle closing
            await self._process_candle(candle)
            
            self.candles_processed += 1
            
            # Progress
            if i % 100 == 0:
                progress = (i / len(candles)) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{len(candles)})")
            
            # Sleep to simulate time passing
            if self.speed < float('inf'):
                await asyncio.sleep(1.0 / self.speed)
        
        self.end_time = datetime.now(timezone.utc)
        
        # Restore database writes
        if not self.enable_db:
            self.engine.db_writer.write_candle = original_write
        
        self._print_summary()
    
    async def _process_candle(self, candle: Dict):
        """Process a single candle through the engine"""
        # Convert to engine's candle format if needed
        # For now, just trigger indicator calculation
        
        snap, snap_dict = self.engine.indicator_engine.update(
            candle['timeframe'], candle
        )
        
        # Track detections
        # (In real implementation, hook into detection callbacks)
    
    def _print_summary(self):
        """Print backtest summary"""
        duration = (self.end_time - self.start_time).total_seconds()
        
        logger.info("="*60)
        logger.info("📊 BACKTEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Candles processed: {self.candles_processed}")
        logger.info(f"Processing speed: {self.candles_processed/duration:.1f} candles/s")
        logger.info("="*60)