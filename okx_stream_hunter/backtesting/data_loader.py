"""
Historical Data Loader
Load historical data from database or files
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    import asyncpg
except ImportError:
    asyncpg = None

from ..utils.logger import get_logger


logger = get_logger(__name__)


class HistoricalDataLoader:
    """
    Load historical market data for backtesting.
    
    Sources:
    - Neon database
    - JSON files
    - CSV files
    """
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Connect to database"""
        if not self.db_url or not asyncpg:
            logger.warning("Database not available for historical data")
            return
        
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=3)
            logger.info("Connected to database for historical data")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.pool:
            await self.pool.close()
    
    async def load_from_database(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Load candles from database.
        
        Returns:
            List of candle dicts
        """
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        table_name = f"candles_{timeframe}"
        
        query = f"""
            SELECT 
                symbol, open_time, close_time,
                open, high, low, close, volume, trades
            FROM {table_name}
            WHERE symbol = $1
                AND open_time >= $2
                AND open_time < $3
            ORDER BY open_time ASC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_time, end_time)
        
        candles = [
            {
                'symbol': row['symbol'],
                'timeframe': timeframe,
                'open_time': row['open_time'].isoformat(),
                'close_time': row['close_time'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'trades': row['trades'],
            }
            for row in rows
        ]
        
        logger.info(
            f"Loaded {len(candles)} candles for {symbol} {timeframe} "
            f"from {start_time} to {end_time}"
        )
        
        return candles
    
    def load_from_json(self, filepath: str) -> List[Dict]:
        """Load data from JSON file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    
    def save_to_json(self, data: List[Dict], filepath: str):
        """Save data to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} records to {filepath}")