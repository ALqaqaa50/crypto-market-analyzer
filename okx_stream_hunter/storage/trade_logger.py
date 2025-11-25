"""
PHASE 4: Trade Logger
Persistent logging of experiences to disk with automatic rotation
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs all trading experiences to disk for offline training
    Supports automatic file rotation (daily files)
    """
    
    def __init__(
        self,
        storage_dir: str = "storage/experiences",
        format: str = "parquet",
        rotation_interval: str = "daily"
    ):
        self.storage_dir = Path(storage_dir)
        self.format = format.lower()
        self.rotation_interval = rotation_interval
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_date = datetime.utcnow().date()
        self.current_file = None
        self.buffer = []
        self.buffer_size_limit = 100
        
        # Statistics
        self.stats = {
            'total_logged': 0,
            'files_created': 0,
            'last_flush': None,
            'errors': 0
        }
        
        logger.info(f"📝 TradeLogger initialized (dir={storage_dir}, format={format})")
    
    def log_decision(
        self,
        timestamp: datetime,
        symbol: str,
        market_features: Dict[str, Any],
        ai_decision: Dict[str, Any],
        risk_context: Optional[Dict[str, Any]] = None
    ):
        """Log AI decision step"""
        record = {
            'timestamp': timestamp.isoformat(),
            'type': 'decision',
            'symbol': symbol,
            'price': market_features.get('price', 0),
            'bid': market_features.get('bid', 0),
            'ask': market_features.get('ask', 0),
            'spread': market_features.get('spread', 0),
            'volume_24h': market_features.get('volume_24h', 0),
            
            # Orderflow metrics
            'buy_volume': market_features.get('buy_volume', 0),
            'sell_volume': market_features.get('sell_volume', 0),
            'orderbook_imbalance': market_features.get('orderbook_imbalance', 0),
            'spoof_risk': market_features.get('spoof_risk', 0),
            
            # AI decision
            'action': ai_decision.get('direction', 'NEUTRAL'),
            'confidence': ai_decision.get('confidence', 0),
            'regime': ai_decision.get('regime', 'unknown'),
            'reason': ai_decision.get('reason', ''),
            
            # Risk context
            'current_equity': risk_context.get('equity', 0) if risk_context else 0,
            'daily_pnl': risk_context.get('daily_pnl', 0) if risk_context else 0,
            'open_positions': risk_context.get('open_positions', 0) if risk_context else 0,
        }
        
        self._add_to_buffer(record)
    
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        market_features: Dict[str, Any],
        ai_decision: Dict[str, Any],
        execution_result: Dict[str, Any],
        risk_context: Optional[Dict[str, Any]] = None
    ):
        """Log trade execution"""
        record = {
            'timestamp': timestamp.isoformat(),
            'type': 'trade',
            'symbol': symbol,
            'price': market_features.get('price', 0),
            'bid': market_features.get('bid', 0),
            'ask': market_features.get('ask', 0),
            
            # AI decision
            'action': ai_decision.get('direction', 'NEUTRAL'),
            'confidence': ai_decision.get('confidence', 0),
            'regime': ai_decision.get('regime', 'unknown'),
            
            # Execution
            'trade_id': execution_result.get('trade_id', ''),
            'order_id': execution_result.get('order_id', ''),
            'filled_size': execution_result.get('filled_size', 0),
            'avg_fill_price': execution_result.get('avg_fill_price', 0),
            'sl': execution_result.get('sl', 0),
            'tp': execution_result.get('tp', 0),
            
            # Risk
            'current_equity': risk_context.get('equity', 0) if risk_context else 0,
            'position_size': execution_result.get('filled_size', 0),
        }
        
        self._add_to_buffer(record)
    
    def log_trade_outcome(
        self,
        timestamp: datetime,
        symbol: str,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        direction: str,
        size: float,
        pnl: float,
        pnl_pct: float,
        duration_seconds: float,
        max_favorable_move: float = 0,
        max_adverse_move: float = 0,
        exit_reason: str = ''
    ):
        """Log completed trade outcome"""
        record = {
            'timestamp': timestamp.isoformat(),
            'type': 'outcome',
            'symbol': symbol,
            'trade_id': trade_id,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_seconds': duration_seconds,
            'max_favorable_move': max_favorable_move,
            'max_adverse_move': max_adverse_move,
            'exit_reason': exit_reason,
            'win': pnl > 0
        }
        
        self._add_to_buffer(record)
    
    def _add_to_buffer(self, record: Dict[str, Any]):
        """Add record to buffer and flush if needed"""
        self.buffer.append(record)
        self.stats['total_logged'] += 1
        
        # Check if need rotation
        current_date = datetime.utcnow().date()
        if current_date != self.current_date:
            self.flush()
            self.current_date = current_date
            self.current_file = None
        
        # Flush if buffer full
        if len(self.buffer) >= self.buffer_size_limit:
            self.flush()
    
    def flush(self):
        """Write buffer to disk"""
        if not self.buffer:
            return
        
        try:
            filepath = self._get_current_filepath()
            
            df_new = pd.DataFrame(self.buffer)
            
            # Append to existing file or create new
            if filepath.exists():
                if self.format == "parquet":
                    df_existing = pd.read_parquet(filepath)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_parquet(filepath, index=False)
                else:  # csv
                    df_new.to_csv(filepath, mode='a', header=False, index=False)
            else:
                if self.format == "parquet":
                    df_new.to_parquet(filepath, index=False)
                else:
                    df_new.to_csv(filepath, index=False)
                
                self.stats['files_created'] += 1
            
            self.buffer.clear()
            self.stats['last_flush'] = datetime.utcnow().isoformat()
            
            logger.debug(f"💾 Flushed {len(df_new)} records to {filepath.name}")
        
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ TradeLogger flush error: {e}")
    
    def _get_current_filepath(self) -> Path:
        """Get current log file path"""
        if self.current_file is None:
            date_str = self.current_date.strftime('%Y-%m-%d')
            ext = 'parquet' if self.format == 'parquet' else 'csv'
            self.current_file = self.storage_dir / f"trades_{date_str}.{ext}"
        
        return self.current_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'current_file': str(self.current_file) if self.current_file else None
        }
    
    def load_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: Optional[int] = None
    ) -> pd.DataFrame:
        """Load historical data from logs"""
        if days_back:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days_back)
        
        if not start_date:
            start_date = datetime.utcnow().date() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow().date()
        
        dfs = []
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            ext = 'parquet' if self.format == 'parquet' else 'csv'
            filepath = self.storage_dir / f"trades_{date_str}.{ext}"
            
            if filepath.exists():
                try:
                    if self.format == 'parquet':
                        df = pd.read_parquet(filepath)
                    else:
                        df = pd.read_csv(filepath)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
            
            current += timedelta(days=1)
        
        if not dfs:
            logger.warning("No data files found in date range")
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"📖 Loaded {len(combined)} records from {len(dfs)} files")
        
        return combined


# Global instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger(
    storage_dir: str = "storage/experiences",
    format: str = "parquet"
) -> TradeLogger:
    """Get global trade logger instance"""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger(storage_dir=storage_dir, format=format)
    return _trade_logger


def reset_trade_logger(storage_dir: str = "storage/experiences", format: str = "parquet"):
    """Reset global logger (for testing)"""
    global _trade_logger
    _trade_logger = TradeLogger(storage_dir=storage_dir, format=format)
