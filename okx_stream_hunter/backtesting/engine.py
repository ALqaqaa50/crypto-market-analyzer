"""
🔥 Backtesting Engine - AI Strategy Backtesting & Optimization
"""
import asyncio
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from ..core.ai_brain import AIBrain
from ..core.trading_engine import TradingEngine, TradingEngineConfig
from ..integrations.risk_manager import RiskManager, RiskConfig
from ..integrations.position_manager import PositionManager
from ..utils.logger import get_logger
from .data_loader import HistoricalDataLoader


logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Backtest results and metrics"""
    
    # Period
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Trading stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L
    total_pnl: float
    total_pnl_pct: float
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    
    # Performance
    candles_processed: int
    signals_generated: int
    
    # Config used
    config: Dict[str, Any]


class BacktestEngine:
    """
    🔥 Advanced Backtest Engine for AI Strategies
    
    Features:
    - Replay historical market data
    - Test AI Brain strategies
    - Complete trading simulation with Risk & Position management
    - Performance metrics & reporting
    - Parameter optimization support
    """
    
    def __init__(
        self,
        ai_brain: AIBrain,
        data_loader: HistoricalDataLoader,
        speed_multiplier: float = 100.0,  # Fast replay by default
        initial_balance: float = 1000.0,
        risk_config: Optional[RiskConfig] = None,
        trading_config: Optional[TradingEngineConfig] = None,
    ):
        """
        Args:
            ai_brain: AI Brain for signal generation
            data_loader: Historical data loader
            speed_multiplier: Replay speed (100 = 100x faster)
            initial_balance: Starting capital
            risk_config: Risk management config
            trading_config: Trading engine config
        """
        self.ai_brain = ai_brain
        self.loader = data_loader
        self.speed = speed_multiplier
        
        # Initialize trading components (paper trading mode)
        self.risk_config = risk_config or RiskConfig(account_balance=initial_balance)
        self.trading_config = trading_config or TradingEngineConfig()
        
        self.risk_manager = RiskManager(config=self.risk_config)
        self.position_manager = PositionManager()
        
        # Don't need real executor in backtest
        self.trading_engine = None  # Will be initialized without executor
        
        # Backtest state
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.current_balance: float = initial_balance
        
        # Metrics
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.candles_processed = 0
        self.signals_generated = 0
    
    async def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> BacktestResult:
        """
        Run backtest for specified period.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_time: Start of backtest period
            end_time: End of backtest period
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info("="*60)
        logger.info("🔄 STARTING AI BACKTEST")
        logger.info("="*60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {start_time} to {end_time}")
        logger.info(f"Speed: {self.speed}x")
        logger.info(f"Initial Balance: ${self.current_balance:.2f}")
        logger.info("="*60)
        
        self.start_time = datetime.now(timezone.utc)
        
        # Load historical data
        candles = await self.loader.load_from_database(
            symbol, timeframe, start_time, end_time
        )
        
        if not candles:
            logger.error("No historical data found")
            return self._create_empty_result()
        
        logger.info(f"Loaded {len(candles)} candles")
        
        # Replay candles
        for i, candle in enumerate(candles, 1):
            await self._process_candle(candle)
            
            self.candles_processed += 1
            
            # Progress
            if i % 100 == 0:
                progress = (i / len(candles)) * 100
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Balance: ${self.current_balance:.2f} | "
                    f"Trades: {len(self.trades)}"
                )
            
            # Sleep to simulate time passing
            if self.speed < float('inf'):
                await asyncio.sleep(1.0 / self.speed)
        
        self.end_time = datetime.now(timezone.utc)
        
        # Generate result
        result = self._calculate_results()
        self._print_summary(result)
        
        return result
    
    async def _process_candle(self, candle: Dict) -> None:
        """Process a single candle - feed to AI Brain and simulate trading"""
        
        # Update AI Brain with candle data
        # Simulate ticker update
        ticker = {
            "last": candle.get("close"),
            "ts": candle.get("timestamp"),
        }
        self.ai_brain.update_from_ticker(ticker)
        
        # Get AI signal
        signal = self.ai_brain.build_signal()
        self.signals_generated += 1
        
        # Simulate trading decision
        direction = signal["direction"]
        confidence = signal["confidence"]
        price = signal.get("price")
        
        if direction == "flat" or not price:
            # Close any open position
            self._close_position_if_open(price or candle.get("close"))
            return
        
        # Check if should open position
        if confidence < 0.35:  # Min confidence
            return
        
        # Calculate SL/TP
        sl_pct = 0.01
        if direction == "long":
            sl_price = price * (1.0 - sl_pct)
            tp_price = price * (1.0 + sl_pct * 2.0)
        else:
            sl_price = price * (1.0 + sl_pct)
            tp_price = price * (1.0 - sl_pct * 2.0)
        
        # Risk assessment
        risk_assessment = self.risk_manager.assess_trade(
            symbol=self.ai_brain.symbol,
            direction=direction,
            entry_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            confidence=confidence,
        )
        
        if not risk_assessment.approved:
            return
        
        # Open position (paper trading)
        current_position = self.position_manager.get_position(self.ai_brain.symbol)
        
        if not current_position:
            # Open new position
            position = self.position_manager.open_position(
                symbol=self.ai_brain.symbol,
                direction=direction,
                size=risk_assessment.position_size,
                entry_price=price,
                tp_price=risk_assessment.tp_price,
                sl_price=risk_assessment.sl_price,
                reason=signal.get("reason", ""),
                confidence=confidence,
            )
            
            self.risk_manager.register_position(
                symbol=self.ai_brain.symbol,
                direction=direction,
                size=risk_assessment.position_size,
                entry_price=price,
            )
        
        # Check TP/SL on current candle
        self._check_tp_sl(candle)
        
        # Record equity
        self._record_equity(candle.get("timestamp"))
    
    def _check_tp_sl(self, candle: Dict) -> None:
        """Check if TP or SL hit in current candle"""
        
        position = self.position_manager.get_position(self.ai_brain.symbol)
        if not position:
            return
        
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
        
        hit_tp = False
        hit_sl = False
        exit_price = close
        
        # Check TP/SL
        if position.direction == "long":
            if position.tp_price and high >= position.tp_price:
                hit_tp = True
                exit_price = position.tp_price
            elif position.sl_price and low <= position.sl_price:
                hit_sl = True
                exit_price = position.sl_price
        else:  # short
            if position.tp_price and low <= position.tp_price:
                hit_tp = True
                exit_price = position.tp_price
            elif position.sl_price and high >= position.sl_price:
                hit_sl = True
                exit_price = position.sl_price
        
        if hit_tp or hit_sl:
            self._close_position(exit_price, "tp_hit" if hit_tp else "sl_hit")
    
    def _close_position_if_open(self, price: float) -> None:
        """Close position if open (flat signal)"""
        position = self.position_manager.get_position(self.ai_brain.symbol)
        if position:
            self._close_position(price, "signal_flat")
    
    def _close_position(self, exit_price: float, reason: str) -> None:
        """Close position and record trade"""
        
        position = self.position_manager.get_position(self.ai_brain.symbol)
        if not position:
            return
        
        # Calculate P&L
        if position.direction == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        pnl = pnl_pct * position.size * position.entry_price
        
        # Update balance
        self.current_balance += pnl
        self.risk_config.account_balance = self.current_balance
        
        # Close in manager
        self.position_manager.close_position(
            symbol=self.ai_brain.symbol,
            close_price=exit_price,
            reason=reason,
        )
        
        is_win = pnl > 0
        self.risk_manager.close_position(
            symbol=self.ai_brain.symbol,
            exit_price=exit_price,
            is_win=is_win,
        )
        
        # Record trade
        trade = {
            "entry_time": position.opened_at.isoformat(),
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "is_win": is_win,
        }
        self.trades.append(trade)
        
        logger.debug(
            f"{'✅ WIN' if is_win else '❌ LOSS'}: "
            f"{position.direction} ${position.entry_price:.2f} → ${exit_price:.2f} | "
            f"PnL: ${pnl:.2f} ({pnl_pct:.2%})"
        )
    
    def _record_equity(self, timestamp: Any) -> None:
        """Record current equity"""
        position = self.position_manager.get_position(self.ai_brain.symbol)
        unrealized_pnl = position.unrealized_pnl if position else 0.0
        
        self.equity_curve.append({
            "timestamp": timestamp,
            "balance": self.current_balance,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": self.current_balance + unrealized_pnl,
        })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results"""
        
        if not self.trades:
            return self._create_empty_result()
        
        wins = [t for t in self.trades if t["is_win"]]
        losses = [t for t in self.trades if not t["is_win"]]
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_pnl_pct = (total_pnl / self.risk_config.account_balance) * 100
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
        
        best_trade = max((t["pnl"] for t in self.trades), default=0.0)
        worst_trade = min((t["pnl"] for t in self.trades), default=0.0)
        
        # Calculate max drawdown
        max_equity = self.risk_config.account_balance
        max_dd = 0.0
        for point in self.equity_curve:
            equity = point["total_equity"]
            if equity > max_equity:
                max_equity = equity
            dd = max_equity - equity
            if dd > max_dd:
                max_dd = dd
        
        max_dd_pct = (max_dd / max_equity) * 100 if max_equity > 0 else 0.0
        
        # Sharpe ratio (simplified)
        returns = [t["pnl_pct"] for t in self.trades]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        std_dev = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5 if returns else 0.0
        sharpe = (avg_return / std_dev * (252**0.5)) if std_dev > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        return BacktestResult(
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=duration,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            candles_processed=self.candles_processed,
            signals_generated=self.signals_generated,
            config={
                "risk_config": asdict(self.risk_config),
                "trading_config": asdict(self.trading_config),
            },
        )
    
    def _create_empty_result(self) -> BacktestResult:
        """Create empty result for failed backtest"""
        return BacktestResult(
            start_time=self.start_time or datetime.now(timezone.utc),
            end_time=self.end_time or datetime.now(timezone.utc),
            duration_seconds=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            candles_processed=self.candles_processed,
            signals_generated=self.signals_generated,
            config={},
        )
    
    def _print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary"""
        logger.info("="*60)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Candles: {result.candles_processed}")
        logger.info(f"Signals: {result.signals_generated}")
        logger.info("")
        logger.info("TRADING STATS:")
        logger.info(f"  Total Trades: {result.total_trades}")
        logger.info(f"  Wins: {result.winning_trades} | Losses: {result.losing_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.1%}")
        logger.info("")
        logger.info("P&L:")
        logger.info(f"  Total P&L: ${result.total_pnl:.2f} ({result.total_pnl_pct:+.2f}%)")
        logger.info(f"  Best Trade: ${result.best_trade:.2f}")
        logger.info(f"  Worst Trade: ${result.worst_trade:.2f}")
        logger.info(f"  Avg Win: ${result.avg_win:.2f}")
        logger.info(f"  Avg Loss: ${result.avg_loss:.2f}")
        logger.info("")
        logger.info("RISK METRICS:")
        logger.info(f"  Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info("="*60)
        
        # Save to file
        try:
            with open("backtest_result.json", "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.info("Results saved to backtest_result.json")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def export_trades(self, filename: str = "backtest_trades.json") -> None:
        """Export all trades to JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.trades, f, indent=2)
            logger.info(f"Trades exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export trades: {e}")
    
    def export_equity_curve(self, filename: str = "backtest_equity.json") -> None:
        """Export equity curve to JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.equity_curve, f, indent=2, default=str)
            logger.info(f"Equity curve exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export equity curve: {e}")