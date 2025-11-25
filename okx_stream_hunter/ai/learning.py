# okx_stream_hunter/ai/learning.py
"""
🔥 Self-Learning Engine - Pattern Recognition & Auto-Optimization
"""

import asyncio
import json
import logging
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("ai.learning")


@dataclass
class TradePattern:
    """Identified trading pattern"""
    pattern_id: str
    pattern_type: str  # trend_follow, mean_reversion, breakout, etc.
    
    # Entry conditions
    entry_conditions: Dict[str, Any]
    
    # Performance metrics
    total_occurrences: int
    successful_trades: int
    failed_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    
    # Confidence
    confidence: float
    last_seen: str


@dataclass
class HyperparameterSet:
    """Hyperparameter configuration"""
    name: str
    parameters: Dict[str, Any]
    
    # Performance
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Optimization score
    score: float
    last_updated: str


class SelfLearningEngine:
    """
    🔥 Self-Learning AI Engine
    
    Features:
    - Pattern recognition (identify winning setups)
    - Historical analysis (learn from past trades)
    - Auto hyperparameter tuning (optimize settings)
    - Performance tracking and adaptation
    """
    
    def __init__(
        self,
        data_dir: str = "data/learning",
        min_pattern_occurrences: int = 10,
        pattern_lookback_days: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_pattern_occurrences = min_pattern_occurrences
        self.pattern_lookback_days = pattern_lookback_days
        
        # Pattern storage
        self.patterns: Dict[str, TradePattern] = {}
        self.pattern_history = deque(maxlen=1000)
        
        # Hyperparameter sets
        self.hyperparameter_sets: Dict[str, HyperparameterSet] = {}
        self.current_hyperparams: Optional[str] = None
        
        # Trade history for learning
        self.trade_history = deque(maxlen=500)
        
        # Performance tracking
        self.learning_iterations = 0
        self.last_optimization = None
        
        self._running = False
        self._learning_task: Optional[asyncio.Task] = None
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    async def start(self) -> None:
        """Start self-learning engine"""
        logger.info("🧠 Self-Learning Engine starting...")
        
        # Load existing patterns and hyperparameters
        self._load_patterns()
        self._load_hyperparameters()
        self._load_trade_history()
        
        self._running = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        
        logger.info(
            f"✅ Self-Learning Engine started: "
            f"{len(self.patterns)} patterns, "
            f"{len(self.hyperparameter_sets)} hyperparameter sets"
        )
    
    async def stop(self) -> None:
        """Stop self-learning engine"""
        logger.info("Stopping Self-Learning Engine...")
        self._running = False
        
        if self._learning_task and not self._learning_task.done():
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        # Save all data
        self._save_patterns()
        self._save_hyperparameters()
        self._save_trade_history()
        
        logger.info("✅ Self-Learning Engine stopped")
    
    # ============================================================
    # Pattern Recognition
    # ============================================================
    
    def record_trade(
        self,
        signal_data: Dict[str, Any],
        entry_price: float,
        exit_price: float,
        pnl: float,
        duration_minutes: int,
        success: bool,
    ) -> None:
        """Record a completed trade for learning"""
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal_data,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": (exit_price - entry_price) / entry_price,
            "duration_minutes": duration_minutes,
            "success": success,
        }
        
        self.trade_history.append(trade)
        logger.info(f"📝 Trade recorded: PnL={pnl:.2f}, Success={success}")
        
        # Trigger pattern analysis
        asyncio.create_task(self._analyze_new_trade(trade))
    
    async def _analyze_new_trade(self, trade: Dict) -> None:
        """Analyze new trade and update patterns"""
        try:
            # Extract features from signal
            signal = trade["signal"]
            
            # Create pattern signature
            pattern_sig = self._create_pattern_signature(signal)
            
            if pattern_sig in self.patterns:
                # Update existing pattern
                pattern = self.patterns[pattern_sig]
                pattern.total_occurrences += 1
                
                if trade["success"]:
                    pattern.successful_trades += 1
                    pattern.avg_profit = (
                        pattern.avg_profit * (pattern.successful_trades - 1) + trade["pnl"]
                    ) / pattern.successful_trades
                else:
                    pattern.failed_trades += 1
                    pattern.avg_loss = (
                        pattern.avg_loss * (pattern.failed_trades - 1) + abs(trade["pnl"])
                    ) / pattern.failed_trades
                
                # Update metrics
                pattern.win_rate = pattern.successful_trades / pattern.total_occurrences
                pattern.profit_factor = (
                    pattern.avg_profit / pattern.avg_loss if pattern.avg_loss > 0 else 0
                )
                pattern.confidence = self._calculate_pattern_confidence(pattern)
                pattern.last_seen = datetime.now(timezone.utc).isoformat()
                
                logger.info(
                    f"📊 Pattern updated: {pattern.pattern_id}, "
                    f"win_rate={pattern.win_rate:.2%}, confidence={pattern.confidence:.2f}"
                )
            
            else:
                # Create new pattern
                pattern = TradePattern(
                    pattern_id=pattern_sig,
                    pattern_type=self._classify_pattern_type(signal),
                    entry_conditions=self._extract_entry_conditions(signal),
                    total_occurrences=1,
                    successful_trades=1 if trade["success"] else 0,
                    failed_trades=0 if trade["success"] else 1,
                    win_rate=1.0 if trade["success"] else 0.0,
                    avg_profit=trade["pnl"] if trade["success"] else 0.0,
                    avg_loss=abs(trade["pnl"]) if not trade["success"] else 0.0,
                    profit_factor=0.0,
                    confidence=0.3,  # Low initial confidence
                    last_seen=datetime.now(timezone.utc).isoformat(),
                )
                
                self.patterns[pattern_sig] = pattern
                logger.info(f"🆕 New pattern created: {pattern_sig}")
        
        except Exception as e:
            logger.exception(f"Error analyzing trade: {e}")
    
    def _create_pattern_signature(self, signal: Dict) -> str:
        """Create unique signature for a signal pattern"""
        # Extract key features
        direction = signal.get("direction", "unknown")
        confidence_bucket = int(signal.get("confidence", 0) * 10) / 10  # Round to 0.1
        regime = signal.get("regime", "unknown")
        
        # Create signature
        sig = f"{direction}_{confidence_bucket}_{regime}"
        
        # Add orderflow if available
        if "orderflow" in signal:
            of = signal["orderflow"]
            sig += f"_of{int(of * 10)}"
        
        return sig
    
    def _classify_pattern_type(self, signal: Dict) -> str:
        """Classify pattern type from signal"""
        # Simple classification based on available data
        regime = signal.get("regime", "unknown")
        
        if regime in ["trending_up", "trending_down"]:
            return "trend_follow"
        elif regime == "ranging":
            return "mean_reversion"
        elif "breakout" in signal.get("reason", "").lower():
            return "breakout"
        else:
            return "unknown"
    
    def _extract_entry_conditions(self, signal: Dict) -> Dict[str, Any]:
        """Extract entry conditions from signal"""
        return {
            "confidence": signal.get("confidence", 0),
            "regime": signal.get("regime", "unknown"),
            "orderflow": signal.get("orderflow", 0),
            "price": signal.get("price", 0),
        }
    
    def _calculate_pattern_confidence(self, pattern: TradePattern) -> float:
        """Calculate pattern confidence score"""
        # Factors: win rate, sample size, profit factor
        
        # Win rate component (0-0.4)
        win_rate_score = pattern.win_rate * 0.4
        
        # Sample size component (0-0.3)
        sample_score = min(pattern.total_occurrences / 50, 1.0) * 0.3
        
        # Profit factor component (0-0.3)
        pf_score = min(pattern.profit_factor / 3.0, 1.0) * 0.3 if pattern.profit_factor > 0 else 0
        
        confidence = win_rate_score + sample_score + pf_score
        return min(1.0, confidence)
    
    def get_best_patterns(self, top_n: int = 10) -> List[TradePattern]:
        """Get top performing patterns"""
        # Filter by minimum occurrences
        valid_patterns = [
            p for p in self.patterns.values()
            if p.total_occurrences >= self.min_pattern_occurrences
        ]
        
        # Sort by confidence
        valid_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return valid_patterns[:top_n]
    
    def should_take_trade(self, signal: Dict) -> Tuple[bool, float]:
        """
        Determine if trade should be taken based on learned patterns
        
        Returns: (should_take, confidence_adjustment)
        """
        pattern_sig = self._create_pattern_signature(signal)
        
        if pattern_sig in self.patterns:
            pattern = self.patterns[pattern_sig]
            
            if pattern.total_occurrences >= self.min_pattern_occurrences:
                # Use historical performance
                if pattern.win_rate >= 0.55 and pattern.confidence >= 0.6:
                    logger.info(
                        f"✅ Pattern recognized: {pattern_sig}, "
                        f"win_rate={pattern.win_rate:.2%}"
                    )
                    return True, pattern.confidence
                else:
                    logger.warning(
                        f"⚠️ Low-performing pattern: {pattern_sig}, "
                        f"win_rate={pattern.win_rate:.2%}"
                    )
                    return False, 0.0
        
        # Unknown pattern - use original signal confidence
        return True, signal.get("confidence", 0.5)
    
    # ============================================================
    # Hyperparameter Optimization
    # ============================================================
    
    def register_hyperparameter_set(
        self,
        name: str,
        parameters: Dict[str, Any],
    ) -> None:
        """Register a hyperparameter configuration"""
        hyperparam = HyperparameterSet(
            name=name,
            parameters=parameters,
            total_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            score=0.0,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        
        self.hyperparameter_sets[name] = hyperparam
        logger.info(f"📝 Hyperparameter set registered: {name}")
    
    def update_hyperparameter_performance(
        self,
        name: str,
        trade_result: Dict,
    ) -> None:
        """Update hyperparameter set performance"""
        if name not in self.hyperparameter_sets:
            logger.warning(f"Unknown hyperparameter set: {name}")
            return
        
        hp = self.hyperparameter_sets[name]
        hp.total_trades += 1
        hp.total_pnl += trade_result.get("pnl", 0)
        
        # Recalculate metrics
        if hp.total_trades > 0:
            recent_trades = [
                t for t in self.trade_history
                if t.get("hyperparameters") == name
            ][-100:]  # Last 100 trades
            
            if recent_trades:
                wins = sum(1 for t in recent_trades if t.get("success", False))
                hp.win_rate = wins / len(recent_trades)
                
                # Calculate score (composite metric)
                hp.score = self._calculate_hyperparameter_score(hp)
                hp.last_updated = datetime.now(timezone.utc).isoformat()
                
                logger.info(
                    f"📊 Hyperparameter {name}: "
                    f"trades={hp.total_trades}, win_rate={hp.win_rate:.2%}, score={hp.score:.3f}"
                )
    
    def _calculate_hyperparameter_score(self, hp: HyperparameterSet) -> float:
        """Calculate composite score for hyperparameter set"""
        # Score = (win_rate * 0.5) + (profit_factor * 0.3) + (sample_size_factor * 0.2)
        
        win_rate_score = hp.win_rate * 0.5
        
        # Profit factor (normalized)
        pf = hp.total_pnl / max(hp.total_trades, 1)
        pf_score = min(abs(pf) / 100, 1.0) * 0.3 if pf > 0 else 0
        
        # Sample size factor
        sample_score = min(hp.total_trades / 100, 1.0) * 0.2
        
        return win_rate_score + pf_score + sample_score
    
    def get_best_hyperparameters(self) -> Optional[HyperparameterSet]:
        """Get best performing hyperparameter set"""
        if not self.hyperparameter_sets:
            return None
        
        # Filter sets with minimum trades
        valid_sets = [
            hp for hp in self.hyperparameter_sets.values()
            if hp.total_trades >= 20
        ]
        
        if not valid_sets:
            return None
        
        # Return highest scoring set
        best = max(valid_sets, key=lambda hp: hp.score)
        return best
    
    async def optimize_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """
        Auto-optimize hyperparameters using grid search
        
        Returns optimized parameter set
        """
        logger.info("🔧 Starting hyperparameter optimization...")
        
        # Define search space
        search_space = {
            "confidence_threshold": [0.4, 0.5, 0.6, 0.7],
            "tp_multiplier": [1.5, 2.0, 2.5, 3.0],
            "sl_multiplier": [0.8, 1.0, 1.2, 1.5],
            "max_position_size": [0.01, 0.02, 0.03, 0.05],
        }
        
        # Test combinations (simplified - in production use more sophisticated methods)
        best_score = 0.0
        best_params = None
        
        # For demo: test a few random combinations
        import random
        
        for i in range(10):  # Test 10 random combinations
            params = {
                key: random.choice(values)
                for key, values in search_space.items()
            }
            
            # Simulate performance (in production: backtest)
            score = await self._evaluate_hyperparameters(params)
            
            logger.info(f"Tested params: {params}, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = params
        
        if best_params:
            logger.info(f"✅ Best hyperparameters found: {best_params}, score={best_score:.3f}")
            self.register_hyperparameter_set("optimized_" + datetime.now().strftime("%Y%m%d"), best_params)
            return best_params
        
        return None
    
    async def _evaluate_hyperparameters(self, params: Dict) -> float:
        """Evaluate hyperparameter set (simplified simulation)"""
        # In production: run backtest with these parameters
        # For now: estimate based on historical patterns
        
        await asyncio.sleep(0.1)  # Simulate computation
        
        # Simple heuristic scoring
        score = (
            params.get("confidence_threshold", 0.5) * 0.3 +
            (2.0 / params.get("tp_multiplier", 2.0)) * 0.3 +
            (1.0 / params.get("sl_multiplier", 1.0)) * 0.2 +
            (0.02 / params.get("max_position_size", 0.02)) * 0.2
        )
        
        return score
    
    # ============================================================
    # Learning Loop
    # ============================================================
    
    async def _learning_loop(self) -> None:
        """Background learning and optimization"""
        logger.info("🧠 Learning loop started")
        
        while self._running:
            try:
                self.learning_iterations += 1
                
                # Periodic optimization (every 24 hours)
                if self.learning_iterations % 144 == 0:  # Every ~24h if 10min intervals
                    logger.info("🔧 Triggering periodic optimization...")
                    await self.optimize_hyperparameters()
                    self.last_optimization = datetime.now(timezone.utc)
                
                # Analyze patterns every iteration
                self._analyze_pattern_performance()
                
                # Save data periodically
                if self.learning_iterations % 6 == 0:  # Every hour
                    self._save_patterns()
                    self._save_hyperparameters()
                    self._save_trade_history()
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.exception(f"Error in learning loop: {e}")
                await asyncio.sleep(60)
    
    def _analyze_pattern_performance(self) -> None:
        """Analyze and prune low-performing patterns"""
        if not self.patterns:
            return
        
        # Remove stale patterns (not seen in 30 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.pattern_lookback_days)
        
        stale_patterns = [
            pid for pid, p in self.patterns.items()
            if datetime.fromisoformat(p.last_seen) < cutoff
        ]
        
        for pid in stale_patterns:
            logger.info(f"🗑️ Removing stale pattern: {pid}")
            del self.patterns[pid]
        
        # Log summary
        if self.learning_iterations % 6 == 0:
            logger.info(
                f"📊 Learning Summary: "
                f"{len(self.patterns)} patterns, "
                f"{len(self.hyperparameter_sets)} hyperparam sets, "
                f"{len(self.trade_history)} trades in history"
            )
    
    # ============================================================
    # Persistence
    # ============================================================
    
    def _save_patterns(self) -> None:
        """Save patterns to disk"""
        try:
            file_path = self.data_dir / "patterns.json"
            data = {pid: asdict(p) for pid, p in self.patterns.items()}
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.patterns)} patterns")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def _load_patterns(self) -> None:
        """Load patterns from disk"""
        try:
            file_path = self.data_dir / "patterns.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                self.patterns = {
                    pid: TradePattern(**pdata)
                    for pid, pdata in data.items()
                }
                
                logger.info(f"Loaded {len(self.patterns)} patterns")
        except Exception as e:
            logger.warning(f"Failed to load patterns: {e}")
    
    def _save_hyperparameters(self) -> None:
        """Save hyperparameters to disk"""
        try:
            file_path = self.data_dir / "hyperparameters.json"
            data = {name: asdict(hp) for name, hp in self.hyperparameter_sets.items()}
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.hyperparameter_sets)} hyperparameter sets")
        except Exception as e:
            logger.error(f"Failed to save hyperparameters: {e}")
    
    def _load_hyperparameters(self) -> None:
        """Load hyperparameters from disk"""
        try:
            file_path = self.data_dir / "hyperparameters.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                self.hyperparameter_sets = {
                    name: HyperparameterSet(**hpdata)
                    for name, hpdata in data.items()
                }
                
                logger.info(f"Loaded {len(self.hyperparameter_sets)} hyperparameter sets")
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters: {e}")
    
    def _save_trade_history(self) -> None:
        """Save trade history to disk"""
        try:
            file_path = self.data_dir / "trade_history.json"
            
            with open(file_path, "w") as f:
                json.dump(list(self.trade_history), f, indent=2)
            
            logger.debug(f"Saved {len(self.trade_history)} trades")
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
    
    def _load_trade_history(self) -> None:
        """Load trade history from disk"""
        try:
            file_path = self.data_dir / "trade_history.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                self.trade_history = deque(data, maxlen=500)
                
                logger.info(f"Loaded {len(self.trade_history)} trades")
        except Exception as e:
            logger.warning(f"Failed to load trade history: {e}")


__all__ = ["SelfLearningEngine", "TradePattern", "HyperparameterSet"]
