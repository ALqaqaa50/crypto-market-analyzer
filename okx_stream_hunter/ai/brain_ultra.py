"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Main Orchestrator - Brain Ultra

Coordinates all AI components:
- CNN Layer
- LSTM/Transformer
- Orderflow Module
- RL Agent (placeholder)
- Meta-Reasoner
- Risk Intelligence (placeholder)
- Regime Detector (placeholder)

Subscribes to StreamEngine events and produces real-time decisions.
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging
from collections import deque

from okx_stream_hunter.ai.config import get_config
from okx_stream_hunter.ai.cnn_layer import CNNLayer
from okx_stream_hunter.ai.time_series_layer import TimeSeriesLayer
from okx_stream_hunter.ai.orderflow_module import OrderflowModule
from okx_stream_hunter.ai.meta_reasoner import MetaReasoner
from okx_stream_hunter.core.experience_buffer import get_experience_buffer
from okx_stream_hunter.storage.trade_logger import get_trade_logger
from okx_stream_hunter.ai.model_registry import get_model_registry
from okx_stream_hunter.ai.self_learning_controller import get_self_learning_controller
from okx_stream_hunter.ai.rl_v2.agents.multi_agent import MultiAgentRL
from okx_stream_hunter.ai.regime_dna import RegimeDNAClassifier
from okx_stream_hunter.ai.fusion_v3 import OmegaFusionEngine
from okx_stream_hunter.ai.self_healing import HealthMonitor, AutoRetrainer

logger = logging.getLogger(__name__)


class PrometheusAIBrain:
    """
    PROMETHEUS v7 OMEGA EDITION
    Supreme AI orchestrator for crypto trading
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize all AI components
        logger.info("🧠 Initializing PROMETHEUS AI BRAIN v7 (OMEGA EDITION)")
        
        self.cnn = CNNLayer(self.config.cnn)
        self.lstm = TimeSeriesLayer(self.config.lstm)
        self.orderflow = OrderflowModule(self.config.orderflow)
        self.meta_reasoner = MetaReasoner(self.config.meta)
        
        # PHASE 5: Advanced RL, Regime Detection, Fusion, Self-Healing
        state_dim = 20
        action_dim = 1
        self.rl_multi_agent = MultiAgentRL(state_dim, action_dim, agent_types=['ddpg', 'td3', 'sac'])
        self.regime_classifier = RegimeDNAClassifier()
        self.fusion_engine = OmegaFusionEngine()
        self.health_monitor = HealthMonitor()
        self.auto_retrainer = AutoRetrainer()
        
        # PHASE 4: Experience logging and self-learning
        self.experience_buffer = get_experience_buffer()
        self.trade_logger = get_trade_logger()
        self.model_registry = get_model_registry()
        self.sl_controller = get_self_learning_controller()
        
        # Shadow mode
        self.shadow_model = None
        self._load_shadow_model()
        
        # Market data buffers
        self.candle_buffer = deque(maxlen=self.config.cnn.input_sequence_length)
        self.current_market_data = {}
        
        # Latest outputs
        self.latest_decision = None
        self.latest_cnn_output = None
        self.latest_lstm_output = None
        self.latest_orderflow_output = None
        
        logger.info("✅ PROMETHEUS AI BRAIN initialized successfully")
    
    def on_candle(self, candle: Dict):
        """Handle new candle from StreamEngine"""
        self.candle_buffer.append(candle)
        self.current_market_data['price'] = candle.get('close')
        
        # Update LSTM sequence
        self.lstm.update_sequence({
            'price': candle.get('close'),
            'volume': candle.get('volume', 0),
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'orderbook_imbalance': 0.5,
            'trade_intensity': 0.5,
            'liquidity_depth': 1.0
        })
        
        # PHASE 5: Update regime classifier
        buy_vol = candle.get('buy_volume', candle.get('volume', 0) * 0.5)
        sell_vol = candle.get('sell_volume', candle.get('volume', 0) * 0.5)
        self.regime_classifier.update(
            price=candle.get('close'),
            volume=candle.get('volume', 0),
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            bid=candle.get('close') * 0.9995,
            ask=candle.get('close') * 1.0005
        )
        
        logger.debug(f"Candle received: {candle.get('close')} @ {candle.get('timestamp')}")
    
    def on_orderbook(self, orderbook: Dict):
        """Handle orderbook update from StreamEngine"""
        self.orderflow.process_orderbook(orderbook)
        logger.debug("Orderbook processed")
    
    def on_trades(self, trades: list):
        """Handle trades from StreamEngine"""
        for trade in trades:
            self.orderflow.process_trade(trade)
        logger.debug(f"Processed {len(trades)} trades")
    
    def on_ticker(self, ticker: Dict):
        """Handle ticker update"""
        self.current_market_data.update({
            'price': ticker.get('last'),
            'volume_24h': ticker.get('vol24h', 0),
            'bid': ticker.get('bidPx'),
            'ask': ticker.get('askPx')
        })
        logger.debug(f"Ticker: {ticker.get('last')}")
    
    def get_live_decision(self) -> Optional[Dict]:
        """
        Generate real-time AI decision
        Returns None if not enough data
        """
        
        # Check if we have enough data
        if len(self.candle_buffer) < 20:
            logger.warning("Not enough candle data for prediction")
            return None
        
        try:
            # 1. CNN micro-pattern analysis
            candles_array = list(self.candle_buffer)
            self.latest_cnn_output = self.cnn.predict(candles_array)
            logger.info(f"CNN: {self.latest_cnn_output.direction} @ {self.latest_cnn_output.confidence:.2%}")
            
            # 2. LSTM/Transformer sequence prediction
            self.latest_lstm_output = self.lstm.predict()
            logger.info(f"LSTM: {self.latest_lstm_output.direction} @ {self.latest_lstm_output.confidence:.2%}")
            
            # 3. Orderflow intelligence
            self.latest_orderflow_output = self.orderflow.analyze()
            logger.info(f"Orderflow: {self.latest_orderflow_output.signal} @ {self.latest_orderflow_output.confidence:.2%}")
            
            # 4. RL recommendation (PHASE 5: Multi-Agent RL)
            rl_state = self._build_rl_state()
            rl_action = self.rl_multi_agent.select_action(rl_state, use_voting=True)
            rl_recommendation = {
                'action': 'LONG' if rl_action > 0.2 else ('SHORT' if rl_action < -0.2 else 'NEUTRAL'),
                'confidence': abs(rl_action),
                'raw_value': float(rl_action)
            }
            logger.info(f"RL: {rl_recommendation['action']} @ {rl_recommendation['confidence']:.2%}")
            
            # 5. Detect regime (PHASE 5: Regime DNA Classifier)
            regime_result = self.regime_classifier.classify()
            regime = regime_result['regime']
            regime_confidence = regime_result['confidence']
            logger.info(f"Regime: {regime} @ {regime_confidence:.2%}")
            
            # 6. Risk filters
            risk_filters = self._get_risk_filters_simple()
            
            # 7. META-REASONING: Fuse all signals (PHASE 5: Omega Fusion Engine)
            fusion_signals = {
                'cnn': {
                    'action': self._map_to_numeric(self.latest_cnn_output.direction),
                    'confidence': self.latest_cnn_output.confidence
                },
                'lstm': {
                    'action': self._map_to_numeric(self.latest_lstm_output.direction),
                    'confidence': self.latest_lstm_output.confidence
                },
                'orderflow': {
                    'action': self._map_to_numeric(self.latest_orderflow_output.signal),
                    'confidence': self.latest_orderflow_output.confidence
                },
                'rl': {
                    'action': rl_action,
                    'confidence': rl_recommendation['confidence']
                },
                'meta_reasoner': {
                    'action': 0.0,
                    'confidence': 0.5
                },
                'risk': {
                    'action': 0.0 if risk_filters.get('block_trading', False) else 0.5,
                    'confidence': 1.0
                }
            }
            
            market_state_for_fusion = {
                'volatility': risk_filters.get('atr', 0) / self.current_market_data.get('price', 1),
                'drawdown': 0.0,
                'recent_losses': 0
            }
            
            fusion_result = self.fusion_engine.fuse_signals(
                signals=fusion_signals,
                regime=regime,
                market_state=market_state_for_fusion
            )
            
            fused_action = fusion_result['action']
            fused_confidence = fusion_result['confidence']
            
            # Map back to decision using meta_reasoner for compatibility
            decision = self.meta_reasoner.fuse_signals(
                cnn_output=self.latest_cnn_output,
                lstm_output=self.latest_lstm_output,
                orderflow_output=self.latest_orderflow_output,
                rl_recommendation=rl_recommendation,
                regime=regime,
                risk_filters=risk_filters,
                market_data=self.current_market_data
            )
            
            # Override with fusion engine output
            if abs(fused_action) > 0.3:
                decision.direction = 'LONG' if fused_action > 0 else 'SHORT'
                decision.confidence = fused_confidence
            else:
                decision.direction = 'NEUTRAL'
                decision.confidence = fused_confidence
            
            self.latest_decision = decision
            
            logger.info(
                f"🎯 FINAL DECISION: {decision.direction} @ {decision.confidence:.2%} "
                f"| {decision.reason}"
            )
            
            # PHASE 4: Log decision to experience buffer and disk
            self._log_decision(decision, regime, risk_filters)
            
            # PHASE 5: Health monitoring and self-healing
            predicted_return = fused_action * 0.01
            self.health_monitor.record_prediction(
                predicted_value=predicted_return,
                actual_value=0.0,
                confidence=fused_confidence,
                action=fused_action
            )
            
            health_status = self.health_monitor.check_model_health()
            if health_status['status'] == 'degraded':
                logger.warning(f"⚠️ Model degradation detected: {len(health_status['issues'])} issues")
                should_retrain = self.auto_retrainer.register_degradation(health_status)
                if should_retrain:
                    logger.critical("🔧 Triggering auto-retraining...")
                    # Retraining will happen in background thread
            
            # PHASE 4: Run shadow prediction (no impact on real trading)
            if self.sl_controller.is_shadow_mode_enabled() and self.shadow_model is not None:
                shadow_decision = self._run_shadow_prediction(self.current_market_data)
                if shadow_decision:
                    logger.debug(f"🌑 Shadow: {shadow_decision['direction']} @ {shadow_decision['confidence']:.2%}")
            
            # Convert to dict for API
            return self._decision_to_dict(decision)
        
        except Exception as e:
            logger.error(f"Error generating decision: {e}", exc_info=True)
            return None
    
    def _detect_regime_simple(self) -> str:
        """Simplified regime detection (placeholder)"""
        if len(self.candle_buffer) < 30:
            return 'unknown'
        
        candles = list(self.candle_buffer)
        prices = [c.get('close', 0) for c in candles[-30:]]
        
        # Simple trend detection
        price_change = (prices[-1] - prices[0]) / prices[0]
        volatility = np.std(prices) / np.mean(prices)
        
        if volatility > 0.03:  # 3% volatility
            return 'volatility_expansion'
        elif price_change > 0.02:
            return 'trend_up'
        elif price_change < -0.02:
            return 'trend_down'
        else:
            return 'range'
    
    def _get_risk_filters_simple(self) -> Dict:
        """Simplified risk filters (placeholder)"""
        return {
            'block_trading': False,
            'high_volatility': False,
            'liquidation_danger': False,
            'dangerous_imbalance': False,
            'atr': self.current_market_data.get('price', 0) * 0.015  # 1.5% ATR estimate
        }
    
    def _decision_to_dict(self, decision) -> Dict:
        """Convert decision to API-friendly dict"""
        return {
            'direction': decision.direction,
            'confidence': round(decision.confidence, 4),
            'regime': decision.regime,
            'sl': decision.sl,
            'tp': decision.tp,
            'reason': decision.reason,
            'timestamp': decision.timestamp.isoformat(),
            'features': decision.features,
            'component_votes': decision.component_votes,
            'component_confidences': {
                k: round(v, 4) for k, v in decision.component_confidences.items()
            }
        }
    
    def _log_decision(self, decision, regime: str, risk_filters: Dict):
        """PHASE 4: Log decision to experience buffer and disk"""
        try:
            market_features = {
                'price': self.current_market_data.get('last', 0),
                'bid': self.current_market_data.get('bid', 0),
                'ask': self.current_market_data.get('ask', 0),
                'spread': abs(self.current_market_data.get('ask', 0) - self.current_market_data.get('bid', 0)),
                'volume_24h': self.current_market_data.get('vol24h', 0),
                'buy_volume': self.orderflow.stats.get('buy_volume', 0) if hasattr(self.orderflow, 'stats') else 0,
                'sell_volume': self.orderflow.stats.get('sell_volume', 0) if hasattr(self.orderflow, 'stats') else 0,
                'orderbook_imbalance': 0,  # Placeholder
                'spoof_risk': self.orderflow.stats.get('spoof_risk', 0) if hasattr(self.orderflow, 'stats') else 0,
                'regime': regime
            }
            
            ai_decision = {
                'direction': decision.direction,
                'confidence': decision.confidence,
                'regime': regime,
                'reason': decision.reason,
                'sl': decision.sl,
                'tp': decision.tp
            }
            
            # Log to buffer
            self.experience_buffer.add_decision(
                symbol='BTC-USDT-SWAP',
                market_features=market_features,
                ai_decision=ai_decision
            )
            
            # Log to disk
            self.trade_logger.log_decision(
                timestamp=datetime.utcnow(),
                symbol='BTC-USDT-SWAP',
                market_features=market_features,
                ai_decision=ai_decision,
                risk_context=risk_filters
            )
        
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
    
    def _load_shadow_model(self):
        """Load shadow model if shadow mode enabled"""
        try:
            if not self.sl_controller.is_shadow_mode_enabled():
                logger.debug("Shadow mode not enabled")
                return
            
            # Get best candidate model
            candidate = self.model_registry.get_best_candidate('cnn', 'test_accuracy')
            if not candidate:
                logger.warning("No candidate model found for shadow mode")
                return
            
            # Load shadow model
            from okx_stream_hunter.ai.offline_trainer import OfflineTrainer
            trainer = OfflineTrainer(model_type='cnn')
            self.shadow_model = trainer.load_model(candidate.file_path)
            
            logger.info(f"🌑 Shadow model loaded: {candidate.version_id}")
        
        except Exception as e:
            logger.error(f"Error loading shadow model: {e}")
            self.shadow_model = None
    
    def _run_shadow_prediction(self, market_state: Dict) -> Optional[Dict]:
        """Run shadow model prediction (no impact on real trading)"""
        try:
            if self.shadow_model is None:
                return None
            
            # Prepare features (same as production)
            price = market_state.get('price', 0)
            bid = market_state.get('bid', 0)
            ask = market_state.get('ask', 0)
            volume = market_state.get('volume_24h', 0)
            
            # Simple feature vector
            features = np.array([[price, bid, ask, volume]])
            
            # Run prediction
            prediction = self.shadow_model.predict(features, verbose=0)
            
            # Convert to decision
            direction = "long" if prediction[0][0] > 0.5 else "short"
            confidence = float(prediction[0][0]) if direction == "long" else float(1 - prediction[0][0])
            
            shadow_decision = {
                'direction': direction,
                'confidence': confidence,
                'is_shadow': True
            }
            
            # Log shadow decision separately
            self.trade_logger.log_decision(
                timestamp=datetime.utcnow(),
                symbol='BTC-USDT-SWAP',
                market_features={
                    'price': price,
                    'bid': bid,
                    'ask': ask,
                    'volume_24h': volume,
                    'regime': 'shadow_test'
                },
                ai_decision=shadow_decision,
                risk_context={'shadow_mode': True}
            )
            
            return shadow_decision
        
        except Exception as e:
            logger.error(f"Shadow prediction error: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Get AI brain status"""
        return {
            'active': True,
            'candles_loaded': len(self.candle_buffer),
            'lstm_sequence_length': len(self.lstm.sequence_buffer),
            'orderflow_trades': len(self.orderflow.trades_short),
            'latest_decision': self._decision_to_dict(self.latest_decision) if self.latest_decision else None,
            'cnn_ready': len(self.candle_buffer) >= self.config.cnn.input_sequence_length,
            'lstm_ready': len(self.lstm.sequence_buffer) >= 20,
            'regime': self.regime_classifier.current_regime,
            'regime_confidence': self.regime_classifier.regime_confidence,
            'health_status': self.health_monitor.health_status,
            'fusion_stats': self.fusion_engine.get_fusion_statistics()
        }
    
    def _build_rl_state(self) -> np.ndarray:
        """Build state vector for RL agent"""
        state = [
            self.current_market_data.get('price', 0) / 50000.0,
            self.latest_cnn_output.confidence if self.latest_cnn_output else 0.5,
            self.latest_lstm_output.confidence if self.latest_lstm_output else 0.5,
            self.latest_orderflow_output.confidence if self.latest_orderflow_output else 0.5,
            self.regime_classifier.regime_confidence,
            len(self.candle_buffer) / 100.0,
            0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5
        ]
        return np.array(state, dtype=np.float32)
    
    def _map_to_numeric(self, direction: str) -> float:
        """Map direction string to numeric value"""
        direction = direction.upper() if isinstance(direction, str) else str(direction).upper()
        if 'LONG' in direction or 'BUY' in direction or 'UP' in direction:
            return 1.0
        elif 'SHORT' in direction or 'SELL' in direction or 'DOWN' in direction:
            return -1.0
        else:
            return 0.0


# Global instance
_brain_instance = None

def get_brain() -> PrometheusAIBrain:
    """Get singleton brain instance"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = PrometheusAIBrain()
    return _brain_instance
