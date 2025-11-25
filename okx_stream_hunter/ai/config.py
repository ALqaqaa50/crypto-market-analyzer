"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Configuration Module

All hyperparameters, model configurations, regime thresholds, and risk limits.
Centralized configuration for the entire AI stack.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CNNConfig:
    """CNN Layer Configuration for micro-pattern detection"""
    
    # Architecture
    input_sequence_length: int = 50  # Number of recent candles to analyze
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    dense_layers: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.3
    
    # Feature engineering
    features_to_extract: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'wick_ratio', 'body_ratio', 'spread', 'volatility',
        'momentum', 'rsi', 'macd'
    ])
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs_per_cycle: int = 5


@dataclass
class LSTMConfig:
    """LSTM/Transformer Time-Series Layer Configuration"""
    
    # Architecture
    sequence_length: int = 100  # Longer sequence for pattern learning
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    attention_heads: int = 8  # For transformer attention mechanism
    use_transformer: bool = True  # Use transformer instead of pure LSTM
    dropout_rate: float = 0.3
    
    # Features
    features: List[str] = field(default_factory=lambda: [
        'price', 'volume', 'buy_pressure', 'sell_pressure',
        'orderbook_imbalance', 'trade_intensity', 'liquidity_depth'
    ])
    
    # Training
    learning_rate: float = 0.0005
    batch_size: int = 64
    epochs_per_cycle: int = 10


@dataclass
class OrderflowConfig:
    """Orderflow Neural Layer Configuration"""
    
    # Analysis windows
    micro_window_seconds: int = 5  # Ultra-short term
    short_window_seconds: int = 30
    medium_window_seconds: int = 300  # 5 minutes
    
    # Detection thresholds
    spoofing_threshold: float = 0.7  # Ratio of cancelled vs executed
    aggressive_buy_threshold: float = 0.65  # Buy pressure ratio
    aggressive_sell_threshold: float = 0.35  # Sell pressure ratio
    iceberg_detection_threshold: float = 0.8  # Hidden vs visible volume
    absorption_threshold: float = 0.75  # Volume absorbed without price move
    
    # Volume clustering
    volume_cluster_min_trades: int = 10
    volume_cluster_time_window: float = 2.0  # seconds
    
    # Neural network for orderflow patterns
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    learning_rate: float = 0.001


@dataclass
class RLConfig:
    """Reinforcement Learning Configuration"""
    
    # Algorithm
    algorithm: str = "PPO"  # Proximal Policy Optimization
    
    # State space
    state_features: List[str] = field(default_factory=lambda: [
        'regime', 'price_momentum', 'volatility', 'orderflow_signal',
        'cnn_confidence', 'lstm_confidence', 'position_state',
        'unrealized_pnl', 'time_in_position', 'market_pressure'
    ])
    
    # Action space
    actions: List[str] = field(default_factory=lambda: ['long', 'short', 'close', 'hold'])
    
    # Hyperparameters
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 0.0003
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    replay_buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 4  # Update every N steps
    
    # Reward shaping
    profit_reward_multiplier: float = 100.0
    loss_penalty_multiplier: float = 50.0
    hold_penalty: float = -0.1  # Small penalty for inaction
    risk_penalty_weight: float = 0.2


@dataclass
class RegimeConfig:
    """Market Regime Detection Configuration"""
    
    # Regime types
    regimes: List[str] = field(default_factory=lambda: [
        'range', 'trend_up', 'trend_down', 'volatility_expansion',
        'liquidity_sweep', 'breakout', 'liquidity_void', 'fair_value_gap',
        'order_block', 'htf_pressure'
    ])
    
    # Detection thresholds
    trend_threshold: float = 0.02  # 2% move for trend
    range_atr_multiplier: float = 0.5  # Range if price within 0.5 ATR
    volatility_expansion_threshold: float = 2.0  # 2x normal volatility
    liquidity_sweep_delta: float = 0.005  # 0.5% sweep
    breakout_volume_multiplier: float = 1.5  # 1.5x average volume
    
    # Time windows for detection
    short_window_candles: int = 20
    medium_window_candles: int = 50
    long_window_candles: int = 200
    
    # Indicators
    use_ema: bool = True
    use_atr: bool = True
    use_bollinger_bands: bool = True
    use_volume_profile: bool = True


@dataclass
class RiskConfig:
    """Risk Intelligence Configuration"""
    
    # Volatility filters
    max_volatility_multiplier: float = 3.0  # Block trades if vol > 3x normal
    min_volatility_threshold: float = 0.0001  # Minimum vol to trade
    
    # Position limits
    max_position_size_pct: float = 2.0  # Max 2% of capital per trade
    max_leverage: float = 5.0
    max_drawdown_pct: float = 10.0  # Stop trading if DD > 10%
    max_daily_loss_pct: float = 5.0
    
    # Dynamic SL/TP
    sl_atr_multiplier: float = 2.0  # SL at 2 ATR
    tp_risk_reward_ratio: float = 2.5  # TP at 2.5x risk
    trailing_stop_activation: float = 1.5  # Activate trailing at 1.5 RR
    trailing_stop_distance: float = 0.5  # Trail at 0.5 RR
    
    # Danger detection
    liquidation_cluster_threshold: float = 0.1  # 10% of open interest
    dangerous_imbalance_threshold: float = 0.9  # 90% one-sided
    spoof_trap_score_threshold: float = 0.8
    
    # Dynamic sizing
    kelly_fraction: float = 0.25  # Conservative Kelly
    confidence_scaling: bool = True  # Scale position by AI confidence
    min_confidence_to_trade: float = 0.60  # 60% minimum confidence


@dataclass
class MetaReasonerConfig:
    """Meta-Reasoning Omega Layer Configuration"""
    
    # Model weights (relative importance of each signal)
    cnn_weight: float = 0.20
    lstm_weight: float = 0.25
    orderflow_weight: float = 0.30
    rl_weight: float = 0.15
    risk_weight: float = 0.10
    
    # Fusion method
    fusion_method: str = "weighted_ensemble"  # or "neural_fusion"
    
    # Neural fusion network (if used)
    fusion_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    fusion_learning_rate: float = 0.001
    
    # Decision thresholds
    min_confidence_threshold: float = 0.60
    high_confidence_threshold: float = 0.80
    consensus_required: bool = True  # Require multiple models to agree
    consensus_threshold: int = 3  # Out of 5 models must agree
    
    # Regime-adaptive weighting
    regime_weight_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'trend': {'lstm_weight': 0.35, 'orderflow_weight': 0.25},
        'range': {'orderflow_weight': 0.40, 'cnn_weight': 0.25},
        'volatility_expansion': {'risk_weight': 0.25, 'orderflow_weight': 0.35}
    })


@dataclass
class OptimizationConfig:
    """Auto-Optimization Configuration"""
    
    # Optimization schedule
    optimization_interval_minutes: int = 15
    
    # Hyperparameter search
    hyperparameter_search_iterations: int = 20
    use_bayesian_optimization: bool = True
    
    # Performance metrics to optimize
    target_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'win_rate', 'profit_factor',
        'max_drawdown', 'calmar_ratio'
    ])
    
    # Retraining
    retrain_cnn: bool = True
    retrain_lstm: bool = True
    retrain_rl: bool = True
    retrain_on_samples: int = 1000  # Last N samples
    
    # Threshold adaptation
    adapt_confidence_threshold: bool = True
    adapt_risk_thresholds: bool = True
    adapt_regime_thresholds: bool = True


@dataclass
class MetaReasonerConfig:
    """Meta-Reasoning Omega Layer Configuration"""
    
    fusion_method: str = "weighted_ensemble"
    min_confidence_threshold: float = 0.30
    high_confidence_threshold: float = 0.75


@dataclass
class PrometheusConfig:
    """Master Configuration for PROMETHEUS v7 AI Brain"""
    
    # Sub-configurations
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    orderflow: OrderflowConfig = field(default_factory=OrderflowConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    meta: MetaReasonerConfig = field(default_factory=MetaReasonerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"
    save_models: bool = True
    model_save_path: str = "./models/prometheus_v7"
    
    # Performance
    enable_gpu: bool = True
    num_threads: int = 4
    async_inference: bool = True
    
    # Feature flags
    enable_cnn: bool = True
    enable_lstm: bool = True
    enable_orderflow: bool = True
    enable_rl: bool = True
    enable_auto_optimization: bool = True


# Global configuration instance
CONFIG = PrometheusConfig()


def get_config() -> PrometheusConfig:
    """Get the global configuration instance"""
    return CONFIG


def update_config(**kwargs):
    """Update configuration values dynamically"""
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)


def save_config(filepath: str):
    """Save configuration to file"""
    import json
    import dataclasses
    
    def serialize(obj):
        if dataclasses.is_dataclass(obj):
            return {k: serialize(v) for k, v in dataclasses.asdict(obj).items()}
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(serialize(CONFIG), f, indent=2)


def load_config(filepath: str):
    """Load configuration from file"""
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
        # TODO: Reconstruct dataclass from dict
        pass
