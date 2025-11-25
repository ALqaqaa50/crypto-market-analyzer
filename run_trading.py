#!/usr/bin/env python3
"""
PROMETHEUS v7 - PHASE 5 Autonomous Self-Evolving Trading System Launcher
Complete system with autonomous learning, evolution, and self-healing
"""

import asyncio
import logging
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from okx_stream_hunter.core.autonomous_runtime import start_autonomous_runtime, stop_autonomous_runtime
from okx_stream_hunter.ai.autonomous_loop import get_autonomous_loop
from okx_stream_hunter.core.trading_mode import get_trading_mode_manager
from okx_stream_hunter.notifications import get_telegram_client


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Reduce noise from libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config():
    """Load trading configuration"""
    config_path = Path("okx_stream_hunter/config/trading_config.yaml")
    
    if not config_path.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config():
    """Get default configuration"""
    return {
        # Trading
        'symbol': 'BTC-USDT-SWAP',
        'paper_trading': True,
        'auto_trading': False,
        'initial_balance': 10000.0,
        
        # Risk Management
        'max_risk_per_trade': 0.02,
        'max_daily_drawdown': 0.10,
        'min_confidence_to_trade': 0.60,
        
        # Decision Making
        'decision_interval_seconds': 5,
        'candle_timeframe': 60,
        
        # PHASE 3: Safety Limits
        'circuit_breaker': {
            'daily_loss_limit_pct': 10,
            'max_daily_trades': 20,
            'max_consecutive_losses': 5,
            'single_trade_loss_limit_pct': 5,
            'auto_reset_minutes': 60,
        },
        
        'ai_safety': {
            'confidence_floor': 0.30,
            'confidence_ceiling': 0.95,
            'max_confidence_std': 0.30,
            'max_consecutive_losses': 5,
            'max_drawdown_pct': 15,
        },
        
        'rate_limiter': {
            'base_limit': 10,
            'min_limit': 1,
            'max_limit': 50,
        },
        
        'watchdog': {
            'interval': 10,
            'failure_threshold': 3,
            'recovery_enabled': True,
        },
        
        # System
        'stats_interval': 60,
    }


async def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger("main")
    
    config = load_config()
    
    trading_mode_mgr = get_trading_mode_manager(config.get('trading_mode'))
    telegram = get_telegram_client()
    
    mode_display = "SANDBOX MODE" if trading_mode_mgr.is_sandbox() else "REAL LIVE-MARKET MODE"
    
    logger.info("=" * 80)
    logger.info(f"🔥 PROMETHEUS AI BRAIN v7 (OMEGA EDITION) - PHASE 5.1")
    logger.info(f"🤖 AUTONOMOUS SELF-EVOLVING TRADING SYSTEM")
    logger.info(f"📡 Trading Mode: {mode_display}")
    logger.info("=" * 80)
    
    logger.info(f"📊 Symbol: {config.get('symbol')}")
    logger.info(f"💰 Initial Balance: ${config.get('initial_balance'):,.2f}")
    logger.info(f"📄 Paper Trading: {config.get('paper_trading')}")
    logger.info(f"🤖 Auto Trading: {config.get('auto_trading')}")
    logger.info(f"🎯 Min Confidence: {config.get('min_confidence_to_trade'):.0%}")
    logger.info("")
    
    if trading_mode_mgr.is_real():
        logger.critical("=" * 80)
        logger.critical("⚠️  REAL TRADING MODE ACTIVE")
        logger.critical("⚠️  ALL ORDERS WILL EXECUTE ON LIVE MARKETS")
        logger.critical("⚠️  REAL MONEY AT RISK")
        logger.critical("=" * 80)
        
        safety_check = trading_mode_mgr.get_safety_check()
        if not safety_check[0]:
            logger.critical(f"❌ Safety check failed: {safety_check[1]}")
            logger.critical("Aborting execution")
            return
    else:
        logger.info("=" * 80)
        logger.info("✅ SANDBOX MODE ACTIVE")
        logger.info("✅ All trades are simulated/demo")
        logger.info("✅ No real money at risk")
        logger.info("=" * 80)
    logger.info("")
    logger.info("🛡️ PHASE 3-5 Safety & Intelligence Features:")
    logger.info(f"  ⚡ Circuit Breaker: {config.get('circuit_breaker', {}).get('daily_loss_limit_pct')}% daily loss limit")
    logger.info(f"  🛡️ AI Safety Layer: Active")
    logger.info(f"  🔄 Adaptive Rate Limiter: {config.get('rate_limiter', {}).get('base_limit')} req/s base")
    logger.info(f"  💓 System Watchdog: Auto-recovery enabled")
    logger.info(f"  👮 Trade Supervisor: Real-time monitoring")
    logger.info(f"  🧠 RL Multi-Agent: DDPG + TD3 + SAC ensemble")
    logger.info(f"  🧬 Hyperparameter Evolution: Genetic + Bayesian + PBT")
    logger.info(f"  🎯 Regime DNA Classifier: 12+ market regime types")
    logger.info(f"  🔀 Omega Fusion Engine: Adaptive signal weighting")
    logger.info(f"  🩺 Self-Healing AI: Auto-detect & retrain")
    logger.info(f"  🔬 Omega Backtester: Microstructure simulation")
    logger.info(f"  🤖 Autonomous Loop: Continuous learning & evolution")
    logger.info("=" * 80)
    
    try:
        autonomous_loop = get_autonomous_loop()
        autonomous_loop.start()
        logger.info("🤖 Autonomous Intelligence Loop STARTED")
        
        try:
            await telegram.send_system_restart(f"Starting in {mode_display}")
        except:
            pass
        
        await start_autonomous_runtime(config)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        try:
            await telegram.send_error_alert(str(e), include_trace=True)
        except:
            pass
    finally:
        try:
            autonomous_loop = get_autonomous_loop()
            autonomous_loop.stop()
        except:
            pass
        await stop_autonomous_runtime()
        logger.info("✅ System shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
