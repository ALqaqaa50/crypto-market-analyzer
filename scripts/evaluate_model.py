#!/usr/bin/env python3
"""
PHASE 4: Model Evaluation Script
Evaluate candidate models on historical data
"""

import argparse
import logging
from datetime import datetime

from okx_stream_hunter.storage.trade_logger import get_trade_logger
from okx_stream_hunter.ai.model_registry import get_model_registry
from okx_stream_hunter.ai.offline_trainer import OfflineTrainer
from okx_stream_hunter.backtesting.offline_evaluator import OfflineEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate candidate model')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['cnn', 'lstm', 'rl_policy', 'rl_value'],
                        help='Type of model to evaluate')
    parser.add_argument('--version', type=str, required=True,
                        help='Model version to evaluate')
    parser.add_argument('--days-back', type=int, default=7,
                        help='Number of days to load test data from')
    parser.add_argument('--save-report', action='store_true',
                        help='Save evaluation report to disk')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PHASE 4: MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Days Back: {args.days_back}")
    logger.info("=" * 60)
    
    # Step 1: Load model from registry
    logger.info("\n📚 STEP 1: Loading model from registry...")
    registry = get_model_registry()
    
    model_entry = None
    for model in registry.get_all_models(args.model_type):
        if model.version_id == args.version:
            model_entry = model
            break
    
    if model_entry is None:
        logger.error(f"❌ Model not found: {args.model_type}/{args.version}")
        return
    
    logger.info(f"✅ Found model: {model_entry.file_path}")
    logger.info(f"Status: {model_entry.status}")
    logger.info(f"Registered: {model_entry.registered_at}")
    
    # Step 2: Load model
    logger.info("\n🔄 STEP 2: Loading model...")
    trainer = OfflineTrainer(model_type=args.model_type)
    model = trainer.load_model(model_entry.file_path)
    
    if model is None:
        logger.error("❌ Failed to load model")
        return
    
    logger.info("✅ Model loaded")
    
    # Step 3: Load test data
    logger.info("\n📊 STEP 3: Loading test data...")
    trade_logger = get_trade_logger()
    test_data = trade_logger.load_data(days_back=args.days_back)
    
    if test_data is None or len(test_data) == 0:
        logger.error("❌ No test data available")
        return
    
    logger.info(f"✅ Loaded {len(test_data)} test records")
    
    # Step 4: Evaluate
    logger.info("\n🔬 STEP 4: Running backtest evaluation...")
    evaluator = OfflineEvaluator()
    
    metrics = evaluator.evaluate_model(
        model=model,
        test_data=test_data,
        model_type=args.model_type,
        initial_balance=10000.0
    )
    
    if not metrics:
        logger.error("❌ Evaluation failed")
        return
    
    # Step 5: Display results
    logger.info("\n" + "=" * 60)
    logger.info("📈 EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
    logger.info(f"Total PnL: ${metrics.get('total_pnl', 0):.2f}")
    logger.info(f"Return: {metrics.get('return_pct', 0):.2f}%")
    logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Avg Profit: ${metrics.get('avg_profit', 0):.2f}")
    logger.info(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
    
    # Step 6: Save report
    if args.save_report:
        logger.info("\n💾 STEP 6: Saving evaluation report...")
        
        report_path = evaluator.save_evaluation_report(
            metrics=metrics,
            model_type=args.model_type,
            version_id=args.version
        )
        
        if report_path:
            logger.info(f"✅ Report saved to: {report_path}")
    
    # Step 7: Recommendation
    logger.info("\n" + "=" * 60)
    logger.info("💡 RECOMMENDATION")
    logger.info("=" * 60)
    
    sharpe = metrics.get('sharpe_ratio', 0)
    winrate = metrics.get('win_rate', 0)
    drawdown = abs(metrics.get('max_drawdown', 0))
    
    if sharpe >= 1.0 and winrate >= 55 and drawdown <= 20:
        logger.info("✅ PROMOTE - Model meets all criteria")
        logger.info(f"To promote: registry.promote_to_production('{args.model_type}', '{args.version}')")
    elif sharpe >= 0.5 and winrate >= 50 and drawdown <= 30:
        logger.info("⚠️ MONITOR - Marginal performance, needs more data")
    else:
        logger.info("❌ REJECT - Poor performance, retrain with more/better data")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
