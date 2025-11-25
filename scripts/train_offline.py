#!/usr/bin/env python3
"""
PHASE 4: Offline Training Script
Train models on historical data without blocking live trading
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from okx_stream_hunter.storage.trade_logger import get_trade_logger
from okx_stream_hunter.ai.dataset_builder import DatasetBuilder
from okx_stream_hunter.ai.offline_trainer import OfflineTrainer
from okx_stream_hunter.ai.model_registry import get_model_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train AI models offline')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['cnn', 'lstm', 'rl_policy', 'rl_value'],
                        help='Type of model to train')
    parser.add_argument('--days-back', type=int, default=7,
                        help='Number of days to load data from')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Window size for dataset builder')
    parser.add_argument('--target-type', type=str, default='direction',
                        choices=['direction', 'return', 'outcome'],
                        help='Type of target to predict')
    parser.add_argument('--version-tag', type=str, default=None,
                        help='Version tag for model (default: auto-generated)')
    parser.add_argument('--register', action='store_true',
                        help='Register model in registry after training')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PHASE 4: OFFLINE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Days Back: {args.days_back}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Window Size: {args.window_size}")
    logger.info(f"Target Type: {args.target_type}")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n📊 STEP 1: Loading historical data...")
    trade_logger = get_trade_logger()
    data = trade_logger.load_data(days_back=args.days_back)
    
    if data is None or len(data) == 0:
        logger.error("❌ No data loaded. Run system to collect data first.")
        return
    
    logger.info(f"✅ Loaded {len(data)} records from last {args.days_back} days")
    
    # Step 2: Build dataset
    logger.info("\n🔨 STEP 2: Building dataset...")
    builder = DatasetBuilder(
        window_size=args.window_size,
        prediction_horizon=10,
        target_type=args.target_type
    )
    
    # Define features to use
    features = ['price', 'bid', 'ask', 'spread', 'volume_24h']
    
    X, y = builder.build_from_logs(data, features)
    
    if X is None or len(X) == 0:
        logger.error("❌ Failed to build dataset")
        return
    
    logger.info(f"✅ Dataset built: X.shape={X.shape}, y.shape={y.shape}")
    
    # Step 3: Normalize
    logger.info("\n📐 STEP 3: Normalizing features...")
    X_norm = builder.normalize_features(X)
    
    # Step 4: Train/test split
    logger.info("\n✂️ STEP 4: Splitting train/test...")
    X_train, X_val, y_train, y_val = builder.train_test_split(X_norm, y, test_size=0.2)
    
    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val: X={X_val.shape}, y={y_val.shape}")
    
    # Step 5: Build model
    logger.info(f"\n🏗️ STEP 5: Building {args.model_type} model...")
    trainer = OfflineTrainer(model_type=args.model_type)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, features)
    output_size = 2 if args.target_type == 'direction' else 1
    
    trainer.build_model(input_shape, output_size)
    
    # Step 6: Train
    logger.info(f"\n🎓 STEP 6: Training for {args.epochs} epochs...")
    metrics = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not metrics:
        logger.error("❌ Training failed")
        return
    
    logger.info("✅ Training completed!")
    logger.info(f"Final metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Step 7: Save model
    logger.info("\n💾 STEP 7: Saving model...")
    
    if args.version_tag is None:
        version_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        version_tag = args.version_tag
    
    file_path = trainer.save_model(version_tag, metrics)
    
    if not file_path:
        logger.error("❌ Failed to save model")
        return
    
    logger.info(f"✅ Model saved to: {file_path}")
    
    # Step 8: Register in registry
    if args.register:
        logger.info("\n📚 STEP 8: Registering model...")
        
        registry = get_model_registry()
        
        training_config = {
            'model_type': args.model_type,
            'days_back': args.days_back,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'window_size': args.window_size,
            'target_type': args.target_type,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        success = registry.register_model(
            version_id=version_tag,
            model_type=args.model_type,
            file_path=str(file_path),
            training_config=training_config,
            metrics=metrics,
            status='candidate'
        )
        
        if success:
            logger.info(f"✅ Model registered as candidate: {version_tag}")
        else:
            logger.error("❌ Failed to register model")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ OFFLINE TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Next steps:")
    logger.info(f"1. Evaluate model: python scripts/evaluate_model.py --model-type {args.model_type} --version {version_tag}")
    logger.info(f"2. If good, promote: registry.promote_to_production('{args.model_type}', '{version_tag}')")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
