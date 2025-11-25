"""
PHASE 4: Offline Evaluator
Backtest candidate models offline without affecting live trading
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OfflineEvaluator:
    """
    Evaluate candidate models using historical logged data
    Simulates trading decisions without real execution
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
        self.balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"📊 OfflineEvaluator initialized (balance=${initial_balance:,.2f})")
    
    def evaluate_model(
        self,
        model,
        test_data: pd.DataFrame,
        model_type: str = "cnn"
    ) -> Dict:
        """Evaluate model on historical data"""
        
        if test_data.empty:
            logger.warning("Empty test data")
            return {}
        
        logger.info(f"🔬 Evaluating {model_type} model on {len(test_data)} records")
        
        self.reset()
        
        # Simulate trading based on model predictions
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            
            # Get model prediction
            prediction = self._get_model_prediction(model, row, model_type)
            
            if prediction is None:
                continue
            
            # Simulate trade
            if prediction['direction'] != 'NEUTRAL':
                self._simulate_trade(row, prediction)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"✅ Evaluation complete: {metrics}")
        
        return metrics
    
    def _get_model_prediction(self, model, row: pd.Series, model_type: str) -> Optional[Dict]:
        """Get prediction from model"""
        try:
            # Extract features (simplified)
            features = [
                row.get('price', 0),
                row.get('bid', 0),
                row.get('ask', 0),
                row.get('confidence', 0)
            ]
            
            # Placeholder: In real implementation, use proper feature extraction
            # and model.predict()
            
            # For now, use logged AI decision if available
            if 'action' in row and pd.notna(row['action']):
                direction = row['action']
                confidence = row.get('confidence', 0.5)
                
                return {
                    'direction': direction,
                    'confidence': confidence
                }
            
            return None
        
        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return None
    
    def _simulate_trade(self, row: pd.Series, prediction: Dict):
        """Simulate trade execution"""
        try:
            price = row.get('price', 0)
            direction = prediction['direction']
            confidence = prediction['confidence']
            
            if price <= 0 or confidence < 0.5:
                return
            
            # Calculate position size
            risk_amount = self.balance * self.risk_per_trade
            position_size = risk_amount / price
            
            # Simulate trade outcome
            # In real data, we'd look ahead to see actual outcome
            # For now, use logged PnL if available
            
            pnl = 0
            if 'pnl' in row and pd.notna(row['pnl']):
                pnl = row['pnl']
            else:
                # Estimate based on next price movement (simplified)
                pnl = 0
            
            self.balance += pnl
            self.equity = self.balance
            
            trade_record = {
                'timestamp': row.get('timestamp', datetime.utcnow().isoformat()),
                'direction': direction,
                'entry_price': price,
                'size': position_size,
                'confidence': confidence,
                'pnl': pnl,
                'equity': self.equity
            }
            
            self.trades.append(trade_record)
            self.equity_curve.append(self.equity)
        
        except Exception as e:
            logger.debug(f"Simulate trade error: {e}")
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        
        if not self.trades:
            logger.warning("No trades to evaluate")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_profit = total_pnl / len(self.trades) if self.trades else 0
        
        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] / self.initial_balance for t in self.trades]
        if len(returns) > 1:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Profit factor
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_profit': round(avg_profit, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'profit_factor': round(profit_factor, 2),
            'final_equity': round(self.equity, 2),
            'return_pct': round((self.equity - self.initial_balance) / self.initial_balance * 100, 2)
        }
        
        return metrics
    
    def reset(self):
        """Reset evaluator state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        self.equity_curve = []
    
    def save_evaluation_report(
        self,
        metrics: Dict,
        model_info: Dict,
        output_dir: str = "reports/phase4"
    ) -> str:
        """Save evaluation report"""
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            report_name = f"evaluation_{model_info.get('model_type', 'model')}_{timestamp}"
            
            report = {
                'model_info': model_info,
                'metrics': metrics,
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'initial_balance': self.initial_balance,
                'risk_per_trade': self.risk_per_trade,
                'trades_sample': self.trades[:10]  # First 10 trades
            }
            
            # Save JSON
            json_path = output_path / f"{report_name}.json"
            json_path.write_text(json.dumps(report, indent=2))
            
            # Save human-readable markdown
            md_path = output_path / f"{report_name}.md"
            md_content = self._generate_markdown_report(report)
            md_path.write_text(md_content)
            
            logger.info(f"📄 Evaluation report saved:")
            logger.info(f"   JSON: {json_path}")
            logger.info(f"   Markdown: {md_path}")
            
            return str(json_path)
        
        except Exception as e:
            logger.error(f"❌ Save report error: {e}")
            return ""
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown evaluation report"""
        
        model_info = report['model_info']
        metrics = report['metrics']
        
        md = f"""# Model Evaluation Report

## Model Information
- **Model Type**: {model_info.get('model_type', 'N/A')}
- **Version**: {model_info.get('version_tag', 'N/A')}
- **Framework**: {model_info.get('framework', 'N/A')}
- **Evaluation Date**: {report['evaluation_timestamp']}

## Configuration
- **Initial Balance**: ${report['initial_balance']:,.2f}
- **Risk Per Trade**: {report['risk_per_trade']:.1%}

## Performance Metrics

### Trade Statistics
- **Total Trades**: {metrics['total_trades']}
- **Winning Trades**: {metrics['winning_trades']}
- **Losing Trades**: {metrics['losing_trades']}
- **Win Rate**: {metrics['win_rate']}%

### Profitability
- **Total PnL**: ${metrics['total_pnl']:,.2f}
- **Average Profit**: ${metrics['avg_profit']:,.2f}
- **Return**: {metrics['return_pct']}%
- **Final Equity**: ${metrics['final_equity']:,.2f}

### Risk Metrics
- **Max Drawdown**: {metrics['max_drawdown']}%
- **Sharpe Ratio**: {metrics['sharpe_ratio']}
- **Profit Factor**: {metrics['profit_factor']}

## Recommendation
"""
        
        # Add recommendation based on metrics
        if metrics['win_rate'] >= 55 and metrics['sharpe_ratio'] >= 1.0 and metrics['max_drawdown'] <= 20:
            md += "✅ **CANDIDATE FOR PROMOTION** - Performance exceeds thresholds\n"
        elif metrics['win_rate'] >= 50 and metrics['max_drawdown'] <= 30:
            md += "⚠️ **MONITOR** - Acceptable performance, requires more data\n"
        else:
            md += "❌ **NOT RECOMMENDED** - Performance below thresholds\n"
        
        return md
