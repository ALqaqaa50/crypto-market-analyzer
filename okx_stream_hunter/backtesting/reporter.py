"""
Backtesting Reporter
Generate performance reports
"""
from typing import Dict, List
from datetime import datetime
import json

from ..utils.logger import get_logger


logger = get_logger(__name__)


class BacktestReporter:
    """
    Generate reports from backtest results.
    """
    
    def __init__(self):
        self.metrics: Dict = {}
    
    def add_metric(self, name: str, value):
        """Add a metric"""
        self.metrics[name] = value
    
    def generate_report(self) -> Dict:
        """Generate full report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'metrics': self.metrics,
        }
        return report
    
    def save_report(self, filepath: str):
        """Save report to file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filepath}")
    
    def print_report(self):
        """Print report to console"""
        report = self.generate_report()
        logger.info("="*60)
        logger.info("📈 BACKTEST REPORT")
        logger.info("="*60)
        for key, value in report['metrics'].items():
            logger.info(f"{key}: {value}")
        logger.info("="*60)