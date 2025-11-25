import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Callable
from collections import deque
from okx_stream_hunter.ai.backtester_omega.orderbook_sim import OrderbookSimulator

logger = logging.getLogger(__name__)


class OmegaBacktestEngine:
    def __init__(self, initial_capital: float = 10000.0, commission_rate: float = 0.0006,
                 max_position_size: float = 1.0):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        
        self.orderbook_sim = OrderbookSimulator()
        
        self.capital = initial_capital
        self.position = 0.0
        self.avg_entry_price = 0.0
        
        self.trades = []
        self.equity_curve = []
        self.positions_history = []
        
        self.total_pnl = 0.0
        self.total_commission = 0.0
        
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"OmegaBacktestEngine initialized: capital={initial_capital}, "
                   f"commission={commission_rate*100}%")
    
    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable) -> Dict:
        logger.info(f"Starting backtest with {len(data)} data points")
        
        self.capital = self.initial_capital
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.positions_history = []
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            price = row.get('close', row.get('price', 0))
            volume = row.get('volume', 0)
            
            self.orderbook_sim.update_orderbook(price, volume)
            
            market_state = {
                'price': price,
                'volume': volume,
                'timestamp': row.get('timestamp', i),
                'index': i
            }
            
            for col in data.columns:
                if col not in market_state:
                    market_state[col] = row[col]
            
            action = strategy_func(market_state, self.position, self.capital)
            
            self._execute_action(action, market_state)
            
            current_equity = self._calculate_equity(price)
            self.equity_curve.append({
                'timestamp': market_state['timestamp'],
                'equity': current_equity,
                'position': self.position,
                'price': price
            })
            
            self.positions_history.append(self.position)
        
        final_price = data.iloc[-1].get('close', data.iloc[-1].get('price', 0))
        if self.position != 0:
            self._close_position(final_price, "backtest_end")
        
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest completed: total_return={metrics['total_return_pct']:.2f}%, "
                   f"sharpe={metrics['sharpe_ratio']:.2f}")
        
        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'positions_history': self.positions_history
        }
    
    def _execute_action(self, action: float, market_state: Dict):
        price = market_state['price']
        
        action = np.clip(action, -1.0, 1.0)
        
        target_position = action * self.max_position_size
        
        position_change = target_position - self.position
        
        if abs(position_change) < 0.01:
            return
        
        trade_size = abs(position_change)
        side = 'buy' if position_change > 0 else 'sell'
        
        order = self.orderbook_sim.place_order(side, trade_size, 'market')
        
        if order['status'] == 'pending':
            orders = self.orderbook_sim._process_pending_orders()
            if orders:
                order = orders[0]
        
        if order['status'] == 'filled':
            fill_price = order['fill_price']
            fill_size = order['fill_size']
            slippage = order['slippage']
            
            commission = fill_size * fill_price * self.commission_rate
            self.total_commission += commission
            
            if side == 'buy':
                cost = fill_size * fill_price + commission
                if cost <= self.capital:
                    new_total_size = self.position + fill_size
                    self.avg_entry_price = ((self.position * self.avg_entry_price) + 
                                           (fill_size * fill_price)) / new_total_size
                    self.position = new_total_size
                    self.capital -= cost
                    
                    self.trades.append({
                        'timestamp': market_state['timestamp'],
                        'side': 'buy',
                        'price': fill_price,
                        'size': fill_size,
                        'commission': commission,
                        'slippage': slippage,
                        'position_after': self.position
                    })
            
            else:
                sell_size = min(fill_size, self.position)
                
                if sell_size > 0:
                    proceeds = sell_size * fill_price - commission
                    
                    pnl = (fill_price - self.avg_entry_price) * sell_size - commission
                    self.total_pnl += pnl
                    
                    if pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                    
                    self.position -= sell_size
                    self.capital += proceeds
                    
                    if self.position < 0.001:
                        self.position = 0.0
                        self.avg_entry_price = 0.0
                    
                    self.trades.append({
                        'timestamp': market_state['timestamp'],
                        'side': 'sell',
                        'price': fill_price,
                        'size': sell_size,
                        'commission': commission,
                        'slippage': slippage,
                        'pnl': pnl,
                        'position_after': self.position
                    })
    
    def _close_position(self, price: float, reason: str = "manual"):
        if self.position > 0:
            order = self.orderbook_sim.place_order('sell', self.position, 'market')
            
            fill_price = price
            commission = self.position * fill_price * self.commission_rate
            
            pnl = (fill_price - self.avg_entry_price) * self.position - commission
            self.total_pnl += pnl
            
            proceeds = self.position * fill_price - commission
            self.capital += proceeds
            
            self.trades.append({
                'timestamp': time.time(),
                'side': 'sell',
                'price': fill_price,
                'size': self.position,
                'commission': commission,
                'pnl': pnl,
                'reason': reason,
                'position_after': 0.0
            })
            
            self.position = 0.0
            self.avg_entry_price = 0.0
    
    def _calculate_equity(self, current_price: float) -> float:
        position_value = self.position * current_price
        return self.capital + position_value
    
    def _calculate_metrics(self) -> Dict:
        if len(self.equity_curve) == 0:
            return {}
        
        equities = [e['equity'] for e in self.equity_curve]
        
        final_equity = equities[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        returns = np.diff(equities) / equities[:-1]
        
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)
        
        cummax = np.maximum.accumulate(equities)
        drawdowns = (equities - cummax) / cummax
        max_drawdown = np.min(drawdowns)
        
        win_rate = self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0
        
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t['pnl'] for t in winning_trades])) / abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else float('inf')
        
        return {
            'total_return': final_equity - self.initial_capital,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_commission': self.total_commission,
            'final_equity': final_equity
        }
