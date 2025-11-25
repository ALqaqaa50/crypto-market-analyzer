#!/usr/bin/env python3
"""
PROMETHEUS v7 - Demo Trading Mode
Complete paper trading simulation without real OKX connection
"""

import asyncio
import logging
import sys
import random
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from okx_stream_hunter.notifications import get_telegram_client
from okx_stream_hunter.core.trading_mode import get_trading_mode_manager


class DemoMarketSimulator:
    """Simulate market data and trading"""
    
    def __init__(self):
        self.btc_price = 97500.0  # Starting BTC price
        self.balance = 10000.0
        self.position = None  # {'side': 'long', 'size': 0.1, 'entry': 97500, 'pnl': 0}
        self.trades = []
        self.telegram = get_telegram_client()
        self.mode_mgr = get_trading_mode_manager()
        
    def update_price(self):
        """Simulate price movement"""
        change = random.uniform(-0.002, 0.002)  # ±0.2% movement
        self.btc_price *= (1 + change)
        return self.btc_price
        
    def calculate_confidence(self):
        """Simulate AI confidence"""
        return random.uniform(0.55, 0.85)
        
    async def execute_trade(self, side, size, confidence):
        """Execute simulated trade"""
        price = self.btc_price
        
        logging.info(f"💰 [{self.mode_mgr.get_log_prefix()}] Executing {side.upper()} trade")
        logging.info(f"   Size: {size} BTC")
        logging.info(f"   Price: ${price:,.2f}")
        logging.info(f"   Confidence: {confidence:.1%}")
        
        # Send Telegram notification
        try:
            await self.telegram.send_trade_alert(
                direction=side,
                size=size,
                price=price,
                mode='SANDBOX',
                symbol='BTC/USDT',
                confidence=confidence,
                leverage=1
            )
        except Exception as e:
            logging.warning(f"Telegram send failed: {e}")
        
        # Open position
        if not self.position:
            self.position = {
                'side': side,
                'size': size,
                'entry': price,
                'pnl': 0,
                'confidence': confidence,
                'time': datetime.now()
            }
            
            try:
                await self.telegram.send_position_open(
                    symbol='BTC/USDT',
                    side=side,
                    size=size,
                    price=price,
                    confidence=confidence,
                    mode='SANDBOX'
                )
            except:
                pass
                
            logging.info(f"✅ Position opened: {side.upper()} {size} BTC @ ${price:,.2f}")
            
    async def close_position(self):
        """Close current position"""
        if not self.position:
            return
            
        current_price = self.btc_price
        entry_price = self.position['entry']
        size = self.position['size']
        side = self.position['side']
        
        # Calculate P&L
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        pnl_usd = pnl_pct * entry_price * size
        self.balance += pnl_usd
        
        logging.info(f"🔒 Closing position: {side.upper()}")
        logging.info(f"   Entry: ${entry_price:,.2f}")
        logging.info(f"   Exit: ${current_price:,.2f}")
        logging.info(f"   P&L: ${pnl_usd:,.2f} ({pnl_pct:+.2%})")
        logging.info(f"   Balance: ${self.balance:,.2f}")
        
        # Send Telegram notification
        try:
            await self.telegram.send_position_close(
                symbol='BTC/USDT',
                side=side,
                pnl=pnl_usd,
                confidence=self.position['confidence'],
                mode='SANDBOX',
                entry_price=entry_price,
                exit_price=current_price
            )
        except:
            pass
        
        self.trades.append({
            'side': side,
            'size': size,
            'entry': entry_price,
            'exit': current_price,
            'pnl': pnl_usd,
            'pnl_pct': pnl_pct,
            'time': datetime.now()
        })
        
        self.position = None
        
    def print_status(self):
        """Print current status"""
        logging.info("=" * 80)
        logging.info(f"📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        logging.info(f"💰 Balance: ${self.balance:,.2f}")
        logging.info(f"📈 BTC Price: ${self.btc_price:,.2f}")
        
        if self.position:
            pnl = (self.btc_price - self.position['entry']) / self.position['entry']
            if self.position['side'] == 'short':
                pnl = -pnl
            logging.info(f"📍 Position: {self.position['side'].upper()} {self.position['size']} BTC")
            logging.info(f"   Entry: ${self.position['entry']:,.2f}")
            logging.info(f"   Unrealized P&L: {pnl:+.2%}")
        else:
            logging.info(f"📍 Position: None")
            
        if self.trades:
            win_trades = [t for t in self.trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(self.trades) * 100
            total_pnl = sum(t['pnl'] for t in self.trades)
            logging.info(f"📊 Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:,.2f}")
        
        logging.info("=" * 80)


async def run_demo_trading():
    """Run demo trading simulation"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("demo")
    
    logger.info("=" * 80)
    logger.info("🔥 PROMETHEUS v7 - DEMO TRADING MODE")
    logger.info("🤖 Paper Trading Simulation with Telegram Notifications")
    logger.info("=" * 80)
    
    simulator = DemoMarketSimulator()
    
    # Send startup notification
    try:
        await simulator.telegram.send_system_restart("Starting Demo Trading Mode")
        await simulator.telegram.send_status(
            f"🚀 PROMETHEUS v7 Demo Mode\n"
            f"💰 Starting Balance: ${simulator.balance:,.2f}\n"
            f"📈 BTC Price: ${simulator.btc_price:,.2f}\n"
            f"🎯 Trading Symbol: BTC/USDT"
        )
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")
    
    logger.info("")
    logger.info("✅ System ready - Starting trading simulation")
    if simulator.telegram.enabled:
        logger.info("📱 Telegram notifications: ENABLED ✅")
    else:
        logger.info("📱 Telegram notifications: DISABLED (token issue)")
    logger.info("⏸️  Press Ctrl+C to stop")
    logger.info("")
    
    cycle = 0
    try:
        while True:
            cycle += 1
            
            # Update price
            new_price = simulator.update_price()
            
            # Log price updates every few cycles
            if cycle % 3 == 0:
                logger.info(f"📈 BTC Price: ${new_price:,.2f}")
            
            # Every 10 cycles (~30 seconds), make a trading decision
            if cycle % 10 == 0:
                simulator.print_status()
                
                confidence = simulator.calculate_confidence()
                
                if not simulator.position and confidence > 0.65:
                    # Open new position
                    side = random.choice(['long', 'short'])
                    size = 0.05  # 0.05 BTC
                    await simulator.execute_trade(side, size, confidence)
                    
                elif simulator.position:
                    # Close position based on time or profit target
                    hold_time = (datetime.now() - simulator.position['time']).seconds
                    current_pnl = (new_price - simulator.position['entry']) / simulator.position['entry']
                    if simulator.position['side'] == 'short':
                        current_pnl = -current_pnl
                    
                    # Close if held > 2 minutes or hit 2% profit/loss
                    if hold_time > 120 or abs(current_pnl) > 0.02:
                        await simulator.close_position()
            
            # Wait 3 seconds between updates
            await asyncio.sleep(3)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Shutdown requested")
        
        # Close any open position
        if simulator.position:
            await simulator.close_position()
        
        # Final stats
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"💰 Starting Balance: $10,000.00")
        logger.info(f"💰 Final Balance: ${simulator.balance:,.2f}")
        logger.info(f"📈 Total Return: ${simulator.balance - 10000:+,.2f} ({(simulator.balance/10000-1)*100:+.2f}%)")
        
        if simulator.trades:
            win_trades = [t for t in simulator.trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(simulator.trades) * 100
            total_pnl = sum(t['pnl'] for t in simulator.trades)
            avg_pnl = total_pnl / len(simulator.trades)
            
            logger.info(f"📊 Total Trades: {len(simulator.trades)}")
            logger.info(f"✅ Winning Trades: {len(win_trades)}")
            logger.info(f"❌ Losing Trades: {len(simulator.trades) - len(win_trades)}")
            logger.info(f"📈 Win Rate: {win_rate:.1f}%")
            logger.info(f"💵 Total P&L: ${total_pnl:+,.2f}")
            logger.info(f"📊 Average P&L per Trade: ${avg_pnl:+,.2f}")
        
        logger.info("=" * 80)
        
        # Send final stats to Telegram
        try:
            if simulator.trades:
                win_trades = [t for t in simulator.trades if t['pnl'] > 0]
                win_rate = len(win_trades) / len(simulator.trades) * 100
                total_pnl = sum(t['pnl'] for t in simulator.trades)
                
                await simulator.telegram.send_status(
                    f"🏁 Demo Trading Session Ended\n\n"
                    f"💰 Final Balance: ${simulator.balance:,.2f}\n"
                    f"📈 Return: ${simulator.balance - 10000:+,.2f} ({(simulator.balance/10000-1)*100:+.2f}%)\n"
                    f"📊 Trades: {len(simulator.trades)}\n"
                    f"✅ Win Rate: {win_rate:.1f}%\n"
                    f"💵 Total P&L: ${total_pnl:+,.2f}"
                )
        except:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(run_demo_trading())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
