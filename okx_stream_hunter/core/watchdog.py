"""
System Watchdog - PHASE 3
Health monitoring, heartbeat tracking, auto-recovery
"""

import asyncio
import logging
from typing import Dict, Callable, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class SystemWatchdog:
    """System health monitor with auto-recovery"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        self.heartbeat_interval = config.get('heartbeat_interval_seconds', 10)
        self.component_timeout = config.get('component_timeout_seconds', 30)
        
        self.components = {}
        self.heartbeats = {}
        self.health_history = deque(maxlen=100)
        
        self.recovery_callbacks = {}
        self.alert_callbacks = []
        
        self.stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'recoveries_attempted': 0,
            'recoveries_successful': 0,
            'alerts_sent': 0
        }
        
        logger.info(f"🐕 System Watchdog initialized: {self.heartbeat_interval}s interval")
    
    def register_component(
        self,
        name: str,
        health_check: Optional[Callable] = None,
        recovery_callback: Optional[Callable] = None
    ):
        """Register component for monitoring"""
        self.components[name] = {
            'name': name,
            'health_check': health_check,
            'status': 'unknown',
            'last_check': None,
            'consecutive_failures': 0
        }
        
        self.heartbeats[name] = {
            'last_heartbeat': datetime.now(),
            'beats': 0
        }
        
        if recovery_callback:
            self.recovery_callbacks[name] = recovery_callback
        
        logger.info(f"📝 Component registered: {name}")
    
    def heartbeat(self, component: str):
        """Record component heartbeat"""
        if component in self.heartbeats:
            self.heartbeats[component]['last_heartbeat'] = datetime.now()
            self.heartbeats[component]['beats'] += 1
    
    async def start(self):
        """Start watchdog monitoring"""
        self.running = True
        logger.info("🐕 Watchdog started")
        
        while self.running:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"❌ Watchdog error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    def stop(self):
        """Stop watchdog"""
        self.running = False
        logger.info("🐕 Watchdog stopped")
    
    async def _check_all_components(self):
        """Check health of all registered components"""
        now = datetime.now()
        overall_healthy = True
        
        for name, component in self.components.items():
            try:
                self.stats['total_checks'] += 1
                
                is_healthy = await self._check_component(name, component, now)
                
                if not is_healthy:
                    overall_healthy = False
                    self.stats['failed_checks'] += 1
                    
                    component['consecutive_failures'] += 1
                    
                    if component['consecutive_failures'] >= 3:
                        await self._attempt_recovery(name, component)
                else:
                    component['consecutive_failures'] = 0
                
                component['status'] = 'healthy' if is_healthy else 'unhealthy'
                component['last_check'] = now
                
            except Exception as e:
                logger.error(f"❌ Error checking {name}: {e}")
                component['status'] = 'error'
                overall_healthy = False
        
        self.health_history.append({
            'timestamp': now,
            'overall_healthy': overall_healthy,
            'components': {
                name: comp['status']
                for name, comp in self.components.items()
            }
        })
    
    async def _check_component(self, name: str, component: Dict, now: datetime) -> bool:
        """Check individual component health"""
        if name in self.heartbeats:
            last_beat = self.heartbeats[name]['last_heartbeat']
            elapsed = (now - last_beat).total_seconds()
            
            if elapsed > self.component_timeout:
                logger.warning(f"⚠️ {name} heartbeat timeout: {elapsed:.1f}s")
                return False
        
        if component['health_check']:
            try:
                result = component['health_check']()
                if asyncio.iscoroutine(result):
                    result = await result
                return bool(result)
            except Exception as e:
                logger.error(f"❌ {name} health check failed: {e}")
                return False
        
        return True
    
    async def _attempt_recovery(self, name: str, component: Dict):
        """Attempt to recover failed component"""
        self.stats['recoveries_attempted'] += 1
        
        logger.warning(f"🔧 Attempting recovery: {name}")
        
        if name in self.recovery_callbacks:
            try:
                callback = self.recovery_callbacks[name]
                result = callback()
                if asyncio.iscoroutine(result):
                    result = await result
                
                if result:
                    self.stats['recoveries_successful'] += 1
                    component['consecutive_failures'] = 0
                    logger.info(f"✅ Recovery successful: {name}")
                    return True
                else:
                    logger.error(f"❌ Recovery failed: {name}")
                    
            except Exception as e:
                logger.error(f"❌ Recovery error for {name}: {e}")
        else:
            logger.warning(f"⚠️ No recovery callback for: {name}")
        
        await self._send_alert(name, 'recovery_failed')
        return False
    
    async def _send_alert(self, component: str, alert_type: str):
        """Send alert about component failure"""
        self.stats['alerts_sent'] += 1
        
        alert = {
            'timestamp': datetime.now(),
            'component': component,
            'type': alert_type,
            'severity': 'critical'
        }
        
        logger.critical(f"🚨 ALERT: {component} - {alert_type}")
        
        for callback in self.alert_callbacks:
            try:
                result = callback(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"❌ Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def get_component_status(self, name: str) -> Dict:
        """Get status of specific component"""
        if name not in self.components:
            return {'error': 'Component not found'}
        
        component = self.components[name]
        heartbeat = self.heartbeats.get(name, {})
        
        return {
            'name': name,
            'status': component['status'],
            'last_check': component['last_check'].isoformat() if component['last_check'] else None,
            'consecutive_failures': component['consecutive_failures'],
            'last_heartbeat': heartbeat.get('last_heartbeat', datetime.now()).isoformat(),
            'total_beats': heartbeat.get('beats', 0)
        }
    
    def get_system_health(self) -> Dict:
        """Get overall system health"""
        if not self.health_history:
            return {
                'status': 'unknown',
                'healthy_components': 0,
                'total_components': len(self.components)
            }
        
        latest = self.health_history[-1]
        
        healthy_count = sum(
            1 for status in latest['components'].values()
            if status == 'healthy'
        )
        
        total = len(self.components)
        health_pct = (healthy_count / total * 100) if total > 0 else 0
        
        if health_pct >= 90:
            status = 'healthy'
        elif health_pct >= 70:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'health_percentage': health_pct,
            'healthy_components': healthy_count,
            'total_components': total,
            'components': latest['components'],
            'last_check': latest['timestamp'].isoformat()
        }
    
    def get_stats(self) -> Dict:
        """Get watchdog statistics"""
        recovery_rate = (
            self.stats['recoveries_successful'] / self.stats['recoveries_attempted']
            if self.stats['recoveries_attempted'] > 0 else 0
        )
        
        failure_rate = (
            self.stats['failed_checks'] / self.stats['total_checks']
            if self.stats['total_checks'] > 0 else 0
        )
        
        return {
            **self.stats,
            'recovery_rate': recovery_rate,
            'failure_rate': failure_rate,
            'monitored_components': len(self.components)
        }
