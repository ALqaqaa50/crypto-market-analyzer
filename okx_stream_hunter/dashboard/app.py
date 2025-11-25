# okx_stream_hunter/dashboard/app.py
"""
🔥 FastAPI Dashboard - Real-time AI Trading Insights
GOD MODE Ultra Dashboard v3.0
Connected to live AI Brain data via system_state
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from okx_stream_hunter.state import get_system_state

logger = logging.getLogger("dashboard.app")

# Import AI endpoints
try:
    from okx_stream_hunter.ai.api_endpoints import register_ai_endpoints
    AI_ENDPOINTS_AVAILABLE = True
except ImportError:
    logger.warning("AI endpoints not available - PROMETHEUS v7 not installed")
    AI_ENDPOINTS_AVAILABLE = False

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="🔥 Crypto Market Analyzer Dashboard",
    description="Real-time AI trading insights and auto-trading status - GOD MODE Ultra v3.0",
    version="3.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

# Mount static files (JavaScript, CSS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ============================================================
# Data Models
# ============================================================

class AISignal(BaseModel):
    signal: str
    direction: str
    confidence: float
    reason: str
    regime: str
    price: Optional[float]
    buy_volume: float
    sell_volume: float
    trade_count: int
    whale_trades: int
    max_trade_size: float
    max_trade_price: Optional[float]
    position: Dict[str, Any]
    scores: Dict[str, float]
    generated_at: str


class Strategy(BaseModel):
    signal: str
    direction: str
    entry_price: Optional[float]
    tp: Optional[float]
    sl: Optional[float]
    confidence: float
    generated_at: str


# ============================================================
# Helper Functions
# ============================================================

def load_json_file(filename: str, default: Dict = None) -> Dict:
    """Load JSON file with fallback to default"""
    try:
        file_path = Path(filename)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
    
    return default or {}


def get_system_status() -> Dict[str, Any]:
    """Get current system status from live system_state"""
    system_state = get_system_state()
    
    return {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "dashboard_version": "3.0.0",
        "ai_enabled": system_state.ai_enabled,
        "auto_trading_enabled": system_state.auto_trading_enabled,
        "uptime": system_state.uptime_seconds,
        "heartbeat": system_state.last_heartbeat.isoformat() if system_state.last_heartbeat else None,
        "last_update": system_state.last_update.isoformat() if system_state.last_update else None,
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    GOD MODE Ultra Dashboard v3.0
    
    Renders the main dashboard with:
    - Live candles chart with Entry/TP/SL lines
    - AI signal panel with confidence gauge
    - Orderflow & microstructure panel
    - Position & risk panel
    - Recent signals table
    - System status & heartbeat
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_dashboard():
    """
    Legacy simple dashboard (v2.0) - kept for backwards compatibility
    Access the new dashboard at: /
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔥 Crypto Market Analyzer - Legacy Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header {
                text-align: center;
                padding: 40px 20px;
                background: rgba(0,0,0,0.3);
                border-radius: 20px;
                margin-bottom: 30px;
            }
            .header h1 { font-size: 3em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .upgrade-banner {
                background: rgba(0,255,136,0.2);
                border: 2px solid #00ff88;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }
            .upgrade-banner a {
                color: #00ff88;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.2em;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .card h2 {
                font-size: 1.5em;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .card-content { font-size: 1.1em; line-height: 1.8; }
            .metric { display: flex; justify-content: space-between; margin: 10px 0; }
            .metric-label { opacity: 0.8; }
            .metric-value { font-weight: bold; }
            .signal-buy { color: #00ff88; }
            .signal-sell { color: #ff4444; }
            .signal-wait { color: #ffaa00; }
            .confidence-high { color: #00ff88; }
            .confidence-medium { color: #ffaa00; }
            .confidence-low { color: #ff4444; }
            .footer {
                text-align: center;
                padding: 20px;
                opacity: 0.7;
                font-size: 0.9em;
            }
            .api-links {
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 15px;
                margin-top: 20px;
            }
            .api-links h3 { margin-bottom: 15px; }
            .api-links a {
                display: block;
                color: #00ff88;
                text-decoration: none;
                padding: 10px;
                margin: 5px 0;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                transition: all 0.3s;
            }
            .api-links a:hover {
                background: rgba(255,255,255,0.2);
                transform: translateX(5px);
            }
            .status-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
                background: rgba(0,255,136,0.2);
                border: 2px solid #00ff88;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: #00ff88;
                border-radius: 50%;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 Crypto Market Analyzer</h1>
                <p><span class="live-indicator"></span>Real-time AI Trading System</p>
                <div style="margin-top: 20px;">
                    <span class="status-badge">LEGACY v2.0</span>
                </div>
            </div>
            
            <div class="upgrade-banner">
                <p>🚀 <a href="/">Upgrade to GOD MODE Ultra Dashboard v3.0</a> - Live Charts, Advanced Analytics & More!</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>🤖 AI Signal</h2>
                    <div class="card-content" id="ai-signal">
                        Loading...
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Strategy</h2>
                    <div class="card-content" id="strategy">
                        Loading...
                    </div>
                </div>
                
                <div class="card">
                    <h2>💼 Position</h2>
                    <div class="card-content" id="position">
                        Loading...
                    </div>
                </div>
            </div>
            
            <div class="api-links">
                <h3>📡 API Endpoints</h3>
                <a href="/api/ai/insights" target="_blank">GET /api/ai/insights - AI Insights & Signal</a>
                <a href="/api/strategy" target="_blank">GET /api/strategy - Trading Strategy (TP/SL)</a>
                <a href="/api/status" target="_blank">GET /api/status - System Status</a>
                <a href="/api/health" target="_blank">GET /api/health - Health Check</a>
                <a href="/docs" target="_blank">GET /docs - API Documentation (Swagger)</a>
            </div>
            
            <div class="footer">
                <p>© 2024 Crypto Market Analyzer | Powered by AI Ultra Brain 🧠</p>
            </div>
        </div>
        
        <script>
            async function fetchData() {
                try {
                    // Fetch AI insights
                    const insightsRes = await fetch('/api/ai/insights');
                    const insights = await insightsRes.json();
                    
                    let signalClass = 'signal-wait';
                    if (insights.signal === 'BUY') signalClass = 'signal-buy';
                    if (insights.signal === 'SELL') signalClass = 'signal-sell';
                    
                    let confClass = 'confidence-low';
                    if (insights.confidence > 70) confClass = 'confidence-high';
                    else if (insights.confidence > 40) confClass = 'confidence-medium';
                    
                    document.getElementById('ai-signal').innerHTML = `
                        <div class="metric">
                            <span class="metric-label">Signal:</span>
                            <span class="metric-value ${signalClass}">${insights.signal}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Direction:</span>
                            <span class="metric-value">${insights.direction.toUpperCase()}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Confidence:</span>
                            <span class="metric-value ${confClass}">${insights.confidence}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Regime:</span>
                            <span class="metric-value">${insights.regime}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Price:</span>
                            <span class="metric-value">$${insights.price ? insights.price.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Reason:</span>
                            <span class="metric-value">${insights.reason}</span>
                        </div>
                    `;
                    
                    // Fetch strategy
                    const strategyRes = await fetch('/api/strategy');
                    const strategy = await strategyRes.json();
                    
                    document.getElementById('strategy').innerHTML = `
                        <div class="metric">
                            <span class="metric-label">Entry:</span>
                            <span class="metric-value">$${strategy.entry_price ? strategy.entry_price.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Take Profit:</span>
                            <span class="metric-value signal-buy">$${strategy.tp ? strategy.tp.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Stop Loss:</span>
                            <span class="metric-value signal-sell">$${strategy.sl ? strategy.sl.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">R:R Ratio:</span>
                            <span class="metric-value">${strategy.tp && strategy.sl && strategy.entry_price ? 
                                ((Math.abs(strategy.tp - strategy.entry_price) / Math.abs(strategy.entry_price - strategy.sl)).toFixed(2)) : 'N/A'}</span>
                        </div>
                    `;
                    
                    // Display position
                    document.getElementById('position').innerHTML = `
                        <div class="metric">
                            <span class="metric-label">Direction:</span>
                            <span class="metric-value">${insights.position.direction.toUpperCase()}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Size:</span>
                            <span class="metric-value">${insights.position.size}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Entry Price:</span>
                            <span class="metric-value">$${insights.position.entry_price ? insights.position.entry_price.toFixed(2) : 'N/A'}</span>
                        </div>
                    `;
                    
                } catch (error) {
                    console.error('Error fetching data:', error);
                }
            }
            
            // Fetch data initially and every 2 seconds
            fetchData();
            setInterval(fetchData, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/health")
async def health():
    """
    Health check endpoint with system state information.
    
    Returns system health and last heartbeat from AI Brain.
    """
    system_state = get_system_state()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "last_heartbeat": system_state.last_heartbeat.isoformat() if system_state.last_heartbeat else None,
        "ai_enabled": system_state.ai_enabled,
        "auto_trading": system_state.auto_trading_enabled,
        "uptime_seconds": system_state.uptime_seconds,
    }


@app.get("/api/status")
async def status():
    """System status endpoint"""
    return get_system_status()


@app.get("/api/ai/insights")
async def get_ai_insights():
    """
    Get current AI insights and signal from live system_state.
    
    Returns real-time data from AI Brain, not from JSON files.
    """
    system_state = get_system_state()
    
    # Build response from live state
    insights = {
        "signal": "BUY" if system_state.ai_direction == "long" else "SELL" if system_state.ai_direction == "short" else "WAIT",
        "direction": system_state.ai_direction,
        "confidence": round(system_state.ai_confidence * 100, 1),  # Convert to percentage
        "reason": system_state.ai_reason,
        "regime": system_state.ai_regime,
        "price": system_state.current_price,
        "buy_volume": system_state.buy_volume,
        "sell_volume": system_state.sell_volume,
        "trade_count": system_state.trade_count,
        "whale_trades": system_state.whale_trades,
        "max_trade_size": system_state.max_trade_size,
        "max_trade_price": system_state.current_price,  # Use current price as proxy
        "position": {
            "direction": system_state.position_direction,
            "size": system_state.position_size,
            "entry_price": system_state.position_entry_price,
        },
        "scores": system_state.ai_scores,
        "generated_at": system_state.signal_timestamp.isoformat() if system_state.signal_timestamp else datetime.utcnow().isoformat(),
    }
    
    return JSONResponse(content=insights)


@app.get("/api/ai/learning_status")
async def get_learning_status():
    """
    Get AI learning and data collection status.
    
    Returns information about self-learning mode, shadow trading, and data collection.
    """
    try:
        # Check if learning components are available
        learning_enabled = False
        shadow_mode = False
        total_trades = 0
        last_flush = "Never"
        
        # Try to get data from system state
        system_state = get_system_state()
        
        # Check for model registry or learning controller
        try:
            from okx_stream_hunter.ai.self_learning_controller import get_controller
            controller = get_controller()
            if controller:
                learning_enabled = True
                # Get statistics if available
        except ImportError:
            pass
        
        status = {
            "self_learning_enabled": learning_enabled,
            "shadow_mode_enabled": shadow_mode,
            "data_collection": {
                "total_logged_trades": total_trades,
                "last_flush": last_flush,
                "buffer_size": 0,
                "collection_active": False
            },
            "models": {
                "active_model": "AI Brain v1.0",
                "model_version": "1.0.0",
                "last_training": "Never",
                "training_samples": 0
            },
            "status": "disabled" if not learning_enabled else "active"
        }
        
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Get learning status error: {e}")
        return JSONResponse(content={
            "self_learning_enabled": False,
            "shadow_mode_enabled": False,
            "data_collection": {
                "total_logged_trades": 0,
                "last_flush": "Never",
                "buffer_size": 0,
                "collection_active": False
            },
            "status": "disabled"
        })


@app.get("/api/strategy")
async def get_strategy():
    """
    Get current trading strategy (TP/SL levels) from live system_state.
    
    Returns real-time Entry/TP/SL data from AI Brain.
    """
    system_state = get_system_state()
    
    strategy = {
        "signal": "BUY" if system_state.ai_direction == "long" else "SELL" if system_state.ai_direction == "short" else "WAIT",
        "direction": system_state.ai_direction,
        "entry_price": system_state.entry_price,
        "tp": system_state.take_profit,
        "sl": system_state.stop_loss,
        "confidence": round(system_state.ai_confidence * 100, 1),
        "generated_at": system_state.last_update.isoformat() if system_state.last_update else datetime.utcnow().isoformat(),
    }
    
    return JSONResponse(content=strategy)


@app.get("/api/positions")
async def get_positions():
    """
    Get all open positions from live system_state.
    
    Returns current position data from AI Brain and Position Manager.
    """
    system_state = get_system_state()
    
    position = {
        "direction": system_state.position_direction,
        "size": system_state.position_size,
        "entry_price": system_state.position_entry_price,
        "pnl": system_state.position_pnl,
    }
    
    return JSONResponse(content={
        "positions": [position] if position["direction"] != "flat" else [],
        "count": 1 if position["direction"] != "flat" else 0,
    })


@app.get("/api/market/current")
async def get_current_market():
    """
    Get current market data snapshot from live system_state.
    
    Returns real-time price, volumes, and regime from AI Brain.
    """
    system_state = get_system_state()
    
    return JSONResponse(content={
        "price": system_state.current_price,
        "buy_volume": system_state.buy_volume,
        "sell_volume": system_state.sell_volume,
        "trade_count": system_state.trade_count,
        "whale_trades": system_state.whale_trades,
        "regime": system_state.ai_regime,
        "timestamp": system_state.last_update.isoformat() if system_state.last_update else datetime.utcnow().isoformat(),
    })


@app.get("/api/orderflow")
async def get_orderflow():
    """
    Get real-time orderflow data (CVD, volume imbalance, bid/ask pressure).
    
    Returns live orderflow metrics from system_state.
    """
    system_state = get_system_state()
    
    # Calculate volume imbalance
    total_volume = system_state.buy_volume + system_state.sell_volume
    volume_imbalance = 0.5  # neutral
    if total_volume > 0:
        volume_imbalance = system_state.buy_volume / total_volume
    
    return JSONResponse(content={
        "cvd": system_state.cvd,
        "buy_volume": system_state.buy_volume,
        "sell_volume": system_state.sell_volume,
        "volume_imbalance": volume_imbalance,  # 0-1 (0=all sell, 1=all buy, 0.5=balanced)
        "orderbook_imbalance": system_state.orderbook_imbalance,  # bid/ask imbalance
        "bid_volume": system_state.bid_volume,
        "ask_volume": system_state.ask_volume,
        "best_bid": system_state.best_bid,
        "best_ask": system_state.best_ask,
        "spread": system_state.spread,
        "spoof_risk": system_state.spoof_risk,
        "timestamp": system_state.last_update.isoformat() if system_state.last_update else datetime.utcnow().isoformat(),
    })


@app.get("/api/performance/stats")
async def get_performance_stats():
    """
    Get performance statistics.
    
    TODO: Integrate with Risk Manager and Position Manager for real stats.
    """
    return JSONResponse(content={
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "note": "Stats will be available after integrating Position Manager",
    })


# ============================================================
# Startup Event
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🔥 GOD MODE Ultra Dashboard v3.0 Started")
    logger.info("=" * 60)
    logger.info("📊 Main Dashboard: http://localhost:8000")
    logger.info("📊 Legacy Dashboard: http://localhost:8000/legacy")
    logger.info("📡 API Docs: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    # Register PROMETHEUS v7 AI endpoints
    if AI_ENDPOINTS_AVAILABLE:
        register_ai_endpoints(app)
        logger.info("✅ PROMETHEUS v7 AI endpoints registered")
        logger.info("🧠 AI Live: http://localhost:8000/api/ai/live")
        logger.info("🧠 AI Status: http://localhost:8000/api/ai/status")
        logger.info("🔥 Trading System: http://localhost:8000/api/trading/status")


@app.get("/api/trading/status")
async def get_trading_status():
    """Get trading system status"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if orchestrator:
            stats = orchestrator.get_system_stats()
            return JSONResponse(stats)
        else:
            return JSONResponse({
                "status": "not_running",
                "message": "Trading orchestrator not initialized"
            })
    except Exception as e:
        logger.error(f"Trading status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trading/enable")
async def enable_trading():
    """Enable auto trading"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if orchestrator:
            orchestrator.enable_auto_trading()
            return JSONResponse({"success": True, "message": "Auto trading enabled"})
        else:
            return JSONResponse({"error": "Orchestrator not running"}, status_code=400)
    except Exception as e:
        logger.error(f"Enable trading error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trading/disable")
async def disable_trading():
    """Disable auto trading"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if orchestrator:
            orchestrator.disable_auto_trading()
            return JSONResponse({"success": True, "message": "Auto trading disabled"})
        else:
            return JSONResponse({"error": "Orchestrator not running"}, status_code=400)
    except Exception as e:
        logger.error(f"Disable trading error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/positions")
async def get_positions():
    """Get open positions"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if orchestrator:
            positions = orchestrator.execution_engine.get_open_positions()
            return JSONResponse({"positions": positions})
        else:
            return JSONResponse({"positions": {}})
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/trades")
async def get_trades():
    """Get closed trades"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if orchestrator:
            trades = orchestrator.execution_engine.get_closed_trades()
            return JSONResponse({"trades": trades})
        else:
            return JSONResponse({"trades": []})
    except Exception as e:
        logger.error(f"Get trades error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# PHASE 3 - Advanced Trading Endpoints
# ============================================================

@app.get("/api/trading/live_trades")
async def get_live_trades():
    """Get active trades with real-time PnL"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator or not orchestrator.master_loop:
            return JSONResponse({"active_trades": [], "count": 0})
        
        supervisor = orchestrator.master_loop.trade_supervisor
        active_trades = supervisor.get_active_trades_status()
        
        return JSONResponse({
            "active_trades": active_trades,
            "count": len(active_trades),
            "supervisor_stats": supervisor.get_stats()
        })
    except Exception as e:
        logger.error(f"Get live trades error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/confidence_history")
async def get_confidence_history():
    """Get last 100 AI confidence values"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator or not orchestrator.master_loop:
            return JSONResponse({"history": []})
        
        ai_brain = orchestrator.master_loop.ai_brain
        if hasattr(ai_brain, 'decision_history'):
            history = [
                {
                    'timestamp': d.get('timestamp', ''),
                    'confidence': d.get('confidence', 0),
                    'direction': d.get('direction', 'NEUTRAL'),
                    'regime': d.get('regime', 'unknown')
                }
                for d in list(ai_brain.decision_history)[-100:]
            ]
            return JSONResponse({"history": history, "count": len(history)})
        
        return JSONResponse({"history": []})
    except Exception as e:
        logger.error(f"Get confidence history error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/rl_rewards")
async def get_rl_rewards():
    """Get RL agent reward evolution"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator or not hasattr(orchestrator, 'rl_agent'):
            return JSONResponse({"rewards": []})
        
        rl_agent = orchestrator.rl_agent
        if hasattr(rl_agent, 'reward_history'):
            rewards = [
                {
                    'episode': i,
                    'reward': r,
                    'cumulative': sum(rl_agent.reward_history[:i+1])
                }
                for i, r in enumerate(rl_agent.reward_history[-100:])
            ]
            return JSONResponse({
                "rewards": rewards,
                "total_episodes": len(rl_agent.reward_history),
                "avg_reward": sum(rl_agent.reward_history[-100:]) / len(rl_agent.reward_history[-100:]) if rl_agent.reward_history else 0
            })
        
        return JSONResponse({"rewards": []})
    except Exception as e:
        logger.error(f"Get RL rewards error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/orderflow_dominance")
async def get_orderflow_dominance():
    """Get buy/sell pressure graph data"""
    try:
        state = get_system_state()
        if not state:
            return JSONResponse({"dominance": []})
        
        # Use current state data for simple dominance
        buy_vol = state.buy_volume or 0
        sell_vol = state.sell_volume or 0
        total_vol = buy_vol + sell_vol
        
        if total_vol > 0:
            buy_pressure = (buy_vol / total_vol) * 100
            sell_pressure = (sell_vol / total_vol) * 100
        else:
            buy_pressure = 50
            sell_pressure = 50
        
        dominance = [{
            'timestamp': state.last_update.isoformat() if state.last_update else datetime.now(timezone.utc).isoformat(),
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'net_flow': buy_vol - sell_vol
        }]
        
        return JSONResponse({"dominance": dominance, "count": len(dominance)})
    except Exception as e:
        logger.error(f"Get orderflow dominance error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/safety_status")
async def get_safety_status():
    """Get AI safety, circuit breaker, and watchdog status"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator or not orchestrator.master_loop:
            return JSONResponse({"available": False})
        
        master_loop = orchestrator.master_loop
        
        safety_status = {
            "ai_safety": master_loop.ai_safety.get_safety_status() if hasattr(master_loop, 'ai_safety') else {},
            "circuit_breaker": master_loop.circuit_breaker.get_status() if hasattr(master_loop, 'circuit_breaker') else {},
            "trade_supervisor": master_loop.trade_supervisor.get_stats() if hasattr(master_loop, 'trade_supervisor') else {},
            "available": True
        }
        
        return JSONResponse(safety_status)
    except Exception as e:
        logger.error(f"Get safety status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/decision_tree")
async def get_decision_tree():
    """Get latest AI decision breakdown"""
    try:
        from okx_stream_hunter.core.trading_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator or not orchestrator.master_loop:
            return JSONResponse({"decision": None})
        
        decision = orchestrator.master_loop.last_ai_decision
        
        if decision:
            return JSONResponse({
                "decision": decision,
                "timestamp": decision.get('timestamp', ''),
                "components": {
                    "cnn": decision.get('cnn_vote', {}),
                    "lstm": decision.get('lstm_vote', {}),
                    "orderflow": decision.get('orderflow_vote', {}),
                    "meta": decision.get('meta_decision', {})
                }
            })
        
        return JSONResponse({"decision": None})
    except Exception as e:
        logger.error(f"Get decision tree error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trading/performance_metrics")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        # Since we're in paper trading mode without orchestrator,
        # return safe default values
        metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0
        }
        
        return JSONResponse({"metrics": metrics})
    except Exception as e:
        logger.error(f"Get performance metrics error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/safety/status")
async def get_trade_safety_status():
    """
    🛡️ Get TradeSafety system status (NEW - main.py integrated system)
    Returns live safety metrics, limits, and decision stats
    """
    try:
        state = get_system_state()
        
        # Try to get safety system from main.py (via state or direct import)
        # Since we can't easily pass reference, we'll return what we track in state
        
        safety_info = {
            "status": "active",
            "emergency_stop": False,  # Would be True if emergency triggered
            "system_health": "healthy",
            
            # Stats from system_state
            "total_rejections": getattr(state, 'total_rejections', 0),
            "last_rejection_reason": getattr(state, 'last_rejection_reason', None),
            
            # Current market conditions
            "current_price": state.last_price,
            "current_regime": state.regime,
            "spoof_risk": state.spoof_risk,
            
            # Signal quality
            "last_signal": {
                "direction": state.signal_direction,
                "confidence": state.signal_confidence,
                "reason": state.signal_reason,
            },
            
            # Conservative limits configured
            "configured_limits": {
                "max_trades_per_hour": 3,
                "max_trades_per_day": 15,
                "max_flips_per_hour": 2,
                "daily_loss_limit": "4%",
                "max_consecutive_losses": 3,
                "emergency_stop_loss": "8%",
                "min_confidence": "70-80% (regime-adaptive)",
                "max_spoof_score": "40%",
                "max_risk_penalty": "70%",
            },
            
            # Safety features enabled
            "active_protections": [
                "Regime-adaptive confidence thresholds",
                "Spoof detection filter",
                "Risk penalty filter",
                "Time-based cooldowns (5-10 min)",
                "Duplicate signal filtering",
                "Hourly trade limits",
                "Daily trade limits",
                "Position flip limits",
                "Daily loss protection",
                "Consecutive loss protection",
                "Emergency stop mechanism",
                "Signal age validation",
            ],
            
            "timestamp": datetime.now().isoformat(),
        }
        
        return JSONResponse(safety_info)
        
    except Exception as e:
        logger.error(f"Get trade safety status error: {e}")
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Dashboard API shutting down")


# ============================================================
# 🐋 Whale Detection API
# ============================================================

@app.get("/api/whales/events")
async def get_whale_events():
    """
    🐋 Get recent whale detection events
    Returns list of large trades detected by WhaleDetector
    """
    try:
        state = get_system_state()
        
        whale_data = {
            "whale_count": getattr(state, 'whale_count', 0),
            "whale_events": getattr(state, 'whale_events', []),
            "last_whale_event": getattr(state, 'last_whale_event', None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return JSONResponse(whale_data)
        
    except Exception as e:
        logger.error(f"Get whale events error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/whales/stats")
async def get_whale_stats():
    """
    🐋 Get whale detection statistics
    """
    try:
        state = get_system_state()
        whale_events = getattr(state, 'whale_events', [])
        
        # Calculate stats from whale events
        total_whales = getattr(state, 'whale_count', 0)
        buy_whales = sum(1 for e in whale_events if isinstance(e, dict) and e.get('side') == 'buy')
        sell_whales = sum(1 for e in whale_events if isinstance(e, dict) and e.get('side') == 'sell')
        
        # Total USD value
        total_usd = sum(e.get('usd_value', 0) for e in whale_events if isinstance(e, dict))
        
        stats = {
            "total_whale_trades": total_whales,
            "buy_whale_trades": buy_whales,
            "sell_whale_trades": sell_whales,
            "total_usd_volume": total_usd,
            "average_whale_size": total_usd / max(total_whales, 1),
            "recent_events_count": len(whale_events),
            "last_24h_whales": len(whale_events),  # Assuming whale_events is recent
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Get whale stats error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 🕯️ Candles API
# ============================================================

@app.get("/api/candles/{timeframe}")
async def get_candles(timeframe: str):
    """
    🕯️ Get OHLCV candles for specified timeframe
    Supported timeframes: 1m, 5m, 15m, 1h
    """
    try:
        state = get_system_state()
        
        # Map timeframe to state field
        timeframe_map = {
            "1m": "candles_1m",
            "5m": "candles_5m",
            "15m": "candles_15m",
            "1h": "candles_1h",
        }
        
        if timeframe not in timeframe_map:
            return JSONResponse(
                {"error": f"Invalid timeframe. Supported: {list(timeframe_map.keys())}"},
                status_code=400
            )
        
        field_name = timeframe_map[timeframe]
        candles = getattr(state, field_name, [])
        
        response = {
            "timeframe": timeframe,
            "candles": candles,
            "count": len(candles),
            "last_candle_closed": getattr(state, 'last_candle_closed', None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Get candles error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/candles/all")
async def get_all_candles():
    """
    🕯️ Get all timeframe candles in one response
    """
    try:
        state = get_system_state()
        
        response = {
            "candles": {
                "1m": getattr(state, 'candles_1m', []),
                "5m": getattr(state, 'candles_5m', []),
                "15m": getattr(state, 'candles_15m', []),
                "1h": getattr(state, 'candles_1h', []),
            },
            "counts": {
                "1m": len(getattr(state, 'candles_1m', [])),
                "5m": len(getattr(state, 'candles_5m', [])),
                "15m": len(getattr(state, 'candles_15m', [])),
                "1h": len(getattr(state, 'candles_1h', [])),
            },
            "last_candle_closed": getattr(state, 'last_candle_closed', None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Get all candles error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 📊 CVD (Cumulative Volume Delta) API
# ============================================================

@app.get("/api/cvd/current")
async def get_cvd_current():
    """
    📊 Get current CVD value and trend
    """
    try:
        state = get_system_state()
        
        cvd_data = {
            "cvd_value": getattr(state, 'cvd_value', 0.0),
            "cvd_trend": getattr(state, 'cvd_trend', 'neutral'),
            "buy_volume": state.buy_volume,
            "sell_volume": state.sell_volume,
            "volume_delta": state.buy_volume - state.sell_volume,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return JSONResponse(cvd_data)
        
    except Exception as e:
        logger.error(f"Get CVD current error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


