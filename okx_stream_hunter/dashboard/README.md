# 🔥 GOD MODE Ultra Dashboard v3.0

A professional, multi-panel real-time trading dashboard for the Crypto Market Analyzer (OKX Stream Hunter).

## 🎯 Features

### 📊 4-Panel Main Grid

1. **Live Candles Chart (Panel A)**
   - Real-time candlestick chart using TradingView Lightweight Charts
   - Live price updates every 2-3 seconds
   - Entry, TP, and SL lines overlayed on the chart
   - Signal markers for LONG/SHORT entries
   - Auto-builds 1-minute candles from tick data

2. **AI Signal & Confidence (Panel B)**
   - BUY/SELL/FLAT signal badge
   - Direction indicator (LONG/SHORT/FLAT)
   - Circular confidence gauge (0-100%)
   - Market regime display (trend/range/choppy)
   - Detailed reason with tooltip
   - Detection scores badges:
     - Trend Score
     - Orderflow Score
     - Microstructure Score
     - Spoof Risk

3. **Orderflow & Microstructure (Panel C)**
   - Buy vs Sell volume bar charts
   - Orderbook imbalance meter
   - Spoofing risk indicator
   - Spread and depth metrics
   - Whale trades counter

4. **Position & Risk (Panel D)**
   - Current position details (direction, size, entry)
   - Unrealized PnL
   - Risk per trade percentage
   - Risk:Reward ratio
   - Recent trades history (last 5-10 trades)

### 🎨 Additional Features

- **Top Bar Market Overview**: Symbol, current price, 24h change, spread, regime tags, volatility level, spoof risk, mode (Paper/Live)
- **Recent Signals Table**: History of last 20 signals with time, signal, direction, confidence, price, regime, and reason
- **System Status & Heartbeat**: AI status, auto-trading status, last heartbeat, dashboard version
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Error Handling**: Graceful degradation if API endpoints fail
- **Live Updates**: Auto-refresh without page reload

## 🚀 Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `jinja2` - Template engine

### Running the Dashboard

Start the dashboard server:

```bash
# From project root
python3 -m uvicorn okx_stream_hunter.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```

Or use the included script:

```bash
# From project root
python3 -c "import uvicorn; uvicorn.run('okx_stream_hunter.dashboard.app:app', host='0.0.0.0', port=8000, reload=True)"
```

### Accessing the Dashboard

- **Main Dashboard (v3.0 Ultra)**: http://localhost:8000
- **Legacy Dashboard (v2.0)**: http://localhost:8000/legacy
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📡 API Endpoints

The dashboard consumes the following API endpoints:

| Endpoint | Method | Description | Polling Interval |
|----------|--------|-------------|------------------|
| `/api/ai/insights` | GET | AI signal, scores, position data | 2 seconds |
| `/api/strategy` | GET | Entry, TP, SL, R:R ratio | 5 seconds |
| `/api/status` | GET | System status, AI/trading enabled | 5 seconds |
| `/api/health` | GET | Health check | On demand |
| `/api/market/current` | GET | Current market snapshot | On demand |

## 🎨 Customization

### Adjusting Polling Intervals

Edit `/static/dashboard.js`:

```javascript
const CONFIG = {
    // Polling intervals (in milliseconds)
    INSIGHTS_POLL_INTERVAL: 2000,      // Poll AI insights every 2 seconds
    STRATEGY_POLL_INTERVAL: 5000,      // Poll strategy every 5 seconds
    STATUS_POLL_INTERVAL: 5000,        // Poll system status every 5 seconds
};
```

### Changing Chart Settings

```javascript
const CONFIG = {
    CHART: {
        MAX_CANDLES: 100,               // Maximum candles to display
        CANDLE_INTERVAL: '1m',          // Candle interval
    },
};
```

### Modifying Max History

```javascript
const CONFIG = {
    MAX_SIGNALS_HISTORY: 20,            // Max signals to show in table
    MAX_TRADES_HISTORY: 10,             // Max trades to show in recent trades
};
```

## 🔧 Configuration

### Port Configuration

To change the port, modify the uvicorn command:

```bash
uvicorn okx_stream_hunter.dashboard.app:app --host 0.0.0.0 --port 8080
```

### CORS Configuration

CORS is enabled by default in `app.py` to allow all origins. To restrict:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📱 Mobile Access

The dashboard is fully responsive and can be accessed from mobile devices:

1. **GitHub Codespaces**: Use port forwarding to access from mobile browser
2. **Local Network**: Access via `http://<your-ip>:8000`
3. **Production**: Deploy with a reverse proxy (nginx, Caddy) with HTTPS

### GitHub Codespaces Setup

1. Ports tab → Forward Port 8000
2. Set visibility to "Public"
3. Access the generated URL from your mobile device

## 🎯 How It Works

### JavaScript Polling Architecture

1. **Initialization** (`initDashboard()`):
   - Creates the Lightweight Charts instance
   - Starts initial data fetch
   - Sets up polling intervals

2. **Data Fetching** (every 2-5 seconds):
   - `pollInsights()` → fetches AI signal data
   - `pollStrategy()` → fetches TP/SL levels
   - `pollStatus()` → fetches system status

3. **UI Updates**:
   - `updateTopBar()` → market overview
   - `updateAISignalPanel()` → signal & confidence
   - `updateOrderflowPanel()` → volume & microstructure
   - `updatePositionPanel()` → position & risk
   - `updateSignalsTable()` → recent signals
   - `updateSystemStatus()` → system health

4. **Chart Updates**:
   - `updateChart()` → adds/updates candlesticks
   - `updatePriceLines()` → draws Entry/TP/SL lines

### Error Handling

- If an endpoint fails 3+ times, an error badge appears in the top bar
- Other panels continue working independently
- Graceful fallback to "Loading..." or placeholder text

## 🧪 Testing

### Test with Mock Data

You can test the dashboard without running the full trading system by creating mock JSON files:

**`insights.json`**:
```json
{
  "signal": "BUY",
  "direction": "long",
  "confidence": 85,
  "reason": "Strong orderflow imbalance + trend confirmation",
  "regime": "trend",
  "price": 43250.50,
  "buy_volume": 1250000,
  "sell_volume": 850000,
  "trade_count": 245,
  "whale_trades": 5,
  "max_trade_size": 150000,
  "max_trade_price": 43251.00,
  "position": {
    "direction": "flat",
    "size": 0,
    "entry_price": null
  },
  "scores": {
    "trend_score": 0.85,
    "orderflow_score": 0.78,
    "microstructure_score": 0.72,
    "spoof_risk": 0.15,
    "orderbook_imbalance": 0.65
  },
  "generated_at": "2024-01-15T10:30:45.123Z"
}
```

**`strategy.json`**:
```json
{
  "signal": "BUY",
  "direction": "long",
  "entry_price": 43250.50,
  "tp": 43850.00,
  "sl": 42950.00,
  "confidence": 85,
  "generated_at": "2024-01-15T10:30:45.123Z"
}
```

## 🐛 Troubleshooting

### Chart Not Showing

1. Check browser console for errors
2. Verify Lightweight Charts CDN is loading
3. Ensure `insights.json` has valid price data
4. Check that `/api/ai/insights` returns data

### No Data Updating

1. Verify the trading system is running and generating `insights.json`
2. Check API endpoints are accessible: `curl http://localhost:8000/api/health`
3. Open browser console to see polling errors
4. Check file permissions on `insights.json` and `strategy.json`

### Port Already in Use

```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn okx_stream_hunter.dashboard.app:app --port 8080
```

## 📝 File Structure

```
okx_stream_hunter/dashboard/
├── app.py                    # FastAPI application
├── templates/
│   └── dashboard.html        # Main dashboard template
├── static/
│   └── dashboard.js          # Dashboard JavaScript logic
└── README.md                 # This file
```

## 🔐 Security Notes

- **Production Deployment**: 
  - Use HTTPS (SSL/TLS)
  - Restrict CORS origins
  - Add authentication (OAuth, JWT)
  - Rate limiting
  - Hide sensitive data (API keys, balances)

- **Paper Trading Mode**: Dashboard defaults to showing "Paper Trading" mode. Switch to "Live Trading" only after extensive testing.

## 🚀 Performance

- **Lightweight**: Uses CDN for libraries (no build process)
- **Efficient Polling**: Staggered intervals (2s, 5s) to reduce load
- **Chart Optimization**: Limits to 100 candles, uses efficient rendering
- **Error Recovery**: Continues working even if some endpoints fail

## 📊 Screenshots

### Desktop View
- 2x2 grid layout with all 4 panels visible
- Top bar with market overview
- Recent signals table and system status below

### Mobile View
- Stacked cards for easy scrolling
- Touch-friendly interface
- All features accessible

## 🤝 Contributing

To add new features to the dashboard:

1. **New Panel**: Add HTML to `templates/dashboard.html`
2. **New Data**: Create function in `static/dashboard.js` to fetch and update
3. **New Endpoint**: Add route to `app.py` if needed
4. **Update Polling**: Add to `initDashboard()` if data needs periodic refresh

## 📜 License

Same as parent project (Crypto Market Analyzer).

## 🙏 Credits

- **TradingView Lightweight Charts**: For the beautiful candlestick chart
- **Tailwind CSS**: For rapid UI styling
- **FastAPI**: For the lightning-fast backend
- **Chart.js**: For additional charting capabilities (optional)

---

**🔥 GOD MODE Ultra Dashboard v3.0** - Built for Professional Crypto Traders 🚀
