# 🔥 GOD MODE Ultra Dashboard v3.0 - Quick Start Guide

## ✅ Implementation Complete!

Your professional trading dashboard is ready to use!

---

## 🚀 Quick Start (3 Steps)

### 1. Start the Dashboard Server

```bash
# From project root
python3 -m uvicorn okx_stream_hunter.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open in Browser

Visit: **http://localhost:8000**

### 3. (Optional) Start Trading System

```bash
# In another terminal, start the trading system to generate real data
python3 main.py
```

---

## 📊 What You'll See

### Desktop View:
```
┌─────────────────────────────────────────────────────────┐
│  BTC-USDT-SWAP  |  $43,250.50  |  +2.5%  |  [LIVE]     │
│  Trend | Normal Vol | Clean Orders | Paper Trading      │
└─────────────────────────────────────────────────────────┘
┌─────────────────────┬───────────────────────────────────┐
│   📊 Live Candles   │   🤖 AI Signal & Confidence       │
│   Chart             │   BUY / LONG                      │
│   Entry: $43,250    │   Confidence: 85%                 │
│   TP: $43,850       │   Regime: trend                   │
│   SL: $42,950       │   Scores: [Trend 85% ▸ ...]      │
├─────────────────────┼───────────────────────────────────┤
│   📈 Orderflow &    │   💼 Position & Risk              │
│   Microstructure    │   Direction: FLAT                 │
│   Buy: 1.5M ▓▓▓▓▓   │   Size: 0                         │
│   Sell: 850K ▓▓▓    │   R:R: 2:1                        │
│   Imbalance: 65%    │   Recent Trades: [...]            │
└─────────────────────┴───────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  📡 Recent Signals                                      │
│  Time    | Signal | Direction | Confidence | Price     │
│  10:30   | BUY    | LONG      | 85%        | $43,250   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  ⚡ System Status                                       │
│  AI: ON  |  Trading: OFF  |  Heartbeat: 10:30:45      │
└─────────────────────────────────────────────────────────┘
```

---

## 📱 Access from Mobile

### GitHub Codespaces:
1. Go to **Ports** tab
2. Forward port 8000
3. Set visibility to "Public"
4. Copy the generated URL
5. Open on your mobile browser

### Local Network:
```bash
# Find your IP
hostname -I

# Access from phone
http://<your-ip>:8000
```

---

## 🔧 Configuration

### Adjust Update Intervals

Edit `okx_stream_hunter/dashboard/static/dashboard.js`:

```javascript
const CONFIG = {
    INSIGHTS_POLL_INTERVAL: 2000,   // Poll AI insights every 2 seconds
    STRATEGY_POLL_INTERVAL: 5000,   // Poll strategy every 5 seconds
    STATUS_POLL_INTERVAL: 5000,     // Poll system status every 5 seconds
};
```

**Recommendations:**
- **Fast internet**: Use 2s, 3s, 5s
- **Slow internet**: Use 5s, 10s, 10s
- **Mobile data**: Use 10s, 15s, 15s

### Change Max Candles

```javascript
const CONFIG = {
    CHART: {
        MAX_CANDLES: 100,  // Change to 50 for faster, 200 for more history
    },
};
```

---

## 📡 Available Endpoints

| Endpoint | Description | Used By |
|----------|-------------|---------|
| `/` | Main dashboard v3.0 | Browser |
| `/legacy` | Old dashboard v2.0 | Browser (fallback) |
| `/api/ai/insights` | AI signal, scores, position | Dashboard (2s poll) |
| `/api/strategy` | Entry, TP, SL, R:R | Dashboard (5s poll) |
| `/api/status` | System status, heartbeat | Dashboard (5s poll) |
| `/api/health` | Health check | Monitoring |
| `/docs` | Swagger API docs | Developers |

---

## 🎨 Dashboard Features

### 🟢 Top Bar
- **Symbol**: Trading pair (e.g., BTC-USDT-SWAP)
- **Current Price**: Real-time price with $ sign
- **24h Change**: Percentage change (green if positive, red if negative)
- **Live Indicator**: Green pulsing dot
- **Regime Tag**: Trend/Range/Choppy
- **Volatility Tag**: Low/Normal/High
- **Spoof Risk Tag**: Clean/Medium/High
- **Mode Tag**: Paper Trading / Live Trading

### 📊 Panel A: Live Candles Chart
- **Candlestick Chart**: Green (up) / Red (down)
- **Price Lines**:
  - Blue dashed: Entry price
  - Green solid: Take Profit (TP)
  - Red solid: Stop Loss (SL)
- **Auto-Building**: Creates 1-minute candles from tick data
- **Max History**: Shows last 100 candles
- **Zoom/Pan**: Scroll to zoom, drag to pan

### 🤖 Panel B: AI Signal & Confidence
- **Signal Badge**:
  - 🟢 BUY (green) → Buy signal
  - 🔴 SELL (red) → Sell signal
  - 🟡 WAIT (yellow) → No signal
- **Direction**: LONG / SHORT / FLAT
- **Confidence Gauge**: Circular meter (0-100%)
  - 70%+ → Green (high confidence)
  - 40-70% → Yellow (medium confidence)
  - <40% → Red (low confidence)
- **Regime**: Market type (trend/range/choppy)
- **Reason**: Why the AI made this decision (hover for full text)
- **Detection Scores**: Individual algorithm scores
  - Trend Score
  - Orderflow Score
  - Microstructure Score
  - Spoof Risk
  - And more...

### 📈 Panel C: Orderflow & Microstructure
- **Volume Bars**:
  - Green bar: Buy volume
  - Red bar: Sell volume
  - Width shows relative strength
- **Orderbook Imbalance Meter**:
  - 0% → Heavy sell orders (asks)
  - 50% → Balanced
  - 100% → Heavy buy orders (bids)
- **Spoofing Risk Meter**:
  - 0% → Clean orderbook
  - 50% → Some spoofing
  - 100% → High spoofing risk
- **Additional Metrics**:
  - Spread in basis points
  - Depth imbalance
  - Whale trades count

### 💼 Panel D: Position & Risk
- **Current Position**:
  - Direction: LONG / SHORT / FLAT
  - Size: Position size
  - Entry Price: Entry price
  - Unrealized PnL: Current profit/loss
- **Risk Metrics**:
  - Risk per Trade: % of balance (default 2%)
  - R:R Ratio: Reward to Risk ratio (e.g., 3:1)
- **Recent Trades**: Last 5-10 trades (when available)

### 📡 Recent Signals Table
- Shows last 20 signals
- Columns: Time, Signal, Direction, Confidence, Price, Regime, Reason
- Color-coded for easy reading
- Scrollable if many signals

### ⚡ System Status & Heartbeat
- **AI Status**: ON/OFF
- **Auto Trading Status**: ON/OFF (Paper vs Live)
- **Last Heartbeat**: Timestamp of last system check
- **Dashboard Version**: v3.0 Ultra

---

## 🐛 Troubleshooting

### Problem: Dashboard shows "Loading..." forever

**Solution 1**: Check if trading system is running
```bash
# Start the trading system to generate data
python3 main.py
```

**Solution 2**: Create mock data files
```bash
# insights.json already exists with mock data
# strategy.json already exists with mock data
cat insights.json
cat strategy.json
```

### Problem: Chart not showing

**Solution 1**: Check browser console (F12)
- Look for JavaScript errors
- Check if Lightweight Charts CDN loaded

**Solution 2**: Verify API returns data
```bash
curl http://localhost:8000/api/ai/insights
```

**Solution 3**: Check price field exists
```bash
# insights.json must have "price" field
cat insights.json | grep price
```

### Problem: Port 8000 already in use

**Solution**:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python3 -m uvicorn okx_stream_hunter.dashboard.app:app --port 8080
```

### Problem: API errors in top bar

**Solution 1**: Check server logs
```bash
# Look at terminal where uvicorn is running
# Check for error messages
```

**Solution 2**: Verify file permissions
```bash
ls -la insights.json strategy.json
```

**Solution 3**: Test endpoints manually
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/status
curl http://localhost:8000/api/ai/insights
```

### Problem: Mobile display issues

**Solution**:
- Clear browser cache
- Try different browser (Chrome, Safari, Firefox)
- Check if port forwarding is set to "Public" (in Codespaces)
- Verify mobile data/WiFi connection

---

## 📚 Documentation

### Full Documentation:
- **English Guide**: `okx_stream_hunter/dashboard/README.md`
- **Arabic Guide**: `GOD_MODE_DASHBOARD_v3.0_GUIDE_AR.md`
- **Implementation Summary**: `DASHBOARD_IMPLEMENTATION_SUMMARY.md`

### API Documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🎯 Testing with Mock Data

The project includes mock data files for testing:

### `insights.json`:
```json
{
  "signal": "BUY",
  "direction": "long",
  "confidence": 85,
  "price": 43250.50,
  "scores": { ... },
  ...
}
```

### `strategy.json`:
```json
{
  "entry_price": 43250.50,
  "tp": 43850.00,
  "sl": 42950.00,
  ...
}
```

**You can edit these files to test different scenarios!**

---

## 🔐 Security Notes

### ⚠️ For Production Use:

1. **Enable HTTPS**:
```bash
# Use reverse proxy (nginx, Caddy) with SSL certificate
```

2. **Restrict CORS** in `app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Not "*"
    ...
)
```

3. **Add Authentication**:
- Use OAuth (Google, GitHub)
- Or JWT tokens
- Or Basic Auth with strong password

4. **Hide Sensitive Data**:
- Don't show API keys
- Mask full balance amounts
- Use environment variables

5. **Rate Limiting**:
- Limit API calls per IP
- Prevent abuse

---

## 📊 Performance Tips

### For Best Performance:

1. **Optimize Polling**:
   - Use longer intervals on slow connections
   - Use shorter intervals on fast connections

2. **Reduce Max Candles**:
   - Lower `MAX_CANDLES` to 50 if chart is slow
   - Increase to 200 for more history

3. **Use CDN**:
   - Libraries load from CDN (faster, cached)
   - No build process needed

4. **Close Unused Tabs**:
   - Dashboard uses resources (CPU, RAM)
   - Close if not actively monitoring

---

## 🎉 You're All Set!

Your **GOD MODE Ultra Dashboard v3.0** is ready!

### Next Steps:

1. ✅ **Open the Dashboard**: http://localhost:8000
2. ✅ **Start Trading System**: `python3 main.py` (optional)
3. ✅ **Monitor Live Data**: Watch the panels update in real-time
4. ✅ **Test on Mobile**: Forward port and access from phone
5. ✅ **Customize**: Edit `dashboard.js` to adjust settings

---

## 📞 Support

### If you need help:

1. **Check Console**: Browser console (F12) for JavaScript errors
2. **Check Server Logs**: Terminal where uvicorn is running
3. **Read Documentation**: Full guides in English and Arabic
4. **Test Endpoints**: Use `curl` to verify API responses
5. **Open Issue**: Create GitHub issue with details

---

## 🚀 Enjoy Your Professional Trading Dashboard!

**Built with:**
- FastAPI 🔥
- TradingView Lightweight Charts 📊
- Tailwind CSS 🎨
- Jinja2 📝
- JavaScript ES6+ 💻

**Features:**
- ✅ Real-time updates (2-5 seconds)
- ✅ Live candlestick charts
- ✅ AI signal visualization
- ✅ Position & risk management
- ✅ Mobile-responsive
- ✅ Professional design
- ✅ Error handling
- ✅ No build process needed

---

**🔥 GOD MODE Ultra Dashboard v3.0** - Trade Like a Pro! 🚀💰📈
