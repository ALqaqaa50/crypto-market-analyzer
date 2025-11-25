# 🔥👑 Crypto Market Analyzer - GOD MODE Edition

## نظام تداول ذكاء اصطناعي متقدم مع تداول تلقائي كامل

---

## ✨ المميزات الجديدة

### 1️⃣ **Auto-Trading System (التداول التلقائي)**
- ✅ دخول صفقات Long/Short تلقائياً بناءً على إشارات الـ AI
- ✅ وضع TP (Take Profit) و SL (Stop Loss) تلقائي
- ✅ تتبع الصفقات وإدارتها في الوقت الفعلي
- ✅ Trailing Stop Loss لحماية الأرباح
- ✅ Break-even SL adjustment
- ✅ إغلاق تلقائي عند الوصول للهدف أو الخسارة

**الملفات:**
- `okx_stream_hunter/integrations/position_manager.py` - إدارة الصفقات
- `okx_stream_hunter/integrations/trade_executor.py` - تنفيذ الصفقات عبر OKX API

---

### 2️⃣ **Risk Management (إدارة المخاطر المحترفة)**
- ✅ Dynamic position sizing بناءً على نسبة المخاطرة
- ✅ Kelly Criterion للحجم الأمثل
- ✅ Volatility-adjusted sizing (تكبير/تصغير الحجم حسب التقلبات)
- ✅ حماية رأس المال مع Max Daily Loss
- ✅ Win/Loss streak tracking و adaptive sizing
- ✅ R:R ratio validation (الحد الأدنى 1.5:1)
- ✅ Drawdown protection

**الملف:**
- `okx_stream_hunter/integrations/risk_manager.py`

**مثال:**
```python
from okx_stream_hunter.integrations.risk_manager import RiskManager, RiskConfig

# إنشاء Risk Manager
risk_config = RiskConfig(
    account_balance=1000.0,  # رأس المال
    max_risk_per_trade_pct=0.01,  # 1% مخاطرة لكل صفقة
    max_daily_loss_pct=0.05,  # 5% خسارة يومية قصوى
    min_rr_ratio=1.5,  # R:R ratio أدنى
)
risk_manager = RiskManager(config=risk_config)

# تقييم صفقة
risk_assessment = risk_manager.assess_trade(
    symbol="BTC-USDT-SWAP",
    direction="long",
    entry_price=100000.0,
    sl_price=99000.0,
    confidence=0.75,
    volatility=0.02,
)

if risk_assessment.approved:
    print(f"✅ Trade approved! Size: {risk_assessment.position_size}")
else:
    print(f"❌ Trade rejected: {risk_assessment.reason}")
```

---

### 3️⃣ **Dashboard & Real-time Insights (لوحة التحكم)**
- ✅ واجهة ويب تفاعلية لعرض إشارات الـ AI مباشرةً
- ✅ عرض الثقة (Confidence) والاتجاه (Direction) والنظام (Regime)
- ✅ عرض TP/SL ومستويات الدخول
- ✅ عرض الصفقات المفتوحة والـ P&L
- ✅ API endpoints كاملة للتكامل

**الملف:**
- `okx_stream_hunter/dashboard/app.py`

**تشغيل Dashboard:**
```bash
python3 main.py
```
ثم افتح: `http://localhost:8000`

**API Endpoints:**
- `GET /` - الصفحة الرئيسية (UI تفاعلي)
- `GET /api/ai/insights` - إشارات الـ AI الحالية
- `GET /api/strategy` - استراتيجية التداول (TP/SL)
- `GET /api/status` - حالة النظام
- `GET /api/positions` - الصفقات المفتوحة
- `GET /docs` - API Documentation (Swagger)

---

### 4️⃣ **Trading Engine (محرك التداول الاحترافي)**
- ✅ State Machine كاملة (IDLE → ANALYZING → IN_POSITION → EXIT)
- ✅ Market Regime Detection (Trending/Ranging/Volatile)
- ✅ Adaptive parameters لكل نظام سوق
- ✅ Rate limiting (حد أقصى للصفقات بالساعة/اليوم)
- ✅ Cooldown بعد الخسائر
- ✅ Emergency close all positions

**الملف:**
- `okx_stream_hunter/core/trading_engine.py`

**مثال:**
```python
from okx_stream_hunter.core.trading_engine import TradingEngine, TradingEngineConfig

config = TradingEngineConfig(
    adapt_to_regime=True,  # تكييف حسب نظام السوق
    cooldown_after_loss_seconds=300,  # 5 دقائق راحة بعد الخسارة
    max_trades_per_day=50,
)

engine = TradingEngine(
    ai_brain=brain,
    risk_manager=risk_manager,
    position_manager=position_manager,
    config=config,
)

await engine.start()
```

---

### 5️⃣ **Backtesting & Optimization (اختبار الاستراتيجيات)**
- ✅ اختبار استراتيجيات الـ AI على بيانات تاريخية
- ✅ حساب Win Rate, Sharpe Ratio, Max Drawdown
- ✅ تصدير النتائج والـ equity curve
- ✅ محاكاة كاملة لإدارة المخاطر والصفقات
- ✅ Parameter optimization support

**الملف:**
- `okx_stream_hunter/backtesting/engine.py`

**مثال:**
```python
from okx_stream_hunter.backtesting.engine import BacktestEngine
from okx_stream_hunter.backtesting.data_loader import HistoricalDataLoader
from datetime import datetime, timedelta

# إنشاء Backtest Engine
loader = HistoricalDataLoader(db_pool)
backtest = BacktestEngine(
    ai_brain=brain,
    data_loader=loader,
    speed_multiplier=100.0,  # 100x أسرع
    initial_balance=1000.0,
)

# تشغيل Backtest
result = await backtest.run_backtest(
    symbol="BTC-USDT-SWAP",
    timeframe="1m",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
)

# عرض النتائج
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Total P&L: ${result.total_pnl:.2f}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
```

---

### 6️⃣ **🔥👑 GOD MODE (الوضع النهائي)**

**أقوى وضع تداول في النظام - يجمع كل المكونات:**

- ✅ AI Brain + Real-time Analysis
- ✅ Auto Trading + Position Management
- ✅ Risk Management + Dynamic Sizing
- ✅ Market Regime Adaptation
- ✅ Self-Learning & Performance Tracking
- ✅ Emergency Controls
- ✅ Comprehensive Logging

**الملف:**
- `okx_stream_hunter/core/god_mode.py`

**تشغيل GOD MODE:**

```python
from okx_stream_hunter.core.god_mode import launch_god_mode

# تشغيل (Paper Trading)
god = await launch_god_mode(
    symbol="BTC-USDT-SWAP",
    initial_balance=1000.0,
    enable_live_trading=False,  # Paper trading (آمن)
)

# مراقبة الحالة
status = god.get_status()
print(status)

# إيقاف مؤقت
god.pause()

# استئناف
god.resume()

# إغلاق جميع الصفقات
await god.close_all_positions()

# إيقاف كامل
await god.stop()

# 🚨 إيقاف طارئ
await god.emergency_stop()
```

**⚠️ تحذير:** لتفعيل التداول الحقيقي:
```python
god = await launch_god_mode(enable_live_trading=True)  # حذار! مال حقيقي!
```

---

## 📊 هيكل المشروع المحدّث

```
okx_stream_hunter/
├── core/
│   ├── ai_brain.py          # AI Brain للتحليل
│   ├── trading_engine.py    # 🔥 محرك التداول الجديد
│   ├── god_mode.py          # 👑 GOD MODE
│   ├── auto_trader.py
│   ├── processor.py
│   └── ...
├── integrations/
│   ├── position_manager.py  # 🔥 إدارة الصفقات (جديد)
│   ├── risk_manager.py      # 🔥 إدارة المخاطر (جديد)
│   ├── trade_executor.py    # تحديث: دعم TP/SL
│   └── claude_analyzer.py
├── backtesting/
│   ├── engine.py            # 🔥 محرك Backtesting محسّن
│   ├── data_loader.py
│   └── reporter.py
├── dashboard/
│   ├── app.py               # 🔥 FastAPI Dashboard (جديد)
│   └── __init__.py
└── ...
```

---

## 🚀 البدء السريع

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. إعداد متغيرات البيئة

```bash
# في ملف .env أو تصدير مباشر
export NEON_DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export CLAUDE_API_KEY="your_claude_api_key"
export OKX_API_KEY="your_okx_api_key"
export OKX_SECRET_KEY="your_okx_secret"
export OKX_PASSPHRASE="your_okx_passphrase"
```

تحديث `config/settings.yaml`:
```yaml
database:
  enabled: true
  url: "${NEON_DATABASE_URL}"
```

### 3. تشغيل النظام

**الوضع العادي (AI Brain + Dashboard):**
```bash
python3 main.py
```

**GOD MODE (من داخل Python):**
```python
import asyncio
from okx_stream_hunter.core.god_mode import launch_god_mode

async def main():
    # Paper Trading
    god = await launch_god_mode()
    
    # اترك النظام يعمل
    try:
        while True:
            await asyncio.sleep(60)
            god.print_status()
    except KeyboardInterrupt:
        await god.stop()

asyncio.run(main())
```

### 4. افتح Dashboard
```
http://localhost:8000
```

---

## 📈 أمثلة متقدمة

### مثال 1: Backtesting استراتيجية AI
```python
from datetime import datetime, timedelta
from okx_stream_hunter.core.ai_brain import AIBrain
from okx_stream_hunter.backtesting.engine import BacktestEngine
from okx_stream_hunter.backtesting.data_loader import HistoricalDataLoader

async def backtest_strategy():
    # إنشاء AI Brain
    brain = AIBrain(symbol="BTC-USDT-SWAP")
    
    # تحميل بيانات تاريخية
    loader = HistoricalDataLoader(db_pool)
    
    # إنشاء Backtest
    backtest = BacktestEngine(
        ai_brain=brain,
        data_loader=loader,
        initial_balance=1000.0,
    )
    
    # تشغيل
    result = await backtest.run_backtest(
        symbol="BTC-USDT-SWAP",
        timeframe="1m",
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
    )
    
    # تصدير
    backtest.export_trades("my_trades.json")
    backtest.export_equity_curve("my_equity.json")
    
    return result
```

### مثال 2: تخصيص Risk Management
```python
from okx_stream_hunter.integrations.risk_manager import RiskManager, RiskConfig

# إعداد مخصص للمخاطر
config = RiskConfig(
    account_balance=5000.0,
    max_risk_per_trade_pct=0.02,  # 2% مخاطرة
    max_daily_loss_pct=0.08,  # 8% خسارة يومية
    default_rr_ratio=3.0,  # R:R = 1:3
    enable_volatility_adjustment=True,
    enable_drawdown_protection=True,
    consecutive_losses_limit=3,
)

risk_manager = RiskManager(config=config)
```

### مثال 3: Trading Engine مع Callbacks
```python
from okx_stream_hunter.core.trading_engine import TradingEngine

def on_trade_opened(signal, size, tp, sl):
    print(f"🔔 New trade: {signal['direction']} @ {signal['price']}")
    # أرسل إشعار Telegram/Discord/Email
    send_notification(f"Opened {signal['direction']} position")

def on_trade_closed(position, is_win):
    result = "WIN ✅" if is_win else "LOSS ❌"
    print(f"🔔 Trade closed: {result} - P&L: ${position.realized_pnl:.2f}")
    send_notification(f"Closed position: {result}")

engine = TradingEngine(
    ai_brain=brain,
    on_trade_opened=on_trade_opened,
    on_trade_closed=on_trade_closed,
)

await engine.start()
```

---

## 🎯 استراتيجيات التداول المدعومة

### 1. Trend Following
- يتكيف مع الاتجاهات الصاعدة/الهابطة
- يزيد حجم الصفقة في الاتجاهات القوية
- TP أوسع لالتقاط المزيد من الحركة

### 2. Mean Reversion (Range Trading)
- يستفيد من الارتداد في النطاقات
- TP أضيق لالتقاط أرباح سريعة
- ثقة أعلى مطلوبة قبل الدخول

### 3. Volatility Breakout
- يقلل الحجم في الأسواق المتقلبة
- SL أوسع لتجنب الخروج المبكر
- يتطلب ثقة عالية جداً

---

## 📚 الوثائق الكاملة

### API Documentation
```
http://localhost:8000/docs
```

### ملفات JSON الناتجة
- `insights.json` - إشارات الـ AI الحالية
- `strategy.json` - TP/SL والاستراتيجية
- `backtest_result.json` - نتائج Backtesting
- `god_mode_session_*.json` - ملخص جلسة GOD MODE
- `god_mode_learning_data.json` - بيانات التعلّم

---

## ⚠️ تحذيرات مهمة

1. **التداول الحقيقي خطر:** استخدم Paper Trading أولاً
2. **اختبر الاستراتيجيات:** استخدم Backtesting قبل التداول
3. **إدارة المخاطر:** لا تخاطر بأكثر من 1-2% لكل صفقة
4. **مراقبة يومية:** راقب الأداء وأوقف النظام عند الحاجة
5. **متغيرات البيئة:** تأكد من ضبط API Keys بشكل صحيح

---

## 🆘 استكشاف الأخطاء

### المشكلة: Dashboard لا يعمل
```bash
# تأكد من تثبيت uvicorn
pip install uvicorn fastapi

# تحقق من المنفذ
netstat -tulpn | grep 8000
```

### المشكلة: قاعدة البيانات لا تتصل
```bash
# تحقق من URL
echo $NEON_DATABASE_URL

# اختبار الاتصال
psql $NEON_DATABASE_URL -c "SELECT 1"
```

### المشكلة: OKX API لا يعمل
- تأكد من صحة API Keys
- تأكد من تفعيل Trading في OKX
- تحقق من IP Whitelist في OKX

---

## 🤝 المساهمة

نرحب بالمساهمات! افتح Issue أو Pull Request.

---

## 📝 الترخيص

MIT License

---

## 🔥 ملخص سريع

```bash
# تثبيت
pip install -r requirements.txt

# إعداد
export NEON_DATABASE_URL="..."
export CLAUDE_API_KEY="..."

# تشغيل
python3 main.py

# Dashboard
http://localhost:8000

# GOD MODE
python3 -c "import asyncio; from okx_stream_hunter.core.god_mode import launch_god_mode; asyncio.run(launch_god_mode())"
```

---

**🔥👑 Happy Trading with GOD MODE! 🚀**
