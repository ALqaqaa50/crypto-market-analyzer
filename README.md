# 🚀 Crypto Market Analyzer

**نظام تحليل متقدم لأسواق العملات الرقمية مع دعم البيانات الحية من OKX**

## 📋 نظرة عامة

نظام تحليل شامل ومتطور لأسواق العملات الرقمية يوفر:

- **📊 بيانات حية متعددة الأطر الزمنية** - من 1 ثانية إلى يوم كامل
- **🐋 كشف تحركات الحيتان** - تتبع الصفقات الكبيرة والتلاعب بدفتر الأوامر
- **📈 تحليل حجم التداول المتقدم** - VWAP, CVD, Volume Profile
- **🎯 كشف الأنماط** - مستويات الدعم والمقاومة التلقائية
- **⚡ أداء عالي** - معالجة غير متزامنة مع تجميع الكتابة
- **🗄️ تخزين Neon PostgreSQL** - مع دعم TimescaleDB للبيانات الزمنية
- **🧪 Backtesting Engine** - اختبار الاستراتيجيات على البيانات التاريخية
- **🔍 مراقبة الجودة** - التحقق من صحة البيانات وكشف الشذوذ

## 🏗️ البنية المعمارية

```
crypto-market-analyzer/
├── config/                      # ملفات التكوين
│   ├── settings.yaml           # إعدادات النظام
│   └── schema.sql              # قاعدة البيانات
├── okx_stream_hunter/          # الحزمة الرئيسية
│   ├── core/                   # المكونات الأساسية
│   │   ├── rate_limiter.py    # تحديد معدل الطلبات
│   │   └── shutdown.py        # إدارة الإغلاق الآمن
│   ├── modules/                # وحدات التحليل
│   │   ├── candles/           # بناء الشموع
│   │   ├── validation/        # التحقق من البيانات
│   │   ├── health/            # مراقبة الصحة
│   │   ├── whales/            # كشف الحيتان
│   │   ├── volume/            # تحليل الحجم
│   │   └── patterns/          # كشف الأنماط
│   ├── storage/                # طبقة التخزين
│   │   └── neon_writer.py     # كاتب Neon DB
│   ├── backtesting/            # محرك الاختبار الخلفي
│   │   ├── data_loader.py     # تحميل البيانات
│   │   ├── engine.py          # محرك التنفيذ
│   │   └── reporter.py        # تقارير الأداء
│   ├── performance/            # تحسين الأداء
│   │   └── optimizer.py       # محسن النظام
│   ├── config/                 # محمل التكوين
│   │   └── loader.py
│   └── utils/                  # أدوات مساعدة
│       └── logger.py
└── requirements.txt            # المتطلبات
```

## 🚀 البدء السريع

### المتطلبات الأساسية

- Python 3.9+
- PostgreSQL 14+ (يفضل Neon مع TimescaleDB)
- حساب OKX API (اختياري للبيانات الحية)

### التثبيت

```bash
# استنساخ المستودع
git clone https://github.com/YOUR_USERNAME/crypto-market-analyzer.git
cd crypto-market-analyzer

# إنشاء بيئة افتراضية
python3 -m venv venv
source venv/bin/activate  # على Windows: venv\Scripts\activate

# تثبيت المتطلبات
pip install -r requirements.txt
```

### إعداد قاعدة البيانات

```bash
# إنشاء قاعدة البيانات على Neon
# قم بزيارة: https://neon.tech

# تطبيق المخطط
psql $NEON_DATABASE_URL -f config/schema.sql
```

### التكوين

قم بتحرير `config/settings.yaml`:

```yaml
# إعدادات التداول
trading:
  symbol: "BTC-USDT-SWAP"
  timeframes: ["1m", "5m", "15m", "1h", "4h"]

# إعدادات قاعدة البيانات
database:
  url: "${NEON_DATABASE_URL}"
  pool_min_size: 2
  pool_max_size: 10
  batch_size: 100
  batch_timeout: 5
```

أو استخدم متغيرات البيئة:

```bash
export NEON_DATABASE_URL="postgresql://user:pass@host/db"
export OKX_API_KEY="your_api_key"
export OKX_SECRET_KEY="your_secret_key"
export OKX_PASSPHRASE="your_passphrase"
```

## 📊 الميزات الرئيسية

### 1. كشف تحركات الحيتان 🐋

```python
from okx_stream_hunter.modules.whales import WhaleDetector

detector = WhaleDetector(
    large_trade_threshold=100000,  # 100K USD
    whale_trade_threshold=500000   # 500K USD
)

# كشف تلقائي من البيانات الحية
whale_trades = await detector.detect_from_trades(trades)
```

**المؤشرات المتاحة:**
- حجم الصفقة الكبيرة
- اتجاه الحيتان (شراء/بيع)
- ضغط دفتر الأوامر
- تحركات صناع السوق

### 2. تحليل الحجم المتقدم 📈

#### VWAP (Volume Weighted Average Price)
```python
from okx_stream_hunter.modules.volume import VWAPCalculator

vwap = VWAPCalculator()
vwap_value = vwap.calculate(candles)
```

#### CVD (Cumulative Volume Delta)
```python
from okx_stream_hunter.modules.volume import CVDCalculator

cvd = CVDCalculator()
cvd_value = cvd.calculate(trades)
```

#### Volume Profile
```python
from okx_stream_hunter.modules.volume import VolumeProfileCalculator

profile = VolumeProfileCalculator(num_bins=50)
poc, value_area = profile.calculate(candles)
```

### 3. كشف الأنماط 🎯

```python
from okx_stream_hunter.modules.patterns import SupportResistanceDetector

detector = SupportResistanceDetector(
    lookback_period=100,
    min_touches=2,
    tolerance=0.002
)

levels = detector.detect(candles)
```

### 4. Backtesting 🧪

```python
from okx_stream_hunter.backtesting import BacktestEngine, BacktestReporter

# تحميل البيانات
loader = DataLoader(db_url)
data = await loader.load_candles("BTC-USDT-SWAP", "1h", start_date, end_date)

# تشغيل الاختبار
engine = BacktestEngine(initial_capital=10000)
results = await engine.run(data, strategy)

# إنشاء التقرير
reporter = BacktestReporter()
report = reporter.generate_report(results)
```

## 🔧 التكوين المتقدم

### إعدادات كشف الحيتان

```yaml
whales:
  large_trade_threshold: 100000      # حد الصفقة الكبيرة (USD)
  whale_trade_threshold: 500000      # حد صفقة الحوت (USD)
  orderbook_imbalance_threshold: 0.3 # حد عدم التوازن
  mm_pressure_threshold: 50000       # حد ضغط صناع السوق
```

### إعدادات التحقق من البيانات

```yaml
validation:
  price_spike_threshold: 0.05        # 5% حد القفزة السعرية
  volume_spike_threshold: 10.0       # 10x حد قفزة الحجم
  max_gap_seconds: 120               # الحد الأقصى للفجوة الزمنية
```

### إعدادات الأداء

```yaml
performance:
  enable_profiling: true
  profile_interval: 300              # كل 5 دقائق
  memory_threshold_mb: 500           # تحذير عند 500MB
  cpu_threshold_percent: 80          # تحذير عند 80%
```

## 📈 أمثلة الاستخدام

### مثال 1: مراقبة البيانات الحية

```python
import asyncio
from okx_stream_hunter.core import StreamEngine

async def main():
    engine = StreamEngine()
    
    # الاشتراك في البيانات الحية
    await engine.subscribe_trades("BTC-USDT-SWAP")
    await engine.subscribe_orderbook("BTC-USDT-SWAP")
    
    # بدء المعالجة
    await engine.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### مثال 2: تحليل البيانات التاريخية

```python
from okx_stream_hunter.backtesting import DataLoader
from okx_stream_hunter.modules.whales import WhaleDetector

async def analyze_historical():
    loader = DataLoader(db_url)
    
    # تحميل البيانات
    trades = await loader.load_trades(
        symbol="BTC-USDT-SWAP",
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    # تحليل الحيتان
    detector = WhaleDetector()
    whale_trades = detector.detect_from_trades(trades)
    
    print(f"Found {len(whale_trades)} whale trades")
```

## 🧪 الاختبار

```bash
# تشغيل جميع الاختبارات
pytest

# اختبار وحدة معينة
pytest tests/test_whales.py

# مع تغطية الكود
pytest --cov=okx_stream_hunter
```

## 📊 قاعدة البيانات

### الجداول الرئيسية

- **candles_*** - شموع متعددة الأطر الزمنية (1s, 5s, 1m, 3m, 5m, 15m, 1h, 4h, 1d)
- **indicators** - المؤشرات الفنية المحسوبة
- **market_events** - أحداث السوق (تصفيات، معدلات التمويل، إلخ)
- **orderbook_snapshots** - لقطات دفتر الأوامر
- **health_metrics** - مقاييس صحة النظام
- **system_logs** - سجلات النظام
- **data_quality_logs** - سجلات جودة البيانات

### سياسات الاحتفاظ

- شموع 1s: 7 أيام
- شموع 5s: 14 يوم
- شموع 1m: 30 يوم
- شموع 1h: سنة واحدة
- شموع 1d: إلى الأبد

## 🔒 الأمان

- **لا تشارك مفاتيح API** في الكود المصدري
- استخدم متغيرات البيئة أو ملفات `.env`
- قم بتفعيل IP whitelisting على OKX
- استخدم مفاتيح API للقراءة فقط عند الإمكان

## 🤝 المساهمة

المساهمات مرحب بها! يرجى:

1. Fork المستودع
2. إنشاء فرع للميزة (`git checkout -b feature/amazing-feature`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push إلى الفرع (`git push origin feature/amazing-feature`)
5. فتح Pull Request

## 📝 الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

## 🙏 شكر وتقدير

- [OKX](https://www.okx.com) - لتوفير API قوي
- [Neon](https://neon.tech) - لقاعدة بيانات PostgreSQL بدون خادم
- [TimescaleDB](https://www.timescale.com) - لامتدادات السلاسل الزمنية

## 📧 التواصل

لأي أسئلة أو اقتراحات، يرجى فتح issue على GitHub.

---

**تحذير:** هذا النظام للأغراض التعليمية والبحثية. التداول ينطوي على مخاطر. استخدم على مسؤوليتك الخاصة.
