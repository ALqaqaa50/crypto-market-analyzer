# 🚀 النظام المفعَّل بالكامل - دليل الاستخدام

## ✨ الميزات الجديدة المفعَّلة

تم تفعيل **100%** من المكونات الخاملة! النظام الآن يعمل بكامل طاقته.

---

## 🐋 كشف الحيتان (Whale Detection)

### ما هو؟
نظام يكشف الصفقات الكبيرة (الحيتان) في الوقت الفعلي.

### كيف يعمل؟
- يفحص كل صفقة تلقائياً
- يكشف الأوامر الكبيرة (>$100k افتراضياً)
- يحسب القيمة بالدولار والحجم النسبي

### كيف تستخدمه؟

#### 1. عبر Dashboard API:
```bash
# الحصول على آخر أحداث الحيتان
curl http://localhost:8000/api/whales/events

# الحصول على إحصائيات الحيتان
curl http://localhost:8000/api/whales/stats
```

#### 2. في اللوج:
```
🐋 WHALE DETECTED! Side=BUY, Size=125.50, USD=$3,250,000, Magnitude=8.3x
```

#### 3. عبر SystemState:
```python
from okx_stream_hunter.state import get_system_state

state = get_system_state()
print(f"Whale count: {state.whale_count}")
print(f"Last whale: {state.last_whale_event}")
print(f"Recent whales: {state.whale_events}")
```

---

## 📊 CVD - Cumulative Volume Delta

### ما هو؟
مؤشر يقيس الفرق التراكمي بين حجم الشراء والبيع.

### كيف يعمل؟
- يحسب: `CVD = إجمالي الشراء - إجمالي البيع`
- CVD إيجابي → ضغط شراء (bullish)
- CVD سالب → ضغط بيع (bearish)

### كيف تستخدمه؟

#### 1. عبر API:
```bash
curl http://localhost:8000/api/cvd/current
```

الرد:
```json
{
  "cvd_value": 1250.5,
  "cvd_trend": "bullish",
  "buy_volume": 3500,
  "sell_volume": 2250,
  "volume_delta": 1250
}
```

#### 2. عبر SystemState:
```python
state = get_system_state()
print(f"CVD: {state.cvd_value}")
print(f"Trend: {state.cvd_trend}")  # bullish/bearish/neutral
```

---

## 🕯️ بناء الشموع (Candles)

### ما هو؟
نظام يبني شموع OHLCV من الصفقات الفورية.

### الإطارات الزمنية المتاحة:
- **1m** - دقيقة واحدة
- **5m** - 5 دقائق
- **15m** - 15 دقيقة
- **1h** - ساعة واحدة

### كيف يعمل؟
- يبني الشموع تلقائياً من كل صفقة
- يحسب Open, High, Low, Close, Volume
- يخزن آخر 100 شمعة لكل إطار زمني

### كيف تستخدمه؟

#### 1. الحصول على شموع إطار زمني محدد:
```bash
# شموع دقيقة واحدة
curl http://localhost:8000/api/candles/1m

# شموع 5 دقائق
curl http://localhost:8000/api/candles/5m

# شموع ساعة
curl http://localhost:8000/api/candles/1h
```

#### 2. الحصول على جميع الإطارات:
```bash
curl http://localhost:8000/api/candles/all
```

الرد:
```json
{
  "candles": {
    "1m": [
      {
        "symbol": "BTC-USDT-SWAP",
        "timeframe": "1m",
        "open": 42100.5,
        "high": 42150.2,
        "low": 42095.1,
        "close": 42140.8,
        "volume": 125.5,
        "trades": 342
      }
    ],
    "5m": [...],
    "15m": [...],
    "1h": [...]
  },
  "counts": {
    "1m": 100,
    "5m": 50,
    "15m": 25,
    "1h": 12
  }
}
```

#### 3. في اللوج:
```
🕯️ Candle closed: BTC-USDT-SWAP 1m O=42100.50 H=42150.20 L=42095.10 C=42140.80 V=125.50
```

---

## 🤖 AI Brain مع البيانات الجديدة

الآن AI Brain يستقبل:
- ✅ أسعار الصفقات (Trades)
- ✅ بيانات OrderBook
- ✅ أحداث الحيتان 🐋
- ✅ CVD في الوقت الفعلي 📊
- ✅ الشموع المكتملة 🕯️

### استخدام AI:
```bash
# الحصول على إشارة AI
curl http://localhost:8000/api/ai/insights
```

الرد:
```json
{
  "signal": "long",
  "confidence": 0.78,
  "direction": "long",
  "reason": "Strong buying pressure + whale accumulation",
  "regime": "trending",
  "price": 42140.5,
  "cvd_value": 1250.5,
  "cvd_trend": "bullish",
  "whale_count": 5
}
```

---

## 💚 Health Monitor

### ما هو؟
نظام مراقبة صحة النظام بالكامل.

### كيف يعمل؟
- يفحص صحة النظام كل 60 ثانية
- يتتبع: ticks, errors, candles, DB writes
- يفحص صحة قاعدة البيانات

### كيف تستخدمه؟
```bash
curl http://localhost:8000/api/health
```

---

## 📝 استخدام SystemState (للمطورين)

```python
from okx_stream_hunter.state import get_system_state

# الحصول على الحالة العامة
state = get_system_state()

# بيانات الحيتان
print(f"Whale count: {state.whale_count}")
print(f"Whale events: {state.whale_events}")

# بيانات CVD
print(f"CVD value: {state.cvd_value}")
print(f"CVD trend: {state.cvd_trend}")

# بيانات الشموع
print(f"1m candles: {len(state.candles_1m)}")
print(f"5m candles: {len(state.candles_5m)}")
print(f"Last candle closed: {state.last_candle_closed}")

# بيانات AI
print(f"Signal: {state.ai_direction}")
print(f"Confidence: {state.ai_confidence}")
print(f"Regime: {state.ai_regime}")
```

---

## 🧪 اختبار النظام

### 1. اختبار سريع:
```bash
./test_activation.sh
```

### 2. اختبار يدوي:
```bash
# تشغيل النظام
python main.py

# في نافذة أخرى - اختبار APIs
curl http://localhost:8000/api/whales/events
curl http://localhost:8000/api/candles/1m
curl http://localhost:8000/api/cvd/current
curl http://localhost:8000/api/ai/insights
```

### 3. Dashboard:
افتح المتصفح: `http://localhost:8000`

---

## 📊 APIs الجديدة - ملخص سريع

| Endpoint | الوصف | مثال |
|----------|-------|------|
| `/api/whales/events` | أحداث الحيتان | `curl http://localhost:8000/api/whales/events` |
| `/api/whales/stats` | إحصائيات الحيتان | `curl http://localhost:8000/api/whales/stats` |
| `/api/candles/1m` | شموع دقيقة | `curl http://localhost:8000/api/candles/1m` |
| `/api/candles/5m` | شموع 5 دقائق | `curl http://localhost:8000/api/candles/5m` |
| `/api/candles/15m` | شموع 15 دقيقة | `curl http://localhost:8000/api/candles/15m` |
| `/api/candles/1h` | شموع ساعة | `curl http://localhost:8000/api/candles/1h` |
| `/api/candles/all` | جميع الشموع | `curl http://localhost:8000/api/candles/all` |
| `/api/cvd/current` | CVD الحالي | `curl http://localhost:8000/api/cvd/current` |

---

## ⚙️ التخصيص

### تغيير عتبة كشف الحيتان:
```python
# في okx_stream_hunter/modules/whales/detector.py
self.min_usd_value = 100_000  # تغيير هنا (افتراضي $100k)
```

### تغيير نافذة CVD:
```python
# في okx_stream_hunter/core/processor.py
self.cvd_engine = CVDEngine(window_size=1000)  # تغيير 1000 إلى القيمة المرغوبة
```

### تغيير الإطارات الزمنية للشموع:
```python
# في okx_stream_hunter/core/processor.py
self.candle_timeframes = ["1m", "5m", "15m", "1h"]  # إضافة أو إزالة إطارات
```

---

## 🐛 استكشاف الأخطاء

### لا تظهر أحداث حيتان:
1. تأكد من وجود صفقات كبيرة (>$100k)
2. تحقق من اللوج: `grep "WHALE DETECTED" logs/*.log`

### الشموع فارغة:
1. تأكد من استقبال صفقات: `curl http://localhost:8000/api/orderflow`
2. انتظر دقيقة واحدة على الأقل لإغلاق أول شمعة

### CVD يعطي 0:
1. تأكد من استقبال صفقات
2. CVD يبدأ من 0 ويتراكم تدريجياً

---

## 📚 المزيد من المعلومات

- 📖 **تقرير التفعيل الكامل:** `ACTIVATION_REPORT.md`
- 🌐 **Dashboard:** `http://localhost:8000`
- 📝 **اللوج:** `logs/stream.log`
- 🧪 **اختبار:** `./test_activation.sh`

---

## ✅ الخلاصة

**النظام الآن مفعَّل بالكامل ويعمل على:**
- ✅ كشف الحيتان 🐋
- ✅ حساب CVD 📊
- ✅ بناء الشموع 🕯️
- ✅ مراقبة الصحة 💚
- ✅ توليد إشارات AI 🤖
- ✅ Dashboard APIs 🌐

**استمتع بالنظام الكامل! 🚀**
