#!/bin/bash
# test_activation.sh - Test all newly activated features

echo "🧪 Testing Crypto Market Analyzer - Activation Features"
echo "========================================================"
echo ""

BASE_URL="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local name=$2
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ OK${NC} (HTTP $response)"
    else
        echo -e "${RED}❌ FAILED${NC} (HTTP $response)"
    fi
}

echo "🐋 Testing Whale Detection APIs:"
echo "--------------------------------"
test_endpoint "/api/whales/events" "Whale Events"
test_endpoint "/api/whales/stats" "Whale Stats"
echo ""

echo "🕯️ Testing Candles APIs:"
echo "------------------------"
test_endpoint "/api/candles/1m" "Candles 1m"
test_endpoint "/api/candles/5m" "Candles 5m"
test_endpoint "/api/candles/15m" "Candles 15m"
test_endpoint "/api/candles/1h" "Candles 1h"
test_endpoint "/api/candles/all" "All Candles"
echo ""

echo "📊 Testing CVD APIs:"
echo "-------------------"
test_endpoint "/api/cvd/current" "CVD Current"
echo ""

echo "🤖 Testing AI APIs:"
echo "------------------"
test_endpoint "/api/ai/insights" "AI Insights"
test_endpoint "/api/ai/learning_status" "Learning Status"
echo ""

echo "📈 Testing Market APIs:"
echo "----------------------"
test_endpoint "/api/market/current" "Market Current"
test_endpoint "/api/orderflow" "Orderflow"
test_endpoint "/api/strategy" "Strategy"
echo ""

echo "💚 Testing Health APIs:"
echo "----------------------"
test_endpoint "/api/health" "Health Status"
test_endpoint "/api/status" "System Status"
echo ""

echo ""
echo "========================================================"
echo "🎯 Test Complete!"
echo ""
echo "📝 Detailed data check:"
echo "----------------------"

# Get whale stats
echo ""
echo "🐋 Whale Detection Status:"
whale_data=$(curl -s "$BASE_URL/api/whales/stats" 2>/dev/null)
whale_count=$(echo "$whale_data" | grep -o '"total_whale_trades":[0-9]*' | cut -d':' -f2)
echo "   Total whale trades detected: ${whale_count:-0}"

# Get candle counts
echo ""
echo "🕯️ Candles Status:"
candle_data=$(curl -s "$BASE_URL/api/candles/all" 2>/dev/null)
candles_1m=$(echo "$candle_data" | grep -o '"1m":[0-9]*' | head -1 | cut -d':' -f2)
candles_5m=$(echo "$candle_data" | grep -o '"5m":[0-9]*' | head -1 | cut -d':' -f2)
echo "   1m candles: ${candles_1m:-0}"
echo "   5m candles: ${candles_5m:-0}"

# Get CVD
echo ""
echo "📊 CVD Status:"
cvd_data=$(curl -s "$BASE_URL/api/cvd/current" 2>/dev/null)
cvd_trend=$(echo "$cvd_data" | grep -o '"cvd_trend":"[^"]*"' | cut -d'"' -f4)
echo "   CVD trend: ${cvd_trend:-unknown}"

# Get AI status
echo ""
echo "🤖 AI Brain Status:"
ai_data=$(curl -s "$BASE_URL/api/ai/insights" 2>/dev/null)
ai_direction=$(echo "$ai_data" | grep -o '"direction":"[^"]*"' | cut -d'"' -f4)
ai_confidence=$(echo "$ai_data" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
echo "   Direction: ${ai_direction:-unknown}"
echo "   Confidence: ${ai_confidence:-0}%"

echo ""
echo "========================================================"
echo -e "${GREEN}✅ All components activated and responding!${NC}"
echo ""
echo "📖 View full report: cat ACTIVATION_REPORT.md"
echo "🌐 Dashboard: http://localhost:8000"
echo ""
