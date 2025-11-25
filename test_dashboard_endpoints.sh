#!/bin/bash

echo "=========================================="
echo "🔥 GOD MODE Dashboard API Endpoint Test"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8000"

echo "1. Testing /api/health"
echo "----------------------------------------"
curl -s "$BASE_URL/api/health" | python3 -m json.tool
echo ""

echo "2. Testing /api/status"
echo "----------------------------------------"
curl -s "$BASE_URL/api/status" | python3 -m json.tool
echo ""

echo "3. Testing /api/ai/insights"
echo "----------------------------------------"
curl -s "$BASE_URL/api/ai/insights" | python3 -m json.tool | head -30
echo ""

echo "4. Testing /api/orderflow"
echo "----------------------------------------"
curl -s "$BASE_URL/api/orderflow" | python3 -m json.tool
echo ""

echo "5. Testing /api/strategy"
echo "----------------------------------------"
curl -s "$BASE_URL/api/strategy" | python3 -m json.tool
echo ""

echo "6. Testing /api/positions"
echo "----------------------------------------"
curl -s "$BASE_URL/api/positions" | python3 -m json.tool
echo ""

echo "7. Testing /api/market/current"
echo "----------------------------------------"
curl -s "$BASE_URL/api/market/current" | python3 -m json.tool
echo ""

echo "8. Testing Dashboard Root /"
echo "----------------------------------------"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/")
echo "HTTP Status Code: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Dashboard HTML is accessible"
else
    echo "❌ Dashboard HTML returned HTTP $HTTP_CODE"
fi
echo ""

echo "9. Testing /static/dashboard.js"
echo "----------------------------------------"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/static/dashboard.js")
echo "HTTP Status Code: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ dashboard.js is accessible"
    echo "File size: $(curl -s "$BASE_URL/static/dashboard.js" | wc -c) bytes"
else
    echo "❌ dashboard.js returned HTTP $HTTP_CODE"
fi
echo ""

echo "=========================================="
echo "✅ All endpoint tests completed!"
echo "=========================================="
