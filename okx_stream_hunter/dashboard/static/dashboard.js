/**
 * 🔥 GOD MODE Ultra Dashboard v3.0
 * Real-time Trading Dashboard JavaScript
 * 
 * This file handles:
 * - Real-time data polling from FastAPI endpoints
 * - Chart rendering (candlesticks with TradingView Lightweight Charts)
 * - UI updates for all panels
 * - Error handling and graceful degradation
 */

// ============================================================
// Configuration
// ============================================================

const CONFIG = {
    // Base URL - auto-detect or use window.location.origin for Codespaces compatibility
    BASE_URL: window.location.origin,
    
    // Polling intervals (in milliseconds)
    INSIGHTS_POLL_INTERVAL: 500,       // Poll AI insights every 500ms (2Hz)
    ORDERFLOW_POLL_INTERVAL: 500,      // Poll orderflow every 500ms
    STRATEGY_POLL_INTERVAL: 2000,      // Poll strategy every 2 seconds
    STATUS_POLL_INTERVAL: 5000,        // Poll system status every 5 seconds
    
    // API endpoints
    ENDPOINTS: {
        AI_INSIGHTS: '/api/ai/insights',
        ORDERFLOW: '/api/orderflow',
        STRATEGY: '/api/strategy',
        STATUS: '/api/status',
        HEALTH: '/api/health',
        MARKET: '/api/market/current',
    },
    
    // Chart settings
    CHART: {
        MAX_CANDLES: 100,               // Maximum candles to display
        CANDLE_INTERVAL: '1m',          // Candle interval (not used yet, for future)
    },
    
    // UI settings
    MAX_SIGNALS_HISTORY: 20,            // Max signals to show in table
    MAX_TRADES_HISTORY: 10,             // Max trades to show in recent trades
};

// ============================================================
// Global State
// ============================================================

const state = {
    chart: null,                        // Lightweight chart instance
    candleSeries: null,                 // Candlestick series
    priceLine: null,                    // Current price line
    entryLine: null,                    // Entry price line
    tpLine: null,                       // Take profit line
    slLine: null,                       // Stop loss line
    candles: [],                        // Candle data array
    signals: [],                        // Signal history
    lastInsight: null,                  // Last AI insight
    lastStrategy: null,                 // Last strategy data
    lastStatus: null,                   // Last system status
    errors: {                           // Error tracking
        insights: 0,
        strategy: 0,
        status: 0,
    },
};

// ============================================================
// Utility Functions
// ============================================================

/**
 * Fetch data from API endpoint with error handling
 */
async function fetchAPI(endpoint) {
    const url = `${CONFIG.BASE_URL}${endpoint}`;
    try {
        console.log(`[FETCH] ${url}`);
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
            mode: 'cors',
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`[FETCH ERROR] ${url} - Status: ${response.status}, Body: ${errorText}`);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        console.log(`[FETCH SUCCESS] ${url}`, data);
        return { success: true, data };
    } catch (error) {
        console.error(`[FETCH FAILED] ${url}:`, error);
        return { success: false, error: error.message };
    }
}

/**
 * Format price with appropriate decimals
 */
function formatPrice(price) {
    if (price == null) return '--';
    return parseFloat(price).toFixed(2);
}

/**
 * Format percentage with sign
 */
function formatPercent(value) {
    if (value == null) return '--';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

/**
 * Format number with K/M suffixes
 */
function formatNumber(num) {
    if (num == null) return '--';
    if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(2)}K`;
    return num.toFixed(2);
}

/**
 * Format timestamp to readable time
 */
function formatTime(timestamp) {
    if (!timestamp) return '--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
}

/**
 * Get color for confidence level
 */
function getConfidenceColor(confidence) {
    if (confidence >= 70) return '#00ff88';
    if (confidence >= 40) return '#ffaa00';
    return '#ff4444';
}

/**
 * Truncate text with ellipsis
 */
function truncate(text, maxLength = 50) {
    if (!text) return '--';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// ============================================================
// Chart Functions
// ============================================================

/**
 * Initialize the lightweight chart
 */
function initChart() {
    const chartContainer = document.getElementById('candles-chart');
    if (!chartContainer) return;
    
    // Clear loading message
    chartContainer.innerHTML = '';
    
    // Create chart
    state.chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: 350,
        layout: {
            background: { color: 'transparent' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
        },
        timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            timeVisible: true,
            secondsVisible: false,
        },
    });
    
    // Add candlestick series
    state.candleSeries = state.chart.addCandlestickSeries({
        upColor: '#00ff88',
        downColor: '#ff4444',
        borderUpColor: '#00ff88',
        borderDownColor: '#ff4444',
        wickUpColor: '#00ff88',
        wickDownColor: '#ff4444',
    });
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (state.chart && chartContainer) {
            state.chart.applyOptions({ 
                width: chartContainer.clientWidth 
            });
        }
    });
    
    console.log('📊 Chart initialized');
}

/**
 * Update chart with new candle data
 */
function updateChart(price, timestamp) {
    if (!state.candleSeries || !price) return;
    
    const time = Math.floor(new Date(timestamp).getTime() / 1000);
    
    // Check if we need to create a new candle or update existing
    if (state.candles.length === 0) {
        // First candle
        const candle = {
            time,
            open: price,
            high: price,
            low: price,
            close: price,
        };
        state.candles.push(candle);
        state.candleSeries.setData([candle]);
    } else {
        const lastCandle = state.candles[state.candles.length - 1];
        
        // Check if we should create a new candle (1 minute interval)
        if (time - lastCandle.time >= 60) {
            // New candle
            const candle = {
                time,
                open: price,
                high: price,
                low: price,
                close: price,
            };
            state.candles.push(candle);
            
            // Keep only last N candles
            if (state.candles.length > CONFIG.CHART.MAX_CANDLES) {
                state.candles.shift();
            }
            
            state.candleSeries.setData(state.candles);
        } else {
            // Update last candle
            lastCandle.close = price;
            lastCandle.high = Math.max(lastCandle.high, price);
            lastCandle.low = Math.min(lastCandle.low, price);
            
            state.candleSeries.update(lastCandle);
        }
    }
}

/**
 * Draw price lines on chart (Entry, TP, SL)
 */
function updatePriceLines(entry, tp, sl) {
    if (!state.candleSeries) return;
    
    // Remove old lines
    if (state.entryLine) {
        state.candleSeries.removePriceLine(state.entryLine);
    }
    if (state.tpLine) {
        state.candleSeries.removePriceLine(state.tpLine);
    }
    if (state.slLine) {
        state.candleSeries.removePriceLine(state.slLine);
    }
    
    // Draw entry line
    if (entry) {
        state.entryLine = state.candleSeries.createPriceLine({
            price: entry,
            color: '#007bff',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: 'Entry',
        });
    }
    
    // Draw TP line
    if (tp) {
        state.tpLine = state.candleSeries.createPriceLine({
            price: tp,
            color: '#00ff88',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: 'TP',
        });
    }
    
    // Draw SL line
    if (sl) {
        state.slLine = state.candleSeries.createPriceLine({
            price: sl,
            color: '#ff4444',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: 'SL',
        });
    }
    
    // Update text below chart
    document.getElementById('chart-entry').textContent = entry ? `$${formatPrice(entry)}` : '--';
    document.getElementById('chart-tp').textContent = tp ? `$${formatPrice(tp)}` : '--';
    document.getElementById('chart-sl').textContent = sl ? `$${formatPrice(sl)}` : '--';
}

// ============================================================
// UI Update Functions
// ============================================================

/**
 * Update top bar with market overview
 */
function updateTopBar(insight) {
    if (!insight) {
        console.warn('[updateTopBar] No insight data provided');
        return;
    }
    console.log('[updateTopBar] Updating with:', insight);
    
    // Update price
    const price = insight.price;
    if (price) {
        document.getElementById('current-price').textContent = `$${formatPrice(price)}`;
    }
    
    // Update regime tag
    const regimeTag = document.getElementById('regime-tag');
    regimeTag.textContent = insight.regime || 'Unknown';
    regimeTag.className = 'tag';
    if (insight.regime === 'trend') {
        regimeTag.classList.add('tag-trend');
    } else if (insight.regime === 'range') {
        regimeTag.classList.add('tag-range');
    }
    
    // Update volatility tag (derive from scores if available)
    const volatilityTag = document.getElementById('volatility-tag');
    const volatility = insight.scores?.volatility || 0.5;
    if (volatility < 0.3) {
        volatilityTag.textContent = 'Low Volatility';
        volatilityTag.className = 'tag tag-volatility-low';
    } else if (volatility < 0.7) {
        volatilityTag.textContent = 'Normal Volatility';
        volatilityTag.className = 'tag tag-volatility-normal';
    } else {
        volatilityTag.textContent = 'High Volatility';
        volatilityTag.className = 'tag tag-volatility-high';
    }
    
    // Update spoof risk tag
    const spoofTag = document.getElementById('spoof-tag');
    const spoofRisk = insight.scores?.spoof_risk || 0;
    if (spoofRisk < 0.3) {
        spoofTag.textContent = 'Clean Orders';
        spoofTag.className = 'tag tag-spoof-clean';
    } else if (spoofRisk < 0.7) {
        spoofTag.textContent = 'Medium Spoof';
        spoofTag.className = 'tag tag-spoof-medium';
    } else {
        spoofTag.textContent = 'High Spoof Risk';
        spoofTag.className = 'tag tag-spoof-high';
    }
    
    // Update last update time
    document.getElementById('last-update').textContent = formatTime(insight.generated_at);
    
    // Update spread (TODO: get from API if available)
    document.getElementById('spread').textContent = '0.5 bps';
    
    // Update 24h volume (placeholder)
    document.getElementById('volume-24h').textContent = formatNumber(insight.buy_volume + insight.sell_volume);
}

/**
 * Update AI Signal Panel (Panel B)
 */
function updateAISignalPanel(insight) {
    if (!insight) {
        console.warn('[updateAISignalPanel] No insight data provided');
        return;
    }
    console.log('[updateAISignalPanel] Updating with:', insight);
    
    // Update signal badge
    const signalBadge = document.getElementById('signal-badge');
    signalBadge.textContent = insight.signal || 'WAIT';
    signalBadge.className = 'signal-badge';
    if (insight.signal === 'BUY') {
        signalBadge.classList.add('signal-buy');
    } else if (insight.signal === 'SELL') {
        signalBadge.classList.add('signal-sell');
    } else {
        signalBadge.classList.add('signal-flat');
    }
    
    // Update direction
    const directionEl = document.getElementById('signal-direction');
    directionEl.textContent = (insight.direction || 'flat').toUpperCase();
    directionEl.className = '';
    if (insight.direction === 'long') {
        directionEl.classList.add('direction-long');
    } else if (insight.direction === 'short') {
        directionEl.classList.add('direction-short');
    } else {
        directionEl.classList.add('direction-flat');
    }
    
    // Update confidence gauge
    const confidence = insight.confidence || 0;
    const confidenceCircle = document.getElementById('confidence-circle');
    const confidenceText = document.getElementById('confidence-text');
    
    const circumference = 2 * Math.PI * 65;
    const offset = circumference - (confidence / 100) * circumference;
    
    confidenceCircle.style.strokeDashoffset = offset;
    confidenceCircle.style.stroke = getConfidenceColor(confidence);
    confidenceText.textContent = `${confidence}%`;
    confidenceText.style.color = getConfidenceColor(confidence);
    
    // Update regime
    document.getElementById('signal-regime').textContent = insight.regime || '--';
    
    // Update reason (with tooltip for full text)
    const reasonEl = document.getElementById('signal-reason');
    reasonEl.textContent = truncate(insight.reason, 40);
    reasonEl.setAttribute('data-tooltip', insight.reason || 'No reason available');
    
    // Update score badges
    const scoreContainer = document.getElementById('score-badges');
    scoreContainer.innerHTML = '';
    
    if (insight.scores) {
        for (const [key, value] of Object.entries(insight.scores)) {
            const badge = document.createElement('span');
            badge.className = 'score-badge';
            
            const scorePercent = (value * 100).toFixed(0);
            
            if (value >= 0.7) badge.classList.add('score-high');
            else if (value >= 0.4) badge.classList.add('score-medium');
            else badge.classList.add('score-low');
            
            badge.innerHTML = `
                <span>${key.replace(/_/g, ' ')}</span>
                <span class="font-bold">${scorePercent}%</span>
            `;
            
            scoreContainer.appendChild(badge);
        }
    }
}

/**
 * Update Orderflow & Microstructure Panel (Panel C)
 */
function updateOrderflowPanel(insight) {
    if (!insight) {
        console.warn('[updateOrderflowPanel] No insight data provided');
        return;
    }
    console.log('[updateOrderflowPanel] Updating with:', insight);
    
    const buyVol = insight.buy_volume || 0;
    const sellVol = insight.sell_volume || 0;
    const totalVol = buyVol + sellVol;
    
    // Update volume bars
    document.getElementById('buy-volume-text').textContent = formatNumber(buyVol);
    document.getElementById('sell-volume-text').textContent = formatNumber(sellVol);
    
    if (totalVol > 0) {
        const buyPercent = (buyVol / totalVol) * 100;
        const sellPercent = (sellVol / totalVol) * 100;
        
        document.getElementById('buy-bar').style.width = `${buyPercent}%`;
        document.getElementById('sell-bar').style.width = `${sellPercent}%`;
    }
    
    // Update orderbook imbalance meter (from scores if available)
    const imbalance = insight.scores?.orderbook_imbalance || 0.5;
    const imbalancePercent = (imbalance * 100).toFixed(0);
    
    document.getElementById('imbalance-value').textContent = `${imbalancePercent}%`;
    document.getElementById('imbalance-meter').style.width = `${imbalancePercent}%`;
    
    // Update spoofing risk meter
    const spoofRisk = insight.scores?.spoof_risk || 0;
    const spoofPercent = (spoofRisk * 100).toFixed(0);
    
    document.getElementById('spoof-value').textContent = `${spoofPercent}%`;
    document.getElementById('spoof-meter').style.width = `${spoofPercent}%`;
    
    // Update additional metrics
    document.getElementById('micro-spread').textContent = '0.5 bps'; // TODO: get from API
    document.getElementById('depth-imbalance').textContent = imbalancePercent + '%';
    document.getElementById('whale-trades').textContent = insight.whale_trades || 0;
}

/**
 * Update Position & Risk Panel (Panel D)
 */
function updatePositionPanel(insight, strategy) {
    if (!insight) {
        console.warn('[updatePositionPanel] No insight data provided');
        return;
    }
    console.log('[updatePositionPanel] Updating with insight:', insight, 'strategy:', strategy);
    
    const position = insight.position || {};
    
    // Update current position
    const direction = position.direction || 'flat';
    const directionEl = document.getElementById('pos-direction');
    directionEl.textContent = direction.toUpperCase();
    directionEl.className = 'metric-value';
    if (direction === 'long') directionEl.style.color = '#00ff88';
    else if (direction === 'short') directionEl.style.color = '#ff4444';
    else directionEl.style.color = '#ffaa00';
    
    document.getElementById('pos-size').textContent = position.size || 0;
    document.getElementById('pos-entry').textContent = position.entry_price 
        ? `$${formatPrice(position.entry_price)}` 
        : '$--';
    
    // Calculate unrealized PnL (placeholder)
    const pnl = 0; // TODO: calculate from current price vs entry
    const pnlEl = document.getElementById('pos-pnl');
    pnlEl.textContent = `$${pnl.toFixed(2)}`;
    pnlEl.style.color = pnl >= 0 ? '#00ff88' : '#ff4444';
    
    // Update risk metrics
    document.getElementById('risk-per-trade').textContent = '2.0%'; // TODO: get from risk manager
    
    // Calculate R:R ratio
    if (strategy && strategy.entry_price && strategy.tp && strategy.sl) {
        const reward = Math.abs(strategy.tp - strategy.entry_price);
        const risk = Math.abs(strategy.entry_price - strategy.sl);
        const rrRatio = risk > 0 ? (reward / risk).toFixed(2) : '--';
        document.getElementById('rr-ratio').textContent = `${rrRatio}:1`;
    } else {
        document.getElementById('rr-ratio').textContent = '--';
    }
    
    // TODO: Update recent trades (will need trades history endpoint)
    // For now, show placeholder
    const tradesContainer = document.getElementById('recent-trades');
    if (tradesContainer.children.length === 1) {
        // Keep placeholder for now
    }
}

/**
 * Update signals history table
 */
function updateSignalsTable(insight) {
    if (!insight) {
        console.warn('[updateSignalsTable] No insight data provided');
        return;
    }
    console.log('[updateSignalsTable] Adding signal:', insight);
    
    // Add to signals history
    state.signals.unshift({
        time: insight.generated_at,
        signal: insight.signal,
        direction: insight.direction,
        confidence: insight.confidence,
        price: insight.price,
        regime: insight.regime,
        reason: insight.reason,
    });
    
    // Keep only last N signals
    if (state.signals.length > CONFIG.MAX_SIGNALS_HISTORY) {
        state.signals.pop();
    }
    
    // Update table
    const tbody = document.getElementById('signals-tbody');
    tbody.innerHTML = '';
    
    if (state.signals.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-gray-500 py-4">
                    Waiting for signals...
                </td>
            </tr>
        `;
        return;
    }
    
    state.signals.forEach(signal => {
        const row = document.createElement('tr');
        
        // Signal color
        let signalClass = '';
        if (signal.signal === 'BUY') signalClass = 'text-green-400 font-bold';
        else if (signal.signal === 'SELL') signalClass = 'text-red-400 font-bold';
        else signalClass = 'text-yellow-400';
        
        // Direction color
        let directionClass = '';
        if (signal.direction === 'long') directionClass = 'text-green-400';
        else if (signal.direction === 'short') directionClass = 'text-red-400';
        else directionClass = 'text-yellow-400';
        
        // Confidence color
        const confColor = getConfidenceColor(signal.confidence);
        
        row.innerHTML = `
            <td>${formatTime(signal.time)}</td>
            <td class="${signalClass}">${signal.signal}</td>
            <td class="${directionClass}">${signal.direction.toUpperCase()}</td>
            <td style="color: ${confColor}; font-weight: 600;">${signal.confidence}%</td>
            <td>$${formatPrice(signal.price)}</td>
            <td>${signal.regime}</td>
            <td class="text-sm">${truncate(signal.reason, 30)}</td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Update system status panel
 */
function updateSystemStatus(status) {
    if (!status) {
        console.warn('[updateSystemStatus] No status data received');
        return;
    }
    
    console.log('[updateSystemStatus] Updating with:', status);
    
    // Update AI status
    const aiStatus = document.getElementById('status-ai');
    if (aiStatus) {
        if (status.ai_enabled) {
            aiStatus.className = 'status-badge status-on';
            aiStatus.innerHTML = '<span>●</span> ON';
        } else {
            aiStatus.className = 'status-badge status-off';
            aiStatus.innerHTML = '<span>●</span> OFF';
        }
    }
    
    // Update trading status (handle both field names)
    const tradingStatus = document.getElementById('status-trading');
    if (tradingStatus) {
        const autoTrading = status.auto_trading_enabled || status.auto_trading || false;
        if (autoTrading) {
            tradingStatus.className = 'status-badge status-on';
            tradingStatus.innerHTML = '<span>●</span> ON';
        } else {
            tradingStatus.className = 'status-badge status-off';
            tradingStatus.innerHTML = '<span>●</span> OFF';
        }
    }
    
    // Update heartbeat time (use heartbeat field if available, fallback to timestamp)
    const heartbeatEl = document.getElementById('heartbeat-time');
    if (heartbeatEl) {
        const heartbeatTime = status.heartbeat || status.last_heartbeat || status.timestamp;
        heartbeatEl.textContent = formatTime(heartbeatTime);
    }
    
    // Update mode tag in top bar
    const modeTag = document.getElementById('mode-tag');
    if (modeTag) {
        const autoTrading = status.auto_trading_enabled || status.auto_trading || false;
        if (autoTrading) {
            modeTag.textContent = 'Live Trading';
            modeTag.className = 'tag tag-live';
        } else {
            modeTag.textContent = 'Paper Trading';
            modeTag.className = 'tag tag-paper';
        }
    }
}

// ============================================================
// Data Polling Functions
// ============================================================

/**
 * Poll AI insights endpoint
 */
async function pollInsights() {
    console.log('[pollInsights] Starting...');
    const result = await fetchAPI(CONFIG.ENDPOINTS.AI_INSIGHTS);
    
    if (result.success) {
        console.log('[pollInsights] SUCCESS - Updating UI with data:', result.data);
        state.lastInsight = result.data;
        state.errors.insights = 0;
        
        // Remove error badge if it exists
        const errorBadge = document.getElementById('error-insights');
        if (errorBadge) errorBadge.remove();
        
        // Update UI
        try {
            updateTopBar(result.data);
            updateAISignalPanel(result.data);
            updateOrderflowPanel(result.data);
            updatePositionPanel(result.data, state.lastStrategy);
            updateSignalsTable(result.data);
            
            // Update chart
            if (result.data.price) {
                updateChart(result.data.price, result.data.generated_at);
            }
            console.log('[pollInsights] UI updated successfully');
        } catch (error) {
            console.error('[pollInsights] Error updating UI:', error);
        }
    } else {
        state.errors.insights++;
        console.error(`[pollInsights] FAILED (${state.errors.insights} consecutive errors):`, result.error);
        
        if (state.errors.insights > 3) {
            // Show error badge after 3 consecutive failures
            const topBar = document.querySelector('.top-bar');
            if (topBar && !document.getElementById('error-insights')) {
                const errorBadge = document.createElement('span');
                errorBadge.id = 'error-insights';
                errorBadge.className = 'error-badge';
                errorBadge.textContent = 'API Error: Insights';
                topBar.appendChild(errorBadge);
                console.error('[pollInsights] Error badge displayed');
            }
        }
    }
}

/**
 * Poll strategy endpoint
 */
async function pollStrategy() {
    console.log('[pollStrategy] Starting...');
    const result = await fetchAPI(CONFIG.ENDPOINTS.STRATEGY);
    
    if (result.success) {
        console.log('[pollStrategy] SUCCESS:', result.data);
        state.lastStrategy = result.data;
        state.errors.strategy = 0;
        
        // Update price lines on chart
        try {
            updatePriceLines(
                result.data.entry_price,
                result.data.tp,
                result.data.sl
            );
            
            // Update position panel with R:R ratio
            updatePositionPanel(state.lastInsight, result.data);
            console.log('[pollStrategy] UI updated');
        } catch (error) {
            console.error('[pollStrategy] Error updating UI:', error);
        }
    } else {
        state.errors.strategy++;
        console.error(`[pollStrategy] FAILED (${state.errors.strategy} errors):`, result.error);
    }
}

/**
 * Poll system status endpoint
 */
async function pollStatus() {
    console.log('[pollStatus] Starting...');
    const result = await fetchAPI(CONFIG.ENDPOINTS.STATUS);
    
    if (result.success) {
        console.log('[pollStatus] SUCCESS:', result.data);
        state.lastStatus = result.data;
        state.errors.status = 0;
        
        // Update system status panel
        try {
            updateSystemStatus(result.data);
            console.log('[pollStatus] UI updated');
        } catch (error) {
            console.error('[pollStatus] Error updating UI:', error);
        }
    } else {
        state.errors.status++;
        console.error(`[pollStatus] FAILED (${state.errors.status} errors):`, result.error);
    }
}

/**
 * Poll orderflow data (CVD, volume imbalance, bid/ask pressure)
 */
async function pollOrderflow() {
    console.log('[pollOrderflow] Starting...');
    const result = await fetchAPI(CONFIG.ENDPOINTS.ORDERFLOW);
    
    if (result.success) {
        console.log('[pollOrderflow] SUCCESS:', result.data);
        state.lastOrderflow = result.data;
        state.errors.orderflow = 0;
        
        // Update orderflow indicators
        try {
            updateOrderflowIndicators(result.data);
            console.log('[pollOrderflow] UI updated');
        } catch (error) {
            console.error('[pollOrderflow] Error updating UI:', error);
        }
    } else {
        state.errors.orderflow = state.errors.orderflow || 0;
        state.errors.orderflow++;
        console.error(`[pollOrderflow] FAILED (${state.errors.orderflow} errors):`, result.error);
    }
}

/**
 * Update orderflow indicators in UI
 */
function updateOrderflowIndicators(data) {
    // Update CVD display if element exists
    const cvdEl = document.getElementById('cvd-value');
    if (cvdEl && data.cvd !== undefined) {
        cvdEl.textContent = data.cvd.toFixed(2);
        cvdEl.className = data.cvd > 0 ? 'positive' : data.cvd < 0 ? 'negative' : 'neutral';
    }
    
    // Update volume imbalance
    const imbalanceEl = document.getElementById('volume-imbalance');
    if (imbalanceEl && data.volume_imbalance !== undefined) {
        const pct = (data.volume_imbalance * 100).toFixed(1);
        imbalanceEl.textContent = `${pct}%`;
        imbalanceEl.className = data.volume_imbalance > 0.6 ? 'bullish' : data.volume_imbalance < 0.4 ? 'bearish' : 'neutral';
    }
    
    // Update orderbook imbalance
    const obImbalanceEl = document.getElementById('orderbook-imbalance');
    if (obImbalanceEl && data.orderbook_imbalance !== undefined) {
        const pct = (data.orderbook_imbalance * 100).toFixed(1);
        obImbalanceEl.textContent = `${pct}%`;
        obImbalanceEl.className = data.orderbook_imbalance > 0.6 ? 'bullish' : data.orderbook_imbalance < 0.4 ? 'bearish' : 'neutral';
    }
    
    // Update spoof risk
    const spoofEl = document.getElementById('spoof-risk');
    if (spoofEl && data.spoof_risk !== undefined) {
        const risk = (data.spoof_risk * 100).toFixed(1);
        spoofEl.textContent = `${risk}%`;
        spoofEl.className = data.spoof_risk > 0.7 ? 'high-risk' : data.spoof_risk > 0.4 ? 'medium-risk' : 'low-risk';
    }
}

// ============================================================
// Initialization
// ============================================================

/**
 * Initialize dashboard
 */
function initDashboard() {
    console.log('🔥 GOD MODE Ultra Dashboard v3.0 initializing...');
    console.log(`📡 Base URL: ${CONFIG.BASE_URL}`);
    console.log(`🌐 Window Location: ${window.location.href}`);
    
    // Initialize chart
    try {
        initChart();
        console.log('✅ Chart initialized');
    } catch (error) {
        console.error('❌ Chart initialization failed:', error);
    }
    
    // Start immediate polling (fire once before intervals)
    console.log('🚀 Starting initial data fetch...');
    pollInsights().then(() => console.log('✅ Initial insights loaded'));
    pollOrderflow().then(() => console.log('✅ Initial orderflow loaded'));
    pollStrategy().then(() => console.log('✅ Initial strategy loaded'));
    pollStatus().then(() => console.log('✅ Initial status loaded'));
    
    // Set up intervals
    console.log('⏰ Setting up polling intervals...');
    const intervals = {
        insights: setInterval(pollInsights, CONFIG.INSIGHTS_POLL_INTERVAL),
        orderflow: setInterval(pollOrderflow, CONFIG.ORDERFLOW_POLL_INTERVAL),
        strategy: setInterval(pollStrategy, CONFIG.STRATEGY_POLL_INTERVAL),
        status: setInterval(pollStatus, CONFIG.STATUS_POLL_INTERVAL),
    };
    
    console.log('✅ Dashboard initialized');
    console.log(`📊 Polling intervals: Insights=${CONFIG.INSIGHTS_POLL_INTERVAL}ms, Orderflow=${CONFIG.ORDERFLOW_POLL_INTERVAL}ms, Strategy=${CONFIG.STRATEGY_POLL_INTERVAL}ms, Status=${CONFIG.STATUS_POLL_INTERVAL}ms`);
    console.log('🔍 Open browser DevTools Console to see live data flow');
    
    // Store intervals globally for debugging
    window.dashboardIntervals = intervals;
    
    // PHASE 3: Initialize advanced monitoring
    initPhase3Monitoring();
}

// ============================================================
// PHASE 3: Advanced Monitoring
// ============================================================

let phase3Charts = {
    confidenceChart: null,
    rlRewardsChart: null,
    orderflowDominanceChart: null,
    pnlChart: null
};

function initPhase3Monitoring() {
    console.log('🚀 Initializing PHASE 3 Advanced Monitoring...');
    
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('⚠️ Chart.js not loaded, loading dynamically...');
        loadChartJS(() => {
            setupPhase3Charts();
            startPhase3Polling();
        });
    } else {
        setupPhase3Charts();
        startPhase3Polling();
    }
}

function loadChartJS(callback) {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
    script.onload = callback;
    document.head.appendChild(script);
}

function setupPhase3Charts() {
    // Confidence Chart
    const confidenceCtx = document.getElementById('confidenceChart');
    if (confidenceCtx) {
        phase3Charts.confidenceChart = new Chart(confidenceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'AI Confidence',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Confidence %' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    // RL Rewards Chart
    const rlCtx = document.getElementById('rlRewardsChart');
    if (rlCtx) {
        phase3Charts.rlRewardsChart = new Chart(rlCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative Reward',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: 'Cumulative Reward' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    // Orderflow Dominance Chart
    const orderflowCtx = document.getElementById('orderflowDominanceChart');
    if (orderflowCtx) {
        phase3Charts.orderflowDominanceChart = new Chart(orderflowCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Buy Pressure',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    },
                    {
                        label: 'Sell Pressure',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Pressure %' }
                    }
                },
                plugins: {
                    legend: { display: true, position: 'top' }
                }
            }
        });
    }
    
    // Live PnL Chart
    const pnlCtx = document.getElementById('livePnLChart');
    if (pnlCtx) {
        phase3Charts.pnlChart = new Chart(pnlCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Live PnL ($)',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: 'PnL ($)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    console.log('📊 PHASE 3 charts initialized');
}

function startPhase3Polling() {
    // Live trades polling
    setInterval(updateLiveTrades, 1000);
    
    // Confidence history polling
    setInterval(updateConfidenceHistory, 2000);
    
    // RL rewards polling
    setInterval(updateRLRewards, 5000);
    
    // Orderflow dominance polling
    setInterval(updateOrderflowDominance, 1000);
    
    // Safety status polling
    setInterval(updateSafetyStatus, 3000);
    
    // Performance metrics polling
    setInterval(updatePerformanceMetrics, 5000);
    
    // PHASE 4: Learning status polling
    setInterval(updateLearningStatus, 10000);
    updateLearningStatus(); // Initial call
    
    console.log('🔄 PHASE 3 polling started');
}

async function updateLiveTrades() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/live_trades`);
        const data = await response.json();
        
        const tradesContainer = document.getElementById('liveTradesContainer');
        if (!tradesContainer) return;
        
        if (!data.active_trades || data.active_trades.length === 0) {
            tradesContainer.innerHTML = '<div class="text-muted">No active trades</div>';
            return;
        }
        
        let html = '<table class="table table-sm"><thead><tr><th>ID</th><th>Direction</th><th>Entry</th><th>Current PnL</th><th>Confidence</th></tr></thead><tbody>';
        
        data.active_trades.forEach(trade => {
            const pnlClass = trade.current_pnl >= 0 ? 'text-success' : 'text-danger';
            html += `
                <tr>
                    <td>${trade.trade_id.substring(0, 8)}</td>
                    <td><span class="badge bg-${trade.direction === 'BUY' ? 'success' : 'danger'}">${trade.direction}</span></td>
                    <td>$${trade.entry_price.toFixed(2)}</td>
                    <td class="${pnlClass}">$${trade.current_pnl.toFixed(2)}</td>
                    <td>${(trade.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        tradesContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating live trades:', error);
    }
}

async function updateConfidenceHistory() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/confidence_history`);
        const data = await response.json();
        
        if (!data.history || !phase3Charts.confidenceChart) return;
        
        const labels = data.history.map(h => new Date(h.timestamp).toLocaleTimeString());
        const values = data.history.map(h => h.confidence * 100);
        
        phase3Charts.confidenceChart.data.labels = labels.slice(-50);
        phase3Charts.confidenceChart.data.datasets[0].data = values.slice(-50);
        phase3Charts.confidenceChart.update('none');
        
    } catch (error) {
        console.error('Error updating confidence history:', error);
    }
}

async function updateRLRewards() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/rl_rewards`);
        const data = await response.json();
        
        if (!data.rewards || !phase3Charts.rlRewardsChart) return;
        
        const labels = data.rewards.map(r => `Ep ${r.episode}`);
        const values = data.rewards.map(r => r.cumulative);
        
        phase3Charts.rlRewardsChart.data.labels = labels.slice(-50);
        phase3Charts.rlRewardsChart.data.datasets[0].data = values.slice(-50);
        phase3Charts.rlRewardsChart.update('none');
        
    } catch (error) {
        console.error('Error updating RL rewards:', error);
    }
}

async function updateOrderflowDominance() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/orderflow_dominance`);
        const data = await response.json();
        
        if (!data.dominance || !phase3Charts.orderflowDominanceChart) return;
        
        const labels = data.dominance.map(d => new Date(d.timestamp).toLocaleTimeString());
        const buyPressure = data.dominance.map(d => d.buy_pressure);
        const sellPressure = data.dominance.map(d => d.sell_pressure);
        
        phase3Charts.orderflowDominanceChart.data.labels = labels.slice(-20);
        phase3Charts.orderflowDominanceChart.data.datasets[0].data = buyPressure.slice(-20);
        phase3Charts.orderflowDominanceChart.data.datasets[1].data = sellPressure.slice(-20);
        phase3Charts.orderflowDominanceChart.update('none');
        
    } catch (error) {
        console.error('Error updating orderflow dominance:', error);
    }
}

async function updateSafetyStatus() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/safety_status`);
        const data = await response.json();
        
        const safetyContainer = document.getElementById('safetyStatusContainer');
        if (!safetyContainer || !data.available) return;
        
        const aiSafety = data.ai_safety || {};
        const circuitBreaker = data.circuit_breaker || {};
        
        let html = `
            <div class="row">
                <div class="col-md-6">
                    <h6>AI Safety</h6>
                    <div>Health: <span class="badge bg-${aiSafety.health_score > 0.8 ? 'success' : aiSafety.health_score > 0.5 ? 'warning' : 'danger'}">${(aiSafety.health_score * 100).toFixed(0)}%</span></div>
                    <div>Emergency Stop: <span class="badge bg-${aiSafety.emergency_stop ? 'danger' : 'success'}">${aiSafety.emergency_stop ? 'ACTIVE' : 'OK'}</span></div>
                </div>
                <div class="col-md-6">
                    <h6>Circuit Breaker</h6>
                    <div>Status: <span class="badge bg-${circuitBreaker.is_open ? 'danger' : 'success'}">${circuitBreaker.is_open ? 'OPEN' : 'CLOSED'}</span></div>
                    <div>Daily PnL: $${(circuitBreaker.daily_pnl || 0).toFixed(2)}</div>
                </div>
            </div>
        `;
        
        safetyContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating safety status:', error);
    }
}

async function updatePerformanceMetrics() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/trading/performance_metrics`);
        const data = await response.json();
        
        const metricsContainer = document.getElementById('performanceMetricsContainer');
        if (!metricsContainer || !data.metrics) return;
        
        const m = data.metrics;
        
        let html = `
            <div class="row">
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value ${m.win_rate >= 50 ? 'text-success' : 'text-danger'}">${m.win_rate}%</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-label">Total PnL</div>
                        <div class="metric-value ${m.total_pnl >= 0 ? 'text-success' : 'text-danger'}">$${m.total_pnl}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">${m.sharpe_ratio}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value text-danger">$${m.max_drawdown}</div>
                    </div>
                </div>
            </div>
        `;
        
        metricsContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating performance metrics:', error);
    }
}

// PHASE 4: Update Learning Status
async function updateLearningStatus() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/api/ai/learning_status`);
        const data = await response.json();
        
        const learningContainer = document.getElementById('learningStatusContainer');
        if (!learningContainer || !data) return;
        
        const enabled = data.enabled;
        const shadowMode = data.shadow_mode;
        
        let statusBadge = enabled 
            ? '<span class="status-badge status-on"><span>●</span> Enabled</span>'
            : '<span class="status-badge status-off"><span>●</span> Disabled</span>';
        
        let shadowBadge = shadowMode
            ? '<span class="status-badge status-on"><span>●</span> Shadow ON</span>'
            : '<span class="status-badge status-off"><span>●</span> Shadow OFF</span>';
        
        let html = `
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="metric">
                        <span class="metric-label">Self-Learning</span>
                        ${statusBadge}
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="metric">
                        <span class="metric-label">Shadow Mode</span>
                        ${shadowBadge}
                    </div>
                </div>
            </div>
            
            <div class="mb-4">
                <div class="text-sm font-semibold mb-2 text-gray-400">Data Collection:</div>
                <div class="metric">
                    <span class="metric-label">Total Logged Trades</span>
                    <span class="metric-value">${data.data?.total_logged_trades || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Flush</span>
                    <span class="metric-value text-sm">${data.data?.last_flush || 'Never'}</span>
                </div>
            </div>
        `;
        
        // Production Models
        if (data.production_models && Object.keys(data.production_models).length > 0) {
            html += `
                <div class="mb-4">
                    <div class="text-sm font-semibold mb-2 text-gray-400">Production Models:</div>
            `;
            
            for (const [modelType, modelInfo] of Object.entries(data.production_models)) {
                const winRate = modelInfo.metrics?.win_rate || 0;
                const sharpe = modelInfo.metrics?.sharpe_ratio || 0;
                
                html += `
                    <div class="metric">
                        <span class="metric-label">${modelType.toUpperCase()}</span>
                        <span class="metric-value text-sm">
                            ${modelInfo.version} 
                            <span class="text-xs text-gray-400">(WR: ${winRate.toFixed(1)}%, Sharpe: ${sharpe.toFixed(2)})</span>
                        </span>
                    </div>
                `;
            }
            
            html += `</div>`;
        }
        
        // Best Candidates
        if (data.best_candidates && Object.keys(data.best_candidates).length > 0) {
            html += `
                <div class="mb-4">
                    <div class="text-sm font-semibold mb-2 text-gray-400">Best Candidates:</div>
            `;
            
            for (const [modelType, candidateInfo] of Object.entries(data.best_candidates)) {
                const winRate = candidateInfo.metrics?.win_rate || 0;
                const sharpe = candidateInfo.metrics?.sharpe_ratio || 0;
                
                html += `
                    <div class="metric">
                        <span class="metric-label">${modelType.toUpperCase()}</span>
                        <span class="metric-value text-sm">
                            ${candidateInfo.version} 
                            <span class="text-xs text-gray-400">(WR: ${winRate.toFixed(1)}%, Sharpe: ${sharpe.toFixed(2)})</span>
                        </span>
                    </div>
                `;
            }
            
            html += `</div>`;
        }
        
        learningContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating learning status:', error);
        const learningContainer = document.getElementById('learningStatusContainer');
        if (learningContainer) {
            learningContainer.innerHTML = '<div class="text-gray-400">Error loading learning status</div>';
        }
    }
}

// Start dashboard when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}

// ============================================================
// Export for debugging (optional)
// ============================================================

window.dashboardState = state;
window.dashboardConfig = CONFIG;
window.phase3Charts = phase3Charts;
