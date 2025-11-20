-- =====================================================
-- OKX Stream Hunter - Neon Database Schema
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- 1. CANDLES TABLES (Multi-Timeframe)
-- =====================================================

-- Template for candle tables
CREATE TABLE IF NOT EXISTS candles_template (
    id BIGSERIAL,
    symbol TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    trades INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, open_time)
);

-- Create individual candle tables for each timeframe
CREATE TABLE IF NOT EXISTS candles_1s (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_5s (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_1m (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_3m (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_5m (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_15m (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_1h (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_4h (LIKE candles_template INCLUDING ALL);
CREATE TABLE IF NOT EXISTS candles_1d (LIKE candles_template INCLUDING ALL);

-- Convert to hypertables for better time-series performance
SELECT create_hypertable('candles_1s', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_5s', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_1m', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_3m', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_5m', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_15m', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_1h', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_4h', 'open_time', if_not_exists => TRUE);
SELECT create_hypertable('candles_1d', 'open_time', if_not_exists => TRUE);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_candles_1s_symbol_time ON candles_1s(symbol, open_time DESC);
CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_time ON candles_1m(symbol, open_time DESC);
CREATE INDEX IF NOT EXISTS idx_candles_1h_symbol_time ON candles_1h(symbol, open_time DESC);

-- =====================================================
-- 2. INDICATORS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    close NUMERIC(20, 8),
    rsi_14 NUMERIC(10, 4),
    ema_20 NUMERIC(20, 8),
    ema_50 NUMERIC(20, 8),
    macd NUMERIC(20, 8),
    macd_signal NUMERIC(20, 8),
    macd_hist NUMERIC(20, 8),
    bb_middle NUMERIC(20, 8),
    bb_upper NUMERIC(20, 8),
    bb_lower NUMERIC(20, 8),
    stoch_rsi NUMERIC(10, 4),
    atr_14 NUMERIC(20, 8),
    volume NUMERIC(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicators_symbol_tf_time 
    ON indicators(symbol, timeframe, timestamp DESC);

-- =====================================================
-- 3. MARKET EVENTS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS market_events (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL, -- 'liquidation', 'funding_rate', 'open_interest', 'whale_trade'
    event_data JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_events_symbol_type_time 
    ON market_events(symbol, event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_events_data 
    ON market_events USING GIN(event_data);

-- =====================================================
-- 4. ORDER BOOK SNAPSHOTS
-- =====================================================

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC(20, 8),
    best_ask NUMERIC(20, 8),
    bid_volume NUMERIC(20, 8),
    ask_volume NUMERIC(20, 8),
    mm_pressure NUMERIC(20, 8), -- bid_volume - ask_volume
    spread NUMERIC(20, 8),
    levels_data JSONB, -- full order book data
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time 
    ON orderbook_snapshots(symbol, timestamp DESC);

-- =====================================================
-- 5. HEALTH METRICS
-- =====================================================

CREATE TABLE IF NOT EXISTS health_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    metric_data JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_health_metrics_name_time 
    ON health_metrics(metric_name, timestamp DESC);

-- =====================================================
-- 6. SYSTEM LOGS
-- =====================================================

CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    level TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message TEXT NOT NULL,
    context JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_logs_level_time 
    ON system_logs(level, timestamp DESC);

-- =====================================================
-- 7. DATA QUALITY LOGS
-- =====================================================

CREATE TABLE IF NOT EXISTS data_quality_logs (
    id BIGSERIAL PRIMARY KEY,
    issue_type TEXT NOT NULL, -- 'price_spike', 'missing_data', 'duplicate', 'invalid_value'
    symbol TEXT,
    timeframe TEXT,
    details JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_logs_type_time 
    ON data_quality_logs(issue_type, timestamp DESC);

-- =====================================================
-- 8. RETENTION POLICIES
-- =====================================================

-- Keep 1s candles for 7 days
SELECT add_retention_policy('candles_1s', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep 5s candles for 14 days
SELECT add_retention_policy('candles_5s', INTERVAL '14 days', if_not_exists => TRUE);

-- Keep 1m candles for 30 days
SELECT add_retention_policy('candles_1m', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep 1h candles for 1 year
SELECT add_retention_policy('candles_1h', INTERVAL '365 days', if_not_exists => TRUE);

-- Keep 1d candles forever (no retention policy)

-- Health metrics retention: 30 days
SELECT add_retention_policy('health_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- System logs retention: 60 days
SELECT add_retention_policy('system_logs', INTERVAL '60 days', if_not_exists => TRUE);

-- =====================================================
-- 9. MATERIALIZED VIEWS FOR COMMON QUERIES
-- =====================================================

-- Latest candles per timeframe (for quick access)
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_candles AS
SELECT DISTINCT ON (symbol, timeframe)
    symbol,
    timeframe,
    open_time,
    close_time,
    open,
    high,
    low,
    close,
    volume,
    trades
FROM (
    SELECT 'BTC-USDT-SWAP' as symbol, '1m' as timeframe, * FROM candles_1m
    UNION ALL
    SELECT 'BTC-USDT-SWAP', '5m', * FROM candles_5m
    UNION ALL
    SELECT 'BTC-USDT-SWAP', '1h', * FROM candles_1h
    UNION ALL
    SELECT 'BTC-USDT-SWAP', '1d', * FROM candles_1d
) combined
ORDER BY symbol, timeframe, open_time DESC;

CREATE UNIQUE INDEX ON latest_candles(symbol, timeframe);

-- Refresh policy for materialized view
CREATE OR REPLACE FUNCTION refresh_latest_candles()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY latest_candles;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- COMPLETE ✅
-- =====================================================