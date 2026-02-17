-- ============================================================================
-- SNOWFLAKE TABLE CREATION SCRIPTS
-- E-commerce Batch Processing Pipeline
-- ============================================================================

-- Create database and schema
CREATE DATABASE IF NOT EXISTS ECOMMERCE_DW;
USE DATABASE ECOMMERCE_DW;

CREATE SCHEMA IF NOT EXISTS RAW_DATA;
CREATE SCHEMA IF NOT EXISTS CLEANED_DATA;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;
CREATE SCHEMA IF NOT EXISTS ML_FEATURES;

USE SCHEMA RAW_DATA;

-- ============================================================================
-- 1. RAW TRANSACTIONS TABLE (Stage 1: Data Ingestion)
-- ============================================================================
CREATE OR REPLACE TABLE RAW_TRANSACTIONS (
    transaction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    amount DECIMAL(12,2),
    category VARCHAR(50),
    payment_method VARCHAR(50),
    device_type VARCHAR(20),
    country VARCHAR(50),
    merchant_id VARCHAR(20),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    fraud_type VARCHAR(50),
    ingestion_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    source_file VARCHAR(255)
);

COMMENT ON TABLE RAW_TRANSACTIONS IS 'Raw transaction data ingested from CSV files';

-- ============================================================================
-- 2. CLEANED TRANSACTIONS TABLE (Stage 2: Data Cleaning)
-- ============================================================================
USE SCHEMA CLEANED_DATA;

CREATE OR REPLACE TABLE CLEANED_TRANSACTIONS (
    transaction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    category VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    country VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(20) NOT NULL,
    is_fraudulent BOOLEAN DEFAULT FALSE,
    fraud_type VARCHAR(50),
    -- Data quality flags
    is_duplicate BOOLEAN DEFAULT FALSE,
    is_outlier BOOLEAN DEFAULT FALSE,
    had_missing_values BOOLEAN DEFAULT FALSE,
    cleaning_action VARCHAR(100),
    -- Metadata
    ingestion_timestamp TIMESTAMP_NTZ,
    cleaning_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    source_file VARCHAR(255),
    -- Primary key
    PRIMARY KEY (transaction_id)
);

COMMENT ON TABLE CLEANED_TRANSACTIONS IS 'Cleaned and validated transaction data';

-- ============================================================================
-- 3. HOURLY ANALYTICS SUMMARY (Stage 3: Aggregation)
-- ============================================================================
USE SCHEMA ANALYTICS;

CREATE OR REPLACE TABLE HOURLY_CATEGORY_SUMMARY (
    summary_hour TIMESTAMP_NTZ NOT NULL,
    category VARCHAR(50) NOT NULL,
    transaction_count INTEGER,
    total_amount DECIMAL(15,2),
    avg_amount DECIMAL(12,2),
    min_amount DECIMAL(12,2),
    max_amount DECIMAL(12,2),
    unique_customers INTEGER,
    unique_merchants INTEGER,
    fraud_count INTEGER DEFAULT 0,
    fraud_amount DECIMAL(15,2) DEFAULT 0,
    top_payment_method VARCHAR(50),
    top_country VARCHAR(50),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (summary_hour, category)
);

COMMENT ON TABLE HOURLY_CATEGORY_SUMMARY IS 'Hourly aggregated metrics by product category';

-- Hourly overall summary
CREATE OR REPLACE TABLE HOURLY_OVERALL_SUMMARY (
    summary_hour TIMESTAMP_NTZ NOT NULL PRIMARY KEY,
    transaction_count INTEGER,
    total_amount DECIMAL(15,2),
    avg_amount DECIMAL(12,2),
    unique_customers INTEGER,
    unique_merchants INTEGER,
    fraud_count INTEGER DEFAULT 0,
    fraud_rate_percent DECIMAL(5,2),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE HOURLY_OVERALL_SUMMARY IS 'Hourly overall transaction summary';

-- Daily category summary
CREATE OR REPLACE TABLE DAILY_CATEGORY_SUMMARY (
    summary_date DATE NOT NULL,
    category VARCHAR(50) NOT NULL,
    transaction_count INTEGER,
    total_amount DECIMAL(15,2),
    avg_amount DECIMAL(12,2),
    fraud_count INTEGER DEFAULT 0,
    fraud_rate_percent DECIMAL(5,2),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (summary_date, category)
);

COMMENT ON TABLE DAILY_CATEGORY_SUMMARY IS 'Daily aggregated metrics by product category';

-- ============================================================================
-- 4. ML FEATURES TABLE (Stage 4: Feature Engineering)
-- ============================================================================
USE SCHEMA ML_FEATURES;

CREATE OR REPLACE TABLE FRAUD_DETECTION_FEATURES (
    transaction_id VARCHAR(50) NOT NULL PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    category VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    country VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(20) NOT NULL,
    
    -- Time-based features
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_night_time BOOLEAN,
    
    -- Amount-based features
    amount_zscore DECIMAL(10,4),
    amount_percentile DECIMAL(5,2),
    amount_vs_category_avg DECIMAL(10,4),
    amount_vs_customer_avg DECIMAL(10,4),
    
    -- Customer behavior features (rolling windows)
    customer_txn_count_1h INTEGER,
    customer_txn_count_24h INTEGER,
    customer_avg_amount_24h DECIMAL(12,2),
    customer_max_amount_24h DECIMAL(12,2),
    customer_total_amount_24h DECIMAL(15,2),
    customer_txn_count_7d INTEGER,
    customer_avg_amount_7d DECIMAL(12,2),
    
    -- Velocity features
    time_since_last_txn_minutes DECIMAL(10,2),
    txn_velocity_1h DECIMAL(10,4),
    
    -- Merchant features
    merchant_txn_count_1h INTEGER,
    merchant_avg_amount_1h DECIMAL(12,2),
    
    -- Category features
    category_txn_count_1h INTEGER,
    category_avg_amount_1h DECIMAL(12,2),
    
    -- Risk indicators
    is_high_risk_country BOOLEAN,
    is_high_risk_payment_method BOOLEAN,
    is_new_device BOOLEAN,
    
    -- Composite risk score
    risk_score DECIMAL(5,2),
    
    -- Target variable (for training)
    is_fraudulent BOOLEAN,
    
    -- Metadata
    feature_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- Foreign key reference
    FOREIGN KEY (transaction_id) REFERENCES CLEANED_DATA.CLEANED_TRANSACTIONS(transaction_id)
);

COMMENT ON TABLE FRAUD_DETECTION_FEATURES IS 'Engineered features for fraud detection ML model';

-- Customer profile features (for reference)
CREATE OR REPLACE TABLE CUSTOMER_PROFILE_FEATURES (
    customer_id VARCHAR(20) NOT NULL PRIMARY KEY,
    first_seen_date DATE,
    total_lifetime_txns INTEGER,
    total_lifetime_amount DECIMAL(15,2),
    avg_transaction_amount DECIMAL(12,2),
    preferred_category VARCHAR(50),
    preferred_payment_method VARCHAR(50),
    preferred_device_type VARCHAR(20),
    preferred_country VARCHAR(50),
    unique_merchants_count INTEGER,
    days_since_first_txn INTEGER,
    fraud_history_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE CUSTOMER_PROFILE_FEATURES IS 'Customer profile features for ML model';

-- ============================================================================
-- 5. CREATE STAGES FOR FILE LOADING
-- ============================================================================
USE SCHEMA RAW_DATA;

-- Create stage for CSV files
CREATE OR REPLACE STAGE TRANSACTION_CSV_STAGE
    DIRECTORY = (ENABLE = TRUE)
    FILE_FORMAT = (
        TYPE = 'CSV'
        FIELD_DELIMITER = ','
        SKIP_HEADER = 1
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        NULL_IF = ('NULL', 'null', '')
    );

-- ============================================================================
-- 6. CREATE VIEWS FOR MONITORING
-- ============================================================================
USE SCHEMA ANALYTICS;

-- Data quality monitoring view
CREATE OR REPLACE VIEW DATA_QUALITY_MONITORING AS
SELECT 
    DATE(cleaning_timestamp) as processing_date,
    COUNT(*) as total_records,
    SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) as duplicate_count,
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as outlier_count,
    SUM(CASE WHEN had_missing_values THEN 1 ELSE 0 END) as missing_value_count,
    SUM(CASE WHEN cleaning_action IS NOT NULL THEN 1 ELSE 0 END) as cleaned_count,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(DISTINCT merchant_id) as unique_merchants
FROM CLEANED_DATA.CLEANED_TRANSACTIONS
GROUP BY DATE(cleaning_timestamp);

-- Fraud monitoring view
CREATE OR REPLACE VIEW FRAUD_MONITORING AS
SELECT 
    DATE(timestamp) as transaction_date,
    category,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate_percent,
    SUM(amount) as total_amount,
    SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END) as fraud_amount
FROM CLEANED_DATA.CLEANED_TRANSACTIONS
GROUP BY DATE(timestamp), category;

-- ============================================================================
-- 7. CREATE INDEXES (if using Enterprise edition with search optimization)
-- ============================================================================
-- Note: These are optional and require appropriate Snowflake edition

-- ALTER TABLE CLEANED_DATA.CLEANED_TRANSACTIONS ADD SEARCH OPTIMIZATION;
-- ALTER TABLE ML_FEATURES.FRAUD_DETECTION_FEATURES ADD SEARCH OPTIMIZATION;

-- ============================================================================
-- 8. SAMPLE QUERIES FOR VALIDATION
-- ============================================================================

-- Check raw data count
-- SELECT COUNT(*) FROM RAW_DATA.RAW_TRANSACTIONS;

-- Check cleaned data quality
-- SELECT 
--     cleaning_action,
--     COUNT(*) as count
-- FROM CLEANED_DATA.CLEANED_TRANSACTIONS
-- WHERE cleaning_action IS NOT NULL
-- GROUP BY cleaning_action;

-- Check hourly summary
-- SELECT * FROM ANALYTICS.HOURLY_CATEGORY_SUMMARY 
-- ORDER BY summary_hour DESC, category 
-- LIMIT 10;

-- Check ML features
-- SELECT 
--     transaction_id,
--     amount,
--     customer_txn_count_1h,
--     customer_avg_amount_24h,
--     risk_score,
--     is_fraudulent
-- FROM ML_FEATURES.FRAUD_DETECTION_FEATURES
-- ORDER BY risk_score DESC
-- LIMIT 10;
