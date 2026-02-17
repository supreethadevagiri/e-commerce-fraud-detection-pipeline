# E-Commerce Fraud Detection Data Schema

## Overview
This document describes the data schema for the synthetic e-commerce fraud detection dataset.

## Record Structure

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `transaction_id` | STRING (UUID) | Unique identifier for each transaction | "TXN-A1B2C3D4E5F67890" |
| `customer_id` | STRING | Unique identifier for the customer | "CUST-12345678" |
| `timestamp` | DATETIME | Transaction timestamp (YYYY-MM-DD HH:MM:SS) | "2024-01-15 14:32:18" |
| `amount` | FLOAT | Transaction amount in USD | 299.99 |
| `product_category` | STRING | Category of purchased product | "Electronics" |
| `payment_method` | STRING | Method used for payment | "Credit Card" |
| `device_type` | STRING | Device used for the transaction | "Mobile - iOS" |
| `location` | STRING | Country/region of transaction | "USA" |
| `is_fraud` | BOOLEAN | Target variable - True if fraudulent | false |

## Field Details

### transaction_id
- **Format**: `TXN-{16-char hex}`
- **Generation**: UUID-based, unique per transaction
- **Example**: `TXN-A1B2C3D4E5F67890`

### customer_id
- **Format**: `CUST-{8-digit number}`
- **Generation**: Random 8-digit number with leading zeros
- **Example**: `CUST-00123456`

### timestamp
- **Format**: `YYYY-MM-DD HH:MM:SS`
- **Range**: Real-time generation within specified duration
- **Timezone**: Local system time

### amount
- **Type**: Float with 2 decimal places
- **Normal Range**: $5.00 - $500.00
- **Fraud Pattern**: Often very small ($0.50-$5.00) for testing or very large ($500-$15,000)
- **Outlier Range**: $0.01-$0.99 or $15,000-$100,000

### product_category
**Possible Values:**
- Electronics
- Clothing
- Home & Garden
- Sports & Outdoors
- Books & Media
- Health & Beauty
- Toys & Games
- Food & Beverages
- Automotive
- Jewelry & Watches
- Pet Supplies
- Office Supplies

**Fraud Patterns:**
- Fraudsters often target: Electronics, Jewelry & Watches

### payment_method
**Possible Values:**
- Credit Card
- Debit Card
- PayPal
- Apple Pay
- Google Pay
- Cryptocurrency
- Bank Transfer
- Buy Now Pay Later

**Fraud Patterns:**
- Higher fraud risk: Cryptocurrency, Buy Now Pay Later

### device_type
**Possible Values:**
- Desktop
- Mobile - iOS
- Mobile - Android
- Tablet - iOS
- Tablet - Android
- Smart TV
- Gaming Console
- Unknown Device

**Fraud Patterns:**
- Higher fraud risk: Unknown Device, Desktop (for bot attacks)

### location
**Possible Values:**
- USA
- UK
- Canada
- Germany
- France
- Australia
- Japan
- Brazil
- India
- Mexico
- Spain
- Italy
- Netherlands
- Singapore
- Nigeria (high-risk)
- Romania (high-risk)
- Russia (high-risk)
- Unknown (high-risk)

### is_fraud (Target Variable)
- **Type**: Boolean
- **True Rate**: ~5% (configurable)
- **Determination**: Based on multiple risk factors:
  - Transaction amount (very high or very low)
  - Payment method (cryptocurrency, BNPL)
  - Device type (unknown devices)
  - Location (high-risk countries)

## Data Quality Challenges

### 1. Missing Values (5% of records)
- **Affected Fields**: `product_category`, `payment_method`, `device_type`, `location`
- **Pattern**: Random 1-2 fields set to NULL per affected record
- **Detection**: Check for NULL/None values

### 2. Duplicate Records (2% of records)
- **Pattern**: Exact duplicate of previous transaction with slightly modified timestamp (+1-60 seconds)
- **Detection**: Same transaction_id with different timestamp
- **Marker**: Internal `_is_duplicate` field (for tracking)

### 3. Outliers in Amount (1% of records)
- **Pattern**: Extreme values outside normal range
- **Types**:
  - Very High: $15,000 - $100,000
  - Very Low: $0.01 - $0.99
- **Detection**: Statistical outlier detection (Z-score, IQR)

## Data Types Summary

| Field | JSON Type | Python Type | SQL Type | Notes |
|-------|-----------|-------------|----------|-------|
| transaction_id | string | str | VARCHAR(32) | Primary Key |
| customer_id | string | str | VARCHAR(16) | Foreign Key |
| timestamp | string | str | DATETIME | ISO format |
| amount | number | float | DECIMAL(10,2) | Currency |
| product_category | string/null | str/None | VARCHAR(50) | Nullable |
| payment_method | string/null | str/None | VARCHAR(30) | Nullable |
| device_type | string/null | str/None | VARCHAR(30) | Nullable |
| location | string/null | str/None | VARCHAR(30) | Nullable |
| is_fraud | boolean | bool | BOOLEAN | Target |

## Sample JSON Record

```json
{
  "transaction_id": "TXN-A1B2C3D4E5F67890",
  "customer_id": "CUST-12345678",
  "timestamp": "2024-01-15 14:32:18",
  "amount": 299.99,
  "product_category": "Electronics",
  "payment_method": "Credit Card",
  "device_type": "Mobile - iOS",
  "location": "USA",
  "is_fraud": false
}
```

## Sample JSON Record with Missing Values

```json
{
  "transaction_id": "TXN-B2C3D4E5F6G78901",
  "customer_id": "CUST-87654321",
  "timestamp": "2024-01-15 14:33:45",
  "amount": 150.00,
  "product_category": null,
  "payment_method": "PayPal",
  "device_type": null,
  "location": "UK",
  "is_fraud": false
}
```

## Sample Fraudulent Record

```json
{
  "transaction_id": "TXN-C3D4E5F6G7H89012",
  "customer_id": "CUST-11111111",
  "timestamp": "2024-01-15 14:35:02",
  "amount": 8500.00,
  "product_category": "Electronics",
  "payment_method": "Cryptocurrency",
  "device_type": "Unknown Device",
  "location": "Nigeria",
  "is_fraud": true
}
```

## Data Generation Rates

| Metric | Value |
|--------|-------|
| Default Rate | 1,000 records/hour |
| Default Duration | 6 hours |
| Total Records | ~6,000+ records |
| Records/Second | ~0.28 |
| Batch Size | 10-second batches |

## File Format

- **Format**: JSON Lines (JSONL)
- **Extension**: `.jsonl`
- **Encoding**: UTF-8
- **Structure**: One JSON object per line

## Kafka Message Format

- **Key**: `customer_id` (for partitioning)
- **Value**: Full JSON record
- **Topic**: `ecommerce-transactions` (configurable)
- **Serialization**: JSON with UTF-8 encoding
