# Data Quality Policies

## DQ-001: Airbnb Analytics Data Quality Standards

This document outlines the data quality standards and policies for the Airbnb Analytics platform.

### Overview

All data flowing through our analytics pipeline must meet specific quality standards to ensure business decisions are made on reliable information.

---

## Column-Level Quality Standards

### Sentiment Values Policy

**Policy ID**: DQ-SEN-001  
**Last Updated**: 2024-06-15  
**Owner**: Data Governance Team

The `sentiment` column in reviews data must contain one of the following values:
- `positive` - Indicates favorable review sentiment
- `neutral` - Indicates neither positive nor negative sentiment
- `negative` - Indicates unfavorable review sentiment

**Recent Amendment (2024-12-01)**:  
Due to the introduction of our new NLP sentiment analysis pipeline (v2.0), the following additional values are now **temporarily accepted** during the transition period:
- `mixed` - Reviews containing both positive and negative elements
- `unknown` - Reviews that could not be classified by the NLP model

**Action Required**: Analytics Engineering team should update accepted_values tests to include `mixed` and `unknown` until the NLP team completes the model retraining by Q1 2025.

### Room Type Values Policy

**Policy ID**: DQ-RT-001  
**Last Updated**: 2024-09-20  
**Owner**: Product Analytics Team

The `room_type` column in listings data supports:
- `Entire home/apt` - Complete property rental
- `Private room` - Private room in shared space
- `Shared room` - Shared sleeping space
- `Hotel room` - Hotel-style accommodation

**Amendment Notice (2024-11-15)**:  
The business has approved a new room type category:
- `Studio` - Studio apartments (subset of Entire home/apt for better categorization)

This value should be added to the accepted values list for the `room_type` column.

---

## Price Data Standards

### Price Range Policy

**Policy ID**: DQ-PRC-001  
**Last Updated**: 2024-08-01  
**Owner**: Finance & Analytics

Standard price ranges for Airbnb listings:
- **Minimum**: $0.01 (free listings are not permitted)
- **Maximum**: $10,000 per night (soft cap)

**Exception Cases**:
1. **Luxury Properties**: Listings tagged as "Luxury" category may exceed $10,000
2. **Promotional Rates**: $0.00 is allowed for promotional "first night free" campaigns
3. **Data Corrections**: Negative prices indicate data errors and should be filtered

**Recommended Action**: For prices outside the standard range:
- $0.00 → Verify if promotional campaign, else flag
- > $10,000 → Verify luxury tag, else cap at $10,000
- < $0 → Filter and report to data engineering

---

## Null Value Handling

### Not Null Standards

**Policy ID**: DQ-NULL-001  
**Last Updated**: 2024-07-10

#### Critical Columns (Must Never Be Null)
- `listing_id` - Primary identifier
- `host_id` - Foreign key to hosts
- `review_id` - Primary identifier
- `review_date` - Required for time-series analysis

#### Conditionally Nullable Columns
- `host_id` in listings - May be null during host account migration (grace period: 72 hours)
- `price` - May be null for draft listings

**Handling Missing host_id**:  
If `host_id` is null in `dim_listing`:
1. Check if listing is in "pending_migration" status
2. If yes, exclude from analytics but don't fail pipeline
3. If no, investigate as data quality issue

---

## Referential Integrity

### Foreign Key Relationships

**Policy ID**: DQ-FK-001

All foreign key relationships must maintain referential integrity EXCEPT:

1. **Historical Reviews**: Reviews may reference listings that have been soft-deleted
   - Action: LEFT JOIN instead of INNER JOIN, provide default "Listing Unavailable" for display
   
2. **Migration Period**: During host migration windows, orphaned records are expected
   - Check `_etl_loaded_at` timestamp - if within 24 hours, snooze alert

---

## Data Freshness Requirements

| Table | Max Staleness | Alert Level |
|-------|--------------|-------------|
| fact_reviews | 6 hours | WARN |
| dim_listing | 24 hours | ERROR |
| dim_host | 24 hours | ERROR |
| agg_* tables | 48 hours | WARN |

---

## Contact

For data quality questions or policy exceptions:
- **Slack**: #data-quality-team
- **Email**: dq-team@company.com
- **On-call**: See PagerDuty rotation

