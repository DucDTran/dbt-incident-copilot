# Naming Conventions Guide

## dbt Project Naming Standards

This document defines the naming conventions for the Airbnb Analytics dbt project.

---

## Model Naming

### Prefix Convention

| Prefix | Description | Example |
|--------|-------------|---------|
| `dim_` | Dimension table | `dim_host`, `dim_listing` |
| `fact_` | Fact table | `fact_reviews`, `fact_bookings` |
| `agg_` | Aggregated/mart table | `agg_daily_reviews` |
| `stg_` | Staging model | `stg_airbnb__listings` |
| `int_` | Intermediate model | `int_reviews_enriched` |

### General Rules

1. Use **snake_case** for all model names
2. Keep names **descriptive but concise** (max 30 characters preferred)
3. Avoid abbreviations unless widely understood
4. Use singular nouns for dimensions (`dim_host`, not `dim_hosts`)
5. Use descriptive names for facts (`fact_reviews`, not `fact_review_data`)

---

## Column Naming

### Standard Columns

| Column Type | Convention | Examples |
|-------------|------------|----------|
| Primary Key | `{entity}_id` | `listing_id`, `host_id` |
| Foreign Key | `{referenced_entity}_id` | `host_id` in dim_listing |
| Date | `{event}_date` or `date` | `review_date`, `created_date` |
| Timestamp | `{event}_at` | `created_at`, `updated_at` |
| Boolean | `is_{condition}` or `has_{thing}` | `is_superhost`, `has_reviews` |
| Count | `{thing}_count` | `review_count`, `listing_count` |
| Amount | `{measure}_amount` or just measure | `price`, `total_revenue` |
| Percentage | `{measure}_pct` | `occupancy_pct` |

### Categorical Columns

For categorical/enum columns:
- Use descriptive names: `room_type`, `sentiment`, `status`
- Document allowed values in schema.yml
- Values should be lowercase with underscores for multi-word

**Example - sentiment column**:
```yaml
- name: sentiment
  description: Review sentiment classification
  tests:
    - accepted_values:
        values: ['positive', 'neutral', 'negative', 'mixed', 'unknown']
```

### Metadata Columns

Standard metadata columns (added by ETL):
- `_etl_loaded_at` - When record was loaded
- `_etl_source` - Source system identifier
- `_etl_batch_id` - Batch processing identifier

---

## Test Naming

### Generic Tests

Generic tests are automatically named by dbt:
```
{test_type}_{model}_{column}
```

Examples:
- `unique_dim_host_host_id`
- `not_null_fact_reviews_review_date`
- `accepted_values_fact_reviews_sentiment`

### Singular Tests

Custom singular tests should follow:
```
assert_{description}
```

Examples:
- `assert_all_review_dates_are_recent`
- `assert_no_orphan_reviews`
- `assert_price_consistency`

---

## Variable Naming

### dbt Variables

Use descriptive names with snake_case:

```yaml
vars:
  min_name_length: 2
  min_listing_name_length: 5
  min_nights_min: 1
  max_nights_outlier: 3650
  min_price: 0.01
  max_price_cap: 10000
```

### Naming Pattern for Thresholds

- `min_{column}` - Minimum allowed value
- `max_{column}` - Maximum allowed value
- `{column}_outlier` - Outlier threshold
- `{column}_cap` - Hard cap value

---

## Macro Naming

### Custom Macros

```
{verb}_{object}
```

Examples:
- `calculate_sentiment_score`
- `join_listings_to_hosts`
- `filter_active_records`
- `safe_cast`

### Test Macros

```
test_{condition}
```

Examples:
- `test_values_in_range`
- `test_no_future_dates`

---

## Schema and Database Naming

### Schema Convention

```
{project}_{layer}
```

Examples:
- `airbnb_raw` - Raw/staging data
- `airbnb_core` - Core dimension and fact tables
- `airbnb_mart` - Business-level aggregations
- `airbnb_seed` - Seed data

### Database Objects

- Tables: Same as model name
- Views: Same as model name (materialization determines object type)
- Temp tables: `tmp_{model_name}_{timestamp}`

---

## Documentation Standards

### Model Descriptions

Every model should have:
1. One-line summary
2. Grain (what does one row represent)
3. Update frequency
4. Owner/contact

Example:
```yaml
- name: fact_reviews
  description: |
    Fact table for reviews (one per review).
    Grain: One row per review.
    Updated: Hourly via incremental load.
    Owner: Analytics Engineering
```

### Column Descriptions

Required for:
- All primary keys
- All foreign keys  
- Business-critical columns
- Calculated columns

---

## File Organization

```
models/
├── core/
│   ├── dim/
│   │   └── dim_*.sql
│   └── fact/
│       └── fact_*.sql
├── mart/
│   └── agg_*.sql
├── staging/
│   └── stg_*.sql
└── schema.yml
```

---

## Contact

For naming convention questions:
- **Slack**: #analytics-engineering
- **Standards Owner**: ae-standards@company.com

