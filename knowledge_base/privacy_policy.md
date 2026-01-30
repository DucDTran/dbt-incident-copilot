# Privacy and PII Handling Policy

## Data Privacy Guidelines for Analytics

This document outlines privacy requirements and PII handling procedures for the Airbnb Analytics platform.

---

## PII Classification

### Personally Identifiable Information (PII)

The following fields are classified as PII and require special handling:

| Field | Classification | Handling |
|-------|---------------|----------|
| `host_name` | Direct PII | Hash in public reports |
| `reviewer_name` | Direct PII | Hash in public reports |
| `email` | Direct PII | Never expose in analytics |
| `phone` | Direct PII | Never expose in analytics |
| `address` | Direct PII | Aggregate to city level |
| `host_id` | Indirect PII | Pseudonymize for external |
| `listing_id` | Indirect PII | OK for internal analytics |

### Sensitivity Levels

1. **Public**: Can be shared externally (aggregated metrics)
2. **Internal**: Visible to all employees (de-identified)
3. **Restricted**: Need-to-know basis (contains PII)
4. **Confidential**: Highly sensitive (financial, legal)

---

## Data Handling Rules

### PP-001: Name Fields

**Applies to**: `host_name`, `reviewer_name`

#### Internal Analytics
- Names can be used for:
  - Debugging data quality issues
  - Individual host/guest lookup (authorized users)
  - Customer support context

#### External/Public Analytics
- Names must be:
  - Hashed using SHA-256
  - Truncated to initials only
  - Removed entirely

#### Validation Rules
- Minimum length: 2 characters (to catch single-letter placeholders)
- Maximum length: 100 characters
- Must not contain only special characters
- Must not be known test values ("Test User", "John Doe", etc.)

### PP-002: Review Content

**Applies to**: Review text and comments

#### Storage
- Full text stored in restricted tables
- Sentiment score (derived) is public

#### Analytics Use
- Never display raw review text in dashboards
- Aggregate sentiment is acceptable
- Word clouds require PII scrubbing

---

## Test Considerations

### PII-Related Test Failures

When tests fail on PII columns:

1. **not_null on host_name**: 
   - May fail due to GDPR deletion requests
   - Check if record is in `deleted_users` table
   - Acceptable to have NULL for deleted users

2. **length checks on names**:
   - Minimum 2 characters catches placeholder data
   - Names like "A" or "." indicate data issues
   - Single-character names from certain regions are valid (verify)

3. **reviewer_name validation**:
   - Same rules as host_name
   - Additional check for anonymized reviews ("Anonymous", "Guest")

### Handling Failures

For PII-related test failures:

```sql
-- Example: Exclude known deleted users
WHERE host_id NOT IN (SELECT id FROM deleted_users)
  AND host_name IS NOT NULL
  AND LENGTH(host_name) >= 2
```

---

## Compliance Requirements

### GDPR (EU Users)

- Right to deletion: Users can request data removal
- Right to access: Users can request their data
- Data minimization: Only collect necessary data

**Impact on Analytics**:
- Some records may have NULLed PII fields
- Do not fail pipelines on GDPR-deleted records
- Maintain audit trail of deletions

### CCPA (California Users)

- Similar to GDPR
- Additional disclosure requirements
- Opt-out mechanisms

---

## Data Retention

### Retention Periods

| Data Type | Retention | Notes |
|-----------|-----------|-------|
| Raw PII | 2 years | Then anonymize |
| Aggregated metrics | 7 years | No PII |
| Audit logs | 5 years | Required by law |
| Test results | 90 days | Operational data |

### Anonymization Process

After retention period:
1. Replace names with hash
2. Generalize dates to month/year
3. Aggregate geographic data to region
4. Remove all direct identifiers

---

## Incident Response

### PII Exposure Incident

If PII is accidentally exposed:

1. **Immediate**: Remove/restrict access
2. **Within 1 hour**: Notify Security team
3. **Within 24 hours**: Document incident
4. **Within 72 hours**: Regulatory notification (if required)

### Contact

- **Security Team**: security@company.com
- **Privacy Officer**: privacy@company.com
- **Incident Hotline**: #security-incidents

---

## Analytics Engineering Guidelines

### DO
- Use aggregated metrics in dashboards
- Hash PII in development environments
- Document PII columns in schema.yml
- Use row-level security for sensitive tables

### DON'T
- Expose raw PII in Looker/Tableau
- Copy PII to personal databases
- Share screenshots containing PII
- Log PII in application logs

---

## Code Examples

### Hashing PII for Reports

```sql
-- Hash name for external reports
SELECT 
  SHA256(CAST(host_name AS STRING)) as host_name_hash,
  -- Other non-PII fields
  listing_count,
  avg_rating
FROM agg_host_performance
```

### Filtering Deleted Users

```sql
-- Exclude GDPR-deleted records from analytics
SELECT *
FROM dim_host
WHERE deletion_requested_at IS NULL
  OR deletion_requested_at > CURRENT_DATE()
```

---

## Approval Required

The following actions require Privacy team approval:
- New PII field collection
- PII export to external systems
- Retention policy exceptions
- Third-party data sharing

**Contact**: privacy-review@company.com

