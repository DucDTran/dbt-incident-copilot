# Incident Response Playbook

## dbt Test Failure Response Guide

This playbook provides step-by-step guidance for responding to common dbt test failures.

---

## Quick Reference: Failure Types

| Error Type | Severity | Typical Cause | First Action |
|------------|----------|---------------|--------------|
| accepted_values | Medium | New category in source | Check business rules |
| not_null | High | Data quality issue | Check source system |
| unique | Critical | Duplicate processing | Check incremental logic |
| relationships | High | Orphan records | Check upstream model |
| values_in_range | Medium | Outlier data | Verify business limits |

---

## Playbook: accepted_values Failures

### INC-AV-001: New Values in Categorical Column

**Symptom**: Test fails with "unexpected values: [x, y, z]"

**Investigation Steps**:

1. **Check Business Context**
   - Search knowledge base for column policies
   - Check recent business announcements for new categories
   - Verify if this is a known transition period

2. **Analyze the Data**
   ```sql
   SELECT DISTINCT {column}, COUNT(*) 
   FROM {model}
   WHERE {column} NOT IN (/* current accepted values */)
   GROUP BY 1
   ORDER BY 2 DESC
   ```

3. **Determine Root Cause**
   - If new valid category → Update test
   - If data error → Fix at source
   - If temporary → Snooze and monitor

**Resolution Options**:

| Option | When to Use | Risk |
|--------|-------------|------|
| Add to accepted_values | Business confirmed new value | Low |
| Filter at source | Confirmed data error | Medium |
| Snooze 24-48h | Known transition, temporary | Low |

**Example Fix**:
```yaml
# In schema.yml
- name: sentiment
  tests:
    - accepted_values:
        values: ['positive', 'neutral', 'negative', 'mixed', 'unknown']
```

---

## Playbook: not_null Failures

### INC-NN-001: NULL Values in Required Column

**Symptom**: Test fails with "Got X results, configured to fail if != 0"

**Investigation Steps**:

1. **Identify Affected Records**
   ```sql
   SELECT * 
   FROM {model}
   WHERE {column} IS NULL
   LIMIT 100
   ```

2. **Check for Patterns**
   - Are all NULLs from same source?
   - Same time period?
   - Related to a specific host/listing?

3. **Consult Business Rules**
   - Is NULL acceptable in some cases?
   - Check for migration/deletion policies

**Resolution Options**:

| Option | When to Use | Risk |
|--------|-------------|------|
| COALESCE with default | Business allows default | Low |
| Filter NULL rows | Records are invalid | Medium |
| Change to WARN severity | NULLs are acceptable | Low |
| Fix source system | Data quality issue | High |

**Example Fix** (COALESCE):
```sql
SELECT
  COALESCE(host_id, -1) as host_id,  -- -1 indicates unknown
  ...
FROM source
```

---

## Playbook: relationships Failures

### INC-REL-001: Orphan Records (FK Violation)

**Symptom**: Test fails with "Got X results" for relationship test

**Investigation Steps**:

1. **Find Orphan Records**
   ```sql
   SELECT child.*
   FROM {child_model} child
   LEFT JOIN {parent_model} parent 
     ON child.{fk_column} = parent.{pk_column}
   WHERE parent.{pk_column} IS NULL
   ```

2. **Check Timing**
   - Were parents recently deleted?
   - Is there a timing issue between model runs?
   - Check `_etl_loaded_at` for both models

3. **Verify Business Context**
   - Are orphans expected (soft deletes)?
   - Historical data preservation?

**Resolution Options**:

| Option | When to Use | Risk |
|--------|-------------|------|
| Filter orphans | Invalid data | Medium |
| Use LEFT JOIN | Preserve history | Low |
| Fix upstream | Missing parent data | High |
| Adjust run order | Timing issue | Medium |

**Example Fix** (LEFT JOIN):
```sql
SELECT 
  reviews.*,
  COALESCE(listings.listing_name, 'Listing Unavailable') as listing_name
FROM fact_reviews reviews
LEFT JOIN dim_listing listings USING (listing_id)
```

---

## Playbook: values_in_range Failures

### INC-VR-001: Out-of-Range Values

**Symptom**: Test fails with values outside min/max bounds

**Investigation Steps**:

1. **Identify Outliers**
   ```sql
   SELECT {column}, COUNT(*)
   FROM {model}
   WHERE {column} < {min} OR {column} > {max}
   GROUP BY 1
   ORDER BY 1
   ```

2. **Analyze Distribution**
   - Are these true outliers or errors?
   - Has the business range changed?

3. **Check Business Rules**
   - Are there exception categories?
   - Should the range be updated?

**Resolution Options**:

| Option | When to Use | Risk |
|--------|-------------|------|
| Update range bounds | Business confirms new range | Low |
| Clamp values | Cap at boundary | Medium |
| Filter outliers | Confirmed errors | Medium |
| Add exception logic | Valid edge cases | Low |

**Example Fix** (Clamp):
```sql
SELECT
  GREATEST(LEAST(price, 10000), 0.01) as price,  -- Clamp to range
  ...
FROM source
```

---

## Escalation Matrix

| Severity | Response Time | Escalation |
|----------|--------------|------------|
| Critical (unique, PK) | 15 minutes | Page on-call |
| High (not_null FK) | 1 hour | Slack channel |
| Medium (accepted_values) | 4 hours | Ticket |
| Low (warnings) | 24 hours | Backlog |

---

## Communication Templates

### Slack Update Template

```
🔴 *dbt Test Failure Alert*
*Test*: {test_name}
*Model*: {model_name}
*Severity*: {severity}
*Failed Sample Rows*: {count}

*Status*: Investigating
*ETA*: {time}

*Next Update*: 30 minutes
```

### Resolution Template

```
✅ *Incident Resolved*
*Test*: {test_name}
*Root Cause*: {description}
*Resolution*: {action_taken}
*Prevention*: {future_steps}

*Resolved by*: {engineer}
*Duration*: {time}
```

---

## Post-Incident Actions

After resolving any test failure:

1. **Document** the incident and resolution
2. **Update** runbooks if new scenario
3. **Consider** adding monitoring/alerting
4. **Review** if test configuration needs adjustment
5. **Share** learnings with team

---

## Contact

- **On-call Engineer**: See PagerDuty
- **Analytics Engineering Lead**: ae-lead@company.com
- **Data Platform Team**: #data-platform

