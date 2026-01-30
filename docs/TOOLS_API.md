# Tools API Documentation

This document provides comprehensive API documentation for all tools available
to the dbt Co-Work agent for investigating test failures.

## Overview

The agent uses a suite of specialized tools to gather context about test failures:

| Tool | Purpose | Used By |
|------|---------|---------|
| `get_model_lineage` | Trace model dependencies | Investigator |
| `read_model_sql` | Read SQL transformation code | Investigator |
| `read_schema_definition` | Read schema.yml configuration | Investigator |
| `search_knowledge_base` | Find business rules | Investigator |
| `execute_sql` | Query data warehouse (read-only) | Investigator |
| `get_test_details` | Get test failure details | Investigator |
| `propose_fix` | Generate fix recommendations | Fix Proposer |

---

## Tool Reference

### get_model_lineage

Get the upstream and downstream dependencies for a dbt model.

**Function Signature:**
```python
def adk_get_model_lineage(model_name: str) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | Name of the dbt model (e.g., "fact_reviews") |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "data": {
    "model_name": "fact_reviews",
    "upstream_models": ["stg_reviews", "dim_listing"],
    "upstream_sources": ["raw.reviews"],
    "downstream_models": ["report_reviews_summary"],
    "tests": ["not_null", "accepted_values"]
  }
}
```

**Example Usage:**
```python
result = adk_get_model_lineage("fact_reviews")
lineage = json.loads(result)
print(f"Upstream: {lineage['data']['upstream_models']}")
```

---

### read_model_sql

Read the SQL transformation code for a dbt model.

**Function Signature:**
```python
def adk_read_model_sql(model_name: str) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | Name of the dbt model |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "path": "models/marts/fact_reviews.sql",
  "lines": 45,
  "content": "SELECT\n    review_id,\n    ..."
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Model not found: unknown_model"
}
```

---

### read_schema_definition

Read the schema.yml definition for a dbt model including column definitions,
tests, and descriptions.

**Function Signature:**
```python
def adk_read_schema_definition(model_name: str) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | Name of the dbt model |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "path": "models/marts/schema.yml",
  "model_definition": {
    "name": "fact_reviews",
    "description": "Review facts table",
    "columns": [
      {
        "name": "sentiment",
        "description": "Review sentiment classification",
        "tests": [
          {"accepted_values": {"values": ["positive", "neutral", "negative"]}}
        ]
      }
    ]
  },
  "full_content": "version: 2\nmodels:\n  ..."
}
```

---

### search_knowledge_base

Search the business rules knowledge base for relevant documentation using
semantic search.

**Function Signature:**
```python
def adk_search_knowledge_base(
    query: str,
    context_column: str = "",
    context_model: str = ""
) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `str` | Yes | Search query (business term, policy name, etc.) |
| `context_column` | `str` | No | Column name for context |
| `context_model` | `str` | No | Model name for context |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "results": [
    {
      "document": "Business Rules",
      "section_title": "Review Sentiment Analysis",
      "content": "BR-SEN-001: Sentiment Classification...",
      "relevance_score": 0.89,
      "is_semantic": true
    }
  ]
}
```

**Best Practices:**
- Use specific business terms in your query
- Include column and model context for better results
- Check `is_semantic` to know if embeddings were used

---

### execute_sql

Execute a read-only SQL query against the data warehouse to investigate
data patterns.

**Function Signature:**
```python
def adk_execute_sql(sql: str, limit: int = 100) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sql` | `str` | Yes | SQL SELECT query (read-only) |
| `limit` | `int` | No | Maximum rows to return (default: 100) |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "data": {
    "columns": ["sentiment", "count"],
    "rows": [
      {"sentiment": "positive", "count": 1234},
      {"sentiment": "mixed", "count": 47}
    ],
    "row_count": 2,
    "execution_time_ms": 234
  }
}
```

**Security:**
- Only SELECT queries are allowed
- INSERT, UPDATE, DELETE, DROP are blocked
- Queries are automatically limited

**Table Naming:**
Use fully qualified BigQuery format:
```sql
SELECT * FROM `project_id.dataset_name.table_name` LIMIT 10
```

**Error Handling:**
If SQL execution fails (auth, permissions), continue investigation using
other tools. Many issues can be diagnosed from code analysis alone.

---

### get_test_details

Get detailed information about a specific test failure from Elementary.

**Function Signature:**
```python
def adk_get_test_details(test_id: str) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `test_id` | `str` | Yes | Unique test identifier |

**Returns:**
JSON string containing:
```json
{
  "status": "success",
  "data": {
    "test_id": "test.airbnb_analytics.accepted_values_fact_reviews_sentiment",
    "test_name": "accepted_values_fact_reviews_sentiment",
    "model_name": "fact_reviews",
    "column_name": "sentiment",
    "status": "fail",
    "error_message": "Found unexpected values: ['mixed', 'unknown']",
    "failed_rows": 47,
    "failed_row_samples": [
      {"review_id": 98234, "sentiment": "mixed"}
    ],
    "test_type": "generic",
    "severity": "ERROR",
    "executed_at": "2024-12-18T10:30:00Z"
  }
}
```

---

### propose_fix

Generate fix recommendations based on the investigation findings.

**Function Signature:**
```python
def adk_propose_fix(
    test_name: str,
    model_name: str,
    column_name: str,
    root_cause: str,
    fix_options: str
) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `test_name` | `str` | Yes | Name of the failing test |
| `model_name` | `str` | Yes | Name of the dbt model |
| `column_name` | `str` | Yes | Column name (if applicable) |
| `root_cause` | `str` | Yes | Diagnosed root cause |
| `fix_options` | `str` | Yes | JSON array of fix options |

**Fix Options Format:**
```json
[
  {
    "id": "fix_1",
    "title": "Add new accepted values",
    "description": "Update accepted_values test to include 'mixed' and 'unknown'",
    "fix_type": "schema",
    "risk_level": "low",
    "rationale": "Based on BR-SEN-001, these are valid sentiment values",
    "pros": "Quick fix, aligns with business rules",
    "cons": "May mask data quality issues",
    "when_appropriate": "When new values are legitimate per business rules",
    "code_change": {
      "action": "add_accepted_values",
      "file_path": "models/marts/schema.yml",
      "details": {
        "model": "fact_reviews",
        "column": "sentiment",
        "new_values": ["mixed", "unknown"]
      }
    }
  }
]
```

**Fix Types:**
| Type | Description | Example Actions |
|------|-------------|-----------------|
| `schema` | Modify schema.yml | add_accepted_values, update_severity, add_test |
| `sql` | Modify SQL code | add_where_clause, add_coalesce, add_case_statement |

**Risk Levels:**
| Level | Description |
|-------|-------------|
| `low` | Minimal impact, easy to revert |
| `medium` | Some impact, requires review |
| `high` | Significant impact, thorough testing needed |

---

## Error Handling

All tools return errors in a consistent format:

```json
{
  "status": "error",
  "message": "Description of the error",
  "data": null
}
```

**Common Errors:**
| Error | Cause | Recovery |
|-------|-------|----------|
| Model not found | Invalid model name | Check dbt manifest |
| SQL auth error | Missing credentials | Continue with code analysis |
| Knowledge base empty | No documents indexed | Check KNOWLEDGE_BASE_PATH |

**Best Practice:**
Always check `status` field before processing results. If a tool fails,
continue the investigation using other tools when possible.

---

## Usage Examples

### Complete Investigation Flow

```python
# 1. Get test details
test_details = adk_get_test_details("test.airbnb.accepted_values_sentiment")

# 2. Get model lineage
lineage = adk_get_model_lineage("fact_reviews")

# 3. Read SQL code
sql_code = adk_read_model_sql("fact_reviews")

# 4. Read schema definition
schema = adk_read_schema_definition("fact_reviews")

# 5. Search for business rules
rules = adk_search_knowledge_base(
    query="sentiment classification rules",
    context_column="sentiment",
    context_model="fact_reviews"
)

# 6. Optionally query data
data_sample = adk_execute_sql("""
    SELECT sentiment, COUNT(*) as cnt
    FROM `project.dataset.fact_reviews`
    GROUP BY sentiment
""")

# 7. Propose fixes based on findings
fixes = adk_propose_fix(
    test_name="accepted_values_sentiment",
    model_name="fact_reviews",
    column_name="sentiment",
    root_cause="New sentiment values from upgraded NLP system",
    fix_options=json.dumps([...])
)
```

---

## Configuration

Tool behavior can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DBT_PROJECT_PATH` | Path to dbt project | Required |
| `KNOWLEDGE_BASE_PATH` | Path to knowledge base | Required |
| `BIGQUERY_PROJECT_ID` | BigQuery project | Required for SQL |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | true |
| `CACHE_ENABLED` | Enable embedding cache | true |
