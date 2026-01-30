"""
Fix recommendation prompts for the dbt Co-Work.

These prompts are used to generate and enhance fix recommendations.
"""

from typing import Dict, Any, List

def get_fix_enhancement_prompt(
    test_name: str,
    model_name: str,
    column_name: str,
    error_message: str,
    diagnosis: str,
    sql_excerpt: str,
    options_text: str,
) -> str:

    return f"""Based on this test failure analysis, enhance these fix options with specific, actionable recommendations.

## Test Failure
- Test: {test_name}
- Model: {model_name}
- Column: {column_name}
- Error: {error_message}

## Diagnosis
{diagnosis}

## SQL Code (excerpt)
```sql
{sql_excerpt}
```

## Available Fix Options
{options_text}

For each option, provide:
1. **Specific code change** - The exact SQL or YAML code to add/modify
2. **Why this works** - technical explanation
3. **Business impact** - who/what is affected
4. **Pros** - key advantages or strengths of this option
5. **Cons** - key tradeoffs, limitations, or risks
6. **When appropriate** - when this option is the best choice (conditions, scenarios, or constraints)

**IMPORTANT: Code Snippet Format**
- For SQL changes: Provide complete, executable SQL statements or specific clauses
  - WHERE clause: "WHERE column_name IS NOT NULL" or "WHERE column_name IN ('value1', 'value2')"
  - COALESCE: "COALESCE(column_name, 'default_value') AS column_name"
  - Column replacement: "COALESCE(column_name, 0) AS column_name"
- For YAML schema changes: Provide the complete test definition
  - Example: "- accepted_values:\n    arguments:\n      values: ['value1', 'value2', 'value3']"
- Be specific and include the exact code that should be added/modified

Format as JSON array with enhanced options:
```json
[
{{
    "id": "option_id",
    "title": "The clear action title",
    "description": "The description of what this fix does",
    "code_snippet": "The exact SQL or YAML code to add/modify",
    "technical_reason": "The technical reason why this solves the problem",
    "business_impact": "The business impact of this fix",
    "pros": "Key advantages or strengths of this option",
    "cons": "Key tradeoffs, limitations, or risks of this option",
    "when_appropriate": "When this option is the best choice (conditions, scenarios, or constraints)"
}}
]
```

Return ONLY the JSON array, no other text."""
