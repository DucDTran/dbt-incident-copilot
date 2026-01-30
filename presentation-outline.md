## dbt Co-Work – Presentation Outline

### 1. Key Features of the Technology

- **Multi-Agent Orchestration Architecture**
  - **Investigator Agent**: Gathers context via lineage, SQL analysis, and data queries
  - **Diagnostician Agent**: Analyzes findings to identify root cause and business impact
  - **Fix Proposer Agent**: Generates actionable, code-ready solutions
  - Prevents token limits and ensures focused, high-quality outputs at each stage

- **Deep Data & Business Context**
  - Reads dbt manifest for lineage and dependency graphs
  - Inspects model SQL, schema YAML, and test configurations
  - Integrates Elementary results (BigQuery) and local knowledge base of policies

- **Decision-Support Oriented Fix Recommendations**
  - Generates multiple fix options per incident with risk assessment
  - Each option includes **Pros**, **Cons**, and **When Appropriate**
  - **Actionable Resolutions**: "One-Click" Dry Run and Apply capabilities
  - **Diff Preview**: Side-by-side comparison of proposed changes before application

- **Transparent Investigation Steps & Traceability**
  - **Live Investigation Stream**: Visualizes the agent's thought process
  - **Detailed Steps**: Expandable view of every tool call, input, and output
  - **Performance Metrics**: Tracks execution time and token usage for each step
  - **Audit Trail**: Complete history of how the diagnosis was reached

- **Safe Execution & Observability**
  - Read-only SQL investigation with strict query limits
  - **Dry Run Verification**: Compiles dbt code to validate syntax before saving
  - **Repo Management**: Automatic backups of files before modification
  - **Elementary Integration**: Fallback robust querying for test failure data

---

### 2. Potential Use Cases

- **Data Quality Incident Response**
  - Not-null / accepted_values / unique test failures
  - Automatically identifies upstream causes and recommends fixes

- **Schema Evolution & New Business Logic**
  - New categories, product types, or states appearing in source data
  - Proposes schema/test updates aligned with business rules

- **Test Configuration Tuning**
  - Overly strict or noisy tests (e.g., severity levels, thresholds)
  - Suggests calibrated changes with pros/cons and impact notes

- **Pipeline Debugging & Regression Analysis**
  - Unexpected query results or broken transformations
  - Traces lineage, surfaces SQL + schema context, and diagnoses likely root cause

- **Onboarding, Enablement, and Audit Support**
  - New analytics engineers learning the dbt project
  - Auditors or data governance teams reviewing how incidents were resolved

---

### 3. Brief Technical Overview

- **Architecture at a Glance**
  - **Mission Control**: Dashboard for high-level incident overview and stats
  - **Resolution Studio (Labs)**: Interactive environment for deep-dive investigation
  - **Intelligence**: Google ADK-based Multi-Agent system (Investigator, Diagnostician, Fix Proposer)
  - **Data Context**: dbt project artifacts, BigQuery (Elementary), Markdown Knowledge Base

- **Core Agent Loop (Multi-Stage)**
  1. **Triaging**: Receive failing test details from Elementary
  2. **Investigation**:
     - Agent autonomously selects tools to read lineage, SQL, and Schema
     - Executes read-only queries to fetch failed rows (with robust fallback)
  3. **Diagnosis**: Synthesize findings into Root Cause, Evidence, and Impact
  4. **Solution Generation**: Propose fix options using `agentic_fix_tool`
  5. **Resolution**: User reviews, runs Dry Run, and Applies the fix

- **Implementation Highlights**
  - **Backend**: Python 3.10+, Google Gemini 2.0 Flash, BigQuery
  - **Frontend**: Streamlit with custom component styling
  - **Agent Tooling**: Specialized tools for:
     - `repo_tool`: Safe file reading/writing with backups
     - `elementary_tool`: Robust test result fetching
     - `agentic_fix_tool`: Intelligent schema/SQL modification
     - `knowledge_base_tool`: Semantic search for policies

- **Potential Improvements & Production Enhancements**
  - **CI/CD Integration**
    - Automatically trigger investigations on test failures in CI/CD pipelines
    - Auto-create pull requests with proposed fixes for review
    - Integrate with GitHub Actions, GitLab CI, or Jenkins
    - Enable automated fix application with approval workflows
  
  - **MCP Server Integration**
    - Connect to Confluence via MCP (Model Context Protocol) server
    - Pull business rules and policies directly from Confluence pages
    - Keep knowledge base synchronized with organizational documentation
    - Enable real-time updates without manual file management
  
  - **Production BigQuery Integration**
    - Replace mock Elementary data with real BigQuery queries
    - Direct connection to production data warehouse for live test results
    - Real-time data sampling and validation during investigations
    - Support for multiple BigQuery projects and datasets
  
  - **Additional Enhancements**
    - Multi-project support for organizations with multiple dbt projects
    - Slack/Teams notifications for critical incidents
    - Agent memory and learning from historical resolutions
    - Custom tool development framework for domain-specific integrations
    - Enhanced security with role-based access control
    - Performance optimizations for large-scale deployments
