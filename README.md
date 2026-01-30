# 🚀 dbt Co-Work

**An Agentic AI Platform for Analytics Engineering Incident Resolution**

dbt Co-Work transforms pipeline failure resolution from a manual "hunt-and-peck" process into an automated, strategic workflow. It detects pipeline failures, investigates root causes using internal business context, and presents engineers with "One-Click" resolution options.

![dbt Co-Work](https://img.shields.io/badge/dbt-Co--Pilot-e94560?style=for-the-badge)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![Gemini 2.0](https://img.shields.io/badge/Gemini-2.0_Flash-green?style=for-the-badge)

## The Problem: The "Data Detective" Tax

Why do Analytics Engineers spend **30-50%** of their time debugging instead of building?

1.  **Complexity at Scale**: A single failure in a DAG of 500+ models requires tracing lineage across dozens of files to find the origin.
2.  **Context Switching Hell**: Debugging forces you to toggle between the Error Logs, your IDE (SQL), the Data Warehouse (Data), and Documentation/Slack (Context).
3.  **Brittle Fixes**: "Quick patches" aimed at silencing an alert often ignore broader business rules or upstream root causes, leading to recurring incidents (the "Whac-A-Mole" effect).
4.  **Operational Toil**: Senior engineers get bogged down in repetitive support tickets, blocking high-value strategic work.

**dbt Co-Work eliminates this tax.** It does the heavy lifting of investigation and diagnosis instantly, so engineers can simply review the findings and approve the solution.

## ✨ Key Technology Features

### 🤖 Autonomous AI Agent
- **Google ADK Integration**: Built on Google's Agent Development Kit (ADK) with Gemini 2.0 Flash for intelligent, autonomous decision-making
- **Multi-Tool Orchestration**: Seamlessly coordinates 8+ specialized tools (lineage analysis, SQL execution, knowledge base search, etc.)
- **Context-Aware Reasoning**: Maintains session context and understands model dependencies, business rules, and data patterns
- **Streaming Investigation**: Real-time step-by-step investigation display with transparent reasoning process

### 🔍 Advanced Investigation Capabilities
- **Model Lineage Analysis**: Automatically traces upstream dependencies and downstream impacts
- **SQL Code Analysis**: Reads and analyzes dbt model SQL to understand transformation logic
- **Schema Definition Parsing**: Extracts column definitions, constraints, and test configurations
- **Data Warehouse Queries**: Direct SQL execution (read-only) to verify actual data values and patterns
- **Business Context Integration**: Semantic search across knowledge base for relevant policies and rules

### 🛠️ Intelligent Fix Generation
- **Multi-Option Recommendations**: Generates 4-5 contextual fix options with risk assessment
- **Code Change Generation**: Produces ready-to-apply SQL and schema changes
- **Dry Run Simulation**: Validates fixes before application using dbt compile
- **Diff Visualization**: Side-by-side code comparison with syntax highlighting
- **Decision Guidance**: Pros, cons, and when-to-use recommendations for each fix option

### 📊 Real-Time Observability
- **Elementary Integration**: Connects to Elementary test results stored in BigQuery
- **Live Investigation Stream**: Watch the agent investigate in real-time with step-by-step updates
- **Investigation Steps**: View all tool calls with metadata, JSON responses, and execution time tracking
- **Comprehensive Context Panel**: Displays error messages, SQL code, schema definitions, and business rules
- **Complete Audit Trail**: Full investigation history with tool call details, responses, and timing information

## 🎯 Potential Use Cases

### 1. **Data Quality Incident Response**
**Scenario**: A `not_null` test fails on a critical dimension table
- **Agent Action**: Investigates upstream sources, checks for data pipeline issues, proposes data quality fixes
- **Value**: Reduces MTTR (Mean Time To Resolution) from hours to minutes

### 2. **Schema Evolution Management**
**Scenario**: New values appear in a column (e.g., new product categories)
- **Agent Action**: Identifies the source of new values, checks business rules, proposes schema updates or data filters
- **Value**: Automates the decision-making process for schema changes

### 3. **Test Configuration Tuning**
**Scenario**: Tests are too strict or too lenient for business needs
- **Agent Action**: Analyzes test results, consults data quality policies, suggests severity adjustments
- **Value**: Ensures tests align with business requirements without manual review

### 4. **Data Pipeline Debugging**
**Scenario**: A transformation produces unexpected results
- **Agent Action**: Traces lineage, queries actual data, identifies root cause in upstream models
- **Value**: Accelerates debugging by automatically gathering context

### 5. **Onboarding & Knowledge Transfer**
**Scenario**: New team members need to understand data quality standards
- **Agent Action**: Demonstrates investigation process, shows relevant business rules, explains fix rationale
- **Value**: Serves as an interactive learning tool for data engineering best practices

### 6. **Compliance & Audit Support**
**Scenario**: Need to document why certain data quality issues were handled in specific ways
- **Agent Action**: Provides complete investigation trail with business rule references
- **Value**: Creates audit-ready documentation automatically

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           dbt Co-Work                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Streamlit  │    │   ADK Agent  │    │     Knowledge Base       │  │
│  │   Dashboard  │◄──►│   (Gemini)   │◄──►│   (Business Rules)       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                                           │
│         │                   │                                           │
│         ▼                   ▼                                           │
│  ┌──────────────┐    ┌──────────────┐                                  │
│  │  Resolution  │    │  Tool Suite  │                                  │
│  │   Actions    │    │  - Manifest  │                                  │
│  │  - Dry Run   │    │  - Repo Read │                                  │
│  │  - Apply     │    │  - Elementary│                                  │
│  │  - Diff View │    │  - KB Search │                                  │
│  └──────────────┘    └──────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  dbt Project    │    │    BigQuery     │
│  (Local Files)  │    │  (Elementary)   │
└─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Core Engine** | dbt Core (Airbnb Analytics Project) |
| **Observability** | Elementary (test results to BigQuery) |
| **Intelligence** | Google ADK + Gemini 2.0 Flash |
| **Interface** | Streamlit (Decision Support Studio) |
| **Database** | BigQuery (with local mock for demo) |
| **RAG** | File Search via Knowledge Base |

## 🔬 Technical Overview

### Multi-Agent Architecture
The system uses **Google's Agent Development Kit (ADK)** with a **Multi-Agent Architecture** that splits investigation into three specialized agents to prevent output token truncation and provide better results:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ INVESTIGATOR │───►│ DIAGNOSTICIAN│───►│ FIX PROPOSER │      │
│  │    Agent     │    │    Agent     │    │    Agent     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                   │                    │               │
│        ▼                   ▼                    ▼               │
│  • Gathers context   • Analyzes findings  • Generates fixes    │
│  • Uses 8+ tools     • Identifies root    • Creates options    │
│  • Streams progress    cause              • Provides rationale │
│                      • Business impact                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why Multi-Agent?**
- **Prevents token truncation**: Each agent has a focused task with manageable output
- **Better separation of concerns**: Investigation → Analysis → Fix generation
- **Improved reliability**: Each agent can retry independently
- **Streaming visibility**: Users see real-time progress at each stage

### Agent Roles

| Agent | Role | Capabilities |
|-------|------|-------|
| **Investigator** | Gathers all relevant context | • Lineage & dependency analysis <br> • SQL & Schema inspection <br> • Data warehousing querying (BigQuery) <br> • Knowledge Base semantic search |
| **Diagnostician** | Analyzes findings, identifies root cause | • Root cause identification <br> • Business impact assessment <br> • Evidence synthesis |
| **Fix Proposer** | Generates fix options with rationale | • SQL/YAML code generation <br> • Fix strategy evaluation <br> • Pros/Cons analysis |

### Tool Ecosystem
The agents have access to a suite of 8+ specialized tools:

1. **`repo_tool`**: Safe file reading/writing with automatic backups and diff generation
2. **`elementary_tool`**: Robust fetching of test results and failed row samples from BigQuery
3. **`agentic_fix_tool`**: Intelligent generation of Schema YAML and SQL fixes (not templates)
4. **`knowledge_base_tool`**: Semantic search across markdown business rules and policies
5. **`manifest_tool`**: Lineage and node dependency extraction from dbt manifest
6. **`sql_tool`**: Read-only execution of queries with strict limits and safety checks

### Investigation Flow
```
1. Test Failure Detected → Elementary/BigQuery
2. User Clicks "Investigate" in Mission Control → Multi-Agent Pipeline Starts
3. INVESTIGATOR Agent:
   ├─ Analyze test failure details & failed row samples
   ├─ Read model lineage (upstream/downstream)
   ├─ Examine SQL transformation logic & Schema definitions
   ├─ Query actual data (with fallback to compiled SQL)
   └─ Search business rules knowledge base
4. DIAGNOSTICIAN Agent:
   ├─ Synthesizes gathered context
   ├─ Identifies root cause (Data vs. Code vs. Config)
   └─ Assesses downstream business impact
5. FIX PROPOSER Agent:
   ├─ Generates 3-5 distinct fix options
   ├─ Calculates Pros, Cons, and "When to use"
   └─ Prepares code changes (SQL/YAML)
6. Resolution Studio (UI):
   ├─ User reviews Diagnosis & Options
   ├─ "Dry Run": Compiles code to verify syntax
   └─ "Apply": Commits changes to the repo
```

### Knowledge Base Integration
- **Semantic Search**: Uses Gemini embeddings for context-aware document retrieval
- **Fallback Search**: Keyword-based search when embeddings unavailable
- **Business Context**: Maps test failures to relevant policies, rules, and playbooks
- **File-Based**: Simple Markdown files in `knowledge_base/` directory

### Security & Safety
- **Read-Only SQL**: All SQL queries are validated to prevent write operations
- **Query Limits**: Automatic LIMIT enforcement (default 100 rows, max 1000)
- **Timeout Protection**: 30-second query timeout prevents long-running queries
- **Data Processing Limits**: 10MB maximum bytes billed per query
- **Credential Isolation**: Service account credentials with minimal required permissions

## 📦 Quick Start

1. **Create Virtual Environment**
   ```bash
   cd /Users/duc.tran/dbt-copilot
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp config.example.env config.env
   # Edit config.env with your Gemini API key:
   # GOOGLE_API_KEY=your_key_here
   ```

4. **Run the Application**
   ```bash
   streamlit run app/main.py
   ```

5. **Open Dashboard**
   Navigate to: http://localhost:8501

## 🎯 Features

### A. Mission Control (Dashboard)
The command center for managing incidents.

- 📊 **Stats Overview**: Total failures, critical errors, warnings, affected models
- 🚨 **Active Incidents**: Sorted by severity and time
- 🔍 **Quick Actions**: Investigate or snooze incidents directly

### B. Resolution Studio (Labs)
Deep-dive into individual incidents with AI-powered investigation.

| Panel | Content |
|-------|---------|
| **Left** | **Context**: Error Log + SQL Code + Schema Definition |
| **Center** | **Intelligence**: AI Diagnosis + Traceable Investigation Steps |
| **Right** | **Resolution**: Decision Matrix with actionable Fix Options |

### C. AI-Recommended Fix Options
The agent generates contextual fix options based on:
- Test type and error message
- Model lineage and dependencies
- Business rules from knowledge base

Each option includes:
- **Pros & Cons**: Balanced assessment of the fix
- **Rationale**: Why the agent chose this solution
- **Risk Assessment**: High/Medium/Low impact analysis

**Example Options:**
- ✨ **Option A**: Update Logic (add new accepted values)
- 🔧 **Option B**: Data Quality Fix (filter bad rows)
- ⏸️ **Option C**: Snooze/Warn (known issue handling)

### D. Execution Actions
- 🧪 **Dry Run**: Compiles code to verify syntax before applying
- 📝 **Diff View**: Visual red/green code comparison
- ✅ **Apply**: Commits changes to local files (with automatic backup)

## 🔧 Agent Tools

The Multi-Agent system uses these specialized tools to interact with your project:

| Tool | Description | Use Case |
|------|-------------|----------|
| `manifest_tool` | Parse `target/manifest.json` | Understand model lineage and graph |
| `repo_tool` | Safe Read/Write/Diff | View code, apply fixes, backup files |
| `elementary_tool` | Query Elementary data | Get test results and failed rows (robust fallback) |
| `knowledge_base_tool` | Semantic Search | Find relevant business policies |
| `sql_tool` | Execute Read-Only SQL | Verify data in BigQuery (with limits) |
| `agentic_fix_tool` | Generative Fixer | Create valid SQL/YAML schema fixes (not templates) |

## 📚 Knowledge Base

The agent consults local Markdown files for business context:

```
knowledge_base/
├── data_quality_policies.md    # Data quality standards
├── business_rules.md           # Domain-specific logic
├── naming_conventions.md       # Column and model naming
├── privacy_policy.md           # PII handling guidelines
└── incident_playbook.md        # Response procedures
```

## 🔄 Injecting Test Failures

To test with real dbt failures:

```bash
# Add failing data to the Airbnb project
python scripts/inject_failing_data.py

# Run dbt tests to see failures
cd /Users/duc.tran/airbnb-dbt-project/dbt
dbt test

# Restore original data
python scripts/inject_failing_data.py --restore
```

## 🔮 Future Enhancements

- [ ] Slack/Teams notifications
- [ ] Git integration for PR creation and CI/CD
- [ ] Multi-project support
- [ ] BigQuery and dbt MCP for consistent and standardized tools calling
- [ ] Agent memory and learning
- [ ] Support for more data warehouses (Snowflake, Databricks)
- [ ] Automated regression testing after fixes

**Questions?** Open an issue or reach out to me @ dinhductran189@gmail.com.

