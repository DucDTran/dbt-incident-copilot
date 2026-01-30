# ❓ dbt Co-Work: Frequently Asked Questions

This document captures key questions and answers to help you understand the **dbt Co-Work** project, its architecture, and its capabilities.

---

## 🌟 General Understanding

### Q: What is dbt Co-Work?
**A:** dbt Co-Work is an **Agentic AI Platform** designed to automate the resolution of analytics engineering incidents. Specifically, it focuses on diagnosing and fixing **dbt test failures** by autonomously investigating root causes, checking business rules, and proposing actionable code fixes.

### Q: How is this different from tools like GitHub Copilot?
**A:** While standard coding assistants focus on *code completion* and *syntax*, dbt Co-Work focuses on **investigation** and **data context**.
- **Autonomy:** It doesn't just suggest code as you type; it proactively investigates a failure by tracing lineage and querying data.
- **Context:** It understands the *data* (via BigQuery), not just the *code*.
- **Business Awareness:** It consults a semantic Knowledge Base to ensure fixes comply with business policies.

### Q: What problem does it solve?
**A:** It solves the "hunt-and-peck" nature of debugging data pipelines. Instead of an engineer manually tracing dependencies, running ad-hoc queries, and checking docs, dbt Co-Work does this automatically and presents a "One-Click" resolution screen.

---

## 🏗️ Architecture & Agents

### Q: How does the Multi-Agent architecture work?
**A:** The system handles complexity by splitting the workload across three specialized agents using the **Google ADK (Agent Development Kit)**:
1.  **🕵️ Investigator Agent:** Gathers facts. It reads lineage, SQL, schema, test details, and business rules. It *does not* diagnose.
2.  **🧠 Diagnostician Agent:** Analyzes the facts. It produces a structured diagnosis (Root Cause, Evidence, Impact, Severity).
3.  **🛠️ Fix Proposer Agent:** Takes the diagnosis and generates concrete fix options (SQL or Schema changes).

### Q: What tools are available to the agents?
**A:** The agents have access to a suite of 8+ specialized tools:
-   **`get_model_lineage`**: Traces upstream/downstream dependencies.
-   **`read_model_sql` / `read_schema_definition`**: inspections code files.
-   **`execute_sql`**: Runs queries against BigQuery (read-only) to verify data.
-   **`search_knowledge_base`**: Semantic search for business rules policies.
-   **`adk_propose_fix`**: The "action" tool to formalize fix recommendations.

---

## 💡 Features & Capabilities

### Q: How does the "Fix" generation work?
**A:** The **Fix Proposer Agent** generates 4-5 potential solutions. These can be:
-   **Schema Fixes:** Updating `schema.yml` (e.g., adding `accepted_values`, relaxing severity).
-   **SQL Fixes:** Modifying `model.sql` (e.g., adding `WHERE` clauses, `COALESCE` logic).

### Q: What happens if the Code Generation fails?
**A:** The system uses a robust fallback mechanism:
1.  **Deterministic Fixes:** First, it tries to apply the fix using code parsers (YAML/Regex).
2.  **Generative Fallback:** If that fails, it uses Gemini to essentially "rewrite" the file with the fix applied.
3.  **Manual Fallback:** If the AI fix also fails, the option is still presented to the user with a "Requires Manual Intervention" flag, so no idea is lost.

### Q: Does it handle dbt variables (`vars`)?
**A:** Yes. If a fix involves a value (like `minimum_nights`) that is defined as a project variable in `dbt_project.yml`, the agent is smart enough to proposed updating the **variable** itself rather than hardcoding the value in the schema file.

### Q: How does it ensure fixes are "Business Compliant"?
**A:** The **Investigator** tool retrieves relevant policies from the `knowledge_base/` folder (markdown files) based on the test failure context. The **Fix Proposer** is then explicitly instructed to prioritize fixes that align with these retrieved rules.

---

## 🚀 Usage & Operations

### Q: How do I run the project?
**A:**
1.  Ensure you have a `config.env` file (copied from `config.example.env`).
2.  Run the start script:
    ```bash
    ./run.sh
    ```
    Or manually:
    ```bash
    streamlit run app/main.py
    ```
3.  Access the web UI at `http://localhost:8501`.

### Q: What are the key configuration requirements?
**A:**
-   **Google API Key:** For Gemini models.
-   **BigQuery Credentials:** Service account for accessing data/metadata.
-   **dbt Project Path:** Local path to the dbt project you want to debug.
