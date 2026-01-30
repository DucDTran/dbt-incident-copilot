"""Labs - Investigation and fix recommendation UI."""

import streamlit as st
import asyncio
import difflib
from datetime import datetime
from typing import Dict, Any, List
import re
import logging

# Import multi-agent copilot (preferred for better output handling)
from app.agent.multi_agent_copilot import (
    MultiAgentCopilot,
    Investigation,
    InvestigationStep,
    Diagnosis,
    InvestigationContext,
)

from app.agent.tools import (
    tool_read_repo,
    get_model_lineage,
)
from app.agent.tools.repo_tool import find_file_by_model_name, find_schema_file, get_file_diff
from app.agent.tools.dbt_tool import run_dry_run, run_dbt_test, check_dbt_installed
from app.agent.tools.elementary_tool import get_failed_rows_for_test
from app.ui.components import icon
import yaml
import pandas as pd

logger = logging.getLogger(__name__)



def go_to_dashboard():
    """Navigation callback to go to dashboard."""
    st.session_state.page = "home"
    st.session_state.sidebar_nav = "Dashboard"
    st.session_state.selected_incident = None
    st.session_state.fix_applied = False
    st.session_state.apply_result = None
    st.session_state.test_result = None
    clear_investigation_state()



def go_to_fixes():
    """Navigation callback to go to fixes page."""
    st.session_state.page = "fixes"
    # Ensure we stay on Labs page
    if "current_page" not in st.session_state or st.session_state.current_page != "Labs":
        st.session_state.current_page = "Labs"


def go_back_to_investigation():
    """Navigation callback to go back to investigation view."""
    st.session_state.page = "resolution"
    # Ensure we stay on Labs page
    if "current_page" not in st.session_state or st.session_state.current_page != "Labs":
        st.session_state.current_page = "Labs"


def restart_investigation():
    """Callback to restart investigation."""
    clear_investigation_state()


def render_resolution_studio():
    """Render the Labs (detail) view."""
    
    # Check if we have a selected incident
    if "selected_incident" not in st.session_state or st.session_state.selected_incident is None:
        st.info("Select an incident from Dashboard to investigate")
        st.button("Back to Dashboard", on_click=go_to_dashboard, icon=':material/arrow_back:')
        return
    
    incident = st.session_state.selected_incident
    
    # Auto-clear state if we switched incidents
    current_test_id = incident.get("test_id")
    if "investigating_test_id" in st.session_state:
        if st.session_state.investigating_test_id != current_test_id:
            clear_investigation_state()
            st.session_state.investigating_test_id = current_test_id
    else:
        st.session_state.investigating_test_id = current_test_id
    
    # Header
    st.divider()
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Back", icon=':material/arrow_back:', on_click=go_to_dashboard)
    
    with col2:
        model = incident.get('model_name', 'Unknown')
        column = incident.get('column_name', '')
        st.markdown(f"#### Investigating: {model}.{column}", unsafe_allow_html=True, text_alignment='left')
    
    st.divider()
    
    # Two-panel layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_context_panel(incident)
    
    with col_right:
        render_investigation_panel(incident)


def clear_investigation_state():
    """Clear all investigation-related session state."""
    keys_to_clear = [
        "investigation", "investigation_steps", "diagnosis", 
        "fix_options", "selected_fix", "show_diff", "dry_run_result",
        "investigation_context", "show_fixes", "run_test_result"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def extract_diagnosis_conclusion(full_diagnosis: str) -> str:
    """
    Extract the diagnosis conclusion from the agent response.
    
    Returns the complete diagnosis with Root Cause, Evidence, and Impact Assessment.
    Returns empty string with logging if diagnosis is incomplete.
    """
    if not full_diagnosis:
        return ""
    
    # Clean up any legacy <final_diagnosis> tags if they exist (but don't rely on them)
    full_diagnosis = re.sub(r'</?final_diagnosis>', '', full_diagnosis, flags=re.IGNORECASE).strip()
    
    # Fallback: Try to find diagnosis sections without tags
    # This is less reliable but better than nothing
    root_cause_match = re.search(
        r'#{1,3}\s*Root\s*Cause\s*\n(.*?)(?=#{1,3}\s*|\Z)',
        full_diagnosis,
        re.IGNORECASE | re.DOTALL
    )
    
    if not root_cause_match:
        print("WARNING: Could not find <final_diagnosis> tags or Root Cause section in agent response")
        print("Agent may not have completed diagnosis section")
        return ""
    
    # Extract all diagnosis sections
    sections = {}
    section_patterns = {
        'root_cause': r'#{1,3}\s*Root\s*Cause\s*\n(.*?)(?=#{1,3}|\Z)',
        'evidence': r'#{1,3}\s*Evidence\s*\n(.*?)(?=#{1,3}|\Z)',
        'impact': r'#{1,3}\s*Impact\s*Assessment\s*\n(.*?)(?=#{1,3}|\Z)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, full_diagnosis, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
    
    if not sections:
        print("WARNING: Found diagnosis sections but they are incomplete")
        return ""
    
    # Reconstruct diagnosis
    diagnosis = ""
    if sections.get('root_cause'):
        diagnosis += f"### Root Cause\n{sections['root_cause']}\n\n"
    if sections.get('evidence'):
        diagnosis += f"### Evidence\n{sections['evidence']}\n\n"
    if sections.get('impact'):
        diagnosis += f"### Impact Assessment\n{sections['impact']}\n"
    
    return diagnosis.strip()


def extract_step_contents(full_diagnosis: str) -> Dict[str, str]:
    """
    Extract step-by-step content from the agent's output.
    
    Returns a dict mapping step numbers and titles to their content.
    Example: {"step_1": "The test checks...", "step_2": "Found 3 upstream..."}
    """
    if not full_diagnosis:
        return {}
    
    step_contents = {}
    
    # Pattern to match "Step N: Title" or "**Step N: Title**"
    # This pattern is more flexible with whitespace
    step_pattern = r'(?:\*\*)?Step\s+(\d+)[:\s]+([^\*\n]+?)(?:\*\*)?\s*\n(.*?)(?=(?:\*\*)?Step\s+\d+|<final_diagnosis>|$)'
    
    matches = re.finditer(step_pattern, full_diagnosis, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        step_num = match.group(1)
        step_title = match.group(2).strip()
        step_content = match.group(3).strip()
        
        if step_content:  # Only add if there's actual content
            # Use both number and title for reference
            step_contents[f"step_{step_num}"] = step_content
            
            # Also add a mapped version by type
            title_lower = step_title.lower()
            
            if any(word in title_lower for word in ["understand", "test", "analyze"]):
                step_contents["analyze"] = step_content
            elif any(word in title_lower for word in ["lineage", "dependencies", "upstream"]):
                step_contents["lineage"] = step_content
            elif any(word in title_lower for word in ["sql", "code", "transformation"]):
                step_contents["sql"] = step_content
            elif any(word in title_lower for word in ["schema", "definition", "configuration"]):
                step_contents["schema"] = step_content
            elif any(word in title_lower for word in ["knowledge", "business", "rules", "policies"]):
                step_contents["knowledge"] = step_content
    
    return step_contents


def generate_schema_excerpt(model_def: Dict[str, Any], column_name: str = "") -> str:
    """
    Generate a YAML excerpt showing only the relevant model and column.
    
    Args:
        model_def: The model definition dictionary from schema.yml
        column_name: The column to highlight (optional)
        
    Returns:
        YAML-formatted string excerpt
    """
    if not model_def:
        return "# No model definition found"
    
    # Build a focused excerpt
    excerpt_lines = []
    
    # Model header
    model_name = model_def.get("name", "unknown")
    excerpt_lines.append(f"- name: {model_name}")
    
    # Model description if exists
    if model_def.get("description"):
        desc = model_def["description"][:100] + "..." if len(model_def.get("description", "")) > 100 else model_def.get("description", "")
        excerpt_lines.append(f'  description: "{desc}"')
    
    # Columns section
    columns = model_def.get("columns", [])
    if columns:
        excerpt_lines.append("  columns:")
        
        # Find the target column and show it with context
        target_col = None
        other_cols = []
        
        for col in columns:
            if col.get("name") == column_name:
                target_col = col
            else:
                other_cols.append(col)
        
        # Show other columns briefly
        if other_cols:
            excerpt_lines.append(f"    # ... {len(other_cols)} other column(s) ...")
        
        # Show the target column in full
        if target_col:
            excerpt_lines.append("")
            excerpt_lines.append(f"    # ========== TARGET COLUMN ==========")
            excerpt_lines.append(f"    - name: {target_col.get('name', 'unknown')}")
            
            if target_col.get("description"):
                excerpt_lines.append(f'      description: "{target_col.get("description")}"')
            
            if target_col.get("data_type"):
                excerpt_lines.append(f"      data_type: {target_col.get('data_type')}")
            
            # Show tests in detail
            tests = target_col.get("tests", [])
            if tests:
                excerpt_lines.append("      tests:")
                for test in tests:
                    if isinstance(test, str):
                        excerpt_lines.append(f"        - {test}")
                    elif isinstance(test, dict):
                        for test_name, test_config in test.items():
                            excerpt_lines.append(f"        - {test_name}:")
                            if isinstance(test_config, dict):
                                for key, val in test_config.items():
                                    if isinstance(val, list):
                                        excerpt_lines.append(f"            {key}:")
                                        for item in val:
                                            excerpt_lines.append(f"              - {item}")
                                    else:
                                        excerpt_lines.append(f"            {key}: {val}")
            excerpt_lines.append(f"    # ======================================")
    
    return '\n'.join(excerpt_lines)


def get_step_content(step_action: str, step_contents: Dict[str, str]) -> str:
    """Get the extracted content for a specific step based on its action name."""
    action_lower = step_action.lower()
    
    # Map step actions to content keys - check for tool names too
    if "analyzing" in action_lower or "test failure" in action_lower:
        return step_contents.get("analyze", "")
    elif "lineage" in action_lower or "get_model_lineage" in action_lower:
        return step_contents.get("lineage", "")
    elif "sql" in action_lower or "model sql" in action_lower or "read_model_sql" in action_lower or "execute_sql" in action_lower or "adk_execute_sql" in action_lower:
        return step_contents.get("sql", "")
    elif "schema" in action_lower or "read_schema" in action_lower:
        return step_contents.get("schema", "")
    elif "knowledge" in action_lower or "business" in action_lower or "search_knowledge" in action_lower or "search_knowledge_base" in action_lower:
        return step_contents.get("knowledge", "")
    elif "propose_fix" in action_lower or "adk_propose_fix" in action_lower:
        return step_contents.get("fixes", "")
    elif "diagnosis" in action_lower:
        return step_contents.get("diagnosis", "")
    elif "file" in action_lower or "read_file" in action_lower:
        return step_contents.get("file", "")
    elif "test details" in action_lower or "get_test_details" in action_lower:
        return step_contents.get("test_details", "")
    
    return ""


def render_context_panel(incident: Dict[str, Any]):
    """Render the left panel with context information."""

    with st.expander("Test Context", expanded=True):
        test_description = incident.get('test_description') or incident.get('test_name', 'Unknown')
        st.markdown(f"**{icon('info', 18)} Test Description**: `{test_description}`", unsafe_allow_html=True)
        st.markdown(f"**{icon('table_chart', 18)} Table Name**: `{incident.get('table_name', 'Unknown')}`", unsafe_allow_html=True)
        st.markdown(f"**{icon('density_small', 18)} Column Name**: `{incident.get('column_name', 'Unknown')}`", unsafe_allow_html=True)
        st.markdown(f"**{icon('category', 18)} Test Category**: `{incident.get('test_short_name', 'Unknown')}`", unsafe_allow_html=True)

    
    # Error Log
    with st.expander("Error Log", expanded=True):
        error_msg = incident.get("error_message", "No error message available")
        st.error(error_msg)
    
    # Model SQL
    with st.expander("Model SQL", expanded=True):
        model_name = incident.get("model_name", "")
        
        if model_name:
            file_result = find_file_by_model_name(model_name)
            
            if file_result["status"] == "success":
                sql_result = tool_read_repo(file_result["data"]["path"])
                
                if sql_result["status"] == "success":
                    st.code(sql_result["data"]["content"], language="sql", wrap_lines=True)
                else:
                    st.warning(f"Could not read SQL: {sql_result['message']}")
            else:
                st.warning(f"Model file not found: {file_result['message']}")
    
    # Schema Definition - Show only the relevant model and column excerpt
    with st.expander("Schema Definition", expanded=True):
        model_name = incident.get("model_name", "")
        column_name = incident.get("column_name", "")
        
        if model_name:
            schema_result = find_schema_file(model_name)
            
            if schema_result["status"] == "success":
                model_def = schema_result["data"].get("model_definition", {})
                
                # Generate focused YAML excerpt
                excerpt = generate_schema_excerpt(model_def, column_name)
                
                st.caption(f"Excerpt from `{schema_result['data'].get('relative_path', 'schema.yml')}`")
                st.code(excerpt, language="yaml", wrap_lines=True)
                
                # Option to view full schema
                with st.expander("View Full Schema", expanded=False):
                    yaml_result = tool_read_repo(schema_result["data"]["path"])
                    if yaml_result["status"] == "success":
                        st.code(yaml_result["data"]["content"], language="yaml", wrap_lines=True)
            else:
                st.warning("Schema file not found")


def render_investigation_panel(incident: Dict[str, Any]):
    """Render the right panel with agent investigation."""
    
    # Display Failed Sample Rows above Agent Investigation
    test_id = incident.get("test_id")
    failed_rows = []
    total_count = 0
    
    if test_id:
        failed_rows_result = get_failed_rows_for_test(test_id, limit=10)
        
        if failed_rows_result["status"] == "success":
            failed_rows = failed_rows_result["data"].get("failed_rows", [])
            total_count = failed_rows_result["data"].get("total_count", 0)
    
    # Always show the Failed Sample Rows expander
    if failed_rows:
        expander_label = f"Failed Sample Rows"
    else:
        expander_label = "Failed Sample Rows"
    
    with st.expander(expander_label, expanded=True):
        if failed_rows:
            # Convert to DataFrame for better display
            try:
                df = pd.DataFrame(failed_rows)
                # Replace None/NaN values with empty string for better display
                df = df.fillna('')
                
                # Style the dataframe with JetBrains Mono font and render as HTML
                styled_df = df.style.set_table_styles([
                    {
                        'selector': 'th, td',
                        'props': [
                            ('font-family', "'JetBrains Mono', monospace"),
                            ('font-size', '13px'),
                            ('padding', '8px'),
                            ('text-align', 'left')
                        ]
                    },
                    {
                        'selector': 'table',
                        'props': [
                            ('font-family', "'JetBrains Mono', monospace"),
                            ('font-size', '13px'),
                            ('border-collapse', 'collapse'),
                            ('width', '100%'),
                            ('max-width', '100%'),
                            ('table-layout', 'auto')
                        ]
                    },
                    {
                        'selector': 'thead th',
                        'props': [
                            ('background-color', '#ff683b'),
                            ('font-weight', '600')
                        ]
                    }
                ]).set_table_attributes('class="dataframe"')
                
                # Render as HTML wrapped in a container div for proper containment
                html = styled_df.to_html(escape=False, index=False)
                # Wrap in a container div with overflow handling
                # Use proper string formatting to avoid escaping issues
                wrapped_html = (
                    '<div style="width: 100%; overflow-x: auto; max-width: 100%;">'
                    '<style>'
                    '.dataframe {'
                    'width: 100% !important;'
                    'max-width: 100% !important;'
                    'font-family: "JetBrains Mono", monospace !important;'
                    'font-size: 13px !important;'
                    '}'
                    '.dataframe th, .dataframe td {'
                    'font-family: "JetBrains Mono", monospace !important;'
                    'font-size: 13px !important;'
                    'white-space: nowrap;'
                    'overflow: hidden;'
                    'text-overflow: ellipsis;'
                    'max-width: 300px;'
                    '}'
                    '</style>'
                    + html +
                    '</div>'
                )
                st.markdown(wrapped_html, unsafe_allow_html=True)
            except Exception as e:
                # If DataFrame conversion fails, show as JSON
                st.json(failed_rows[:10])  # Show first 10 rows
                if len(failed_rows) > 10:
                    st.caption(f"Showing first 10 of {len(failed_rows)} rows")
                st.caption(f"DataFrame error: {str(e)}")
        else:
            # Display empty datatable with styling
            empty_df = pd.DataFrame()
            styled_empty_df = empty_df.style.set_table_styles([
                {
                    'selector': 'th, td',
                    'props': [
                        ('font-family', "'JetBrains Mono', monospace"),
                        ('font-size', '13px')
                    ]
                },
                {
                    'selector': 'table',
                    'props': [
                        ('width', '100%'),
                        ('max-width', '100%')
                    ]
                }
            ])
            html = styled_empty_df.to_html(escape=False, index=False)
            wrapped_html = (
                '<div style="width: 100%; overflow-x: auto; max-width: 100%;">'
                '<style>'
                '.dataframe {'
                'width: 100% !important;'
                'max-width: 100% !important;'
                'font-family: "JetBrains Mono", monospace !important;'
                '}'
                '</style>'
                + html +
                '</div>'
            )
            st.markdown(wrapped_html, unsafe_allow_html=True)
    
    st.markdown("**Multi-Agent Investigation**")
    
    # Agent badge mapping
    AGENT_BADGES = {
        "investigator": (":material/motion_play:", "Investigator"),
        "diagnostician": (":material/medical_services:", "Diagnostician"),
        "fix_proposer": (":material/build:", "Fix Proposer"),
    }
    
    # Initialize state
    if "investigation" not in st.session_state:
        st.session_state.investigation = None
        st.session_state.investigation_steps = []
        st.session_state.diagnosis = None
        st.session_state.investigation_context = {}
    
    if "show_fixes" not in st.session_state:
        st.session_state.show_fixes = False
    
    # Not started yet
    if st.session_state.investigation is None:
        
        if st.button("Start Investigation", type="primary", use_container_width=True, icon=':material/play_circle:'):
            # Run investigation with live streaming display
            run_investigation_with_live_display(incident)
    else:
        # Investigation is complete; show a summary of steps (without internal thinking)
        steps = st.session_state.investigation_steps
        display_steps = [s for s in steps if "fix" not in s.get("action", "").lower() and "recommendation" not in s.get("action", "").lower()]
        
        success_count = sum(1 for s in display_steps if s.get("status") == "success")
        error_count = sum(1 for s in display_steps if s.get("status") == "error")
        
        # Count steps by agent
        agent_counts = {}
        for s in display_steps:
            agent = s.get("agent", "")
            if agent:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Get total investigation time if available
        total_time = st.session_state.get("investigation_total_time")
        time_display = ""
        if total_time is not None:
            if total_time < 60:
                time_display = f"Total time: {total_time:.1f}s"
            else:
                minutes = int(total_time // 60)
                seconds = total_time % 60
                time_display = f"Total time: {minutes}m {seconds:.1f}s"
        
        with st.expander(f"Investigation Steps ({success_count} success, {error_count} errors)\n\n{time_display}", expanded=False):
            # Show agent summary
            if agent_counts:
                agent_summary = " • ".join([
                    f"{AGENT_BADGES.get(a, ('🤖', a))[0]} {AGENT_BADGES.get(a, ('🤖', a))[1]}: {c}"
                    for a, c in agent_counts.items()
                ])
                st.caption(f"**Agents:** {agent_summary}")
                st.divider()
            
            for step in display_steps:
                status = step.get("status", "thinking")
                status_icon = {"success": ":material/check_circle:", "error": ":material/error:", "thinking": ":material/hourglass_empty:", "warning": ":material/warning:"}.get(status, ":material/hourglass_empty:")
                action = step.get("action", "Unknown step")
                tool = step.get("tool_name") or ""
                output = step.get("output_summary") or ""
                timestamp = step.get("timestamp") or ""
                details = step.get("details") or {}
                time_used = details.get("time_used_seconds")
                agent_name = step.get("agent", "")
                
                # Get agent badge
                badge_emoji, badge_name = AGENT_BADGES.get(agent_name, ("", ""))
                
                # Show agent badge if available
                if badge_emoji:
                    st.markdown(f"{badge_emoji} **[{badge_name}]** {status_icon} {action}")
                else:
                    st.markdown(f"{status_icon} **{action}**")
                
                meta_bits = []
                if tool:
                    meta_bits.append(f"Tool: `{tool}`")
                if timestamp:
                    meta_bits.append(f"Time: {timestamp}")
                if time_used is not None:
                    meta_bits.append(f"Time used: {time_used}s")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))
                
                if output and "received response" not in str(output).lower():
                    st.caption(f"Response: {output}")
                
                # For tool calls, show the full response as JSON if available
                if tool and details:
                    import json
                    raw = details.get("raw_response")
                    response_text = details.get("response_text")
                    json_obj = None
                    if isinstance(raw, dict):
                        json_obj = raw
                    elif isinstance(response_text, str):
                        try:
                            json_obj = json.loads(response_text)
                        except Exception:
                            json_obj = {"raw_response": response_text}
                    
                    if json_obj is not None:
                        with st.expander("Response", expanded=False):
                            st.json(json_obj)
            
            # Display total investigation time after all steps
            total_time = st.session_state.get("investigation_total_time")
            if total_time is not None:
                if total_time < 60:
                    st.info(f"**Total investigation time:** {total_time:.1f} seconds", icon=':material/avg_time:')
                else:
                    minutes = int(total_time // 60)
                    seconds = total_time % 60
                    st.info(f"**Total investigation time:** {minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} seconds", icon=':material/avg_time:')
        
        # Then show diagnosis (extract just the conclusion part)
        if st.session_state.diagnosis:
            diagnosis_text = extract_diagnosis_conclusion(st.session_state.diagnosis)
            
            # If extraction failed, try a more aggressive cleanup
            if not diagnosis_text or len(diagnosis_text.strip()) < 10:
                import re
                # Try to find any section that looks like diagnosis
                # Remove all Step patterns first
                cleaned = re.sub(r'(?:\*\*)?Step\s*\d+[:\s].*?(?=\n|$)', '', st.session_state.diagnosis, flags=re.IGNORECASE | re.MULTILINE)
                # Remove propose_fix references
                cleaned = re.sub(r'propose_fix.*?$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
                # Remove tool call patterns
                cleaned = re.sub(r'Calling\s+\w+.*?$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
                cleaned = cleaned.strip()
                
                # If we still have substantial content, use it
                if cleaned and len(cleaned) > 20:
                    diagnosis_text = cleaned
                else:
                    # Last resort: use original but try to find diagnosis-like content
                    # Look for any markdown headers or bold sections
                    diagnosis_sections = re.findall(r'(?:###?\s*|\*\*)[^\n]*(?:Root|Cause|Evidence|Impact|Diagnosis)[^\n]*(?:\*\*)?.*?(?=(?:###?\s*|\*\*)[^\n]*(?:Root|Cause|Evidence|Impact|Diagnosis)|$)', st.session_state.diagnosis, re.IGNORECASE | re.DOTALL)
                    if diagnosis_sections:
                        diagnosis_text = '\n\n'.join(diagnosis_sections).strip()
                    else:
                        diagnosis_text = st.session_state.diagnosis
            
            if diagnosis_text and len(diagnosis_text.strip()) > 10:
                st.success("AI Diagnosis (powered by Gemini)")
                st.markdown(diagnosis_text)
            else:
                st.warning("Diagnosis was generated but appears to be empty or could not be extracted.")
                # Show raw diagnosis in expander for debugging
                with st.expander("View raw diagnosis (for debugging)"):
                    st.text(st.session_state.diagnosis[:1000] if st.session_state.diagnosis else "No diagnosis available")
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("See Recommended Fixes", type="primary", use_container_width=True, on_click=go_to_fixes)
            with col2:
                st.button("Re-investigate", use_container_width=True, on_click=restart_investigation)
        else:
            st.warning("No diagnosis generated. Try re-investigating.")
            st.button("Re-investigate", use_container_width=True, on_click=restart_investigation)


def run_investigation_with_live_display(incident: Dict[str, Any]):
    """
    Run investigation with streaming display using MultiAgentCopilot.
    
    Uses the multi-agent architecture:
    1. Investigator Agent - gathers context using tools
    2. Diagnostician Agent - analyzes and produces diagnosis
    3. Fix Proposer Agent - generates fix recommendations
    """
    # Use MultiAgentCopilot for better token management
    agent = MultiAgentCopilot()
    steps = []
    diagnosis = None
    diagnosis_obj = None  # Store Diagnosis object
    fix_options = []
    context = {}
    
    # Track investigation start time
    investigation_start_time = datetime.now()
    
    # Agent badge colors for visual distinction
    AGENT_BADGES = {
        "investigator": ("", "Investigator"),
        "diagnostician": ("", "Diagnostician"),
        "fix_proposer": ("", "Fix Proposer"),
    }
    
    # Use st.status for live updating container
    with st.status("🚀 Starting multi-agent investigation...", expanded=True) as status:
        
        async def run_multi_agent():
            nonlocal diagnosis, diagnosis_obj, fix_options, context
            
            async for step in agent.investigate(incident, stream_steps=True):
                # Get agent badge
                agent_name = getattr(step, 'agent', None) or ""
                badge_emoji, badge_name = AGENT_BADGES.get(agent_name, ("🤖", "Agent"))
                
                step_data = {
                    "action": step.action,
                    "tool_name": step.tool_name,
                    "output_summary": step.output_summary,
                    "status": step.status,
                    "timestamp": step.timestamp.strftime("%H:%M:%S"),
                    "details": step.details,
                    "agent": agent_name,
                }
                steps.append(step_data)
                
                # Display step in real-time
                status_icon = {
                    "success": ":material/check_circle:",
                    "error": ":material/error:",
                    "thinking": ":material/hourglass_empty:",
                    "warning": ":material/warning:",
                }.get(step.status, ":material/hourglass_empty:")
                
                # Update the status label based on agent phase
                if "investigator" in agent_name.lower():
                    status.update(label=f":material/search: Investigator gathering context...", state="running")
                elif "diagnostician" in agent_name.lower():
                    status.update(label=f":material/medical_services: Diagnostician analyzing findings...", state="running")
                elif "fix_proposer" in agent_name.lower():
                    status.update(label=f":material/build: Fix Proposer generating options...", state="running")
                elif "diagnosis" in step.action.lower():
                    status.update(label="Generating diagnosis...", state="running")
                elif "fix" in step.action.lower():
                    status.update(label="Generating fix recommendations...", state="running")
                else:
                    status.update(label=f"{status_icon} {step.action}", state="running")
                
                # Write step details with agent badge
                if agent_name:
                    st.write(f"{badge_emoji} **[{badge_name}]** {status_icon} {step.action}")
                else:
                    st.write(f"{status_icon} **{step.action}**")
                    
                if step.tool_name:
                    st.caption(f"Tool: `{step.tool_name}`")
                if step.output_summary:
                    st.caption(step.output_summary)
            
            # Get the completed investigation
            test_id = incident.get("test_id", "unknown")
            investigation = agent.get_investigation(test_id)
            
            if investigation:
                # Handle new Diagnosis dataclass
                if isinstance(investigation.diagnosis, Diagnosis):
                    diagnosis_obj = investigation.diagnosis
                    diagnosis = diagnosis_obj.to_text()
                else:
                    diagnosis = investigation.diagnosis
                    
                fix_options = investigation.fix_options
                context = investigation.raw_context if hasattr(investigation, 'raw_context') else {}
        
        asyncio.run(run_multi_agent())
        
        # Mark as complete
        status.update(label=":material/check_circle: Investigation complete!", state="complete", expanded=False)
    
    # Calculate total investigation time
    investigation_end_time = datetime.now()
    total_duration = (investigation_end_time - investigation_start_time).total_seconds()
    
    # Store in session state
    st.session_state.investigation = True
    st.session_state.investigation_steps = steps
    st.session_state.diagnosis = diagnosis
    st.session_state.fix_options = fix_options if fix_options else []
    st.session_state.investigation_context = context
    st.session_state.investigation_total_time = total_duration
    
    # Rerun to show the final results
    st.rerun()


def render_fixes_page():
    """Render the AI-Recommended Fixes page."""
    
    # Check if we have investigation data
    if "selected_incident" not in st.session_state or st.session_state.selected_incident is None:
        st.info("No incident selected. Go back to Dashboard.")
        st.button("Back to Dashboard", on_click=go_to_dashboard, icon=':material/arrow_back:')
        return
    
    incident = st.session_state.selected_incident
    
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("Back", icon=':material/arrow_back:', on_click=go_back_to_investigation)
    
    with col2:
        model = incident.get('model_name', 'Unknown')
        column = incident.get('column_name', '')
        st.markdown(f"#### AI Recommended fixes for: {model}.{column}", unsafe_allow_html=True, text_alignment='left')
    
    st.divider()
    
    # Check if we have fix options
    if "fix_options" not in st.session_state or not st.session_state.fix_options:
        st.warning("No fix options available.")
        
        # Check investigation steps to provide more context
        investigation_steps = st.session_state.get("investigation_steps", [])
        fix_tool_steps = [s for s in investigation_steps if "adk_propose_fix" in (s.get("tool_name") or "").lower() or "propose_fix" in (s.get("tool_name") or "").lower() or "fix" in (s.get("action") or "").lower()]
        
        if investigation_steps:
            st.info("Investigation was completed, but no fix options were generated. This could mean:")
            st.markdown("""
            - The agent did not call the `adk_propose_fix` tool during investigation
            - The agent called `adk_propose_fix` but the response was invalid or couldn't be parsed
            - The investigation determined that no automated fixes are available
            
            Check the investigation steps for more details.
            """)
            
            # Show relevant steps
            if fix_tool_steps:
                with st.expander("Fix Generation Steps", expanded=True):
                    for step in fix_tool_steps:
                        status_icon = {"success": ":material/check_circle:", "error": ":material/error:", "warning": ":material/warning:", "thinking": ":material/hourglass_empty:"}.get(step.get("status", "thinking"), ":material/hourglass_empty:")
                        st.markdown(f"{status_icon} **{step.get('action', 'Unknown')}**")
                        st.caption(step.get('output_summary', ''))
                        if step.get('details'):
                            st.json(step['details'])
        else:
            st.info("Please run the investigation first.")
        
        if st.button("Go to Investigation"):
            st.session_state.page = "resolution"
            st.rerun()
        return
    
    options = st.session_state.fix_options
    
    # AI attribution
    st.info(f"Gemini AI generated {len(options)} fix option(s) based on the investigation")
    
    # Initialize selected option
    if "selected_fix" not in st.session_state:
        st.session_state.selected_fix = None
    
    # Two column layout for Options and Details
    col_options, col_details = st.columns([1, 1])
    
    with col_options:
        st.markdown("### Available Options")
        
        for idx, option in enumerate(options):
            title = option.get('title', 'Unknown')
            description = option.get('description', '')
            pros = option.get("pros", "")
            cons = option.get("cons", "")
            when_appropriate = option.get("when_appropriate", "")
            is_selected = st.session_state.selected_fix == idx
            
            # Create a bordered container for each option
            with st.container(border=True):
                if is_selected:
                    st.success(f"**Option {chr(65 + idx)}: {title}**")
                else:
                    st.markdown(f"**Option {chr(65 + idx)}: {title}**")
                
                st.caption(description)
                
                if st.button(f"Select Option {chr(65 + idx)}", key=f"fix_select_{idx}", use_container_width=True):
                    st.session_state.selected_fix = idx
                    st.session_state.show_diff = False
                    st.session_state.dry_run_result = None
                    st.rerun()
    
    with col_details:
        st.markdown("### Option Details")
        
        if st.session_state.selected_fix is not None:
            selected = options[st.session_state.selected_fix]
            
            st.info(f"Selected: Option {chr(65 + st.session_state.selected_fix)} - {selected.get('title', '')}")
            
            # Pros, Cons, and When Appropriate
            pros = selected.get("pros", "")
            cons = selected.get("cons", "")
            when_appropriate = selected.get("when_appropriate", "")
            
            if pros or cons or when_appropriate:
                with st.container(border=True):
                    if pros:
                        st.markdown(f"**:material/check: Pros:** {pros}")
                    if cons:
                        st.markdown(f"**:material/close: Cons:** {cons}")
                    if when_appropriate:
                        st.markdown(f"**:material/info: When Appropriate:** {when_appropriate}")
            
            # AI Analysis
            ai_reason = selected.get("ai_technical_reason") or selected.get("rationale", "")
            ai_impact = selected.get("ai_business_impact", "")
            ai_code = selected.get("ai_code_snippet", "")
            code_changes = selected.get("code_changes", {})
            
            if ai_reason:
                st.markdown("**Why this works:**")
                st.markdown(ai_reason)
            
            if ai_impact:
                st.markdown("**Business Impact:**")
                st.markdown(ai_impact)
            
            # Show code snippet (AI-generated or pre-computed)
            # Show code content (Ready-to-apply)
            if code_changes:
                with st.expander("**Suggested Code (Ready to Apply):**", expanded=False):
                    st.info(":material/lightbulb: Full file content that will be applied.")
                    for file_path, content in code_changes.items():
                        st.caption(f"File: `{file_path}`")
                        st.code(content, language="yaml" if ".yml" in file_path else "sql", wrap_lines=True)
            elif ai_code:
                with st.expander("**Suggested Code (Snippet):**", expanded=False):
                    st.code(ai_code, language="sql" if "sql" in ai_code.lower() or "select" in ai_code.lower() else "yaml", wrap_lines=True)
            
            # Business justification
            if selected.get("business_justification"):
                with st.expander("Business Justification"):
                    st.markdown(selected["business_justification"])
        else:
            st.info("Select an option from the left to see details")
    
    # ============================================================
    # Action Buttons and Preview Section (Full Width - Below Options)
    # ============================================================
    
    if st.session_state.selected_fix is not None:
        selected = options[st.session_state.selected_fix]
        
        st.divider()
        
        # Action buttons in full width
        st.markdown("### Actions")
        
        # Helper to generate/show diff
        def show_diff_preview():
            st.session_state.show_diff = not st.session_state.get("show_diff", False)

        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
             if st.button("Preview Impact", use_container_width=True, icon=":material/analytics:", help="Analyze the impact of this fix on data and downstream models"):
                 render_preview_impact(selected, incident)
                 
        with col2:
             if st.button("Run Test", use_container_width=True, icon=":material/lab_research:", help="Apply fix temporarily and run dbt test to verify it works"):
                 render_run_test(selected, incident)
                 
        with col3:
             st.button("Code Diff Preview", use_container_width=True, on_click=show_diff_preview, icon=":material/compare:")
        
        # Show Diff Section in expander
        if st.session_state.get("show_diff", False):
            code_changes = selected.get("code_changes", {})
            ai_code_snippet = selected.get("ai_code_snippet", "")
            
            with st.expander("Code Diff Preview", expanded=True, icon=":material/compare:"):
                if code_changes:
                    for file_path, content in code_changes.items():
                        st.caption(f"File: `{file_path}`")
                        diff_preview = selected.get("diff_preview") or ai_code_snippet
                        
                        if diff_preview:
                            st.code(diff_preview, language="diff", wrap_lines=True)
                        else:
                            st.code(content, language="sql", wrap_lines=True)
                elif ai_code_snippet:
                    st.code(ai_code_snippet, language="diff" if ("+" in ai_code_snippet and "-" in ai_code_snippet) else "sql", wrap_lines=True)
                else:
                    st.info("No code changes generated for preview.")

        # Show Preview Impact Result in expander
        if st.session_state.get("preview_impact_result"):
            result = st.session_state.preview_impact_result
            risk_level = result.get("data", {}).get("risk_assessment", {}).get("level", "unknown")
            risk_icon = {"low": ":material/check_circle:", "medium": ":material/warning:", "high": ":material/error:"}.get(risk_level, ":material/help:")
            with st.expander(f"Preview Impact {risk_icon}", expanded=True, icon=":material/analytics:"):
                render_preview_impact_result_content()
        
        # Show Run Test Result in expander
        if st.session_state.get("run_test_result"):
            result = st.session_state.run_test_result
            status = result.get("status", "error")
            status_emoji = {
                "success": ":material/check_circle:", 
                "partial": ":material/error_outline:", 
                "error": ":material/error:"}.get(status, ":material/help:") 
            with st.expander(f"Test Results {status_emoji}", expanded=True, icon=":material/lab_research:"):
                render_run_test_result_content()






def render_preview_impact(selected: Dict[str, Any], incident: Dict[str, Any]):
    """Analyze and display the impact of applying a fix."""
    from app.agent.tools import preview_impact
    
    model_name = incident.get("model_name", "")
    column_name = incident.get("column_name")
    
    with st.status(f"Analyzing Impact...", expanded=True, state="running") as status:
        st.write(f"{icon('analytics', 16)} Analyzing change type...", unsafe_allow_html=True)
        st.write(f"{icon('account_tree', 16)} Checking model lineage...", unsafe_allow_html=True)
        st.write(f"{icon('assessment', 16)} Assessing risk level...", unsafe_allow_html=True)
        
        # Execute preview impact
        result = preview_impact(selected, model_name, column_name)
        
        if result["status"] == "success":
            risk_level = result.get("data", {}).get("risk_assessment", {}).get("level", "unknown")
            status.update(label=f"Impact Analysis Complete (Risk: {risk_level.upper()})", state="complete", expanded=False)
        else:
            status.update(label="Impact Analysis Failed", state="error", expanded=True)
             
    st.session_state.preview_impact_result = result
    st.session_state.show_diff = False


def render_preview_impact_result():
    """Render the impact preview results with header."""
    st.markdown(f"### {icon('analytics', 20)} Impact Analysis", unsafe_allow_html=True)
    render_preview_impact_result_content()


def render_preview_impact_result_content():
    """Render the impact preview results content (without header - for use in expander)."""
    result = st.session_state.preview_impact_result
    data = result.get("data", {})
    
    if result.get("status") != "success":
        st.error(f"**Analysis Failed**: {result.get('message')}")
        return
    
    # Fix title and model
    fix_title = data.get("fix_title", "Unknown Fix")
    model_name = data.get("model_name", "Unknown")
    risk = data.get("risk_assessment", {})
    risk_level = risk.get("level", "unknown")
    
    # Risk level color coding
    risk_colors = {"low": ":material/check_circle:", "medium": ":material/warning:", "high": ":material/error:"}
    risk_icon = risk_colors.get(risk_level, ":material/help:")
    
    # Header with risk indicator
    st.markdown(f"**{fix_title}** on `{model_name}` {risk_icon} Risk: **{risk_level.upper()}**")
    
    # Main metrics row
    change_summary = data.get("change_summary", {})
    change_type = change_summary.get("change_type", "unknown")
    files_modified = change_summary.get("files_modified", 0)
    downstream_count = data.get("lineage", {}).get("downstream_count", 0)
    
    st.markdown("##### Overview")
    st.markdown(f"""
    | Change Type | Files Modified | Downstream Models |
    | :--- | :--- | :--- |
    | {change_type.replace("_", " ").title()} | {files_modified} | {downstream_count} |
    """)
    
    # Change Summary
    st.markdown(f"#### Change Summary", unsafe_allow_html=True)
    st.write(change_summary.get("description", "No description available"))
    
    # Data Impact
    data_impact = data.get("data_impact", {})
    
    impact_items = []
    if data_impact.get("changes_test_only"):
        impact_items.append("Test configuration only - no data changes")
    if data_impact.get("modifies_output"):
        impact_items.append("Modifies model output data")
    if data_impact.get("removes_rows"):
        impact_items.append("May exclude rows from output")
    if data_impact.get("adds_default_values"):
        impact_items.append("Adds default values for NULLs")
    
    if impact_items:
        st.markdown("**Data Impact:**")
        for item in impact_items:
            st.markdown(f"- {item}")
    
    st.caption(f"Note: {data_impact.get('estimated_impact', 'Review changes for impact')}")
    
    # Downstream Impact
    lineage = data.get("lineage", {})
    downstream_impact = lineage.get("downstream_impact", [])
    
    if downstream_impact:
        with st.expander(f"Downstream Impact ({len(downstream_impact)} models)", expanded=False):
            for dm in downstream_impact:
                impact_level = dm.get("impact_level", "unknown")
                level_icon = {"low": ":material/check_circle:", "medium": ":material/warning:", "high": ":material/error:"}.get(impact_level, ":material/help:")
                st.markdown(f"{level_icon} `{dm.get('model', 'Unknown')}` - {dm.get('reason', 'Unknown')}")
    
    # Risk Assessment
    st.markdown(f"#### Risk Assessment", unsafe_allow_html=True)
    
    # Risk factors
    factors = risk.get("factors", [])
    if factors:
        st.markdown("**Risk Factors:**")
        for factor in factors:
            st.markdown(f"- {factor}")
    
    # Recommendations
    recommendations = risk.get("recommendations", [])
    if recommendations:
        st.markdown("**Recommendations:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")


def render_run_test(selected: Dict[str, Any], incident: Dict[str, Any]):
    """Execute dbt test to verify if the fix resolves the issue."""
    model_name = incident.get("model_name", "")
    test_name = incident.get("test_name", "")
    
    # Check if this fix has code changes
    code_changes = selected.get("code_changes", {})
    fix_type = selected.get("fix_type", "")
    
    # Non-code fix types don't need testing
    NON_CODE_FIX_TYPES = {"snooze", "manual_review", "alert_suppression", "ignore"}
    
    if fix_type in NON_CODE_FIX_TYPES:
        st.info(f"""
        **This is a non-code fix option**
        
        Fix type: `{fix_type}`
        
        This option doesn't modify code and cannot be tested. It's an operational action like:
        - Snoozing the alert temporarily
        - Marking for manual review
        - Suppressing the alert
        """)
        return
    
    if not code_changes:
        # This shouldn't happen with mandatory code_changes, but handle gracefully
        st.error("""
        **No Code Changes Available**
        
        This fix option is missing generated code. This is unexpected.
        Please try re-running the investigation or selecting a different fix option.
        """)
        return
    
    with st.status("🧪 Running Test...", expanded=True, state="running") as status:
        st.write(f"{icon('check_circle', 16)} Checking dbt installation...", unsafe_allow_html=True)
        dbt_available = check_dbt_installed()
        
        if not dbt_available:
            st.warning("dbt not found - cannot run actual tests")
            status.update(label="dbt not available", state="error")
            st.session_state.run_test_result = {
                "status": "error",
                "message": "dbt is not installed. Cannot run tests.",
                "data": {"dbt_not_found": True}
            }
            return
        
        st.write(f"{icon('edit', 16)} Temporarily applying code changes...", unsafe_allow_html=True)
        for file_path in code_changes.keys():
            st.caption(f"  → {file_path}")
        
        st.write(f"{icon('science', 16)} Running: `dbt test --select {test_name or model_name}`...", unsafe_allow_html=True)
        
        # Execute the test
        result = run_dbt_test(selected, model_name, test_name)
        
        if result["status"] == "success":
            status.update(label=f"Test Passed! Fix works.", state="complete", expanded=False)
        elif result["status"] == "partial":
            status.update(label=f"Partial Success", state="complete", expanded=True)
        else:
            status.update(label=f"Test Failed", state="error", expanded=True)
    st.session_state.run_test_result = result
    st.session_state.show_diff = False
    st.session_state.dry_run_result = None


def render_run_test_result():
    """Render the run test results with header."""
    st.markdown(f"### {icon('science', 20)} Test Verification Results", unsafe_allow_html=True)
    render_run_test_result_content()


def render_run_test_result_content():
    """Render the run test results content (without header - for use in expander)."""
    result = st.session_state.run_test_result
    data = result.get("data", {})
    
    if result.get("status") == "success":
        st.markdown(f":material/check_circle: **{result.get('message')}**")
        st.markdown(f"""
        The fix for **{data.get('fix_title', 'this issue')}** has been verified:
        - Model: `{data.get('model_name')}`
        - Test: `{data.get('test_name') or 'all model tests'}`
        """)
        
        # Show test results summary
        test_results = data.get("test_results", [])
        if test_results:
            st.markdown("**Test Results:**")
            for t in test_results:
                status_icon = ":material/check_circle:" if t.get("status") == "pass" else ":material/error:"
                st.caption(f"{status_icon} {t.get('test_name', 'Unknown test')}")
        
        # Recommend next steps
        st.info(":material/thumb_up: **Recommended**: Apply this fix permanently using the 'Apply Fix' button.")
        
    elif result.get("status") == "partial":
        st.markdown(f":material/warning: **{result.get('message')}**")
        
        test_results = data.get("test_results", [])
        if test_results:
            st.markdown("**Test Results:**")
            for t in test_results:
                if t.get("status") == "pass":
                    st.caption(f":material/check_circle: {t.get('test_name')}")
                elif t.get("status") == "fail":
                    failures = t.get("failures", "?")
                    st.caption(f":material/error: {t.get('test_name')} ({failures} failures)")
                elif t.get("status") == "warn":
                    st.caption(f":material/warning: {t.get('test_name')} (warning)")
                else:
                    st.caption(f":material/report: {t.get('test_name')} (error)")
        
        st.info("Consider reviewing the failing tests or trying a different fix option.")
        
    else:
        st.markdown(f":material/error: **{result.get('message')}**")
        
        # Show what went wrong
        if data.get("dbt_not_found"):
            st.warning("dbt CLI is not installed or not in PATH.")
        elif data.get("timeout"):
            st.warning("The test took too long to complete. Try running it manually.")
        else:
            test_results = data.get("test_results", [])
            if test_results:
                st.markdown("**Failed Tests:**")
                for t in test_results:
                    if t.get("status") in ["fail", "error"]:
                        failures = t.get("failures", "")
                        fail_info = f" ({failures} failures)" if failures else ""
                        st.caption(f":material/error: {t.get('test_name')}{fail_info}")
            
            # Show stderr if available
            stderr = data.get("stderr")
            if stderr:
                with st.expander("Error Details"):
                    st.code(stderr, language="bash")
        
        st.info(":material/lightbulb: **Tip**: Try selecting a different fix option, or check the error details above.")
    
    # Show modified code content
    code_changes = data.get("code_changes", {})
    if code_changes:
        with st.expander("Modified File Content", expanded=False):
            for file_path, content in code_changes.items():
                st.caption(f"File: `{file_path}`")
                st.code(content, language="sql" if file_path.endswith(".sql") else "yaml")

    # Show dbt output for debugging
    stdout = data.get("stdout")
    if stdout:
        with st.expander("Full dbt Output", expanded=False):
            st.code(stdout, language="bash")


def render_test_verification(incident: Dict[str, Any]):
    """Run dbt test to verify the fix worked."""
    model_name = incident.get("model_name", "")
    test_name = incident.get("test_name", "")
    
    with st.status("Running tests to verify fix...", expanded=True) as status:
        st.write(f"{icon('science', 16)} Running dbt test for `{model_name}`...", unsafe_allow_html=True)
        
        result = run_dbt_test(model_name, test_name)
        
        if result.get("data", {}).get("test_passed"):
            st.write(f"{icon('check_circle', 16)} Test passed!", unsafe_allow_html=True)
            status.update(label="Test passed! Fix verified.", state="complete")
        else:
            st.write(f"{icon('error', 16)} Test still failing", unsafe_allow_html=True)
            status.update(label="Test failed - fix may need adjustment", state="error")
    
    st.session_state.test_result = result
    st.rerun()


def render_test_result():
    """Render the test verification results."""
    result = st.session_state.test_result
    data = result.get("data", {})
    
    st.markdown(f"### {icon('science', 20)} Test Verification Results", unsafe_allow_html=True)
    
    if data.get("test_passed"):
        st.success(f"**Test Passed!** The fix for `{data.get('model_name', '')}` has been verified.")
        st.balloons()
        
        # Show test output even when passed
        if data.get("stdout"):
            with st.expander("Test Output", expanded=False):
                st.code(data["stdout"], language="text", wrap_lines=True)
        
        if data.get("stderr"):
            with st.expander("Test Warnings/Errors", expanded=False):
                st.code(data["stderr"], language="text", wrap_lines=True)
        
        # Show option to go back
        st.button("Return to Dashboard", type="primary", use_container_width=True, on_click=go_to_dashboard)
    else:
        st.error(f"**Test Failed** - The fix may need adjustment.")
        
        # Show test output
        if data.get("stdout"):
            with st.expander("Test Output", expanded=True):
                st.code(data["stdout"], language="text", wrap_lines=True)
        
        if data.get("stderr"):
            with st.expander("Error Details", expanded=True):
                st.code(data["stderr"], language="text", wrap_lines=True)
        
        # Show return button
        if st.button("Return to Fixes", type="primary"):
            st.session_state.test_result = None
            st.rerun()


def render_diff_view(selected: Dict[str, Any], incident: Dict[str, Any]):
    """Render a side-by-side diff view showing current vs proposed code."""
    
    st.markdown(f"### {icon('compare', 20)} Code Diff Preview", unsafe_allow_html=True)
    
    code_changes = selected.get("code_changes", {})
    ai_code = selected.get("ai_code_snippet", "")
    fix_type = selected.get("fix_type", "")
    model_name = incident.get("model_name", "")
    column_name = incident.get("column_name", "")
    
    if not code_changes and not ai_code:
        st.info(f"No automatic code changes for this fix type ({fix_type}). See dry run for guidance.")
        return
    
    # If we have code_changes, show the file diff
    if code_changes:
        for file_path, new_content in code_changes.items():
            original_result = tool_read_repo(file_path)
            
            if original_result["status"] == "success":
                original_content = original_result["data"]["content"]
                
                render_side_by_side_diff(
                    original_content, 
                    new_content, 
                    file_path,
                    original_result["data"].get("relative_path", file_path)
                )
            else:
                st.error(f"Could not read original file: {original_result['message']}")
        return
    
    # If we only have AI code snippet, show before/after with the relevant file
    if ai_code:
        # Determine which file to show based on fix type
        original_content = None
        file_path = None
        relative_path = None
        
        # For schema-related fixes, show schema.yml
        if fix_type in ["update_accepted_values", "update_range"] or "schema" in ai_code.lower() or "tests:" in ai_code.lower():
            schema_result = find_schema_file(model_name)
            if schema_result["status"] == "success":
                file_path = schema_result["data"]["path"]
                relative_path = schema_result["data"].get("relative_path", file_path)
                file_result = tool_read_repo(file_path)
                if file_result["status"] == "success":
                    original_content = file_result["data"]["content"]
        
        # For SQL-related fixes, show model SQL
        if original_content is None:
            sql_result = find_file_by_model_name(model_name)
            if sql_result["status"] == "success":
                file_path = sql_result["data"]["path"]
                relative_path = sql_result["data"].get("relative_path", file_path)
                file_result = tool_read_repo(file_path)
                if file_result["status"] == "success":
                    original_content = file_result["data"]["content"]
        
        if original_content:
            # Show the current file and the AI suggestion side by side
            st.markdown(f"**{relative_path or 'Current File'}**")
            st.caption("Compare current code with AI-suggested changes")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown(f"{icon('remove_circle', 16)} **BEFORE (Current)**", unsafe_allow_html=True)
                with st.container(border=True):
                    # Show excerpt of relevant section
                    excerpt = _extract_relevant_excerpt(original_content, column_name, model_name)
                    st.code(excerpt, language="yaml" if ".yml" in str(file_path) else "sql", wrap_lines=True)
            
            with col_right:
                st.markdown(f"{icon('add_circle', 16)} **AFTER (AI Suggested)**", unsafe_allow_html=True)
                with st.container(border=True):
                    st.code(ai_code, language="yaml" if ":" in ai_code and "-" in ai_code else "sql", wrap_lines=True)
            
            st.caption("Apply the suggested changes manually to the appropriate location in the file.")
        else:
            # Fallback: just show the AI suggestion
            st.markdown("**AI-Suggested Code Change:**")
            st.code(ai_code, language="sql", wrap_lines=True)
            st.caption("Apply this code snippet manually to the appropriate file.")


def _extract_relevant_excerpt(content: str, column_name: str, model_name: str) -> str:
    """Extract relevant section from content based on column/model name."""
    lines = content.splitlines()
    
    # Try to find the column or model in the content
    target_line = -1
    for i, line in enumerate(lines):
        if column_name and column_name in line:
            target_line = i
            break
        if model_name and model_name in line:
            target_line = i
    
    if target_line >= 0:
        # Show 5 lines before and 10 lines after the target
        start = max(0, target_line - 5)
        end = min(len(lines), target_line + 15)
        excerpt_lines = lines[start:end]
        
        # Add indicators
        if start > 0:
            excerpt_lines.insert(0, f"# ... (lines 1-{start} omitted) ...")
        if end < len(lines):
            excerpt_lines.append(f"# ... (lines {end+1}-{len(lines)} omitted) ...")
        
        return '\n'.join(excerpt_lines)
    
    # If no match found, show first 30 lines
    if len(lines) > 30:
        return '\n'.join(lines[:30]) + f"\n# ... ({len(lines) - 30} more lines) ..."
    return content


def render_side_by_side_diff(
    original: str, 
    modified: str, 
    file_path: str,
    display_path: str = None
):
    """Render a true side-by-side Before/After comparison with color coding."""
    
    display_path = display_path or file_path
    
    # Generate the diff
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()
    
    # Calculate statistics from unified diff
    diff = list(difflib.unified_diff(original_lines, modified_lines, lineterm=''))
    additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    if additions == 0 and deletions == 0:
        st.info("No changes detected in this file")
        return
    
    # File header
    st.markdown(f"**{display_path}**")
    st.caption(f"+{additions} additions | -{deletions} deletions")
    
    # Generate side-by-side HTML
    before_html, after_html = generate_side_by_side_html(original_lines, modified_lines)
    
    # Two column layout for side-by-side
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"{icon('remove_circle', 16)} **BEFORE**", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(before_html, unsafe_allow_html=True)
    
    with col_right:
        st.markdown(f"{icon('add_circle', 16)} **AFTER**", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(after_html, unsafe_allow_html=True)
    
    # Also show unified diff for easy copying
    with st.expander("Unified Diff (copyable)", expanded=False):
        diff_text = '\n'.join(diff)
        st.code(diff_text, language="diff", wrap_lines=True)


def generate_side_by_side_html(original_lines: list, modified_lines: list) -> tuple:
    """
    Generate HTML for side-by-side Before/After display.
    
    Returns:
        Tuple of (before_html, after_html)
    """
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
    
    before_parts = ['<div style="font-family: JetBrains Mono, monospace; font-size: 12px; line-height: 1.6;">']
    after_parts = ['<div style="font-family: JetBrains Mono, monospace; font-size: 12px; line-height: 1.6;">']
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Unchanged lines - show in both columns (gray)
            for line in original_lines[i1:i2]:
                escaped = _escape_html(line) if line else '&nbsp;'
                before_parts.append(
                    f'<div style="color: #8b949e; padding: 2px 8px; white-space: pre-wrap;">{escaped}</div>'
                )
                after_parts.append(
                    f'<div style="color: #8b949e; padding: 2px 8px; white-space: pre-wrap;">{escaped}</div>'
                )
        
        elif tag == 'delete':
            # Lines only in original - show in red on left, empty on right
            for line in original_lines[i1:i2]:
                escaped = _escape_html(line) if line else '&nbsp;'
                before_parts.append(
                    f'<div style="background-color: #4d1a1a; color: #f47067; padding: 2px 8px; '
                    f'border-left: 3px solid #f85149; white-space: pre-wrap;">{escaped}</div>'
                )
                after_parts.append(
                    f'<div style="background-color: #2d2d2d; color: #666; padding: 2px 8px; '
                    f'white-space: pre-wrap;">&nbsp;</div>'
                )
        
        elif tag == 'insert':
            # Lines only in modified - show empty on left, green on right
            for line in modified_lines[j1:j2]:
                escaped = _escape_html(line) if line else '&nbsp;'
                before_parts.append(
                    f'<div style="background-color: #2d2d2d; color: #666; padding: 2px 8px; '
                    f'white-space: pre-wrap;">&nbsp;</div>'
                )
                after_parts.append(
                    f'<div style="background-color: #1a4d1a; color: #7ee787; padding: 2px 8px; '
                    f'border-left: 3px solid #3fb950; white-space: pre-wrap;">{escaped}</div>'
                )
        
        elif tag == 'replace':
            # Lines changed - show old in red, new in green
            old_lines = original_lines[i1:i2]
            new_lines = modified_lines[j1:j2]
            
            max_lines = max(len(old_lines), len(new_lines))
            
            for idx in range(max_lines):
                # Before column (deletions in red)
                if idx < len(old_lines):
                    escaped = _escape_html(old_lines[idx]) if old_lines[idx] else '&nbsp;'
                    before_parts.append(
                        f'<div style="background-color: #4d1a1a; color: #f47067; padding: 2px 8px; '
                        f'border-left: 3px solid #f85149; white-space: pre-wrap;">{escaped}</div>'
                    )
                else:
                    before_parts.append(
                        f'<div style="background-color: #2d2d2d; padding: 2px 8px;">&nbsp;</div>'
                    )
                
                # After column (additions in green)
                if idx < len(new_lines):
                    escaped = _escape_html(new_lines[idx]) if new_lines[idx] else '&nbsp;'
                    after_parts.append(
                        f'<div style="background-color: #1a4d1a; color: #7ee787; padding: 2px 8px; '
                        f'border-left: 3px solid #3fb950; white-space: pre-wrap;">{escaped}</div>'
                    )
                else:
                    after_parts.append(
                        f'<div style="background-color: #2d2d2d; padding: 2px 8px;">&nbsp;</div>'
                    )
    
    before_parts.append('</div>')
    after_parts.append('</div>')
    
    return '\n'.join(before_parts), '\n'.join(after_parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )
