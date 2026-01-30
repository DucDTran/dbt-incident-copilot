"""
dbt Co-Work Agent using Google ADK (Agent Development Kit).

This agent autonomously investigates dbt test failures by:
1. Analyzing test failure details
2. Reading model lineage from dbt manifest
3. Examining SQL code and schema definitions
4. Searching business rules knowledge base
5. Generating AI-powered diagnosis and fix recommendations
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import ast
import re

from google import genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from app.config import get_settings
from app.prompts import (
    COPILOT_SYSTEM_INSTRUCTION,
    get_investigation_prompt,
)

# Configure genai client globally with API key
# This MUST happen before creating any ADK agents
_settings = get_settings()
_genai_client = genai.Client(api_key=_settings.google_api_key)

from app.agent.tools import (
    tool_read_manifest,
    get_model_lineage,
    tool_read_repo,
    tool_query_elementary,
    get_failed_tests,
    tool_consult_knowledge_base,
)
from app.agent.tools.agentic_fix_tool import adk_propose_fix as _adk_propose_fix
from app.agent.tools.knowledge_base_tool import search_for_business_rule
from app.agent.tools.repo_tool import find_file_by_model_name, find_schema_file
from app.agent.tools.sql_tool import adk_execute_sql


@dataclass
class InvestigationStep:
    """A single step in the agent's investigation."""
    timestamp: datetime
    action: str
    tool_name: Optional[str]
    input_summary: str
    output_summary: str
    status: str  # 'success', 'error', 'thinking'
    details: Optional[Dict[str, Any]] = None


@dataclass
class Investigation:
    """Complete investigation for a test failure."""
    test_result: Dict[str, Any]
    steps: List[InvestigationStep] = field(default_factory=list)
    diagnosis: Optional[str] = None
    fix_options: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


# ============================================================
# ADK Tool Definitions
# These functions are exposed to the LlmAgent as callable tools
# ============================================================

def adk_get_model_lineage(model_name: str) -> str:
    """
    Get the upstream and downstream dependencies for a dbt model.
    
    Args:
        model_name: The name of the dbt model to get lineage for.
        
    Returns:
        JSON string with lineage information including upstream sources,
        upstream models, downstream models, and associated tests.
    """
    result = get_model_lineage(model_name)
    return json.dumps(result, indent=2, default=str)


def adk_read_model_sql(model_name: str) -> str:
    """
    Read the SQL code for a dbt model.
    
    Args:
        model_name: The name of the dbt model to read SQL for.
        
    Returns:
        The SQL code content of the model, or an error message.
    """
    file_result = find_file_by_model_name(model_name)
    if file_result["status"] != "success":
        return json.dumps({"status": "error", "message": file_result["message"]})
    
    sql_path = file_result["data"]["path"]
    sql_result = tool_read_repo(sql_path)
    
    if sql_result["status"] == "success":
        return json.dumps({
            "status": "success",
            "path": sql_result["data"]["relative_path"],
            "lines": sql_result["data"]["lines"],
            "content": sql_result["data"]["content"]
        }, indent=2)
    else:
        return json.dumps({"status": "error", "message": sql_result["message"]})


def adk_read_schema_definition(model_name: str) -> str:
    """
    Read the schema.yml definition for a dbt model including column definitions and tests.
    
    Args:
        model_name: The name of the dbt model to read schema for.
        
    Returns:
        The YAML schema content and model definition, or an error message.
    """
    schema_result = find_schema_file(model_name)
    if schema_result["status"] != "success":
        return json.dumps({"status": "error", "message": schema_result["message"]})
    
    schema_path = schema_result["data"]["path"]
    yaml_result = tool_read_repo(schema_path)
    
    if yaml_result["status"] == "success":
        return json.dumps({
            "status": "success",
            "path": schema_result["data"]["relative_path"],
            "model_definition": schema_result["data"]["model_definition"],
            "full_content": yaml_result["data"]["content"]
        }, indent=2, default=str)
    else:
        return json.dumps({"status": "error", "message": yaml_result["message"]})


def adk_search_knowledge_base(query: str, context_column: str = "", context_model: str = "") -> str:
    """
    Search the business rules knowledge base for relevant documentation.
    
    Args:
        query: The search query - can be a test name, column name, or concept.
        context_column: Optional column name for context.
        context_model: Optional model name for context.
        
    Returns:
        Relevant business rules, data quality policies, and documentation.
    """
    # Try semantic search first
    from app.agent.tools.knowledge_base_tool import get_knowledge_base
    kb = get_knowledge_base()
    
    search_query = f"{query} {context_column} {context_model}".strip()
    results = kb.search(search_query, top_k=5)
    
    if results:
        return json.dumps({
            "status": "success",
            "results": results
        }, indent=2, default=str)
    
    # Fallback to targeted search
    kb_result = search_for_business_rule(context_column, context_model, query)
    return json.dumps(kb_result, indent=2, default=str)


def adk_read_file(file_path: str) -> str:
    """
    Read any file from the dbt project repository.
    
    Args:
        file_path: Path to the file (relative to dbt project root).
        
    Returns:
        The file content, or an error message.
    """
    result = tool_read_repo(file_path)
    return json.dumps(result, indent=2, default=str)


def adk_get_test_details(test_id: str) -> str:
    """
    Get detailed information about a specific test from Elementary.
    
    Args:
        test_id: The unique identifier of the test.
        
    Returns:
        Test details including error message, failed rows, and sample data.
    """
    from app.agent.tools.elementary_tool import get_test_details
    result = get_test_details(test_id)
    return json.dumps(result, indent=2, default=str)


def adk_propose_fix(
    test_name: str,
    model_name: str,
    column_name: str,
    root_cause: str,
    fix_options: str
) -> str:
    """
    Propose fix options for a test failure based on investigation.
    
    This is the main tool for generating fixes. Call this after completing
    your investigation to formalize your fix recommendations.
    
    Args:
        test_name: Name of the failing test.
        model_name: Name of the dbt model.
        column_name: Name of the column (if applicable).
        root_cause: The diagnosed root cause of the failure.
        fix_options: JSON string containing an array of fix options, each with:
            - id: Unique identifier (e.g., "fix_1")
            - title: Clear action title
            - description: What this fix does
            - fix_type: One of 'schema' or 'sql'
            - code_change: Object with fix details:
                - For schema: {"action": "add_accepted_values|update_range|change_severity", "details": {...}}
                - For sql: {"action": "add_where_clause|add_coalesce|replace_column", "code": "..."}
            - rationale: Why this fix works
            - pros: Key advantages or strengths of this option
            - cons: Key tradeoffs, limitations, or risks of this option
            - when_appropriate: When this option is the best choice (conditions, scenarios, or constraints)
            
    Returns:
        JSON string with validated fix options ready for UI display.
    """
    return _adk_propose_fix(test_name, model_name, column_name, root_cause, fix_options)


# ============================================================
# ADK Agent Setup
# ============================================================


class CopilotAgent:
    """
    The dbt Co-Work Agent that investigates test failures 
    using Google ADK (Agent Development Kit) for autonomous tool use.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.investigations: Dict[str, Investigation] = {}
        self._agent = self._create_agent()
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._agent,
            app_name="dbt-copilot",
            session_service=self._session_service,
        )
    
    def _create_agent(self) -> LlmAgent:
        """Create the ADK LlmAgent with tools."""
        return LlmAgent(
            name="dbt_copilot",
            model=self.settings.gemini_model,
            instruction=COPILOT_SYSTEM_INSTRUCTION,
            tools=[
                adk_get_model_lineage,
                adk_read_model_sql,
                adk_read_schema_definition,
                adk_search_knowledge_base,
                adk_read_file,
                adk_get_test_details,
                adk_execute_sql,  # SQL execution for data investigation
                adk_propose_fix,  # Agentic fix generation
            ],
        )
    
    async def investigate(
        self, 
        test_result: Dict[str, Any],
        stream_steps: bool = True
    ):

        investigation = Investigation(test_result=test_result)
        test_id = test_result.get("test_id", "unknown")
        self.investigations[test_id] = investigation
        
        model_name = test_result.get("model_name", "")
        column_name = test_result.get("column_name", "")
        test_name = test_result.get("test_name", "")
        error_message = test_result.get("error_message", "")
        failed_rows = test_result.get("failed_rows", 0)
        
        # Step 1: Initial Analysis
        step = InvestigationStep(
            timestamp=datetime.now(),
            action="Analyzing test failure",
            tool_name=None,
            input_summary=f"Test: {test_name}",
            output_summary=f"Model: {model_name}, Column: {column_name}",
            status="success",
        )
        investigation.steps.append(step)
        if stream_steps:
            yield step
        
        # Create the investigation prompt for the agent
        investigation_prompt = get_investigation_prompt(
            test_name=test_name,
            model_name=model_name,
            column_name=column_name,
            error_message=error_message,
            failed_rows=failed_rows,
            test_id=test_id,
        )

        # Step 2: Agent Investigation (ADK handles tool calls automatically)
        step = InvestigationStep(
            timestamp=datetime.now(),
            action="Agent investigating with tools",
            tool_name="adk_agent",
            input_summary="Autonomous investigation",
            output_summary="",
            status="thinking",
        )
        investigation.steps.append(step)
        if stream_steps:
            yield step
        
        try:
            # Create a session for this investigation
            session = await self._session_service.create_session(
                app_name="dbt-copilot",
                user_id=f"investigation_{test_id}",
            )
            
            # Collect tool calls and responses for step tracking
            tool_steps = []
            diagnosis_text = ""
            pending_tool_calls = {}  # Track pending calls by name
            
            # Run the agent and stream events
            async for event in self._runner.run_async(
                user_id=f"investigation_{test_id}",
                session_id=session.id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=investigation_prompt)]
                ),
            ):
                # Track tool calls using ADK's get_function_calls() method
                function_calls = event.get_function_calls()
                if function_calls:
                    for fc in function_calls:
                        tool_step = InvestigationStep(
                            timestamp=datetime.now(),
                            action=f"Using {fc.name}",
                            tool_name=fc.name,
                            input_summary=str(fc.args)[:100] if fc.args else "",
                            output_summary="",
                            status="thinking",
                        )
                        tool_steps.append(tool_step)
                        pending_tool_calls[fc.name] = tool_step
                        investigation.steps.append(tool_step)
                        if stream_steps:
                            yield tool_step
                
                # Track tool responses using ADK's get_function_responses() method
                function_responses = event.get_function_responses()
                if function_responses:
                    for fr in function_responses:
                        tool_name = fr.name if hasattr(fr, 'name') else None
                        
                        # Extract response - handle different response formats
                        if hasattr(fr, 'response'):
                            raw_response = fr.response
                            # If it's already a string, use it
                            if isinstance(raw_response, str):
                                response_text = raw_response
                            # If it's a dict, try to extract the actual result
                            elif isinstance(raw_response, dict):
                                # Check for common ADK response formats
                                if 'result' in raw_response:
                                    response_text = raw_response['result']
                                elif 'content' in raw_response:
                                    response_text = raw_response['content']
                                else:
                                    # Try to serialize to JSON
                                    response_text = json.dumps(raw_response)
                            else:
                                response_text = str(raw_response)
                        else:
                            response_text = str(fr)
                        
                        # Find the matching tool step
                        if tool_name and tool_name in pending_tool_calls:
                            tool_step = pending_tool_calls[tool_name]
                        elif tool_steps:
                            tool_step = tool_steps[-1]
                        else:
                            continue
                        
                        # Compute how long this tool call took (approximate)
                        try:
                            elapsed = (datetime.now() - tool_step.timestamp).total_seconds()
                        except Exception:
                            elapsed = None
                        
                        tool_step.output_summary = f"Received response ({len(response_text)} chars)"
                        tool_step.status = "success"
                        details = tool_step.details or {}
                        details.update({
                            "response_preview": response_text[:200],
                            "response_text": response_text,
                        })
                        # If the raw response is structured, keep it too
                        if isinstance(raw_response, dict):
                            details["raw_response"] = raw_response
                        if elapsed is not None:
                            details["time_used_seconds"] = round(elapsed, 3)
                        tool_step.details = details
                        
                        # Store context based on tool name (check both tool_name from response and tool_step.tool_name)
                        tool_name_lower = (tool_name or "").lower() if tool_name else ""
                        step_tool_name_lower = (tool_step.tool_name or "").lower() if tool_step.tool_name else ""
                        combined_tool_name = f"{tool_name_lower} {step_tool_name_lower}"
                        
                        # Check if this is a propose_fix response (most important to capture correctly)
                        if "propose_fix" in combined_tool_name or "adk_propose" in combined_tool_name:
                            # Extract agentic fix options from propose_fix response
                            investigation.context["agentic_fix_response"] = response_text
                            investigation.context["propose_fix_tool_name"] = tool_name or tool_step.tool_name
                            investigation.context["propose_fix_response_length"] = len(response_text)
                        elif "lineage" in combined_tool_name:
                            investigation.context["lineage"] = response_text
                        elif "sql" in combined_tool_name or "read_model" in combined_tool_name:
                            investigation.context["sql_code"] = response_text
                        elif "schema" in combined_tool_name or "read_schema" in combined_tool_name:
                            investigation.context["schema"] = response_text
                        elif "knowledge" in combined_tool_name or "search" in combined_tool_name:
                            investigation.context["business_rules"] = response_text
                        
                        # Also check response content for propose_fix indicators (fallback)
                        if "agentic_fix_response" not in investigation.context:
                            response_lower = response_text.lower()[:500] if response_text else ""
                            if '"status":"success"' in response_lower and '"options"' in response_lower and '"fix_type"' in response_lower:
                                # This looks like a propose_fix response even if tool name didn't match
                                investigation.context["agentic_fix_response"] = response_text
                                investigation.context["propose_fix_detected_by_content"] = True
                        
                        if stream_steps:
                            yield tool_step
                
                # Capture the final text response (diagnosis)
                if event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            diagnosis_text += part.text
            
            # Update the agent investigation step
            step.output_summary = f"Completed with {len(tool_steps)} tool call(s)"
            step.status = "success"
            step.details = {"tool_calls": [t.tool_name for t in tool_steps]}
            if stream_steps:
                yield step
            
            # Store diagnosis (UI will extract final diagnosis sections as needed)
            if diagnosis_text:
                # Clean up any legacy <final_diagnosis> tags if they appear
                import re
                cleaned = re.sub(r'</?final_diagnosis>', '', diagnosis_text, flags=re.IGNORECASE).strip()
                investigation.diagnosis = cleaned
            else:
                investigation.diagnosis = "Agent did not provide a diagnosis."
            
            investigation.context["agent_response"] = diagnosis_text
            
            # Check if propose_fix was called - if not, send a follow-up prompt
            propose_fix_called = any(
                "propose_fix" in (t.tool_name or "").lower() 
                for t in tool_steps
            )
            
            if not propose_fix_called and diagnosis_text:
                # Send a follow-up message to force propose_fix call
                follow_up_step = InvestigationStep(
                    timestamp=datetime.now(),
                    action="Requesting fix proposals",
                    tool_name="adk_agent",
                    input_summary="Agent did not call propose_fix, sending follow-up",
                    output_summary="",
                    status="thinking",
                )
                investigation.steps.append(follow_up_step)
                if stream_steps:
                    yield follow_up_step
                
                follow_up_prompt = f"""
                You have completed your investigation and provided a diagnosis. However, you have not yet called the propose_fix tool.
                CRITICAL: You MUST call the propose_fix tool now with your recommended fix options. The proposed fixes should be specific to the root cause and evidence found, as well as aligned with the business context. This is a required step.
                Based on your investigation, please call propose_fix with 4-5 fix options. Each option should include:
                - id: unique identifier
                - title: clear action title
                - description: what this fix does
                - fix_type: "schema" or "sql"
                - risk_level: "low", "medium", or "high"
                - rationale: why this fix works
                - confidence: number between 0.0 and 1.0
                - code_change: object with action and details/code

                Call propose_fix now with your fix recommendations."""
                
                try:
                    follow_up_diagnosis = ""
                    follow_up_tool_steps = []
                    
                    async for event in self._runner.run_async(
                        user_id=f"investigation_{test_id}",
                        session_id=session.id,
                        new_message=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=follow_up_prompt)]
                        ),
                    ):
                        # Track propose_fix calls from follow-up
                        function_calls = event.get_function_calls()
                        if function_calls:
                            for fc in function_calls:
                                if "propose_fix" in fc.name.lower():
                                    tool_step = InvestigationStep(
                                        timestamp=datetime.now(),
                                        action=f"Using {fc.name}",
                                        tool_name=fc.name,
                                        input_summary=str(fc.args)[:100] if fc.args else "",
                                        output_summary="",
                                        status="thinking",
                                    )
                                    follow_up_tool_steps.append(tool_step)
                                    investigation.steps.append(tool_step)
                                    if stream_steps:
                                        yield tool_step
                        
                        # Track propose_fix responses from follow-up (use same extraction logic as main flow)
                        function_responses = event.get_function_responses()
                        if function_responses:
                            for fr in function_responses:
                                tool_name = fr.name if hasattr(fr, 'name') else None
                                if tool_name and "propose_fix" in tool_name.lower():
                                    # Extract response - handle different response formats (same as main flow)
                                    if hasattr(fr, 'response'):
                                        raw_response = fr.response
                                        # If it's already a string, use it
                                        if isinstance(raw_response, str):
                                            response_text = raw_response
                                        # If it's a dict, try to extract the actual result
                                        elif isinstance(raw_response, dict):
                                            # Check for common ADK response formats
                                            if 'result' in raw_response:
                                                response_text = raw_response['result']
                                            elif 'content' in raw_response:
                                                response_text = raw_response['content']
                                            else:
                                                # Try to serialize to JSON
                                                response_text = json.dumps(raw_response)
                                        else:
                                            response_text = str(raw_response)
                                    else:
                                        response_text = str(fr)
                                    
                                    investigation.context["agentic_fix_response"] = response_text
                                    investigation.context["propose_fix_tool_name"] = tool_name
                                    investigation.context["propose_fix_response_length"] = len(response_text)
                                    
                                    # Update the tool step
                                    if follow_up_tool_steps:
                                        tool_step = follow_up_tool_steps[-1]
                                        tool_step.output_summary = f"Received response ({len(response_text)} chars)"
                                        tool_step.status = "success"
                                        if stream_steps:
                                            yield tool_step
                    
                    follow_up_step.output_summary = f"Follow-up completed"
                    follow_up_step.status = "success"
                    if stream_steps:
                        yield follow_up_step
                        
                except Exception as e:
                    follow_up_step.output_summary = f"Follow-up error: {str(e)}"
                    follow_up_step.status = "error"
                    if stream_steps:
                        yield follow_up_step
            
        except Exception as e:
            step.output_summary = f"Error: {str(e)}"
            step.status = "error"
            if stream_steps:
                yield step
            
            # Fallback to manual investigation
            investigation.diagnosis = f"Agent investigation failed: {str(e)}. Please review the test manually."
        
        # Step 3: Diagnosis Complete
        step = InvestigationStep(
            timestamp=datetime.now(),
            action="Generating diagnosis with AI",
            tool_name="gemini",
            input_summary="Analyzing all gathered context",
            output_summary="Diagnosis complete",
            status="success",
            details={"diagnosis_preview": investigation.diagnosis[:200] + "..." if len(investigation.diagnosis) > 200 else investigation.diagnosis}
        )
        investigation.steps.append(step)
        if stream_steps:
            yield step
        
        # Step 4: Extract agentic fix options or fall back to hardcoded generation
        agentic_fix_response = investigation.context.get("agentic_fix_response")
        use_agentic_fixes = False
        
        # Check if propose_fix was called at all
        propose_fix_called = any(
            "propose_fix" in (step.tool_name or "").lower() 
            for step in investigation.steps
        )
        
        if agentic_fix_response:
            # Try to extract fix options from the agent's propose_fix response
            try:
                # Handle different response formats
                fix_data = None
                
                if isinstance(agentic_fix_response, dict):
                    # Already a dict, use it directly
                    fix_data = agentic_fix_response
                elif isinstance(agentic_fix_response, str):
                    # Try to parse as JSON
                    try:
                        fix_data = json.loads(agentic_fix_response)
                    except json.JSONDecodeError:
                        # Might be a string representation of a dict (like "{'result': '...'}")
                        # Try to extract JSON from it
                        try:
                            # Try parsing as Python literal
                            parsed = ast.literal_eval(agentic_fix_response)
                            if isinstance(parsed, dict):
                                # If it has a 'result' key with a JSON string, parse that
                                if 'result' in parsed and isinstance(parsed['result'], str):
                                    fix_data = json.loads(parsed['result'])
                                else:
                                    fix_data = parsed
                        except (ValueError, SyntaxError):
                            # Last resort: try to find JSON in the string
                            # Look for JSON object pattern
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', agentic_fix_response)
                            if json_match:
                                try:
                                    fix_data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    raise ValueError("Could not parse response as JSON")
                            else:
                                raise ValueError("Could not parse response as JSON")
                else:
                    fix_data = agentic_fix_response
                
                # Extract the actual fix data - handle nested structures
                if isinstance(fix_data, dict):
                    # Check if the actual data is nested in a 'result' or 'data' field
                    if 'result' in fix_data and isinstance(fix_data['result'], str):
                        # Result is a JSON string, parse it
                        fix_data = json.loads(fix_data['result'])
                    elif 'data' in fix_data and isinstance(fix_data['data'], str):
                        # Data is a JSON string, parse it
                        inner_data = json.loads(fix_data['data'])
                        fix_data = {'status': fix_data.get('status', 'success'), 'data': inner_data}
                
                # Now check for the actual fix options
                if fix_data and fix_data.get("status") == "success":
                    options = None
                    if "data" in fix_data:
                        if isinstance(fix_data["data"], dict):
                            options = fix_data["data"].get("options")
                        elif isinstance(fix_data["data"], str):
                            # Data might be a JSON string
                            try:
                                data_dict = json.loads(fix_data["data"])
                                options = data_dict.get("options")
                            except json.JSONDecodeError:
                                pass
                    
                    if options and isinstance(options, list) and len(options) > 0:
                        investigation.fix_options = options
                        use_agentic_fixes = True
                        
                        step = InvestigationStep(
                            timestamp=datetime.now(),
                            action="Generated agentic fix recommendations",
                            tool_name="adk_propose_fix",
                            input_summary=f"Agent proposed fixes for {test_name}",
                            output_summary=f"Generated {len(investigation.fix_options)} agentic fix option(s)",
                            status="success",
                            details={"option_titles": [o.get("title", "Unknown") for o in investigation.fix_options]}
                        )
                    else:
                        # Log if no options found
                        step = InvestigationStep(
                            timestamp=datetime.now(),
                            action="Fix Generation Issues",
                            tool_name="adk_propose_fix",
                            input_summary="Validating agent proposals",
                            output_summary=f"Response parsed but no options found",
                            status="error",
                            details={
                                "status": fix_data.get("status") if fix_data else None,
                                "has_data": "data" in fix_data if fix_data else False,
                                "data_type": type(fix_data.get("data")).__name__ if fix_data and "data" in fix_data else None,
                                "has_options": bool(options) if options else False,
                                "raw_response_preview": str(agentic_fix_response)[:500]
                            }
                        )
                        investigation.steps.append(step)
                else:
                    # Log if status was not success or no options
                    step = InvestigationStep(
                        timestamp=datetime.now(),
                        action="Fix Generation Issues",
                        tool_name="adk_propose_fix",
                        input_summary="Validating agent proposals",
                        output_summary=f"Invalid response: {fix_data.get('message', 'No options provided') if fix_data else 'Could not parse response'}",
                        status="error",
                        details={
                            "raw_response": str(fix_data)[:500] if fix_data else str(agentic_fix_response)[:500],
                            "status": fix_data.get("status") if fix_data else None,
                            "has_data": "data" in fix_data if fix_data else False,
                            "has_options": bool(fix_data.get("data", {}).get("options")) if fix_data else False
                        }
                    )
                    investigation.steps.append(step)

            except (json.JSONDecodeError, TypeError, KeyError, ValueError, SyntaxError) as e:
                # Log parsing error
                step = InvestigationStep(
                    timestamp=datetime.now(),
                    action="Fix Generation Error",
                    tool_name="adk_propose_fix",
                    input_summary="Parsing tool response",
                    output_summary=f"Failed to parse fix options: {str(e)}",
                    status="error",
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "raw_response": str(agentic_fix_response)[:1000] if agentic_fix_response else "None",
                        "response_type": type(agentic_fix_response).__name__ if agentic_fix_response else None
                    }
                )
                investigation.steps.append(step)
        
        if not use_agentic_fixes:
            # If agent didn't propose fixes, log why
            if propose_fix_called:
                # Tool was called but response wasn't captured or was invalid
                step = InvestigationStep(
                    timestamp=datetime.now(),
                    action="Fix Generation",
                    tool_name="adk_propose_fix",
                    input_summary=f"Checking for agentic fix proposals",
                    output_summary="propose_fix was called but no valid fix options were extracted",
                    status="error",
                    details={
                        "propose_fix_called": True,
                        "has_response": bool(agentic_fix_response),
                        "tool_calls": [t.tool_name for t in tool_steps],
                        "option_titles": []
                    }
                )
            else:
                # Tool was never called
                step = InvestigationStep(
                    timestamp=datetime.now(),
                    action="Fix Generation",
                    tool_name="adk_agent",
                    input_summary=f"Checking for agentic fix proposals",
                    output_summary="Agent did not call propose_fix tool. No fix options generated.",
                    status="warning",
                    details={
                        "propose_fix_called": False,
                        "tool_calls": [t.tool_name for t in tool_steps],
                        "option_titles": [],
                        "note": "The agent should call propose_fix at the end of investigation to generate fix options."
                    }
                )
            
        investigation.steps.append(step)
        if stream_steps:
            yield step
        
        investigation.completed_at = datetime.now()
        self.investigations[test_id] = investigation

    def get_investigation(self, test_id: str) -> Optional[Investigation]:
        """Get a stored investigation by test ID."""
        return self.investigations.get(test_id)


def run_investigation(test_result: Dict[str, Any]) -> Investigation:

    agent = CopilotAgent()
    
    async def _run():
        async for step in agent.investigate(test_result, stream_steps=False):
            pass
        test_id = test_result.get("test_id", "unknown")
        return agent.investigations.get(test_id)
    
    return asyncio.run(_run())
