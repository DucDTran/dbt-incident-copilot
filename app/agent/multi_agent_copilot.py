"""
Multi-Agent dbt Co-Work using Google ADK.

This module implements a multi-agent architecture to avoid output token limits
and improve investigation quality:

1. **Investigator Agent**: Gathers context using tools (lineage, SQL, schema, etc.)
2. **Diagnostician Agent**: Analyzes context and produces structured diagnosis  
3. **Fix Proposer Agent**: Generates fix options based on diagnosis

Each agent has a focused responsibility and smaller output requirements,
preventing token truncation issues.

Features:
- Specialized agents with clear responsibilities
- Context passing between agents
- Langfuse integration for tracing
- Rate limiting for API calls
- Cached embeddings for knowledge base
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import logging

from google import genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from app.config import get_settings
from app.core.observability import get_tracer, TokenUsage, TraceStatus
from app.core.rate_limiter import RateLimiter
from app.prompts.multi_agent_prompts import (
    INVESTIGATOR_SYSTEM_INSTRUCTION,
    DIAGNOSTICIAN_SYSTEM_INSTRUCTION,
    FIX_PROPOSER_SYSTEM_INSTRUCTION,
    get_investigator_prompt,
    get_diagnostician_prompt,
    get_fix_proposer_prompt,
)

# Import tools
from app.agent.tools import (
    tool_read_repo,
    get_model_lineage,
)
from app.agent.tools.agentic_fix_tool import adk_propose_fix as _adk_propose_fix
from app.agent.tools.knowledge_base_tool import search_for_business_rule
from app.agent.tools.repo_tool import find_file_by_model_name, find_schema_file
from app.agent.tools.sql_tool import adk_execute_sql
from app.agent.tools.elementary_tool import get_test_details

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class InvestigationStep:
    """A single step in the agent's investigation."""
    timestamp: datetime
    action: str
    tool_name: Optional[str]
    input_summary: str
    output_summary: str
    status: str  # 'success', 'error', 'thinking'
    agent: Optional[str] = None  # Which agent performed this step
    details: Optional[Dict[str, Any]] = None


@dataclass 
class InvestigationContext:
    """Context gathered during investigation."""
    test_details: Optional[str] = None
    schema_definition: Optional[str] = None
    sql_code: Optional[str] = None
    lineage: Optional[str] = None
    business_rules: Optional[str] = None
    data_samples: Optional[str] = None
    
    def to_summary(self) -> str:
        """Generate a text summary of all gathered context."""
        sections = []
        
        if self.test_details:
            sections.append(f"### Test Details\n{self.test_details}")
        
        if self.schema_definition:
            sections.append(f"### Schema Definition\n```yaml\n{self.schema_definition}\n```")
        
        if self.sql_code:
            sections.append(f"### SQL Code\n```sql\n{self.sql_code}\n```")
        
        if self.lineage:
            sections.append(f"### Model Lineage\n{self.lineage}")
        
        if self.business_rules:
            sections.append(f"### Business Rules\n{self.business_rules}")
        
        if self.data_samples:
            sections.append(f"### Data Samples\n{self.data_samples}")
        
        return "\n\n".join(sections)


@dataclass
class Diagnosis:
    """Structured diagnosis from the Diagnostician Agent."""
    root_cause: str = ""
    evidence: List[str] = field(default_factory=list)
    impact_assessment: str = ""
    category: str = ""  # schema_mismatch, data_quality, etc.
    severity: str = ""  # critical, high, medium, low
    confidence: str = ""  # high, medium, low
    raw_text: str = ""  # Full diagnosis text
    
    def to_text(self) -> str:
        """Convert to formatted text."""
        if self.raw_text:
            return self.raw_text
        
        text = f"### Root Cause\n{self.root_cause}\n\n"
        text += "### Evidence\n"
        for e in self.evidence:
            text += f"- {e}\n"
        text += f"\n### Impact Assessment\n{self.impact_assessment}"
        return text


@dataclass
class Investigation:
    """Complete investigation for a test failure."""
    test_result: Dict[str, Any]
    steps: List[InvestigationStep] = field(default_factory=list)
    context: InvestigationContext = field(default_factory=InvestigationContext)
    diagnosis: Optional[Diagnosis] = None
    fix_options: List[Dict[str, Any]] = field(default_factory=list)
    raw_context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Token tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


# ============================================================
# Tool Functions for Investigator Agent
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
    from app.agent.tools.knowledge_base_tool import get_knowledge_base
    kb = get_knowledge_base()
    
    search_query = f"{query} {context_column} {context_model}".strip()
    results = kb.search(search_query, top_k=5)
    
    if results:
        return json.dumps({
            "status": "success",
            "results": results
        }, indent=2, default=str)
    
    kb_result = search_for_business_rule(context_column, context_model, query)
    return json.dumps(kb_result, indent=2, default=str)


def adk_get_test_details(test_id: str) -> str:
    """
    Get detailed information about a specific test from Elementary.
    
    Args:
        test_id: The unique identifier of the test.
        
    Returns:
        Test details including error message, failed rows, and sample data.
    """
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
    
    Args:
        test_name: Name of the failing test.
        model_name: Name of the dbt model.
        column_name: Name of the column (if applicable).
        root_cause: The diagnosed root cause of the failure.
        fix_options: JSON string containing an array of fix options.
        
    Returns:
        JSON string with validated fix options ready for UI display.
    """
    return _adk_propose_fix(test_name, model_name, column_name, root_cause, fix_options)


# ============================================================
# Multi-Agent Copilot
# ============================================================

class MultiAgentCopilot:
    """
    Multi-agent dbt Co-Work for investigating test failures.
    
    Uses three specialized agents:
    1. Investigator: Gathers context using tools
    2. Diagnostician: Analyzes context and produces diagnosis
    3. Fix Proposer: Generates fix options
    
    This architecture prevents output token truncation by giving each
    agent a focused task with smaller output requirements.
    """
    
    def __init__(self) -> None:
        """Initialize the multi-agent copilot."""
        self.settings = get_settings()
        self.investigations: Dict[str, Investigation] = {}
        self._tracer = get_tracer()
        self._rate_limiter = RateLimiter.get_instance()
        
        # Initialize genai client
        self._client = genai.Client(api_key=self.settings.google_api_key)
        
        # Create specialized agents
        self._investigator = self._create_investigator_agent()
        self._diagnostician = self._create_diagnostician_agent()
        self._fix_proposer = self._create_fix_proposer_agent()
        
        # Session services for each agent
        self._session_service = InMemorySessionService()
        
        # Runners for each agent
        self._investigator_runner = Runner(
            agent=self._investigator,
            app_name="dbt-copilot-investigator",
            session_service=self._session_service,
        )
        self._diagnostician_runner = Runner(
            agent=self._diagnostician,
            app_name="dbt-copilot-diagnostician",
            session_service=self._session_service,
        )
        self._fix_proposer_runner = Runner(
            agent=self._fix_proposer,
            app_name="dbt-copilot-fix-proposer",
            session_service=self._session_service,
        )
    
    def _create_investigator_agent(self) -> LlmAgent:
        """Create the Investigator Agent with tools."""
        return LlmAgent(
            name="investigator",
            model=self.settings.agent.get_investigator_model(),
            instruction=INVESTIGATOR_SYSTEM_INSTRUCTION,
            tools=[
                adk_get_model_lineage,
                adk_read_model_sql,
                adk_read_schema_definition,
                adk_search_knowledge_base,
                adk_get_test_details,
                adk_execute_sql,
            ],
        )
    
    def _create_diagnostician_agent(self) -> LlmAgent:
        """Create the Diagnostician Agent (no tools needed)."""
        return LlmAgent(
            name="diagnostician",
            model=self.settings.agent.get_diagnostician_model(),
            instruction=DIAGNOSTICIAN_SYSTEM_INSTRUCTION,
            tools=[],  # Diagnostician doesn't use tools
        )
    
    def _create_fix_proposer_agent(self) -> LlmAgent:
        """Create the Fix Proposer Agent with propose_fix tool."""
        return LlmAgent(
            name="fix_proposer",
            model=self.settings.agent.get_fix_proposer_model(),
            instruction=FIX_PROPOSER_SYSTEM_INSTRUCTION,
            tools=[adk_propose_fix],
        )
    
    async def investigate(
        self,
        test_result: Dict[str, Any],
        stream_steps: bool = True,
    ) -> AsyncGenerator[InvestigationStep, None]:
        """
        Investigate a test failure using the multi-agent approach.
        
        Args:
            test_result: Test failure details from Elementary
            stream_steps: Whether to yield steps as they occur
            
        Yields:
            InvestigationStep objects as the investigation progresses
        """
        test_id = test_result.get("test_id", "unknown")
        test_name = test_result.get("test_name", "")
        model_name = test_result.get("model_name", "")
        column_name = test_result.get("column_name", "")
        error_message = test_result.get("error_message", "")
        failed_rows = test_result.get("failed_rows", 0)
        previous_fix_attempt = test_result.get("previous_fix_attempt", None)
        failed_fix_titles = test_result.get("failed_fix_titles", [])  # Track fixes that already failed
        
        # Create investigation record
        investigation = Investigation(test_result=test_result)
        self.investigations[test_id] = investigation
        
        # Start trace
        with self._tracer.trace_investigation(
            test_id=test_id,
            test_name=test_name,
            model_name=model_name,
            metadata={"column_name": column_name, "error_message": error_message},
        ) as trace_ctx:
            
            # ========================================
            # Phase 1: Investigation (Investigator Agent)
            # ========================================
            step = InvestigationStep(
                timestamp=datetime.now(),
                action="Starting investigation",
                tool_name=None,
                input_summary=f"Test: {test_name}",
                output_summary="Investigator Agent gathering context...",
                status="thinking",
                agent="investigator",
            )
            investigation.steps.append(step)
            if stream_steps:
                yield step
            
            try:
                investigation_context = await self._run_investigator(
                    test_id=test_id,
                    test_name=test_name,
                    model_name=model_name,
                    column_name=column_name,
                    error_message=error_message,
                    failed_rows=failed_rows,
                    investigation=investigation,
                    stream_steps=stream_steps,
                    trace_ctx=trace_ctx,
                    previous_fix_attempt=previous_fix_attempt,
                )
                
                # Yield tool call steps from investigator
                if stream_steps:
                    for inv_step in investigation.steps[-10:]:  # Last 10 steps
                        if inv_step.agent == "investigator" and inv_step.tool_name:
                            yield inv_step
                
                step.status = "success"
                step.output_summary = f"Context gathered: {len(investigation_context.to_summary())} chars"
                if stream_steps:
                    yield step
                
            except Exception as e:
                logger.error(f"Investigator failed: {e}")
                step.status = "error"
                step.output_summary = f"Investigation error: {str(e)}"
                if stream_steps:
                    yield step
                investigation_context = InvestigationContext()
            
            # ========================================
            # Phase 2: Diagnosis (Diagnostician Agent)
            # ========================================
            step = InvestigationStep(
                timestamp=datetime.now(),
                action="Analyzing findings",
                tool_name=None,
                input_summary="Context from investigation",
                output_summary="Diagnostician Agent analyzing...",
                status="thinking",
                agent="diagnostician",
            )
            investigation.steps.append(step)
            if stream_steps:
                yield step
            
            try:
                diagnosis = await self._run_diagnostician(
                    test_id=test_id,
                    test_name=test_name,
                    model_name=model_name,
                    column_name=column_name,
                    error_message=error_message,
                    failed_rows=failed_rows,
                    investigation_context=investigation_context,
                    trace_ctx=trace_ctx,
                )
                
                investigation.diagnosis = diagnosis
                step.status = "success"
                step.output_summary = f"Diagnosis complete: {diagnosis.category} ({diagnosis.severity})"
                step.details = {"root_cause_preview": diagnosis.root_cause[:200]}
                if stream_steps:
                    yield step
                
            except Exception as e:
                logger.error(f"Diagnostician failed: {e}")
                step.status = "error"
                step.output_summary = f"Diagnosis error: {str(e)}"
                if stream_steps:
                    yield step
                diagnosis = Diagnosis(
                    root_cause="Unable to diagnose - please review manually",
                    raw_text=f"Diagnosis failed: {str(e)}",
                )
                investigation.diagnosis = diagnosis
            
            # ========================================
            # Phase 3: Fix Proposals (Fix Proposer Agent)
            # ========================================
            step = InvestigationStep(
                timestamp=datetime.now(),
                action="Generating fix options",
                tool_name=None,
                input_summary="Diagnosis and context",
                output_summary="Fix Proposer Agent generating options...",
                status="thinking",
                agent="fix_proposer",
            )
            investigation.steps.append(step)
            if stream_steps:
                yield step
            
            try:
                fix_options = await self._run_fix_proposer(
                    test_id=test_id,
                    test_name=test_name,
                    model_name=model_name,
                    column_name=column_name,
                    error_message=error_message,
                    diagnosis=diagnosis,
                    investigation_context=investigation_context,
                    trace_ctx=trace_ctx,
                    failed_fix_titles=failed_fix_titles,
                )
                
                investigation.fix_options = fix_options
                step.status = "success"
                step.output_summary = f"Generated {len(fix_options)} fix options"
                step.details = {"option_titles": [f.get("title", "Unknown") for f in fix_options]}
                if stream_steps:
                    yield step
                
            except Exception as e:
                logger.error(f"Fix Proposer failed: {e}")
                step.status = "error"
                step.output_summary = f"Fix generation error: {str(e)}"
                if stream_steps:
                    yield step
            
            # ========================================
            # Complete Investigation
            # ========================================
            investigation.completed_at = datetime.now()
            
            final_step = InvestigationStep(
                timestamp=datetime.now(),
                action="Investigation complete",
                tool_name=None,
                input_summary="All phases completed",
                output_summary=f"Found {len(investigation.fix_options)} fix options",
                status="success",
                details={
                    "total_steps": len(investigation.steps),
                    "diagnosis_category": diagnosis.category if diagnosis else None,
                    "fix_count": len(investigation.fix_options),
                }
            )
            investigation.steps.append(final_step)
            if stream_steps:
                yield final_step
    
    async def _run_investigator(
        self,
        test_id: str,
        test_name: str,
        model_name: str,
        column_name: str,
        error_message: str,
        failed_rows: int,
        investigation: Investigation,
        stream_steps: bool,
        trace_ctx,
        previous_fix_attempt: str = None,
    ) -> InvestigationContext:
        """Run the Investigator Agent to gather context."""
        
        # Rate limit check
        await self._rate_limiter.acquire("gemini")
        
        # Create investigation prompt
        prompt = get_investigator_prompt(
            test_name=test_name,
            model_name=model_name,
            column_name=column_name,
            error_message=error_message,
            failed_rows=failed_rows,
            test_id=test_id,
            previous_fix_attempt=previous_fix_attempt,
        )
        
        # Create session for this investigation
        session = await self._session_service.create_session(
            app_name="dbt-copilot-investigator",
            user_id=f"investigation_{test_id}",
        )
        
        context = InvestigationContext()
        response_text = ""
        
        # Run investigator and track tool calls
        async for event in self._investigator_runner.run_async(
            user_id=f"investigation_{test_id}",
            session_id=session.id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)]
            ),
        ):
            # Track tool calls
            function_calls = event.get_function_calls()
            if function_calls:
                for fc in function_calls:
                    step = InvestigationStep(
                        timestamp=datetime.now(),
                        action=f"Calling {fc.name}",
                        tool_name=fc.name,
                        input_summary=str(fc.args)[:100] if fc.args else "",
                        output_summary="",
                        status="thinking",
                        agent="investigator",
                    )
                    investigation.steps.append(step)
            
            # Track tool responses
            function_responses = event.get_function_responses()
            if function_responses:
                for fr in function_responses:
                    tool_name = fr.name if hasattr(fr, 'name') else None
                    response = fr.response if hasattr(fr, 'response') else str(fr)
                    
                    if isinstance(response, dict):
                        response_str = json.dumps(response)
                    else:
                        response_str = str(response)
                    
                    # Store context based on tool
                    if tool_name:
                        tool_lower = tool_name.lower()
                        if "lineage" in tool_lower:
                            context.lineage = response_str
                        elif "sql" in tool_lower or "read_model" in tool_lower:
                            context.sql_code = response_str
                        elif "schema" in tool_lower:
                            context.schema_definition = response_str
                        elif "knowledge" in tool_lower or "search" in tool_lower:
                            context.business_rules = response_str
                        elif "test_details" in tool_lower:
                            context.test_details = response_str
                        elif "execute_sql" in tool_lower:
                            context.data_samples = response_str
                    
                    # Update step
                    for step in reversed(investigation.steps):
                        if step.tool_name and tool_name and tool_name in step.tool_name:
                            step.status = "success"
                            step.output_summary = f"Received {len(response_str)} chars"
                            step.details = {"response_preview": response_str[:200]}
                            
                            # Log to trace
                            trace_ctx.add_tool_call(
                                tool_name=tool_name,
                                input_data=step.input_summary,
                                output_data=response_str[:500],
                            )
                            break
            
            # Capture final response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
        
        # Store raw context
        investigation.raw_context["investigation_response"] = response_text
        investigation.context = context
        
        return context
    
    async def _run_diagnostician(
        self,
        test_id: str,
        test_name: str,
        model_name: str,
        column_name: str,
        error_message: str,
        failed_rows: int,
        investigation_context: InvestigationContext,
        trace_ctx,
    ) -> Diagnosis:
        """Run the Diagnostician Agent to analyze findings."""
        
        # Rate limit check
        await self._rate_limiter.acquire("gemini")
        
        # Create diagnosis prompt
        prompt = get_diagnostician_prompt(
            test_name=test_name,
            model_name=model_name,
            column_name=column_name,
            error_message=error_message,
            failed_rows=failed_rows,
            investigation_context=investigation_context.to_summary(),
        )
        
        # Create session
        session = await self._session_service.create_session(
            app_name="dbt-copilot-diagnostician",
            user_id=f"diagnosis_{test_id}",
        )
        
        response_text = ""
        
        # Run diagnostician
        async for event in self._diagnostician_runner.run_async(
            user_id=f"diagnosis_{test_id}",
            session_id=session.id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)]
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
        
        # Parse diagnosis from response
        diagnosis = self._parse_diagnosis(response_text)
        
        # Log generation to trace
        trace_ctx.add_generation(
            name="diagnostician",
            model=self.settings.agent.get_diagnostician_model(),
            input_text=prompt[:500],
            output_text=response_text[:500],
        )
        
        return diagnosis
    
    async def _run_fix_proposer(
        self,
        test_id: str,
        test_name: str,
        model_name: str,
        column_name: str,
        error_message: str,
        diagnosis: Diagnosis,
        investigation_context: InvestigationContext,
        trace_ctx,
        failed_fix_titles: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the Fix Proposer Agent to generate fix options."""
        
        # Rate limit check
        await self._rate_limiter.acquire("gemini")
        
        # Create fix proposal prompt
        prompt = get_fix_proposer_prompt(
            test_name=test_name,
            model_name=model_name,
            column_name=column_name,
            error_message=error_message,
            diagnosis=diagnosis.to_text(),
            schema_definition=investigation_context.schema_definition or "",
            sql_code=investigation_context.sql_code or "",
            business_rules=investigation_context.business_rules or "",
            failed_fix_titles=failed_fix_titles or [],
        )
        
        # Create session
        session = await self._session_service.create_session(
            app_name="dbt-copilot-fix-proposer",
            user_id=f"fix_{test_id}",
        )
        
        fix_options = []
        response_text = ""
        
        logger.info(f"Running Fix Proposer for test {test_id}")
        
        # Run fix proposer
        async for event in self._fix_proposer_runner.run_async(
            user_id=f"fix_{test_id}",
            session_id=session.id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)]
            ),
        ):
            # Track function calls (tool invocations)
            function_calls = event.get_function_calls()
            if function_calls:
                for fc in function_calls:
                    logger.info(f"Fix Proposer function call detected: name={getattr(fc, 'name', 'unknown')}")
                    # Log the ACTUAL arguments the agent is passing
                    if hasattr(fc, 'args'):
                        logger.info(f"Function call args: {fc.args}")
            
            # Track function responses (tool results)
            function_responses = event.get_function_responses()
            if function_responses:
                for fr in function_responses:
                    tool_name = fr.name if hasattr(fr, 'name') else None
                    logger.debug(f"Fix Proposer function response: tool_name={tool_name}")
                    
                    try:
                        raw_response = getattr(fr, 'response', None)
                        
                        if raw_response is None:
                            logger.warning("fr.response is None")
                            continue
                        
                        # Parse the response
                        if isinstance(raw_response, str):
                            fix_data = json.loads(raw_response)
                        elif isinstance(raw_response, dict):
                            fix_data = raw_response
                        else:
                            logger.warning(f"Unexpected response type: {type(raw_response)}")
                            try:
                                fix_data = json.loads(str(raw_response))
                            except:
                                continue
                        
                        # ADK wraps tool responses in {"result": "..."} - unwrap it
                        if "result" in fix_data and len(fix_data) == 1:
                            inner = fix_data["result"]
                            if isinstance(inner, str):
                                fix_data = json.loads(inner)
                            elif isinstance(inner, dict):
                                fix_data = inner
                        
                        if fix_data.get("status") == "success":
                            data = fix_data.get("data", {})
                            options = data.get("options", [])
                            logger.info(f"Fix Proposer generated {len(options)} options")
                            
                            if options:
                                fix_options = options
                                trace_ctx.add_tool_call(
                                    tool_name="adk_propose_fix",
                                    input_data={"test_name": test_name},
                                    output_data=f"Generated {len(options)} options",
                                )
                            else:
                                logger.warning("Options array is empty - agent may have passed invalid fix_options")
                        else:
                            logger.warning(f"Tool returned status={fix_data.get('status')}, message={fix_data.get('message')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                    except Exception as e:
                        logger.error(f"Error processing response: {e}", exc_info=True)
            
            # Capture text response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
        
        # If no fix options from tool call, try to parse from response text
        if not fix_options and response_text:
            logger.info("No fix options from tool call, attempting to parse from response text")
            parsed_options = self._parse_fix_options_from_text(response_text)
            if parsed_options:
                fix_options = parsed_options
                logger.info(f"Extracted {len(fix_options)} fix options from response text")
        
        # Log generation
        trace_ctx.add_generation(
            name="fix_proposer",
            model=self.settings.agent.get_fix_proposer_model(),
            input_text=prompt[:500],
            output_text=response_text[:500] if response_text else f"Generated {len(fix_options)} fixes",
        )
        
        logger.info(f"Fix Proposer completed with {len(fix_options)} options")
        return fix_options
    
    def _parse_fix_options_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback parser to extract fix options from agent's text response.
        
        This handles cases where the agent describes fixes in text rather than
        calling the adk_propose_fix tool.
        """
        import re
        
        fix_options = []
        
        # Try to find JSON blocks in the response
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        json_matches = re.findall(json_pattern, text)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                # Check if it's a fix options structure
                if 'fix_options' in data:
                    for opt in data['fix_options']:
                        if opt.get('title'):
                            fix_options.append(opt)
                elif data.get('title') and data.get('fix_type'):
                    # Single fix option
                    fix_options.append(data)
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, try to parse structured text
        if not fix_options:
            # Look for numbered fix options
            fix_pattern = r'(?:#{1,3}\s*)?(?:Option|Fix)\s*(\d+)[:\s]*([^\n]+)\n([\s\S]*?)(?=(?:#{1,3}\s*)?(?:Option|Fix)\s*\d+|$)'
            matches = re.findall(fix_pattern, text, re.IGNORECASE)
            
            for idx, (num, title, content) in enumerate(matches):
                fix_type = "schema" if any(kw in content.lower() for kw in ['schema', 'accepted_values', 'severity']) else "sql"
                
                fix_options.append({
                    "id": f"fix_{num}",
                    "title": title.strip(),
                    "description": content.strip()[:500],
                    "fix_type": fix_type,
                    "rationale": "",
                    "pros": "",
                    "cons": "",
                    "when_appropriate": "",
                })
        
        return fix_options
    
    def _parse_diagnosis(self, text: str) -> Diagnosis:
        """Parse structured diagnosis from agent response."""
        import re
        
        diagnosis = Diagnosis(raw_text=text)
        
        # Extract Root Cause
        root_cause_match = re.search(
            r'###?\s*Root\s*Cause\s*\n(.*?)(?=###?|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if root_cause_match:
            diagnosis.root_cause = root_cause_match.group(1).strip()
        
        # Extract Evidence
        evidence_match = re.search(
            r'###?\s*Evidence\s*\n(.*?)(?=###?|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            diagnosis.evidence = [
                e.strip().lstrip('-').strip()
                for e in evidence_text.split('\n')
                if e.strip() and e.strip() != '-'
            ]
        
        # Extract Impact Assessment
        impact_match = re.search(
            r'###?\s*Impact\s*Assessment\s*\n(.*?)(?=###?|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if impact_match:
            diagnosis.impact_assessment = impact_match.group(1).strip()
        
        # Extract Classification
        category_match = re.search(
            r'\*\*Category\*\*:\s*(\w+)',
            text,
            re.IGNORECASE
        )
        if category_match:
            diagnosis.category = category_match.group(1).lower()
        
        severity_match = re.search(
            r'\*\*Severity\*\*:\s*(\w+)',
            text,
            re.IGNORECASE
        )
        if severity_match:
            diagnosis.severity = severity_match.group(1).lower()
        
        confidence_match = re.search(
            r'\*\*Confidence\*\*:\s*(\w+)',
            text,
            re.IGNORECASE
        )
        if confidence_match:
            diagnosis.confidence = confidence_match.group(1).lower()
        
        return diagnosis
    
    def get_investigation(self, test_id: str) -> Optional[Investigation]:
        """Get a stored investigation by test ID."""
        return self.investigations.get(test_id)


# ============================================================
# Convenience Functions
# ============================================================

def create_multi_agent_copilot() -> MultiAgentCopilot:
    """Create a new MultiAgentCopilot instance."""
    return MultiAgentCopilot()


async def run_multi_agent_investigation(
    test_result: Dict[str, Any],
) -> Investigation:
    """
    Run a complete investigation using the multi-agent approach.
    
    Args:
        test_result: Test failure details
        
    Returns:
        Completed Investigation object
    """
    copilot = create_multi_agent_copilot()
    
    async for step in copilot.investigate(test_result, stream_steps=False):
        pass  # Consume all steps
    
    test_id = test_result.get("test_id", "unknown")
    return copilot.get_investigation(test_id)
