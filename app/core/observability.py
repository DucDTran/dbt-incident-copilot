"""
Langfuse Observability Integration for dbt Co-Work.

This module provides comprehensive tracing, metrics, and observability
using Langfuse for LLM operations.

Features:
- Automatic tracing of agent investigations
- Token usage tracking and cost estimation
- Latency metrics for tool calls
- Error tracking and debugging
- Generation metadata (model, temperature, etc.)

Usage:
    from app.core.observability import get_tracer, trace_generation
    
    # Get the global tracer
    tracer = get_tracer()
    
    # Trace an investigation
    with tracer.trace_investigation(test_id, test_name) as trace:
        trace.add_generation("investigator", prompt, response, usage)
        trace.add_tool_call("get_lineage", input_data, output_data)
    
    # Using decorator
    @trace_generation("diagnostician")
    async def generate_diagnosis(context):
        ...
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from enum import Enum

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class TraceStatus(str, Enum):
    """Status of a trace or span."""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Token usage statistics for a generation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class GenerationMetadata:
    """Metadata for a generation (LLM call)."""
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


@dataclass
class TraceSpan:
    """A span within a trace, representing a single operation."""
    name: str
    span_type: str  # "generation", "tool", "agent"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.RUNNING
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[TokenUsage] = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def complete(
        self,
        output_data: Any = None,
        token_usage: Optional[TokenUsage] = None,
        status: TraceStatus = TraceStatus.SUCCESS,
    ) -> None:
        """Mark span as complete."""
        self.end_time = datetime.now()
        self.output_data = output_data
        self.token_usage = token_usage
        self.status = status
    
    def fail(self, error: str) -> None:
        """Mark span as failed."""
        self.end_time = datetime.now()
        self.error = error
        self.status = TraceStatus.ERROR


@dataclass
class Trace:
    """A complete trace for an investigation or operation."""
    trace_id: str
    name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.RUNNING
    spans: list[TraceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get total duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    @property
    def total_tokens(self) -> TokenUsage:
        """Get total token usage across all spans."""
        total = TokenUsage()
        for span in self.spans:
            if span.token_usage:
                total = total + span.token_usage
        return total
    
    def add_span(self, span: TraceSpan) -> None:
        """Add a span to the trace."""
        self.spans.append(span)
    
    def complete(self, status: TraceStatus = TraceStatus.SUCCESS) -> None:
        """Mark trace as complete."""
        self.end_time = datetime.now()
        self.status = status
    
    def fail(self) -> None:
        """Mark trace as failed."""
        self.end_time = datetime.now()
        self.status = TraceStatus.ERROR


class LangfuseTracer:
    """
    Langfuse integration for observability and tracing.
    
    Provides comprehensive tracing of agent operations including:
    - Investigation traces with nested spans
    - Generation tracking with token usage
    - Tool call metrics
    - Error tracking
    
    When Langfuse is not configured, falls back to local logging.
    """
    
    _instance: Optional["LangfuseTracer"] = None
    
    def __init__(self) -> None:
        """Initialize Langfuse tracer."""
        from app.config import get_settings
        
        self._settings = get_settings()
        self._langfuse_client = None
        self._enabled = False
        self._local_traces: dict[str, Trace] = {}
        
        self._init_langfuse()
    
    def _init_langfuse(self) -> None:
        """Initialize Langfuse client if configured."""
        langfuse_settings = self._settings.langfuse
        
        if not langfuse_settings.enabled:
            logger.info("Langfuse tracing disabled")
            return
        
        try:
            from langfuse import Langfuse
            import os
            
            # Set environment variables for Langfuse SDK v3
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_settings.public_key or ""
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_settings.secret_key or ""
            os.environ["LANGFUSE_HOST"] = langfuse_settings.host
            
            self._langfuse_client = Langfuse()
            
            # Verify connection
            if self._langfuse_client.auth_check():
                self._enabled = True
                logger.info(f"Langfuse tracing enabled (host: {langfuse_settings.host})")
            else:
                logger.error("Langfuse authentication failed. Check your credentials.")
            
        except ImportError:
            logger.warning(
                "Langfuse package not installed. Install with: pip install langfuse"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
    
    @classmethod
    def get_instance(cls) -> "LangfuseTracer":
        """Get or create singleton tracer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance and cls._instance._langfuse_client:
            cls._instance._langfuse_client.flush()
        cls._instance = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled."""
        return self._enabled
    
    def create_trace(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Trace:
        """
        Create a new trace for an investigation.
        
        Args:
            trace_id: Unique trace identifier
            name: Human-readable trace name
            metadata: Additional metadata
            tags: Tags for filtering
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            New Trace object
        """
        trace = Trace(
            trace_id=trace_id,
            name=name,
            metadata=metadata or {},
            tags=tags or [],
            user_id=user_id,
            session_id=session_id,
        )
        
        self._local_traces[trace_id] = trace
        
        # Store the Langfuse span for this trace
        if self._enabled and self._langfuse_client:
            try:
                # Create a root span that acts as the trace in Langfuse v3
                langfuse_span = self._langfuse_client.start_observation(
                    as_type="span",
                    name=name,
                    input={"trace_id": trace_id},
                    metadata=metadata,
                )
                # Update trace attributes
                langfuse_span.update_trace(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )
                # Store the span reference
                trace.metadata["_langfuse_span"] = langfuse_span
                trace.metadata["_langfuse_trace_id"] = langfuse_span.trace_id
            except Exception as e:
                logger.error(f"Failed to create Langfuse trace: {e}")
        
        return trace
    
    def add_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        token_usage: Optional[TokenUsage] = None,
        metadata: Optional[dict[str, Any]] = None,
        generation_metadata: Optional[GenerationMetadata] = None,
    ) -> TraceSpan:
        """
        Add a generation (LLM call) to a trace.
        
        Args:
            trace_id: Parent trace ID
            name: Generation name (e.g., "investigator", "diagnostician")
            model: Model name
            input_text: Prompt/input text
            output_text: Response/output text
            token_usage: Token usage statistics
            metadata: Additional metadata
            generation_metadata: Model configuration
            
        Returns:
            Created TraceSpan
        """
        span = TraceSpan(
            name=name,
            span_type="generation",
            input_data=input_text,
            output_data=output_text,
            token_usage=token_usage,
            metadata={
                "model": model,
                **(metadata or {}),
                **(generation_metadata.__dict__ if generation_metadata else {}),
            },
        )
        span.complete(output_text, token_usage)
        
        # Add to local trace
        if trace_id in self._local_traces:
            self._local_traces[trace_id].add_span(span)
        
        # Send to Langfuse
        if self._enabled and self._langfuse_client:
            try:
                # Build usage dict for Langfuse v3
                usage_details = None
                if token_usage:
                    usage_details = {
                        "input": token_usage.prompt_tokens,
                        "output": token_usage.completion_tokens,
                        "total": token_usage.total_tokens,
                    }
                
                # Create a generation observation using the new API
                generation = self._langfuse_client.start_observation(
                    as_type="generation",
                    name=name,
                    model=model,
                    input=input_text,
                    metadata=span.metadata,
                )
                # Update with output and usage, then end
                generation.update(
                    output=output_text,
                    usage_details=usage_details,
                )
                generation.end()
            except Exception as e:
                logger.error(f"Failed to log generation to Langfuse: {e}")
        
        return span
    
    def add_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: Optional[float] = None,
        status: TraceStatus = TraceStatus.SUCCESS,
        error: Optional[str] = None,
    ) -> TraceSpan:
        """
        Add a tool call to a trace.
        
        Args:
            trace_id: Parent trace ID
            tool_name: Name of the tool called
            input_data: Tool input parameters
            output_data: Tool output
            duration_ms: Execution duration in milliseconds
            status: Call status
            error: Error message if failed
            
        Returns:
            Created TraceSpan
        """
        span = TraceSpan(
            name=tool_name,
            span_type="tool",
            input_data=input_data,
            output_data=output_data,
            status=status,
            error=error,
        )
        
        if duration_ms is not None:
            span.metadata["duration_ms"] = duration_ms
        
        if status == TraceStatus.ERROR:
            span.fail(error or "Unknown error")
        else:
            span.complete(output_data)
        
        # Add to local trace
        if trace_id in self._local_traces:
            self._local_traces[trace_id].add_span(span)
        
        # Send to Langfuse
        if self._enabled and self._langfuse_client:
            try:
                # Create a span for the tool call using new API
                tool_span = self._langfuse_client.start_observation(
                    as_type="span",
                    name=tool_name,
                    input=input_data,
                    metadata={"duration_ms": duration_ms} if duration_ms else None,
                )
                tool_span.update(
                    output=output_data,
                    level="ERROR" if status == TraceStatus.ERROR else "DEFAULT",
                    status_message=error if error else None,
                )
                tool_span.end()
            except Exception as e:
                logger.error(f"Failed to log tool call to Langfuse: {e}")
        
        return span
    
    def complete_trace(
        self,
        trace_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        output: Optional[Any] = None,
    ) -> Optional[Trace]:
        """
        Complete a trace.
        
        Args:
            trace_id: Trace ID to complete
            status: Final status
            output: Final output/result
            
        Returns:
            Completed Trace or None if not found
        """
        trace = self._local_traces.get(trace_id)
        if trace:
            trace.complete(status)
            if output:
                trace.metadata["output"] = output
        
        # Update Langfuse - end the root span
        if self._enabled and self._langfuse_client and trace:
            try:
                langfuse_span = trace.metadata.get("_langfuse_span")
                if langfuse_span:
                    langfuse_span.update(
                        output=output,
                        level="ERROR" if status == TraceStatus.ERROR else "DEFAULT",
                    )
                    langfuse_span.end()
                # Flush to ensure data is sent
                self._langfuse_client.flush()
            except Exception as e:
                logger.error(f"Failed to complete Langfuse trace: {e}")
        
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self._local_traces.get(trace_id)
    
    def get_trace_stats(self, trace_id: str) -> dict[str, Any]:
        """
        Get statistics for a trace.
        
        Returns:
            Dict with duration, token usage, span counts, etc.
        """
        trace = self._local_traces.get(trace_id)
        if not trace:
            return {}
        
        token_usage = trace.total_tokens
        
        return {
            "trace_id": trace_id,
            "name": trace.name,
            "status": trace.status.value,
            "duration_ms": trace.duration_ms,
            "span_count": len(trace.spans),
            "generation_count": sum(1 for s in trace.spans if s.span_type == "generation"),
            "tool_call_count": sum(1 for s in trace.spans if s.span_type == "tool"),
            "total_prompt_tokens": token_usage.prompt_tokens,
            "total_completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
            "error_count": sum(1 for s in trace.spans if s.status == TraceStatus.ERROR),
        }
    
    @contextmanager
    def trace_investigation(
        self,
        test_id: str,
        test_name: str,
        model_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Context manager for tracing an investigation.
        
        Args:
            test_id: Test identifier (used as trace ID)
            test_name: Test name
            model_name: dbt model name
            metadata: Additional metadata
            
        Yields:
            InvestigationTraceContext for adding spans
            
        Example:
            with tracer.trace_investigation(test_id, test_name) as ctx:
                ctx.add_generation("investigator", prompt, response, usage)
                ctx.add_tool_call("get_lineage", input, output)
        """
        trace = self.create_trace(
            trace_id=test_id,
            name=f"Investigation: {test_name}",
            metadata={
                "test_id": test_id,
                "test_name": test_name,
                "model_name": model_name,
                **(metadata or {}),
            },
            tags=["investigation", model_name] if model_name else ["investigation"],
        )
        
        ctx = InvestigationTraceContext(self, trace)
        
        try:
            yield ctx
            self.complete_trace(test_id, TraceStatus.SUCCESS)
        except Exception as e:
            self.complete_trace(test_id, TraceStatus.ERROR)
            raise
    
    def flush(self) -> None:
        """Flush any pending data to Langfuse."""
        if self._enabled and self._langfuse_client:
            try:
                self._langfuse_client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


class InvestigationTraceContext:
    """Context helper for investigation tracing."""
    
    def __init__(self, tracer: LangfuseTracer, trace: Trace) -> None:
        self._tracer = tracer
        self._trace = trace
    
    @property
    def trace_id(self) -> str:
        return self._trace.trace_id
    
    def add_generation(
        self,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        token_usage: Optional[TokenUsage] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceSpan:
        """Add a generation to this trace."""
        return self._tracer.add_generation(
            self._trace.trace_id,
            name,
            model,
            input_text,
            output_text,
            token_usage,
            metadata,
        )
    
    def add_tool_call(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: Optional[float] = None,
        status: TraceStatus = TraceStatus.SUCCESS,
        error: Optional[str] = None,
    ) -> TraceSpan:
        """Add a tool call to this trace."""
        return self._tracer.add_tool_call(
            self._trace.trace_id,
            tool_name,
            input_data,
            output_data,
            duration_ms,
            status,
            error,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get current trace statistics."""
        return self._tracer.get_trace_stats(self._trace.trace_id)


def get_tracer() -> LangfuseTracer:
    """Get the global Langfuse tracer instance."""
    return LangfuseTracer.get_instance()


def trace_generation(
    name: str,
    model: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace a generation function.
    
    Args:
        name: Generation name for tracing
        model: Model name (auto-detected if not provided)
        
    Example:
        @trace_generation("diagnostician")
        async def generate_diagnosis(trace_id: str, context: str) -> str:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            start_time = time.monotonic()
            
            # Get trace_id from kwargs or first arg
            trace_id = kwargs.get("trace_id") or (args[0] if args else "unknown")
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.monotonic() - start_time) * 1000
                
                # Log successful generation
                logger.debug(
                    f"Generation '{name}' completed",
                    extra={"trace_id": trace_id, "duration_ms": duration_ms}
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.error(
                    f"Generation '{name}' failed: {e}",
                    extra={"trace_id": trace_id, "duration_ms": duration_ms}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            start_time = time.monotonic()
            trace_id = kwargs.get("trace_id") or (args[0] if args else "unknown")
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.debug(
                    f"Generation '{name}' completed",
                    extra={"trace_id": trace_id, "duration_ms": duration_ms}
                )
                return result
            except Exception as e:
                logger.error(f"Generation '{name}' failed: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
