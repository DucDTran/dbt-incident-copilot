"""
Agent module for dbt Co-Work.

Contains the ADK agent implementations:
- CopilotAgent: Single-agent implementation (legacy)
- MultiAgentCopilot: Multi-agent implementation (recommended)

The multi-agent approach prevents output token truncation by splitting
the investigation into specialized agents:
1. Investigator: Gathers context using tools
2. Diagnostician: Analyzes context and produces diagnosis
3. Fix Proposer: Generates fix options
"""

from app.agent.copilot_agent import CopilotAgent, run_investigation
from app.agent.multi_agent_copilot import (
    MultiAgentCopilot,
    create_multi_agent_copilot,
    run_multi_agent_investigation,
    Investigation,
    InvestigationStep,
    InvestigationContext,
    Diagnosis,
)

__all__ = [
    # Single-agent (legacy)
    "CopilotAgent",
    "run_investigation",
    # Multi-agent (recommended)
    "MultiAgentCopilot",
    "create_multi_agent_copilot",
    "run_multi_agent_investigation",
    # Data classes
    "Investigation",
    "InvestigationStep",
    "InvestigationContext",
    "Diagnosis",
]

