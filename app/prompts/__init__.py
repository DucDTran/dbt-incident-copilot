"""
Centralized prompt management for dbt Co-Work.

All prompts are stored in this module for easy management and version tracking.

This module provides:
- Single-agent prompts (legacy): COPILOT_SYSTEM_INSTRUCTION
- Multi-agent prompts: INVESTIGATOR_*, DIAGNOSTICIAN_*, FIX_PROPOSER_*
"""

from app.prompts.agent_prompts import (
    COPILOT_SYSTEM_INSTRUCTION,
    get_investigation_prompt,
)
from app.prompts.fix_prompts import (
    get_fix_enhancement_prompt,
)
from app.prompts.multi_agent_prompts import (
    INVESTIGATOR_SYSTEM_INSTRUCTION,
    DIAGNOSTICIAN_SYSTEM_INSTRUCTION,
    FIX_PROPOSER_SYSTEM_INSTRUCTION,
    get_investigator_prompt,
    get_diagnostician_prompt,
    get_fix_proposer_prompt,
)

__all__ = [
    # Single-agent (legacy)
    "COPILOT_SYSTEM_INSTRUCTION",
    "get_investigation_prompt",
    "get_fix_enhancement_prompt",
    # Multi-agent
    "INVESTIGATOR_SYSTEM_INSTRUCTION",
    "DIAGNOSTICIAN_SYSTEM_INSTRUCTION",
    "FIX_PROPOSER_SYSTEM_INSTRUCTION",
    "get_investigator_prompt",
    "get_diagnostician_prompt",
    "get_fix_proposer_prompt",
]
