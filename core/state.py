from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.messages import BaseMessage

class TrialPhase(Enum):
    INITIALIZATION = "initialization"
    EVIDENCE_COLLECTION = "evidence_collection"
    ARGUMENT_EXCHANGE = "argument_exchange"
    VERDICT = "verdict"
    COMPLETED = "completed"

@dataclass
class AgentState:
    """State management for each agent in the trial workflow"""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = "judge"  # Default to judge as the next agent
    thought_step: Optional[str] = None
    cot_finished: bool = False
    trial_phase: TrialPhase = TrialPhase.INITIALIZATION
    current_speaker: Optional[str] = None
    argument_count: int = 0
