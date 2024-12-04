from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState

# class TrialPhase(Enum):
#     INITIALIZATION = "initialization"
#     EVIDENCE_COLLECTION = "evidence_collection"
#     ARGUMENT_EXCHANGE = "argument_exchange"
#     VERDICT = "verdict"
#     COMPLETED = "completed"

# @dataclass
# class AgentState:
#     """State management for each agent in the trial workflow"""
#     messages: List[BaseMessage] = field(default_factory=list)
#     next: str = "judge"  # Default to judge as the next agent
#     thought_step: Optional[int] = 0
    # cot_finished: bool = False
    # trial_phase: TrialPhase = TrialPhase.INITIALIZATION
    # current_speaker: Optional[str] = None
    # argument_count: int = 0

class AgentState(MessagesState):
    """State for each agent node in the graph"""
    next: str  # Where to route to next
    thought_step: Optional[int] = 0  # Current step in chain of thought
    caller: Optional[str] = None  # Who called the agent