from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class TrialPhase(Enum):
    INITIALIZATION = "initialization"
    EVIDENCE_COLLECTION = "evidence_collection"
    ARGUMENT_EXCHANGE = "argument_exchange"
    VERDICT = "verdict"
    COMPLETED = "completed"

@dataclass
class TrialState:
    """State management for a single trial"""
    trial_id: str
    case_details: Dict[str, Any]
    phase: TrialPhase = TrialPhase.INITIALIZATION
    current_speaker: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    fact_checks: List[Dict[str, Any]] = field(default_factory=list)
    argument_count: int = 0
