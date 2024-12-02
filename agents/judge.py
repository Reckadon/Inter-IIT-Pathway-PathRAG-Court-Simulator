from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseAgent, AgentState, AgentResponse

class JudgeDecision(TypedDict):
    """Judge's decision on next steps"""
    next_agent: Literal["lawyer", "prosecutor", "retriever", "END"]
    reasoning: str
    fact_check: Optional[Dict[str, Any]]

class JudgeAgent(BaseAgent):
    """Agent representing the judge who manages the trial flow"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are an impartial judge managing a legal trial. Your role is to:
        1. Ensure fair trial proceedings by alternating between lawyer and prosecutor
        2. Fact-check arguments thoroughly using evidence
        3. Request additional information when needed
        4. Keep track of argument strength and validity
        5. Make a final verdict when sufficient arguments have been presented
        
        For each argument you review:
        - Assess factual accuracy against evidence
        - Check logical consistency
        - Evaluate if more information is needed
        - Determine if counter-arguments are needed
        - Consider if the trial is ready for verdict
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    def get_thought_steps(self) -> List[str]:
        """Get judge-specific chain of thought steps"""
        return [
            "1. Review current trial state and latest arguments",
            "2. Assess factual accuracy and evidence support",
            "3. Evaluate argument completeness",
            "4. Consider need for counter-arguments",
            "5. Determine next steps",
            "6. Formulate response and direction"
        ]
    
    async def process(self, state: AgentState) -> AgentResponse:
        """Process current state with judge-specific logic"""
        # Add judge context to state
        messages = state["messages"] + [
            SystemMessage(content="""
                Consider the trial state carefully. You must decide:
                1. If the latest argument needs fact-checking
                2. If more evidence is needed (call retriever)
                3. Which agent should speak next (lawyer/prosecutor)
                4. If the trial is ready for a verdict
                
                Maintain trial fairness by alternating speakers unless there's a strong reason not to.
            """)
        ]
        
        # Update state with judge context
        state["messages"] = messages
        
        # Process through chain of thought
        response = await super().process(state)
        
        # If chain of thought is complete, parse decision
        if response["cot_finished"]:
            decision = self._parse_judge_decision(response["messages"][-1].content)
            response["next"] = decision["next_agent"]
            
            # Add fact check if performed
            if decision["fact_check"]:
                response["messages"].append(
                    HumanMessage(
                        content=f"Fact Check Results: {decision['fact_check']}",
                        name="judge"
                    )
                )
        
        return response
    
    def _determine_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine next agent based on judge's decision"""
        # Extract decision from last message
        decision = self._parse_judge_decision(result["messages"][-1].content)
        return decision["next_agent"]
    
    def _parse_judge_decision(self, content: str) -> JudgeDecision:
        """Parse judge's decision from response content"""
        # Default decision structure
        decision: JudgeDecision = {
            "next_agent": "lawyer",  # Default to lawyer if unclear
            "reasoning": "",
            "fact_check": None
        }
        
        content_lower = content.lower()
        
        # Check for fact-check indicators
        if any(term in content_lower for term in ["fact check", "verify", "accuracy"]):
            decision["fact_check"] = {
                "validity": self._assess_validity(content),
                "feedback": content
            }
        
        # Check for verdict readiness
        if any(term in content_lower for term in ["conclude", "verdict", "decision", "ruling"]):
            decision["next_agent"] = "END"
            decision["reasoning"] = "Trial ready for verdict"
            return decision
        
        # Check for retriever need
        if any(term in content_lower for term in ["need information", "more evidence", "research"]):
            decision["next_agent"] = "retriever"
            decision["reasoning"] = "Additional information required"
            return decision
        
        # Determine next speaker
        if "prosecutor" in content_lower:
            decision["next_agent"] = "prosecutor"
        elif "lawyer" in content_lower:
            decision["next_agent"] = "lawyer"
            
        decision["reasoning"] = content
        return decision
    
    def _assess_validity(self, content: str) -> float:
        """Assess validity score from content"""
        if "invalid" in content.lower():
            return 0.0
        elif "partially valid" in content.lower():
            return 0.5
        elif "valid" in content.lower():
            return 1.0
        return 0.5  # Default to partial validity if unclear