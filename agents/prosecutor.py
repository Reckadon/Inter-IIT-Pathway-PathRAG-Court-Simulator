from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseAgent, AgentState, AgentResponse

class ProsecutionArgument(TypedDict):
    """Structured prosecution argument"""
    argument: str
    evidence_cited: List[str]
    legal_basis: List[str]
    needs_information: bool
    information_request: Optional[str]
    strength_assessment: float

class ProsecutorAgent(BaseAgent):
    """Agent representing the prosecution"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a prosecutor representing the state. Your role is to:
        1. Present evidence against the defendant systematically
        2. Build compelling arguments based on facts and law
        3. Challenge defense claims with evidence
        4. Ensure all legal requirements are met
        5. Maintain burden of proof standards
        
        For each argument:
        - Link evidence to specific charges
        - Cite relevant statutes and precedents
        - Address defense counter-arguments
        - Identify evidence gaps
        - Maintain ethical prosecution standards
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    def get_thought_steps(self) -> List[str]:
        """Get prosecutor-specific chain of thought steps"""
        return [
            "1. Review case evidence and defense claims",
            "2. Identify key legal elements to prove",
            "3. Analyze evidence strength for each element",
            "4. Evaluate counter-argument needs",
            "5. Structure prosecution argument",
            "6. Validate legal sufficiency"
        ]
    
    async def process(self, state: AgentState) -> AgentResponse:
        """Process current state with prosecutor-specific logic"""
        # Add prosecution context to state
        messages = state["messages"] + [
            SystemMessage(content="""
                As prosecutor, focus on:
                1. Elements of the charged offenses
                2. Evidence supporting each element
                3. Addressing defense challenges
                4. Meeting burden of proof
                5. Identifying evidence gaps
                
                Maintain ethical prosecution while pursuing justice.
            """)
        ]
        
        # Update state with prosecution context
        state["messages"] = messages
        
        # Process through chain of thought
        response = await super().process(state)
        
        # If chain of thought is complete, structure the argument
        if response["cot_finished"]:
            prosecution_arg = self._structure_prosecution(response["messages"][-1].content)
            
            # Check if we need more information
            if prosecution_arg["needs_information"]:
                response["next"] = "retriever"
                response["messages"].append(
                    HumanMessage(
                        content=f"Information Request: {prosecution_arg['information_request']}",
                        name="prosecutor"
                    )
                )
            else:
                response["next"] = "judge"
                response["messages"].append(
                    HumanMessage(
                        content=self._format_prosecution(prosecution_arg),
                        name="prosecutor"
                    )
                )
        
        return response
    
    def _determine_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine next agent based on prosecution needs"""
        prosecution_arg = self._structure_prosecution(result["messages"][-1].content)
        return "retriever" if prosecution_arg["needs_information"] else "judge"
    
    def _structure_prosecution(self, content: str) -> ProsecutionArgument:
        """Parse and structure the prosecution argument"""
        # Default prosecution structure
        prosecution: ProsecutionArgument = {
            "argument": content,
            "evidence_cited": [],
            "legal_basis": [],
            "needs_information": False,
            "information_request": None,
            "strength_assessment": 0.5
        }
        
        content_lower = content.lower()
        
        # Extract evidence citations
        evidence_markers = ["evidence shows", "as proven by", "exhibits demonstrate", "witness testimony"]
        for marker in evidence_markers:
            if marker in content_lower:
                start_idx = content_lower.index(marker)
                end_idx = content.find(".", start_idx)
                if end_idx != -1:
                    evidence = content[start_idx:end_idx].strip()
                    prosecution["evidence_cited"].append(evidence)
        
        # Extract legal basis
        legal_markers = ["pursuant to", "under section", "statute requires", "law states"]
        for marker in legal_markers:
            if marker in content_lower:
                start_idx = content_lower.index(marker)
                end_idx = content.find(".", start_idx)
                if end_idx != -1:
                    legal_ref = content[start_idx:end_idx].strip()
                    prosecution["legal_basis"].append(legal_ref)
        
        # Check for information needs
        info_markers = ["require additional evidence", "need investigation", "further proof needed"]
        if any(marker in content_lower for marker in info_markers):
            prosecution["needs_information"] = True
            for marker in info_markers:
                if marker in content_lower:
                    start_idx = content_lower.index(marker)
                    end_idx = content.find(".", start_idx)
                    if end_idx != -1:
                        prosecution["information_request"] = content[start_idx:end_idx].strip()
                        break
        
        # Assess argument strength
        strength_markers = {
            "conclusively proves": 0.9,
            "strongly demonstrates": 0.8,
            "indicates": 0.6,
            "suggests": 0.5,
            "may show": 0.3
        }
        for marker, strength in strength_markers.items():
            if marker in content_lower:
                prosecution["strength_assessment"] = strength
                break
        
        return prosecution
    
    def _format_prosecution(self, prosecution: ProsecutionArgument) -> str:
        """Format structured prosecution argument"""
        formatted = []
        
        # Main argument
        formatted.append(prosecution["argument"])
        
        # Legal basis
        if prosecution["legal_basis"]:
            formatted.append("\nLegal Basis:")
            for basis in prosecution["legal_basis"]:
                formatted.append(f"- {basis}")
        
        # Evidence citations
        if prosecution["evidence_cited"]:
            formatted.append("\nEvidence:")
            for evidence in prosecution["evidence_cited"]:
                formatted.append(f"- {evidence}")
        
        # Strength assessment
        strength_level = "Strong" if prosecution["strength_assessment"] > 0.7 else \
                        "Moderate" if prosecution["strength_assessment"] > 0.4 else "Weak"
        formatted.append(f"\nArgument Strength: {strength_level}")
        
        return "\n".join(formatted)