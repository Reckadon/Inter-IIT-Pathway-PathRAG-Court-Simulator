from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseAgent, AgentState, AgentResponse

class ArgumentResponse(TypedDict):
    """Structured argument response"""
    argument: str
    evidence_used: List[str]
    needs_information: bool
    information_request: Optional[str]
    confidence: float

class LawyerAgent(BaseAgent):
    """Agent representing the defense counsel"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a skilled defense attorney. Your role is to:
        1. Analyze case details and evidence thoroughly
        2. Build strong arguments supporting your client
        3. Use relevant laws and precedents effectively
        4. Counter prosecution's arguments strategically
        5. Request additional information when needed
        
        For each argument:
        - Ground claims in available evidence
        - Cite specific laws and precedents
        - Address opposing arguments directly
        - Identify information gaps
        - Maintain professional conduct
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    def get_thought_steps(self) -> List[str]:
        """Get lawyer-specific chain of thought steps"""
        return [
            "1. Review case context and recent developments",
            "2. Analyze available evidence and precedents",
            "3. Identify defense strategy opportunities",
            "4. Evaluate information completeness",
            "5. Structure argument components",
            "6. Validate argument strength"
        ]
    
    async def process(self, state: AgentState) -> AgentResponse:
        """Process current state with lawyer-specific logic"""
        # Add defense context to state
        messages = state["messages"] + [
            SystemMessage(content="""
                As defense counsel, consider:
                1. Current evidence supporting your client
                2. Weaknesses in prosecution's arguments
                3. Relevant legal precedents and statutes
                4. Information gaps needing research
                5. Strategic timing of arguments
                
                Maintain focus on your client's interests while adhering to legal ethics.
            """)
        ]
        
        # Update state with defense context
        state["messages"] = messages
        
        # Process through chain of thought
        response = await super().process(state)
        
        # If chain of thought is complete, structure the argument
        if response["cot_finished"]:
            argument = self._structure_argument(response["messages"][-1].content)
            
            # Check if we need more information
            if argument["needs_information"]:
                response["next"] = "retriever"
                response["messages"].append(
                    HumanMessage(
                        content=f"Information Request: {argument['information_request']}",
                        name="lawyer"
                    )
                )
            else:
                response["next"] = "judge"
                response["messages"].append(
                    HumanMessage(
                        content=self._format_argument(argument),
                        name="lawyer"
                    )
                )
        
        return response
    
    def _determine_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine next agent based on argument needs"""
        argument = self._structure_argument(result["messages"][-1].content)
        return "retriever" if argument["needs_information"] else "judge"
    
    def _structure_argument(self, content: str) -> ArgumentResponse:
        """Parse and structure the argument from response content"""
        # Default argument structure
        argument: ArgumentResponse = {
            "argument": content,
            "evidence_used": [],
            "needs_information": False,
            "information_request": None,
            "confidence": 0.5
        }
        
        content_lower = content.lower()
        
        # Extract evidence references
        evidence_markers = ["evidence shows", "according to", "as demonstrated by", "records indicate"]
        for marker in evidence_markers:
            if marker in content_lower:
                # Extract evidence reference following the marker
                start_idx = content_lower.index(marker)
                end_idx = content.find(".", start_idx)
                if end_idx != -1:
                    evidence = content[start_idx:end_idx].strip()
                    argument["evidence_used"].append(evidence)
        
        # Check for information needs
        info_markers = ["need more information", "requires research", "additional evidence needed"]
        if any(marker in content_lower for marker in info_markers):
            argument["needs_information"] = True
            # Extract information request
            for marker in info_markers:
                if marker in content_lower:
                    start_idx = content_lower.index(marker)
                    end_idx = content.find(".", start_idx)
                    if end_idx != -1:
                        argument["information_request"] = content[start_idx:end_idx].strip()
                        break
        
        # Assess confidence
        confidence_markers = {
            "strongly believe": 0.9,
            "confident": 0.8,
            "suggest": 0.6,
            "possibly": 0.4,
            "uncertain": 0.3
        }
        for marker, conf in confidence_markers.items():
            if marker in content_lower:
                argument["confidence"] = conf
                break
        
        return argument
    
    def _format_argument(self, argument: ArgumentResponse) -> str:
        """Format structured argument for presentation"""
        formatted = []
        
        # Main argument
        formatted.append(argument["argument"])
        
        # Evidence citations
        if argument["evidence_used"]:
            formatted.append("\nEvidence cited:")
            for evidence in argument["evidence_used"]:
                formatted.append(f"- {evidence}")
        
        # Confidence indicator
        confidence_level = "High" if argument["confidence"] > 0.7 else \
                         "Moderate" if argument["confidence"] > 0.4 else "Low"
        formatted.append(f"\nConfidence Level: {confidence_level}")
        
        return "\n".join(formatted)