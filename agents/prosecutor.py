from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from .base import AgentState
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os
from langchain_core.messages.utils import get_buffer_string


from dotenv import load_dotenv
load_dotenv()


# class ProsecutorResponse(BaseModel):
#     """Structured prosecutor response"""
#     response: str = Field(description="The prosecutor's argument or response")
#     next_agent: Literal["self", "judge", "retriever"] = Field(
#         description="Next step in the legal process"
#     )

class ProsecutorAgent:
    """Agent representing the prosecution"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        self.llm = llm or ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
        self.tools = tools or []
        
        self.system_prompt = """You are a skilled prosecutor in a specialized AI-driven legal system. Your role is to present compelling arguments against the defendant while ensuring justice.

ROLE AND RESPONSIBILITIES:
1. Case Building
   - Analyze evidence thoroughly
   - Establish clear links to legal violations
   - Meet burden of proof requirements
   - Build systematic case progression

2. Evidence Management
   - Present evidence effectively
   - Validate evidence reliability
   - Request additional investigation when needed
   - Challenge defense evidence appropriately

3. Legal Application
   - Apply relevant laws accurately
   - Cite appropriate precedents
   - Meet procedural requirements
   - Maintain prosecution standards



PROSECUTION CRITERIA:
1. Evidence Strength
   - Is evidence sufficient for charges?
   - Are all elements proven?
   - Are sources reliable?
   - Is additional evidence needed?

2. Legal Framework
   - Are all legal elements addressed?
   - Are precedents applicable?
   - Is burden of proof met?

3. Strategic Considerations
   - Are defense arguments addressed?
   - Is timing appropriate?
   - Are weak points covered?

You will go through the following chain of thought steps:
1. EVIDENCE & CHARGE ANALYSIS
2. LEGAL FRAMEWORK DEVELOPMENT
3. ARGUMENT CONSTRUCTION
4. VALIDATION & STRENGTHENING

Do only the current step at a time.

Remember: Your goal is to ensure justice through effective prosecution while maintaining ethical standards."""

    def get_thought_steps(self) -> List[str]:
        """Get prosecutor-specific chain of thought steps"""
        return [
            "1. EVIDENCE & CHARGE ANALYSIS:\n" +
            "   - Review available evidence and case status\n" +
            "   - Match evidence to charge elements\n" +
            "   - Identify proof gaps\n" +
            "   - Assess defense's recent arguments",

            "2. LEGAL FRAMEWORK DEVELOPMENT:\n" +
            "   - Identify applicable laws and precedents\n" +
            "   - Structure legal requirements\n" +
            "   - Plan evidence presentation\n" +
            "   - set 'next_agent' as 'retriever'",

            "3. ARGUMENT CONSTRUCTION:\n" +
            "   - Build systematic prosecution case\n" +
            "   - Link evidence to legal elements\n" +
            "   - Address defense arguments\n" +
            "   - Strengthen weak points",

            "4. VALIDATION & STRENGTHENING:\n" +
            "   - Review argument completeness\n" +
            "   - Verify evidence citations\n" +
            "   - Assess burden of proof\n" +
            "   - Polish presentation"
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with prosecutor-specific logic"""
        
        messages = [
            {"role": "system", "content": self.system_prompt + "\n'current_task': " + self.get_thought_steps()[state["thought_step"]]}
        ] + state["messages"]

        result = self.llm.invoke(messages)
        
        if state["thought_step"] == 0 or state["thought_step"] == 2:
            response = {
                "messages": [HumanMessage(content=result.content, name="prosecutor")],
                "next": "self",
                "thought_step": state["thought_step"]+1,
                "caller": "prosecutor"
            }
        elif state["thought_step"] == 1 :
            response = {
                "messages": [HumanMessage(content=result.content, name="prosecutor")],
                "next": "retriever",
                "thought_step": 2
            }
        elif state["thought_step"] == 3:
            response = {
                "messages": [HumanMessage(content=result.content, name="prosecutor")],
                "next": "judge",
                "thought_step": 0
            }
        else:
            raise ValueError("Invalid thought step")
            
        return response
    


    # def _determine_next_agent(self, result: Dict[str, Any]) -> str:
    #     """Determine next agent based on prosecution needs"""
    #     prosecution_arg = self._structure_prosecution(result["messages"][-1].content)
    #     return "retriever" if prosecution_arg["needs_information"] else "judge"
    
    # def _structure_prosecution(self, content: str) -> ProsecutionArgument:
    #     """Parse and structure the prosecution argument"""
    #     # Default prosecution structure
    #     prosecution: ProsecutionArgument = {
    #         "argument": content,
    #         "evidence_cited": [],
    #         "legal_basis": [],
    #         "needs_information": False,
    #         "information_request": None,
    #         "strength_assessment": 0.5
    #     }
        
    #     content_lower = content.lower()
        
    #     # Extract evidence citations
    #     evidence_markers = ["evidence shows", "as proven by", "exhibits demonstrate", "witness testimony"]
    #     for marker in evidence_markers:
    #         if marker in content_lower:
    #             start_idx = content_lower.index(marker)
    #             end_idx = content.find(".", start_idx)
    #             if end_idx != -1:
    #                 evidence = content[start_idx:end_idx].strip()
    #                 prosecution["evidence_cited"].append(evidence)
        
    #     # Extract legal basis
    #     legal_markers = ["pursuant to", "under section", "statute requires", "law states"]
    #     for marker in legal_markers:
    #         if marker in content_lower:
    #             start_idx = content_lower.index(marker)
    #             end_idx = content.find(".", start_idx)
    #             if end_idx != -1:
    #                 legal_ref = content[start_idx:end_idx].strip()
    #                 prosecution["legal_basis"].append(legal_ref)
        
    #     # Check for information needs
    #     info_markers = ["require additional evidence", "need investigation", "further proof needed"]
    #     if any(marker in content_lower for marker in info_markers):
    #         prosecution["needs_information"] = True
    #         for marker in info_markers:
    #             if marker in content_lower:
    #                 start_idx = content_lower.index(marker)
    #                 end_idx = content.find(".", start_idx)
    #                 if end_idx != -1:
    #                     prosecution["information_request"] = content[start_idx:end_idx].strip()
    #                     break
        
    #     # Assess argument strength
    #     strength_markers = {
    #         "conclusively proves": 0.9,
    #         "strongly demonstrates": 0.8,
    #         "indicates": 0.6,
    #         "suggests": 0.5,
    #         "may show": 0.3
    #     }
    #     for marker, strength in strength_markers.items():
    #         if marker in content_lower:
    #             prosecution["strength_assessment"] = strength
    #             break
        
    #     return prosecution
    
    # def _format_prosecution(self, prosecution: ProsecutionArgument) -> str:
    #     """Format structured prosecution argument"""
    #     formatted = []
        
    #     # Main argument
    #     formatted.append(prosecution["argument"])
        
    #     # Legal basis
    #     if prosecution["legal_basis"]:
    #         formatted.append("\nLegal Basis:")
    #         for basis in prosecution["legal_basis"]:
    #             formatted.append(f"- {basis}")
        
    #     # Evidence citations
    #     if prosecution["evidence_cited"]:
    #         formatted.append("\nEvidence:")
    #         for evidence in prosecution["evidence_cited"]:
    #             formatted.append(f"- {evidence}")
        
    #     # Strength assessment
    #     strength_level = "Strong" if prosecution["strength_assessment"] > 0.7 else \
    #                     "Moderate" if prosecution["strength_assessment"] > 0.4 else "Weak"
    #     formatted.append(f"\nArgument Strength: {strength_level}")
        
    #     return "\n".join(formatted)