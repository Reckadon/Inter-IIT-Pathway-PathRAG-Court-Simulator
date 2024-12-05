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
        
        self.system_prompt = """
"You are a professional prosecutor advocating for the opposing party in a courtroom simulation. Your role is to challenge the defense's arguments and present evidence and laws to support the prosecution's case."
"Analyze and counter the defense lawyer's claims effectively, relying on factual accuracy, logical reasoning, and legal provisions."
"Collaborate with the Law Retriever to extract relevant laws and case studies and use the Web Searcher to source additional factual or contextual evidence."
"Your arguments should reflect high levels of objectivity, clarity, and persuasive reasoning, adhering to the principles of fairness and justice."
"Respond to the judge's observations or corrections with due diligence and adapt your arguments to maintain a strong prosecutorial stance."

you will go through the following chain of thought steps:
1. Review current state and plan a strategy
2. Identify the legal information needed to support the argument
3. Assess if information from the web is required
4. Argument construction

Do only current task at a time. Do not confuse with precedent cases. Avoid very long responses.
"""

    def get_thought_steps(self) -> List[str]:
        """Get prosecutor-specific chain of thought steps"""
        return [
            "1. Review the case files, analyze the user's arguments to identify weaknesses or inconsistencies in their claims and plan a strategy to build strong arguments against the defendant, ensuring they are logically sound and factually supported.",
            "2. Determine the specific legal information(e.g., laws, IPCs, precedents) required to strengthen your arguments or refute the user's points. Clearly ask the law retriever agent for the necessary details.",
            "3. Evaluate if additional web-based information is needed. If yes, ask the web searcher agent with specific details. If not, reply only with the keyword 'none.'",
            "4. Construct a comprehensive argument or counterargument based on the retrieved data and your planed strategy. Write the response as live dialogue (avoid bullet points), maintaining logical coherence and factual accuracy."
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with prosecutor-specific logic"""
        
        messages = [
            {"role": "system", "content": self.system_prompt + "\n'current_task': " + self.get_thought_steps()[state["thought_step"]]}
        ] + state["messages"]

        result = self.llm.invoke(messages)
        
        if state["thought_step"] == 0:
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
                "thought_step": 2,
                "caller": "prosecutor"
            }
        elif state["thought_step"] == 2:
            response = {
                "messages": [HumanMessage(content=result.content, name="prosecutor")],
                "next": self.is_web_search_needed(result.content),
                "thought_step": 3,
                "caller": "prosecutor"
            }
        elif state["thought_step"] == 3:
            response = {
                "messages": [HumanMessage(content=result.content, name="prosecutor")],
                "next": "judge",
                "thought_step": 0,
                "caller": "prosecutor"
            }
        else:
            raise ValueError("Invalid thought step")
            
        return response
    
    def is_web_search_needed(self, content: str) -> Literal["self", "web_searcher"]:
        if "none" in content.lower():
            return "self"
        else:
            return "web_searcher"

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