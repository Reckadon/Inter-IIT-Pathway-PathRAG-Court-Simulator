from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from .base import AgentState

class LawyerResponse(TypedDict):
    """Structured lawyer response"""
    response: str
    next_step: Literal["self", "judge", "retriever"]

class LawyerAgent:
    """Agent representing the defense counsel"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.tools = tools or []
        
        self.system_prompt = """You are a skilled defense attorney in a specialized AI-driven legal system. Your role is to build and present compelling arguments for your client.

ROLE AND RESPONSIBILITIES:
1. Case Analysis
   - Thoroughly analyze case details and evidence
   - Identify key legal principles and precedents
   - Spot potential weaknesses in prosecution's case
   - Develop strategic defense angles

2. Evidence Handling
   - Evaluate strength of available evidence
   - Request additional information when needed
   - Challenge questionable evidence
   - Present evidence effectively

3. Argument Construction
   - Build logically sound arguments
   - Support claims with specific evidence
   - Address opposing arguments proactively
   - Maintain professional conduct

AVAILABLE NEXT STEPS:
- "self": Continue your chain of thought
- "judge": Present completed argument to judge
- "retriever": Request additional evidence or legal references

ARGUMENT CRITERIA:
1. Evidence Support
   - Are claims supported by concrete evidence?
   - Is additional information needed?
   - Are sources credible and relevant?

2. Legal Foundation
   - Are arguments grounded in law?
   - Are relevant precedents cited?
   - Are legal principles properly applied?

3. Strategic Value
   - Does argument advance client's interests?
   - Are weak points addressed?
   - Is timing appropriate?

You will go through the following chain of thought steps:
1. CASE & EVIDENCE ANALYSIS
2. LEGAL STRATEGY DEVELOPMENT
3. ARGUMENT CONSTRUCTION
4. VALIDATION & REFINEMENT

Do only the current step at a time.

Remember: Your goal is to present the strongest possible defense while maintaining ethical and professional standards."""

    def get_thought_steps(self) -> List[str]:
        """Get lawyer-specific chain of thought steps"""
        return [
            "1. CASE & EVIDENCE ANALYSIS:\n" +
            "   - Review current case status and available evidence\n" +
            "   - Identify key facts supporting defense\n" +
            "   - Spot gaps requiring additional information\n" +
            "   - Assess prosecution's recent arguments",

            "2. LEGAL STRATEGY DEVELOPMENT:\n" +
            "   - Determine optimal legal approach\n" +
            "   - Identify relevant laws and precedents\n" +
            "   - Plan counter-arguments to prosecution\n" +
            "   - Consider timing and emphasis",

            "3. ARGUMENT CONSTRUCTION:\n" +
            "   - Build logical argument structure\n" +
            "   - Connect evidence to legal principles\n" +
            "   - Anticipate counter-arguments\n" +
            "   - Craft persuasive narrative",

            "4. VALIDATION & REFINEMENT:\n" +
            "   - Review argument for logical consistency\n" +
            "   - Verify evidence citations\n" +
            "   - Check for potential weaknesses\n" +
            "   - Polish final presentation"
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with lawyer-specific logic"""
        
        # if state["thought_step"] >= 0:
        messages = [
            {"role": "human", "content": self.system_prompt, 
                 "current_task": self.get_thought_steps()[state["thought_step"]]},
        ] + state["messages"]
        # else:
        #     messages = [
        #         {"role": "lawyer", "content": self.system_prompt, 
        #          "current_task": "Initial case review and strategy planning"}
        #     ] + state["messages"]

        result = self.llm.with_structured_output(LawyerResponse).invoke(messages)
        
        if 0 <= state["thought_step"] < len(self.get_thought_steps())-1:
            response = {
                "messages": [HumanMessage(content=result["response"], name="lawyer")],
                "next": result["next_step"],
                "thought_step": state["thought_step"]+1,
                "caller": "lawyer"
            }
        else:
            response = {
                "messages": [HumanMessage(content=result["response"], name="lawyer")],
                "next": result["next_step"],
                "thought_step": 0
            }
            
        return response