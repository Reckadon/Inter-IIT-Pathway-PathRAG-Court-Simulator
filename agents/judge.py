from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from agents.base import AgentState

class JudgeDecision(TypedDict):
    """Judge's decision on next steps"""
    response: str
    next_step:  Literal["self", "lawyer", "prosecutor", "retriever", "END"]

class JudgeAgent:
    """Agent representing the judge who manages the trial flow"""
    
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
        
        self.system_prompt = """You are an impartial judge presiding over a legal trial in a specialized AI-driven legal system. Your role is critical in ensuring fair proceedings and making informed decisions.

ROLE AND RESPONSIBILITIES:
1. Trial Management
   - Maintain order and fairness in proceedings
   - Ensure balanced participation between lawyer and prosecutor
   - Monitor the logical flow and coherence of arguments
   - Prevent repetitive or circular arguments

2. Evidence Evaluation
   - Assess the credibility and relevance of presented evidence
   - Verify factual claims against available documentation
   - Request additional evidence when necessary
   - Identify gaps in evidence that need addressing

3. Decision Making
   - Make informed decisions about trial progression
   - Determine when sufficient evidence has been presented
   - Evaluate when counter-arguments are needed
   - Assess when the trial is ready for a verdict

AVAILABLE NEXT STEPS:
- "lawyer": Direct the defense lawyer to present arguments or respond
- "prosecutor": Allow the prosecutor to present charges or counter-arguments
- "retriever": Request additional evidence or documentation
- "self": Continue your own chain of thought
- "END": Conclude the trial when sufficient evidence and arguments have been presented

DECISION CRITERIA:
1. Evidence Sufficiency
   - Is there enough evidence to support current claims?
   - Are there gaps in the evidence that need filling?
   - Is the evidence credible and relevant?

2. Argument Balance
   - Have both sides had fair opportunity to present their case?
   - Are there unanswered counter-arguments?
   - Is there a need for clarification or elaboration?

3. Trial Progress
   - Has the case been thoroughly examined?
   - Are there remaining crucial points to address?
   - Is there enough information for a fair verdict?

You will go through the following chain of thought steps :
1. ARGUMENT ANALYSIS & FACT CHECK ASSESSMENT
2. EVIDENCE & CONSISTENCY EVALUATION
3. TRIAL STATE REVIEW
4. RESPONSE & DIRECTION FORMULATION

do only current step at a time.

Remember: Your primary goal is to ensure justice through a thorough, fair, and efficient trial process."""
        

        

    def get_thought_steps(self) -> List[str]:
        """Get judge-specific chain of thought steps"""
        return [
            "1. ARGUMENT ANALYSIS & FACT CHECK ASSESSMENT:\n" +
            "   - Evaluate if the latest argument is logically sound\n" +
            "   - Identify any claims that require factual verification\n" +
            "   - Check for any inconsistencies or gaps in reasoning",
            "   - Determine if additional information (laws/web search) is needed for you (should call retriever)\n" +

            "2. EVIDENCE & CONSISTENCY EVALUATION:\n" +
            "   - Cross-reference claims with available information, if retrieved\n" +
            "   - Assess internal consistency of the argument\n" +
            "   - Evaluate strength of supporting evidence\n" +
            "   - Identify any logical fallacies or weak reasoning",

            "3. TRIAL STATE REVIEW:\n" +
            "   - Assess overall progress of the trial\n" +
            "   - Review strength of prosecution and defense cases\n" +
            "   - Evaluate if key points have been adequately addressed\n" +
            "   - Determine if case is ready for verdict or needs more arguments",

            "4. RESPONSE & DIRECTION FORMULATION:\n" +
            "   - Provide specific feedback on current arguments\n" +
            "   - Determine next speaker (lawyer/prosecutor)\n" +
            "   - Ensure fair alternation between parties unless compelling reason exists\n" +
            "   - Give clear instructions for next phase of trial"
        ]
    async def process(self, state: AgentState) -> AgentState:
        """Process current state with judge-specific logic"""

       
        if state["thought_step"] >= 0:
            messages = [
                {"role": "human", "content": self.system_prompt, "current_task": self.get_thought_steps()[state["thought_step"]]},
            ] + state["messages"]
        else:
            messages = [
                {"role": "human ", "content": self.system_prompt, "current_task": "Start of trial, choose the first speaker"}
            ] + state["messages"]


        result = self.llm.with_structured_output(JudgeDecision).invoke(messages)
        
        if 0 <= state["thought_step"] < len(self.get_thought_steps())-1:
            response = {
                "messages": [HumanMessage(content=result["response"], name="judge")],
                "next": result["next_step"],
                "thought_step": state["thought_step"]+1,
                "caller": "judge"
            }
        else:
            response = {
                "messages": [HumanMessage(content=result["response"], name="judge") ],
                "next": result["next_step"],
                "thought_step": 0
            }
          
        return response
    
    # def _parse_judge_decision(self, content: str) -> JudgeDecision:
    #     """Parse judge's decision from response content"""
    #     # Default decision structure
    #     decision: JudgeDecision = {
    #         "next_agent": "lawyer",  # Default to lawyer if unclear
    #         "reasoning": "",
    #         "fact_check": None
    #     }
        
    #     content_lower = content.lower()
        
    #     # Check for fact-check indicators
    #     if any(term in content_lower for term in ["fact check", "verify", "accuracy"]):
    #         decision["fact_check"] = {
    #             "validity": self._assess_validity(content),
    #             "feedback": content
    #         }
        
    #     # Check for verdict readiness
    #     if any(term in content_lower for term in ["conclude", "verdict", "decision", "ruling"]):
    #         decision["next_agent"] = "END"
    #         decision["reasoning"] = "Trial ready for verdict"
    #         return decision
        
    #     # Check for retriever need
    #     if any(term in content_lower for term in ["need information", "more evidence", "research"]):
    #         decision["next_agent"] = "retriever"
    #         decision["reasoning"] = "Additional information required"
    #         return decision
        
    #     # Determine next speaker
    #     if "prosecutor" in content_lower:
    #         decision["next_agent"] = "prosecutor"
    #     elif "lawyer" in content_lower:
    #         decision["next_agent"] = "lawyer"
            
    #     decision["reasoning"] = content
    #     return decision
    
    # def _assess_validity(self, content: str) -> float:
    #     """Assess validity score from content"""
    #     if "invalid" in content.lower():
    #         return 0.0
    #     elif "partially valid" in content.lower():
    #         return 0.5
    #     elif "valid" in content.lower():
    #         return 1.0
    #     return 0.5  # Default to partial validity if unclear