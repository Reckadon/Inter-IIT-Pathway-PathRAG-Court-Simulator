from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from agents.base import AgentState
from langchain_groq import ChatGroq
import getpass
import os

# os.environ["GROQ_API_KEY"] = getpass.getpass()
from dotenv import load_dotenv
load_dotenv()

class JudgeDecision(BaseModel):
    """Judge's structured decision output"""
    response: str = Field(description="The judge's response and comments")
    next_agent: Literal["lawyer", "prosecutor", "END"] = Field(
        description="Next agent to speak in the trial or END if verdict is given in response"
    )

class JudgeAgent:
    """Agent representing the judge who manages the trial flow"""
    
    def __init__(
        self,  
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        self.llm = llm or ChatGroq(model="llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))
        self.tools = tools or []
        
        self.system_prompt = """
"You are a presiding judge overseeing a courtroom simulation. Your primary role is to evaluate the arguments presented by the lawyer and prosecutor for logical consistency, factual accuracy, and adherence to legal principles."
"Point out inconsistencies, hallucinations, or errors in the agents' arguments and provide constructive feedback to help refine them."
"Call upon the Law Retriever and Web Searcher agents as necessary to verify or clarify legal and factual claims made during the arguments."
"Monitor the proceedings and identify when no new points are being raised, all conflicts and rebuttals have been adequately addressed, and the case is ready for a verdict, When the arguments have reached this stage, request final statements from both the lawyer and prosecutor before delivering your impartial verdict."
"Summarize the case before delivering a verdict, outlining the key points of contention and the reasoning behind your decision."
"Facilitate a fair and structured discussion, ensuring that both parties have equal opportunity to present their case."
"Your decisions and comments should be impartial, grounded in logic, and aimed at maintaining the integrity of the courtroom process."

you will go through the following chain of thought steps:
1. Review arguments
2. legal data retrieval
3. web search
4. check if trial is ready for verdict
5. give response based on above steps

Do only current task at a time. Do not confuse with precedent cases. Avoid very long responses.
"""
        

        

    def get_thought_steps(self) -> List[str]:
        """Get judge-specific chain of thought steps"""
        return [
            "1. Listen to the arguments presented by both the lawyer and prosecutor. Note their key points and claims. Identify potential hallucinations or logical errors or factual errors in latest argument.",
            "2. Determine the specific legal data (e.g., laws, IPCs, legal case precedents) required for cross verificaton of identified errors. Clearly ask the law retriever agent for the necessary legal data.",
            "3. Evaluate if additional web-based information is needed. If yes, ask the web searcher agent with specific details. If not, reply only with the keyword 'none.'",
            "4. Assess the current state of the case, analyze and determine if it is ready for a verdict. Atleast 10 arguments should be present before verdict.",
            """5. Provide constructive feedback or comments, pointing out logical flaws, factual inconsistencies, or unsupported claims in the arguments if present based on retrieverd data. 
            From previous thought step, only if trail is ready for verdict, ask for final statements from both lawyer and prosecutor., if already asked for final statements, summarize the case and deliver verdict with keyphrase "Given Verdict".
            Write the response as live dialogue (avoid bullet points), Maintain an impartial tone.""" 
        ]
    async def process(self, state: AgentState) -> AgentState:
        """Process current state with judge-specific logic"""

       
        # if state["thought_step"] >= 0:
        messages = [
            {"role": "system", "content": self.system_prompt + "\n'current_task': " + self.get_thought_steps()[state["thought_step"]]}
        ] + state["messages"]
        # else:
        #     messages = [
        #         {"role": "system", "content": self.system_prompt + "\n'current_task': 'Start of trial, choose the first speaker'"}
        #     ] + state["messages"]

        # print(f"prompt: {messages}")
        if state["thought_step"] != 4:
            result = self.llm.invoke(messages)
        else:
            result = self.llm.with_structured_output(JudgeDecision).invoke(messages)
        
        if state["thought_step"] == 0 or state["thought_step"] == 3:
            response = {
                "messages": [HumanMessage(content=result.content, name="judge")],
                "next": "self",
                "thought_step": state["thought_step"]+1,
                "caller": "judge"
            }
        elif state["thought_step"] == 1:
            response = {
                "messages": [HumanMessage(content=result.content, name="judge") ],
                "next": "retriever",
                "thought_step": 2,
                "caller": "judge"
            }
        elif state["thought_step"] == 2:
            response = {
                "messages": [HumanMessage(content=result.content, name="judge") ],
                "next": self.is_web_search_needed(result.content),
                "thought_step": 3,
                "caller": "judge"
            }
        elif state["thought_step"] == 4:
            response = {
                "messages": [HumanMessage(content=result.response, name="judge")],
                "next": result.next_agent,
                "thought_step": 0,
                "caller": "judge"
            }
        else:
            raise ValueError("Invalid thought step")

        return response
    
    def is_web_search_needed(self, content: str) -> Literal["self", "web_searcher"]:
        if "none" in content.lower():
            return "self"
        else:
            return "web_searcher"
    
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