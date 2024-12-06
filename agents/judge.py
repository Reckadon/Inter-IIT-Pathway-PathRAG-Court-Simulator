from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from agents.base import AgentState


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
        llms,
        tools: Optional[List[BaseTool]] = None,
    ):
        # self.llm = llm or ChatGroq(model="llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))
        self.llms = llms
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

       
        messages = [
            {"role": "system", "content": self.system_prompt + "\n'current_task': " + self.get_thought_steps()[state["thought_step"]]}
        ] + state["messages"]

        if state["thought_step"] != 4:
            for i, llm in enumerate(self.llms):
                try:
                    result = llm.invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")
                    continue

            # result = self.llm.invoke(messages)
        else:
            for i,llm in enumerate(self.llms):
                try:
                    result = llm.with_structured_output(JudgeDecision).invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")
                    continue
            # result = self.llm.with_structured_output(JudgeDecision).invoke(messages)
        
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
   