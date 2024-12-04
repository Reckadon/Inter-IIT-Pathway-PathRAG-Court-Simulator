from typing import Dict, List, Optional, Any, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState

class AgentState(MessagesState):
    """State for each agent node in the graph"""
    next: str  # Where to route to next
    thought_step: Optional[int] = 0  # Current step in chain of thought
    caller: Optional[str] = None  # Who called the agent

# class AgentResponse(TypedDict):
#     """Standard response format for agents"""
#     messages: List[BaseMessage]
#     next: str
#     thought_step: Optional[str]
#     cot_finished: bool

# class BaseAgent:
#     """Base agent class with Chain of Thought reasoning"""
    
#     def __init__(
#         self,
#         llm: Optional[BaseChatModel] = None,
#         tools: Optional[List[BaseTool]] = None,
#         system_prompt: str = ""
#     ):
#         self.llm = llm or ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             temperature=0,
#             convert_system_message_to_human=True
#         )
#         self.tools = tools or []
#         self.system_prompt = system_prompt
#         self.react_agent = create_react_agent(self.llm, self.tools)
        
#     def get_thought_steps(self) -> List[str]:
#         """Get chain of thought steps - override in subclasses"""
#         return [
#             "1. Analyze current situation and context",
#             "2. Identify key points to address", 
#             "3. Plan response strategy",
#             "4. Determine if additional information is needed",
#             "5. Formulate response",
#             "6. Validate response"
#         ]
    
#     def get_next_thought_step(self, current_step: Optional[str]) -> Optional[str]:
#         """Get next step in chain of thought sequence"""
#         steps = self.get_thought_steps()
#         if not current_step:
#             return steps[0]
#         try:
#             current_idx = steps.index(current_step)
#             if current_idx < len(steps) - 1:
#                 return steps[current_idx + 1]
#         except ValueError:
#             pass
#         return None
        
#     async def process(self, state: AgentState) -> AgentResponse:
#         """Process current state and return response"""
#         # If in middle of chain of thought, continue
#         if state.thought_step and not state.cot_finished:
#             return await self._continue_chain_of_thought(state)
            
#         # Otherwise start new chain of thought
#         return await self._start_chain_of_thought(state)
    
#     async def _start_chain_of_thought(self, state: AgentState) -> AgentResponse:
#         """Start new chain of thought sequence"""
#         first_step = self.get_thought_steps()[0]
#         result = await self.react_agent.ainvoke(state)
        
#         return {
#             "messages": result["messages"],
#             "next": "self",  # Continue chain of thought
#             "thought_step": first_step,
#             "cot_finished": False
#         }
    
#     async def _continue_chain_of_thought(self, state: AgentState) -> AgentResponse:
#         """Continue existing chain of thought sequence"""
#         result = await self.react_agent.ainvoke(state)
        
#         next_step = self.get_next_thought_step(state.thought_step)
#         if next_step:
#             # Continue chain of thought
#             return {
#                 "messages": result["messages"],
#                 "next": "self",
#                 "thought_step": next_step,
#                 "cot_finished": False
#             }
#         else:
#             # Chain of thought complete
#             return {
#                 "messages": result["messages"],
#                 "next": self._determine_next_agent(result),
#                 "thought_step": None,
#                 "cot_finished": True
#             }
    
#     def _determine_next_agent(self, result: Dict[str, Any]) -> str:
#         """Determine next agent based on result - override in subclasses"""
#         raise NotImplementedError