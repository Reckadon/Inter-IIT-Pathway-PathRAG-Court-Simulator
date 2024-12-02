from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool

class BaseAgent:
    """Base agent class with Chain of Thought reasoning"""
    
    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: str = ""
    ):
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.tools = tools or []
        self.system_prompt = system_prompt
        
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Chain of Thought reasoning"""
        
        steps = [
            "1. Analyze current situation and context",
            "2. Identify key points to address",
            "3. Plan response strategy",
            "4. Determine if additional information is needed",
            "5. Formulate response",
            "6. Validate response"
        ]
        
        thoughts = []
        for step in steps:
            thought = await self._execute_thought_step(step, context, thoughts)
            thoughts.append({"step": step, "thought": thought})
        
        return {
            "thoughts": thoughts,
            "final_response": thoughts[-1]["thought"]
        }
    
    async def _execute_thought_step(
        self, 
        step: str, 
        context: Dict[str, Any], 
        previous_thoughts: List[Dict[str, Any]]
    ) -> str:
        """Execute a single thought step"""
        
        messages = [
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=f"""
                Current step: {step}
                Context: {context}
                Previous thoughts: {previous_thoughts}
                
                Think through this step carefully and provide your reasoning.
            """)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content