from typing import Dict, Any
from .base import BaseAgent

class ProsecutorAgent(BaseAgent):
    """Agent representing the prosecution"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a prosecutor representing the state. Your role is to:
        1. Present evidence against the defendant
        2. Build compelling arguments based on facts
        3. Apply relevant laws and precedents
        4. Counter defense arguments effectively
        5. Maintain professional conduct throughout the trial
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    async def make_argument(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a prosecution argument"""
        thought_process = await self.think(context)
        
        if "need_information" in thought_process["final_response"].lower():
            return {
                "type": "information_request",
                "details": thought_process["final_response"]
            }
        
        return {
            "type": "argument",
            "content": thought_process["final_response"],
            "reasoning": thought_process["thoughts"]
        }