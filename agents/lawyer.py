from typing import Dict, Any
from .base import BaseAgent

class LawyerAgent(BaseAgent):
    """Agent representing the defense counsel"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a skilled defense attorney. Your role is to:
        1. Analyze case details and evidence thoroughly
        2. Build strong arguments in favor of your client
        3. Use relevant laws and precedents to support your case
        4. Counter prosecution's arguments effectively
        5. Maintain professional conduct throughout the trial
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    async def make_argument(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a legal argument"""
        thought_process = await self.think(context)
        
        # Check if we need additional information
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