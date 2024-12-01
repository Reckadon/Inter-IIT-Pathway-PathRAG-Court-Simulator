from typing import Dict, Any
from .base import BaseAgent

class JudgeAgent(BaseAgent):
    """Agent representing the judge"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are an impartial judge. Your role is to:
        1. Ensure fair trial proceedings
        2. Fact-check arguments from both sides
        3. Evaluate evidence objectively
        4. Apply laws and precedents correctly
        5. Make well-reasoned decisions
        """
        super().__init__(system_prompt=system_prompt, **kwargs)
    
    async def fact_check(self, argument: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fact check an argument"""
        fact_check_context = {
            **context,
            "argument": argument
        }
        
        thought_process = await self.think(fact_check_context)
        
        return {
            "type": "fact_check",
            "validity": self._determine_validity(thought_process["final_response"]),
            "reasoning": thought_process["thoughts"],
            "feedback": thought_process["final_response"]
        }
    
    async def make_verdict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make final verdict"""
        thought_process = await self.think(context)
        
        return {
            "type": "verdict",
            "decision": thought_process["final_response"],
            "reasoning": thought_process["thoughts"]
        }
    
    def _determine_validity(self, response: str) -> float:
        """Determine validity score from response"""
        # Simple implementation - can be made more sophisticated
        if "valid" in response.lower():
            return 1.0
        elif "partially valid" in response.lower():
            return 0.5
        return 0.0