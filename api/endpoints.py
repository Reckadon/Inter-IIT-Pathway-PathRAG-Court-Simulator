from typing import Dict, Any
from pydantic import BaseModel
from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent

docs = "docs"

workflow = TrialWorkflow(
    lawyer=LawyerAgent(),
    prosecutor=ProsecutorAgent(),
    judge=JudgeAgent(),
    retriever=RetrieverAgent(docs)
)

@serve_callable
async def process_case(case_details: Dict[str, Any]) -> Dict[str, Any]:
    """Process a legal case through the workflow"""
    try:
        result = await workflow.run(case_details)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        } 