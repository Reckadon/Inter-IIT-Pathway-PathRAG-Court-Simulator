from typing import Dict, Any
from pydantic import BaseModel
from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent
from tools.retrievers import create_law_retriever, create_web_retriever

# Initialize components
law_retriever = create_law_retriever()
web_retriever = create_web_retriever()

workflow = TrialWorkflow(
    lawyer=LawyerAgent(),
    prosecutor=ProsecutorAgent(),
    judge=JudgeAgent(),
    retriever=RetrieverAgent(
        vector_store_tool=law_retriever,
        web_search_tool=web_retriever
    )
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