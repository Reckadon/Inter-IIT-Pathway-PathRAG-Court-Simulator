# from api.server import main
from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent
import asyncio
    

if __name__ == "__main__":
    print("running the app.py file...\n")
    # main()
    docs = "/documents"

    workflow = TrialWorkflow(
        lawyer=LawyerAgent(),
        prosecutor=ProsecutorAgent(),
        judge=JudgeAgent(),
        retriever=RetrieverAgent(docs)
    )

    # workflow.visualize()
    res = asyncio.run(workflow.run(case_details={'title': "New case regareding murder of Rajat Moona", "description": "description"}))
    print(res)