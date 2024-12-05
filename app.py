# from api.server import main
from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent, FetchingAgent
import asyncio


if __name__ == "__main__":
    print("running the app.py file...\n")
    # main()
    docs = "/documents"

    workflow = TrialWorkflow(
        lawyer=LawyerAgent(),
        prosecutor=ProsecutorAgent(),
        judge=JudgeAgent(),
        retriever=RetrieverAgent(docs),
        kanoon_fetcher = FetchingAgent()
    )

    # workflow.visualize()
    res = asyncio.run(workflow.run(user_prompt="New case regareding murder of Rajat Moona"))
    print(res)