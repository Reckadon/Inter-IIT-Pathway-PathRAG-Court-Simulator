# from api.server import main
from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent, FetchingAgent, WebSearcherAgent
import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# def initialize_llm():
#     try:
#         llm = ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY'])
#     except:
#         llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
#     return llm
# def initialize_llm_1():
#     try:    
#         llm = ChatGroq(model="groq/gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY'])
#     except:
#         llm = ChatGoogleGenerativeAI(model="google/gemini-1.5-flash", api_key=os.getenv('GOOGLE_API_KEY'))
#     return llm


if __name__ == "__main__":
    print("running the app.py file...\n")
    llm_0 = ChatGroq(model="groq/gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY'])
    llms =[ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY']),ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=os.environ['GROQ_API_KEY']),ChatGroq(model="", groq_api_key=os.environ['GROQ_API_KEY'])]
    workflow = TrialWorkflow(
        lawyer=LawyerAgent(llms=llms),
        prosecutor=ProsecutorAgent(llms=llms),
        judge=JudgeAgent(llms=llms),
        retriever=RetrieverAgent(llms=llms),
        kanoon_fetcher = FetchingAgent(llms=llms),
        web_searcher = WebSearcherAgent(llm=llm_0)
    )

    workflow.visualize()

    # res = asyncio.run(workflow.run(user_prompt="New case regareding murder of Rajat Moona"))
    # print(res)

    user_prompt = """Case Title
State vs. Alex Martin

Case Summary
Alex Martin, a 32-year-old software engineer, has been accused of committing theft under Section 378 of the Indian Penal Code (IPC). The alleged theft involves stealing a high-value laptop from their colleague, John Davis, at the workplace. Alex claims innocence, asserting that the laptop in question was mistakenly taken, believing it to be their own.

Case Details
Incident Description:
On September 15, 2024, at around 3:30 PM, John Davis reported that his laptop, valued at ₹1,20,000, was missing from his desk at XYZ Tech Pvt. Ltd. Security footage shows Alex Martin leaving the office with a laptop that closely resembles John's. Alex returned the laptop the next morning, claiming it was taken by mistake.

Evidence:

CCTV Footage: Video showing Alex Martin picking up the laptop from John's desk and leaving the office premises.
Witness Statement: Sarah Khan, a co-worker, testified that Alex Martin was seen near John's desk shortly before the laptop went missing.
Laptop Serial Number: John provided proof of ownership through the serial number of the laptop.
Defendant's Claim:
Alex Martin asserts that the laptop was taken accidentally, as it closely resembles their own laptop. They claim they were in a rush and did not verify the laptop's serial number before leaving the office.

Legal Charges
Section 378, IPC (Theft):
Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property, is said to commit theft.

Section 403, IPC (Dishonest Misappropriation of Property):
Whoever dishonestly misappropriates or converts to their own use any movable property is liable under this section.
"""
    asyncio.run(workflow.run(user_prompt=user_prompt))