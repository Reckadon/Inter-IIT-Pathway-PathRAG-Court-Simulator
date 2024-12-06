from core.workflow import TrialWorkflow
from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent, FetchingAgent, WebSearcherAgent
import asyncio
from langchain_groq import ChatGroq
import os
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from langchain_huggingface import HuggingFaceEndpoint
import json

# Initialize FastAPI app
app = FastAPI()

# Initialize LLMs
llm_0 = ChatGroq(model="groq/gemma2-9b-it", groq_api_key=os.environ["GROQ_API_KEY"])
llms =[
    ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.environ['GROQ_API_KEY']),
    ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=os.environ['GROQ_API_KEY']),
    ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY']),
    ChatGroq(model="gemma-7b-it", groq_api_key=os.environ['GROQ_API_KEY']),
    ChatGroq(model="mixtral-8x7b-32768", groq_api_key=os.environ['GROQ_API_KEY'])
    # HuggingFaceEndpoint(repo_id ="Qwen/QwQ-32B-Preview", huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'])
]

# Initialize Workflow
workflow = TrialWorkflow(
    lawyer=LawyerAgent(llms=llms),
    prosecutor=ProsecutorAgent(llms=llms),
    judge=JudgeAgent(llms=llms),
    retriever=RetrieverAgent(llms=llms),
    kanoon_fetcher=FetchingAgent(llms=llms),
    web_searcher=WebSearcherAgent(llm=llm_0),
)

# Visualize workflow
workflow.visualize()

@app.post("/stream_workflow")
async def stream_workflow(user_prompt: str = Body(..., embed=True)):
    async def event_generator():
        async for state in workflow.run(user_prompt=user_prompt):
            # # Ensure state is serialized properly
            # if isinstance(state["state"], str):
            #     # Parse string-like dictionaries back into JSON
            #     state["state"] = json.loads(state["state"].replace("'", '"'))  # Convert single quotes to double quotes if needed
            yield f"data: {json.dumps(state)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
