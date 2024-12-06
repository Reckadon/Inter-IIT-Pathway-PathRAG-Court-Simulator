from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from core.pathway_store import PathwayVectorStore
from .base import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages.utils import get_buffer_string
import os
from dotenv import load_dotenv
load_dotenv()


def create_law_retriever(private=False) -> BaseTool:
    """Create vector store retriever for legal documents"""
    vector_store = None
    if private:
        vector_store = PathwayVectorStore('private', './private_documents', 8765)
    else:
        vector_store = PathwayVectorStore('public', './public_documents', 8766)

    client = vector_store.get_client()
    
    retriever = client.as_retriever()

    return retriever
   
class RetrieverResponse(BaseModel):
    """Structured retriever response"""
    # response: str = Field(description="The retriever's assessment of the retrieved content")
    is_enough: bool = Field(description="Whether the retrieved content is enough to answer the request")

class Queries(BaseModel):
    private_query: str = Field(description="Query for private retriever (user case files or documents). 'none' if not needed")  
    public_query: str = Field(description="Query for public retriever (public docs like IPC, legal case precedents, etc) 'none' if not needed")
class RetrieverAgent:
    """Agent for retrieving and analyzing legal documents from vector store"""
    
    def __init__(
        self,
        llms,
        # **kwargs
    ):
        self.private_retriever = create_law_retriever(private=True)
        self.public_retriever = create_law_retriever(private=False)
        # self.llm = llm or ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
        self.llms = llms
        self.system_prompt = """
"You are a legal research assistant specializing in retrieving relevant legal provisions, case laws, and statutes from a vector database of the Indian Penal Code (IPC) and related legal documents."
"Formulate queries based on inputs from the judge, lawyer, or prosecutor, ensuring precision in the retrieval process."
"Evaluate the retrieved text for relevance and clarity before sharing it with the requesting agent. If the retrieval is insufficient, refine the query and attempt again."
"you have acsess to private retrierver (contains user case files or documents) and public retriever (contains public docs like IPC, legal case precedents, etc)"
"Your role is critical in supporting the legal arguments by providing accurate and contextually appropriate legal references."
"Ensure that your outputs are succinct, relevant, and formatted for easy understanding by the requesting agent."

you will go through the following chain of thought steps:
1. Analyze the information request.
2. Form the queries.
3. Assess the retrieved results.
4. Provide accurate excerpts of information

Do only current task at a time. Avoid very long responses.
"""

    def get_thought_steps(self) -> List[str]:
        """Get retriever-specific chain of thought steps"""
        return [
            "1. Analyze the information request received from the lawyer or prosecutor and Note the key words and points.",
            "2. Form very good queries to Retrieve the most relevant text needed. one query for each private retriever (user case files or documents) or public retriever (contains public docs like IPC, legal case precedents, etc) 'none' if a specific retriever is not needed.",
            "3. Assess the retrieved results and determine If they are relevent and enough. if yes, set 'is_enough' to True. if not, set 'is_enough' false",
            "4. Provide the lawyer or prosecutor with accurate excerpts of relevant laws based on the request, ensuring clarity.If no relevant law is found, respond with 'No relevant law found in database.'"
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with retriever-specific logic"""
        
        messages = [
            {"role": "system", "content": self.system_prompt + f"\n'current_task': {self.get_thought_steps()[0]}"}
        ] + state["messages"]

        # info_analysis = self.llm.invoke(messages)

        for i,llm in enumerate(self.llms):
            try:
                info_analysis = llm.invoke(messages)
                break
            except Exception as e:
                print(f"LLM {i} failed with error: {e}")
                continue

            
        for i in range(1): # max 5 iterations
            #formulate query
            messages.append({"role": "system", "content": "need_info: " + info_analysis.content + "\n" + "current_task: " + self.get_thought_steps()[1]})
            # queries = self.llm.with_structured_output(Queries).invoke(messages)
            for i,llm in enumerate(self.llms):
                try:
                    queries = llm.with_structured_output(Queries).invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")

            #retrieve
            private_retrieved_content = self.private_retriever.invoke(queries.private_query) if queries.private_query.lower() != 'none' else 'None'
            public_retrieved_content = self.public_retriever.invoke(queries.public_query) if queries.public_query.lower() != 'none' else 'None'

            #assess
            messages.append({"role": "system", "content": "private_retrieved_content: " + str(private_retrieved_content) + "\npublic_retrieved_content: " + str(public_retrieved_content) + "\ncurrent_task: " + self.get_thought_steps()[2]})
            # assessment = self.llm.with_structured_output(RetrieverResponse).invoke(messages)
            for i,llm in enumerate(self.llms):
                try:
                    assessment = llm.with_structured_output(RetrieverResponse).invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")
                    continue

            #continue
            if assessment.is_enough:
                break

                
            
        
        messages.append({"role": "system", "content": "private_retrieved_content: " + str(private_retrieved_content) + "\npublic_retrieved_content: " + str(public_retrieved_content) + "\ncurrent_task: " + self.get_thought_steps()[3]})
        # result = self.llm.invoke(messages)
        for i,llm in enumerate(self.llms):
            try:
                result = llm.invoke(messages)
                break
            except Exception as e:
                print(f"LLM {i} failed with error: {e}")
                continue

        
        
        response = {
            "messages": [HumanMessage(content=result.content, name="retriever")],
            "next": state["caller"],
            "thought_step": state["thought_step"],
            "caller": "retriever"
        }
   
            
        return response
    
                    
        