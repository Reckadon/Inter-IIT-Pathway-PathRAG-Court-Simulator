from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from core.pathway_store import PathwayVectorStore
from core.config import settings
from pydantic import Field
from langchain_community.vectorstores.pathway import PathwayVectorClient


def create_law_retriever(docs: List[Any]) -> BaseTool:
    """Create vector store retriever for legal documents"""
    # Initialize PathwayVectorStore with embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model=settings.EMBEDDING_MODEL,
    #     google_api_key=settings.GOOGLE_API_KEY,
    #     task_type="retrieval_document"
    # )
    public_store = PathwayVectorStore('public', './public_documents', 8765)
    public_client = public_store.get_client()
    
    private_store = PathwayVectorStore('private', "./private_documents", 8766)
    private_client = private_store.get_client()

    retriever = public_client.as_retriever()
    pvt_retriever = private_client.as_retriever()

    # Create retriever tool
    return create_retriever_tool(
        retriever,
        "search_laws",
        "Search through laws, statutes, and constitution documents"
    )

# class WebRetrieverTool(BaseTool):
#     """Tool for retrieving information from web sources using Tavily"""
    
#     name: str = "search_web"
#     description: str = "Search the web for relevant legal information and precedents"
#     tavily_tool: TavilySearchResults = Field(default_factory=lambda: TavilySearchResults(max_results=3))
    
#     def __init__(self, max_results: int = 3, **kwargs):
#         super().__init__(**kwargs)
#         self.tavily_tool = TavilySearchResults(max_results=max_results)
    
#     def _run(self, query: str) -> Dict[str, Any]:
#         """Execute the tool synchronously"""
#         results = self.tavily_tool.run(query)
#         return {
#             "type": "web_search",
#             "results": results,
#             "query": query
#         }
    
#     async def _arun(self, query: str) -> Dict[str, Any]:
#         """Execute the tool asynchronously"""
#         results = await self.tavily_tool.arun(query)
#         return {
#             "type": "web_search",
#             "results": results,
#             "query": query
#         }

# def create_web_retriever(max_results: int = 3) -> WebRetrieverTool:
#     """Create web search tool"""
#     return WebRetrieverTool(max_results=max_results)