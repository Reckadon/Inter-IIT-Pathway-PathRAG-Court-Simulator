from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from core.pathway_store import PathwayVectorStore
from core.config import settings

class LawRetrieverTool(BaseTool):
    """Tool for retrieving information from legal documents using PathwayVectorStore"""
    
    name = "search_laws"
    description = "Search through laws, statutes, and constitution documents"
    
    def __init__(
        self,
        docs: List[Any],
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
    ):
        super().__init__()
        self.vectorstore = PathwayVectorStore()
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        
        # Initialize and index documents
        self.vectorstore.initialize(embeddings=self.embeddings)
        self.vectorstore.add_documents(docs)
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute the tool synchronously"""
        results = self.vectorstore.similarity_search(query)
        return {
            "type": "legal_search",
            "results": results,
            "query": query
        }
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Execute the tool asynchronously"""
        results = await self.vectorstore.asimilarity_search(query)
        return {
            "type": "legal_search",
            "results": results,
            "query": query
        }

class WebRetrieverTool(BaseTool):
    """Tool for retrieving information from web sources using Tavily"""
    
    name = "search_web"
    description = "Search the web for relevant legal information and precedents"
    
    def __init__(self, max_results: int = 3):
        super().__init__()
        self.tavily_tool = TavilySearchResults(max_results=max_results)
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute the tool synchronously"""
        results = self.tavily_tool.run(query)
        return {
            "type": "web_search",
            "results": results,
            "query": query
        }
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Execute the tool asynchronously"""
        results = await self.tavily_tool.arun(query)
        return {
            "type": "web_search",
            "results": results,
            "query": query
        }

def create_law_retriever(docs: List[Any]) -> LawRetrieverTool:
    """Create vector store retriever for legal documents"""
    return LawRetrieverTool(docs=docs)

def create_web_retriever(max_results: int = 3) -> WebRetrieverTool:
    """Create web search tool"""
    return WebRetrieverTool(max_results=max_results)