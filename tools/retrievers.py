from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults

def create_law_retriever(docs):
    """Create vector store retriever for legal documents"""
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="legal-docs",
        embedding=OpenAIEmbeddings(),
    )
    
    law_retriever = create_retriever_tool(
        vectorstore.as_retriever(),
        "search_laws",
        "Search through laws, statutes, and constitution documents",
    )
    return law_retriever

def create_web_retriever():
    """Create web search tool"""
    web_tool = TavilySearchResults(max_results=3)
    return web_tool 