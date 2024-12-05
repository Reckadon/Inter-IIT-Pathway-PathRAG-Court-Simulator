"""Agent implementations for the Legal RAG system"""
# from .base import BaseAgent
from .lawyer import LawyerAgent
from .prosecutor import ProsecutorAgent
from .judge import JudgeAgent
from .retriever import RetrieverAgent
from .kanoon_fetcher import FetchingAgent
from .web_search import WebSearcherAgent

__all__ = [
    # 'BaseAgent',
    'LawyerAgent',
    'ProsecutorAgent',
    'JudgeAgent',
    'RetrieverAgent',
    'FetchingAgent',
    'WebSearcherAgent'
]