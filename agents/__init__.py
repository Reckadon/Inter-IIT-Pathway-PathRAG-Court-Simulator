"""Agent implementations for the Legal RAG system"""
# from .base import BaseAgent
from .lawyer import LawyerAgent
from .prosecutor import ProsecutorAgent
from .judge import JudgeAgent
from .retriever import RetrieverAgent

__all__ = [
    # 'BaseAgent',
    'LawyerAgent',
    'ProsecutorAgent',
    'JudgeAgent',
    'RetrieverAgent'
]