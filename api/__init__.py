"""API endpoints for the Legal RAG system"""

from .server import create_app
from .endpoints import process_case

__all__ = [
    'create_app',
    'process_case'
] 