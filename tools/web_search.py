from typing import Dict, Any, List, Optional
from langchain.tools import Tool
import random
from datetime import datetime, timedelta

class LegalWebSearchTool:
    """Simulated web search tool with sample legal data"""
    
    def __init__(self):
        self.legal_database = {
            "cases": self._generate_sample_cases(),
            "statutes": self._generate_sample_statutes(),
            "precedents": self._generate_sample_precedents()
        }
        
        self.tool = Tool(
            name="legal_web_search",
            func=self.search,
            description="Search for legal information, cases, and precedents"
        )
    
    def search(self, query: str) -> Dict[str, Any]:
        """Simulate web search with sample legal data"""
        # Simulate search delay
        import time
        time.sleep(0.5)
        
        results = []
        query = query.lower()
        
        # Search cases
        for case in self.legal_database["cases"]:
            if (query in case["title"].lower() or 
                query in case["summary"].lower() or 
                any(query in tag.lower() for tag in case["tags"])):
                results.append({
                    "type": "case",
                    "content": case,
                    "relevance": self._calculate_relevance(query, case)
                })
        
        # Search statutes
        for statute in self.legal_database["statutes"]:
            if (query in statute["title"].lower() or 
                query in statute["description"].lower()):
                results.append({
                    "type": "statute",
                    "content": statute,
                    "relevance": self._calculate_relevance(query, statute)
                })
        
        # Search precedents
        for precedent in self.legal_database["precedents"]:
            if (query in precedent["title"].lower() or 
                query in precedent["summary"].lower()):
                results.append({
                    "type": "precedent",
                    "content": precedent,
                    "relevance": self._calculate_relevance(query, precedent)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return {
            "query": query,
            "results": results[:5],  # Return top 5 results
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_relevance(self, query: str, item: Dict[str, Any]) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Title match
        if query in item["title"].lower():
            score += 1.0
        
        # Content match
        content = item.get("summary", "") or item.get("description", "")
        if query in content.lower():
            score += 0.5
        
        # Tag match
        if "tags" in item and any(query in tag.lower() for tag in item["tags"]):
            score += 0.3
        
        # Recent cases get slight boost
        if "date" in item:
            days_old = (datetime.now() - datetime.fromisoformat(item["date"])).days
            score += max(0, 0.2 - (days_old / 365))  # Small boost for newer items
        
        return score
    
    def _generate_sample_cases(self) -> List[Dict[str, Any]]:
        """Generate sample legal cases"""
        return [
            {
                "title": "Smith v. Johnson Electronics",
                "summary": "Contract dispute over software development project delivery.",
                "date": (datetime.now() - timedelta(days=45)).isoformat(),
                "outcome": "Ruled in favor of plaintiff",
                "damages": "$150,000",
                "tags": ["contract law", "software development", "breach of contract"],
                "key_points": [
                    "Missed delivery deadlines",
                    "Incomplete feature implementation",
                    "Payment disputes"
                ]
            },
            {
                "title": "Roberts v. City Council",
                "summary": "Zoning law dispute regarding commercial property usage.",
                "date": (datetime.now() - timedelta(days=90)).isoformat(),
                "outcome": "Settled out of court",
                "damages": None,
                "tags": ["zoning law", "property law", "municipal law"],
                "key_points": [
                    "Zoning regulation interpretation",
                    "Historical district considerations",
                    "Community impact assessment"
                ]
            },
            # Add more sample cases...
        ]
    
    def _generate_sample_statutes(self) -> List[Dict[str, Any]]:
        """Generate sample legal statutes"""
        return [
            {
                "title": "Commercial Contract Act",
                "description": "Regulations governing commercial contracts and agreements.",
                "section": "15.2",
                "jurisdiction": "Federal",
                "key_provisions": [
                    "Written contract requirements",
                    "Breach remedies",
                    "Dispute resolution procedures"
                ]
            },
            {
                "title": "Digital Services Protection Act",
                "description": "Framework for digital service provider responsibilities.",
                "section": "8.4",
                "jurisdiction": "Federal",
                "key_provisions": [
                    "Service level agreements",
                    "Data protection requirements",
                    "Consumer rights"
                ]
            },
            # Add more sample statutes...
        ]
    
    def _generate_sample_precedents(self) -> List[Dict[str, Any]]:
        """Generate sample legal precedents"""
        return [
            {
                "title": "Tech Solutions Inc. v. DataCorp (2022)",
                "summary": "Established framework for evaluating software project delays.",
                "date": (datetime.now() - timedelta(days=365)).isoformat(),
                "impact": "High",
                "cited_by": 15,
                "key_holdings": [
                    "Force majeure in tech contracts",
                    "Reasonable delay standards",
                    "Documentation requirements"
                ]
            },
            {
                "title": "Metropolitan Development v. Historic Board (2021)",
                "summary": "Defined balance between development rights and preservation.",
                "date": (datetime.now() - timedelta(days=700)).isoformat(),
                "impact": "Medium",
                "cited_by": 8,
                "key_holdings": [
                    "Historical significance criteria",
                    "Economic impact consideration",
                    "Public interest balance"
                ]
            },
            # Add more sample precedents...
        ] 