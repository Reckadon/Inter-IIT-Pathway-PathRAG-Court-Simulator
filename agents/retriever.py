from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from ..core.pathway_store import PathwayVectorStore

class RetrieverAgent:
    """Agent handling information retrieval from both legal documents and web"""
    
    def __init__(
        self,
        vector_store: PathwayVectorStore,
        web_search_tool: BaseTool,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        max_attempts: int = 3
    ):
        self.vector_store = vector_store
        self.web_search_tool = web_search_tool
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.max_attempts = max_attempts
        
        self.system_prompt = """You are an information retrieval specialist for legal cases. Your role is to:
        1. Analyze information needs and determine best source (legal documents or web)
        2. Formulate effective search queries
        3. Validate retrieved information for relevance and completeness
        4. Reformulate queries if needed
        5. Format information for legal context
        6. Track and cite sources properly
        
        When searching legal documents:
        - Focus on laws, statutes, and constitutional references
        - Look for precedent cases
        - Verify article and section numbers
        
        When searching web:
        - Focus on recent case precedents
        - Look for expert legal interpretations
        - Verify source credibility
        """
    
    async def retrieve_information(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve and validate information from appropriate sources"""
        
        # Determine search strategy
        strategy_response = await self._get_llm_response(
            f"""Analyze this query and determine the best search strategy:
            Query: {query}
            Context: {context}
            
            Should we search legal documents (laws, statutes, constitution) or web sources?
            Consider the type of information needed and provide your reasoning.
            """
        )
        
        search_strategy = self._parse_search_strategy(strategy_response)
        
        attempts = 0
        while attempts < self.max_attempts:
            # Generate optimized query
            optimized_query = await self._get_llm_response(
                f"""Formulate an effective search query based on:
                Original Query: {query}
                Search Strategy: {search_strategy}
                Attempt: {attempts + 1} of {self.max_attempts}
                Previous Results: {'None' if attempts == 0 else 'Insufficient'}
                
                Provide only the reformulated query without explanation.
                """
            )
            
            # Perform search based on strategy
            if search_strategy["use_legal_docs"]:
                results = await self._search_legal_documents(
                    optimized_query,
                    search_strategy.get("filters", {})
                )
            else:
                results = await self._search_web(optimized_query)
            
            # Validate results
            validation = await self._get_llm_response(
                f"""Validate these search results:
                Original Query: {query}
                Results: {results}
                
                Are these results sufficient and relevant? Consider:
                1. Do they directly address the query?
                2. Are the sources reliable?
                3. Is the information complete?
                4. Is additional information needed?
                
                Reply with either 'sufficient' or 'insufficient' followed by your reasoning.
                """
            )
            
            if "sufficient" in validation.lower():
                return {
                    "type": "information",
                    "results": results,
                    "source": "legal_docs" if search_strategy["use_legal_docs"] else "web",
                    "query_history": [{
                        "query": optimized_query,
                        "attempt": attempts + 1
                    }]
                }
            
            attempts += 1
        
        return {
            "type": "information_failure",
            "message": "Could not find satisfactory information after maximum attempts",
            "last_results": results,
            "query_history": [{
                "query": optimized_query,
                "attempt": attempts
            }]
        }
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        messages = [
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
    
    async def _search_legal_documents(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search legal documents in vector store"""
        results = await self.vector_store.similarity_search(
            query,
            k=4,
            metadata_filter=self._build_metadata_filter(filters)
        )
        return self._format_results(results, source_type="legal")
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search web using provided tool"""
        results = await self.web_search_tool.ainvoke(query)
        return self._format_results(results, source_type="web")
    
    def _parse_search_strategy(self, response: str) -> Dict[str, Any]:
        """Parse agent's search strategy decision"""
        use_legal_docs = any(term in response.lower() 
                           for term in ["law", "statute", "constitution", "legal code"])
        
        return {
            "use_legal_docs": use_legal_docs,
            "filters": {
                "document_type": ["law", "statute", "constitution"] if use_legal_docs else None,
                "jurisdiction": self._extract_jurisdiction(response)
            }
        }
    
    def _build_metadata_filter(self, filters: Dict[str, Any]) -> Optional[str]:
        """Build metadata filter for vector store"""
        filter_parts = []
        
        if doc_types := filters.get("document_type"):
            type_conditions = [f"type == '{t}'" for t in doc_types]
            filter_parts.append(f"({' || '.join(type_conditions)})")
            
        if jurisdiction := filters.get("jurisdiction"):
            filter_parts.append(f"jurisdiction == '{jurisdiction}'")
            
        return " && ".join(filter_parts) if filter_parts else None
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        source_type: str
    ) -> List[Dict[str, Any]]:
        """Format results with consistent structure"""
        formatted = []
        for result in results:
            formatted.append({
                "content": result.get("content", result.get("text", "")),
                "source": result.get("source", "Unknown"),
                "source_type": source_type,
                "metadata": result.get("metadata", {}),
                "relevance_score": result.get("score", 0.0)
            })
        return formatted
    
    def _extract_jurisdiction(self, response: str) -> Optional[str]:
        """Extract jurisdiction from agent's response"""
        if "federal" in response.lower():
            return "federal"
        elif "state" in response.lower():
            return "state"
        return None 