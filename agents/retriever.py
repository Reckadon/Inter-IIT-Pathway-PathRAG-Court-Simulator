from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tools.retrievers import create_law_retriever, create_web_retriever

class RetrieverAgent:
    """Agent handling information retrieval from both legal documents and web"""
    
    def __init__(
        self,
        docs: List[Any],  # Documents to index
        llm: Optional[ChatGoogleGenerativeAI] = None,
        max_attempts: int = 3
    ):
        self.law_retriever = create_law_retriever(docs)
        self.web_retriever = create_web_retriever()
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
                results = await self.law_retriever.ainvoke(optimized_query)
            else:
                results = await self.web_retriever.ainvoke(optimized_query)
            
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
    
    def _extract_jurisdiction(self, response: str) -> Optional[str]:
        """Extract jurisdiction from agent's response"""
        if "federal" in response.lower():
            return "federal"
        elif "state" in response.lower():
            return "state"
        return None