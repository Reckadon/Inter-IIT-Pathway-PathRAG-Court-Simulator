from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseAgent, AgentState, AgentResponse
from tools.retrievers import create_law_retriever#, create_web_retriever

class RetrievalResult(TypedDict):
    """Structured retrieval result"""
    findings: List[str]
    sources: List[str]
    relevance_scores: List[float]
    calling_agent: str  # Track who requested the information

class RetrieverAgent(BaseAgent):
    """Agent for retrieving and analyzing information from various sources"""
    
    def __init__(
        self,
        docs: List[Any],
        **kwargs
    ):
        system_prompt = """You are an information retrieval specialist. Your role is to:
        1. Analyze information requests carefully
        2. Search appropriate sources (legal documents or web)
        3. Evaluate source credibility
        4. Assess information relevance
        5. Format findings for legal context
        
        For each search:
        - Determine best search strategy
        - Use most relevant source
        - Validate information accuracy
        - Structure findings clearly
        - Track information sources
        """
        # Create retrieval tools
        tools = [
            create_law_retriever(docs),
            # create_web_retriever()
        ]
        
        super().__init__(system_prompt=system_prompt, tools=tools, **kwargs)
    
    def get_thought_steps(self) -> List[str]:
        """Get retriever-specific chain of thought steps"""
        return [
            "1. Analyze information request",
            "2. Determine search strategy",
            "3. Execute targeted search",
            "4. Evaluate search results",
            "5. Structure findings",
            "6. Validate relevance"
        ]
    
    async def process(self, state: AgentState) -> AgentResponse:
        """Process current state with retriever-specific logic"""
        # Extract calling agent from state
        calling_agent = self._get_calling_agent(state["messages"])
        
        # Add retrieval context to state
        messages = state["messages"] + [
            SystemMessage(content=f"""
                Information requested by: {calling_agent}
                
                Focus on:
                1. Specific information requested
                2. Context of the request
                3. Required level of detail
                4. Source credibility
                5. Result relevance
                
                Structure findings for legal context.
            """)
        ]
        
        # Update state with retrieval context
        state["messages"] = messages
        
        # Process through chain of thought
        response = await super().process(state)
        
        # If chain of thought is complete, structure the findings
        if response["cot_finished"]:
            retrieval_result = self._structure_findings(
                response["messages"][-1].content,
                calling_agent
            )
            
            # Format findings and add to messages
            response["messages"].append(
                HumanMessage(
                    content=self._format_findings(retrieval_result),
                    name="retriever"
                )
            )
            
            # Return control to calling agent
            response["next"] = calling_agent
        
        return response
    
    def _determine_next_agent(self, result: Dict[str, Any]) -> str:
        """Return control to the agent that requested information"""
        return self._get_calling_agent(result["messages"])
    
    def _get_calling_agent(self, messages: List[Dict[str, Any]]) -> str:
        """Determine which agent requested the information"""
        # Look through recent messages in reverse
        for message in reversed(messages):
            if isinstance(message, dict) and "role" in message:
                if message["role"] in ["lawyer", "prosecutor"]:
                    return message["role"]
            elif hasattr(message, "name"):
                if message.name in ["lawyer", "prosecutor"]:
                    return message.name
        return "judge"  # Default to judge if can't determine
    
    def _structure_findings(self, content: str, calling_agent: str) -> RetrievalResult:
        """Parse and structure the retrieval findings"""
        findings: RetrievalResult = {
            "findings": [],
            "sources": [],
            "relevance_scores": [],
            "calling_agent": calling_agent
        }
        
        content_lower = content.lower()
        
        # Extract findings
        finding_markers = ["found that", "research shows", "source indicates", "according to"]
        for marker in finding_markers:
            if marker in content_lower:
                start_idx = content_lower.index(marker)
                end_idx = content.find(".", start_idx)
                if end_idx != -1:
                    finding = content[start_idx:end_idx].strip()
                    findings["findings"].append(finding)
                    
                    # Look for source citation near the finding
                    source_end = content.find(")", end_idx)
                    if source_end != -1:
                        source = content[end_idx+1:source_end].strip("( ")
                        findings["sources"].append(source)
                        
                        # Assess relevance
                        relevance = self._assess_relevance(finding)
                        findings["relevance_scores"].append(relevance)
        
        return findings
    
    def _assess_relevance(self, finding: str) -> float:
        """Assess relevance score of a finding"""
        high_relevance = ["directly relates", "specifically addresses", "exactly matches"]
        medium_relevance = ["related to", "pertains to", "relevant to"]
        low_relevance = ["might relate", "could be relevant", "tangentially"]
        
        finding_lower = finding.lower()
        
        if any(term in finding_lower for term in high_relevance):
            return 1.0
        elif any(term in finding_lower for term in medium_relevance):
            return 0.7
        elif any(term in finding_lower for term in low_relevance):
            return 0.3
        return 0.5
    
    def _format_findings(self, findings: RetrievalResult) -> str:
        """Format structured findings for presentation"""
        formatted = [f"Information Retrieval Results (for {findings['calling_agent']}):\n"]
        
        for i, (finding, source, relevance) in enumerate(zip(
            findings["findings"],
            findings["sources"],
            findings["relevance_scores"]
        )):
            relevance_label = "High" if relevance > 0.7 else "Medium" if relevance > 0.4 else "Low"
            formatted.extend([
                f"Finding {i+1}:",
                f"- Content: {finding}",
                f"- Source: {source}",
                f"- Relevance: {relevance_label}",
                ""
            ])
        
        return "\n".join(formatted)