from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tools.retrievers import create_law_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from core.pathway_store import PathwayVectorStore
from .base import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages.utils import get_buffer_string
import os
from dotenv import load_dotenv
load_dotenv()


def create_law_retriever() -> BaseTool:
    """Create vector store retriever for legal documents"""
    public_store = PathwayVectorStore('public', './documents', 8765)

    public_client = public_store.get_client()
    
    retriever = public_client.as_retriever()

    return retriever
    # Create retriever tool
    # return create_retriever_tool(
    #     retriever,
    #     "search_laws",
    #     "Search through laws, statutes, and constitution documents"
    # )
class RetrieverResponse(BaseModel):
    """Structured retriever response"""
    response: str = Field(description="The retriever's assessment of the retrieved content")
    is_enough: bool = Field(description="Whether the retrieved content is enough to answer the request")


class RetrieverAgent:
    """Agent for retrieving and analyzing legal documents from vector store"""
    
    def __init__(
        self,
        # vector_store_retriever: Any,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ):
        self.vector_store_retriever = create_law_retriever()
        self.llm = llm or ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
        
        self.system_prompt = """You are a legal document retrieval specialist. Your role is to find relevant legal information from a database of laws, statutes, and precedents.

ROLE AND RESPONSIBILITIES:
1. Query Formation
   - Analyze information needs carefully
   - Break down complex requests into specific queries
   - Identify key legal concepts and terms
   - Formulate precise search queries

2. Result Analysis
   - Evaluate retrieved documents for relevance
   - Assess if information meets the request
   - Identify gaps requiring additional queries
   - Determine if more specific searches are needed

3. Information Synthesis
   - Combine relevant findings
   - Present information clearly
   - Highlight key legal points
   - Indicate if more retrieval is needed

RETRIEVAL CRITERIA:
1. Query Effectiveness
   - Are search terms specific enough?
   - Do they capture legal concepts?
   - Are variations needed?

2. Result Adequacy
   - Do results answer the request?
   - Is information complete?
   - Are more queries needed?

3. Legal Relevance
   - Are documents legally relevant?
   - Do they address the specific issue?
   - Are they authoritative sources?

You will go through the following chain of thought steps:
1. INFORMATION NEED ANALYSIS
2. QUERY FORMULATION & RETRIEVAL
3. RESULT ASSESSMENT
4. CONTINUATION DECISION

Do only the current step at a time.

Remember: Your goal is to find the most relevant legal information. If results are insufficient, continue with refined queries."""

    def get_thought_steps(self) -> List[str]:
        """Get retriever-specific chain of thought steps"""
        return [
            "1. INFORMATION NEED ANALYSIS:\n" +
            "   - Understand the legal information needed\n" +
            "   - Identify key legal concepts\n" +
            "   - Break down complex requests\n" +
            "   - Plan search approach",

            "2. QUERY FORMULATION & RETRIEVAL:\n" +
            "   - Create specific legal queries\n" +
            "   - Use appropriate legal terminology\n" +
            "   - Execute vector store search\n" +
            "   - Collect initial results",

            "3. RESULT ASSESSMENT:\n" +
            "   - Evaluate document relevance\n" +
            "   - Check legal applicability\n" +
            "   - Identify information gaps\n" +
            "   - Determine result sufficiency",

            "4. FINAL RESULT SYNTHESIS:\n" +
            "   - Combine relevant findings\n" +
            "   - Present information clearly\n" +
            "   - Highlight key legal points\n" +
            "   - Indicate if exact information not found"
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with retriever-specific logic"""
        
        messages = [
            {"role": "system", "content": self.system_prompt + f"\n'current_task': {self.get_thought_steps()[0]}"}
        ] + state["messages"]

        info_analysis = self.llm.invoke(messages)

            
        
        for i in range(5): # max 5 iterations
            #formulate query
            messages.append({"role": "system", "content": "need_info: " + info_analysis.content + "\n" + "current_task: " + self.get_thought_steps()[1]})
            query = self.llm.invoke(messages)

            #retrieve
            retrieved_content = self.vector_store_retriever.invoke(query.content)

            #assess
            messages.append({"role": "system", "content": "retrieved_content: " + retrieved_content.content + "\n" + "current_task: " + self.get_thought_steps()[2]})
            assessment = self.llm.with_structured_output(RetrieverResponse).invoke(messages)

            #continue
            if assessment.is_enough:
                break
        
        messages.append({"role": "system", "content": "assessment: " + assessment.response + "\n" + "current_task: " + self.get_thought_steps()[3]})
        result = self.llm.invoke(messages)

        
        
        # if 0 <= state["thought_step"]+1 < len(self.get_thought_steps())-1:
        response = {
            "messages": [HumanMessage(content=result["response"], name="retriever")],
            "next": state["caller"],
            "thought_step": state["thought_step"]
        }
        # else:
        #     response = {
        #         "messages": [result["response"]],
        #         "next": result["next_step"],
        #         "thought_step": 0
        #     }
            
        return response
    
    # def _extract_query(self, content: str) -> str:
    #     """Extract or formulate search query from message content"""
    #     # Look for explicit query indicators
    #     query_markers = [
    #         "need information about",
    #         "find legal precedent for",
    #         "search for statute regarding",
    #         "locate law concerning",
    #         "find regulations about"
    #     ]
    #     content_lower = content.lower()
        
    #     for marker in query_markers:
    #         if marker in content_lower:
    #             start_idx = content_lower.index(marker) + len(marker)
    #             end_idx = content.find(".", start_idx)
    #             if end_idx != -1:
    #                 return content[start_idx:end_idx].strip()
        
    #     # If no explicit markers, use key legal terms
    #     legal_terms = ["law", "statute", "regulation", "code", "precedent", "case"]
    #     for term in legal_terms:
    #         if term in content_lower:
    #             start_idx = content_lower.index(term)
    #             end_idx = content.find(".", start_idx)
    #             if end_idx != -1:
    #                 return content[start_idx:end_idx].strip()
        
    #     return content  # Return full content if no specific query found







# from typing import Dict, Any, List, Optional, TypedDict
# from langchain_core.messages import HumanMessage, SystemMessage
# from .base import BaseAgent, AgentState, AgentResponse
# from tools.retrievers import create_law_retriever

# class RetrievalResult(TypedDict):
#     """Structured retrieval result"""
#     findings: List[str]
#     sources: List[str]
#     relevance_scores: List[float]
#     calling_agent: str  # Track who requested the information

# class RetrieverAgent(BaseAgent):
#     """Agent for retrieving and analyzing information from various sources"""
    
#     def __init__(
#         self,
#         docs: List[Any],
#         **kwargs
#     ):
#         system_prompt = """You are an information retrieval specialist. Your role is to:
#         1. Analyze information requests carefully
#         2. Search appropriate sources (legal documents)
#         3. Evaluate source credibility
#         4. Assess information relevance
#         5. Format findings for legal context
        
#         For each search:
#         - Determine best search strategy
#         - Use most relevant source
#         - Validate information accuracy
#         - Structure findings clearly
#         - Track information sources
#         """
#         # Create retrieval tools
#         tools = [
#             create_law_retriever(docs),
#             # create_web_retriever()
#         ]
        
#         super().__init__(system_prompt=system_prompt, tools=tools, **kwargs)
    
#     def get_thought_steps(self) -> List[str]:
#         """Get retriever-specific chain of thought steps"""
#         return [
#             "1. Analyze information request",
#             "2. Determine search strategy",
#             "3. Execute targeted search",
#             "4. Evaluate search results",
#             "5. Structure findings",
#             "6. Validate relevance"
#         ]
    
#     async def process(self, state: AgentState) -> AgentResponse:
#         """Process current state with retriever-specific logic"""
#         # Extract calling agent from state
#         calling_agent = self._get_calling_agent(state["messages"])
        
#         # Add retrieval context to state
#         messages = state["messages"] + [
#             SystemMessage(content=f"""
#                 Information requested by: {calling_agent}
                
#                 Focus on:
#                 1. Specific information requested
#                 2. Context of the request
#                 3. Required level of detail
#                 4. Source credibility
#                 5. Result relevance
                
#                 Structure findings for legal context.
#             """)
#         ]
        
#         # Update state with retrieval context
#         state["messages"] = messages
        
#         # Process through chain of thought
#         response = await super().process(state)
        
#         # If chain of thought is complete, structure the findings
#         if response["cot_finished"]:
#             retrieval_result = self._structure_findings(
#                 response["messages"][-1].content,
#                 calling_agent
#             )
            
#             # Format findings and add to messages
#             response["messages"].append(
#                 HumanMessage(
#                     content=self._format_findings(retrieval_result),
#                     name="retriever"
#                 )
#             )
            
#             # Return control to calling agent
#             response["next"] = calling_agent
        
#         return response
    
#     def _determine_next_agent(self, result: Dict[str, Any]) -> str:
#         """Return control to the agent that requested information"""
#         return self._get_calling_agent(result["messages"])
    
#     def _get_calling_agent(self, messages: List[Dict[str, Any]]) -> str:
#         """Determine which agent requested the information"""
#         # Look through recent messages in reverse
#         for message in reversed(messages):
#             if isinstance(message, dict) and "role" in message:
#                 if message["role"] in ["lawyer", "prosecutor"]:
#                     return message["role"]
#             elif hasattr(message, "name"):
#                 if message.name in ["lawyer", "prosecutor"]:
#                     return message.name
#         return "judge"  # Default to judge if can't determine
    
#     def _structure_findings(self, content: str, calling_agent: str) -> RetrievalResult:
#         """Parse and structure the retrieval findings"""
#         findings: RetrievalResult = {
#             "findings": [],
#             "sources": [],
#             "relevance_scores": [],
#             "calling_agent": calling_agent
#         }
        
#         content_lower = content.lower()
        
#         # Extract findings
#         finding_markers = ["found that", "research shows", "source indicates", "according to"]
#         for marker in finding_markers:
#             if marker in content_lower:
#                 start_idx = content_lower.index(marker)
#                 end_idx = content.find(".", start_idx)
#                 if end_idx != -1:
#                     finding = content[start_idx:end_idx].strip()
#                     findings["findings"].append(finding)
                    
#                     # Look for source citation near the finding
#                     source_end = content.find(")", end_idx)
#                     if source_end != -1:
#                         source = content[end_idx+1:source_end].strip("( ")
#                         findings["sources"].append(source)
                        
#                         # Assess relevance
#                         relevance = self._assess_relevance(finding)
#                         findings["relevance_scores"].append(relevance)
        
#         return findings
    
#     def _assess_relevance(self, finding: str) -> float:
#         """Assess relevance score of a finding"""
#         high_relevance = ["directly relates", "specifically addresses", "exactly matches"]
#         medium_relevance = ["related to", "pertains to", "relevant to"]
#         low_relevance = ["might relate", "could be relevant", "tangentially"]
        
#         finding_lower = finding.lower()
        
#         if any(term in finding_lower for term in high_relevance):
#             return 1.0
#         elif any(term in finding_lower for term in medium_relevance):
#             return 0.7
#         elif any(term in finding_lower for term in low_relevance):
#             return 0.3
#         return 0.5
    
#     def _format_findings(self, findings: RetrievalResult) -> str:
#         """Format structured findings for presentation"""
#         formatted = [f"Information Retrieval Results (for {findings['calling_agent']}):\n"]
        
#         for i, (finding, source, relevance) in enumerate(zip(
#             findings["findings"],
#             findings["sources"],
#             findings["relevance_scores"]
#         )):
#             relevance_label = "High" if relevance > 0.7 else "Medium" if relevance > 0.4 else "Low"
#             formatted.extend([
#                 f"Finding {i+1}:",
#                 f"- Content: {finding}",
#                 f"- Source: {source}",
#                 f"- Relevance: {relevance_label}",
#                 ""
#             ])
        
#         return "\n".join(formatted)
    