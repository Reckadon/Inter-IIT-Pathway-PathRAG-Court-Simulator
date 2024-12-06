from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
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


def create_law_retriever(private=False) -> BaseTool:
    """Create vector store retriever for legal documents"""
    if private:
        vector_store = PathwayVectorStore('private', './private_documents', 8765)
    else:
        vector_store = PathwayVectorStore('public', './public_documents', 8766)

    client = vector_store.get_client()
    
    retriever = client.as_retriever()

    return retriever
    # Create retriever tool
    # return create_retriever_tool(
    #     retriever,
    #     "search_laws",
    #     "Search through laws, statutes, and constitution documents"
    # )
class RetrieverResponse(BaseModel):
    """Structured retriever response"""
    # response: str = Field(description="The retriever's assessment of the retrieved content")
    is_enough: bool = Field(description="Whether the retrieved content is enough to answer the request")

class Queries(BaseModel):
    private_query: str = Field(description="Query for private retriever (user case files or documents). 'none' if not needed")  
    public_query: str = Field(description="Query for public retriever (public docs like IPC, legal case precedents, etc) 'none' if not needed")
class RetrieverAgent:
    """Agent for retrieving and analyzing legal documents from vector store"""
    
    def __init__(
        self,
        llms,
        # **kwargs
    ):
        self.private_retriever = create_law_retriever(private=True)
        self.public_retriever = create_law_retriever(private=False)
        # self.llm = llm or ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
        self.llms = llms
        self.system_prompt = """
"You are a legal research assistant specializing in retrieving relevant legal provisions, case laws, and statutes from a vector database of the Indian Penal Code (IPC) and related legal documents."
"Formulate queries based on inputs from the judge, lawyer, or prosecutor, ensuring precision in the retrieval process."
"Evaluate the retrieved text for relevance and clarity before sharing it with the requesting agent. If the retrieval is insufficient, refine the query and attempt again."
"you have acsess to private retrierver (contains user case files or documents) and public retriever (contains public docs like IPC, legal case precedents, etc)"
"Your role is critical in supporting the legal arguments by providing accurate and contextually appropriate legal references."
"Ensure that your outputs are succinct, relevant, and formatted for easy understanding by the requesting agent."

you will go through the following chain of thought steps:
1. Analyze the information request.
2. Form the queries.
3. Assess the retrieved results.
4. Provide accurate excerpts of information

Do only current task at a time. Avoid very long responses.
"""

    def get_thought_steps(self) -> List[str]:
        """Get retriever-specific chain of thought steps"""
        return [
            "1. Analyze the information request received from the lawyer or prosecutor and Note the key words and points.",
            "2. Form very good queries to Retrieve the most relevant text needed. one query for each private retriever (user case files or documents) or public retriever (contains public docs like IPC, legal case precedents, etc) 'none' if a specific retriever is not needed.",
            "3. Assess the retrieved results and determine If they are relevent and enough. if yes, set 'is_enough' to True. if not, set 'is_enough' false",
            "4. Provide the lawyer or prosecutor with accurate excerpts of relevant laws based on the request, ensuring clarity.If no relevant law is found, respond with 'No relevant law found in database.'"
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Process current state with retriever-specific logic"""
        
        messages = [
            {"role": "system", "content": self.system_prompt + f"\n'current_task': {self.get_thought_steps()[0]}"}
        ] + state["messages"]

        # info_analysis = self.llm.invoke(messages)

        for i,llm in enumerate(self.llms):
            try:
                info_analysis = llm.invoke(messages)
                break
            except Exception as e:
                print(f"LLM {i} failed with error: {e}")
                continue

            
        for i in range(1): # max 5 iterations
            #formulate query
            messages.append({"role": "system", "content": "need_info: " + info_analysis.content + "\n" + "current_task: " + self.get_thought_steps()[1]})
            # queries = self.llm.with_structured_output(Queries).invoke(messages)
            for i,llm in enumerate(self.llms):
                try:
                    queries = llm.with_structured_output(Queries).invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")

            #retrieve
            private_retrieved_content = self.private_retriever.invoke(queries.private_query) if queries.private_query.lower() != 'none' else 'None'
            public_retrieved_content = self.public_retriever.invoke(queries.public_query) if queries.public_query.lower() != 'none' else 'None'

            #assess
            messages.append({"role": "system", "content": "private_retrieved_content: " + str(private_retrieved_content) + "\npublic_retrieved_content: " + str(public_retrieved_content) + "\ncurrent_task: " + self.get_thought_steps()[2]})
            # assessment = self.llm.with_structured_output(RetrieverResponse).invoke(messages)
            for i,llm in enumerate(self.llms):
                try:
                    assessment = llm.with_structured_output(RetrieverResponse).invoke(messages)
                    break
                except Exception as e:
                    print(f"LLM {i} failed with error: {e}")
                    continue

            #continue
            if assessment.is_enough:
                break

                
            
        
        messages.append({"role": "system", "content": "private_retrieved_content: " + str(private_retrieved_content) + "\npublic_retrieved_content: " + str(public_retrieved_content) + "\ncurrent_task: " + self.get_thought_steps()[3]})
        # result = self.llm.invoke(messages)
        for i,llm in enumerate(self.llms):
            try:
                result = llm.invoke(messages)
                break
            except Exception as e:
                print(f"LLM {i} failed with error: {e}")
                continue

        
        
        response = {
            "messages": [HumanMessage(content=result.content, name="retriever")],
            "next": state["caller"],
            "thought_step": state["thought_step"],
            "caller": "retriever"
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
    