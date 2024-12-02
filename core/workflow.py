from typing import Dict, Any, List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent
from .state import AgentState

class TrialWorkflow:
    """Manages the trial workflow using LangGraph"""
    
    def __init__(
        self,
        lawyer: LawyerAgent,
        prosecutor: ProsecutorAgent,
        judge: JudgeAgent,
        docs: List[Any],
        retriever: Optional[RetrieverAgent] = None,
    ):
        self.lawyer = lawyer
        self.prosecutor = prosecutor
        self.judge = judge
        self.retriever = retriever or RetrieverAgent(docs=docs)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the trial workflow graph"""
        # Initialize the graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("lawyer", self._lawyer_node)
        workflow.add_node("prosecutor", self._prosecutor_node)
        workflow.add_node("retriever", self._retriever_node)
        
        # Start with judge
        workflow.add_edge("START", "judge")
        
        # Judge manages the flow
        workflow.add_conditional_edges(
            "judge",
            self._route_from_judge,
            {
                "lawyer": "lawyer",
                "prosecutor": "prosecutor",
                "END": END
            }
        )
        
        # Lawyer can go to judge or retriever
        workflow.add_conditional_edges(
            "lawyer",
            self._route_from_agent,
            {
                "judge": "judge",
                "retriever": "retriever",
                "self": "lawyer"  # For Chain of Thought
            }
        )
        
        # Prosecutor can go to judge or retriever
        workflow.add_conditional_edges(
            "prosecutor",
            self._route_from_agent,
            {
                "judge": "judge",
                "retriever": "retriever",
                "self": "prosecutor"  # For Chain of Thought
            }
        )
        
        # Retriever returns to calling agent
        workflow.add_conditional_edges(
            "retriever",
            self._route_from_retriever,
            {
                "lawyer": "lawyer",
                "prosecutor": "prosecutor"
            }
        )
        
        return workflow.compile()
    
    async def _judge_node(self, state: AgentState) -> AgentState:
        """Judge node processing"""
        return await self.judge.process(state)
    
    async def _lawyer_node(self, state: AgentState) -> AgentState:
        """Lawyer node processing"""
        return await self.lawyer.process(state)
    
    async def _prosecutor_node(self, state: AgentState) -> AgentState:
        """Prosecutor node processing"""
        return await self.prosecutor.process(state)
    
    async def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever node processing"""
        return await self.retriever.process(state)
    
    def _route_from_judge(self, state: AgentState) -> str:
        """Route based on judge's decision"""
        # Check if verdict is ready
        if any("verdict" in msg.content.lower() 
               for msg in state["messages"] if hasattr(msg, "content")):
            return "END"
        return state["next"]
    
    def _route_from_agent(self, state: AgentState) -> str:
        """Route from lawyer or prosecutor"""
        # Check for Chain of Thought continuation
        if not state["cot_finished"]:
            return "self"
        return state["next"]
    
    def _route_from_retriever(self, state: AgentState) -> str:
        """Route from retriever back to calling agent"""
        # Get the last message to determine calling agent
        for msg in reversed(state["messages"]):
            if hasattr(msg, "name"):
                if msg.name in ["lawyer", "prosecutor"]:
                    return msg.name
        return "judge"  # Default to judge if can't determine
    
    async def run(self, case_details: Dict[str, Any]) -> Dict[str, Any]:
        """Run the trial workflow"""
        # Initialize state
        initial_state = AgentState(
            messages=[
                HumanMessage(content=f"New case: {case_details['title']}\n\n{case_details['description']}")
            ],
            next="judge",
            thought_step=None,
            cot_finished=True
        )
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract results
        return {
            "messages": final_state["messages"],
            "verdict": next(
                (msg for msg in reversed(final_state["messages"]) 
                 if hasattr(msg, "name") and msg.name == "judge"),
                None
            )
        }
    
    def visualize(self):
        """Visualize the workflow graph"""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
            print("IPython display not available. Install IPython to visualize the graph.")


if __name__ == "__main__":
    workflow = TrialWorkflow(lawyer, prosecutor, judge, docs)
    workflow.visualize() 
