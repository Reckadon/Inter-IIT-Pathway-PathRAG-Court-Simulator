from typing import Dict, Any, List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START,  END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
import os

from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent, FetchingAgent, WebSearcherAgent
from agents import AgentState

class TrialWorkflow:
    """Manages the trial workflow using LangGraph"""
    
    def __init__(
        self,
        lawyer: LawyerAgent,
        prosecutor: ProsecutorAgent,
        judge: JudgeAgent,
        retriever: RetrieverAgent,
        kanoon_fetcher: FetchingAgent,
        web_searcher: WebSearcherAgent
    ):
        self.lawyer = lawyer
        self.prosecutor = prosecutor
        self.judge = judge
        self.retriever = retriever # or RetrieverAgent(docs=docs)
        self.kanoon_fetcher = kanoon_fetcher
        self.web_searcher = web_searcher
        self.memory = MemorySaver()
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the trial workflow graph"""
        # Initialize the graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("kanoon_fetcher", self._kanoon_fetcher_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("lawyer", self._lawyer_node)
        workflow.add_node("prosecutor", self._prosecutor_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("web_searcher", self._web_search_node)
        workflow.add_node("user_feedback", self._user_feedback_node)
        
        # Start with judge
        workflow.add_edge(START, "kanoon_fetcher")
        workflow.add_edge("kanoon_fetcher", "prosecutor")
        workflow.add_edge("user_feedback", "lawyer")
        
        # Judge manages the flow
        workflow.add_conditional_edges(
            "judge",
            self._route_from_judge,
            {
                "lawyer": "lawyer",
                "prosecutor": "prosecutor",
                "retriever": "retriever",
                "self": "judge",
                "web_searcher": "web_searcher",
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
                "web_searcher": "web_searcher",
                "user_feedback": "user_feedback",
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
                "web_searcher": "web_searcher",
                "self": "prosecutor"  # For Chain of Thought
            }
        )
        
        # Retriever returns to calling agent
        workflow.add_conditional_edges(
            "retriever",
            self._route_from_retriever,
            {
                "lawyer": "lawyer",
                "prosecutor": "prosecutor",
                "judge": "judge"
            }
        )

        # Web Search returns to calling agent
        workflow.add_conditional_edges(
            "web_searcher",
            self._route_from_retriever,
            {
                "judge": "judge",
                "lawyer": "lawyer",
                "prosecutor": "prosecutor"
            }
        )
        
        return workflow.compile(checkpointer=self.memory, interrupt_before=["user_feedback"])
    
    async def _kanoon_fetcher_node(self, state: AgentState) -> AgentState:
        """Kanoon Fetcher node processing"""
        return await self.kanoon_fetcher.process(state)
    
    async def _judge_node(self, state: AgentState) -> AgentState:
        """Judge node processing"""
        # print(f"Judge node processing with state: {state}")
        return await self.judge.process(state)
    
    async def _lawyer_node(self, state: AgentState) -> AgentState:
        """Lawyer node processing"""
        # print(f"Lawyer node processing with state: {state}")
        return await self.lawyer.process(state)
    
    async def _prosecutor_node(self, state: AgentState) -> AgentState:
        """Prosecutor node processing"""
        # print(f"Prosecutor node processing with state: {state}")
        return await self.prosecutor.process(state)
    
    async def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever node processing"""
        # print(f"Retriever node processing with state: {state}")
        return await self.retriever.process(state)
    
    async def _web_search_node(self, state: AgentState) -> AgentState:
        """Web Search node processing"""
        # print(f"Web Search node processing with state: {state}")
        return await self.web_searcher.process(state)
    
    async def _user_feedback_node(self, state: AgentState) -> AgentState:
        """User feedback node processing"""
        # print(f"User feedback node processing with state: {state}")
        pass
    
    def _route_from_judge(self, state: AgentState) -> str:
        """Route based on judge's decision"""
        # Check if verdict is ready
        # if any("verdict" in msg.content.lower() 
        #        for msg in state["messages"] if hasattr(msg, "content")):
        #     return "END"
        return state["next"]
    
    def _route_from_agent(self, state: AgentState) -> str:
        """Route from lawyer or prosecutor"""
        # Check for Chain of Thought continuation
        # if not state["cot_finished"]:
        #     return "self"
        return state["next"]
    
    def _route_from_retriever(self, state: AgentState) -> str:
        """Route from retriever back to calling agent"""
        # Get the last message to determine calling agent
        # for msg in reversed(state["messages"]):
        #     if hasattr(msg, "name"):
        #         if msg.name in ["lawyer", "prosecutor"]:
        #             return msg.name
        return state["next"]  # Default to judge if can't determine
    
    async def run(self, user_prompt: str) -> Dict[str, Any]:
        """Run the trial workflow"""
        # Initialize state
        initial_state = AgentState(
            messages=[
                HumanMessage(content=user_prompt)
            ],
            next="kanoon_fetcher",
            thought_step=0,
        )

        print(f"Initial state: {initial_state}")

        thread = {"configurable": {"thread_id": "1"}}
        
        async for a in self.graph.astream(initial_state,thread):
            print(a)
            print("-"*100)

        user_input = "argument is not strong"

        while True:
            self.graph.update_state(values={"user_feedback": user_input}, as_node="user_feedback")  

            async for a in self.graph.astream(None,thread):
                print(a)
                print("-"*100)
            try:
                if a.judge.next == 'END':
                    break
            except:
                pass

        
        # Run the workflow
        # final_state = await self.graph.ainvoke(initial_state)
        
        # Extract results
        # return {
        #     "messages": final_state["messages"],
        #     "verdict": next(
        #         (msg for msg in reversed(final_state["messages"]) 
        #          if hasattr(msg, "name") and msg.name == "judge"),
        #         None
        #     )
        # }
    
    def visualize(self):
        """Visualize the workflow graph"""

        png_graph = self.graph.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")

    #     # Clean up by removing the image file
    #     # os.remove(graph_path)

    # def visualize(self):
    #     """Visualize the workflow graph"""
    #     try:
    #         from IPython.display import Image, display
    #         display(Image(self.graph.get_graph().draw_mermaid_png()))
    #     except ImportError:
    #         print("IPython display not available. Install IPython to visualize the graph.")


