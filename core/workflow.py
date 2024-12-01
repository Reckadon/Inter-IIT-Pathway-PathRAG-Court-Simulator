from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from agents import LawyerAgent, ProsecutorAgent, JudgeAgent, RetrieverAgent
from .state import TrialState, TrialPhase
from .pathway_store import PathwayVectorStore

class TrialWorkflow:
    """Manages the trial workflow using LangGraph"""
    
    def __init__(
        self,
        lawyer: LawyerAgent,
        prosecutor: ProsecutorAgent,
        judge: JudgeAgent,
        retriever: RetrieverAgent,
        vector_store: Optional[PathwayVectorStore] = None
    ):
        self.lawyer = lawyer
        self.prosecutor = prosecutor
        self.judge = judge
        self.retriever = retriever
        self.vector_store = vector_store or PathwayVectorStore()
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the trial workflow graph"""
        workflow = StateGraph(TrialState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_trial)
        workflow.add_node("collect_evidence", self._collect_evidence)
        workflow.add_node("select_speaker", self._select_speaker)
        workflow.add_node("make_argument", self._make_argument)
        workflow.add_node("retrieve_info", self._retrieve_information)
        workflow.add_node("fact_check", self._fact_check)
        workflow.add_node("make_verdict", self._make_verdict)
        
        # Add edges
        workflow.add_edge("initialize", "collect_evidence")
        workflow.add_edge("collect_evidence", "select_speaker")
        
        # Main trial loop
        workflow.add_conditional_edges(
            "select_speaker",
            self._should_continue_trial,
            {
                "continue": "make_argument",
                "verdict": "make_verdict",
            }
        )
        
        workflow.add_conditional_edges(
            "make_argument",
            self._needs_information,
            {
                "yes": "retrieve_info",
                "no": "fact_check"
            }
        )
        
        workflow.add_edge("retrieve_info", "make_argument")
        workflow.add_edge("fact_check", "select_speaker")
        workflow.add_edge("make_verdict", END)
        
        return workflow.compile()
    
    async def _initialize_trial(self, state: TrialState) -> TrialState:
        """Initialize the trial state"""
        state.phase = TrialPhase.INITIALIZATION
        state.messages.append({
            "role": "system",
            "content": f"Trial {state.trial_id} initialized with case: {state.case_details['title']}"
        })
        return state
    
    async def _collect_evidence(self, state: TrialState) -> TrialState:
        """Collect initial evidence"""
        state.phase = TrialPhase.EVIDENCE_COLLECTION
        
        # Collect evidence using retriever
        evidence = await self.retriever.retrieve_information(
            query=state.case_details["description"],
            context={"case_details": state.case_details}
        )
        
        state.evidence.extend(evidence["results"])
        state.messages.append({
            "role": "system",
            "content": "Initial evidence collection completed"
        })
        return state
    
    async def _select_speaker(self, state: TrialState) -> TrialState:
        """Select next speaker"""
        state.phase = TrialPhase.ARGUMENT_EXCHANGE
        
        # Alternate between lawyer and prosecutor
        if not state.current_speaker or state.current_speaker == "prosecutor":
            state.current_speaker = "lawyer"
        else:
            state.current_speaker = "prosecutor"
        
        return state
    
    async def _make_argument(self, state: TrialState) -> TrialState:
        """Generate argument from current speaker"""
        agent = self.lawyer if state.current_speaker == "lawyer" else self.prosecutor
        
        argument = await agent.make_argument({
            "case_details": state.case_details,
            "evidence": state.evidence,
            "messages": state.messages,
            "fact_checks": state.fact_checks
        })
        
        state.messages.append({
            "role": state.current_speaker,
            "content": argument["content"],
            "reasoning": argument["reasoning"]
        })
        
        state.argument_count += 1
        return state
    
    async def _retrieve_information(self, state: TrialState) -> TrialState:
        """Retrieve additional information"""
        last_message = state.messages[-1]
        info = await self.retriever.retrieve_information(
            query=last_message["content"],
            context={
                "case_details": state.case_details,
                "current_speaker": state.current_speaker
            }
        )
        
        state.evidence.extend(info["results"])
        return state
    
    async def _fact_check(self, state: TrialState) -> TrialState:
        """Perform fact checking by judge"""
        last_argument = state.messages[-1]
        
        fact_check = await self.judge.fact_check(
            argument=last_argument,
            context={
                "case_details": state.case_details,
                "evidence": state.evidence,
                "messages": state.messages
            }
        )
        
        state.fact_checks.append(fact_check)
        state.messages.append({
            "role": "judge",
            "content": fact_check["feedback"],
            "validity": fact_check["validity"]
        })
        
        return state
    
    async def _make_verdict(self, state: TrialState) -> TrialState:
        """Generate final verdict"""
        state.phase = TrialPhase.VERDICT
        
        verdict = await self.judge.make_verdict({
            "case_details": state.case_details,
            "evidence": state.evidence,
            "messages": state.messages,
            "fact_checks": state.fact_checks
        })
        
        state.messages.append({
            "role": "judge",
            "content": verdict["decision"],
            "reasoning": verdict["reasoning"]
        })
        
        state.phase = TrialPhase.COMPLETED
        return state
    
    def _should_continue_trial(self, state: TrialState) -> str:
        """Determine if trial should continue"""
        # Check judge's last fact check for trial continuation signal
        if state.fact_checks:
            last_check = state.fact_checks[-1]
            if "conclude trial" in last_check["feedback"].lower():
                return "verdict"
        return "continue"
    
    def _needs_information(self, state: TrialState) -> str:
        """Check if current argument needs more information"""
        last_message = state.messages[-1]
        return "yes" if "need_information" in last_message.get("content", "").lower() else "no"
    
    async def run(self, case_details: Dict[str, Any]) -> Dict[str, Any]:
        """Run the trial workflow"""
        initial_state = TrialState(
            trial_id=str(uuid.uuid4()),
            case_details=case_details
        )
        
        final_state = await self.graph.ainvoke(initial_state)
        return {
            "trial_id": final_state.trial_id,
            "verdict": final_state.messages[-1],
            "history": final_state.messages,
            "evidence": final_state.evidence,
            "fact_checks": final_state.fact_checks
        } 