from .internet_data_retriever.internet_data import DataRetrievalCrew
from .base import AgentState

class WebSearchAgent:
    def __init__(self):
        self.data_retriever_crew = DataRetrievalCrew

    def process(self, state: AgentState) -> AgentState:
        return self.data_retriever_crew(state["messages"][-1].content).run()
