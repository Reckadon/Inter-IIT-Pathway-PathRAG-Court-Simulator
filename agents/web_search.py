from .Internet_data_retriever.internet_data import DataRetrievalCrew
from .base import AgentState
from langchain_core.messages import HumanMessage

class WebSearcherAgent:
    def __init__(self):
        self.data_retriever_crew = DataRetrievalCrew

    def process(self, state: AgentState) -> AgentState:
        result = self.data_retriever_crew(state["messages"][-1].content).run()
        # print(result)
        # print(type(result))
        return {
            "messages": [HumanMessage(content=result.raw, name="web_searcher")],
            "next": state["caller"],
            "thought_step": state["thought_step"],
            "caller": "web_searcher"
        }
