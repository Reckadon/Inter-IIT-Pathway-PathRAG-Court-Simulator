import os
from crewai import Crew, Process
from textwrap import dedent
from .agents import DataRetrieverAgents
from .tasks import RetrievalTasks

from dotenv import load_dotenv
# load_dotenv()

# Debug prints
print("Loading environment variables...")
load_dotenv()
print("GROQ_API_KEY loaded:", "Yes" if os.environ.get('GROQ_API_KEY') else "No")
print("Actual GROQ_API_KEY:", "Present" if os.environ.get('GROQ_API_KEY') is not None else "Missing")

class DataRetrievalCrew:
    def __init__(self, argument):
        self.argument = argument
   

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = DataRetrieverAgents()
        tasks = RetrievalTasks()

        # Define your custom agents and tasks here
        legal_researcher = agents.legal_researcher()
        legal_assistant = agents.legal_assistant()

        # Define tasks
        search_queries_task = tasks.generate_search_queries(legal_researcher, argument)
        retrieve_info_task = tasks.retrieve_information(legal_researcher, argument)
        formulate_counterargument_task = tasks.formulate_counterargument(legal_researcher, argument)
        # evaluate_counterargument_task = tasks.evaluate_counterargument(legal_assistant, argument)

        # Create the crew with sequential processing
        crew1 = Crew(
            agents=[legal_researcher, legal_assistant],
            tasks=[
                search_queries_task,
                retrieve_info_task,
                formulate_counterargument_task,
                # evaluate_counterargument_task
            ],
            process=Process.sequential,
        )

        result = crew1.kickoff()
        return result

        # if 'yes' in str(result).lower()[:10]:
        #     return formulate_counterargument_task.output
        
        # else:

        #     request_additional_info_task = tasks.request_additional_information(legal_assistant, argument, formulate_counterargument_task.output)
        #     refine_counterargument_task = tasks.refine_counterargument(legal_assistant, argument, formulate_counterargument_task.output)

        #     crew2 = Crew(
        #     agents=[legal_assistant],
        #     tasks=[request_additional_info_task, refine_counterargument_task],
        #     process=Process.sequential,
        #     # task_routing=routing
        #     )

        #     result = crew2.kickoff()
        #     return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Counter-Arguement Generator")
    print("-------------------------------")
    argument = str(input(dedent("""Enter argument: """)))

    info_gatherer_crew = DataRetrievalCrew(argument)
    counter_argument = info_gatherer_crew.run()
    print("\n\n########################")
    print("## Here is your information gatherer crew run result:")
    print("########################\n")
    print(counter_argument)
