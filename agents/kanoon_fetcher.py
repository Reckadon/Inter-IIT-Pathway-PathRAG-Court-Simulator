import os
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .base import AgentState
from .misc.filestorage import FileStorage
from .misc.ik import IKApi
import argparse
from dotenv import load_dotenv

# Class to handle keyword extraction from legal documents
class KeywordExtractorAgent:
    def __init__(self,
        documents: List[Any], # List of documents to process
        llms # List of LLMs for handling tasks
    ):
        self.documents = documents
        # self.llm = llm or ChatGoogleGenerativeAI(
        #     model="gemini-1.5-flash",
        #     temperature=0
        # )
        self.llms = llms
        # Define the system prompt for the task
        self.system_prompt = {
            "role": "system",
            "content": """You are an assistant that helps users extract relevant keywords from legal documents for their legal case. Your tasks are:

            1. Understand the user's legal case based on the provided description.
            2. Analyze the provided documents related to the case.
            3. Identify and extract keywords, phrases, and legal terms that are most relevant to the user's case.
            4. Provide the user with a list of these keywords to help them search for supportive cases and information on law websites.
            5. Ensure that the keywords are specific, relevant, and cover all aspects of the user's case.

            Guidelines:

            - Focus on legal terms, case identifiers, statutory references, key legal concepts, and any unique aspects of the case.
            - Consider synonyms and related terms that might be used in legal databases.
            - Do not include irrelevant information or overly general terms.
            """
        }
    async def extract_keywords(self, user_case: str) -> Dict[str, Any]:
        """Extract relevant keywords based on the user's case and documents."""
        documents_content = "\n".join([doc.content for doc in self.documents])
        prompt = f"""User Case Description:
{user_case}

Relevant Documents:
{documents_content}

Based on the above user case and documents, extract a list of relevant keywords, phrases, and legal terms that the user can use to search for supportive cases and information. The keywords should be specific to the user's case and cover all important aspects.

Provide the list of keywords in bullet point format.
"""
        response = self._get_llm_response(prompt)
        keywords = self._parse_keywords(response)
        return keywords

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from the Gemini-pro LLM."""
        # response = self.llm.invoke([
        #     {"role": "system", "content": self.system_prompt['content']},
        #     {"role": "user", "content": prompt}
        # ])
        for i,llm in enumerate(self.llms):
            try:
                response = llm.invoke([
                    {"role": "system", "content": self.system_prompt['content']},
                    {"role": "user", "content": prompt}
                ])
                break
            except Exception as e:
                print(f"LLM {i} failed with error: {e}")
                continue
        return response.content

    def _parse_keywords(self, response: str) -> Dict[str, Any]:
        """Parse the response to extract keywords."""
        lines = response.strip().split("\n")
        keywords = [
            line.strip("- ").strip() 
            for line in lines
            if line.strip()
        ]
        # Return the keywords only once in the correct structure
        return {"type": "keywords", "keywords": keywords}


load_dotenv()


class Document:
    def __init__(self, content: str):
        self.content = content

class FetchingAgent:
    """Agent responsible for fetching relevant docs from the kanoon api"""
    
    def __init__(self, llms):
        self.llms = llms
        print("initialised kanoon fetcher...")
        # super().__init__(**kwargs)

    
    async def process(self, state: AgentState) -> AgentState:
        """Process current state with fetching-specific logic"""
        kanoon_api_key = os.getenv("KANOON_API_KEY")
        if not kanoon_api_key:
            raise ValueError("KANOON_API_KEY not found in environment variables.")

        data_directory = "public_documents"
        os.makedirs(data_directory, exist_ok=True)
        filestorage = FileStorage(data_directory)

        args = argparse.Namespace(
            token=kanoon_api_key,
            datadir=data_directory,
            maxpages=2,  # Limit number of pages
            maxcites=0,
            maxcitedby=0,
            orig=False,
            pathbysrc=True
        )

        # Initialize Indian Kanoon API client
        ikapi = IKApi(args, filestorage)

        # Sample user case description
        user_case = """
            I am involved in a dispute with my employer over wrongful termination. They claim that I violated company policy, but I believe I was terminated due to discrimination based on my age. I have documentation of my performance reviews and emails that suggest I was meeting all job requirements.
            """
        # Sample relevant documents
        documents = [
            Document(
                content="""Company Policy Document:
            - All employees must adhere to the code of conduct.
            - Equal opportunity employment is provided regardless of age, race, or gender.
            - Termination procedures require a formal review process.
            """
            ),
            Document(
                content="""Email from Manager:
                "Your performance has been excellent over the past year. Keep up the good work!"
                """
            ),
            Document(
                content="""Performance Review:
                - Exceeds expectations in all areas.
                - No violations of company policy noted.
                """
            ),
        ]

        # Step1: Extract Keywords
        agent = KeywordExtractorAgent(documents=documents, llms=self.llms)
        keywords_result = await agent.extract_keywords(user_case=state["messages"][-1].content)  # Await the coroutine

        # Step 2: Use Extracted Keywords for Searching Relevant Cases
        print("Extracted Keywords:")
        keywords = keywords_result["keywords"]
        for keyword in keywords:
            print(f"- {keyword}")

        # Specify max_docs per keyword
        MAX_DOCS_PER_KEYWORD = 2

        all_doc_ids = []
        for keyword in keywords[:5]: # Use only the first 5 keywords
            print(f"Searching for keyword: {keyword}")
            doc_ids = ikapi.save_search_results(keyword, max_docs=MAX_DOCS_PER_KEYWORD)
            all_doc_ids.extend(doc_ids)
            
        # Print the total number of documents fetched
        print(f"Total documents fetched: {len(all_doc_ids)}")
