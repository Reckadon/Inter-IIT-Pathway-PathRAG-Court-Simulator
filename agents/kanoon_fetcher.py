import os
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from .base import BaseAgent, AgentState, AgentResponse
from .misc.filestorage import FileStorage
from .misc.ik import IKApi
from groq import Groq
import argparse
from dotenv import load_dotenv


class KeywordExtractorAgent:
    def __init__(self, documents: List[Any]):
        self.documents = documents

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found in environment variables.")
        self.client = Groq(api_key=api_key)

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

        # Initialize chat history with the system prompt
        self.chat_history = [self.system_prompt]

    def extract_keywords(self, user_case: str) -> Dict[str, Any]:
        """Extract relevant keywords based on the user's case and documents."""
        documents_content = "\n".join([doc.content for doc in self.documents])

        self.chat_history.append({
            "role": "user",
            "content": f"""User Case Description:
{user_case}

Relevant Documents:
{documents_content}

Based on the above user case and documents, extract a list of TOP 5 relevant keywords, phrases, or legal terms that the user can use to search for supportive cases and information. The keywords should be specific to the user's case and cover all important aspects. Do Not provide anything else than the keywords. No reasoning is needed.

Provide the list of just the 5 most relevant keywords in bullet point format.
"""
        })

        # Send the request
        print("Sending request to Groq API...")
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=self.chat_history,
            max_tokens=100,
            temperature=1.2
        )
        print("Response received from Groq API.")

        return self._parse_keywords(response.choices[0].message.content)

    def _parse_keywords(self, response: str) -> Dict[str, Any]:
        """Parse the response to extract keywords."""
        lines = response.strip().split("\n")
        keywords = [line.strip("- ").strip() for line in lines if line.strip()]
        return {"type": "keywords", "keywords": keywords}


load_dotenv()


class Document:
    def __init__(self, content: str):
        self.content = content

class FetchingAgent(BaseAgent):
    """Agent responsible for fetching relevant docs from the kanoon api"""
    
    def __init__(self, **kwargs):
        print("initialised kanoon fetcher...")
        super().__init__(**kwargs)

    
    async def process(self, state: AgentState) -> AgentResponse:
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

        # docs = state["messages"][-1].content
        
        # Extract Keywords
        agent = KeywordExtractorAgent(documents=documents)
        keywords_result = agent.extract_keywords(user_case=state["messages"][-1].content)

        # Step 2: Use Extracted Keywords for Searching Relevant Cases
        print("Extracted Keywords:")
        keywords = keywords_result["keywords"][1:]  # Skip the first line
        for keyword in keywords:
            print(f"- {keyword}")

        # Specify max_docs per keyword
        MAX_DOCS_PER_KEYWORD = 2

        all_doc_ids = []
        for keyword in keywords[:5]:
            print(f"Searching for keyword: {keyword}")
            doc_ids = ikapi.save_search_results(keyword, max_docs=MAX_DOCS_PER_KEYWORD)
            all_doc_ids.extend(doc_ids)

        print(f"Total documents fetched: {len(all_doc_ids)}")