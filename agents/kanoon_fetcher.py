import os
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .base import AgentState
from .misc.filestorage import FileStorage
from .misc.ik import IKApi
import argparse
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Class to handle keyword extraction from legal documents
class KeywordExtractorAgent:
    def __init__(self,
        documents: List[Any],
        llms
    ):
        self.documents = documents
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
        documents_content = "\n".join([doc for doc in self.documents])
        # print("case files from user:",documents_content)
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
        """Get response from the LLM."""
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

        # List to store the content of the text files uploaded by user
        folder_path = 'private_documents'
        documents = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a text file
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(content)

        # Extract Keywords
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

        # converting fetched pdfs to texts
        # Path to the 'public' directory
        base_path = "public_documents"

        # Loop through all subfolders in the base directory
        for top_level_folder in os.listdir(base_path):
            top_level_path = os.path.join(base_path, top_level_folder)
            
            # Skip if not a folder
            if not os.path.isdir(top_level_path):
                continue
            
            # Text content to combine for the current top-level folder
            combined_text = []
            
            # Process each subfolder within the top-level folder
            for subfolder in os.listdir(top_level_path):
                subfolder_path = os.path.join(top_level_path, subfolder)
                
                # Skip if not a folder
                if not os.path.isdir(subfolder_path):
                    continue
                
                # Process each file within the subfolder
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    
                    # Skip toc.txt
                    if file_name == "toc.txt":
                        continue
                    
                    # Check if the file is a PDF
                    if file_name.endswith(".pdf"):
                        try:
                            # Load and extract text using PyPDFLoader
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()
                            
                            # Append extracted text to the combined list
                            for doc in documents:
                                combined_text.append(doc.page_content)
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
            
            # Write the combined text to a single file named after the top-level folder
            if combined_text:
                output_file_path = os.path.join(base_path, f"{top_level_folder}.txt")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write("\n\n".join(combined_text))
                print(f"Created combined text file: {output_file_path}")

