import os
from groq import Groq
from typing import Dict, Any, List


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

Based on the above user case and documents, extract a list of relevant keywords, phrases, and legal terms that the user can use to search for supportive cases and information. The keywords should be specific to the user's case and cover all important aspects.

Provide the list of keywords in bullet point format.
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
