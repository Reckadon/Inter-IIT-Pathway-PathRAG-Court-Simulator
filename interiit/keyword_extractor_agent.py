
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class KeywordExtractorAgent:

    def __init__(
        self,
        documents: List[Any],
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        self.documents = documents
        self.llm = llm or ChatGoogleGenerativeAI(
            model="chat-bison",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.system_prompt = """You are an assistant that helps users extract relevant keywords from legal documents for their legal case. Your tasks are:

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

    async def extract_keywords(
        self,
        user_case: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract relevant keywords from documents based on the user's case description."""
        documents_content = "\n".join([doc.content for doc in self.documents])
        prompt = f"""User Case Description:
{user_case}

Relevant Documents:
{documents_content}

Based on the above user case and documents, extract a list of relevant keywords, phrases, and legal terms that the user can use to search for supportive cases and information. The keywords should be specific to the user's case and cover all important aspects.

Provide the list of keywords in bullet point format.
"""
        response = await self._get_llm_response(prompt)
        keywords = self._parse_keywords(response)
        return {
            "type": "keywords",
            "keywords": keywords
        }

    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        messages = [
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = await self.llm.invoke(messages)
        return response.content

    def _parse_keywords(self, response: str) -> List[str]:

        lines = response.strip().split('\n')
        keywords = [line.strip('- ').strip() for line in lines if line.strip()]
        return keywords