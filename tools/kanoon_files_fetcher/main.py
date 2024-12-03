import asyncio
import os
import argparse
from keyword_extractor_agent import KeywordExtractorAgent
from dotenv import load_dotenv
from ik import IKApi
from filestorage import FileStorage


load_dotenv()


class Document:
    def __init__(self, content: str):
        self.content = content


async def main():

    kanoon_api_key = os.getenv("KANOON_API_KEY")
    if not kanoon_api_key:
        raise ValueError("KANOON_API_KEY not found in environment variables.")

    data_directory = "data/kanoon_results"
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

    # Extract Keywords
    agent = KeywordExtractorAgent(documents=documents)
    keywords_result = agent.extract_keywords(user_case=user_case)

    # Step 2: Use Extracted Keywords for Searching Relevant Cases
    print("Extracted Keywords:")
    keywords = keywords_result["keywords"][1:]  # Skip the first line
    for keyword in keywords:
        print(f"- {keyword}")

    # Specify max_docs per keyword
    MAX_DOCS_PER_KEYWORD = 10

    all_doc_ids = []
    for keyword in keywords[:5]:
        print(f"Searching for keyword: {keyword}")
        doc_ids = ikapi.save_search_results(keyword, max_docs=MAX_DOCS_PER_KEYWORD)
        all_doc_ids.extend(doc_ids)

    print(f"Total documents fetched: {len(all_doc_ids)}")


if __name__ == "__main__":
    asyncio.run(main())
