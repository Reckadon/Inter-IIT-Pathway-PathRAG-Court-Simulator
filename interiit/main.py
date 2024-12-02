import asyncio
from .keyword_extractor_agent import KeywordExtractorAgent
from dotenv import load_dotenv
load_dotenv()


class Document:
    def __init__(self, content: str):
        self.content = content

async def main():
    # Sample user case description
    user_case = """
I am involved in a dispute with my employer over wrongful termination. They claim that I violated company policy, but I believe I was terminated due to discrimination based on my age. I have documentation of my performance reviews and emails that suggest I was meeting all job requirements.
"""
    # Sample relevant documents
    documents = [
        Document(content="""Company Policy Document:
- All employees must adhere to the code of conduct.
- Equal opportunity employment is provided regardless of age, race, or gender.
- Termination procedures require a formal review process.
"""),
        Document(content="""Email from Manager:
"Your performance has been excellent over the past year. Keep up the good work!"
"""),
        Document(content="""Performance Review:
- Exceeds expectations in all areas.
- No violations of company policy noted.
"""),
    ]

    agent = KeywordExtractorAgent(documents=documents)

    keywords_result = await agent.extract_keywords(user_case=user_case)

    # Print the extracted keywords
    print("Extracted Keywords:")
    for keyword in keywords_result["keywords"]:
        print(f"- {keyword}")

if __name__ == "__main__":
    asyncio.run(main())