import asyncio
import aiohttp
from typing import Dict, Any

async def test_api():
    async with aiohttp.ClientSession() as session:
        # Test case processing
        case = {
            "title": "Software Contract Dispute",
            "description": "Dispute over software development project delivery",
            "client_name": "TechCorp Inc.",
            "case_type": "Contract Dispute",
            "key_facts": [
                "Project started January 2024",
                "Missed deadlines in March 2024",
                "Payment dispute of $200,000"
            ]
        }
        
        async with session.post(
            "http://localhost:8000/process_case",
            json=case
        ) as response:
            result = await response.json()
            print("\nCase Processing Result:")
            print(result)
        
        # Test legal search
        query = "software development contract breach"
        async with session.get(
            f"http://localhost:8000/search_legal_data?query={query}"
        ) as response:
            result = await response.json()
            print("\nLegal Search Result:")
            print(result)

if __name__ == "__main__":
    asyncio.run(test_api()) 