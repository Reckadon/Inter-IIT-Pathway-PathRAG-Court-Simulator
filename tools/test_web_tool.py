from typing import Dict, Any
import asyncio
from .web_search import LegalWebSearchTool
from rich.console import Console
from rich.table import Table

async def test_web_search():
    console = Console()
    search_tool = LegalWebSearchTool()
    
    while True:
        console.print("\n[bold cyan]Legal Search Tool[/bold cyan]")
        console.print("Enter search query (or 'quit' to exit):")
        
        query = input("> ").strip()
        if query.lower() == 'quit':
            break
        
        with console.status("[bold green]Searching legal database..."):
            results = search_tool.search(query)
        
        # Display results
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Type", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Content", style="white")
        table.add_column("Relevance", justify="right", style="yellow")
        
        for result in results["results"]:
            content = result["content"]
            table.add_row(
                result["type"],
                content["title"],
                content.get("summary", content.get("description", "N/A"))[:100] + "...",
                f"{result['relevance']:.2f}"
            )
        
        console.print(table)

if __name__ == "__main__":
    asyncio.run(test_web_search()) 