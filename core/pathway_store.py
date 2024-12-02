import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings

PATHWAY_PORT = 8765


class PathwayVectorStore:
    def __init__(self, name = "pathway_vector_store", path="documents/lorem.md", port=PATHWAY_PORT):
        """Initialize the Store with the docs from given path"""
        self.name = name
        self.path = path
        self.port = port

        try:
            self.data_sources = []
            self.data_sources.append(
                pw.io.fs.read(
                    path,
                    format="binary",
                    mode="streaming",
                    with_metadata=True,
                )
            )

            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, separator=" ")

            embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            self.vector_server = VectorStoreServer.from_langchain_components(
                *self.data_sources,
                splitter=text_splitter,
                embedder=embeddings_model,
            )

            print(f"Starting VectorStoreServer: '{self.name}'...")
            self.vector_server.run_server(
                host="127.0.0.1",
                port=port,
                threaded=True,
                with_cache=False,
            )
            time.sleep(30)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    def add_item(self, item_name, quantity, price):
        """
        Add an item to the Vector store.

        Parameters:
        item_name (str): The name of the item.
        quantity (int): The quantity of the item.
        price (float): The price of the item.
        """
        if item_name in self.inventory:
            self.inventory[item_name]['quantity'] += quantity
        else:
            self.inventory[item_name] = {'quantity': quantity, 'price': price}

    def query_item(self, item_name):
        """
        Query the details of an item in the store.

        Parameters:
        item_name (str): The name of the item to query.

        Returns:
        dict: A dictionary containing the item's quantity and price, or None if not found.
        """
        return self.inventory.get(item_name, None)
    



if __name__ == "__main__":
    

    print("\nmaking client...\n")
    client = VectorStoreClient(
        host="127.0.0.1",
        port=PATHWAY_PORT,
    )
    query = "What is lorem?"
    docs = client(query, k=1)
    print(docs)
















# from typing import List, Dict, Any, Optional
# from pathlib import Path
# import pathway as pw
# from pathway.xpacks.llm.vector_store import VectorStoreServer
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.pathway import PathwayVectorClient
# from core.config import settings

# class PathwayVectorStore:
#     """Pathway-based vector store implementation using Google's Gemini"""
    
#     def __init__(
#         self,
#         data_dir: str = "./data",
#         host: str = "127.0.0.1",
#         port: int = 8666,
#         cache_dir: str = "./cache"
#     ):
#         self.data_dir = Path(data_dir)
#         self.host = host
#         self.port = port
#         self.cache_dir = Path(cache_dir)
#         self.server = None
#         self.client = None
        
#     async def initialize(self, embeddings: Optional[GoogleGenerativeAIEmbeddings] = None) -> None:
#         """Initialize the vector store server and client"""
#         try:
#             # Create directories
#             self.data_dir.mkdir(parents=True, exist_ok=True)
#             self.cache_dir.mkdir(parents=True, exist_ok=True)
            
#             # Setup data stream from filesystem
#             data = pw.io.fs.read(
#                 str(self.data_dir),
#                 format="binary",
#                 mode="streaming",
#                 with_metadata=True
#             )
            
#             # Initialize embeddings with API key
#             embeddings = embeddings or GoogleGenerativeAIEmbeddings(
#                 model=settings.EMBEDDING_MODEL,
#                 google_api_key=settings.GOOGLE_API_KEY,
#                 task_type="retrieval_document"
#             )
            
#             # Test embeddings
#             test_result = await embeddings.aembed_query("test")
#             if not test_result:
#                 raise ValueError("Embeddings test failed")
            
#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )
            
#             # Create and start server
#             self.server = VectorStoreServer.from_langchain_components(
#                 data,
#                 embedder=embeddings,
#                 splitter=splitter
#             )
            
#             self.server.run_server(
#                 self.host,
#                 port=self.port,
#                 with_cache=True,
#                 cache_backend=pw.persistence.Backend.filesystem(str(self.cache_dir)),
#                 threaded=True
#             )
            
#             # Initialize client
#             self.client = PathwayVectorClient(
#                 host=self.host,
#                 port=self.port
#             )
            
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize vector store: {str(e)}")
    
#     async def similarity_search(
#         self,
#         query: str,
#         k: int = 4,
#         metadata_filter: Optional[str] = None
#     ) -> List[Dict[str, Any]]:
#         """Perform similarity search"""
#         if not self.client:
#             raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
#         try:
#             results = await self.client.asimilarity_search(
#                 query,
#                 k=k,
#                 metadata_filter=metadata_filter
#             )
#             return results if results else []
            
#         except Exception as e:
#             raise RuntimeError(f"Similarity search failed: {str(e)}")
    
#     def get_statistics(self) -> Dict[str, Any]:
#         """Get vector store statistics"""
#         if not self.client:
#             raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
#         try:
#             return self.client.get_vectorstore_statistics()
#         except Exception as e:
#             raise RuntimeError(f"Failed to get statistics: {str(e)}")
    
#     def get_files(self) -> List[Dict[str, Any]]:
#         """Get list of indexed files"""
#         if not self.client:
#             raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
#         try:
#             return self.client.get_input_files()
#         except Exception as e:
#             raise RuntimeError(f"Failed to get files: {str(e)}") 