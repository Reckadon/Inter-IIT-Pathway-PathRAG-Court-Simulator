import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings

PATHWAY_PORT = 8765


class VecStore:
    def __init__(self, name, path, port):
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