import pathway as pw
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pathway import PathwayVectorClient
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings


class PathwayVectorStore:
    def __init__(self, name, path, port):
        """
        Initialize the Store with the docs from given path.
        Parameters: 
        name: name to give the database - eg. public
        path: path to the directory containing the files to feed into db - eg. /data
        port: port to use for the vector store - eg. 8765

        """
        self.name = name
        self.path = path
        self.port = port
        self.vector_server = None
        self.client = None

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

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)

            embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            print(f"making VectorStore: '{self.name}'...")
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
            
            print("\nmaking client using langchain's pathwayvectorclient...\n")
            self.client = PathwayVectorClient(
                host="127.0.0.1",
                port=port,
            )                           


        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    # def add_item(self, item_name, quantity, price):
    #     """
    #     Add an item to the Vector store.

    #     Parameters:
    #     item_name (str): The name of the item.
    #     quantity (int): The quantity of the item.
    #     price (float): The price of the item.
    #     """
    #     if item_name in self.inventory:
    #         self.inventory[item_name]['quantity'] += quantity
    #     else:
    #         self.inventory[item_name] = {'quantity': quantity, 'price': price}

    def query_store(self, query_text):
        """
        Query the vec store using the text provided.

        Parameters:
        query_text (str): The query.

        Returns:
        
        """
        docs = self.client.similarity_search(query_text, k=2)
        return docs
    



if __name__ == "__main__":
    public_db = PathwayVectorStore('public', './documents', 8765)
    print(public_db.query_store("what is lorem"))
