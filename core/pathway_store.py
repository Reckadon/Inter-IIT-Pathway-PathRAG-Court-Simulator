import os
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata/"

import pathway as pw
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pathway import PathwayVectorClient
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.parsers import OpenParse
import time
from langchain_huggingface import HuggingFaceEmbeddings

# @pw.udf
# def strip_metadata(docs: list[tuple[str, dict]]) -> list[str]:
#     return [doc[0] for doc in docs]

parser = OpenParse(table_args=None, image_args=None, parse_images = False)

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "law-ai/InLegalBERT"
)

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
            self.data_sources = pw.io.fs.read(
                path,
                format="binary",
                mode="streaming",
                with_metadata=True,
            )

            # Apply the parser to the PDF data
            # self.documents = self.data_sources.select(data=parser(pw.this.data))

            # no splitter needed as OpenParse handles it
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)   

            print(f"\nmaking VectorStore: '{self.name}'... with docs {self.data_sources}\n")
            self.vector_server = VectorStoreServer.from_langchain_components(
                self.data_sources,
                parser= parser,
                splitter=None,
                embedder=embeddings_model,
            )

            # print(f"Starting VectorStoreServer: '{self.name}'...")
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
            print("\n made client..")


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

    def get_client(self):
        """
        get the client for the vector store.

        Returns:
        PathwayVectorClient
        """
        return self.client
    



if __name__ == "__main__":    # example usage
    public_db = PathwayVectorStore('xyztest', './documents', 8765)
    print('making a query')
    result = public_db.get_client().as_retriever().invoke("using lorem")
    for entry in result:
        print(entry, "\n")
