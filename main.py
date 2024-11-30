import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings

PATHWAY_PORT = 8765

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./documents",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, separator=" ")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    splitter=text_splitter,
    embedder=embeddings_model,
)

if __name__ == "__main__":
    print("Starting VectorStoreServer...")
    vector_server.run_server(
        host="127.0.0.1",
        port=PATHWAY_PORT,
        threaded=True,
        with_cache=False,
    )
    time.sleep(30)

    print("\nmaking client...\n")
    client = VectorStoreClient(
        host="127.0.0.1",
        port=PATHWAY_PORT,
    )
    query = "What is lorem?"
    docs = client(query, k=1)
    print(docs)