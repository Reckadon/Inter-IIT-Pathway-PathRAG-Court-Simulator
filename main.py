import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer
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

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    embedder=embeddings_model,
    splitter=text_splitter,
)

if __name__ == "__main__":
    print("Starting VectorStoreServer...")
    vector_server.run_server(
        host="0.0.0.0",
        port=PATHWAY_PORT + 1,
        threaded=True,
        with_cache=False,
    )