import pathway as pw
from langchain.text_splitter import CharacterTextSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer
import os
import time
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

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

huggingface_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings_model = HuggingFaceEmbeddings(model=huggingface_model)

vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    embedder=embeddings_model,
    splitter=text_splitter,
)
vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT+1, threaded=True, with_cache=False)
time.sleep(30)
