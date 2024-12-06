# PathRAG Court

---

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

### Running the project: 🚀

- Since Pathway is meant to be deployed in a containerized manner, and single-machine deployments can easily be achieved using Docker - we opted to make the whole app be able to run in a docker container. The dependencies are automatically installed into the container using the `requirements.txt` file.
- Therefore to run the project, you need to have **Docker** installed in your environment.
- First, go into the project directory, and setup a `.env` file (have to make the file) containing all the **API Keys** required to run the LLMs and fetching tools.:

```env
GOOGLE_API_KEY= <your_key_here>
GROQ_API_KEY= <your_key_here>
HUGGINGFACE_API_KEY= <your_key_here>
SERPER_API_KEY= <your_key_here>
KANOON_API_KEY= <your_key_here>
// Other API keys for any LLM you would like to use
```

- Then, run the following command:

```terminal
docker build -t pathwaytest .
```

`pathwaytest` can be replaced with any name you want to give to the image.
Then, run the container with the following command:

```terminal
docker run -it -p 8000:8000 --rm --env-file .env pathwaytest
```

_Use the same name as above_, this will expose the backend FastAPI server at port 8000.

---

### Architecture Diagram: 🏛️

![Architecture diagram](./architecture.png)

### Problem Solved: 🎯

//TODO

## References

1. Indian Penal Code PDF: [link](https://www.iitk.ac.in/wc/data/IPC_186045.pdf)
1. PDF Parsing reference: [Multimodal RAG for PDFs with Text, Images, and Charts](https://pathway.com/developers/templates/multimodal-rag)
1. Vector store reference: [Data Indexing](https://pathway.com/developers/user-guide/llm-xpack/vectorstore_pipeline/)
