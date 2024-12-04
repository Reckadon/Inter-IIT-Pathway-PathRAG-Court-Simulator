# Inter-IIT-High-Prep-Pathway

---

### Running the project: 🚀

To run the project, you need to have **Docker** installed in your environment.
First, run:

```terminal
docker build -t pathwaytest .
```

`pathwaytest` can be replaced with any name you want to give to the image.
Then, run the container with the following command:

```terminal
docker run -it --rm --env-file .env pathwaytest
```

_Use the same name as above_

---


## References
Indian Penal Code PDF: [link](https://www.iitk.ac.in/wc/data/IPC_186045.pdf)
PDF Parsing reference: [Multimodal RAG for PDFs with Text, Images, and Charts](https://pathway.com/developers/templates/multimodal-rag)
Vector store reference: [Data Indexing](https://pathway.com/developers/user-guide/llm-xpack/vectorstore_pipeline/)
