# DocuMind AI

### An AI-Powered Research Paper Assistant Using Retrieval-Augmented Generation (RAG)

## About the Project

Reading and understanding research papers can take a lot of time, especially when you need to find specific information across multiple pages.

**DocuMind AI** is a research paper assistant built to make this process easier. It allows users to upload a research paper in PDF format, ask questions about its content, and generate structured summaries.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document and provide context-based answers using a Large Language Model (LLM).

It combines semantic search, keyword-based retrieval and hybrid retrieval to improve the process of finding relevant information from research papers.

## Features

- **PDF Processing:** Extracts text from research papers while keeping track of page numbers.
- **Text Chunking:** Splits extracted text into smaller, overlapping chunks for better retrieval.
- **Semantic Search:** Uses BGE embeddings to understand the meaning of queries and document content.
- **Vector Search:** Uses FAISS to find semantically similar document chunks.
- **Keyword Search:** Uses BM25 to retrieve passages based on keyword relevance.
- **Hybrid Retrieval:** Combines semantic and keyword search using Reciprocal Rank Fusion (RRF).
- **Question Answering:** Uses Qwen 2.5 to generate answers based on retrieved document content.
- **Source Page References:** Displays the page numbers of retrieved passages alongside answers.
- **Document Summarization:** Generates structured summaries using the Map-Reduce approach.
- **Streamlit Interface:** Provides a simple interface for uploading papers, asking questions and generating summaries.

## Project Architecture

The overall workflow of DocuMind AI is:

```text
Research Paper PDF
        |
        v
PDF Text Extraction
        |
        v
Text Chunking
        |
        v
BGE Embeddings
        |
   +----+----+
   |         |
   v         v
 FAISS     BM25
   |         |
   +----+----+
        |
        v
Hybrid Retrieval (RRF)
        |
        v
Retrieved Context
        |
        v
Qwen 2.5 LLM
        |
        v
Context-Aware Answer
        |
        v
Retrieved Source Pages
```

## Technology Stack

The following tools and technologies were used to build the project:

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| PDF Processing | PyMuPDF |
| Tokenization | BERT Tokenizer |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Search | FAISS |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) |
| Large Language Model | Qwen 2.5 1.5B Instruct |
| NLP Framework | Hugging Face Transformers |
| User Interface | Streamlit |

## Models Used

### 1. Embedding Model – BGE

`BAAI/bge-base-en-v1.5`

This model converts document chunks and user queries into numerical vector representations. These embeddings are used to find passages that are semantically relevant to a user's question.

### 2. Language Model – Qwen 2.5

`Qwen/Qwen2.5-1.5B-Instruct`

Qwen is used to generate answers based on the context retrieved from the uploaded research paper. It is also used in the document summarization pipeline.

## Retrieval Configuration

The following configuration was used during document processing and retrieval:

| Parameter | Value |
|---|---:|
| Chunk Size | 500 tokens |
| Chunk Overlap | 100 tokens |
| Embedding Dimensions | 768 |
| Semantic Retrieval Candidates | 10 |
| BM25 Retrieval Candidates | 10 |
| Final Retrieved Chunks | 6 |
| RRF Constant (k) | 60 |

For semantic retrieval, FAISS uses normalized embeddings with inner-product similarity. BM25 handles keyword-based retrieval, and RRF combines the rankings from both methods to produce the final retrieved results.

## Project Structure

```text
DocuMind-AI/
│
├── DocuMind_AI_Development.ipynb
├── DocuMind_Streamlit.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `DocuMind_AI_Development.ipynb` – Contains the development process, implementation and experiments.
- `DocuMind_Streamlit.py` – Contains the Streamlit application and its backend functionality.
- `requirements.txt` – Lists the Python dependencies required to run the project.
- `README.md` – Project documentation.
- `.gitignore` – Specifies files and folders that should not be uploaded to GitHub.

## Installation and Setup

Follow these steps to run DocuMind AI on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
```

### 2. Navigate to the Project Folder

```bash
cd DocuMind-AI
```

### 3. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run DocuMind_Streamlit.py
```

Once the application starts, open the local URL displayed in your terminal to access the interface.

## How to Use

1. Open the DocuMind AI application.
2. Upload a research paper in PDF format.
3. Wait for the document to be extracted, divided into chunks and indexed.
4. Enter a question related to the uploaded paper.
5. View the generated answer along with the retrieved source page references.
6. Use the summarization feature to generate a structured summary of the document.

## Development and Experiments

The development notebook contains the step-by-step implementation of the project.

It covers the following areas:

- Extracting and preprocessing text from PDF documents.
- Creating page-aware text chunks.
- Generating embeddings using BGE.
- Building a FAISS vector index.
- Implementing keyword retrieval using BM25.
- Combining retrieval methods using Reciprocal Rank Fusion.
- Generating answers using Qwen 2.5.
- Implementing Map-Reduce document summarization.
- Comparing embedding models and retrieval approaches.
- Evaluating retrieval performance.

These experiments helped in selecting the embedding model and retrieval approach used in the final application.

## Limitations

There are a few limitations in the current version:

- The application processes one research paper at a time.
- Generated answers may sometimes contain inaccurate or unsupported information.
- Retrieved page references identify relevant passages but are not independently verified citations for individual claims.
- Map-Reduce summarization may introduce factual inconsistencies when combining summaries from different chunks.
- Model performance depends on document quality and available computational resources.

## Author

**Yashwanth Kumar Chapalamadugu**

B.Tech – Artificial Intelligence and Machine Learning

---

*Built as a hands-on project to explore Retrieval-Augmented Generation, hybrid information retrieval and Large Language Models.*