# Simple Semantic Search Engine

---

## Overview

The system supports versatile ingestion capabilities, allowing you to seamlessly load local PDF or TXT documents via the PyMuPDF4LLMLoader as well as fetch and parse web pages using the WebBaseLoader. Once ingested, text is split into overlapping chunks with the RecursiveCharacterTextSplitter to preserve context across boundaries. For representation, mean‑pooled Transformer embeddings are computed entirely on CPU for efficiency. These vectors are then indexed in a FAISS HNSW store with memory‑mapped persistence, and you can optionally switch to a Qdrant backend for advanced disk‑backed storage. The user interface is powered by Streamlit, featuring built‑in caching, query‑term highlighting in result snippets, and download links for source files. Finally, the entire application is packaged in a multi‑stage Docker build, minimizing image size and including healthchecks for robust deployment.

---

## Project Layout

```
Simple-Semantic-Search-Engine/
├── data/
├── semantic_search_engine/
│   ├── __init__.py
│   ├── base_loader.py
│   ├── embedder.py
│   ├── engine.py
│   ├── file_loader.py
│   ├── logger.py
│   ├── settings.py
│   ├── splitter.py
│   ├── url_loader.py
│   └── vectorstore.py
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── README.md
└── requirements.txt

```

---


## Quickstart

### 1. Clone & Install Locally

```bash
git clone https://github.com/yourusername/semantic-search-engine.git
cd semantic-search-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `.pdf` or `.txt` files in the `data/` folder at the project root.

### 3. Run with Streamlit

```bash
streamlit run web_app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker

Build and run with Docker:

```bash
docker build -t semantic-search .
docker run -p 8501:8501 -p 8000:8000 semantic-search
```

Port **8501** hosts the Streamlit UI

Port **8000** exposes Prometheus metrics (optional)

---

## Usage

Ingest by uploading or dropping files via the sidebar or by entering a URL and clicking Start Ingestion. 

Search by entering a query to receive one highlighted snippet per document, each accompanied by a download link.
