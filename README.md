# Hybrid RAG Architecture for BBC News

This project builds a retrieval-augmented generation (RAG) pipeline over a BBC news dataset using Weaviate for vector storage, metadata filtering, hybrid search, and reranking, with Ollama-based embedding and generation models.

The project demonstrates a complete workflow:

- load the BBC dataset
- split documents into chunks
- index them in Weaviate
- rewrite user queries for more effective retrieval
- run hybrid search with reranking
- generate answers with an LLM grounded in retrieved context

---

## Project Structure

```text
rag-hybrid-architecture/
├── docker-compose.yml
├── README.md
├── data/
│   └── bbc_data.joblib
└── src/
    ├── chunking.py
    └── pipeline.ipynb
```

### Key files

- [src/chunking.py](src/chunking.py): text chunking logic for creating retrieval units
- [src/pipeline.ipynb](src/pipeline.ipynb): end-to-end notebook for indexing, querying, and generation
- [docker-compose.yml](docker-compose.yml): Weaviate and reranker service setup
- [data/bbc_data.joblib](data/bbc_data.joblib): serialized BBC dataset used for indexing

---

## Architecture Overview

The system combines:

- semantic search using vector embeddings
- hybrid search blending keyword + vector retrieval
- query rewriting for better search intent
- reranking to improve result relevance
- LLM response generation with retrieved chunks as context

At a high level:

1. News articles are loaded from the BBC dataset.
2. Each article is divided into smaller text chunks.
3. Chunks are stored in a Weaviate collection with metadata like title, link, date, description, and chunk index.
4. A user query is rewritten to improve retrieval quality.
5. Hybrid search retrieves candidate chunks.
6. The reranker reorders the results by relevance.
7. The top relevant chunks are passed to an LLM, which answers using only those documents.

---

## Tech Stack

- Python
- Weaviate
- Ollama
- Docker Compose
- pandas
- joblib
- tqdm

### Models
- **Embedding**: `nomic-embed-text` (via `text2vec-ollama`)
- **Generation / Query Rewriting**: `qwen2.5` (via `generative-ollama`)
- **Reranking**: `cross-encoder-ms-marco-MiniLM-L-6-v2` (via `reranker-transformers`)
  
---

## Prerequisites

Before running the project, make sure you have:

- Docker Desktop or Docker Engine installed
- Python 3.10+ recommended
- Ollama installed and running locally
- Access to the Ollama-compatible API endpoint used by `CLOUDFLARE_URL`
- Internet access for pulling model artifacts if needed

---

## Setup

### Option A: Local machine setup

#### 1. Start dependencies

From the project root:

```bash
docker compose up -d
```

This starts:

- Weaviate on `http://localhost:8090`
- gRPC on port `50051`
- the reranker service used by the Weaviate collection

#### 2. Install Python packages

```bash
pip install weaviate-client ollama pandas joblib tqdm
```

#### 3. Pull Ollama models

If not already available locally, pull the models used by the notebook:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5
```

#### 4. Configure the external Ollama endpoint

In normal local usage, you may set a stable endpoint such as:

```python
CLOUDFLARE_URL = "https://your-custom-domain-or-tunnel-url"
```

When using a temporary Cloudflare tunnel in Colab, this value changes every time the tunnel is recreated. In that case, you must re-run the tunnel command and update `CLOUDFLARE_URL` to the newly generated URL before creating or querying the Weaviate collection.

### Option B: Google Colab setup

This project was originally run in Colab with Ollama exposed through a Cloudflare tunnel. The setup used was:

```python
!apt-get update -qq
!apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time, os


os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"
os.environ["OLLAMA_ORIGINS"] = "*"

process = subprocess.Popen("ollama serve", shell=True, env=os.environ)
time.sleep(4)

import requests
r = requests.get("http://localhost:11434")
print(r.status_code, r.text)
!ollama pull nomic-embed-text
!ollama pull qwen2.5
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
import re

tunnel_process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:11434"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

time.sleep(8)

public_url = None
for _ in range(30):
    line = tunnel_process.stdout.readline()
    if not line:
        break
    print(line, end="")
    match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        break

print(public_url)
import requests
requests.get(public_url)
```

This exposes Ollama through a temporary public URL, which is then assigned to `CLOUDFLARE_URL` in the notebook. Because the tunnel URL is generated fresh on each run, the value must be updated every time you restart the tunnel or re-open the Colab session.

---

## Data Pipeline

The dataset is loaded from:

```python
bbc_data = joblib.load(r'...\data\bbc_data.joblib')
```

Each object is expected to contain article metadata and text content, including fields such as:

- title
- description
- link
- guid
- pubDate
- article_content

The chunking process is implemented in [src/chunking.py](src/chunking.py) and uses a paragraph-based chunking strategy with minimum and maximum length constraints.

---
### Chunking Strategy

The chunking logic (`src/chunking.py`) uses a paragraph-based mixed strategy:

- Articles are split by paragraph (`\n\n`).
- Consecutive short paragraphs are merged together until they reach a
  minimum length (`min_length=25` words), avoiding tiny, low-value chunks.
- Paragraphs (or merged groups) that exceed a maximum length
  (`max_length=350` words) are split into fixed-size slices with no overlap.
- The result is a list of chunks per article, each tagged with a
  `chunk_index` and the article's original metadata (title, date, link, etc).

---

## Weaviate Collection Setup


The collection uses:

- `text2vec-ollama` for vectorization
- `generative-ollama` for answer generation
- `reranker-transformers` for relevance ranking


---

## Query Pipeline

The notebook includes the following stages:

### Query rewriting

A helper function calls Ollama to rewrite the user's query into a more search-friendly form:

```python
def query_rewriting(query):
    rewrite_prompt = f"""
    You are an expert search system optimizer.
    Rewrite the following user query to be more descriptive and comprehensive for a semantic search database.
    Include relevant financial synonyms and context.
    Return ONLY the rewritten query text, nothing else.

    Original query: {query}
    """
```

### Hybrid retrieval

The hybrid search combines vector similarity and keyword relevance:

```python
response = collection.query.hybrid(
    query=query,
    alpha=alpha,
    limit=num_retrieval,
    rerank=Rerank(prop="chunk", query=query),
    return_metadata=MetadataQuery(score=True, explain_score=True)
)
```

This retrieves relevant chunks, filters to positive rerank scores, and assembles context for the LLM.

### Full RAG flow

The notebook then builds a final prompt using the retrieved chunks and asks Ollama to answer:

```python
def full_pipeline(query, num_retrieval, num_retrieval_ranker, alpha):
    improved_query = query_rewriting(query)
    final_prompt = f"""You are a smart and professional assistant. Answer the user's question based on the attached documents only.
    If the answer is not in the documents, say "I do not have enough information".

    Question: {query}

    Documents:
    {hybrid_search(query=improved_query, num_retrieval=num_retrieval, num_retrieval_ranker=num_retrieval_ranker, alpha=alpha)}
    """
```

---

## Usage

Open [src/pipeline.ipynb](src/pipeline.ipynb) and run the cells in order.

Typical flow:

1. Connect to Weaviate
2. Load the BBC joblib dataset
3. Chunk the article content
4. Create the collection
5. Load chunks into the collection
6. Query with `full_pipeline(...)`
7. Inspect the generated result

If you are using Colab, run the tunnel setup block first so that `CLOUDFLARE_URL` resolves to a valid Cloudflare URL before creating the Weaviate collection.

Example:

```python
user_query = "Tell me the economic situation of the US in 2023."
res = full_pipeline(query=user_query, num_retrieval=10, num_retrieval_ranker=3, alpha=0.4)
print(res)
```

---

## Notes and Caveats

- The project assumes Ollama and Weaviate are running and reachable.
- The external `CLOUDFLARE_URL` is temporary and changes on each new Cloudflare tunnel session.
- Reassign it every time the tunnel is restarted or recreated.
- The prompt explicitly restricts the model to answer only from the retrieved context.
- If no valid context is found, the response should say that there is not enough information.
- The repository is intended as a research/demo pipeline rather than a production-ready application.

---


## Summary

This project is a practical example of a hybrid RAG system using Weaviate, Ollama, and reranking to answer questions over BBC news articles. It is designed to be easy to follow in a notebook and to illustrate the full indexing and retrieval pipeline from raw documents to generated answers.
