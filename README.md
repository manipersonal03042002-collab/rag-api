# Enterprise RAG System

This project implements an advanced, enterprise-grade Retrieval-Augmented Generation (RAG) pipeline. It moves beyond standard introductory tutorials by integrating robust data ingestion, multi-turn conversational memory, query optimization, and relevance filtering to drastically reduce LLM hallucinations and provide accurate, cited answers.

---

## 🏗️ End-to-End Architecture

The system operates in five main stages:

### 1. Data Ingestion & Parsing (`src/data_loader.py`)
- **Multi-Format Support**: Automatically recursively scans the `data/` folder and loads PDF, TXT, CSV, Excel, Word, and JSON files.
- **Robustness**: Includes fallback mechanisms (e.g., attempting UTF-8 encoding for text files and falling back to Latin-1) and utilizes strict JSON schema parsing to prevent crashes on complex data.
- **Source Tracking**: Attaches the original filename as metadata to every document parsed, which is essential for accurate citations later.

### 2. Chunking & Embedding (`src/embedding.py`)
- **Semantic Chunking**: Uses a `RecursiveCharacterTextSplitter` to break large documents into readable chunks (1000 characters) with a 200-character overlap to preserve paragraph context.
- **Local Embeddings**: Uses the lightweight, fast `SentenceTransformer` model (`all-MiniLM-L6-v2`) to convert text chunks into high-dimensional numerical vectors completely locally, saving API costs and increasing privacy.

### 3. Vector Storage (`src/vectorStore.py`)
- **FAISS Engine**: Uses Facebook AI Similarity Search (FAISS) for lightning-fast L2 distance vector matching.
- **Incremental Updates**: Features an `add_documents()` method allowing new files to be indexed without requiring a full pipeline rebuild.
- **Persistence**: Automatically saves the FAISS index and a metadata pickle file to disk (`faiss_store/`) for instantaneous loading on subsequent runs.
- **Health Monitoring**: Includes a `stats()` method to monitor total active vectors and unique document sources.

### 4. Advanced Retrieval (`src/search.py`)
This is where the true enterprise logic occurs. The retrieval phase uses several advanced techniques before ever passing context to the LLM:
- **Conversation-Aware Query Rewriting**: In an interactive chat, users often ask follow-up questions using pronouns (e.g., "What does *it* do?"). The system uses the recent chat history to ask the LLM to rewrite this into a standalone query (e.g., "What does Artificial Intelligence do?") so the vector search finds the right data.
- **Multi-Query Retrieval**: A single query might miss relevant documents. The system asks the LLM to generate 3 semantic variations of the user's query, executes FAISS searches for all of them, and then merges and deduplicates the results to drastically improve recall.
- **Relevance Score Filtering**: FAISS returns distance metrics. The system implements a strict `SCORE_THRESHOLD` (1.5). Any chunk that exceeds this distance is discarded, ensuring the LLM is not fed irrelevant "noise" that causes hallucinations.

### 5. LLM Generation & Memory Management (`src/search.py`)
- **Context Injection & Citation**: The filtered chunks are injected into a strict prompt instructing the Groq LLM (`llama-3.3-70b-versatile`) to answer *only* using the provided context. It appends the deduplicated document names at the bottom of the response as `📄 Sources:`.
- **Dynamic History Compression**: To prevent hitting LLM token limits and to save API costs during long chats, the system implements `_summarize_history()`. When a chat exceeds 10 turns, the oldest messages are compressed into a single concise summary block, while the most recent 6 messages are kept intact for immediate conversational flow.

---

## 📁 Project Structure

```text
RAGbot/
├── app.py                  # Main entry point for One-Shot Q&A and Interactive Chat
├── README.md               # This documentation file
├── .env                    # Environment variables (GROQ_API_KEY)
├── data/                   # Directory to drop PDF, TXT, CSV, JSON, Word files
├── faiss_store/            # Auto-generated persistent vector database
└── src/
    ├── data_loader.py      # Robust multi-format document ingestion
    ├── embedding.py        # Chunking logic and SentenceTransformer embedding
    ├── vectorStore.py      # FAISS indexing, metadata management, saving/loading
    └── search.py           # Orchestration, multi-query, filtering, LLM prompts
```

---

## 🚀 How to Run

1. **Prerequisites**: Ensure you have a `.env` file in the root directory containing your API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
2. **Add Data**: Drop any supported files into the `data/` folder or its subdirectories.
3. **Execute**:
   ```bash
   python app.py
   ```
   *On the very first run, the system will detect that no index exists, parse all files in `data/`, generate embeddings, and save the vector store. On subsequent runs, it will load instantly.*
4. **Interact**: The terminal will launch into a Chat Mode where you can ask questions, ask follow-ups, and receive fully cited answers based exclusively on your local data. Type `quit` to exit.
