


# Example usage
# pyrefly: ignore [missing-import]
from src.data_loader import load_all_documents
# pyrefly: ignore [missing-import]
from src.embedding import EmbeddingPipeline
# pyrefly: ignore [missing-import]
from src.vectorStore import FaissVectorStore
# pyrefly: ignore [missing-import]
from src.search import RAGSearch

if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    # RAG search usage
    query = "What is cricket?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    answer = rag_search.search_and_answer(query, top_k=3)
    print("Answer:", answer)
    
    # Interactive chat usage
    chat_history = []
    rag_search.interactive_chat(chat_history)

    



