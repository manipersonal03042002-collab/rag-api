# pyrefly: ignore [missing-import]
from src.search import RAGSearch

if __name__ == "__main__":
    # RAGSearch auto-loads or builds the vector store on init
    rag_search = RAGSearch()

    # One-shot Q&A
    query = "What is Artificial Intelligence?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    answer = rag_search.search_and_answer(query, top_k=3)
    print("Answer:", answer)

    # Interactive multi-turn chat with memory
    chat_history = []
    rag_search.interactive_chat(chat_history)
