import os 
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from src.vectorStore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

# L2 distance threshold — results with distance above this are considered not relevant
SCORE_THRESHOLD = 1.5

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2",  llm_model: str = "llama-3.3-70b-versatile"):
        self.vector_store = FaissVectorStore(persist_dir, embedding_model )
        #load or build vector store
        faiss_path = os.path.join(persist_dir, "faiss_index.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[WARN] Vector store not found. Building from scratch...")
            # pyrefly: ignore [missing-import]
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            print("[INFO] Found existing vector store. Loading...")
            self.vector_store.load()
        self.llm = ChatGroq(model_name=llm_model, api_key=os.getenv("GROQ_API_KEY"))
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def _rewrite_query(self, query: str, chat_history: list) -> str:
        """
        Rewrite a follow-up question into a self-contained standalone query
        using recent chat history so FAISS retrieval gets the right context.
        Only rewrites if there is history; otherwise returns query as-is.
        """
        if not chat_history:
            return query
        # Use last 4 messages (2 turns) to keep the prompt short
        recent = chat_history[-4:]
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        prompt = (
            "Given the conversation history below, rewrite the last user question "
            "as a fully self-contained standalone question. "
            "Output only the rewritten question, nothing else.\n\n"
            f"History:\n{history_str}\n\n"
            f"Last question: {query}\n\n"
            "Standalone question:"
        )
        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            print(f"[INFO] Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception:
            # Fall back to original query if rewrite fails
            return query

    def _filter_results(self, results: list) -> list:
        """Filter out FAISS results that exceed the distance threshold (not relevant)."""
        filtered = [r for r in results if r["distance"] < SCORE_THRESHOLD and r["metadata"]]
        return filtered

    def _format_sources(self, results: list) -> str:
        """Build a deduplicated source citation string from result metadata."""
        sources = list({r["metadata"].get("source", "unknown") for r in results if r["metadata"]})
        return ", ".join(sources) if sources else "unknown"

    def _multi_query_retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Multi-query retrieval: generate 3 query variants using the LLM,
        retrieve results for each, then merge and deduplicate by FAISS index.
        This improves recall for ambiguous or broad questions.
        """
        # Generate query variants
        prompt = (
            "Generate 3 different ways to ask the following question. "
            "Output only the 3 questions, one per line, no numbering or extra text.\n\n"
            f"Original question: {query}"
        )
        try:
            response = self.llm.invoke(prompt)
            variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        except Exception:
            variants = []
        # Always include the original query
        all_queries = [query] + variants[:3]
        seen_indices = set()
        merged_results = []
        for q in all_queries:
            results = self.vector_store.query(q, top_k)
            for r in results:
                if r["index"] not in seen_indices:
                    seen_indices.add(r["index"])
                    merged_results.append(r)
        # Sort merged results by distance (best first)
        merged_results.sort(key=lambda r: r["distance"])
        print(f"[INFO] Multi-query retrieved {len(merged_results)} unique chunks across {len(all_queries)} queries.")
        return merged_results

    def _summarize_history(self, chat_history: list) -> list:
        """
        Compress chat history when it exceeds 10 messages.
        Summarizes the oldest messages into a single assistant message to prevent
        context overflow in long conversations.
        """
        if len(chat_history) <= 10:
            return chat_history
        # Keep the last 6 messages intact; summarize everything before
        old_turns = chat_history[:-6]
        recent_turns = chat_history[-6:]
        old_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in old_turns
        )
        prompt = (
            "Summarize the following conversation history into a single concise paragraph "
            "that captures the key topics discussed. Output only the summary.\n\n"
            f"{old_str}"
        )
        try:
            response = self.llm.invoke(prompt)
            summary_text = response.content.strip()
            print("[INFO] Chat history summarized to prevent context overflow.")
        except Exception:
            summary_text = "[Earlier conversation summarized]"
        compressed = [{"role": "assistant", "content": f"[Summary of earlier conversation]: {summary_text}"}]
        return compressed + recent_turns

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k)
        results = self._filter_results(results)
        if not results:
            return "No relevant context found in the documents."
        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts)
        sources = self._format_sources(results)
        prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
        response = self.llm.invoke(prompt)
        return f"{response.content.strip()}\n\n📄 Sources: {sources}"

    def search_and_answer(self, query: str, top_k: int = 5) -> str:
        # Multi-query retrieval for better recall, then apply score threshold filter
        results = self._multi_query_retrieve(query, top_k)
        results = self._filter_results(results)
        if not results:
            return "I don't have relevant information about that in my documents."
        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts)
        sources = self._format_sources(results)
        prompt = f"""You are an AI assistant.
Use the following context to answer the question accurately.
If the answer is not in the context, say "Information not found in the documents".

Question: {query}

Context:
{context}

Answer:"""
        response = self.llm.invoke(prompt)
        return f"{response.content.strip()}\n\n📄 Sources: {sources}"

    def chat(self, chat_history: list) -> str:
        """Send raw chat history to the LLM. Use for general conversation without RAG retrieval."""
        try:
            response = self.llm.invoke(chat_history)
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat_with_context(self, query: str, context: str, chat_history: list) -> str:
        """Generate a response using the query, retrieved context, and conversation history."""
        # Format chat history as readable turns for the LLM
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]
        )
        prompt = f"""You are a helpful AI assistant. Use the context below to answer the user's question.
If the answer is not in the context, say "I don't have that information in my documents".

Context:
{context}

Conversation so far:
{history_str}

User: {query}
Assistant:"""
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def interactive_chat(self, chat_history: list):
        print("\nChat Mode Activated. Type 'quit' to exit.\n")
        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() == 'quit':
                print("\nExiting chat mode...")
                break

            # Compress history if it has grown too long
            chat_history = self._summarize_history(chat_history)

            # Rewrite follow-up questions into self-contained queries for better retrieval
            standalone_query = self._rewrite_query(query, chat_history)

            # Multi-query retrieval using the rewritten query for better recall
            results = self._multi_query_retrieve(standalone_query, top_k=5)
            results = self._filter_results(results)

            if not results:
                print("AI: I don't have information about that in my documents.\n")
                # Still update history so the LLM knows the conversation flow
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": "I don't have information about that in my documents."})
                continue

            texts = [r["metadata"].get("text", "") for r in results]
            context = "\n\n".join(texts)
            sources = self._format_sources(results)

            response = self.chat_with_context(query, context, chat_history)
            print(f"AI: {response}\n📄 Sources: {sources}\n")

            # Update history after generating the response
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response})

  
    # Example usage (you can add this to the bottom of src/search.py)
if __name__ == "__main__":
    # Test the RAGSearch class
    rag_search = RAGSearch()
    print("\n" + "="*60)
    print("TESTING RAGSEARCH SYSTEM")
    print("="*60 + "\n")

    # Test question
    test_query = "What is  Artificial Intelligence, When It was introduced?"
    print(f"Question: {test_query}")
    print("\n" + "-"*60)
    print("Search & Summarize Mode:")
    summary = rag_search.search_and_summarize(test_query)
    print(summary)

    print("\n" + "-"*60)
    print("Search & Answer Mode:")
    answer = rag_search.search_and_answer(test_query)
    print(answer)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)