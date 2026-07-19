import os 
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from src.vectorStore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

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

    def search_and_summarize(self, query:str, top_k:int = 5) -> str:
        results = self.vector_store.query(query, top_k)
        # Extract relevant texts from the top results
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant context found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    
    
    def search_and_answer(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k)
        # Extract relevant texts from the top results
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant context found."

        # Construct a clear, professional prompt
        prompt = f"""
        You are an AI assistant.
        Use the following context to answer the question accurately.
        If the answer is not in the context, say "Information not found in the documents".

        Question: {query}

        Context:
        {context}

        Answer:
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def chat(self, chat_history: list[any], max_tokens: int = 200):
        """Generate a response using the chat history."""
        # The ChatGroq model expects a list of messages (chat history)
        # We can directly pass the chat history if it is in the correct format
        # Or we can format it if needed. Assuming it's a list of message dicts
        try:
            response = self.llm.invoke(chat_history)
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat_with_context(self, query: str, context: str, chat_history: list[any], max_tokens: int = 200):
        """Generate a response using the query, context, and chat history."""
        # Construct the prompt with context and chat history
        prompt = f"""
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        User: {query}
        AI: """
        
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def interactive_chat(self, chat_history: list[any],): 
        print("\nChat Mode Activated. Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            query = input("You: ")
            
            if query.lower() == 'quit':
                print("\nExiting chat mode...")
                break
            
            # Search for relevant context
            results = self.vector_store.query(query, top_k=5)
            texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
            context = "\n\n".join(texts)
            
            if not context:
                print("AI: I don't have information about that.")
                continue
            
            # Generate response using chat_with_context
            response = self.chat_with_context(query, context, chat_history)
            print(f"AI: {response}\n")
            
            # Update chat history
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