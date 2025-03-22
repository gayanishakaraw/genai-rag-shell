from langchain.vectorstores import FAISS

class SearchHelper:
    
    def __init__(self):
        self.index_name = "genai-rag-shell-index"
        self.vector_store = None
        
    def search(self, query, embedding_model):
        self.vector_store = self.load_vector_store(embedding_model)
        retrieved_chunks = self.vector_store.similarity_search(query, k=2)
        return retrieved_chunks
    
    def save_vector_store(self, chunked_text, embedding):
        self.vector_store = FAISS.from_texts(chunked_text, embedding)
        self.vector_store.save_local(self.index_name)
        
    def load_vector_store(self, embedding_model):
        return FAISS.load_local(self.index_name, embedding_model, allow_dangerous_deserialization=True)
