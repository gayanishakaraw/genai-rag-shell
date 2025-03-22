from langchain.embeddings.openai import OpenAIEmbeddings


class EmbeddingHelper:
    # Initialize OpenAI Embeddings
    def __init__(self, api_key, model_name="text-embedding-ada-002"):
        self.embedding_model = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

    # Convert text chunks into embeddings
    def embed_documents(self, chunks):
        return self.embedding_model.embed_documents(chunks)
    
    def get_embedding_model(self):
        return self.embedding_model