import streamlit as st
from openai import OpenAI
from document_retriever import DocumentRetriever
import logging
import os
log_path = "logfile.log"
from text_chunker import TextChunker
from embedding_helper import EmbeddingHelper
from search_helper import SearchHelper

if not os.path.exists(log_path):
    with open(log_path, 'w') as file:
        file.write("")

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Initialize OpenAI API
openai_api_key = st.secrets["openai_api_key"]
openai = OpenAI(api_key=openai_api_key)

# Initialize Document Retriever
document_retriever = DocumentRetriever(directory="documents")

# Initialize Text Chunker and Embedding Helper
text_chunker = TextChunker()
# Initialize Embedding Helper
embedding_helper = EmbeddingHelper(api_key = openai_api_key)
# Initialize Search Helper
search_helper = SearchHelper()
relevant_docs = document_retriever.retrieve()
        
chunked_text = text_chunker.chunk_text(relevant_docs)
LOGGER.info(f"### Chunked Text .....")
# LOGGER.info(chunked_text)
LOGGER.info(f"### \n")

chunk_embeddings = embedding_helper.embed_documents(chunked_text)
LOGGER.info(f"### Embedded Documents .....")

embedding_model = embedding_helper.get_embedding_model()
search_helper.save_vector_store(chunked_text, embedding_model)

LOGGER.info(f"### Application is starting up.....")

# Streamlit UI
st.title("GenAI Chatbot with RAG")
st.write("Ask questions based on local documents.")

# User input
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question:
        
        LOGGER.info(f"### user_question.....")
        LOGGER.info(user_question)
        LOGGER.info(f"### \n")
        
        # Retrieve relevant documents
        retrieved_chunks = search_helper.search(user_question, embedding_model)
        LOGGER.info(f"### Search Results .....")
        LOGGER.info(retrieved_chunks)
        
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        # Generate response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer the question based on the following documents context:\n\n{context}\n\nQuestion: {user_question}\nAnswer:"}
            ]
        )
        
        LOGGER.info(f"### Response .....")
        LOGGER.info(response.choices[0].message.content)

        # Display response
        st.write(response.choices[0].message.content.strip())
    else:
        st.write("Please enter a question.")
