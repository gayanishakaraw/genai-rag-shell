from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text):
        """Splits text into chunks with overlap for better context preservation."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Prefer splitting at paragraphs, sentences, spaces
        )
        return text_splitter.split_text(text)

