import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Paths - Updated for new directory structure
    BASE_DIR = Path(__file__).parent.parent.parent  # knowledge-base-system root
    INBOX_DIR = BASE_DIR / "data" / "inbox"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "data" / "logs"
    
    # Pinecone settings
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "personal-knowledge-base")
    
    # ChromaDB settings
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "vector_db" / "chroma_db"))
    
    # Embeddings settings
    EMBEDDINGS_CACHE_PATH = os.getenv("EMBEDDINGS_CACHE_PATH", str(DATA_DIR / "cache" / "embeddings"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Processing settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # Storage mode: local, pinecone, or hybrid
    STORAGE_MODE = os.getenv("STORAGE_MODE", "hybrid")
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.txt', '.md', '.markdown', '.docx', '.doc',
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp',
        '.c', '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.swift',
        '.html', '.htm', '.xml', '.json', '.yaml', '.yml'
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.INBOX_DIR, cls.PROCESSED_DIR, cls.DATA_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
