try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import os

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "face_verification_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "politicians")

# Cache settings
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 60))  # Cache duration in seconds