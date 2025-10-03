from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "face_verification_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "politicians")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Cache settings
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 60))  # Cache duration in seconds