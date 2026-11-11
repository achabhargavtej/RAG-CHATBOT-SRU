
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url or not api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from .env")

try:
    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30
    )

    print("Connecting to Qdrant Cloud...")

    collections = client.get_collections()

    print("Connection successful!")
    print("Existing collections:")

    for collection in collections.collections:
        print("-", collection.name)

except Exception as e:
    print("Connection failed:", str(e))
