
import os
import uuid
import chromadb

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "college_documents"
VECTOR_SIZE = 384
BATCH_SIZE = 64

# Connect to local ChromaDB
print("Connecting to local ChromaDB...")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

total_chunks = chroma_collection.count()
print(f"Found {total_chunks} chunks in ChromaDB.")

# Connect to Qdrant Cloud
print("Connecting to Qdrant Cloud...")

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

# Create collection if it doesn't exist
existing = [
    c.name for c in qdrant_client.get_collections().collections
]

if COLLECTION_NAME not in existing:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    print("Created Qdrant collection:", COLLECTION_NAME)
else:
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print("Collection already exists.")

    vector_config = info.config.params.vectors
    if not isinstance(vector_config, models.VectorParams):
        raise ValueError("Unexpected Qdrant vector configuration.")

    if vector_config.size != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE} dimensions, "
            f"but collection has {vector_config.size}."
        )

# Migrate data in batches
migrated = 0

for offset in range(0, total_chunks, BATCH_SIZE):

    records = chroma_collection.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "metadatas", "embeddings"]
    )

    points = []

    for i, original_id in enumerate(records["ids"]):

        document = records["documents"][i]
        metadata = records["metadatas"][i] or {}
        embedding = records["embeddings"][i]

        if document is None or embedding is None:
            raise ValueError(
                f"Missing document or embedding for {original_id}"
            )

        if len(embedding) != VECTOR_SIZE:
            raise ValueError(
                f"Invalid embedding dimension for {original_id}"
            )

        # Qdrant accepts UUIDs or unsigned integer IDs.
        # Generate a deterministic UUID from the original Chroma ID.
        qdrant_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, original_id)
        )

        points.append(
            models.PointStruct(
                id=qdrant_id,
                vector=embedding.tolist(),
                payload={
                    "document": document,
                    "metadata": metadata,
                    "original_chroma_id": original_id
                }
            )
        )

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        migrated += len(points)
        print(f"Migrated {migrated}/{total_chunks} chunks")

# Verify migration
result = qdrant_client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)

print("\nMigration completed.")
print("ChromaDB chunks:", total_chunks)
print("Chunks processed:", migrated)
print("Qdrant record count:", result.count)

if result.count == total_chunks:
    print("SUCCESS: All chunks are present in Qdrant!")
else:
    print("WARNING: Record counts do not match.")
