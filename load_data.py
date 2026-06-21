import chromadb
import json

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="fyugp_navigator",
    metadata={"hnsw:space": "cosine"}
)

# Load chunks from JSON
with open("data/chunks.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]

# Add chunks to ChromaDB
documents = []
ids = []
metadatas = []

for chunk in chunks:
    documents.append(chunk["content"])
    ids.append(chunk["id"])
    metadatas.append({"topic": chunk["topic"]})

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Successfully loaded {len(chunks)} chunks into ChromaDB")