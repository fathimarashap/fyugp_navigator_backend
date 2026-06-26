import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load chunks from JSON
with open("data/chunks.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
documents = [chunk["content"] for chunk in chunks]
ids = [chunk["id"] for chunk in chunks]
topics = [chunk["topic"] for chunk in chunks]

# Build TF-IDF vectorizer and matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Save everything needed for retrieval later
with open("tfidf_data.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "documents": documents,
        "ids": ids,
        "topics": topics
    }, f)

print(f"Successfully indexed {len(chunks)} chunks using TF-IDF")