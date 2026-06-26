import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_semester_refs(text):
    """Converts 'Sem 4', 'Semester 4', 'sem4' etc into a single token 'semesternum4'
    so TF-IDF treats the semester number as one strong, exact-matching term
    instead of diluting it across generic digit tokens."""
    text = re.sub(r'\bsem(?:ester)?\.?\s*(\d)\b', r'semesternum\1', text, flags=re.IGNORECASE)
    return text


# Load chunks from JSON
with open("data/chunks.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
documents = [normalize_semester_refs(chunk["content"]) for chunk in chunks]
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
        "documents": [chunk["content"] for chunk in chunks],  # store ORIGINAL text for the LLM context
        "ids": ids,
        "topics": topics
    }, f)

print(f"Successfully indexed {len(chunks)} chunks using TF-IDF with semester normalization")