import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_semester_refs(text):
    return re.sub(r'\bsem(?:ester)?\.?\s*(\d)\b', r'semesternum\1', text, flags=re.IGNORECASE)


with open("data/chunks.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]

boosted_documents = []
for chunk in chunks:
    content = chunk["content"]
    match = re.match(r'semester(\d)_courses', chunk["id"])
    if match:
        num = match.group(1)
        boost = f"Semester {num} Semester {num} Semester {num} course list courses subjects. " * 3
        content = boost + content
    boosted_documents.append(normalize_semester_refs(content))

ids = [chunk["id"] for chunk in chunks]
topics = [chunk["topic"] for chunk in chunks]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(boosted_documents)

with open("tfidf_data.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "documents": [chunk["content"] for chunk in chunks],  # original text for LLM context
        "ids": ids,
        "topics": topics
    }, f)

print(f"Successfully indexed {len(chunks)} chunks with semester boosting")