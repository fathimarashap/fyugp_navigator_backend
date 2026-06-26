import os
import pickle
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)

# Load TF-IDF data once at startup
with open("tfidf_data.pkl", "rb") as f:
    tfidf_data = pickle.load(f)

vectorizer = tfidf_data["vectorizer"]
tfidf_matrix = tfidf_data["tfidf_matrix"]
documents = tfidf_data["documents"]

# Initialize Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_relevant_chunks(question, n_results=3):
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-n_results:][::-1]
    return [documents[i] for i in top_indices]

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""You are a helpful assistant for BSc Computer Science Honours students under the CUFYUGP Regulations 2024 (University of Calicut). You help students understand their FYUGP degree programme rules, eligibility criteria, grading system, and regulations.

Answer the student question using only the information provided in the context below. If the answer is not in the context, say I do not have information about that. Please check with your department or refer to the official syllabus document. Do not make up information.

If a student asks about syllabus content or what topics are covered in a specific subject, tell them to check the Syllabus section of the app.

Context:
{context}

Student Question: {question}

Answer:"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "")

    if not question:
        return jsonify({"error": "No message provided"}), 400

    chunks = get_relevant_chunks(question)
    prompt = build_prompt(question, chunks)

    message = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = message.choices[0].message.content
    return jsonify({"reply": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")