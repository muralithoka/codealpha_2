import json

from flask import Flask, render_template, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources (runs only the first time)
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

# -----------------------------
# Load FAQ Data
# -----------------------------

with open("faq_data.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

stop_words = set(stopwords.words("english"))


# -----------------------------
# Text Preprocessing
# -----------------------------

def preprocess(text):

    tokens = word_tokenize(text.lower())

    filtered = [
        word for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(filtered)


processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        suggestions=questions[:6]
    )


# -----------------------------
# Chat API
# -----------------------------

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json.get("message", "")

    processed_message = preprocess(user_message)

    user_vector = vectorizer.transform([processed_message])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score > 0.30:

        response = answers[best_match]

    else:

        response = (
            "Sorry, I couldn't find a matching answer. "
            "Please try asking in a different way."
        )

    return jsonify({

        "answer": response,

        "score": round(float(score), 2)

    })


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)