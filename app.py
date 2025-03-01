from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = os.path.join(os.getcwd(), "spam_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Function to preprocess text before prediction
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Webpage route (HTML interface)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]
        processed_text = preprocess_text(email_text)
        transformed_text = vectorizer.transform([processed_text])

        # Get spam probability
        probabilities = model.predict_proba(transformed_text)[0]
        spam_prob = round(probabilities[1] * 100, 2)  # Convert to percentage

        prediction = "Spam" if spam_prob > 50 else "Ham"
        return render_template("index.html", prediction=prediction, spam_prob=spam_prob)

    return render_template("index.html", prediction=None)

# New API route for spam detection
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON request data
        email_text = data.get("email_text", "")  # Extract email text

        if not email_text:
            return jsonify({"error": "No email text provided"}), 400

        processed_text = preprocess_text(email_text)
        transformed_text = vectorizer.transform([processed_text])

        # Get spam probability
        probabilities = model.predict_proba(transformed_text)[0]
        spam_prob = round(probabilities[1] * 100, 2)

        # Determine spam or ham
        prediction = "Spam" if spam_prob > 50 else "Ham"

        return jsonify({"prediction": prediction, "spam_probability": spam_prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


