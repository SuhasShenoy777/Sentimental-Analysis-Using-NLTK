from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained pipeline (vectorizer + model together)
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__, template_folder="templates")
CORS(app)

# Serve the frontend
@app.route("/")
def home():
    return render_template("index.html")

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words)

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]  # ✅ FIXED

    return jsonify({"prediction": "Positive" if prediction == 'pos' else "Negative"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
