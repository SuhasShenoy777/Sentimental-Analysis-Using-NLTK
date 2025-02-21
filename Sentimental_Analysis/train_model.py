import nltk
from nltk.corpus import movie_reviews
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download dataset if not available
nltk.download('movie_reviews')

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle dataset
random.shuffle(documents)

# Prepare data
X_text = [" ".join(words) for words, label in documents]  # Convert words to sentences
y_labels = [label for words, label in documents]  # Labels (pos/neg)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels, test_size=0.2, random_state=42)

# Create a text processing + Naive Bayes pipeline
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
model = make_pipeline(vectorizer, MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Vectorizer and model saved successfully!")
