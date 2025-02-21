# Sentiment Analyzer for Internship Project

This project is a simple Sentiment Analysis web application that determines whether a given text expresses a **positive** or **negative** sentiment. The app is built using **Python (Flask)** for the backend and **HTML, CSS, and JavaScript** for the frontend. It utilizes **NLTK** and **Scikit-learn** for sentiment classification using a **Naïve Bayes model** trained on the `movie_reviews` dataset.

## Features
- Analyze text sentiment (Positive or Negative)
- User-friendly interface with a clean UI
- Machine Learning model trained with NLTK's `movie_reviews` dataset
- Backend built with Flask
- Frontend built using HTML, CSS, and JavaScript

## Project Structure
📂 Sentiment-Analyzer → The main project folder.
📄 app.py → The Flask backend that processes user input and returns sentiment analysis results.
📄 train_model.py → A script to train and save the sentiment analysis model.
📄 model.pkl → The trained sentiment analysis model (serialized using pickle).
📄 vectorizer.pkl → The text vectorizer (used to convert text into numerical format).
📄 index.html → The frontend HTML page where users input text.
📄 styles.css → The CSS file that styles the web app.
📄 README.md → The documentation file explaining the project.

### 1. Clone the Repository

git clone (https://github.com/SuhasShenoy777/Sentimental-Analysis-Using-NLTK.git)
cd sentiment-analyzer

### 2. Install Dependencies
Ensure you have Python 3.7+ installed, then run:
pip install -r requirements.txt
If requirements.txt is not provided, manually install the dependencies:
pip install flask nltk scikit-learn

### 3. Train the Model (Optional)
If you want to retrain the model, run:

python train_model.py
This will create/update model.pkl and vectorizer.pkl.

### 4. Run the Flask App
Start the application using:

python app.py
By default, the app runs on http://127.0.0.1:5000/.

### Usage
Open the web browser and go to http://127.0.0.1:5000/.
Enter a sentence to analyze.
Click the "Analyze" button.
The app will display whether the sentiment is Positive or Negative.

### Screenshots

provided in the "Sentimental_Analysis" folder for :
1)Positive Sentiment Example
2)Negative Sentiment Example


### Technologies Used
Python (Flask)
Natural Language Toolkit (NLTK)
Scikit-learn
HTML, CSS, JavaScript
Bootstrap (for styling, if used)

Contributing
Feel free to fork this repository and submit pull requests!

