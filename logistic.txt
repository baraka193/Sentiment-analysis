import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load components
model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit App
st.title("🧠 Sentiment Analyzer")
st.write("Type a sentence below to find out if it's **Positive**, **Neutral**, or **Negative**.")

# User input
user_input = st.text_area("Enter your sentence:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:
        cleaned = preprocess(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)
        sentiment = encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Sentiment: **{sentiment}**")
