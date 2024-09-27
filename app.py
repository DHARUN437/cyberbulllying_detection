import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load('LinearSVCTuned.pkl')
vectorizer = joblib.load('tfidfvectoizer.pkl')

# Load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Page title and description
st.title('🚨 Cyberbullying Detection App 🚨')
st.write("""
    This app uses Machine Learning to detect potential cyberbullying in text messages. Enter some text, and the model will predict if the content could be considered bullying or not.
""")

# Input text area
st.subheader("Enter the text to analyze:")
user_input = st.text_area("Type your message here...")

# Button for submission
if st.button('Analyze Text'):
    if user_input:
        # Preprocessing the input
        data = [user_input]
        tfidf_vector = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=joblib.load(open("tfidfvectoizer.pkl", "rb")))
        preprocessed_data = tfidf_vector.fit_transform(data)

        # Prediction
        prediction = model.predict(preprocessed_data)[0]

        # Display result
        if prediction == 1:
            st.markdown("Please be mindful of the language used")
            st.error("### ⚠️ **The content is considered as Bullying**")
            
        else:
            st.markdown("No harmful intent detected")
            st.success("### ✅ **The content is not considered as Bullying**")
    else:
        st.warning("Please enter some text to analyze.")
