import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Input
from keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
model = load_model('C:/Users/Abhi K Thakkar/Desktop/GenAI Udemy/Generative-AI-Engineer-Portfolio/DeepLearning_Fundamentals/RNN/simple_rnn_imdb.h5')

def preprocess_text(text):
    """
    Preprocess a given text for the model.

    Parameters
    ----------
    text : str
        The text to be preprocessed.

    Returns
    -------
    preprocessed_input : array
        The preprocessed input for the model.
    """
    # Split the text into words
    text = text.lower()
    words = text.split()
    # Encode the words using the word index
    encoded_text = [word_index.get(word, 2) + 3 for word in words]
    # Pad the encoded text with zeros to the maximum length
    padded_text = sequence.pad_sequences([encoded_text], maxlen=2500)
    # Return the preprocessed input
    return padded_text

def predict_sentiment(review):
    """
    Predict sentiment of a given review based on the trained model.

    Parameters
    ----------
    review : str
        The text to be analyzed.

    Returns
    -------
    sentiment : str
        The sentiment of the review.
    prediction : float
        The prediction of the model.
    """
    preprocessed_input = preprocess_text(review)
    # Get the prediction of the model
    prediction = model.predict(preprocessed_input)
    # Determine the sentiment based on the prediction
    sentiment = "Positive" if prediction > 0.5 else 'Negative'
    # Return the sentiment and the prediction
    return sentiment, prediction

import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")
st.markdown("---")
user_input = st.text_area("Enter a movie review:")

if st.button("Classify"):
    sentiment, prediction = predict_sentiment(user_input)
    st.write(f"The sentiment of the review is: {sentiment}")
    st.write(f"The prediction value (0-1) of the model is: {prediction}")