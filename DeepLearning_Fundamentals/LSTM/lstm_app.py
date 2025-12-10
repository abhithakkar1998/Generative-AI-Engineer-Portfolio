import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = load_model('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/DeepLearning_Fundamentals/LSTM/lstm_macbeth_model.h5')
with open('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/DeepLearning_Fundamentals/LSTM/tokenizer_macbeth.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    if len(token_list) > max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Keep only the last max_sequence_len-1 tokens
    
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    # Get the index of the highest probability word
    predicted_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    
    return None


st.title("LSTM Next Word Prediction")
input_text = st.text_input("Enter text to predict the next word:", "to be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.write(f"Input Text: '{input_text}' -> Predicted Next Word: '{next_word}'")
    else:
        st.write("Could not predict the next word.")