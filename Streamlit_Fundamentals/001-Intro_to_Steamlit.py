import streamlit as st
import pandas as pd
import numpy as np

##Title of the application
st.title("Hello Streamlit")

df = pd.DataFrame({
        'first_col': [1,2,3],
        'second_col': [10,20,30]
    })

##Display DataFrame
st.write("Here is the dataframe")
st.write(df)

##Create a line chart
chart_data = pd.DataFrame(
        np.random.randn(20,3), columns=['a','b','c']
    )

st.line_chart(chart_data)


name = st.text_input("Enter your name:")
age = st.slider("Select you age", min_value=1, max_value=100, value=18)
options = ["Python", "Java", "C++", "JavaScript"]
choice = st.selectbox("Choose your favorite language:", options)
st.write(f"You selected {choice}")

if name and age:
    st.write(f"Hello {name}. Welcome to Streamlit Demo")
    st.write(f"Your age is {age}")