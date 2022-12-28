from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

HALO INI TEGAR

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

review = st.text_area(label="Masukkan review (dalam bahasa Inggris):",
                      placeholder="Contoh: I like this course...")

model = pickle.load(open("Pickle_RL_Model.pkl", 'rb'))

# blahblah 1
df = pd.read_csv("./data_review_waterloo.csv")
df.reviews = df.reviews.astype(str)
review_df = df[['reviews', 'course_rating']]
review = review_df.reviews.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review)
sentiment_label = review_df.course_rating.factorize()

# blahblah 2


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]
