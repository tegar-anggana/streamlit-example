from keras.models import load_model
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
# import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
# Analisis Sentimen terhadap Review Mahasiswa
"""

input_review = st.text_area(label="Masukkan review (dalam bahasa Inggris):",
                            placeholder="Contoh: I like this course...")
analisis_button = st.button(label="Analisis")

st.write(analisis_button)

model = load_model('my_h5_model.h5')
df = pd.read_csv("./data_review_waterloo.csv")
df.reviews = df.reviews.astype(str)
review_df = df[['reviews', 'course_rating']]
review = review_df.reviews.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review)
sentiment_label = review_df.course_rating.factorize()


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]
