import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("movies.csv")

genre_df = movies_df['genres'].str.get_dummies("|")

st.write(genre_df)

similarity_matrix = cosine_similarity(genre_df)

random_movies = movies_df['title'].sample(5)

movie = st.text_input("Search for a movie")

if not movie:
  movie0 = st.button(random_movies[0])
  movie1 = st.button(random_movies[1])
  movie2 = st.button(random_movies[2])
  movie3 = st.button(random_movies[3])
  movie4 = st.button(random_movies[4])
