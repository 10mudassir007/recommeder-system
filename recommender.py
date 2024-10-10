import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("movies.csv")

genre_df = movies_df['genres'].str.get_dummies("|")

st.write(genre_df)

similarity_matrix = cosine_similarity(genre_df)
