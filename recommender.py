import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


movies_df = pd.read_csv("movies.csv")
links = pd.read_csv("imdbLinks.csv")
genre_df = movies_df['genres'].str.get_dummies("|")
st.title("Movie Recommendation System")
@st.cache_data
def calculate_similarity_matrix(genre_df):
    similarity_matrix = cosine_similarity(genre_df)
    return similarity_matrix

similarity_matrix = calculate_similarity_matrix(genre_df)

@st.cache_data
def vectorize_data(movies):
    vec = TfidfVectorizer()
    vectorized = vec.fit_transform(movies['title'])
    return vectorized,vec

def search_movie(title):
    vectorized_data,vec = vectorize_data(movies_df)
    title_vector = vec.transform([title])

    similarity_scores = cosine_similarity(title_vector, vectorized_data).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    results = [(movies_df['title'][i], similarity_scores[i]) for i in similar_indices]

    return results[0][0]

if 'search' not in st.session_state:
    st.session_state['search'] = ""

movie = st.text_input("Search",placeholder='Search for movies',key="search")
if 'random_movies' not in st.session_state:
    st.session_state['random_movies'] = movies_df['title'].sample(5)
col1,col2 = st.columns(2)
random_movies = st.session_state['random_movies']
movie1,id = None,0
if not movie:
    with col1:
        for i,movie in enumerate(random_movies,0):
            if st.button(movie):
                movie1 = movie
                id = i
                st.write("Selected Movie:",movie1)
                movie_id = movies_df[movies_df['title'] == random_movies.iloc[id]].index
                similar_movies_idx = similarity_matrix[movie_id[0]].argsort()[::-1][1:7]
                similar_movies = movies_df['title'].iloc[similar_movies_idx]
                movies_links = links.iloc[similar_movies_idx].to_numpy()
                if movie1 in similar_movies:
                    index = np.where(similar_movies == movie1)[0][0]
        
                    similar_movies = np.delete(similar_movies,index)
                with col2:
                    st.subheader("Recommended Movies")
                    for i,j in enumerate(similar_movies[:5]):
                        st.write(f"\t{i+1}.{j}:[IMDb]({movies_links[i][0][:-1]})")
else:
    result = search_movie(movie)
    id = movies_df[movies_df['title'] == result].index[0]
    similar_movies_idx = similarity_matrix[id].argsort()[::-1][1:7]
    similar_movies = movies_df['title'].iloc[similar_movies_idx].to_numpy()
    movies_links = links.iloc[similar_movies_idx].to_numpy()
    if result in similar_movies:
        index = np.where(similar_movies == result)[0][0]
        
        similar_movies = np.delete(similar_movies,index)
    st.subheader("Recommended Movies")
    for i,j in enumerate(similar_movies[:5]):
        st.write(f"\t{i+1}.{j}:[IMDb]({movies_links[i][0][:-1]})")
if st.button("Refresh", key='refresh'):
    st.session_state['random_movies'] = movies_df['title'].sample(5)
    del st.session_state['search']
    st.rerun()
