%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Library Management System")

@st.cache_data
def load_data():
    df = pd.read_csv('google_books_dataset.csv')
    df['description'] = df['description'].fillna('')
    if 'is_available' not in df.columns:
        df['is_available'] = True
    return df

df = load_data()

# Sidebar Search
st.sidebar.header("Search Books")
query = st.sidebar.text_input("Enter keyword (title, author, etc.)")

if query:
    mask = (df['title'].str.contains(query, case=False, na=False) | 
            df['authors'].str.contains(query, case=False, na=False))
    results = df[mask].head(10)
    st.write(f"### Search Results for '{query}'")
    st.dataframe(results[['title', 'authors', 'categories', 'is_available']])

# Recommendation Section
st.header("AI Recommendations")
selected_book = st.selectbox("Select a book to get recommendations:", df['title'].values)

if st.button("Recommend"):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    idx = df[df['title'] == selected_book].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = cosine_sim.argsort()[-6:-1][::-1]
    recs = df.iloc[sim_indices]
    st.write("Books similar to your choice:")
    st.table(recs[['title', 'authors']])
