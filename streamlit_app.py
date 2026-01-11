import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse

st.set_page_config(layout="wide", page_title="Movie Recommendation System")

# Custom CSS for improved UI
st.markdown("""
<style>
    .movie-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        height: 280px; /* Fixed height for consistent alignment */
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 10px;
    }
    .movie-poster {
        width: 120px;
        height: 180px;
        object-fit: cover;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .movie-info {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .rank {
        font-size: 12px;
        color: #888;
        margin: 0;
        margin-bottom: 5px;
    }
    .title {
        font-size: 14px;
        font-weight: bold;
        color: white;
        margin: 0;
        line-height: 1.2;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_meta = pd.read_csv("movies_with_content_meta.csv")
    df_numeric = pd.read_csv("movies_with_numeric_score.csv")
    df = df_meta.merge(df_numeric[['id', 'numeric_score']], on='id', how='left', suffixes=('', '_dup'))
    if 'numeric_score_dup' in df.columns:
        df['numeric_score'] = df['numeric_score_dup'].fillna(df['numeric_score'])
        df.drop('numeric_score_dup', axis=1, inplace=True)
    
    story_vectorizer = joblib.load("story_tfidf_vectorizer.joblib")
    story_matrix = sparse.load_npz("story_tfidf.npz")
    creators_vectorizer = joblib.load("creators_tfidf.joblib")
    creators_matrix = sparse.load_npz("creators_tfidf.npz")
    
    title_to_index = {row['title'].lower().strip(): idx for idx, row in df.iterrows()}
    
    return df, story_matrix, creators_matrix, title_to_index

def _normalize_01(arr):
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)

def compute_story_similarity(selected_indices, story_matrix):
    sel = story_matrix[selected_indices]
    user_vec = sel.mean(axis=0)
    if not sparse.issparse(user_vec):
        user_vec = sparse.csr_matrix(user_vec)
    sims = story_matrix.dot(user_vec.T).A.flatten()
    return _normalize_01(sims)

def compute_creator_similarity(selected_indices, creators_matrix):
    sel = creators_matrix[selected_indices]
    user_vec = sel.mean(axis=0)
    if not sparse.issparse(user_vec):
        user_vec = sparse.csr_matrix(user_vec)
    sims = creators_matrix.dot(user_vec.T).A.flatten()
    return _normalize_01(sims)

def recommend_movies(selected_titles, df, story_matrix, creators_matrix, title_to_index, top_k=10):
    selected_indices = []
    unmatched = []
    for title in selected_titles:
        lower_title = title.lower().strip()
        if lower_title in title_to_index:
            selected_indices.append(title_to_index[lower_title])
        else:
            unmatched.append(title)
    if unmatched:
        raise ValueError(f"The following titles were not found: {', '.join(unmatched)}")
    if not selected_indices:
        raise ValueError("No valid titles selected")
    
    story_sim = compute_story_similarity(selected_indices, story_matrix)
    creator_sim = compute_creator_similarity(selected_indices, creators_matrix)
    
    numeric = df['numeric_score'].fillna(0).values
    numeric = _normalize_01(numeric)
    
    final_score = 0.55 * story_sim + 0.25 * creator_sim + 0.20 * numeric
    
    res = df[['title', 'poster_path']].copy()
    res['final_score'] = final_score
    res['index'] = df.index
    
    res = res[~res['index'].isin(selected_indices)]
    res = res.sort_values('final_score', ascending=False).head(top_k)
    
    return res

df, story_matrix, creators_matrix, title_to_index = load_data()

st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")

st.markdown('<div class="centered">', unsafe_allow_html=True)
movies = st.multiselect(
    "Choose one or more movies:",
    list(title_to_index.keys()),
    help="Start typing to search for movies"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="centered">', unsafe_allow_html=True)
recommend_button = st.button("Recommend")
st.markdown('</div>', unsafe_allow_html=True)

if recommend_button:
    if not movies:
        st.info("Select at least one movie to get recommendations.")
    else:
        with st.spinner("Computing recommendations..."):
            try:
                recommendations = recommend_movies(movies, df, story_matrix, creators_matrix, title_to_index, top_k=10)
                st.subheader("Top 10 Recommended Movies")
                recs_list = recommendations.to_dict('records')
                for row in range(2):
                    cols = st.columns(5)
                    for col in range(5):
                        idx = row * 5 + col
                        if idx < len(recs_list):
                            rec = recs_list[idx]
                            with cols[col]:
                                poster_url = f"https://image.tmdb.org/t/p/w342{rec['poster_path']}" if pd.notna(rec['poster_path']) and rec['poster_path'] else ""
                                poster_html = f'<img src="{poster_url}" class="movie-poster" />' if poster_url else '<div class="movie-poster" style="background-color: #333; display: flex; align-items: center; justify-content: center; color: white;">No poster</div>'
                                st.markdown(f"""
<div class="movie-card">
    {poster_html}
    <div class="movie-info">
        <p class="rank">Rank #{idx + 1}</p>
        <p class="title">{rec['title']}</p>
    </div>
</div>
""", unsafe_allow_html=True)
            except ValueError as e:
                st.error(str(e))

st.markdown('</div>', unsafe_allow_html=True)