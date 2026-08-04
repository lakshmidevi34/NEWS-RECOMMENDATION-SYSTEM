import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page config
st.set_page_config(
    page_title="News Recommender",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .article-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .similarity-score {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load preprocessed data"""
    df = pd.read_pickle('models/news_data.pkl')
    with open('models/categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    return df, categories

@st.cache_resource
def build_category_model(df, category, max_features=3000):
    """Build model for specific category"""
    df_subset = df[df['category'] == category].reset_index(drop=True)
    
    if df_subset.shape[0] < 50:
        max_features = min(max_features, 1000)
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df_subset['content'])
    
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)
    
    return df_subset, vectorizer, tfidf_matrix, nn_model

def recommend_by_index(df_subset, tfidf_matrix, nn_model, index, top_k=5):
    """Get recommendations based on article index"""
    n_samples = tfidf_matrix.shape[0]
    k = min(top_k + 1, n_samples)
    
    distances, indices = nn_model.kneighbors(tfidf_matrix[index], n_neighbors=k)
    distances = distances.flatten()
    indices = indices.flatten()
    similarities = 1 - distances
    
    results = []
    for dist, idx, sim in zip(distances[1:], indices[1:], similarities[1:]):
        row = df_subset.iloc[idx]
        results.append({
            'headline': row['headline'],
            'short_description': row['short_description'],
            'link': row.get('link', ''),
            'similarity': float(sim),
            'global_id': int(row['global_id'])
        })
    return pd.DataFrame(results)

def recommend_by_query(query_text, df_subset, vectorizer, tfidf_matrix, nn_model, top_k=5):
    """Get recommendations based on text query"""
    qv = vectorizer.transform([query_text])
    n_samples = tfidf_matrix.shape[0]
    k = min(top_k, n_samples)
    
    distances, indices = nn_model.kneighbors(qv, n_neighbors=k)
    distances = distances.flatten()
    indices = indices.flatten()
    similarities = 1 - distances
    
    results = []
    for dist, idx, sim in zip(distances, indices, similarities):
        row = df_subset.iloc[idx]
        results.append({
            'headline': row['headline'],
            'short_description': row['short_description'],
            'link': row.get('link', ''),
            'similarity': float(sim),
            'global_id': int(row['global_id'])
        })
    return pd.DataFrame(results)

def main():
    # Header
    st.markdown('<div class="main-header">📰 News Article Recommender</div>', unsafe_allow_html=True)
    
    # Check if models exist. If not, download and train them.
    if not os.path.exists('models/news_data.pkl') or not os.path.exists('models/categories.pkl'):
        with st.status("📥 Recommendation models not found. Downloading dataset and training models (takes ~15s)...", expanded=True) as status:
            st.write("Downloading news dataset from Kaggle...")
            from model import train_and_save
            train_and_save()
            status.update(label="✓ Models built successfully!", state="complete", expanded=False)
            
    # Load data
    df, categories = load_data()
    
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Category selection
    selected_category = st.sidebar.selectbox(
        "📂 Select Category",
        categories,
        help="Choose a news category to explore"
    )
    
    # Number of recommendations
    top_k = st.sidebar.slider(
        "🔢 Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5,
        help="How many similar articles to show"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 **Total Articles:** {len(df):,}\n\n📁 **Categories:** {len(categories)}")
    
    # Build model for selected category
    with st.spinner(f"Building model for {selected_category}..."):
        df_subset, vectorizer, tfidf_matrix, nn_model = build_category_model(df, selected_category)
    
    st.success(f"✓ Loaded **{len(df_subset):,}** articles from **{selected_category}** category")
    
    # Main content area
    tab1, tab2 = st.tabs(["🔍 Search by Article", "✍️ Search by Query"])
    
    with tab1:
        st.markdown("### Select an article to find similar ones")
        
        # Show sample articles
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create article options
            article_options = [(f"{row['headline'][:100]}...", idx) 
                             for idx, row in df_subset.head(200).iterrows()]
            
            selected_article = st.selectbox(
                "Choose an article:",
                options=range(len(article_options)),
                format_func=lambda x: article_options[x][0]
            )
        
        with col2:
            st.markdown("**Or enter article index:**")
            manual_index = st.number_input(
                "Article Index",
                min_value=0,
                max_value=len(df_subset)-1,
                value=selected_article,
                help="Enter any index from 0 to " + str(len(df_subset)-1)
            )
        
        if st.button("🔍 Find Similar Articles", key="btn1", type="primary"):
            idx = manual_index
            
            # Show selected article
            st.markdown("---")
            st.markdown("### 📄 Selected Article")
            selected_row = df_subset.iloc[idx]
            st.markdown(f"**{selected_row['headline']}**")
            st.write(selected_row['short_description'])
            if selected_row.get('link'):
                st.markdown(f"[🔗 Read more]({selected_row['link']})")
            
            # Get recommendations
            st.markdown("---")
            st.markdown(f"### 🎯 Top {top_k} Similar Articles")
            
            with st.spinner("Finding similar articles..."):
                recommendations = recommend_by_index(df_subset, tfidf_matrix, nn_model, idx, top_k)
            
            if recommendations.empty:
                st.warning("No recommendations found.")
            else:
                for i, row in recommendations.iterrows():
                    with st.container():
                        col_a, col_b = st.columns([4, 1])
                        with col_a:
                            st.markdown(f"**{i+1}. {row['headline']}**")
                            st.write(row['short_description'])
                            if row['link']:
                                st.markdown(f"[🔗 Read article]({row['link']})")
                        with col_b:
                            st.markdown(f"<div class='similarity-score'>{row['similarity']:.2%}</div>", 
                                      unsafe_allow_html=True)
                        st.markdown("---")
    
    with tab2:
        st.markdown("### Enter your own search query")
        
        query_text = st.text_input(
            "Type your search query:",
            placeholder="e.g., 'climate change impact', 'technology innovation', 'sports news'",
            help="Enter keywords or a short phrase"
        )
        
        if st.button("🔍 Search", key="btn2", type="primary"):
            if not query_text.strip():
                st.warning("Please enter a search query.")
            else:
                st.markdown("---")
                st.markdown(f"### 🎯 Top {top_k} Results for: *'{query_text}'*")
                
                with st.spinner("Searching..."):
                    recommendations = recommend_by_query(
                        query_text, df_subset, vectorizer, tfidf_matrix, nn_model, top_k
                    )
                
                if recommendations.empty:
                    st.warning("No results found. Try different keywords.")
                else:
                    for i, row in recommendations.iterrows():
                        with st.container():
                            col_a, col_b = st.columns([4, 1])
                            with col_a:
                                st.markdown(f"**{i+1}. {row['headline']}**")
                                st.write(row['short_description'])
                                if row['link']:
                                    st.markdown(f"[🔗 Read article]({row['link']})")
                            with col_b:
                                st.markdown(f"<div class='similarity-score'>{row['similarity']:.2%}</div>", 
                                          unsafe_allow_html=True)
                            st.markdown("---")

if __name__ == "__main__":
    main() 
