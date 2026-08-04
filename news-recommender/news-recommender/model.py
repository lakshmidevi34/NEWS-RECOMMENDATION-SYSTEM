import os
import glob
import kagglehub
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class NewsRecommender:
    def __init__(self):
        self.df = None
        self.categories = []
        self.models = {}  # Store models per category
        
    def download_and_load_data(self):
        """Download dataset and load into DataFrame"""
        print("Downloading dataset...")
        path = kagglehub.dataset_download("rmisra/news-category-dataset")
        print(f"Downloaded dataset path: {path}")
        
        # Find JSON file
        json_files = glob.glob(os.path.join(path, "*.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
        
        if not json_files:
            raise FileNotFoundError(f"No .json found in {path}")
        
        json_path = json_files[0]
        print(f"Using JSON file: {json_path}")
        
        # Load dataset
        self.df = pd.read_json(json_path, lines=True)
        print(f"Rows loaded: {len(self.df)}")
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the dataset"""
        print("Preprocessing data...")
        
        # Combine headline and description
        self.df['headline'] = self.df['headline'].fillna('').astype(str)
        self.df['short_description'] = self.df['short_description'].fillna('').astype(str)
        self.df['content'] = (self.df['headline'] + ' ' + self.df['short_description']).str.strip()
        
        # Drop empty content
        self.df = self.df[self.df['content'].str.len() > 0].reset_index(drop=True)
        
        # Drop duplicates
        before = len(self.df)
        self.df.drop_duplicates(subset=['content'], inplace=True)
        self.df = self.df.reset_index(drop=True)
        print(f"Dropped {before - len(self.df)} duplicate rows. Remaining: {len(self.df)}")
        
        # Add global ID
        self.df['global_id'] = self.df.index
        
        # Get categories
        self.categories = sorted(self.df['category'].unique())
        print(f"Found {len(self.categories)} categories")
        
        return self.df
    
    def build_category_model(self, category, max_features=3000, ngram_range=(1, 2)):
        """Build TF-IDF and NearestNeighbors model for a specific category"""
        print(f"Building model for category: {category}")
        
        df_subset = self.df[self.df['category'] == category].reset_index(drop=True)
        
        if df_subset.shape[0] < 50:
            max_features = min(max_features, 1000)
        
        # Build TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )
        tfidf_matrix = vectorizer.fit_transform(df_subset['content'])
        
        # Build NearestNeighbors
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        nn_model.fit(tfidf_matrix)
        
        self.models[category] = {
            'df_subset': df_subset,
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'nn_model': nn_model
        }
        
        print(f"Model built for {category} with {len(df_subset)} articles")
        return self.models[category]
    
    def save_models(self, output_dir='models'):
        """Save all models and data to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main dataframe
        self.df.to_pickle(os.path.join(output_dir, 'news_data.pkl'))
        
        # Save categories
        with open(os.path.join(output_dir, 'categories.pkl'), 'wb') as f:
            pickle.dump(self.categories, f)
        
        print(f"Saved main data to {output_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load saved models and data"""
        self.df = pd.read_pickle(os.path.join(model_dir, 'news_data.pkl'))
        
        with open(os.path.join(model_dir, 'categories.pkl'), 'rb') as f:
            self.categories = pickle.load(f)
        
        print(f"Loaded data with {len(self.df)} articles and {len(self.categories)} categories")


def train_and_save():
    """Main training function"""
    recommender = NewsRecommender()
    
    # Download and preprocess
    recommender.download_and_load_data()
    recommender.preprocess_data()
    
    # Save processed data
    recommender.save_models()
    
    print("\n✓ Training complete! Models saved to 'models/' directory")
    print(f"✓ Total articles: {len(recommender.df)}")
    print(f"✓ Total categories: {len(recommender.categories)}")


if __name__ == "__main__":
    train_and_save() 
