"""
WhoDunIt Dataset Trope Extraction and Vectorization

This script mines frequent clue-like phrases ("tropes") from the WhoDunIt dataset
and represents each story as a vector indicating the presence or frequency of these tropes.

Author: Assistant
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TropeExtractor:
    """
    A class to extract and vectorize tropes from mystery story texts.
    """
    
    def __init__(self, ngram_range=(2, 4), min_df=5, max_df=0.8, max_features=1000):
        """
        Initialize the TropeExtractor.
        
        Parameters:
        - ngram_range: tuple, range of n-grams to consider (default: 2-4 grams)
        - min_df: int, minimum document frequency for a phrase to be included
        - max_df: float, maximum document frequency (to filter out too common phrases)
        - max_features: int, maximum number of features to keep
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vectorizer = None
        self.trope_matrix = None
        self.feature_names = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Clean and preprocess the text for better trope extraction.
        
        Parameters:
        - text: str, raw text to preprocess
        
        Returns:
        - str: cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation that might be part of tropes
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '', text)
        
        # Remove standalone numbers (but keep numbers that are part of words)
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()
    
    def is_mystery_relevant(self, phrase):
        """
        Check if a phrase is potentially relevant to mystery/detective stories.
        
        Parameters:
        - phrase: str, the phrase to check
        
        Returns:
        - bool: True if phrase seems mystery-relevant
        """
        # Common mystery-related keywords and patterns
        mystery_keywords = [
            'murder', 'kill', 'death', 'dead', 'body', 'corpse', 'victim',
            'suspect', 'alibi', 'motive', 'evidence', 'clue', 'witness',
            'detective', 'police', 'investigate', 'crime', 'blood', 'weapon',
            'knife', 'gun', 'poison', 'confession', 'guilty', 'innocent',
            'secret', 'lie', 'truth', 'mystery', 'suspicious', 'strange',
            'disappear', 'missing', 'found', 'discover', 'reveal', 'hide',
            'footprint', 'fingerprint', 'note', 'letter', 'diary', 'will',
            'inherit', 'money', 'blackmail', 'threaten', 'revenge', 'jealous'
        ]
        
        # Check if phrase contains any mystery-relevant words
        phrase_lower = phrase.lower()
        for keyword in mystery_keywords:
            if keyword in phrase_lower:
                return True
        
        # Check for certain patterns that might indicate mystery elements
        mystery_patterns = [
            r'\b(suddenly|quietly|secretly|mysteriously)\b',
            r'\b(room|house|door|window|stairs|basement|attic)\b',
            r'\b(night|dark|shadow|light|candle|lamp)\b',
            r'\b(scream|shout|whisper|voice|sound|noise)\b',
            r'\b(key|lock|safe|drawer|box|chest)\b',
            r'\b(blood|stain|mark|scratch|bruise)\b'
        ]
        
        for pattern in mystery_patterns:
            if re.search(pattern, phrase_lower):
                return True
        
        return False
    
    def extract_tropes(self, texts):
        """
        Extract tropes from a collection of texts using TF-IDF vectorization.
        
        Parameters:
        - texts: list, collection of story texts
        
        Returns:
        - scipy.sparse matrix: TF-IDF matrix
        - list: feature names (tropes)
        """
        print(f"Preprocessing {len(texts)} texts...")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print(f"Extracting n-grams ({self.ngram_range[0]}-{self.ngram_range[1]})...")
        
        # Create TF-IDF vectorizer with specified parameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=None,  # We'll filter later
            stop_words='english',
            lowercase=True,
            # Use default word tokenization
        )
        
        # Fit and transform the texts
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Initial features extracted: {len(feature_names)}")
        
        # Filter for mystery-relevant phrases
        print("Filtering for mystery-relevant phrases...")
        mystery_relevant_indices = []
        mystery_relevant_names = []
        
        for i, phrase in enumerate(feature_names):
            if self.is_mystery_relevant(phrase):
                mystery_relevant_indices.append(i)
                mystery_relevant_names.append(phrase)
        
        print(f"Mystery-relevant features: {len(mystery_relevant_names)}")
        
        # Keep only mystery-relevant features
        if mystery_relevant_indices:
            tfidf_matrix = tfidf_matrix[:, mystery_relevant_indices]
            feature_names = mystery_relevant_names
        
        # If we still have too many features, keep the top ones by variance
        if len(feature_names) > self.max_features:
            print(f"Reducing to top {self.max_features} features by variance...")
            feature_variances = np.array(tfidf_matrix.toarray()).var(axis=0)
            top_indices = np.argsort(feature_variances)[-self.max_features:]
            tfidf_matrix = tfidf_matrix[:, top_indices]
            feature_names = [feature_names[i] for i in top_indices]
        
        self.trope_matrix = tfidf_matrix
        self.feature_names = feature_names
        
        print(f"Final trope vocabulary size: {len(self.feature_names)}")
        
        return tfidf_matrix, feature_names
    
    def get_trope_dataframe(self, story_ids=None):
        """
        Convert the trope matrix to a pandas DataFrame.
        
        Parameters:
        - story_ids: list, optional story identifiers
        
        Returns:
        - pandas.DataFrame: DataFrame with stories as rows and tropes as columns
        """
        if self.trope_matrix is None:
            raise ValueError("Tropes haven't been extracted yet. Call extract_tropes() first.")
        
        # Convert sparse matrix to dense for DataFrame creation
        dense_matrix = self.trope_matrix.toarray()
        
        # Create DataFrame
        if story_ids is None:
            story_ids = [f"story_{i}" for i in range(dense_matrix.shape[0])]
        
        df = pd.DataFrame(
            dense_matrix,
            index=story_ids,
            columns=self.feature_names
        )
        
        return df
    
    def visualize_top_tropes(self, top_n=20, figsize=(12, 8)):
        """
        Visualize the top N most frequent tropes.
        
        Parameters:
        - top_n: int, number of top tropes to visualize
        - figsize: tuple, figure size for the plot
        """
        if self.trope_matrix is None:
            raise ValueError("Tropes haven't been extracted yet. Call extract_tropes() first.")
        
        # Calculate mean TF-IDF scores for each trope
        mean_scores = np.array(self.trope_matrix.mean(axis=0)).flatten()
        
        # Get top N tropes
        top_indices = np.argsort(mean_scores)[-top_n:]
        top_tropes = [self.feature_names[i] for i in top_indices]
        top_scores = mean_scores[top_indices]
        
        # Create visualization
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_tropes)), top_scores)
        plt.yticks(range(len(top_tropes)), top_tropes)
        plt.xlabel('Average TF-IDF Score')
        plt.title(f'Top {top_n} Most Frequent Mystery Tropes')
        plt.tight_layout()
        
        # Add value labels on bars
        for i, score in enumerate(top_scores):
            plt.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=8)
        
        plt.show()
        
        return top_tropes, top_scores


def main():
    """
    Main function to demonstrate the trope extraction workflow.
    """
    print("=== WhoDunIt Dataset Trope Extraction ===")
    print()
    
    # Load the dataset
    print("1. Loading WhoDunIt dataset...")
    try:
        dataset = load_dataset("kjgpta/WhoDunIt")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Use the train split (or the first available split)
        if 'train' in dataset:
            stories = dataset['train']
        else:
            stories = dataset[list(dataset.keys())[0]]
        
        print(f"Number of stories: {len(stories)}")
        print(f"Columns: {stories.column_names}")
        print()
        
        # Extract text data
        texts = stories['text']
        print(f"Sample text length: {len(texts[0])} characters")
        print(f"Sample text preview: {texts[0][:200]}...")
        print()
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using sample data for demonstration...")
        
        # Create sample mystery story data for demonstration
        sample_stories = [
            "The detective found a bloody knife in the library. The suspect had no alibi for the time of murder. A mysterious note was discovered under the victim's body.",
            "The witness saw a strange figure near the crime scene. The victim's diary revealed a dark secret about the inheritance. The butler confessed to hiding evidence.",
            "The murder weapon was missing from the kitchen. Fingerprints on the door handle matched the suspect. The victim received threatening letters before death.",
            "A secret passage led to the victim's room. The alibi was proven false by the detective. Blood stains were found on the suspect's clothes.",
            "The victim's will was changed recently. Suspicious footprints were found in the garden. The housekeeper discovered the body at midnight."
        ]
        
        texts = sample_stories * 20  # Repeat for more data
        print(f"Using {len(texts)} sample stories for demonstration")
        print()
    
    # Initialize the trope extractor
    print("2. Initializing Trope Extractor...")
    extractor = TropeExtractor(
        ngram_range=(2, 4),  # 2-4 word phrases
        min_df=2,           # Appear in at least 2 stories
        max_df=0.8,         # Appear in at most 80% of stories
        max_features=500    # Keep top 500 tropes
    )
    
    print("Parameters:")
    print(f"  - N-gram range: {extractor.ngram_range}")
    print(f"  - Min document frequency: {extractor.min_df}")
    print(f"  - Max document frequency: {extractor.max_df}")
    print(f"  - Max features: {extractor.max_features}")
    print()
    
    # Extract tropes
    print("3. Extracting tropes...")
    tfidf_matrix, feature_names = extractor.extract_tropes(texts)
    print(f"Trope matrix shape: {tfidf_matrix.shape}")
    print()
    
    # Create DataFrame
    print("4. Creating trope DataFrame...")
    story_ids = [f"story_{i:04d}" for i in range(len(texts))]
    trope_df = extractor.get_trope_dataframe(story_ids)
    
    print(f"DataFrame shape: {trope_df.shape}")
    print(f"Sample columns: {list(trope_df.columns[:10])}")
    print()
    
    # Display sample results
    print("5. Sample Results:")
    print("\nFirst 5 rows and 10 columns:")
    print(trope_df.iloc[:5, :10])
    print()
    
    print("Top 10 tropes by average TF-IDF score:")
    mean_scores = trope_df.mean().sort_values(ascending=False)
    for i, (trope, score) in enumerate(mean_scores.head(10).items()):
        print(f"{i+1:2d}. {trope:<25} (avg score: {score:.4f})")
    print()
    
    # Summary statistics
    print("6. Summary Statistics:")
    print(f"Total unique tropes: {len(trope_df.columns)}")
    print(f"Average tropes per story: {(trope_df > 0).sum(axis=1).mean():.1f}")
    print(f"Most common trope: {mean_scores.index[0]} (appears in {(trope_df[mean_scores.index[0]] > 0).sum()} stories)")
    print()
    
    # Visualize top tropes
    print("7. Visualizing top tropes...")
    try:
        top_tropes, top_scores = extractor.visualize_top_tropes(top_n=20)
        print("Visualization displayed!")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Save results
    print("8. Saving results...")
    trope_df.to_csv('mystery_tropes.csv', index=True)
    print("Results saved to 'mystery_tropes.csv'")
    
    # Save trope vocabulary
    trope_vocab = pd.DataFrame({
        'trope': feature_names,
        'avg_tfidf': mean_scores.values
    }).sort_values('avg_tfidf', ascending=False)
    trope_vocab.to_csv('trope_vocabulary.csv', index=False)
    print("Trope vocabulary saved to 'trope_vocabulary.csv'")
    print()
    
    print("=== Trope Extraction Complete ===")
    
    return trope_df, extractor


if __name__ == "__main__":
    trope_df, extractor = main() 