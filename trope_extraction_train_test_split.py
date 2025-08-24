"""
WhoDunIt Dataset Trope Extraction with Train/Test Split (Deduplicated)

This script splits the 78 deduplicated stories into train (54) and test (24) sets,
then extracts tropes by n-gram size from the training set only.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_split_deduplicated_data(test_size=24, random_state=42):
    """Load deduplicated dataset and split into train/test."""
    try:
        # Load deduplicated indices
        indices_df = pd.read_csv('deduplicated_indices.csv')
        selected_indices = indices_df['selected_indices'].tolist()
        print(f"✅ Loading {len(selected_indices)} deduplicated story indices...")
        
        # Load full dataset
        dataset = load_dataset("kjgpta/WhoDunIt")
        stories = dataset['train']
        
        # Get deduplicated stories with metadata
        deduplicated_data = []
        for idx in selected_indices:
            if idx < len(stories):
                story = stories[idx]
                deduplicated_data.append({
                    'original_index': idx,
                    'title': story.get('title', f'Story {idx}'),
                    'author': story.get('author', 'Unknown'),
                    'text': story['text'],
                    'length': len(story['text'])
                })
        
        print(f"✅ Loaded {len(deduplicated_data)} deduplicated stories")
        
        # Split into train and test
        train_data, test_data = train_test_split(
            deduplicated_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # No stratification needed for mystery stories
        )
        
        print(f"📊 Split: {len(train_data)} train, {len(test_data)} test stories")
        
        # Save split information
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_df.to_csv('train_stories_deduplicated.csv', index=False)
        test_df.to_csv('test_stories_deduplicated.csv', index=False)
        
        print(f"✅ Saved train_stories_deduplicated.csv ({len(train_data)} stories)")
        print(f"✅ Saved test_stories_deduplicated.csv ({len(test_data)} stories)")
        
        return train_data, test_data
        
    except Exception as e:
        print(f"❌ Error loading and splitting data: {e}")
        return None, None

def preprocess_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and normalize whitespace
    text = text.lower()
    text = ' '.join(text.split())
    
    return text.strip()

def extract_ngrams_by_size(texts, ngram_size, min_df=5, max_df=0.8, max_features=500):
    """Extract n-grams of a specific size from texts."""
    print(f"\n📊 EXTRACTING {ngram_size}-GRAMS (TRAIN SET):")
    print(f"   Min doc frequency: {min_df}")
    print(f"   Max doc frequency: {max_df}")
    print(f"   Max features: {max_features}")
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(ngram_size, ngram_size),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words='english',
        lowercase=True
    )
    
    try:
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"   ✅ Extracted {len(feature_names)} {ngram_size}-grams from {len(texts)} train stories")
        
        # Calculate statistics for each n-gram
        results = []
        for i, phrase in enumerate(feature_names):
            phrase_scores = tfidf_matrix[:, i].toarray().flatten()
            
            avg_score = np.mean(phrase_scores)
            max_score = np.max(phrase_scores)
            num_docs = np.sum(phrase_scores > 0)
            
            results.append({
                'phrase': phrase,
                'avg_tfidf_score': avg_score,
                'max_tfidf_score': max_score,
                'num_train_stories': num_docs,
                'ngram_size': ngram_size
            })
        
        # Convert to DataFrame and sort by TF-IDF score
        df = pd.DataFrame(results)
        df = df.sort_values('avg_tfidf_score', ascending=False)
        
        return df, vectorizer
        
    except Exception as e:
        print(f"   ❌ Error extracting {ngram_size}-grams: {e}")
        return pd.DataFrame(), None

def display_results(df, ngram_size, top_n=20):
    """Display top results for an n-gram size."""
    if len(df) == 0:
        print(f"   ❌ No {ngram_size}-grams found")
        return
    
    print(f"\n🏆 TOP {top_n} {ngram_size}-GRAMS (TRAIN SET):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Phrase':<40} {'TF-IDF':<10} {'Train Stories':<12}")
    print("-" * 80)
    
    for i, row in df.head(top_n).iterrows():
        print(f"{i+1:<4} {row['phrase']:<40} {row['avg_tfidf_score']:<10.4f} {row['num_train_stories']:<12}")

def main():
    """Main function to extract tropes with train/test split."""
    print("=" * 80)
    print("WhoDunIt Trope Extraction - Train/Test Split (Deduplicated)")
    print("=" * 80)
    
    # Load and split deduplicated dataset
    train_data, test_data = load_and_split_deduplicated_data(test_size=24, random_state=42)
    if train_data is None:
        return None
    
    # Extract texts from train data
    train_texts = [story['text'] for story in train_data]
    
    print(f"\n📋 DATASET SPLIT INFO:")
    print("-" * 40)
    print(f"Total deduplicated stories: {len(train_data) + len(test_data)}")
    print(f"Training stories: {len(train_data)}")
    print(f"Test stories: {len(test_data)}")
    print(f"Train/test ratio: {len(train_data)/(len(train_data) + len(test_data)):.1%}/{len(test_data)/(len(train_data) + len(test_data)):.1%}")
    
    # Configure different thresholds for each n-gram size
    ngram_configs = {
        1: {'min_df': 5, 'max_features': 300},   # 1-grams: min_df=5
        2: {'min_df': 5, 'max_features': 400},   # 2-grams: min_df=5
        3: {'min_df': 3, 'max_features': 200},   # 3-grams: min_df=3 (more lenient)
        4: {'min_df': 3, 'max_features': 100}    # 4-grams: min_df=3 (more lenient)
    }
    
    print(f"\n📋 EXTRACTION CONFIGURATION (TRAIN SET ONLY):")
    print("-" * 50)
    for ngram_size, config in ngram_configs.items():
        print(f"{ngram_size}-grams: min_df={config['min_df']}, max_features={config['max_features']}")
    
    all_results = {}
    all_vectorizers = {}
    
    # Extract each n-gram size separately from training data
    for ngram_size, config in ngram_configs.items():
        df, vectorizer = extract_ngrams_by_size(
            train_texts,
            ngram_size,
            min_df=config['min_df'],
            max_df=0.8,
            max_features=config['max_features']
        )
        all_results[ngram_size] = df
        all_vectorizers[ngram_size] = vectorizer
        
        # Display results
        display_results(df, ngram_size, top_n=20)
    
    # Save results to separate files
    print(f"\n💾 SAVING TRAIN-BASED TROPE RESULTS:")
    print("=" * 50)
    
    for ngram_size, df in all_results.items():
        if len(df) > 0:
            filename = f'tropes_{ngram_size}gram_train_only.csv'
            df.to_csv(filename, index=False)
            print(f"✅ {ngram_size}-grams: {filename} ({len(df)} phrases)")
        else:
            print(f"❌ {ngram_size}-grams: No phrases found")
    
    # Save vectorizers for later use on test set
    print(f"\n💾 SAVING VECTORIZERS FOR TEST SET APPLICATION:")
    print("-" * 50)
    import pickle
    
    for ngram_size, vectorizer in all_vectorizers.items():
        if vectorizer is not None:
            filename = f'vectorizer_{ngram_size}gram_train.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"✅ {ngram_size}-gram vectorizer: {filename}")
    
    # Create summary analysis
    print(f"\n📊 SUMMARY ANALYSIS:")
    print("=" * 40)
    
    total_phrases = 0
    summary_data = []
    
    for ngram_size, df in all_results.items():
        count = len(df)
        total_phrases += count
        
        if count > 0:
            avg_score = df['avg_tfidf_score'].mean()
            max_score = df['avg_tfidf_score'].max()
            min_stories = df['num_train_stories'].min()
            max_stories = df['num_train_stories'].max()
            
            summary_data.append({
                'ngram_size': ngram_size,
                'total_phrases': count,
                'avg_tfidf_score': avg_score,
                'max_tfidf_score': max_score,
                'min_train_stories': min_stories,
                'max_train_stories': max_stories,
                'min_df_used': ngram_configs[ngram_size]['min_df'],
                'train_set_size': len(train_data)
            })
            
            print(f"{ngram_size}-grams: {count:3d} phrases "
                  f"(avg TF-IDF: {avg_score:.4f}, "
                  f"train stories: {min_stories}-{max_stories}, "
                  f"min_df: {ngram_configs[ngram_size]['min_df']})")
        else:
            print(f"{ngram_size}-grams:   0 phrases")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('ngram_extraction_train_summary.csv', index=False)
    print(f"\n✅ Summary saved to: ngram_extraction_train_summary.csv")
    
    print(f"\nTotal phrases extracted from training set: {total_phrases}")
    print(f"Training stories used: {len(train_data)}")
    print(f"Test stories reserved: {len(test_data)}")
    
    # Create combined file with top entries from each size
    print(f"\n📄 CREATING COMBINED TRAIN-BASED FILE:")
    print("-" * 40)
    
    combined_results = []
    for ngram_size, df in all_results.items():
        if len(df) > 0:
            # Take top 50 from each size
            top_phrases = df.head(50).copy()
            combined_results.append(top_phrases)
    
    if combined_results:
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_df.to_csv('tropes_all_sizes_train_only.csv', index=False)
        print(f"✅ Combined file: tropes_all_sizes_train_only.csv ({len(combined_df)} phrases)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("=" * 30)
    print("1. Review tropes_*gram_train_only.csv files for manual selection")
    print("2. Use train_stories_deduplicated.csv for training your model")
    print("3. Use test_stories_deduplicated.csv for testing your model")
    print("4. Apply saved vectorizers to test set when needed")
    print("5. Create final trope vocabulary from training set")
    
    print(f"\n✅ EXTRACTION WITH TRAIN/TEST SPLIT COMPLETE!")
    print("="*80)
    
    return all_results, all_vectorizers, train_data, test_data

if __name__ == "__main__":
    results, vectorizers, train_data, test_data = main()
