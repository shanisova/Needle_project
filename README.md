# Needle Project - Mystery Story Character Classification

A machine learning system for identifying culprits in mystery stories using proximity-based trope analysis and pure alias character extraction.

## 🎯 Project Overview

This project implements a character classification system that:
- Extracts character candidates from story metadata using pure alias-only approach
- Uses proximity features based on categorized mystery tropes
- Trains a logistic regression classifier to identify story culprits
- Provides a Streamlit UI for interactive story analysis

## 📁 Key Files & Components

### 🔄 Data Preparation & Splits

#### `deduplicated_indices.csv`
- **Purpose**: Master index file for train-test split
- **Usage**: Used to ensure consistent data splits across different experiments
- **Content**: Story indices after deduplication process

#### `trope_extraction_train_test_split.py`
- **Purpose**: Extracts tropes based on the train-test split
- **Function**: Processes stories using the deduplicated indices for consistent splits
- **Output**: Creates separate train/test trope extraction files

#### `train_stories_deduplicated.csv`
- **Purpose**: Training set story indices
- **Created by**: `trope_extraction_train_test_split.py`
- **Usage**: Defines which stories are used for model training

#### `test_stories_deduplicated.csv`
- **Purpose**: Test set story indices  
- **Created by**: `trope_extraction_train_test_split.py`
- **Usage**: Defines which stories are used for model evaluation

### 🏷️ Trope Categories & Features

#### `trope_categories_filtered.json`
- **Purpose**: Filtered trope dictionary organized by categories
- **Structure**: `{category_name: [list_of_tropes]}`
- **Categories**: crime_theme, financial_motive, setting_ambience, authority_investigation, etc.
- **Usage**: Core vocabulary for proximity feature computation

#### `final_categories_with_tfidf.csv`
- **Purpose**: Filtered n-grams per category with TF-IDF scores
- **Content**: N-gram terms, their categories, and importance scores
- **Usage**: Analysis of trope importance and category relationships

### 🤖 Classification System

#### `pure_alias_classifier.py`
- **Purpose**: Main trope classifier training and prediction system
- **Features**:
  - Pure alias-only character extraction from metadata
  - Proximity-based feature computation using categorized tropes
  - Multi-word culprit scoring methods (last_word, avg, max)
  - SMOTE for class balancing
  - Comprehensive evaluation metrics
- **Key Classes**:
  - `PureAliasCharacterClassifier`: Main classifier class
  - Proximity feature computation functions
  - Training and evaluation pipeline

#### `pure_alias_classifier_last_word_with_weights.pkl`
- **Purpose**: Trained classifier model (pickled)
- **Method**: Uses 'last_word' scoring for multi-word culprits
- **Features**: Includes category discriminative weights
- **Usage**: Load with pickle for story analysis and predictions

### 🖥️ User Interface

#### `main.py`
- **Purpose**: Streamlit web application for interactive story analysis
- **Features**:
  - Story selection with priority stories support
  - Pure alias trope analysis with category weights
  - Culprit prediction and scoring
  - Trope dictionary reference
  - LLM integration for reasoning

#### `priority_stories.csv`
- **Purpose**: Curated list of high-performing stories for UI
- **Content**: Story indices and titles of best-performing stories
- **Usage**: Prioritizes important stories in the dropdown selection

## 🔄 Workflow

```
1. Data Preparation
   deduplicated_indices.csv → trope_extraction_train_test_split.py
   ↓
   train_stories_deduplicated.csv + test_stories_deduplicated.csv

2. Feature Engineering
   trope_categories_filtered.json + final_categories_with_tfidf.csv
   ↓
   Categorized proximity features

3. Model Training
   pure_alias_classifier.py → pure_alias_classifier_last_word_with_weights.pkl

4. Interactive Analysis
   main.py (Streamlit UI) + priority_stories.csv
```

## 🚀 Quick Start

### Training the Model
```bash
python pure_alias_classifier.py
```

### Running the UI
```bash
streamlit run main.py
```

### Requirements
- Python 3.8+
- Dependencies: `sklearn`, `streamlit`, `datasets`, `pandas`, `numpy`, `imblearn`

## 📊 Performance

- **ROC-AUC**: 0.626 (better than random baseline)
- **PR-AUC**: 0.079 (challenging due to 3.5% positive rate)
- **Recall**: 41% (finds less than half of culprits)
- **Precision**: 12% (high false positive rate)

## 🎯 Key Features

### Pure Alias Approach
- Uses only aliases from `metadata['name_id_map']`
- No synthetic name generation or complex filtering
- Focus on actual character references in text

### Categorized Proximity Features
- 20 trope categories with 151 total tropes
- Three feature types per category: min_dist, count_within_50, kernel_sum
- Category-specific discriminative weights

### Multi-word Culprit Scoring
- **last_word**: Uses only the last word for scoring (best performing)
- **avg**: Averages scores across all words
- **max**: Takes maximum score across all words

## 🔧 Configuration

Key parameters in `pure_alias_classifier.py`:
- `scoring_method`: 'last_word', 'avg', or 'max'
- `categories_file`: Path to trope categories JSON
- SMOTE balancing (disabled if <6 minority samples)

## 📈 Evaluation Metrics

The system provides comprehensive evaluation including:
- Precision-Recall curves
- ROC curves
- Score distribution analysis
- Feature importance by category
- Classification reports
- Ranking metrics (AUC, Average Precision)

## 🎨 UI Features

- **Priority Stories**: Curated high-performing stories shown first
- **Category Weights**: Display of discriminative importance per trope category
- **Trope Dictionary**: Expandable reference of all categories and tropes
- **Interactive Analysis**: Real-time culprit scoring and reasoning
- **Story Index Display**: Shows original dataset indices

---

*For detailed technical documentation, see the docstrings in individual Python files.*