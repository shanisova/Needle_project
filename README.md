# Needle Project - Mystery Story Character Classification

A machine learning system for identifying culprits in mystery stories using proximity-based trope analysis and pure alias character extraction.

## 🎯 Project Overview

This project implements a character classification system that:
- Extracts character candidates from story metadata using pure alias-only approach
- Uses proximity features based on categorized mystery tropes
- Trains a logistic regression classifier to identify story culprits
- Provides a Streamlit UI for interactive story analysis

## 📁 Key Files & Components

### 👥 Character Data & Graph Analysis

#### `character_extraction.py`
- **Purpose**: Extract character names from mystery stories using Ollama LLM
- **Features**:
  - Uses structured output with Pydantic models for reliable character extraction
  - Batched processing with timeout handling for long stories
  - Fallback to simple extraction if structured output fails
  - Saves results to CSV format in `/char/` directory
- **Arguments**:
  - `-s, --story-index`: Index of story to analyze (default: 0)
  - `-c, --batch-size`: Chunk size in characters (default: 2000)
  - `-m, --model`: Ollama model name (default: "llama3.2")
  - `-t, --timeout`: Timeout per chunk in seconds (default: 10)
- **Usage**: `python character_extraction.py -s 0 -c 2000 -m llama3.2 -t 10`

#### `alias_builder.py`
- **Purpose**: Build canonical character aliases from extracted character names
- **Features**:
  - Filters out pronouns, non-capitalized names, places, and fragments
  - Groups characters by surname compatibility and title matching
  - Uses containment rules and initial overlap for merging
  - Outputs both JSON and CSV formats
- **Arguments**:
  - `--input`: Input CSV file with character names
  - `--json-out`: Output JSON file path
  - `--csv-out`: Output CSV file path
  - `--drop-name`: Names to explicitly drop (can be used multiple times)
- **Usage**: `python alias_builder.py --input Story_chars.csv --json-out aliases.json --csv-out aliases.csv`

#### `interaction_extraction.py`
- **Purpose**: Extract character interactions from story text using canonical names
- **Features**:
  - Maps aliases to canonical names from alias builder output
  - Finds co-occurrences of characters in same sentences
  - Counts interaction frequencies
  - Outputs interaction matrix as CSV
- **Arguments**:
  - `--story`: Path to story text file
  - `--aliases`: Path to aliases JSON or CSV file
  - `--output`: Output CSV file path (optional)
- **Usage**: `python interaction_extraction.py --story story.txt --aliases aliases.json`

#### `plot_connection_graph.py`
- **Purpose**: Create connection graphs showing character relationships and victim connections
- **Features**:
  - Builds weighted graphs from interaction data
  - Computes PageRank centrality scores
  - Highlights victim connections and importance
  - Exports node metrics and visualization
- **Arguments**:
  - `story_index`: Story index in WhoDunIt dataset (required)
  - `--out`: Output PNG path (optional, defaults to story directory)
  - `--victim-list`: Custom victim list CSV (optional)
- **Usage**: `python plot_connection_graph.py 0 --out graph.png --victim-list victims.csv`

#### `run_pipeline.py`
- **Purpose**: Run complete character analysis pipeline for a single story
- **Features**:
  - Orchestrates character extraction, alias building, and interaction extraction
  - Handles file organization and error management
  - Creates organized output directory structure
- **Arguments**:
  - `story_index`: Story index to process (required)
  - `--output-dir`: Base output directory (default: "out")
  - `--model`: Ollama model for character extraction (default: "llama3.2")
- **Usage**: `python run_pipeline.py 0 --output-dir out --model llama3.2`

#### `run_full_dataset.py`
- **Purpose**: Process entire WhoDunIt dataset through character analysis pipeline
- **Features**:
  - Batch processing of all stories
  - Progress tracking and error handling
  - Configurable parallel processing
- **Arguments**:
  - `--start-index`: Starting story index (default: 0)
  - `--end-index`: Ending story index (default: all stories)
  - `--output-dir`: Base output directory (default: "out")
  - `--model`: Ollama model name (default: "llama3.2")
- **Usage**: `python run_full_dataset.py --start-index 0 --end-index 100 --output-dir out`

#### `plot_combined_culprit_analysis.py`
- **Purpose**: Comprehensive statistical analysis of character metrics across all stories
- **Features**:
  - Combines character data from all processed stories
  - Generates box plots with log transformation and intelligent outlier removal
  - Performs statistical tests (ANOVA/Kruskal-Wallis) with post-hoc analysis
  - Computes rank-based evaluation metrics for culprit identification
  - Creates metric distribution plots for visual analysis
  - Applies intelligent outlier removal (IQR for most metrics, percentile for victim connections)
- **Arguments**:
  - `--use-saved-data`: Use existing combined data CSV (faster)
  - `--output-dir`: Output directory for plots (default: "culprit_analysis_results")
- **Usage**: `python plot_combined_culprit_analysis.py --use-saved-data`
- **Statistical Tests**:
  - **Normality Testing**: Shapiro-Wilk test for each group
  - **Main Tests**: One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric)
  - **Post-hoc**: Tukey's HSD (ANOVA) or Mann-Whitney U (Kruskal-Wallis)
  - **Key Analysis**: Tests if culprits are significantly different from victims and others

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

### Character Data Pipeline
```
1. Character Extraction
   Story Text → character_extraction.py → Character Names CSV
   
2. Alias Building
   Character Names → alias_builder.py → Canonical Aliases (JSON/CSV)
   
3. Interaction Extraction
   Story Text + Aliases → interaction_extraction.py → Interaction Matrix CSV
   
4. Graph Visualization
   Interaction Matrix → plot_connection_graph.py → Network Graph PNG + Node Metrics CSV
   
5. Full Pipeline
   run_pipeline.py → Complete character analysis for single story
   run_full_dataset.py → Batch processing of entire dataset
   
6. Statistical Analysis
   Combined Character Data → plot_combined_culprit_analysis.py → Statistical Tests + Visualizations
```

### Trope Classification Pipeline
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

### Option 1: Use Pre-existing Data (Recommended)
If you have the required data files from git, you can jump directly to:

#### Run Statistical Analysis
```bash
python plot_combined_culprit_analysis.py --use-saved-data
```

#### Running the UI
```bash
python3 -m streamlit run main.py
```

### Option 2: Complete Pipeline from Scratch
If you want to generate all data yourself, follow this complete workflow:

#### Step 1: Install Dependencies
```bash
pip3 install -r requirements.txt
```

#### Step 2: Set up Ollama (for character extraction)
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull the required model
ollama pull llama3.2
```

#### Step 3: Character Data Pipeline
```bash
# Process a single story (test run)
python run_pipeline.py 0 --output-dir out --model llama3.2

# Process entire dataset (this will take several hours)
python run_full_dataset.py --start-index 0 --end-index 100 --output-dir out --model llama3.2
```

#### Step 4: Train the Classification Model
```bash
python pure_alias_classifier.py
```

#### Step 5: Run Statistical Analysis
```bash
python plot_combined_culprit_analysis.py --use-saved-data
```

#### Step 6: Launch the UI
```bash
python3 -m streamlit run main.py
```

### ⏱️ Time Estimates & Important Notes

#### Processing Times (for complete dataset):
- **Character Extraction**: ~4-6 hours (depends on Ollama model speed)
- **Alias Building**: ~30 minutes
- **Interaction Extraction**: ~2-3 hours
- **Graph Visualization**: ~1 hour
- **Model Training**: ~15-30 minutes
- **Statistical Analysis**: ~5-10 minutes

#### Prerequisites for Full Pipeline:
1. **Ollama Installation**: Required for character extraction
2. **Sufficient Storage**: ~2-3 GB for all processed data
3. **Memory**: 8GB+ RAM recommended for large dataset processing
4. **Patience**: Full dataset processing takes 6-10 hours total

#### Recommended Approach:
1. **Start Small**: Test with `run_pipeline.py 0` first
2. **Use Pre-existing Data**: If available, skip to Step 5-6
3. **Batch Processing**: Use `run_full_dataset.py` for efficiency
4. **Monitor Progress**: Check output directories for intermediate results

#### Troubleshooting:
- **Ollama Issues**: Ensure Ollama is running (`ollama serve`)
- **Memory Errors**: Process smaller batches or increase system memory
- **Missing Dependencies**: Run `pip3 install -r requirements.txt`
- **Streamlit Issues**: Use `python3 -m streamlit run main.py`

### Individual Component Usage

#### Extract Characters from a Story
```bash
python character_extraction.py -s 0 -m llama3.2
```

#### Build Character Aliases
```bash
python alias_builder.py --input char/Story_chars.csv --json-out aliases.json --csv-out aliases.csv
```

#### Extract Character Interactions
```bash
python interaction_extraction.py --story story.txt --aliases aliases.json
```

#### Create Character Connection Graph
```bash
python plot_connection_graph.py 0 --out graph.png
```

#### Run Complete Pipeline for One Story
```bash
python run_pipeline.py 0 --output-dir out --model llama3.2
```

#### Process Entire Dataset
```bash
python run_full_dataset.py --start-index 0 --end-index 100
```

#### Run Statistical Analysis
```bash
python plot_combined_culprit_analysis.py --use-saved-data
```

#### Training the Model
```bash
python pure_alias_classifier.py
```

#### Running the UI
```bash
python3 -m streamlit run main.py
```

### Requirements
- Python 3.8+
- Ollama installed and running (for character extraction)
- Dependencies: See `requirements.txt` for complete list including:
  - Character Analysis: `networkx`, `matplotlib`, `seaborn`, `ollama`, `pydantic`
  - Machine Learning: `sklearn`, `imblearn`, `numpy`, `pandas`
  - Statistical Analysis: `statsmodels`, `scipy`
  - UI: `streamlit`
  - NLP: `spacy`, `nltk`

### 📥 Required Data Files
To use the project smoothly, you need to obtain the following data files from git:

#### Character Data Files
- **`out/` directory**: Contains processed character data from all stories
  - `*/node_metrics.csv`: PageRank and centrality scores for each character
  - `*/Story Title_chars.csv`: Extracted character names
  - `*/Story Title_aliases.csv`: Canonical character aliases
  - `*/Story Title_interactions.csv`: Character interaction matrices

#### Pre-trained Models & Data
- **`pure_alias_classifier_last_word_with_weights.pkl`**: Trained classifier model
- **`trope_categories_filtered.json`**: Filtered trope dictionary by categories
- **`final_categories_with_tfidf.csv`**: N-grams with TF-IDF scores
- **`priority_stories.csv`**: Curated high-performing stories for UI
- **`victim_list.csv`**: Victim character mappings
- **`culprit_analysis_results/combined_character_data.csv`**: Combined character data for statistical analysis

#### Data Split Files
- **`deduplicated_indices.csv`**: Master index for train-test split
- **`train_stories_deduplicated.csv`**: Training set story indices
- **`test_stories_deduplicated.csv`**: Test set story indices

**Note**: These files are essential for running the character analysis pipeline, statistical tests, and the Streamlit UI. Without them, the scripts will not function properly.

## 📊 Performance

### Trope Classification Performance
- **ROC-AUC**: 0.626 (better than random baseline)
- **PR-AUC**: 0.079 (challenging due to 3.5% positive rate)
- **Recall**: 41% (finds less than half of culprits)
- **Precision**: 12% (high false positive rate)

### Statistical Analysis Results
The character metrics show **highly significant differences** between groups:

#### Statistical Significance (p < 0.0001 for all metrics):
- **PageRank**: Kruskal-Wallis p=0.0000 ✅
- **Victim Connection Weight**: Kruskal-Wallis p=0.0000 ✅  
- **Strength**: Kruskal-Wallis p=0.0000 ✅
- **Degree**: Kruskal-Wallis p=0.0000 ✅

#### Key Findings:
- **Culprits vs Others**: All metrics show culprits significantly higher (p=0.0000)
- **Culprits vs Victims**: Significant differences across all metrics
- **Victims vs Others**: Victims consistently higher than others
- **Rank-based Evaluation**: PageRank shows best culprit identification performance

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

## 📁 Output Data Structure

### Character Analysis Outputs
The character analysis pipeline creates organized output directories:

```
out/
├── Story_Title_Index/
│   ├── Story Title_chars.csv          # Extracted character names
│   ├── Story Title_aliases.csv        # Canonical aliases mapping
│   ├── Story Title_interactions.csv   # Character interaction matrix
│   ├── graph_victim_connections.png   # Network visualization
│   └── node_metrics.csv              # PageRank and centrality scores
```

### File Formats

#### Character CSV (`*_chars.csv`)
- `char_id`: Unique character identifier
- `name`: Character name as extracted from text
- `story_title`: Source story title

#### Aliases CSV (`*_aliases.csv`)
- `canonical_name`: Primary character name
- `alias_name`: Alternative name/variant

#### Interactions CSV (`*_interactions.csv`)
- `character1`: First character in interaction
- `character2`: Second character in interaction
- `count`: Number of co-occurrences in same sentences

#### Node Metrics CSV (`node_metrics.csv`)
- `node`: Character name
- `is_victim`: Binary indicator (1 if victim, 0 otherwise)
- `pagerank`: PageRank centrality score
- `victim_connection_weight`: Total interaction weight with victims
- `degree`: Number of connections
- `strength`: Sum of all connection weights

## 🔧 Configuration

### Character Extraction Settings
- **Model**: Choose Ollama model (`llama3.2`, `gemma3:12b`, etc.)
- **Batch Size**: Text chunk size for processing (default: 2000 chars)
- **Timeout**: Per-chunk timeout in seconds (default: 10s)

### Graph Visualization Settings
- **Layout**: Spring layout with weight-based positioning
- **Node Sizing**: Based on victim connections or PageRank
- **Colors**: Red for victims, blue intensity for others
- **Edge Width**: Proportional to interaction frequency

---

*For detailed technical documentation, see the docstrings in individual Python files.*