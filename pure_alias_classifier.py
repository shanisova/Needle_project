#!/usr/bin/env python3
"""
Pure alias-only character classifier with different scoring methods for multi-word culprits.
Uses ONLY the values from metadata['name_id_map'] as character candidates.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import re
import ast
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def normalize_text(text):
    """
    Normalize text for better matching:
    - Lowercase
    - Replace hyphens with spaces
    - Drop dots in titles (Mr., Dr., etc.)
    - Normalize apostrophes
    """
    if not text:
        return ""
    
    text = text.lower()
    # Replace different types of hyphens and dashes with spaces
    text = re.sub(r'[-–—]', ' ', text)
    # Drop dots (especially in titles like Mr., Dr., etc.)
    text = re.sub(r'\.', '', text)
    # Normalize apostrophes
    text = text.replace(''', "'").replace(''', "'")
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def strip_possessive(token):
    """Strip possessive endings from tokens ('s, 's)"""
    import re
    return re.sub(r"(?:'s|'s)$", "", token)

def token_positions(tokens, phrase_tokens):
    """
    Find all positions where phrase_tokens appears in tokens.
    Allow for punctuation between words for more flexible matching.
    Handle possessives by treating "barrymore's" as matching "barrymore".
    """
    if not phrase_tokens:
        return []
    
    L = len(phrase_tokens)
    pos = []
    
    # For single word phrases, do exact matching with possessive handling
    if L == 1:
        target = phrase_tokens[0]
        for i, token in enumerate(tokens):
            # Check exact match or possessive match
            if token == target or strip_possessive(token) == target:
                pos.append(i)
    else:
        # For multi-word phrases, allow some flexibility with punctuation
        for i in range(len(tokens) - L + 1):
            # Check if we have a match allowing for punctuation between words
            match = True
            for j, target_token in enumerate(phrase_tokens):
                if i + j >= len(tokens):
                    match = False
                    break
                current_token = tokens[i + j]
                # Check exact match or possessive match
                if current_token != target_token and strip_possessive(current_token) != target_token:
                    match = False
                    break
            
            if match:
                pos.append(i)
    
    return pos


def compute_categorized_proximity_features(text, aliases, categorized_tropes):
    """
    Compute proximity features for each trope category.
    Returns a feature vector with category-specific proximity scores.
    Uses normalized text for better matching.
    """
    # Normalize and tokenize text
    normalized_text = normalize_text(text)
    toks = re.findall(r"[\w']+|[^\w\s]", normalized_text)
    
    # Find all candidate positions using normalized aliases
    cand_pos = []
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        alias_tokens = normalized_alias.split()
        positions = token_positions(toks, alias_tokens)
        cand_pos.extend(positions)
    
    if not cand_pos:
        # Return zero features for all categories
        features = {}
        for category in categorized_tropes.keys():
            features[f'{category}_min_dist'] = 999999
            features[f'{category}_count_within_50'] = 0
            features[f'{category}_kernel_sum'] = 0.0
        return features
    
    # Compute features for each category
    features = {}
    
    for category, tropes in categorized_tropes.items():
        # Find all positions for this category's tropes using normalized text
        category_trope_pos = []
        for trope in tropes:
            normalized_trope = normalize_text(trope)
            trope_tokens = normalized_trope.split()
            positions = token_positions(toks, trope_tokens)
            category_trope_pos.extend(positions)
        
        if not category_trope_pos:
            features[f'{category}_min_dist'] = 999999
            features[f'{category}_count_within_50'] = 0
            features[f'{category}_kernel_sum'] = 0.0
            continue
        
        # Compute features for each candidate position, then take best
        min_distances = []
        counts_within_50 = []
        kernel_sums = []
        
        for cpos in cand_pos:
            # Distance to nearest trope in this category
            distances = [abs(cpos - tpos) for tpos in category_trope_pos]
            min_dist = min(distances)
            min_distances.append(min_dist)
            
            # Count within 50 tokens
            count_50 = sum(1 for d in distances if d <= 50)
            counts_within_50.append(count_50)
            
            # Gaussian kernel sum (sigma=80)
            kernel_sum = sum(np.exp(-(d/80.0)**2) for d in distances)
            kernel_sums.append(kernel_sum)
        
        # Take best across candidate positions
        features[f'{category}_min_dist'] = min(min_distances)
        features[f'{category}_count_within_50'] = max(counts_within_50)
        features[f'{category}_kernel_sum'] = max(kernel_sums)
    
    return features


def extract_pure_aliases_from_metadata(metadata):
    """
    Extract character names using PURE ALIAS-ONLY approach.
    - Use ONLY the aliases (values) from name_id_map
    - No filtering, no synthesis - just the raw aliases
    """
    if not metadata or 'name_id_map' not in metadata:
        return {}
    
    name_map = metadata['name_id_map']
    candidates = {}
    
    # Simply collect all aliases (values) and normalize them
    for original, replaced in name_map.items():
        alias = replaced.strip()
        if len(alias) >= 1:
            normalized_alias = normalize_text(alias)
            if normalized_alias:
                candidates[normalized_alias] = [normalized_alias]
    
    return candidates


class PureAliasCharacterClassifier:
    def __init__(self, categories_file='trope_categories_filtered.json', scoring_method='last_word'):
        self.categories_file = categories_file
        self.categorized_tropes = None
        self.classifier = None
        self.scaler = None
        self.feature_names = None
        self.scoring_method = scoring_method  # 'last_word', 'avg', or 'max'
        
    def safe_eval_metadata(self, metadata_str):
        if not metadata_str or metadata_str == 'None':
            return None
        try:
            if isinstance(metadata_str, dict):
                return metadata_str
            return ast.literal_eval(metadata_str)
        except:
            return None
    
    def load_categorized_tropes(self):
        """Load the categorized trope vocabulary."""
        with open(self.categories_file, 'r') as f:
            self.categorized_tropes = json.load(f)
        
        total_tropes = sum(len(tropes) for tropes in self.categorized_tropes.values())
        print(f"Loaded {len(self.categorized_tropes)} categories with {total_tropes} total tropes")
        
        # Create feature names
        self.feature_names = []
        for category in self.categorized_tropes.keys():
            self.feature_names.extend([
                f'{category}_min_dist',
                f'{category}_count_within_50',
                f'{category}_kernel_sum'
            ])
        
        print(f"Generated {len(self.feature_names)} category-specific features")
        return self.categorized_tropes
    
    def compute_culprit_score(self, text, candidate):
        """Compute culprit score using different methods for multi-word names."""
        if self.scoring_method == 'last_word':
            # Use only the last word of the candidate for scoring
            candidate_words = candidate.split()
            last_word = candidate_words[-1] if candidate_words else candidate
            return compute_categorized_proximity_features(text, [last_word], self.categorized_tropes)
        
        elif self.scoring_method == 'avg':
            # Average the scores of all words in the candidate
            candidate_words = candidate.split()
            if len(candidate_words) <= 1:
                return compute_categorized_proximity_features(text, [candidate], self.categorized_tropes)
            
            word_features = []
            for word in candidate_words:
                features = compute_categorized_proximity_features(text, [word], self.categorized_tropes)
                word_features.append(features)
            
            # Average across all words
            avg_features = {}
            for feature_name in self.feature_names:
                values = [f.get(feature_name, 0) for f in word_features]
                avg_features[feature_name] = np.mean(values)
            return avg_features
            
        elif self.scoring_method == 'max':
            # Take the maximum score across all words in the candidate
            candidate_words = candidate.split()
            if len(candidate_words) <= 1:
                return compute_categorized_proximity_features(text, [candidate], self.categorized_tropes)
            
            word_features = []
            for word in candidate_words:
                features = compute_categorized_proximity_features(text, [word], self.categorized_tropes)
                word_features.append(features)
            
            # Take maximum across all words
            max_features = {}
            for feature_name in self.feature_names:
                values = [f.get(feature_name, 0) for f in word_features]
                max_features[feature_name] = np.max(values)
            return max_features
        
        else:
            # Default: use full candidate
            return compute_categorized_proximity_features(text, [candidate], self.categorized_tropes)
    
    def is_culprit_match(self, candidate, aliases, culprit_names):
        """Simple culprit matching - culprits are already aliases."""
        candidate_tokens = set(candidate.split())
        
        for culprit in culprit_names:
            culprit_tokens = set(culprit.split())
            
            # Exact match
            if culprit == candidate:
                return True
            
            # Substring match (either direction)
            if culprit in candidate or candidate in culprit:
                return True
                
            # Check if candidate aliases match culprit
            for alias in aliases:
                if culprit in alias or alias in culprit:
                    return True
            
            # For multi-word names, check token overlap
            if len(candidate_tokens) > 1 and len(culprit_tokens) > 1:
                overlap = len(candidate_tokens & culprit_tokens)
                if overlap >= 2:  # At least 2 tokens in common
                    return True
                # Or if one name contains most tokens of the other
                if overlap >= min(len(candidate_tokens), len(culprit_tokens)) * 0.7:
                    return True
        
        return False
    
    def prepare_training_data(self, stories, story_indices):
        """Prepare training data using PURE ALIAS-ONLY character names from metadata."""
        print(f"Preparing PURE ALIAS-ONLY proximity training data...")
        print(f"📋 Using ONLY aliases from metadata name_id_map (scoring: {self.scoring_method})!")
        
        if not self.categorized_tropes:
            self.load_categorized_tropes()
        
        X_rows = []
        y = []
        candidate_info = []
        
        for i, story_idx in enumerate(story_indices):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(story_indices)}")
            
            story = stories[story_idx]
            text = story.get('text', '')
            metadata = self.safe_eval_metadata(story.get('metadata', {}))
            culprit_ids = story.get('culprit_ids', [])
            
            if isinstance(culprit_ids, str):
                try:
                    culprit_ids = ast.literal_eval(culprit_ids)
                except:
                    culprit_ids = []
            
            # Extract character names using pure alias-only approach
            candidate_aliases = extract_pure_aliases_from_metadata(metadata)
            if not candidate_aliases:
                continue
            
            # Normalize culprit names for matching
            culprit_names = [normalize_text(name) for name in culprit_ids]
            
            for candidate, aliases in candidate_aliases.items():
                # Compute categorized proximity features using the specified scoring method
                features = self.compute_culprit_score(text, candidate)
                
                # Create feature vector (ordered by feature_names)
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0))
                
                # Simple culprit matching
                is_culprit = self.is_culprit_match(candidate, aliases, culprit_names)
                
                X_rows.append(feature_vector)
                y.append(1 if is_culprit else 0)
                candidate_info.append({
                    'story_idx': story_idx,
                    'candidate': candidate,
                    'aliases': aliases,
                    'is_culprit': is_culprit,
                    'features': features
                })
        
        X = np.array(X_rows)
        y = np.array(y)
        
        print(f"Prepared {len(X)} examples with {X.shape[1]} categorized features")
        print(f"Class distribution: {np.bincount(y)} (culprits: {np.sum(y)}, non-culprits: {len(y) - np.sum(y)})")
        print(f"Culprit percentage: {np.sum(y)/len(y)*100:.1f}%")
        
        return X, y, candidate_info
    
    def train(self, X, y):
        """Train the character classifier with SMOTE for balancing."""
        print(f"\nTraining PURE ALIAS-ONLY character classifier ({self.scoring_method})...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE for balancing (only if we have enough samples)
        minority_class_count = min(np.bincount(y))
        if minority_class_count >= 6:  # SMOTE needs at least 6 samples for k=5 neighbors
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
        else:
            print(f"⚠️  Skipping SMOTE: only {minority_class_count} minority samples (need ≥6)")
            X_balanced, y_balanced = X_scaled, y
        
        print(f"After SMOTE: {len(X_balanced)} examples")
        print(f"Balanced distribution: {np.bincount(y_balanced)}")
        
        # Train logistic regression
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(X_balanced, y_balanced)
        
        # Compute category weights from learned coefficients
        self._compute_category_weights()
        
        # Show top feature importance
        print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES ({self.scoring_method.upper()}):")
        feature_importance = [(name, coef) for name, coef in zip(self.feature_names, self.classifier.coef_[0])]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature_name, coef) in enumerate(feature_importance[:10]):
            direction = "CULPRIT" if coef > 0 else "NON-CULPRIT"
            print(f"   {i+1:2d}. {feature_name:<40}: {coef:7.4f} ({direction})")
        
        # Show category weights
        if hasattr(self, 'category_weights') and self.category_weights:
            print(f"\n📊 CATEGORY DISCRIMINATIVE WEIGHTS:")
            sorted_weights = sorted(self.category_weights.items(), key=lambda x: x[1], reverse=True)
            for i, (category, weight) in enumerate(sorted_weights[:10]):
                print(f"   {i+1:2d}. {category:<30}: {weight:.3f}")
        
        return self.classifier
    
    def _compute_category_weights(self):
        """Compute per-category discriminative weights from learned coefficients."""
        if not self.classifier or not hasattr(self.classifier, 'coef_'):
            return
        
        # Build coefficient map
        coef_map = {name: coef for name, coef in zip(self.feature_names, self.classifier.coef_[0])}
        
        # Aggregate by category (use absolute magnitude for discriminativeness)
        cat_scores = {}
        for name, coef in coef_map.items():
            # Feature names look like "{category}_{min_dist|count_within_50|kernel_sum}"
            category = name.rsplit('_', 2)[0]
            cat_scores.setdefault(category, []).append(abs(coef))
        
        # Average and normalize to mean 1.0
        raw_weights = {cat: (sum(vals) / len(vals)) for cat, vals in cat_scores.items()}
        mean_w = sum(raw_weights.values()) / max(1, len(raw_weights))
        self.category_weights = {cat: (w / mean_w if mean_w > 0 else 1.0) for cat, w in raw_weights.items()}
        
        print(f"\n✅ Computed category weights (mean normalized to 1.0)")
    
    def predict_all_candidates(self, story_data):
        """Predict scores for all character candidates from metadata."""
        if not self.classifier or not self.scaler:
            print("Model not trained yet")
            return {}, {}
        
        text = story_data.get('text', '')
        metadata = self.safe_eval_metadata(story_data.get('metadata', {}))
        
        # Extract character names using pure alias-only approach
        candidate_aliases = extract_pure_aliases_from_metadata(metadata)
        if not candidate_aliases:
            return {}, {}
        
        candidate_scores = {}
        candidate_details = {}
        
        for candidate, aliases in candidate_aliases.items():
            # Compute features using the specified scoring method
            features = self.compute_culprit_score(text, candidate)
            
            # Create feature vector
            feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])
            
            # Scale and predict
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prob = self.classifier.predict_proba(feature_vector_scaled)[0, 1]
            
            candidate_scores[candidate] = prob
            candidate_details[candidate] = {
                'score': prob,
                'features': features,
                'aliases': aliases
            }
        
        return candidate_scores, candidate_details
    
    def evaluate_ranker(self, stories, story_indices):
        """Evaluate the ranker on stories."""
        print(f"Evaluating PURE ALIAS-ONLY character ranker ({self.scoring_method})...")
        
        all_scores = []
        all_labels = []
        
        for story_idx in story_indices:
            story = stories[story_idx]
            culprit_ids = story.get('culprit_ids', [])
            
            if isinstance(culprit_ids, str):
                try:
                    culprit_ids = ast.literal_eval(culprit_ids)
                except:
                    culprit_ids = []
            
            candidate_scores, candidate_details = self.predict_all_candidates(story)
            if not candidate_scores:
                continue
            
            culprit_names = [normalize_text(name) for name in culprit_ids]
            
            for candidate, score in candidate_scores.items():
                aliases = candidate_details[candidate]['aliases']
                is_culprit = self.is_culprit_match(candidate, aliases, culprit_names)
                
                all_scores.append(score)
                all_labels.append(1 if is_culprit else 0)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
            ap = average_precision_score(all_labels, all_scores)
            
            print(f"AUC: {auc:.3f}")
            print(f"Average Precision: {ap:.3f}")
            print(f"Total examples: {len(all_labels)}")
            print(f"Culprits: {np.sum(all_labels)} ({np.sum(all_labels)/len(all_labels)*100:.1f}%)")
            return auc, ap
        else:
            print("Cannot compute AUC - only one class present")
            return 0, 0


def create_evaluation_plots(classifier, X_test, y_test, y_scores, method_name):
    """Create comprehensive evaluation plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Classifier Evaluation: {method_name.upper()}', fontsize=16, fontweight='bold')
    
    # 1. PR Curve (top-left)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    axes[0, 0].plot(recall, precision, color='darkorange', lw=2, 
                    label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 0].fill_between(recall, precision, alpha=0.2, color='darkorange')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].legend(loc="lower left")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve (top-right)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkblue', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.7)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.2, color='darkblue')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Score Distribution Boxplot (bottom-left)
    culprit_scores = y_scores[y_test == 1]
    non_culprit_scores = y_scores[y_test == 0]
    
    box_data = [non_culprit_scores, culprit_scores]
    box_labels = ['Non-Culprit', 'Culprit']
    colors = ['lightcoral', 'lightgreen']
    
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_ylabel('Prediction Score')
    axes[1, 0].set_title('Score Distribution: Culprit vs Non-Culprit')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add statistics
    culprit_mean = np.mean(culprit_scores) if len(culprit_scores) > 0 else 0
    non_culprit_mean = np.mean(non_culprit_scores) if len(non_culprit_scores) > 0 else 0
    axes[1, 0].text(0.02, 0.98, 
                    f'Non-Culprit μ: {non_culprit_mean:.4f}\nCulprit μ: {culprit_mean:.4f}', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Feature Importance (bottom-right)
    if hasattr(classifier, 'category_weights') and classifier.category_weights:
        categories = list(classifier.category_weights.keys())
        weights = list(classifier.category_weights.values())
        
        # Sort by importance
        sorted_items = sorted(zip(categories, weights), key=lambda x: x[1], reverse=True)
        categories, weights = zip(*sorted_items[:15])  # Top 15
        
        y_pos = np.arange(len(categories))
        bars = axes[1, 1].barh(y_pos, weights, color='steelblue', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([cat.replace('_', ' ').title() for cat in categories])
        axes[1, 1].set_xlabel('Discriminative Weight')
        axes[1, 1].set_title('Top Trope Categories (Feature Importance)')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            axes[1, 1].text(weight + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{weight:.2f}', ha='left', va='center', fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, 'Category weights not available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance (Not Available)')
    
    plt.tight_layout()
    return fig, (pr_auc, roc_auc)

def main():
    """Main training and evaluation function with comprehensive metrics."""
    from sklearn.metrics import classification_report
    
    print("🚀 COMPREHENSIVE CLASSIFIER TRAINING & EVALUATION")
    print("=" * 70)
    print("Pure Alias-Only Classifier with Category Weights & Evaluation Metrics")
    print()
    
    # Load dataset
    print("📚 Loading dataset...")
    dataset = load_dataset("kjgpta/WhoDunIt", split="train")
    print(f"Total stories: {len(dataset)}")
    
    # Load train/test split
    mapping_df = pd.read_csv('train_test_index_mapping.csv')
    train_indices = mapping_df[mapping_df['split'] == 'train']['original_index'].tolist()
    test_indices = mapping_df[mapping_df['split'] == 'test']['original_index'].tolist()
    
    print(f"✅ Using existing train/test split:")
    print(f"   Train stories: {len(train_indices)}")
    print(f"   Test stories: {len(test_indices)}")
    
    # Train with LAST_WORD method (best performing)
    method = 'last_word'
    print(f"\n🎯 Training Pure Alias Classifier ({method.upper()} method)...")
    classifier = PureAliasCharacterClassifier(scoring_method=method)
    
    # Prepare training data
    print("📊 Preparing training data...")
    X_train, y_train, train_info = classifier.prepare_training_data(dataset, train_indices)
    
    if len(X_train) == 0:
        print(f"❌ No training data prepared")
        return
    
    print(f"Training set: {len(X_train)} candidates from {len(train_indices)} stories")
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"Culprit rate: {np.mean(y_train):.3%}")
    
    # Train the classifier
    print("\n🔧 Training classifier...")
    classifier.train(X_train, y_train)
    
    # Prepare test data
    print("📊 Preparing test data...")
    X_test, y_test, test_info = classifier.prepare_training_data(dataset, test_indices)
    print(f"Test set: {len(X_test)} candidates from {len(test_indices)} stories")
    
    # Evaluate on test set
    print("\n📈 Evaluating model performance on test set...")
    y_scores = classifier.classifier.predict_proba(classifier.scaler.transform(X_test))[:, 1]
    
    # Create evaluation plots
    print("📊 Creating evaluation visualizations...")
    fig, (pr_auc, roc_auc) = create_evaluation_plots(classifier, X_test, y_test, y_scores, method)
    
    # Save the plot
    plot_filename = f'comprehensive_evaluation_{method}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Saved evaluation plot: {plot_filename}")
    
    # Print detailed metrics
    print(f"\n📊 TEST SET PERFORMANCE METRICS:")
    print(f"   PR-AUC:  {pr_auc:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    y_pred = (y_scores > 0.5).astype(int)
    print(f"\n📋 Classification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=['Non-Culprit', 'Culprit']))
    
    # Ranking evaluation
    print(f"\n🎯 RANKING EVALUATION:")
    test_auc, test_ap = classifier.evaluate_ranker(dataset, test_indices)
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Test AP:  {test_ap:.4f}")
    
    # Save the trained model with weights
    model_filename = f'pure_alias_classifier_{method}_with_weights.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"\n💾 Saved trained model: {model_filename}")
    
    # Show category weights for UI integration
    if hasattr(classifier, 'category_weights') and classifier.category_weights:
        print(f"\n🏷️  CATEGORY WEIGHTS (for UI integration):")
        sorted_weights = sorted(classifier.category_weights.items(), key=lambda x: x[1], reverse=True)
        for category, weight in sorted_weights:
            print(f"   {category:<30}: {weight:.3f}")
    
    # Test example predictions
    print(f"\n" + "=" * 50)
    print(f"EXAMPLE PREDICTIONS ON TEST SET")
    print("=" * 50)
    
    for i in range(min(3, len(test_indices))):
        story_idx = test_indices[i]
        story = dataset[story_idx]
        title = story.get('title', f'Story {story_idx}')
        culprits = story.get('culprit_ids', [])
        
        if isinstance(culprits, str):
            try:
                culprits = ast.literal_eval(culprits)
            except:
                culprits = []
        
        print(f"\n📚 {title}")
        print(f"   🎯 Actual culprits: {culprits}")
        
        scores, details = classifier.predict_all_candidates(story)
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print(f"   📊 Top 10 predictions:")
            for j, (candidate, score) in enumerate(sorted_scores[:10]):
                normalized_culprits = [normalize_text(str(c)) for c in culprits]
                aliases = details[candidate]['aliases']
                is_culprit = classifier.is_culprit_match(candidate, aliases, normalized_culprits)
                marker = "🎯" if is_culprit else "  "
                print(f"      {j+1:2d}. {marker} {candidate:<20}: {score:.3f}")
        else:
            print(f"   ❌ No predictions available")
    
    print(f"\n✅ Training complete! Model ready for integration.")
    return classifier


if __name__ == "__main__":
    main()
