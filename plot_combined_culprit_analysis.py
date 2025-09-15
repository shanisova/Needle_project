#!/usr/bin/env python3
"""
Combined Culprit Analysis - All Stories Combined

Creates comprehensive scatter plots combining all characters from all stories
to show the overall pattern of PageRank vs Victim Connection across the entire dataset.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re
from datasets import load_dataset
import json
import ast
import glob


def clean_dir_name(title: str) -> str:
    """Clean story title for directory naming"""
    return title.replace(' ', '_').replace("'", '').replace('"', '').replace(':', '').replace(';', '')


def load_alias_mapping(aliases_csv_path: Path) -> Dict[str, str]:
    """Load alias->canonical mapping from CSV"""
    df = pd.read_csv(aliases_csv_path)
    alias_to_can: Dict[str, str] = {}
    for _, row in df.iterrows():
        can = str(row['canonical_name'])
        al = str(row['alias_name']) if not (pd.isna(row['alias_name']) or row['alias_name'] == '') else None
        if al:
            alias_to_can[al] = can
        # also map canonical to itself
        alias_to_can.setdefault(can, can)
    return alias_to_can


def load_culprit_from_dataset(story_index: int) -> List[str]:
    """Load culprit information from the WhoDunIt dataset"""
    try:
        ds = load_dataset("kjgpta/WhoDunIt", split="train")
        story = ds[story_index]
        
        # Check for culprit information in various possible fields
        culprit_fields = ['culprit_ids', 'culprits', 'culprit', 'suspects']
        culprits = []
        
        for field in culprit_fields:
            if field in story and story[field] is not None:
                culprit_data = story[field]
                if isinstance(culprit_data, list):
                    culprits.extend([str(c) for c in culprit_data if c])
                elif isinstance(culprit_data, str) and culprit_data.strip():
                    # Try to parse string representation of list
                    try:
                        # Handle cases like "['muriel asha gauri naman']"
                        parsed = ast.literal_eval(culprit_data)
                        if isinstance(parsed, list):
                            culprits.extend([str(c) for c in parsed if c])
                        else:
                            culprits.append(culprit_data.strip())
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single string
                        culprits.append(culprit_data.strip())
                break
        
        return culprits
    except Exception as e:
        return []


def load_victim_from_list(story_title: str, victim_list_path: Path) -> List[str]:
    """Load victim information from victim_list.csv"""
    victims = []
    
    if not victim_list_path.exists():
        return victims
    
    try:
        df = pd.read_csv(victim_list_path)
        row = df.loc[df['Story'] == story_title]
        if row.empty:
            return victims
        
        # Get victim information from 'Victim(s)' column
        victims_field = str(row.iloc[0].get('Victim(s)', '') or '')
        
        if victims_field.strip().lower() in {'none', '', '—', '-'}:
            return victims
        
        # Parse the victim field (handle lists separated by semicolons or commas)
        def parse_list_field(text: str) -> List[str]:
            t = text.strip()
            t = t.strip('{}')
            parts = re.split(r"[;,]", t) if t else []
            cleaned: List[str] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Remove any (...) or [...] segments
                p = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", p).strip()
                if p:
                    cleaned.append(p)
            return cleaned
        
        victims = parse_list_field(victims_field)
        
    except Exception as e:
        pass
    
    return victims


def try_load_metadata_name_id_map(story_index: int) -> Dict[str, str]:
    """Load name_id_map from dataset metadata"""
    try:
        ds = load_dataset("kjgpta/WhoDunIt", split="train")
        meta = ds[story_index].get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                try:
                    meta = ast.literal_eval(meta)
                except Exception:
                    meta = {}
        if isinstance(meta, dict):
            nim = meta.get('name_id_map', {})
            if isinstance(nim, dict):
                return {str(k): str(v) for k, v in nim.items()}
    except Exception:
        pass
    return {}


def map_names_to_canonical(names: List[str], alias_to_can: Dict[str, str], node_metrics: pd.DataFrame = None, story_index: int = None) -> Set[str]:
    """Map a list of names to canonical names using alias mapping with improved matching"""
    canonical_names = set()
    
    # Build a punctuation-insensitive alias lookup
    def norm_alpha(s: str) -> str:
        return re.sub(r"[^A-Za-z]", "", (s or "")).lower()
    
    alias_norm_to_can = {norm_alpha(a): c for a, c in alias_to_can.items()}
    # Ensure canonicals also resolve to themselves
    for can in set(alias_to_can.values()):
        alias_norm_to_can.setdefault(norm_alpha(can), can)
    
    # Get all available node names for partial matching
    available_nodes = set(node_metrics['node'].tolist()) if node_metrics is not None else set()
    
    # Load metadata name_id_map if available
    name_id_map = {}
    if story_index is not None:
        name_id_map = try_load_metadata_name_id_map(story_index)
    
    for name in names:
        key = norm_alpha(name)
        can = alias_norm_to_can.get(key)
        
        if can:
            canonical_names.add(can)
        else:
            # Try metadata mapping first
            if name_id_map:
                # Try to map through metadata
                for meta_key, meta_value in name_id_map.items():
                    if norm_alpha(meta_key) == key or norm_alpha(meta_value) == key:
                        # Try to find the mapped name in aliases
                        mapped_key = norm_alpha(meta_value)
                        mapped_can = alias_norm_to_can.get(mapped_key)
                        if mapped_can:
                            canonical_names.add(mapped_can)
                            break
                else:
                    # If metadata mapping didn't work, try partial matching
                    name_parts = name.lower().split()
                    matched_parts = []
                    
                    for part in name_parts:
                        part_norm = norm_alpha(part)
                        # Look for exact match first
                        if part_norm in alias_norm_to_can:
                            matched_parts.append(alias_norm_to_can[part_norm])
                        else:
                            # Try partial matching in available nodes
                            for node in available_nodes:
                                if part_norm in norm_alpha(node):
                                    matched_parts.append(node)
                                    break
                    
                    if matched_parts:
                        canonical_names.update(matched_parts)
            else:
                # No metadata available, use partial matching
                name_parts = name.lower().split()
                matched_parts = []
                
                for part in name_parts:
                    part_norm = norm_alpha(part)
                    # Look for exact match first
                    if part_norm in alias_norm_to_can:
                        matched_parts.append(alias_norm_to_can[part_norm])
                    else:
                        # Try partial matching in available nodes
                        for node in available_nodes:
                            if part_norm in norm_alpha(node):
                                matched_parts.append(node)
                                break
                
                if matched_parts:
                    canonical_names.update(matched_parts)
    
    return canonical_names


def find_stories_with_node_metrics() -> List[Tuple[int, str, Path]]:
    """Find all stories that have node_metrics.csv files"""
    stories = []
    
    # Find all node_metrics.csv files
    node_metrics_files = glob.glob("character_data/*/node_metrics.csv")
    
    for metrics_file in node_metrics_files:
        metrics_path = Path(metrics_file)
        story_dir = metrics_path.parent
        
        # Extract story index from directory name
        dir_name = story_dir.name
        # Format: "Story_Title_Index"
        parts = dir_name.split('_')
        if len(parts) >= 2:
            try:
                story_index = int(parts[-1])
                story_title = '_'.join(parts[:-1]).replace('_', ' ')
                stories.append((story_index, story_title, story_dir))
            except ValueError:
                continue
    
    return sorted(stories, key=lambda x: x[0])


def plot_combined_pagerank_vs_victim_connection(all_data: pd.DataFrame, output_dir: Path):
    """Create combined scatter plot of all characters from all stories with better scaling"""
    
    # Create two versions: full scale and zoomed in
    for plot_type, xlim, ylim, suffix in [
        ('full', None, None, ''),
        ('zoomed', (0, 0.15), (0, 5), '_zoomed')
    ]:
        plt.figure(figsize=(14, 10))
        
        # Prepare data
        pagerank = all_data['pagerank'].tolist()
        victim_conn = all_data['victim_connection_weight'].tolist()
        roles = all_data['role'].tolist()
        stories = all_data['story_title'].tolist()
        
        # Color by role
        colors = []
        sizes = []
        for role in roles:
            if role == 'Culprit':
                colors.append('#2E8B57')  # Sea green for culprits
                sizes.append(100)  # Larger dots for culprits
            elif role == 'Victim':
                colors.append('#DC143C')  # Crimson for victims
                sizes.append(80)  # Medium dots for victims
            else:
                colors.append('#4682B4')  # Steel blue for others
                sizes.append(30)  # Smaller dots for others
        
        # Create scatter plot
        scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
        
        # Set axis limits for zoomed version
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.xlabel('PageRank', fontsize=14)
        plt.ylabel('Victim Connection Weight', fontsize=14)
        
        title = 'Combined Analysis: PageRank vs Victim Connection\n(All Characters from All Stories)'
        if plot_type == 'zoomed':
            title += '\n(Zoomed View - Main Data Cluster)'
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E8B57', label='Culprits'),
            Patch(facecolor='#DC143C', label='Victims'),
            Patch(facecolor='#4682B4', label='Other Characters')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        total_chars = len(all_data)
        culprits = len(all_data[all_data['role'] == 'Culprit'])
        victims = len(all_data[all_data['role'] == 'Victim'])
        others = len(all_data[all_data['role'] == 'Other'])
        
        stats_text = f'Total Characters: {total_chars}\nCulprits: {culprits}\nVictims: {victims}\nOthers: {others}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"combined_pagerank_vs_victim_connection{suffix}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  - {plot_type} view: {output_path}")
    
    return output_path


def plot_combined_pagerank_vs_victim_connection_log_scale(all_data: pd.DataFrame, output_dir: Path):
    """Create log-scale scatter plot for better visualization of wide data ranges"""
    
    plt.figure(figsize=(14, 10))
    
    # Prepare data
    pagerank = all_data['pagerank'].tolist()
    victim_conn = all_data['victim_connection_weight'].tolist()
    roles = all_data['role'].tolist()
    
    # Color by role
    colors = []
    sizes = []
    for role in roles:
        if role == 'Culprit':
            colors.append('#2E8B57')  # Sea green for culprits
            sizes.append(100)  # Larger dots for culprits
        elif role == 'Victim':
            colors.append('#DC143C')  # Crimson for victims
            sizes.append(80)  # Medium dots for victims
        else:
            colors.append('#4682B4')  # Steel blue for others
            sizes.append(30)  # Smaller dots for others
    
    # Create scatter plot
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
    
    # Use log scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('PageRank (log scale)', fontsize=14)
    plt.ylabel('Victim Connection Weight (log scale)', fontsize=14)
    plt.title('Combined Analysis: PageRank vs Victim Connection\n(Log Scale - All Characters from All Stories)', fontsize=16, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Culprits'),
        Patch(facecolor='#DC143C', label='Victims'),
        Patch(facecolor='#4682B4', label='Other Characters')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    total_chars = len(all_data)
    culprits = len(all_data[all_data['role'] == 'Culprit'])
    victims = len(all_data[all_data['role'] == 'Victim'])
    others = len(all_data[all_data['role'] == 'Other'])
    
    stats_text = f'Total Characters: {total_chars}\nCulprits: {culprits}\nVictims: {victims}\nOthers: {others}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "combined_pagerank_vs_victim_connection_log_scale.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_combined_metric_distributions(all_data: pd.DataFrame, output_dir: Path):
    """Create distribution plots for each metric by role"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Metric Distributions by Role (All Stories Combined)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('pagerank', 'PageRank'),
        ('strength', 'Strength (Weighted Degree)'),
        ('victim_connection_weight', 'Victim Connection Weight'),
        ('degree', 'Degree (Unweighted)')
    ]
    
    colors = {'Culprit': '#2E8B57', 'Victim': '#DC143C', 'Other': '#4682B4'}
    
    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for role in ['Culprit', 'Victim', 'Other']:
            role_data = all_data[all_data['role'] == role][metric_col]
            if len(role_data) > 0:
                ax.hist(role_data, alpha=0.6, label=role, color=colors[role], bins=20)
        
        ax.set_title(f'{metric_name} Distribution', fontweight='bold')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "combined_metric_distributions.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_improved_scaling_plots(all_data: pd.DataFrame, output_dir: Path):
    """Create improved plots with better scaling to handle outliers"""
    
    # Prepare data
    pagerank = all_data['pagerank'].tolist()
    victim_conn = all_data['victim_connection_weight'].tolist()
    roles = all_data['role'].tolist()
    
    # Color by role with different sizes
    colors = []
    sizes = []
    for role in roles:
        if role == 'Culprit':
            colors.append('#2E8B57')  # Sea green for culprits
            sizes.append(120)  # Larger dots for culprits
        elif role == 'Victim':
            colors.append('#DC143C')  # Crimson for victims
            sizes.append(100)  # Medium dots for victims
        else:
            colors.append('#4682B4')  # Steel blue for others
            sizes.append(40)  # Smaller dots for others
    
    # Create legend elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Culprits'),
        Patch(facecolor='#DC143C', label='Victims'),
        Patch(facecolor='#4682B4', label='Other Characters')
    ]
    
    # Statistics
    total_chars = len(all_data)
    culprits = len(all_data[all_data['role'] == 'Culprit'])
    victims = len(all_data[all_data['role'] == 'Victim'])
    others = len(all_data[all_data['role'] == 'Other'])
    stats_text = f'Total Characters: {total_chars}\nCulprits: {culprits}\nVictims: {victims}\nOthers: {others}'
    
    # 1. Full scale plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
    plt.xlabel('PageRank', fontsize=14)
    plt.ylabel('Victim Connection Weight', fontsize=14)
    plt.title('Combined Analysis: PageRank vs Victim Connection\n(All Characters from All Stories)', fontsize=16, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "combined_pagerank_vs_victim_connection_full.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Zoomed plot (focus on main data cluster)
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
    plt.xlim(0, 0.15)  # Focus on PageRank 0-0.15
    plt.ylim(0, 5)     # Focus on Victim Connection 0-5
    plt.xlabel('PageRank', fontsize=14)
    plt.ylabel('Victim Connection Weight', fontsize=14)
    plt.title('Combined Analysis: PageRank vs Victim Connection\n(Zoomed View - Main Data Cluster)', fontsize=16, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "combined_pagerank_vs_victim_connection_zoomed.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Log scale plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('PageRank (log scale)', fontsize=14)
    plt.ylabel('Victim Connection Weight (log scale)', fontsize=14)
    plt.title('Combined Analysis: PageRank vs Victim Connection\n(Log Scale - All Characters from All Stories)', fontsize=16, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "combined_pagerank_vs_victim_connection_log_scale.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Percentile-based zoom (remove extreme outliers)
    plt.figure(figsize=(14, 10))
    pr_95 = np.percentile(pagerank, 95)
    vc_95 = np.percentile(victim_conn, 95)
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.3)
    plt.xlim(0, pr_95 * 1.1)  # 10% margin above 95th percentile
    plt.ylim(0, vc_95 * 1.1)
    plt.xlabel('PageRank', fontsize=14)
    plt.ylabel('Victim Connection Weight', fontsize=14)
    plt.title(f'Combined Analysis: PageRank vs Victim Connection\n(95th Percentile View - Removes Extreme Outliers)', fontsize=16, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    stats_text_with_percentiles = f'{stats_text}\n\n95th Percentile Limits:\nPageRank: {pr_95:.4f}\nVictim Conn: {vc_95:.1f}'
    plt.text(0.02, 0.98, stats_text_with_percentiles, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "combined_pagerank_vs_victim_connection_95th_percentile.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Improved scaling plots created:")
    print("    - Full scale: combined_pagerank_vs_victim_connection_full.png")
    print("    - Zoomed view: combined_pagerank_vs_victim_connection_zoomed.png")
    print("    - Log scale: combined_pagerank_vs_victim_connection_log_scale.png")
    print("    - 95th percentile: combined_pagerank_vs_victim_connection_95th_percentile.png")
    
    # Print data range info
    print(f"\n📊 Data Range Information:")
    print(f"PageRank range: {min(pagerank):.6f} to {max(pagerank):.6f}")
    print(f"Victim Connection range: {min(victim_conn):.1f} to {max(victim_conn):.1f}")
    print(f"95th percentile PageRank: {pr_95:.6f}")
    print(f"95th percentile Victim Connection: {vc_95:.1f}")


def plot_evaluation(all_data: pd.DataFrame, output_dir: Path):
    """Create evaluation plots: rank-based evaluation and role-based distributions"""
    
    print("Creating evaluation plots...")
    
    # 1. Rank-Based Evaluation
    print("  📊 Computing rank-based evaluation...")
    
    # Group by story for evaluation
    story_groups = all_data.groupby('story_index')
    
    # Evaluation metrics
    top1_accuracy = 0
    top3_accuracy = 0
    top5_accuracy = 0
    mrr_scores = []
    
    # Different scoring methods to test
    scoring_methods = {
        'victim_connection': 'victim_connection_weight',
        'pagerank': 'pagerank',
        'combined_score': lambda x: x['victim_connection_weight'] + 0.3 * x['pagerank'],
        'strength': 'strength'
    }
    
    evaluation_results = {}
    
    for method_name, method in scoring_methods.items():
        print(f"    Testing method: {method_name}")
        
        story_ranks = []
        story_accuracies = {'top1': 0, 'top3': 0, 'top5': 0}
        story_mrr = []
        
        for story_idx, story_data in story_groups:
            # Get culprits for this story
            culprits = story_data[story_data['role'] == 'Culprit']
            if len(culprits) == 0:
                continue
                
            # Calculate scores
            if callable(method):
                story_data['score'] = method(story_data)
            else:
                story_data['score'] = story_data[method]
            
            # Sort by score (descending)
            ranked_data = story_data.sort_values('score', ascending=False).reset_index(drop=True)
            
            # Find culprit ranks
            culprit_ranks = []
            for _, culprit in culprits.iterrows():
                culprit_rank = ranked_data[ranked_data['node'] == culprit['node']].index[0] + 1
                culprit_ranks.append(culprit_rank)
            
            # Record best (lowest) rank for this story
            best_rank = min(culprit_ranks)
            story_ranks.append(best_rank)
            
            # Calculate accuracies for this story
            if best_rank == 1:
                story_accuracies['top1'] += 1
            if best_rank <= 3:
                story_accuracies['top3'] += 1
            if best_rank <= 5:
                story_accuracies['top5'] += 1
            
            # Calculate MRR for this story
            story_mrr.append(1.0 / best_rank)
        
        # Aggregate across all stories
        total_stories = len(story_ranks)
        if total_stories > 0:
            evaluation_results[method_name] = {
                'top1_accuracy': story_accuracies['top1'] / total_stories,
                'top3_accuracy': story_accuracies['top3'] / total_stories,
                'top5_accuracy': story_accuracies['top5'] / total_stories,
                'mrr': np.mean(story_mrr),
                'total_stories': total_stories,
                'mean_rank': np.mean(story_ranks)
            }
    
    # Create rank-based evaluation plot
    plt.figure(figsize=(15, 10))
    
    methods = list(evaluation_results.keys())
    top1_scores = [evaluation_results[m]['top1_accuracy'] for m in methods]
    top3_scores = [evaluation_results[m]['top3_accuracy'] for m in methods]
    top5_scores = [evaluation_results[m]['top5_accuracy'] for m in methods]
    mrr_scores = [evaluation_results[m]['mrr'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.2
    
    plt.bar(x - 1.5*width, top1_scores, width, label='Top-1 Accuracy', color='#2E8B57', alpha=0.8)
    plt.bar(x - 0.5*width, top3_scores, width, label='Top-3 Accuracy', color='#4682B4', alpha=0.8)
    plt.bar(x + 0.5*width, top5_scores, width, label='Top-5 Accuracy', color='#DC143C', alpha=0.8)
    plt.bar(x + 1.5*width, mrr_scores, width, label='Mean Reciprocal Rank', color='#FF8C00', alpha=0.8)
    
    plt.xlabel('Scoring Method', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Rank-Based Evaluation: Culprit Detection Performance\nAcross Different Scoring Methods', fontsize=16, fontweight='bold')
    plt.xticks(x, methods, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, method in enumerate(methods):
        plt.text(i - 1.5*width, top1_scores[i] + 0.01, f'{top1_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        plt.text(i - 0.5*width, top3_scores[i] + 0.01, f'{top3_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        plt.text(i + 0.5*width, top5_scores[i] + 0.01, f'{top5_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        plt.text(i + 1.5*width, mrr_scores[i] + 0.01, f'{mrr_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rank_based_evaluation.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print evaluation results
    print("  📈 Rank-Based Evaluation Results:")
    for method, results in evaluation_results.items():
        print(f"    {method}:")
        print(f"      Top-1 Accuracy: {results['top1_accuracy']:.3f} ({results['top1_accuracy']*100:.1f}%)")
        print(f"      Top-3 Accuracy: {results['top3_accuracy']:.3f} ({results['top3_accuracy']*100:.1f}%)")
        print(f"      Top-5 Accuracy: {results['top5_accuracy']:.3f} ({results['top5_accuracy']*100:.1f}%)")
        print(f"      Mean Reciprocal Rank: {results['mrr']:.3f}")
        print(f"      Mean Rank: {results['mean_rank']:.1f}")
        print(f"      Total Stories: {results['total_stories']}")
    
    # 2. Role-Based Distributions
    print("  📊 Creating role-based distribution plots...")
    
    # Prepare data for boxplots
    roles = ['Culprit', 'Victim', 'Other']
    metrics = ['pagerank', 'victim_connection_weight', 'strength', 'degree']
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for boxplot
        box_data = []
        box_labels = []
        
        for role in roles:
            role_data = all_data[all_data['role'] == role][metric].dropna()
            if len(role_data) > 0:
                box_data.append(role_data)
                box_labels.append(f'{role}\n(n={len(role_data)})')
        
        # Create boxplot
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#2E8B57', '#DC143C', '#4682B4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = ""
        for j, role in enumerate(roles):
            role_data = all_data[all_data['role'] == role][metric].dropna()
            if len(role_data) > 0:
                mean_val = role_data.mean()
                median_val = role_data.median()
                stats_text += f"{role}: μ={mean_val:.3f}, med={median_val:.3f}\n"
        
        ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Role-Based Metric Distributions\nCulprits vs Victims vs Other Characters', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "role_based_distributions.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create violin plots for better distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for violin plot
        violin_data = []
        violin_labels = []
        
        for role in roles:
            role_data = all_data[all_data['role'] == role][metric].dropna()
            if len(role_data) > 0:
                violin_data.append(role_data)
                violin_labels.append(f'{role}\n(n={len(role_data)})')
        
        # Create violin plot
        parts = ax.violinplot(violin_data, positions=range(1, len(violin_data)+1), showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['#2E8B57', '#DC143C', '#4682B4']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(1, len(violin_labels)+1))
        ax.set_xticklabels(violin_labels)
        ax.set_title(f'{metric.replace("_", " ").title()} (Violin Plot)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Role-Based Metric Distributions (Violin Plots)\nCulprits vs Victims vs Other Characters', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "role_based_distributions_violin.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Evaluation plots created:")
    print("    - Rank-based evaluation: rank_based_evaluation.png")
    print("    - Role-based distributions (boxplots): role_based_distributions.png")
    print("    - Role-based distributions (violin plots): role_based_distributions_violin.png")


def main():
    parser = argparse.ArgumentParser(description="Combined culprit analysis for entire dataset")
    parser.add_argument("--output-dir", type=str, default="culprit_analysis_results", help="Output directory for results")
    parser.add_argument("--use-saved-data", action="store_true", help="Use existing combined_character_data.csv instead of processing from scratch")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if we should use saved data
    if args.use_saved_data:
        saved_data_path = output_dir / "combined_character_data.csv"
        if saved_data_path.exists():
            print(f"Using saved data from {saved_data_path}")
            combined_data = pd.read_csv(saved_data_path)
            print(f"Loaded {len(combined_data)} characters from saved data")
        else:
            print(f"Error: Saved data not found at {saved_data_path}")
            print("Run without --use-saved-data to generate the data first")
            return
    else:
        # Find all stories with node metrics
        stories = find_stories_with_node_metrics()
        print(f"Found {len(stories)} stories with node metrics")
        
        # Load victim list
        victim_list_path = Path("/Users/amirtbl/Personal/Needle_project/victim_list.csv")
        
        # Collect all data
        all_data = []
        successful_stories = 0
        
        for story_index, story_title, story_dir in stories:
            print(f"Processing Story {story_index}: {story_title}")
            
            try:
                # Load data
                node_metrics_csv = story_dir / "node_metrics.csv"
                aliases_csv = story_dir / f"{story_title}_aliases.csv"
                
                if not node_metrics_csv.exists() or not aliases_csv.exists():
                    print(f"  Skipping - missing required files")
                    continue
                
                node_metrics = pd.read_csv(node_metrics_csv)
                alias_to_can = load_alias_mapping(aliases_csv)
                
                # Load culprit and victim information
                culprit_names = load_culprit_from_dataset(story_index)
                victim_names = load_victim_from_list(story_title, victim_list_path)
                
                # Map to canonical names
                culprits = map_names_to_canonical(culprit_names, alias_to_can, node_metrics, story_index)
                victims = map_names_to_canonical(victim_names, alias_to_can, node_metrics, story_index)
                
                # Add role information to node metrics
                node_metrics['role'] = 'Other'
                node_metrics.loc[node_metrics['node'].isin(culprits), 'role'] = 'Culprit'
                node_metrics.loc[node_metrics['node'].isin(victims), 'role'] = 'Victim'
                node_metrics['story_title'] = story_title
                node_metrics['story_index'] = story_index
                
                all_data.append(node_metrics)
                successful_stories += 1
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not all_data:
            print("No data collected!")
            return
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined data from {successful_stories} stories:")
        print(f"Total characters: {len(combined_data)}")
        print(f"Culprits: {len(combined_data[combined_data['role'] == 'Culprit'])}")
        print(f"Victims: {len(combined_data[combined_data['role'] == 'Victim'])}")
        print(f"Others: {len(combined_data[combined_data['role'] == 'Other'])}")
    
    # Create combined plots
    print("\nCreating combined plots...")
    
    scatter_path = plot_combined_pagerank_vs_victim_connection(combined_data, output_dir)
    log_scale_path = plot_combined_pagerank_vs_victim_connection_log_scale(combined_data, output_dir)
    dist_path = plot_combined_metric_distributions(combined_data, output_dir)
    
    # Create improved scaling plots
    print("Creating improved scaling plots...")
    create_improved_scaling_plots(combined_data, output_dir)
    
    # Create evaluation plots
    print("Creating evaluation plots...")
    plot_evaluation(combined_data, output_dir)
    
    # Save combined data
    combined_csv = output_dir / "combined_character_data.csv"
    combined_data.to_csv(combined_csv, index=False)
    
    print(f"\n✅ Combined analysis complete!")
    print(f"📊 Plots saved:")
    print(f"  - Combined scatter plot (full & zoomed): {scatter_path}")
    print(f"  - Log-scale scatter plot: {log_scale_path}")
    print(f"  - Metric distributions: {dist_path}")
    print(f"  - Combined data: {combined_csv}")
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    for role in ['Culprit', 'Victim', 'Other']:
        role_data = combined_data[combined_data['role'] == role]
        if len(role_data) > 0:
            print(f"\n{role}s ({len(role_data)} characters):")
            print(f"  - Avg PageRank: {role_data['pagerank'].mean():.4f}")
            print(f"  - Avg Strength: {role_data['strength'].mean():.2f}")
            print(f"  - Avg Victim Connection: {role_data['victim_connection_weight'].mean():.2f}")
            print(f"  - Avg Degree: {role_data['degree'].mean():.2f}")


if __name__ == "__main__":
    main()
