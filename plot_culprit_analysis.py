#!/usr/bin/env python3
"""
Culprit Analysis Plots

Creates two key visualizations:
1. Metric Ranking Bar Chart - Top-k characters ranked by each metric with true culprit highlighted
2. Scatter Plot - PageRank vs Victim Connection with role-based coloring

Loads culprit information from the dataset and matches to canonical names via alias mapping.
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
        print(f"Error loading culprit from dataset: {e}")
        return []


def load_victim_from_list(story_title: str, victim_list_path: Path) -> List[str]:
    """Load victim information from victim_list.csv"""
    victims = []
    
    if not victim_list_path.exists():
        print(f"Warning: Victim list not found at {victim_list_path}")
        return victims
    
    try:
        df = pd.read_csv(victim_list_path)
        row = df.loc[df['Story'] == story_title]
        if row.empty:
            print(f"Warning: No entry found for story '{story_title}' in victim list")
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
        print(f"Error loading victim from list: {e}")
    
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
        if name_id_map:
            print(f"Loaded metadata name_id_map: {name_id_map}")
    
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
                            print(f"Metadata mapping for '{name}': {meta_key} -> {meta_value} -> {mapped_can}")
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
                        # If we found matches, add them
                        canonical_names.update(matched_parts)
                        print(f"Partial match for '{name}': {matched_parts}")
                    else:
                        print(f"Warning: Could not map '{name}' to canonical name")
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
                    # If we found matches, add them
                    canonical_names.update(matched_parts)
                    print(f"Partial match for '{name}': {matched_parts}")
                else:
                    print(f"Warning: Could not map '{name}' to canonical name")
    
    return canonical_names


def plot_metric_rankings(node_metrics: pd.DataFrame, culprits: Set[str], victims: Set[str], 
                        story_title: str, output_dir: Path, top_k: int = 10):
    """Create metric ranking bar charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Metric Rankings: {story_title}', fontsize=16, fontweight='bold')
    
    metrics = [
        ('pagerank', 'PageRank'),
        ('strength', 'Strength (Weighted Degree)'),
        ('victim_connection_weight', 'Victim Connection Weight'),
        ('degree', 'Degree (Unweighted)')
    ]
    
    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric (descending)
        sorted_data = node_metrics.sort_values(metric_col, ascending=False).head(top_k)
        
        characters = sorted_data['node'].tolist()
        values = sorted_data[metric_col].tolist()
        
        # Color bars: green for culprits, red for victims, blue for others
        colors = []
        for char in characters:
            if char in culprits:
                colors.append('#2E8B57')  # Sea green for culprits
            elif char in victims:
                colors.append('#DC143C')  # Crimson for victims
            else:
                colors.append('#4682B4')  # Steel blue for others
        
        bars = ax.bar(range(len(characters)), values, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'Top {top_k} by {metric_name}', fontweight='bold')
        ax.set_xlabel('Characters')
        ax.set_ylabel(metric_name)
        ax.set_xticks(range(len(characters)))
        ax.set_xticklabels(characters, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Culprit'),
        Patch(facecolor='#DC143C', label='Victim'),
        Patch(facecolor='#4682B4', label='Other')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "metric_rankings.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_pagerank_vs_victim_connection(node_metrics: pd.DataFrame, culprits: Set[str], 
                                     victims: Set[str], story_title: str, output_dir: Path):
    """Create PageRank vs Victim Connection scatter plot"""
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    pagerank = node_metrics['pagerank'].tolist()
    victim_conn = node_metrics['victim_connection_weight'].tolist()
    characters = node_metrics['node'].tolist()
    
    # Color and label by role
    colors = []
    labels = []
    for char in characters:
        if char in culprits:
            colors.append('#2E8B57')  # Sea green for culprits
            labels.append('Culprit')
        elif char in victims:
            colors.append('#DC143C')  # Crimson for victims
            labels.append('Victim')
        else:
            colors.append('#4682B4')  # Steel blue for others
            labels.append('Other')
    
    # Create scatter plot
    scatter = plt.scatter(pagerank, victim_conn, c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Add character labels
    for i, char in enumerate(characters):
        plt.annotate(char, (pagerank[i], victim_conn[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('PageRank', fontsize=12)
    plt.ylabel('Victim Connection Weight', fontsize=12)
    plt.title(f'PageRank vs Victim Connection: {story_title}', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Culprit'),
        Patch(facecolor='#DC143C', label='Victim'),
        Patch(facecolor='#4682B4', label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "pagerank_vs_victim_connection.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def analyze_culprit_performance(node_metrics: pd.DataFrame, culprits: Set[str], victims: Set[str]):
    """Analyze how well the metrics identify culprits"""
    
    print(f"\n{'='*60}")
    print("CULPRIT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if not culprits:
        print("No culprit information available")
        return
    
    print(f"True culprits: {', '.join(culprits)}")
    print(f"True victims: {', '.join(victims)}")
    
    # Analyze each metric
    metrics = ['pagerank', 'strength', 'victim_connection_weight', 'degree']
    metric_names = ['PageRank', 'Strength', 'Victim Connection', 'Degree']
    
    for metric, name in zip(metrics, metric_names):
        sorted_data = node_metrics.sort_values(metric, ascending=False)
        characters = sorted_data['node'].tolist()
        
        culprit_ranks = []
        for culprit in culprits:
            if culprit in characters:
                rank = characters.index(culprit) + 1
                culprit_ranks.append(rank)
        
        if culprit_ranks:
            best_rank = min(culprit_ranks)
            print(f"\n{name}:")
            print(f"  Best culprit rank: {best_rank} (out of {len(characters)})")
            if best_rank <= 3:
                print(f"  🎯 Excellent - culprit in top 3!")
            elif best_rank <= 5:
                print(f"  👍 Good - culprit in top 5")
            elif best_rank <= 10:
                print(f"  ⚠️  Moderate - culprit in top 10")
            else:
                print(f"  ❌ Poor - culprit not in top 10")
        else:
            print(f"\n{name}: ❌ Culprit not found in character list")
    
    # Combined score analysis
    print(f"\nCombined Analysis:")
    print(f"  Culprits with high victim connection: {len([c for c in culprits if c in node_metrics['node'].values and node_metrics[node_metrics['node'] == c]['victim_connection_weight'].iloc[0] > 0])}")
    print(f"  Culprits with medium-high PageRank: {len([c for c in culprits if c in node_metrics['node'].values and node_metrics[node_metrics['node'] == c]['pagerank'].iloc[0] > node_metrics['pagerank'].median()])}")


def main():
    parser = argparse.ArgumentParser(description="Create culprit analysis plots")
    parser.add_argument("story_index", type=int, help="Story index in WhoDunIt (0-based)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top characters to show in rankings")
    args = parser.parse_args()
    
    # Load story information
    ds = load_dataset("kjgpta/WhoDunIt", split="train")
    if args.story_index >= len(ds):
        raise IndexError(f"Story index {args.story_index} out of range (N={len(ds)})")
    
    story = ds[args.story_index]
    story_title = story.get("title", f"Story_{args.story_index}")
    
    # Locate pipeline outputs
    story_dir = Path("character_data") / f"{clean_dir_name(story_title)}_{args.story_index}"
    node_metrics_csv = story_dir / "node_metrics.csv"
    aliases_csv = story_dir / f"{story_title}_aliases.csv"
    
    if not node_metrics_csv.exists():
        raise FileNotFoundError(f"Node metrics CSV not found at {node_metrics_csv}")
    if not aliases_csv.exists():
        raise FileNotFoundError(f"Aliases CSV not found at {aliases_csv}")
    
    # Load data
    node_metrics = pd.read_csv(node_metrics_csv)
    alias_to_can = load_alias_mapping(aliases_csv)
    
    # Load culprit information from dataset and victim information from victim list
    culprit_names = load_culprit_from_dataset(args.story_index)
    
    # Load victim information from victim_list.csv
    default_victim_csv = Path("/Users/amirtbl/Personal/Needle_project/victim_list.csv")
    victim_names = load_victim_from_list(story_title, default_victim_csv)
    
    print(f"Raw culprit names from dataset: {culprit_names}")
    print(f"Raw victim names from victim list: {victim_names}")
    
    # Map to canonical names
    culprits = map_names_to_canonical(culprit_names, alias_to_can, node_metrics, args.story_index)
    victims = map_names_to_canonical(victim_names, alias_to_can, node_metrics, args.story_index)
    
    print(f"Mapped culprits: {culprits}")
    print(f"Mapped victims: {victims}")
    
    # Create plots
    ranking_path = plot_metric_rankings(node_metrics, culprits, victims, story_title, story_dir, args.top_k)
    scatter_path = plot_pagerank_vs_victim_connection(node_metrics, culprits, victims, story_title, story_dir)
    
    # Analyze performance
    analyze_culprit_performance(node_metrics, culprits, victims)
    
    print(f"\nPlots saved:")
    print(f"  - Metric rankings: {ranking_path}")
    print(f"  - PageRank vs Victim Connection: {scatter_path}")


if __name__ == "__main__":
    main()
