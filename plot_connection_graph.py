#!/usr/bin/env python3
"""
Plot connection graphs for a single WhoDunIt story using pipeline outputs.

Features:
- Builds a weighted graph from <title>_interactions.csv
- Computes PageRank
- Highlights node importance based on connections to the murdered victim(s)
- Resolves victim(s) from chars CSV (is_victim column); if missing, falls back to
  murdered_victims_by_story_with_aliases.csv and alias matching; finally tries metadata name_id_map

Inputs are the existing pipeline outputs in character_data/<Title_underscored>_<index>/.
"""

from __future__ import annotations

import argparse
import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datasets import load_dataset


def clean_dir_name(title: str) -> str:
    return title.replace(' ', '_').replace("'", '').replace('"', '').replace(':', '').replace(';', '')


def load_alias_mapping(aliases_csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Load canonical->aliases and alias->canonical from CSV produced by alias_builder.py."""
    df = pd.read_csv(aliases_csv_path)
    can_to_aliases: Dict[str, List[str]] = {}
    alias_to_can: Dict[str, str] = {}
    for _, row in df.iterrows():
        can = str(row['canonical_name'])
        al = str(row['alias_name']) if not (pd.isna(row['alias_name']) or row['alias_name'] == '') else None
        can_to_aliases.setdefault(can, [])
        if al:
            can_to_aliases[can].append(al)
            alias_to_can[al] = can
        # also map canonical to itself for resolution convenience
        alias_to_can.setdefault(can, can)
    return can_to_aliases, alias_to_can


def load_interactions(interactions_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(interactions_csv_path)
    expected = {'character1', 'character2', 'count'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing interaction columns: {missing}")
    return df


def try_load_metadata_name_id_map(story_index: int) -> Dict[str, str]:
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


def check_name_compatibility_with_alias_rules(victim_name: str, canonical_name: str) -> bool:
    """Check if victim_name would be grouped with canonical_name using alias_builder rules."""
    try:
        # Import the functions from alias_builder
        import sys
        import importlib.util
        
        # Load alias_builder module
        spec = importlib.util.spec_from_file_location("alias_builder", "alias_builder.py")
        alias_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alias_builder)
        
        # Use the same compatibility check as alias_builder
        return alias_builder.persons_compatible(victim_name, canonical_name)
    except Exception:
        # Fallback to simple substring matching if import fails
        def norm_alpha(s: str) -> str:
            return re.sub(r"[^A-Za-z]", "", (s or "")).lower()
        
        victim_norm = norm_alpha(victim_name)
        canonical_norm = norm_alpha(canonical_name)
        return victim_norm in canonical_norm or canonical_norm in victim_norm


def find_victims(story_dir: Path, story_title: str, story_index: int, alias_to_can: Dict[str, str], victim_list_path: Path) -> Set[str]:
    """
    Return set of canonical victim names from curated CSV keyed by story title.
    If no victim recorded (e.g., 'None'), return empty set.
    """
    victims_can: Set[str] = set()

    curated_path = victim_list_path
    if not curated_path or not curated_path.exists():
        return victims_can

    # Build a punctuation-insensitive alias lookup (alpha-only lower)
    def norm_alpha(s: str) -> str:
        return re.sub(r"[^A-Za-z]", "", (s or "")).lower()

    alias_norm_to_can = {norm_alpha(a): c for a, c in alias_to_can.items()}
    # Ensure canonicals also resolve to themselves
    for can in set(alias_to_can.values()):
        alias_norm_to_can.setdefault(norm_alpha(can), can)

    try:
        df = pd.read_csv(curated_path)
        row = df.loc[df['Story'] == story_title]
        if row.empty:
            return victims_can
        # Support both schemas: 'Victim(s)' (new) or 'Murdered Victim(s)' (old)
        victims_field = str(
            row.iloc[0].get('Victim(s)', row.iloc[0].get('Murdered Victim(s)', '')) or ''
        )
        alias_field = str(row.iloc[0].get('Alias list', '') or '')

        def strip_brackets(text: str) -> str:
            # remove any (...) or [...] segments including surrounding whitespace
            return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", text).strip()

        def parse_list_field(text: str) -> List[str]:
            t = text.strip()
            t = t.strip('{}')
            parts = re.split(r"[;,]", t) if t else []
            cleaned: List[str] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                p = strip_brackets(p)
                if p:
                    cleaned.append(p)
            return cleaned

        # Treat 'none' as no victims
        if victims_field.strip().lower() in {'none', '', '{none}', '—', '-'}:
            victim_aliases: List[str] = []
        else:
            victim_aliases = parse_list_field(victims_field)

        alias_list_additional = parse_list_field(alias_field)
        candidates = victim_aliases + alias_list_additional

        for alias in candidates:
            alias = strip_brackets(alias)
            key = norm_alpha(alias)
            can = alias_norm_to_can.get(key)
            if can:
                victims_can.add(can)
            else:
                # Try to resolve through metadata mapping if direct alias lookup fails
                name_id_map = try_load_metadata_name_id_map(story_index)
                if name_id_map:
                    # Split victim name and try to map each part
                    victim_parts = alias.lower().split()
                    mapped_parts = []
                    for part in victim_parts:
                        # Look for the part in metadata keys
                        for meta_key, meta_value in name_id_map.items():
                            if part == meta_key.lower():
                                mapped_parts.append(meta_value)
                                break
                        else:
                            mapped_parts.append(part)  # Keep original if no mapping found
                    
                    # Try to find the mapped name in aliases
                    if mapped_parts:
                        mapped_name = ' '.join(mapped_parts)
                        mapped_key = norm_alpha(mapped_name)
                        mapped_can = alias_norm_to_can.get(mapped_key)
                        if mapped_can:
                            victims_can.add(mapped_can)
                        else:
                            # Try different combinations and capitalizations
                            for possible_name in [mapped_name, mapped_name.title(), ' '.join(mapped_parts).title()]:
                                possible_key = norm_alpha(possible_name)
                                possible_can = alias_norm_to_can.get(possible_key)
                                if possible_can:
                                    victims_can.add(possible_can)
                                    break
                            else:
                                # Use alias builder matching rules to check compatibility
                                for canonical in set(alias_to_can.values()):
                                    if check_name_compatibility_with_alias_rules(mapped_name, canonical):
                                        victims_can.add(canonical)
                                        break
    except Exception:
        return victims_can

    return victims_can


def build_graph(interactions: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in interactions.iterrows():
        a = str(row['character1'])
        b = str(row['character2'])
        w = int(row['count'])
        if a == b:
            continue
        if G.has_edge(a, b):
            G[a][b]['weight'] += w
        else:
            G.add_edge(a, b, weight=w)
    return G


def compute_victim_connection_scores(G: nx.Graph, victims: Set[str]) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for node in G.nodes():
        total = 0
        for v in victims:
            if G.has_edge(node, v):
                total += int(G[node][v].get('weight', 0))
        scores[node] = total
    return scores


def plot_graph(G: nx.Graph, victims: Set[str], pagerank: Dict[str, float], victim_scores: Dict[str, int], out_path: Path, title: str) -> None:
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, weight='weight', seed=42)

    # Node sizes: if victims known, use victim connection score; else use PageRank
    sizes = []
    if victims:
        for n in G.nodes():
            s = victim_scores.get(n, 0)
            sizes.append(max(200, 50 * (1 + s)))
    else:
        # Scale by PageRank
        pr_vals = [pagerank.get(n, 0.0) for n in G.nodes()]
        max_pr = max(pr_vals) if pr_vals else 1.0
        for n in G.nodes():
            pr = pagerank.get(n, 0.0) / max_pr if max_pr > 0 else 0.0
            sizes.append(max(200, 2000 * pr))

    # Colors: highlight victims in red; others by PageRank (blue intensity)
    pr_vals = [pagerank.get(n, 0.0) for n in G.nodes()]
    max_pr = max(pr_vals) if pr_vals else 1.0
    node_colors = []
    for n in G.nodes():
        if n in victims:
            node_colors.append('#d62728')  # red
        else:
            pr = pagerank.get(n, 0.0) / max_pr if max_pr > 0 else 0
            # map to light->dark blue
            node_colors.append((0.2, 0.2, 0.8 * (0.3 + 0.7 * pr)))

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, alpha=0.85, linewidths=0.5, edgecolors='black')
    # Edge widths by weight
    weights = [G[e[0]][e[1]].get('weight', 1) for e in G.edges()]
    max_w = max(weights) if weights else 1
    scaled_w = [1 + 4 * (w / max_w) for w in weights]
    nx.draw_networkx_edges(G, pos, width=scaled_w, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot connection graphs and victim-aware importance for a story")
    ap.add_argument("story_index", type=int, help="Story index in WhoDunIt (0-based)")
    ap.add_argument("--out", help="Output PNG path (default: store in story's out dir)")
    ap.add_argument("--victim-list", dest="victim_list", help="CSV with columns: Story, Murdered Victim(s), Alias list (overrides default)")
    args = ap.parse_args()

    # Load story
    ds = load_dataset("kjgpta/WhoDunIt", split="train")
    if args.story_index >= len(ds):
        raise IndexError(f"Story index {args.story_index} out of range (N={len(ds)})")
    story = ds[args.story_index]
    story_title = story.get("title", f"Story_{args.story_index}")

    # Locate pipeline outputs
    story_dir = Path("character_data") / f"{clean_dir_name(story_title)}_{args.story_index}"
    aliases_csv = story_dir / f"{story_title}_aliases.csv"
    interactions_csv = story_dir / f"{story_title}_interactions.csv"
    if not aliases_csv.exists() or not interactions_csv.exists():
        raise FileNotFoundError(f"Missing aliases or interactions CSV in {story_dir}")

    can_to_aliases, alias_to_can = load_alias_mapping(aliases_csv)
    interactions = load_interactions(interactions_csv)
    G = build_graph(interactions)

    default_victim_csv = Path("/Users/amirtbl/Personal/Needle_project/victim_list.csv")
    victim_csv = Path(args.victim_list) if args.victim_list else default_victim_csv
    victims = find_victims(story_dir, story_title, args.story_index, alias_to_can, victim_csv)
    # Keep only victims present as nodes
    victims = {v for v in victims if v in G.nodes}

    pagerank = nx.pagerank(G, weight='weight') if len(G) > 0 else {}
    victim_scores = compute_victim_connection_scores(G, victims)

    out_path = Path(args.out) if args.out else (story_dir / "graph_victim_connections.png")
    plot_graph(G, victims, pagerank, victim_scores, out_path, f"{story_title} — Victim-aware graph")

    # Also export a CSV with node metrics
    metrics_csv = story_dir / "node_metrics.csv"
    rows = []
    for n in sorted(G.nodes()):
        rows.append({
            'node': n,
            'is_victim': int(n in victims),
            'pagerank': pagerank.get(n, 0.0),
            'victim_connection_weight': victim_scores.get(n, 0),
            'degree': int(G.degree(n, weight=None)),
            'strength': float(sum(G[n][nbr].get('weight', 0) for nbr in G.neighbors(n))),
        })
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"Saved plot to {out_path}")
    print(f"Saved node metrics to {metrics_csv}")


if __name__ == "__main__":
    main()


