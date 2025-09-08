#!/usr/bin/env python3
"""
Interaction Extraction with Canonical Names

Extract interactions where two characters appear in the same sentence.
Uses canonical names from alias builder and maps aliases to canonicals.

Usage:
    python3 interaction_extraction.py --story path/to/story.txt --aliases path/to/aliases.json
"""

import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple
from itertools import combinations

import pandas as pd


def load_canonical_mapping(aliases_file: str) -> Dict[str, str]:
    """Load aliases and create mapping from alias to canonical name.
    Supports JSON (canonical -> [aliases]) and CSV (canonical_name, alias_name).
    """
    alias_to_canonical: Dict[str, str] = {}

    if aliases_file.lower().endswith('.csv'):
        df = pd.read_csv(aliases_file)
        if 'canonical_name' not in df.columns or 'alias_name' not in df.columns:
            raise ValueError("CSV must contain 'canonical_name' and 'alias_name' columns")
        # Ensure canonicals map to themselves
        for canonical in df['canonical_name'].dropna().astype(str).unique():
            canonical = canonical.strip()
            if canonical:
                alias_to_canonical[canonical] = canonical
        # Map aliases to canonical
        for _, row in df.iterrows():
            canonical = str(row['canonical_name']).strip()
            alias = str(row['alias_name']).strip() if not pd.isna(row['alias_name']) else ''
            if canonical and alias:
                alias_to_canonical[alias] = canonical
        return alias_to_canonical

    # Fallback: JSON
    with open(aliases_file, 'r', encoding='utf-8') as f:
        aliases_data = json.load(f)

    for canonical, aliases in aliases_data.items():
        alias_to_canonical[str(canonical).strip()] = str(canonical).strip()
        for alias in aliases:
            s = str(alias).strip()
            if s:
                alias_to_canonical[s] = str(canonical).strip()
    return alias_to_canonical


def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    # Normalize apostrophes and quotes
    text = text.replace("'", "'").replace("`", "'").replace("'", "'")
    # Normalize dashes
    text = text.replace("—", " ").replace("–", " ").replace("−", "-")
    return text


def find_character_mentions(sentence: str, canonical_mapping: Dict[str, str]) -> Set[str]:
    """Find all character mentions in a sentence and return their canonical names."""
    sentence = normalize_text(sentence)
    found_canonicals = set()
    
    # Sort aliases by length (longest first) to prioritize longer matches
    sorted_aliases = sorted(canonical_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Check each alias/canonical name, longest first
    for alias, canonical in sorted_aliases:
        # Use case-insensitive matching with word boundaries
        # Don't remove punctuation, just normalize apostrophes and dashes
        alias_normalized = normalize_text(alias)
        sentence_normalized = normalize_text(sentence)
        
        # Use word boundaries to avoid partial matches, case-insensitive
        pattern = r'\b' + re.escape(alias_normalized) + r'\b'
        if re.search(pattern, sentence_normalized, re.IGNORECASE):
            found_canonicals.add(canonical)
            # Remove the matched text from sentence to avoid shorter aliases matching the same text
            sentence_normalized = re.sub(pattern, ' ' * len(alias_normalized), sentence_normalized, flags=re.IGNORECASE)
    
    return found_canonicals


def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_pair_counts_from_text(text: str, canonical_mapping: Dict[str, str], chunk_size: int = 5000) -> Dict[Tuple[str, str], int]:
    """Count pair co-occurrences: for each sentence, count all unique character pairs present."""
    chunks = chunk_text(text, chunk_size)
    print(f"Processing {len(chunks)} chunks...")

    pair_to_count: Dict[Tuple[str, str], int] = {}

    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Processing chunk {i+1}/{len(chunks)}...")

        sentences = re.split(r'[.!?]+', chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue

            characters = sorted(find_character_mentions(sentence, canonical_mapping))
            if len(characters) >= 2:
                for c1, c2 in combinations(characters, 2):
                    if c1 == c2:
                        continue
                    key = (c1, c2) if c1 <= c2 else (c2, c1)
                    pair_to_count[key] = pair_to_count.get(key, 0) + 1

    return pair_to_count





def main():
    parser = argparse.ArgumentParser(description="Extract character interactions from text")
    parser.add_argument("--story", required=True, help="Path to story text file")
    parser.add_argument("--aliases", required=True, help="Path to aliases file (CSV or JSON)")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Size of text chunks to process (default: 5000)")
    args = parser.parse_args()
    
    # Load canonical mapping
    print(f"Loading canonical mapping from {args.aliases}...")
    canonical_mapping = load_canonical_mapping(args.aliases)
    print(f"Loaded {len(canonical_mapping)} alias mappings")
    
    # Load story text
    print(f"Loading story from {args.story}...")
    with open(args.story, 'r', encoding='utf-8') as f:
        story_text = f.read()
    print(f"Story length: {len(story_text)} characters")
    
    # Extract pair counts
    print("Extracting interactions (pair counts)...")
    pair_counts = extract_pair_counts_from_text(story_text, canonical_mapping, args.chunk_size)
    print(f"Found {sum(pair_counts.values())} total pair mentions across sentences")

    # Convert to DataFrame with counts
    if pair_counts:
        rows = [{"character1": a, "character2": b, "count": cnt} for (a, b), cnt in pair_counts.items()]
        df = pd.DataFrame(rows, columns=["character1", "character2", "count"])
        df = df.sort_values(["count", "character1", "character2"], ascending=[False, True, True])
    else:
        df = pd.DataFrame(columns=["character1", "character2", "count"])

    # Save output
    output_file = args.output or "interactions.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved interactions to {output_file}")
    
    # Print some examples
    if not df.empty:
        print("\nTop interactions:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['character1']} <-> {row['character2']}  (count={int(row['count'])})")


if __name__ == "__main__":
    main()


