"""
Simple Character and Interaction Extraction using Ollama Structured Outputs

Clean approach with native Ollama structured outputs:
1. Use Pydantic models to define schemas
2. Use Ollama's format parameter for guaranteed structure
3. Parse outputs automatically with validation

Author: Amir
"""

import requests
import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from pydantic import BaseModel
from ollama import chat
import argparse

# Pydantic models for structured outputs
class CharacterList(BaseModel):
    characters: List[str]

class InteractionItem(BaseModel):
    char1: str
    char2: str  
    action: str

class InteractionList(BaseModel):
    interactions: List[InteractionItem]

class Character:
    def __init__(self, char_id: int, name: str, story_title: str):
        self.char_id = char_id
        self.name = name  # Normalized primary name
        self.story_title = story_title
    
    def to_dict(self):
        return {
            'char_id': self.char_id,
            'name': self.name,
            'story_title': self.story_title
        }

class Interaction:
    def __init__(self, char1_name: str, char2_name: str, interaction_description: str):
        self.char1_name = char1_name
        self.char2_name = char2_name
        self.interaction_description = interaction_description
    
    def to_dict(self):
        return {
            'char1_name': self.char1_name,
            'char2_name': self.char2_name,
            'interaction_description': self.interaction_description
        }

class StoryAnalyzer:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
    
    def extract_characters(self, text: str, story_title: str) -> List[Character]:
        """Extract characters using Ollama structured outputs"""
        
        response = chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': f'''
You are analyzing a mystery story to extract character names. Be very careful to extract only proper names of people.

Story Text:
{text}

Rules:
- Extract ONLY proper names of people (e.g., "Detective John Smith", "Mrs. Williams", "Dr. Brown")
- Do NOT extract: pronouns (he, she, I, you, they), generic words (water, house, book), or titles alone
- Always use the MOST COMPLETE and FORMAL version of each name
- If you see "Dr. Gray" and "Mr. Gray", choose the most formal/complete version
- If you see "Inspector Solomon" and just "Solomon", use "Inspector Solomon"
- Consolidate variations of the same person into ONE canonical name

Good examples: "Detective Sarah Williams", "Dr. Michael Stevens", "Inspector Solomon"
Bad examples: "she", "he", "water", "Detective" (alone), "Gray" (without title)

Extract all proper character names, using the most complete form for each person.
'''
            }],
            format=CharacterList.model_json_schema(),
            options={'temperature': 0}
        )
        
        # Parse the structured response
        char_data = CharacterList.model_validate_json(response.message.content)
        
        characters = []
        for i, name in enumerate(char_data.characters, 1):
            if name.strip():
                characters.append(Character(i, name.strip(), story_title))
        
        print(f"Extracted {len(characters)} characters:")
        for char in characters:
            print(f"  {char.char_id}: {char.name}")
        
        return characters
    
    def extract_interactions(self, text: str, characters: List[Character]) -> List[Interaction]:
        """Extract interactions using Ollama structured outputs"""
        char_names = [char.name for char in characters]
        char_list = ", ".join(char_names)
        
        response = chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': f'''
You are finding interactions between specific characters in a mystery story.

Characters in this story: {char_list}

Story Text:
{text}

Rules:
- Find interactions ONLY when two characters (or pronouns referring to them) appear in the SAME SENTENCE
- When you see pronouns (he, she, they, him, her), figure out which character they refer to
- Use ONLY the exact character names from the list above - no pronouns in results
- An interaction occurs when two different characters are mentioned together in one sentence
- Resolve references: if sentence says "Detective Smith questioned her about the case" and "her" refers to "Mrs. Johnson", create interaction between "Detective Smith" and "Mrs. Johnson"
- Use simple actions based on what happens in that sentence: "talked to", "met with", "questioned", "argued with", "appeared with", "mentioned"
- Don't duplicate interactions (if A talked to B, don't also add B talked to A)
- Only create interactions for characters that appear together in the same sentence

Examples:
- "Detective Smith questioned Mrs. Johnson about the murder." → Smith questioned Johnson
- "He told her about the evidence." (if he=Smith, her=Johnson) → Smith told Johnson  
- "Smith was in the garden. Johnson was in the house." → NO interaction (different sentences)

Only create interactions where both characters appear in the same sentence.
'''
            }],
            format=InteractionList.model_json_schema(),
            options={'temperature': 0}
        )
        
        # Parse the structured response
        interaction_data = InteractionList.model_validate_json(response.message.content)
        
        interactions = []
        for item in interaction_data.interactions:
            # Verify characters exist in our list
            if item.char1.strip() in char_names and item.char2.strip() in char_names:
                interactions.append(Interaction(
                    item.char1.strip(), 
                    item.char2.strip(), 
                    item.action.strip()
                ))
        
        print(f"\nExtracted {len(interactions)} interactions:")
        for interaction in interactions:
            print(f"  {interaction.char1_name} -> {interaction.char2_name}: {interaction.interaction_description}")
        
        return interactions

    def chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        overlap = min(200, chunk_size // 10)  # 10% overlap, max 200
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def analyze_story_batched(self, text: str, story_title: str, chunk_size: int = 2000) -> Tuple[List[Character], List[Interaction]]:
        """Analyze a story in batches using structured outputs"""
        print(f"Analyzing story: {story_title}")
        print(f"Text length: {len(text)} characters")
        
        # Split into chunks
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        print(f"Split into {len(chunks)} chunks")
        print("-" * 50)
        
        all_characters = []
        all_interactions = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {i+1}/{len(chunks)} ---")
            print(f"Chunk length: {len(chunk)} characters")
            
            # Extract characters from this chunk
            chunk_characters = self.extract_characters(chunk, f"{story_title}_chunk_{i+1}")
            all_characters.extend(chunk_characters)
            
            # Extract interactions from this chunk
            chunk_interactions = self.extract_interactions(chunk, chunk_characters)
            all_interactions.extend(chunk_interactions)
        
        # Deduplicate characters by name
        unique_characters = []
        seen_names = set()
        char_id = 1
        
        for char in all_characters:
            if char.name not in seen_names:
                seen_names.add(char.name)
                # Update char_id and story_title
                char.char_id = char_id
                char.story_title = story_title
                unique_characters.append(char)
                char_id += 1
        
        # Deduplicate interactions
        unique_interactions = []
        seen_interactions = set()
        
        for interaction in all_interactions:
            # Create a normalized key for deduplication
            key = tuple(sorted([interaction.char1_name, interaction.char2_name]))
            if key not in seen_interactions:
                seen_interactions.add(key)
                unique_interactions.append(interaction)
        
        print(f"\n" + "=" * 50)
        print(f"FINAL RESULTS:")
        print(f"Total unique characters: {len(unique_characters)}")
        print(f"Total unique interactions: {len(unique_interactions)}")
        print("=" * 50)
        
        return unique_characters, unique_interactions
    
    def save_results(self, characters: List[Character], interactions: List[Interaction], story_title: str):
        """Save results to CSV files"""
        # Clean story title for filename
        clean_title = re.sub(r'[^\w\s-]', '', story_title).strip()
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        
        # Save characters
        char_df = pd.DataFrame([char.to_dict() for char in characters])
        char_filename = f"{clean_title}_characters.csv"
        char_df.to_csv(char_filename, index=False)
        print(f"\nSaved characters to: {char_filename}")
        
        # Save interactions
        interaction_df = pd.DataFrame([interaction.to_dict() for interaction in interactions])
        interaction_filename = f"{clean_title}_interactions.csv"
        interaction_df.to_csv(interaction_filename, index=False)
        print(f"Saved interactions to: {interaction_filename}")
        
        return char_filename, interaction_filename

    def analyze_story_batched_parallel(self, text: str, story_title: str, max_workers: int = 4, chunk_size: int = 2000) -> Tuple[List[Character], List[Interaction]]:
        """Analyze story in parallel batches using structured outputs"""
        import concurrent.futures
        
        print(f"Analyzing story: {story_title}")
        print(f"Text length: {len(text)} characters")
        
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        print(f"Split into {len(chunks)} chunks")
        print(f"Using {max_workers} parallel workers")
        print("-" * 50)
        
        def process_chunk(chunk_data):
            """Process a single chunk - designed for parallel execution"""
            i, chunk = chunk_data
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract characters from this chunk
            chunk_characters = self.extract_characters(chunk, f"{story_title}_chunk_{i+1}")
            
            # Extract interactions from this chunk
            chunk_interactions = self.extract_interactions(chunk, chunk_characters)
            
            return chunk_characters, chunk_interactions
        
        # Process chunks in parallel
        all_characters = []
        all_interactions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            chunk_data = [(i, chunk) for i, chunk in enumerate(chunks)]
            future_to_chunk = {executor.submit(process_chunk, data): data for data in chunk_data}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_characters, chunk_interactions = future.result()
                all_characters.extend(chunk_characters)
                all_interactions.extend(chunk_interactions)
        
        # Deduplicate characters by name
        unique_characters = []
        seen_names = set()
        char_id = 1
        
        for char in all_characters:
            if char.name not in seen_names:
                seen_names.add(char.name)
                char.char_id = char_id
                char.story_title = story_title
                unique_characters.append(char)
                char_id += 1
        
        # Deduplicate interactions
        unique_interactions = []
        seen_interactions = set()
        
        for interaction in all_interactions:
            key = tuple(sorted([interaction.char1_name, interaction.char2_name]))
            if key not in seen_interactions:
                seen_interactions.add(key)
                unique_interactions.append(interaction)
        
        print(f"\n" + "=" * 50)
        print(f"PARALLEL PROCESSING COMPLETE:")
        print(f"Total unique characters: {len(unique_characters)}")
        print(f"Total unique interactions: {len(unique_interactions)}")
        print("=" * 50)
        
        return unique_characters, unique_interactions

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Extract characters and interactions from WhoDunIt mystery stories using Ollama LLM")
    
    parser.add_argument('--story-index', '-s', type=int, default=0, 
                        help='Index of story to analyze (default: 0)')
    
    parser.add_argument('--batch', '-b', action='store_true', default=True,
                        help='Use batching for large stories (default: True)')
    
    parser.add_argument('--no-batch', action='store_true',
                        help='Disable batching (process entire story at once)')
    
    parser.add_argument('--parallel', '-p', action='store_true', default=False,
                        help='Use parallel processing for batches (default: False)')
    
    parser.add_argument('--batch-size', '--chunk-size', '-c', type=int, default=2000,
                        help='Size of each text chunk in characters (default: 2000)')
    
    parser.add_argument('--max-workers', '-w', type=int, default=4,
                        help='Maximum number of parallel workers (default: 4)')
    
    parser.add_argument('--model', '-m', type=str, default="llama3.2",
                        help='Ollama model to use (default: llama3.2)')
    
    args = parser.parse_args()
    
    # Handle batch vs no-batch
    use_batching = args.batch and not args.no_batch
    
    print("=" * 60)
    print("CHARACTER AND INTERACTION EXTRACTION")
    print("=" * 60)
    print(f"Story Index: {args.story_index}")
    print(f"Use Batching: {use_batching}")
    print(f"Batch Size: {args.batch_size} characters")
    print(f"Use Parallel: {args.parallel}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    # Run analysis with command line arguments
    analyze_whodunit_story(
        story_index=args.story_index,
        use_batching=use_batching,
        use_parallel=args.parallel,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        model=args.model
    )

def analyze_whodunit_story(story_index: int = 0, use_batching: bool = True, use_parallel: bool = False, max_workers: int = 4, batch_size: int = 2000, model: str = "llama3.2"):
    """Analyze a story from the WhoDunIt dataset"""
    print("Loading WhoDunIt dataset...")
    
    # Load the dataset (simple approach)
    dataset = load_dataset("kjgpta/WhoDunIt")
    train_data = dataset["train"]
    
    if story_index >= len(train_data):
        print(f"Error: Story index {story_index} is out of range. Dataset has {len(train_data)} stories.")
        return
    
    # Get the story
    story = train_data[story_index]
    print("Story structure:", story.keys())
    
    # Extract story data (field names might be different)
    story_title = story.get('title', story.get('Title', f'Story_{story_index}'))
    story_text = story.get('story', story.get('Story', story.get('text', '')))
    
    print(f"Selected story: {story_title}")
    print(f"Story length: {len(story_text)} characters")
    print(f"Using batching: {use_batching}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Initialize analyzer with custom model and batch size
    analyzer = StoryAnalyzer(model_name=model)
    
    # Choose analysis method
    if use_parallel and use_batching:
        print(f"Running PARALLEL BATCHED analysis with {max_workers} workers...")
        characters, interactions = analyzer.analyze_story_batched_parallel(story_text, story_title, max_workers, batch_size)
    elif use_batching:
        print("Running BATCHED analysis...")
        characters, interactions = analyzer.analyze_story_batched(story_text, story_title, batch_size)
    else:
        print("Running SINGLE analysis...")
        characters = analyzer.extract_characters(story_text, story_title)
        interactions = analyzer.extract_interactions(story_text, characters)
    
    # Save results
    char_file, interaction_file = analyzer.save_results(characters, interactions, story_title)
    
    print("=" * 60)
    print("Analysis complete!")
    print(f"Characters: {len(characters)}")
    print(f"Interactions: {len(interactions)}")
    print(f"Files created: {char_file}, {interaction_file}")


def test_small_chunk():
    """Test with just the first 2000 characters of a WhoDunIt story"""
    print("Testing with first 2000 characters...")
    print("=" * 60)
    
    # Load the dataset
    dataset = load_dataset("kjgpta/WhoDunIt")
    train_data = dataset["train"]
    
    # Get the first story
    story = train_data[0]
    print("Story structure:", story.keys())
    
    # Extract story data
    story_title = story.get('title', story.get('Title', 'Story_0'))
    story_text = story.get('story', story.get('Story', story.get('text', '')))
    
    # Take only first 2000 characters
    test_text = story_text[:2000]
    
    print(f"Full story title: {story_title}")
    print(f"Full story length: {len(story_text)} characters")
    print(f"Test chunk length: {len(test_text)} characters")
    print("-" * 50)
    print("First 200 characters of test chunk:")
    print(repr(test_text[:200]))
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = StoryAnalyzer()
    
    # Extract characters
    print("\nStep 1: Extracting characters...")
    characters = analyzer.extract_characters(test_text, story_title + "_test")
    
    # Extract interactions
    print("\nStep 2: Extracting interactions...")
    interactions = analyzer.extract_interactions(test_text, characters)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Characters found: {len(characters)}")
    for char in characters:
        print(f"  - {char.name}")
    
    print(f"\nInteractions found: {len(interactions)}")
    for interaction in interactions:
        print(f"  - {interaction.char1_name} {interaction.interaction_description} {interaction.char2_name}")
    
    return characters, interactions


if __name__ == "__main__":
    main() 