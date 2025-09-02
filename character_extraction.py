"""
Simple Character Extraction using Ollama Structured Outputs
Author: Amir
"""

import re
import pandas as pd
from typing import List
from datasets import load_dataset
from pydantic import BaseModel
from ollama import chat
import argparse
import os

# =========================
# Pydantic models
# =========================
class CharacterList(BaseModel):
    characters: List[str]

# =========================
# Data classes
# =========================
class CharacterData:
    def __init__(self, char_id: int, name: str, story_title: str):
        self.char_id = char_id
        self.name = name  # surface form as extracted
        self.story_title = story_title

    def to_dict(self):
        return {"char_id": self.char_id, "name": self.name, "story_title": self.story_title}

# =========================
# Helpers
# =========================
def clean_title_for_filename(title: str) -> str:
    t = re.sub(r"[^\w\s-]", "", title).strip()
    return re.sub(r"[-\s]+", "_", t)

# =========================
# Analyzer
# =========================
class StoryAnalyzer:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def extract_characters(self, text: str, story_title: str) -> List[CharacterData]:
        response = chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": f"""
You are analyzing a mystery story to extract **character names exactly as they appear in the text**.

Story Text:
{text}

Extract ONLY proper names of people as they appear in the text. Keep all versions of a character's name as they appear. Always preserve capitalization exactly as written. Do not invent or infer names that are not explicitly in the text.

Return the list of unique names found in the story, exactly as written.
"""
            }],
            format=CharacterList.model_json_schema(),
            options={"temperature": 0}
        )
        char_data = CharacterList.model_validate_json(response.message.content)

        characters: List[CharacterData] = []
        for i, name in enumerate(char_data.characters, 1):
            if name.strip():
                characters.append(CharacterData(i, name.strip(), story_title))
        return characters

    def analyze_story_batched(self, text: str, story_title: str, chunk_size: int = 2000) -> List[CharacterData]:
        print(f"Analyzing story: {story_title}")
        # Split text into overlapping chunks
        chunks: List[str] = []
        overlap = min(200, chunk_size // 10)
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        print(f"Split into {len(chunks)} chunks")

        all_characters: List[CharacterData] = []
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1}/{len(chunks)} | len={len(chunk)} ---")
            chunk_chars = self.extract_characters(chunk, f"{story_title}_chunk_{i+1}")
            all_characters.extend(chunk_chars)

        # Deduplicate by surface name
        unique_characters: List[CharacterData] = []
        seen_names = set()
        next_id = 1
        for c in all_characters:
            if c.name not in seen_names:
                seen_names.add(c.name)
                c.char_id = next_id
                c.story_title = story_title
                unique_characters.append(c)
                next_id += 1

        print("=" * 60)
        print(f"Character extraction complete → {len(unique_characters)} unique surface forms")
        print("=" * 60)
        return unique_characters

    def save_characters(self, characters: List[CharacterData], story_title: str):
        out_dir = "/Users/amirtbl/Personal/Needle_project/char"
        os.makedirs(out_dir, exist_ok=True)
        base = clean_title_for_filename(story_title)

        # Characters CSV
        char_df = pd.DataFrame([c.to_dict() for c in characters])
        char_csv = os.path.join(out_dir, f"{base}_chars.csv")
        char_df.to_csv(char_csv, index=False)

        # Print characters to stdout
        print("\nCharacters (unique surface forms):")
        for c in characters:
            print(f"- {c.name}")

        print("=" * 50)
        print(f"SAVED:\n- {char_csv}")
        print("=" * 50)
        return char_csv

# =========================
# Dataset runner
# =========================
def analyze_whodunit_story(story_index: int = 0, batch_size: int = 2000, model: str = "llama3.2"):
    print("Loading WhoDunIt dataset...")
    dataset = load_dataset("kjgpta/WhoDunIt")
    train_data = dataset["train"]

    if story_index >= len(train_data):
        print(f"Error: Story index {story_index} out of range (N={len(train_data)}).")
        return

    story = train_data[story_index]
    story_title = story.get("title", story.get("Title", f"Story_{story_index}"))
    story_text = story.get("story", story.get("Story", story.get("text", "")))

    print(f"Selected: {story_title} | len={len(story_text)}")

    analyzer = StoryAnalyzer(model_name=model)
    chars = analyzer.analyze_story_batched(story_text, story_title, batch_size)
    analyzer.save_characters(chars, story_title)

def main():
    parser = argparse.ArgumentParser(description="Extract characters from WhoDunIt using Ollama")
    parser.add_argument("-s", "--story-index", type=int, default=0, help="Index of story to analyze")
    parser.add_argument("-c", "--batch-size", type=int, default=2000, help="Chunk size (chars)")
    parser.add_argument("-m", "--model", type=str, default="llama3.2", help="Ollama model name")
    args = parser.parse_args()

    print("=" * 60)
    print("CHARACTER EXTRACTION ONLY")
    print("=" * 60)
    print(f"Story Index: {args.story_index}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Model:       {args.model}")
    print("=" * 60)

    analyze_whodunit_story(story_index=args.story_index, batch_size=args.batch_size, model=args.model)

if __name__ == "__main__":
    main()
