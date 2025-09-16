#!/usr/bin/env python3
"""
Complete Pipeline Script

Runs the entire pipeline:
1. Character extraction
2. Alias building
3. Interaction extraction

Loads stories directly from WhoDunIt dataset.
Saves all outputs in organized directories.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
import json
import re
import ast
import shlex


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        # Run without capturing output so we see real-time progress
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR running {description}:")
        print(f"Exit code: {e.returncode}")
        return False


def create_output_directory(story_name, story_index: int):
    """Create output directory for the story, suffixed with story index to avoid collisions."""
    # Clean story name for directory
    clean_name = story_name.replace(' ', '_').replace("'", '').replace('"', '').replace(':', '').replace(';', '')
    output_dir = Path("out") / f"{clean_name}_{story_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_story_from_dataset(story_index: int):
    """Load story from WhoDunIt dataset."""
    print("Loading WhoDunIt dataset...")
    dataset = load_dataset("kjgpta/WhoDunIt", split="train")
    
    if story_index >= len(dataset):
        raise IndexError(f"Story index {story_index} out of range (0..{len(dataset)-1})")
    
    story = dataset[story_index]
    story_title = story.get("title", f"Story_{story_index}")
    story_text = story.get("story", story.get("text", ""))
    
    print(f"Loaded story: {story_title}")
    print(f"Story length: {len(story_text)} characters")
    
    return story_title, story_text


def run_pipeline(story_index: int, story_name=None, skip_extract: bool = False, reuse_chars: bool = False):
    """Run the complete pipeline for a story from the dataset."""
    # Load story from dataset
    story_title, story_text = load_story_from_dataset(story_index)
    
    if not story_name:
        story_name = story_title
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Character Data Collection Pipeline")
    print(f"📖 Story: {story_name}")
    print(f"🔢 Index: {story_index}")
    print(f"{'='*80}")
    
    # Create output directory (include index to disambiguate duplicates)
    output_dir = create_output_directory(story_name, story_index)
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Story length: {len(story_text):,} characters")
    
    # Create temporary story file for processing
    temp_story_file = output_dir / f"{story_name}_temp.txt"
    with open(temp_story_file, 'w', encoding='utf-8') as f:
        f.write(story_text)
    
    try:
        # Step 1: Character extraction (controlled by --reuse-chars and --skip-extract)
        chars_file = output_dir / f"{story_name}_chars.csv"
        if chars_file.exists() and reuse_chars:
            print(f"♻️  Reusing existing characters CSV: {chars_file}")
        else:
            if skip_extract:
                # Build chars CSV from metadata keys as a fallback (no LLM)
                try:
                    from datasets import load_dataset as _ld
                    ds = _ld("kjgpta/WhoDunIt", split="train")
                    meta = ds[story_index].get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            try:
                                meta = ast.literal_eval(meta)
                            except Exception:
                                meta = {}
                    name_id_map = meta.get("name_id_map", {}) if isinstance(meta, dict) else {}
                    keys = [str(k).strip() for k in name_id_map.keys() if str(k).strip()]
                    if not keys:
                        print("⚠️  No metadata keys found; skipping story")
                        return False
                    import pandas as pd
                    rows = [{"char_id": i+1, "name": nm, "story_title": story_name} for i, nm in enumerate(keys)]
                    pd.DataFrame(rows).to_csv(chars_file, index=False)
                    print(f"🧰 Built characters CSV from metadata: {chars_file}")
                except Exception as e:
                    print(f"⚠️  Failed to build characters from metadata: {e}; skipping story")
                    return False
            else:
                if not run_command(
                    f"python3 character_extraction.py -s {story_index} -c 2000 -m llama3.2",
                    "Character Extraction"
                ):
                    print("❌ Character extraction failed!")
                    return False
                
                # Move the generated file to our output directory
                # The script saves to char/ directory, so we need to move it
                import shutil
                char_dir = Path("char")
                if char_dir.exists():
                    for csv_file in char_dir.glob("*.csv"):
                        shutil.move(str(csv_file), str(chars_file))
                        break  # Move the first CSV file found
        
        # Step 2: Alias building (CSV only)
        aliases_csv = output_dir / f"{story_name}_aliases.csv"
        if not run_command(
            f"python3 alias_builder.py --input {shlex.quote(str(chars_file))} --csv-out {shlex.quote(str(aliases_csv))}",
            "Alias Building"
        ):
            print("❌ Alias building failed!")
            return False
        
        # Step 2.5: Skip alias remapping (not needed for CSV-only output)
        print("Skipping alias remapping (CSV output only)...")

        # Step 3: Interaction extraction (adjusted to use CSV)
        interactions_file = output_dir / f"{story_name}_interactions.csv"
        if not run_command(
            f"python3 interaction_extraction.py --story {shlex.quote(str(temp_story_file))} --aliases {shlex.quote(str(aliases_csv))} --output {shlex.quote(str(interactions_file))}",
            "Interaction Extraction"
        ):
            print("❌ Interaction extraction failed!")
            return False
        
        print(f"\n{'='*80}")
        print(f"🎉 Character Data Collection Pipeline Completed Successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Files created:")
        print(f"  • 👥 Characters: {chars_file}")
        print(f"  • 🔗 Aliases: {aliases_csv}")  
        print(f"  • 📈 Interactions: {interactions_file}")
        print(f"{'='*80}")
        print(f"✨ Ready for graph analysis and AI detective insights!")
        
        return True
        
    finally:
        # Clean up temporary story file
        if temp_story_file.exists():
            temp_story_file.unlink()
            print(f"🧹 Cleaned up temporary story file")


def main():
    parser = argparse.ArgumentParser(description="Run complete character analysis pipeline")
    parser.add_argument("story_index", type=int, help="Story index from WhoDunIt dataset (0-based)")
    parser.add_argument("--story-name", help="Custom name for the story (default: use title from dataset)")
    parser.add_argument("--clean", action="store_true", help="Clean up existing output files before running")
    parser.add_argument("--skip-extract", action="store_true", help="Skip LLM character extraction; build from metadata if needed")
    parser.add_argument("--reuse-chars", action="store_true", help="Reuse existing chars CSV if present (default: regenerate)")
    
    args = parser.parse_args()
    
    # Clean up existing outputs if requested
    if args.clean:
        print("🧹 Cleaning up existing output files...")
        for pattern in ["*_chars_aliases.*", "*_interactions*.csv"]:
            os.system(f"rm -f {pattern}")
        print("✅ Cleanup completed")
    
    # Run pipeline
    success = run_pipeline(args.story_index, args.story_name, skip_extract=args.skip_extract, reuse_chars=args.reuse_chars)
    
    if not success:
        print("❌ Pipeline failed!")
        sys.exit(1)
    
    print("🎉 All done!")


if __name__ == "__main__":
    main()


