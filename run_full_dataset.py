#!/usr/bin/env python3
"""
Run Pipeline Over Entire WhoDunIt Dataset

Processes all stories in the dataset and saves outputs in separate directories.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
import shlex


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        # Run without capturing output so we see real-time progress
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {description}:")
        print(f"Exit code: {e.returncode}")
        return False


def create_output_directory(story_name, story_index: int):
    """Create output directory for the story (suffixed with index to avoid collisions)."""
    # Clean story name for directory
    clean_name = story_name.replace(' ', '_').replace("'", '').replace('"', '').replace(':', '').replace(';', '')
    output_dir = Path("character_data") / f"{clean_name}_{story_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline_for_story(story_index: int, story_title: str, story_text: str, skip_extract: bool = True, reuse_chars: bool = False):
    """Run the complete pipeline for a single story."""
    print(f"\n{'='*80}")
    print(f"Processing Story {story_index}: {story_title}")
    print(f"Story length: {len(story_text)} characters")
    print(f"{'='*80}")
    
    # Create output directory (include index)
    output_dir = create_output_directory(story_title, story_index)
    print(f"Output directory: {output_dir}")
    
    # Create temporary story file for processing
    temp_story_file = output_dir / f"{story_title}_temp.txt"
    with open(temp_story_file, 'w', encoding='utf-8') as f:
        f.write(story_text)
    
    try:
        # Step 1: Character extraction (controlled by --reuse-chars and --skip-extract)
        chars_file = output_dir / f"{story_title}_chars.csv"
        if chars_file.exists() and reuse_chars:
            print(f"♻️  Reusing existing characters CSV: {chars_file}")
        else:
            if not skip_extract:
                if not run_command(
                    f"python3 character_extraction.py -s {story_index} -c 2000 -m llama3.2",
                    f"Character Extraction for {story_title}"
                ):
                    print(f"❌ Character extraction failed for {story_title}!")
                    return False
            else:
                # Build chars CSV from metadata keys as a fallback (no LLM)
                try:
                    from datasets import load_dataset as _ld
                    ds2 = _ld("kjgpta/WhoDunIt", split="train")
                    meta = ds2[story_index].get("metadata", {})
                    if isinstance(meta, str):
                        import json, ast
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
                        print(f"⚠️  No metadata keys found for {story_title}; skipping story")
                        return False
                    import pandas as pd
                    rows = [{"char_id": i+1, "name": nm, "story_title": story_title} for i, nm in enumerate(keys)]
                    pd.DataFrame(rows).to_csv(chars_file, index=False)
                    print(f"🧰 Built characters CSV from metadata: {chars_file}")
                except Exception as e:
                    print(f"⚠️  Failed to build characters from metadata for {story_title}: {e}; skipping story")
                    return False
            
            # Move the generated file to our output directory
            import shutil
            char_dir = Path("char")
            if char_dir.exists():
                for csv_file in char_dir.glob("*.csv"):
                    shutil.move(str(csv_file), str(chars_file))
                    break  # Move the first CSV file found
        
        # Step 2: Alias building (CSV only)
        aliases_csv = output_dir / f"{story_title}_aliases.csv"
        if not run_command(
            f"python3 alias_builder.py --input {shlex.quote(str(chars_file))} --csv-out {shlex.quote(str(aliases_csv))}",
            f"Alias Building for {story_title}"
        ):
            print(f"❌ Alias building failed for {story_title}!")
            return False
        
        # Step 3: Interaction extraction using CSV
        interactions_file = output_dir / f"{story_title}_interactions.csv"
        if not run_command(
            f"python3 interaction_extraction.py --story {shlex.quote(str(temp_story_file))} --aliases {shlex.quote(str(aliases_csv))} --output {shlex.quote(str(interactions_file))}",
            f"Interaction Extraction for {story_title}"
        ):
            print(f"❌ Interaction extraction failed for {story_title}!")
            return False
        
        print(f"\n{'='*80}")
        print(f"✅ Pipeline completed successfully for: {story_title}")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        print(f"  - Characters: {chars_file}")
        print(f"  - Aliases (CSV): {aliases_csv}")
        print(f"  - Interactions: {interactions_file}")
        print(f"{'='*80}")
        
        return True
        
    finally:
        # Clean up temporary story file
        if temp_story_file.exists():
            temp_story_file.unlink()
            print(f"🧹 Cleaned up temporary story file for {story_title}")


def run_full_dataset(start_index: int = 0, end_index: int = None, skip_existing: bool = True, skip_extract: bool = True, reuse_chars: bool = False):
    """Run pipeline over the entire WhoDunIt dataset."""
    print("Loading WhoDunIt dataset...")
    dataset = load_dataset("kjgpta/WhoDunIt", split="train")
    
    if end_index is None:
        end_index = len(dataset)
    
    print(f"Dataset contains {len(dataset)} stories")
    print(f"Processing stories {start_index} to {end_index-1}")
    
    successful = 0
    failed = 0
    
    for story_index in range(start_index, end_index):
        story = dataset[story_index]
        story_title = story.get("title", f"Story_{story_index}")
        story_text = story.get("story", story.get("text", ""))
        
        # Check if output already exists
        if skip_existing:
            output_dir = create_output_directory(story_title, story_index)
            interactions_file = output_dir / f"{story_title}_interactions.csv"
            if interactions_file.exists():
                print(f"⏭️  Skipping {story_title} (output already exists)")
                continue
        
        print(f"\n{'='*100}")
        print(f"PROCESSING STORY {story_index + 1}/{end_index}: {story_title}")
        print(f"{'='*100}")
        
        success = run_pipeline_for_story(story_index, story_title, story_text, skip_extract=skip_extract, reuse_chars=reuse_chars)
        
        if success:
            successful += 1
            print(f"✅ Successfully processed {story_title}")
        else:
            failed += 1
            print(f"❌ Failed to process {story_title}")
        
        # Add a separator between stories
        print(f"\n{'='*100}")
        print(f"PROGRESS: {successful} successful, {failed} failed")
        print(f"{'='*100}")
    
    print(f"\n{'='*100}")
    print(f"🎉 DATASET PROCESSING COMPLETE!")
    print(f"Total stories processed: {end_index - start_index}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: character_data/")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline over entire WhoDunIt dataset")
    parser.add_argument("--start", type=int, default=0, help="Starting story index (default: 0)")
    parser.add_argument("--end", type=int, help="Ending story index (exclusive, default: all stories)")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing outputs")
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
    
    # Run full dataset processing
    run_full_dataset(
        start_index=args.start,
        end_index=args.end,
        skip_existing=not args.no_skip,
        skip_extract=args.skip_extract,
        reuse_chars=args.reuse_chars
    )


if __name__ == "__main__":
    main()
