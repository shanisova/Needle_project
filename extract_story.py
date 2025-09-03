#!/usr/bin/env python3
"""
Extract story text from WhoDunIt dataset
"""

from datasets import load_dataset

def extract_story_text(story_index: int = 0):
    """Extract story text from WhoDunIt dataset"""
    print("Loading WhoDunIt dataset...")
    dataset = load_dataset("kjgpta/WhoDunIt", split="train")
    
    if story_index >= len(dataset):
        print(f"Error: Story index {story_index} out of range (0..{len(dataset)-1})")
        return None
    
    story = dataset[story_index]
    story_title = story.get("title", f"Story_{story_index}")
    story_text = story.get("story", story.get("text", ""))
    
    print(f"Extracted story: {story_title}")
    print(f"Story length: {len(story_text)} characters")
    
    return story_title, story_text

if __name__ == "__main__":
    # Extract The D'Arblay Mystery (index 0)
    title, text = extract_story_text(0)
    
    if text:
        # Save to file
        clean_title = title.replace(' ', '_').replace("'", '')
        filename = f"{clean_title}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved story to: {filename}")
