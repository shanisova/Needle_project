#!/usr/bin/env python3
"""
List Unique Story Titles from the WhoDunIt Dataset

Loads the `kjgpta/WhoDunIt` dataset (train split), extracts story titles,
deduplicates them, sorts them, and prints one per line.
"""

from datasets import load_dataset


def extract_unique_titles() -> list[str]:
    dataset = load_dataset("kjgpta/WhoDunIt", split="train")
    titles: list[str] = []
    for index, example in enumerate(dataset):
        # Prefer canonical "title"; fall back to "Title" if needed
        title_value = example.get("title") or example.get("Title")
        if title_value is None:
            # As a last resort, provide a stable placeholder to avoid None values
            title_value = f"Story_{index}"
        titles.append(str(title_value))
    return sorted(set(titles))


def main() -> None:
    unique_titles = extract_unique_titles()
    for t in unique_titles:
        print(t)


if __name__ == "__main__":
    main()


