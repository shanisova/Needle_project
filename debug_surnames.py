#!/usr/bin/env python3
"""
Debug script to test surname matching
"""

from alias_builder import split_name_parts, surnames_compatible, _norm

def test_surname_matching():
    names = ["DArblay", "Darblay", "D'Arblay", "d'arblay"]
    
    print("Testing surname matching:")
    print("=" * 50)
    
    for name in names:
        title, given, surname = split_name_parts(name)
        print(f"Name: {name}")
        print(f"  Title: {title}")
        print(f"  Given: {given}")
        print(f"  Surname: {surname}")
        print(f"  Normalized surname: {_norm(surname) if surname else 'None'}")
        print()
    
    print("Testing compatibility:")
    print("=" * 50)
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[i+1:], i+1):
            compatible = surnames_compatible(name1, name2)
            print(f"{name1} <-> {name2}: {compatible}")

if __name__ == "__main__":
    test_surname_matching()
