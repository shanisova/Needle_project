#!/usr/bin/env python3
"""
Alias Builder (filters -> containment + surname-aware grouping, regex-only)

Usage:
  python3 alias_builder.py \
    --input /path/Story_chars.csv \
    --json-out /path/aliases.json \
    --csv-out  /path/aliases.csv \
    --drop-name "Somerset House" --drop-name "King's Bench-walk"

Input CSV must have a 'name' column (description/story_title optional).
Output JSON: { canonical_name: [aliases...] }
Output CSV : canonical_name,alias_name

Design:
- Hard filters: drop pronouns, non-capitalized, places/org phrases, fragments, possessives.
- Grouping:
  * Seed clusters with longer/more-complete names.
  * Merge if:
      - surnames match (normalized),
      - titles are compatible (Dr≡Doctor etc., block gender mismatches),
      - AND (content-token containment OR overlap with initials allowed).
  * Post-pass collapses cluster canonicals with same rules.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import json as _json
import ast as _ast
try:
    from datasets import load_dataset as _load_dataset
except Exception:
    _load_dataset = None

# =========================
# Lexicons / constants
# =========================
WORD = r"[A-Za-z][A-Za-z'\-]*"

PRONOUNS = {
    "i","me","my","mine","myself","you","your","yours","yourself","yourselves",
    "he","him","his","himself","she","her","hers","herself","it","its","itself",
    "we","us","our","ours","ourselves","they","them","their","theirs","themselves"
}

TITLE_WORDS = {
    "mr","mrs","ms","miss","dr","doctor","prof","professor","sir","lady",
    "inspector","detective","superintendent","supt","sergeant","sgt","constable",
    "chief","officer","judge","justice","father","brother","sister","reverend","rev",
}

MALE_TITLES   = {"mr","sir","father","brother"}
FEMALE_TITLES = {"mrs","ms","miss","lady","sister"}

# Common place-ish tokens (loose; we still allow clear Person forms to survive)
PLACE_TERMS = {
    "wood","cottage","walk","lane","house","bazaar","museum","church","street","road","alley",
    "passage","bridge","yard","bench","court","park","college","university","hotel","inn",
    "somerset","bench-walk","square","lane"
}

ORG_PHRASES = {
    "scotland yard","scotland yard people","madame tussaud's",
    "my friend","my brother","my sister","my father","my mother",
    "my uncle","my aunt","my cousin"
}

SUFFIX_WORDS = {"jr","sr","esq","esquire","ii","iii","iv"}
TITLE_EQUIV  = {"dr":"doctor","prof":"professor","sgt":"sergeant","supt":"superintendent"}

# Common sentence/function words – used to reject full-sentence fragments
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","than","that","this","those","these",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","her","our","their",
    "is","are","was","were","be","been","being","am","do","does","did","have","has","had","shall","will","would",
    "to","of","in","on","at","by","for","from","with","about","over","under","into","onto","without","within",
    "as","not","no","yes","oh","ay","nay","nowt"
}

# Allow some formal patterns after a leading "The" (e.g., titles or ordinals)
ALLOW_AFTER_THE = {
    "great","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"
}

# =========================
# Normalization / token helpers
# =========================
def _normalize_unicode(s: str) -> str:
    s = (s or "")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\u00A0", " ")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _tokens(s: str) -> List[str]:
    return re.findall(WORD, _normalize_unicode(s))

def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^A-Za-z]", "", (s or "")).lower()

def starts_with_capital(s: str) -> bool:
    for ch in s.strip():
        if ch.isalpha():
            return ch.isupper()
    return False

def _norm_title(tok: Optional[str]) -> Optional[str]:
    if not tok: return None
    t = tok.lower().strip(".")
    t = TITLE_EQUIV.get(t, t)
    return t if t in TITLE_WORDS else None

def title_of(name: str) -> Optional[str]:
    toks = _tokens(name)
    if not toks: return None
    return _norm_title(toks[0])

def content_tokens(name: str) -> List[str]:
    """Lowercased tokens with titles removed; suffixes dropped; keep initials."""
    toks = _tokens(name)
    if not toks: return []
    if _norm_title(toks[0]):
        toks = toks[1:]
    ct = [_norm(t) for t in toks if _norm(t)]
    while ct and ct[-1] in SUFFIX_WORDS:
        ct.pop()
    return ct

def split_name(name: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """(title, given_tokens, surname) after normalization/suffix removal."""
    ct = content_tokens(name)
    t  = title_of(name)
    if not ct:
        return (_norm_title(t), [], None)
    if len(ct) == 1:
        return (_norm_title(t), [], ct[0])
    return (_norm_title(t), ct[:-1], ct[-1])

# =========================
# Fuzzy matching
# =========================
def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n+1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,         # deletion
                dp[j-1] + 1,       # insertion
                prev + (a[i-1] != b[j-1])  # substitution
            )
            prev = cur
    return dp[-1]

def token_match(a: str, b: str) -> bool:
    """Exact, initial-vs-full, light fuzz (<=1 for len>=5; <=2 for len>=7)."""
    if not a or not b: return False
    if a == b: return True
    # initial vs full
    if len(a) == 1 and b.startswith(a): return True
    if len(b) == 1 and a.startswith(b): return True
    # fuzz for OCR/typos
    L = levenshtein(a, b)
    if len(a) >= 7 and len(b) >= 7 and L <= 2: return True
    if len(a) >= 5 and len(b) >= 5 and L <= 1: return True
    return False

def tokens_contained(small: List[str], big: List[str]) -> bool:
    """Every token in small can be matched by some token in big (order-agnostic)."""
    if not small: return False
    pool = list(big)
    for t in small:
        hit = False
        for i, u in enumerate(pool):
            if token_match(t, u):
                del pool[i]
                hit = True
                break
        if not hit:
            return False
    return True

# =========================
# Filters
# =========================
def valid_name(name: str) -> bool:
    if not name: return False
    raw = _normalize_unicode(name)
    if not raw: return False

    # pronouns
    if raw.lower() in PRONOUNS:
        return False

    # must start with capital (first alpha)
    if not starts_with_capital(raw):
        return False

    low = raw.lower()
    if any(p in low for p in ORG_PHRASES):
        return False

    toks = _tokens(raw)
    if not toks:
        return False

    # Drop single-word 'God'
    if len(toks) == 1 and toks[0].lower() == "god":
        return False

    # Drop leading "The ..." unless followed by allowed patterns (e.g., The Great, The Second)
    if toks and toks[0].lower() == "the":
        if len(toks) < 2 or toks[1].lower() not in ALLOW_AFTER_THE:
            return False

    # drop lone/bare titles
    if len(toks) == 1 and _norm_title(toks[0]):
        return False

    # trailing dash fragments or lone initial like "J." / "J"
    if re.search(r"[—–-]\s*$", raw) or re.fullmatch(r"[A-Za-z]\.?", raw.strip()):
        return False

    # possessive "John's"
    if re.search(r"[A-Za-z][’']s\b", raw):
        return False

    # place-like: if it contains a place token and is not clearly First Last, drop
    if any(t.lower() in PLACE_TERMS for t in toks):
        caps = [t for t in toks if t and t[0].isupper()]
        if not (len(toks) >= 2 and len(caps) >= 2):
            return False

    # reject names that repeat the same content token (e.g., "Freeman Dr Freeman")
    ct = content_tokens(raw)
    if ct and len(set(ct)) < len(ct):
        return False

    # HARD FILTER: likely full sentences
    # - Too many content tokens (e.g., >=5)
    if len(ct) >= 5:
        return False
    # - Contains sentence-ending punctuation and many tokens
    if ("." in name or "?" in name or "!" in name or "..." in name) and len(toks) >= 5:
        return False
    # - Too many stopwords relative to length (signals a sentence, not a name)
    sw = sum(1 for t in toks if t.lower() in STOPWORDS)
    if len(toks) >= 4 and sw >= 2:
            return False
    # - Low ratio of capitalized tokens for longer strings
    if len(toks) >= 5:
        caps = sum(1 for t in toks if t and t[0].isupper())
        if caps / max(1, len(toks)) < 0.6:
            return False

    return True

# =========================
# Title compatibility & surname match
# =========================
def titles_compatible(t1: Optional[str], t2: Optional[str]) -> bool:
    if not t1 or not t2:
        return True
    # normalize equivalences
    def norm(t: str) -> str:
        return TITLE_EQUIV.get(t, t)
    a, b = norm(t1), norm(t2)
    # gender block
    if (a in MALE_TITLES and b in FEMALE_TITLES) or (a in FEMALE_TITLES and b in MALE_TITLES):
        return False
    # identical or equivalent pairs are fine
    if a == b:
        return True
    # allow Dr≡Doctor, Prof≡Professor
    if {a,b} in ({"dr","doctor"}, {"prof","professor"}):
        return True
    # otherwise different roles: be conservative and block
    return False

def surnames_match(a: str, b: str) -> bool:
    ta, ga, sa = split_name(a)
    tb, gb, sb = split_name(b)
    return bool(sa and sb and token_match(sa, sb))

def given_compatible(ga: List[str], gb: List[str]) -> bool:
    """Empty allowed; otherwise each shorter-side token must match some longer-side token."""
    if not ga or not gb:
        return True
    A, B = (ga, gb) if len(ga) <= len(gb) else (gb, ga)
    used = [False] * len(B)
    for x in A:
        ok = False
        for j, y in enumerate(B):
            if used[j]: continue
            if token_match(x, y):
                used[j] = True
                ok = True
                break
        if not ok:
            return False
    return True

def persons_compatible(a: str, b: str) -> bool:
    """Surname must match exactly; titles must be compatible. Ignore given-name mismatch."""
    ta, ga, sa = split_name(a)
    tb, gb, sb = split_name(b)
    if not sa or not sb or sa != sb:
        return False
    if not titles_compatible(ta, tb):
        return False
    return True

# =========================
# Grouping
# =========================
def canonical_of(group: List[str]) -> str:
    """Most complete name: more content tokens, then longer, then lexicographic."""
    return max(group, key=lambda s: (len(content_tokens(s)), len(s), s))

def group_names(names: List[str]) -> Dict[str, List[str]]:
    """
    1) Seed clusters with longer names first.
    2) Attach others if:
         - persons_compatible (surname+title+given), AND
         - (tokens_contained OR tokens_contained reversed OR reasonable token overlap).
    3) Collapse clusters by comparing canonicals under same rule.
    """
    # sort seeds: (#content tokens desc, length desc)
    def seed_key(n: str) -> Tuple[int, int]:
        return (len(content_tokens(n)), len(n))
    pool = sorted({n.strip() for n in names if n.strip()}, key=seed_key, reverse=True)

    clusters: List[List[str]] = []

    # initial attach
    for name in pool:
        if any(name in c for c in clusters):
            continue
        base_ct = content_tokens(name)
        cluster = [name]
        for other in pool:
            if other == name or any(other in c for c in clusters) or other in cluster:
                continue
            if not persons_compatible(name, other):
                continue
            o_ct = content_tokens(other)
            if tokens_contained(o_ct, base_ct) or tokens_contained(base_ct, o_ct):
                cluster.append(other)
        clusters.append(cluster)

    # post-collapse (compare canonicals)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            can_i = canonical_of(clusters[i])
            j = i + 1
            while j < len(clusters):
                can_j = canonical_of(clusters[j])
                if persons_compatible(can_i, can_j):
                    ci = content_tokens(can_i); cj = content_tokens(can_j)
                    if tokens_contained(ci, cj) or tokens_contained(cj, ci):
                        clusters[i].extend(clusters[j])
                        del clusters[j]
                        changed = True
                        continue
                j += 1
            i += 1

    # build mapping with title clash handling
    mapping: Dict[str, List[str]] = {}
    for g in clusters:
        # detect title clashes in group
        titles = {}
        for n in g:
            t = title_of(n)
            nt = _norm_title(t)
            titles.setdefault(nt, []).append(n)
        # if more than one distinct non-None title, split group by title buckets
        non_none_titles = [k for k in titles.keys() if k is not None]
        if len(set(non_none_titles)) > 1:
            # create separate groups per title and a neutral bucket for None
            for key, members in titles.items():
                can = canonical_of(members)
                aliases = sorted({x for x in members if x != can},
                                 key=lambda s: (len(content_tokens(s)), len(s), s))
                mapping[can] = aliases
        else:
            can = canonical_of(g)
            aliases = sorted({x for x in g if x != can},
                             key=lambda s: (len(content_tokens(s)), len(s), s))
            mapping[can] = aliases
    return mapping

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Alias builder (regex-only)")
    ap.add_argument("--input","-i", required=True, help="CSV with 'name' column (description optional)")
    ap.add_argument("--json-out", help="Output JSON path")
    ap.add_argument("--csv-out", help="Output CSV path")
    ap.add_argument("--drop-name", action="append", default=None, help="Exact surface names to drop")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    if "name" not in df.columns:
        raise ValueError("CSV must contain a 'name' column")

    raw = [str(n).strip() for n in df["name"].tolist() if str(n).strip()]
    drop = set(args.drop_name or [])
    raw = [n for n in raw if n not in drop]

    # hard filters
    filtered = [n for n in raw if valid_name(n)]

    # optional metadata sanity filter (strict): drop any name whose content tokens
    # contain tokens not present in metadata keys (excluding titles). Apply ONLY if
    # metadata keys are present; otherwise skip the filter entirely.
    allowed_raw_keys: List[str] = []
    if _load_dataset is not None:
        try:
            ds = _load_dataset("kjgpta/WhoDunIt", split="train")
            # Infer story index by matching title from CSV or directory name
            story_idx = None
            story_title = None
            if "story_title" in df.columns and df["story_title"].dropna().shape[0] > 0:
                story_title = str(df["story_title"].dropna().iloc[0]).strip()
            if not story_title:
                # Fallback: use parent directory name
                parent = os.path.basename(os.path.dirname(args.input))
                story_title = parent.replace("_", " ").strip()
            # Find index by exact (case-insensitive) title match
            target = story_title.lower() if story_title else None
            if target:
                for i, row in enumerate(ds):
                    t = str(row.get("title", "")).strip().lower()
                    if t == target:
                        story_idx = i
                        break
            # If still unknown, skip metadata usage
            meta = ds[story_idx].get("metadata", {}) if story_idx is not None else {}
            if isinstance(meta, str):
                try:
                    meta = _json.loads(meta)
                except Exception:
                    try:
                        meta = _ast.literal_eval(meta)
                    except Exception:
                        meta = {}
            name_id_map = meta.get("name_id_map", {}) if isinstance(meta, dict) else {}
            # Build key/value pools (as they appear in metadata)
            allowed_raw_keys = [str(k) for k in name_id_map.keys() if _norm(str(k))]
            allowed_raw_vals = [str(v) for v in name_id_map.values() if _norm(str(v))]
            allowed_keys_norm = { _norm(k) for k in allowed_raw_keys }
            allowed_vals_norm = { _norm(v) for v in allowed_raw_vals }

            def keep_by_metadata(nm: str) -> bool:
                _, given, sur = split_name(nm)
                tokens = list(given)
                if sur:
                    tokens.append(sur)
                # every content token must be in allowed set exactly
                return all(tok in allowed for tok in tokens)

            # Decide whether to use keys or values based on which side matches more names
            def name_tokens_ok(nm: str, pool: set) -> bool:
                t, g, s = split_name(nm)
                toks = list(g)
                if s:
                    toks.append(s)
                return bool(toks) and all(tok in pool for tok in toks)

            matches_keys = sum(1 for n in filtered if name_tokens_ok(n, allowed_keys_norm)) if allowed_keys_norm else 0
            matches_vals = sum(1 for n in filtered if name_tokens_ok(n, allowed_vals_norm)) if allowed_vals_norm else 0

            use_side = None
            chosen_norm_pool: set = set()
            chosen_raw_pool: list = []
            if len(allowed_keys_norm) >= 5 or len(allowed_vals_norm) >= 5:
                if matches_keys >= matches_vals and len(allowed_keys_norm) >= 5:
                    use_side = "keys"
                    chosen_norm_pool = allowed_keys_norm
                    chosen_raw_pool = allowed_raw_keys
                elif len(allowed_vals_norm) >= 5:
                    use_side = "values"
                    chosen_norm_pool = allowed_vals_norm
                    chosen_raw_pool = allowed_raw_vals

            if use_side:
                before = len(filtered)
                filtered_after = [n for n in filtered if name_tokens_ok(n, chosen_norm_pool)]
                after = len(filtered_after)
                # Fallback: if the strict filter removes everything (or almost everything), skip it
                if after == 0 or after < max(2, int(0.05 * before)):
                    print(f"[metadata] {use_side} filter too aggressive ({after}/{before}); skipping for this story")
                else:
                    filtered = filtered_after
                    print(f"[metadata] Using {use_side}; kept {after}/{before} names after sanity filter")
            else:
                print("[metadata] Not enough keys for sanity filter; skipping")
        except Exception as e:
            print(f"[metadata] skipped: {e}")

    # grouping
    mapping = group_names(filtered)

    # Add singleton entries for metadata keys not present in mapping (ensure presence in CSV)
    # Add singleton entries from the chosen metadata side if available
    if 'chosen_raw_pool' in locals() and chosen_raw_pool:
        # Build a set of all token-level normals present in existing mapping (canonical + aliases)
        present_token_norms = set()
        def add_tokens(nm: str):
            for tok in content_tokens(nm):
                if tok:
                    present_token_norms.add(tok)
        for can, als in mapping.items():
            add_tokens(can)
            for a in als:
                add_tokens(a)

        added = 0
        for item in chosen_raw_pool:
            norm_item = _norm(item)
            if not norm_item:
                continue
            # Skip adding metadata singleton for bare 'the'
            if norm_item == "the":
                continue
            # Only add singleton if its token does NOT already appear anywhere in existing names
            if norm_item in present_token_norms:
                continue
            if item not in mapping:
                mapping[item] = []
                added += 1
        if added:
            print(f"[metadata] Added {added} singleton canonical entries from metadata {use_side}")

    # outputs
    base = os.path.splitext(os.path.basename(args.input))[0]
    # Only write CSV per user request
    csv_out  = args.csv_out  or base + "_aliases.csv"

    rows: List[Dict[str, str]] = []
    for c, als in mapping.items():
        # Always include a singleton row for the canonical
        rows.append({"canonical_name": c, "alias_name": ""})
        # Then include alias rows
        for a in als:
            rows.append({"canonical_name": c, "alias_name": a})
    pd.DataFrame(rows, columns=["canonical_name", "alias_name"]).to_csv(csv_out, index=False)

    print(f"Loaded: {len(raw)} | Kept after filters: {len(filtered)} | Groups: {len(mapping)}")
    print(f"Wrote: {csv_out}")

if __name__ == "__main__":
    main()
