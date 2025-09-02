#!/usr/bin/env python3
"""
Alias Builder (filters -> containment + overlap + surname-awareness -> optional LLM refinement)

Usage examples:
  # regex-only
  python3 alias_builder.py \
    --input /path/Story_chars.csv \
    --json-out /path/aliases.json \
    --csv-out  /path/aliases.csv

  # with Ollama refinement
  python3 alias_builder.py \
    --input /path/Story_chars.csv \
    --post-llm --ollama-model llama3.2

  # with OpenAI refinement (env var OPENAI_API_KEY required)
  python3 alias_builder.py \
    --input /path/Story_chars.csv \
    --post-llm --openai-model gpt-4o-mini
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================
# Constants & Lexicons
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
    "commissioner","coroner","foreman"
}

# Gendered titles for blocking mismatches (neutral titles omitted)
MALE_TITLES   = {"mr","sir","father","brother"}
FEMALE_TITLES = {"mrs","ms","miss","lady","sister"}

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

SUFFIX_WORDS = {"jr", "sr", "esq", "esquire", "ii", "iii", "iv"}
ROLE_HEADS   = {"coroner", "inspector", "commissioner", "foreman"}
TITLE_EQUIV  = {"dr": "doctor", "prof": "professor"}  # normalize variants

# =========================
# Basic helpers
# =========================
def _normalize_unicode(s: str) -> str:
    """Unify quotes/dashes/commas, strip zero-widths/NBSP, drop parens and their contents, collapse whitespace."""
    s = (s or "")
    # strip zero-widths & BOM
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    # NBSP -> space
    s = s.replace("\u00A0", " ")
    # quotes
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    # dashes
    s = s.replace("\u2014", " ").replace("\u2013", " ").replace("\u2212", "-")
    # remove parentheses and their contents completely (multiple passes to handle nested)
    while re.search(r"\([^)]*\)", s):
        s = re.sub(r"\([^)]*\)", "", s)
    # punctuation -> spaces
    s = re.sub(r"[,\.;:]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _tokens(s: str) -> List[str]:
    s = _normalize_unicode(s)
    return re.findall(WORD, s)

def _norm(s: Optional[str]) -> str:
    """Normalize by removing all non-alphabetic characters and lowercasing."""
    return re.sub(r"[^A-Za-z]", "", (s or "")).lower()

def _norm_punctuation(s: str) -> str:
    """Normalize punctuation differences (dots, apostrophes, etc.) for matching."""
    s = s.lower()
    # Normalize apostrophes and quotes
    s = s.replace("'", "'").replace("`", "'").replace("'", "'")
    # Remove dots from titles and initials
    s = re.sub(r'\b([a-z])\.', r'\1', s)
    # Remove other punctuation but keep spaces
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()

def starts_with_capital(s: str) -> bool:
    s = s.strip()
    for ch in s:
        if ch.isalpha():
            return ch.isupper()
    return False

def _norm_title(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = t.lower().strip(".")
    t = TITLE_EQUIV.get(t, t)
    return t if t in TITLE_WORDS else None

def title_of(name: str) -> Optional[str]:
    toks = _tokens(name)
    if not toks:
        return None
    t0 = toks[0].lower().strip(".")
    t0 = TITLE_EQUIV.get(t0, t0)
    return t0 if t0 in TITLE_WORDS else None

def titles_incompatible(t1: Optional[str], t2: Optional[str]) -> bool:
    """Block if titles are incompatible (different gender, different roles)"""
    if not t1 and not t2:
        return False  # Both have no title - compatible
    
    if not t1 or not t2:
        return False  # One has title, one doesn't - allow if rest of name matches
    
    # Gender incompatibility
    a_m = t1 in MALE_TITLES; a_f = t1 in FEMALE_TITLES
    b_m = t2 in MALE_TITLES; b_f = t2 in FEMALE_TITLES
    if (a_m and b_f) or (a_f and b_m):
        return True
    
    # Different titles (even if same gender) - block unless they're equivalent
    if t1 != t2:
        # Allow some title equivalences
        if (t1 == "dr" and t2 == "doctor") or (t1 == "doctor" and t2 == "dr"):
            return False
        if (t1 == "prof" and t2 == "professor") or (t1 == "professor" and t2 == "prof"):
            return False
        if (t1 == "supt" and t2 == "superintendent") or (t1 == "superintendent" and t2 == "supt"):
            return False
        if (t1 == "sgt" and t2 == "sergeant") or (t1 == "sergeant" and t2 == "sgt"):
            return False
        if (t1 == "rev" and t2 == "reverend") or (t1 == "reverend" and t2 == "rev"):
            return False
        # Different titles - block
        return True
    
    return False  # Same title - compatible

# =========================
# Tokenization / parts
# =========================
def content_tokens(name: str) -> List[str]:
    """
    Lowercased tokens with titles removed, suffixes dropped, punctuation normalized.
    Keep initials as 1-letter tokens and allow initial-vs-full matching.
    """
    toks = _tokens(name)
    if not toks:
        return []
    # strip leading title
    lead = toks[0].lower().strip(".")
    lead = TITLE_EQUIV.get(lead, lead)
    if lead in TITLE_WORDS:
        toks = toks[1:]

    toks = [_norm(t) for t in toks if _norm(t)]
    if not toks:
        return []

    # drop trailing suffixes (allow multiple)
    while toks and toks[-1] in SUFFIX_WORDS:
        toks.pop()
    return toks

def split_name_parts(name: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """
    Returns (title, given_tokens, surname) after normalization/suffix removal.
    Surname is the last remaining token when present.
    """
    raw_toks = _tokens(name)
    title = None
    if raw_toks:
        title = _norm_title(raw_toks[0])

    toks = content_tokens(name)
    if not toks:
        return (title, [], None)
    if len(toks) == 1:
        return (title, [], toks[0])  # single token => assume surname

    given = toks[:-1]
    surname = toks[-1]
    return (title, given, surname)

# =========================
# Matching primitives
# =========================
def levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance (for short name tokens)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]

def token_matches(a: str, b: str) -> bool:
    """
    Allow exact, initial-vs-full, punctuation-normalized, and small-typo fuzzy (edit distance <= 1 for len>=5).
    """
    if not a or not b:
        return False
    if a == b:
        return True
    
    # Check with punctuation normalization
    a_norm = _norm_punctuation(a)
    b_norm = _norm_punctuation(b)
    if a_norm == b_norm:
        return True
    
    # initial vs full
    if len(a) == 1 and b.startswith(a):
        return True
    if len(b) == 1 and a.startswith(b):
        return True
    
    # light fuzziness for likely OCR/typos on longer tokens
    if len(a) >= 5 and len(b) >= 5 and levenshtein(a, b) <= 1:
        return True
    return False

def full_tokens(name: str) -> List[str]:
    """
    Title (normalized) + content tokens (suffixes removed).
    Titles count toward the overlap (e.g., 'doctor').
    """
    t = _norm_title(title_of(name))
    ct = content_tokens(name)
    return ([t] if t else []) + ct

def token_overlap_count(a_tokens: List[str], b_tokens: List[str]) -> int:
    """
    Count of matching tokens between two lists (set-like, order-agnostic), using token_matches.
    """
    if not a_tokens or not b_tokens:
        return 0
    A = list(dict.fromkeys(a_tokens))  # dedupe, preserve order-ish
    B = list(dict.fromkeys(b_tokens))
    used = [False] * len(B)
    count = 0
    for x in A:
        for j, y in enumerate(B):
            if used[j]:
                continue
            if token_matches(x, y):
                used[j] = True
                count += 1
                break
    return count

def initials_compatible(given_a: List[str], given_b: List[str]) -> bool:
    """
    Given names compatible if each shorter-side token matches something on the longer side
    (exact or initial vs full). Either side may be empty.
    """
    if not given_a or not given_b:
        return True
    A, B = (given_a, given_b) if len(given_a) <= len(given_b) else (given_b, given_a)
    used = [False] * len(B)
    for a in A:
        hit = False
        for i, b in enumerate(B):
            if used[i]:
                continue
            if token_matches(a, b):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True

def surnames_compatible(name_a: str, name_b: str) -> bool:
    """
    True if surnames match and given names are compatible (or absent/initials),
    and titles aren't gender-incompatible.
    """
    ta, ga, sa = split_name_parts(name_a)
    tb, gb, sb = split_name_parts(name_b)
    if not sa or not sb or sa != sb:
        return False
    if titles_incompatible(ta, tb):
        return False
    
    # Special rule: if one has a single-letter given name and the other has a full given name,
    # check if the single letter is the first letter of the full name
    if len(ga) == 1 and len(gb) == 1:
        ga_token = ga[0]
        gb_token = gb[0]
        if len(ga_token) == 1 and len(gb_token) > 1:
            # ga is single letter, gb is full name
            if gb_token.lower().startswith(ga_token.lower()):
                return True
        elif len(gb_token) == 1 and len(ga_token) > 1:
            # gb is single letter, ga is full name
            if ga_token.lower().startswith(gb_token.lower()):
                return True
    
    return initials_compatible(ga, gb)

def should_merge_by_overlap(a: str, b: str) -> bool:
    """
    - If min token length == 1: require >=1 overlap (containment) — lets 'Artemus' merge with 'Dr Artemus'.
    - Else (multi-token): require >=2 overlapping tokens AND that surnames are compatible.
      (Prevents 'Dr Simon' drifting into 'Gray' cluster via loose 2-token overlaps.)
    """
    at = full_tokens(a)
    bt = full_tokens(b)
    overlap = token_overlap_count(at, bt)

    if min(len(at), len(bt)) <= 1:
        return overlap >= 1

    if overlap < 2:
        return False
    return surnames_compatible(a, b)

def content_subset(small: List[str], big: List[str]) -> bool:
    """
    True if every token in 'small' can be matched by some token in 'big' (order-agnostic).
    """
    if not small:
        return False
    pool = list(big)
    for t in small:
        hit = False
        for i, u in enumerate(pool):
            if token_matches(t, u):
                hit = True
                del pool[i]
                break
        if not hit:
            return False
    return True

# =========================
# Filters
# =========================
def valid_name(name: str) -> bool:
    if not name:
        return False
    raw = _normalize_unicode(name).strip()
    if not raw:
        return False

    # pronouns
    if raw.lower() in PRONOUNS:
        return False

    # Drop generic "The Something" unless it's a known role head (e.g., The Coroner)
    if raw.lower().startswith("the "):
        head_tokens = _tokens(raw[4:])
        if not head_tokens or head_tokens[0].lower() not in ROLE_HEADS:
            return False

    # must start with capital letter (first alpha char)
    if not starts_with_capital(raw):
        return False

    low = raw.lower()
    if any(p in low for p in ORG_PHRASES):
        return False

    toks = _tokens(raw)
    if not toks:
        return False

    # places / location-like
    if any(t.lower() in PLACE_TERMS for t in toks):
        # allow if clearly "Firstname Lastname" (2+ tokens and both capitalized)
        caps = [t for t in toks if t and t[0].isupper()]
        if not (len(toks) >= 2 and len(caps) >= 2):
            return False

    # trailing dash fragments or lone initial
    if re.search(r"[—–-]\s*$", raw) or re.fullmatch(r"[A-Za-z]\.?$", raw):
        return False

    # possessive surface ("John's")
    if re.search(r"[A-Za-z][’']s\b", raw):
        return False

    # drop bare roles/titles (no real name tokens)
    toks_no_title = _tokens(raw)
    if toks_no_title:
        head = _norm(toks_no_title[0])
        if len(toks_no_title) == 1 and (head in TITLE_WORDS or head in ROLE_HEADS):
            return False
        # if all tokens are titles/roles (e.g., "The Superintendent")
        if all(_norm(t) in TITLE_WORDS or _norm(t) in ROLE_HEADS for t in toks_no_title):
            return False

    return True

# =========================
# Grouping by containment + overlap + surname awareness
# =========================
def canonical_of(group: List[str]) -> str:
    """Pick the 'most complete' name: more content tokens, then longer, then lexicographic."""
    # Use normalized names for comparison to avoid parentheses affecting selection
    return max(group, key=lambda s: (len(content_tokens(_normalize_unicode(s))), len(_normalize_unicode(s)), s))

def check_group_title_consistency(group: List[str]) -> bool:
    """Check if all titles in a group are compatible. If canonical has no title, check aliases for mismatches."""
    if len(group) <= 1:
        return True
    
    canonical = canonical_of(group)
    canonical_title = _norm_title(title_of(canonical))
    
    # If canonical has no title, check if aliases have conflicting titles
    if not canonical_title:
        alias_titles = set()
        for name in group:
            if name != canonical:
                title = _norm_title(title_of(name))
                if title:
                    alias_titles.add(title)
        
        # If multiple different titles in aliases, block the group
        if len(alias_titles) > 1:
            return False
    
    return True

def group_by_containment(names: List[str]) -> Dict[str, List[str]]:
    """
    Start from longer names; attach shorter names whose content tokens are fully contained,
    OR whose token overlap satisfies rules, OR whose surnames match with compatible given names.
    Always block merges when titles are gender-incompatible.
    Then post-collapse clusters if canonical pairs satisfy the same checks.
    """
    def seed_key(n: str) -> Tuple[int, int]:
        return (len(content_tokens(n)), len(n))

    remaining = sorted({n.strip() for n in names if n.strip()}, key=seed_key, reverse=True)
    clusters: List[List[str]] = []

    # --- initial seeding/attachment
    for name in remaining:
        if any(name in c for c in clusters):
            continue
        base_ct = content_tokens(name)
        base_title = _norm_title(title_of(name))
        cluster = [name]

        for other in remaining:
            if other == name or any(other in c for c in clusters):
                continue

            o_ct = content_tokens(other)
            o_title = _norm_title(title_of(other))

            # HARD BLOCK: incompatible titles
            if titles_incompatible(base_title, o_title):
                continue

            if (
                should_merge_by_overlap(name, other) or
                content_subset(o_ct, base_ct) or
                content_subset(base_ct, o_ct) or
                surnames_compatible(name, other)
            ):
                cluster.append(other)

        clusters.append(cluster)

    # --- post-merge collapse (handles stragglers and near-dupes)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            can_i = canonical_of(clusters[i])
            ct_i = content_tokens(can_i)
            t_i = _norm_title(title_of(can_i))
            j = i + 1
            while j < len(clusters):
                can_j = canonical_of(clusters[j])
                ct_j = content_tokens(can_j)
                t_j = _norm_title(title_of(can_j))

                # HARD BLOCK: incompatible titles
                if titles_incompatible(t_i, t_j):
                    j += 1
                    continue

                if (
                    should_merge_by_overlap(can_i, can_j) or
                    content_subset(ct_i, ct_j) or
                    content_subset(ct_j, ct_i) or
                    surnames_compatible(can_i, can_j)
                ):
                    clusters[i].extend(clusters[j])
                    del clusters[j]
                    changed = True
                else:
                    j += 1
            i += 1

    # --- build mapping
    mapping: Dict[str, List[str]] = {}
    for g in clusters:
        # Check title consistency before adding to mapping
        if check_group_title_consistency(g):
            can = canonical_of(g)
            aliases = sorted({x for x in g if x != can}, key=lambda s: (len(content_tokens(s)), len(s), s))
            mapping[can] = aliases
        else:
            # Split inconsistent group into individual entries
            for name in g:
                mapping[name] = []
    return mapping

# =========================
# LLM refinement (optional, uses descriptions)
# =========================
def build_llm_prompt(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> str:
    cluster_lines = []
    for can, als in mapping.items():
        alln = [can] + als
        cluster_lines.append(" - " + ", ".join(alln))

    roster_lines = []
    for _, row in df.iterrows():
        nm = str(row.get("name", "")).strip()
        if not nm:
            continue
        desc = str(row.get("description", "") or "").strip()
        if desc:
            roster_lines.append(f" • {nm}: {desc}")
        else:
            roster_lines.append(f" • {nm}")

    prompt = (
        "You are refining clusters of character names for a mystery story. "
        "Each cluster must contain only surface forms (aliases) that refer to the SAME person.\n\n"
        "Use these rules:\n"
        "- Use the most complete and formal surface name as the canonical (title + first + last, when present).\n"
        "- Titles must be gender-consistent (Mr≠Miss/Mrs/Ms). 'Dr'≡'Doctor'.\n"
        "- Initials can match full first names only when the surname matches.\n"
        "- Do NOT merge people with different full first names unless there is strong evidence from descriptions they are the same person.\n"
        "- Remove non-people (places, organizations) if any slipped in.\n"
        "- If uncertain, keep groups separate.\n\n"
        "Current clusters:\n" + "\n".join(cluster_lines) + "\n\n"
        "All names with descriptions:\n" + "\n".join(roster_lines) + "\n\n"
        "Return STRICT JSON with this schema:\n"
        "{\n"
        '  "clusters": [\n'
        '    { "canonical": "<string>", "aliases": ["<string>", ...] },\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "- Do NOT include the canonical inside its own alias list.\n"
        "- Preserve exact casing and punctuation for all surface names.\n"
        "- No commentary—JSON only."
    )
    return prompt

def refine_with_llm(df: pd.DataFrame, mapping: Dict[str, List[str]],
                    ollama_model: Optional[str] = None,
                    openai_model: Optional[str] = None) -> Dict[str, List[str]]:
    prompt = build_llm_prompt(df, mapping)

    raw = None
    if ollama_model:
        try:
            from ollama import chat as ollama_chat
            resp = ollama_chat(model=ollama_model,
                               messages=[{"role":"user","content":prompt}],
                               options={"temperature":0})
            raw = getattr(resp, "message", {}).get("content", "") if hasattr(resp, "message") else str(resp)
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")

    if raw is None and openai_model:
        try:
            import openai
            client = openai.OpenAI()  # requires OPENAI_API_KEY
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                response_format={"type":"json_object"},
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            print(f"[LLM] OpenAI error: {e}")

    if not raw:
        return mapping

    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return mapping
        try:
            data = json.loads(m.group(0))
        except Exception:
            return mapping

    if not isinstance(data, dict) or "clusters" not in data or not isinstance(data["clusters"], list):
        return mapping

    refined: Dict[str, List[str]] = {}
    for item in data["clusters"]:
        if not isinstance(item, dict):
            continue
        can = item.get("canonical")
        aliases = item.get("aliases", [])
        if not isinstance(can, str) or not isinstance(aliases, list):
            continue
        # clean: drop canonical from aliases, dedupe, keep order-ish
        seen = set()
        out_aliases = []
        for a in aliases:
            s = str(a).strip()
            if not s or s == can or s in seen:
                continue
            seen.add(s)
            out_aliases.append(s)
        refined[can] = out_aliases
    return refined or mapping

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Alias builder (filters -> containment+overlap grouping -> optional LLM refinement)")
    ap.add_argument("--input","-i", required=True, help="CSV with at least 'name' column (optional: 'description')")
    ap.add_argument("--json-out", help="Output JSON path (canonical -> aliases)")
    ap.add_argument("--csv-out", help="Output CSV path (canonical,alias)")
    ap.add_argument("--post-llm", action="store_true", help="Refine clusters with LLM using descriptions")
    ap.add_argument("--ollama-model", default=None, help="Ollama model name (e.g., llama3.2)")
    ap.add_argument("--openai-model", default=None, help="OpenAI model name (e.g., gpt-4o-mini)")
    ap.add_argument("--drop-name", action="append", default=None, help="Exact name(s) to drop before processing")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)

    if "name" not in df.columns:
        raise ValueError("CSV must contain a 'name' column")

    # Collect names (strip empties)
    raw_names = [str(n).strip() for n in df["name"].tolist() if str(n).strip()]

    # Drop-list
    drop = set(args.drop_name or [])
    raw_names = [n for n in raw_names if n not in drop]

    # Preprocess: normalize names (remove parentheses, etc.)
    normalized_names = [_normalize_unicode(n) for n in raw_names]

    # Hard filters
    filtered = [n for n in normalized_names if valid_name(n)]

    # Group by containment + overlap + surname-awareness
    mapping = group_by_containment(filtered)

    # Optional LLM refinement (with descriptions)
    if args.post_llm and (args.ollama_model or args.openai_model):
        mapping = refine_with_llm(df, mapping, ollama_model=args.ollama_model, openai_model=args.openai_model)

    # Write outputs
    base = os.path.splitext(os.path.basename(args.input))[0]
    json_out = args.json_out or base + "_aliases.json"
    csv_out  = args.csv_out  or base + "_aliases.csv"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    rows = [{"canonical_name": c, "alias_name": a} for c, als in mapping.items() for a in als]
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    print(f"Loaded {len(raw_names)} | Kept {len(filtered)} | Groups {len(mapping)}")
    print(f"Wrote {json_out}\nWrote {csv_out}")

if __name__ == "__main__":
    main()
