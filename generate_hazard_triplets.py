#!/usr/bin/env python3
"""Generate 100 training triplets from hazards.json for contrastive learning."""

import json
import re
import random
from html import unescape

random.seed(42)

# Load hazards
with open("/home/kay/.cache/pf2e-codex/extract-pf2e-8.2.0/hazards.json") as f:
    data = json.load(f)

H = []
for h in data:
    name = h["name"]
    lvl = h["system"]["details"]["level"]["value"]
    desc = h["system"]["details"]["description"]
    disable = h["system"]["details"]["disable"]
    is_complex = h["system"]["details"]["isComplex"]
    stealth_dc = h["system"]["attributes"]["stealth"]["details"]
    ac = h["system"]["attributes"]["ac"]["value"]
    hp = h["system"]["attributes"]["hp"]["value"]
    hardness = h["system"]["attributes"]["hardness"]

    # Get trigger/effect from items
    trigger_text = ""
    effect_text = ""
    for item in h.get("items", []):
        itype = item.get("type", "")
        if itype == "action":
            adesc = item.get("system", {}).get("description", {}).get("value", "")

    def strip_html(t):
        return unescape(re.sub(r"<[^>]+>", "", t)).strip()

    desc_clean = strip_html(desc)
    disable_clean = strip_html(disable)

    # Build full text
    full_text = f"--- {name} (Level {lvl}) ---\n"
    full_text += f"Description: {desc_clean}\n"
    stealth_clean = strip_html(str(stealth_dc))
    full_text += f"Stealth: {stealth_clean}\n"
    full_text += f"Disable: {disable_clean}\n"
    if is_complex:
        full_text += f"AC: {ac}, HP: {hp}, Hardness: {hardness}\n"
    if trigger_text:
        full_text += f"Trigger: {strip_html(trigger_text)}\n"
    if effect_text:
        full_text += f"Effect: {strip_html(effect_text)}\n"
    full_text += f"Complex: {'Yes' if is_complex else 'Simple'}"

    H.append({
        "name": name,
        "level": lvl,
        "is_complex": is_complex,
        "text": full_text,
        "desc": desc_clean,
    })

print(f"Loaded {len(H)} hazards")

# Build indices by level
by_level = {}
for idx, h in enumerate(H):
    lvl = h["level"]
    by_level.setdefault(lvl, []).append(idx)

# Also index by complex/simple
simple_idxs = [i for i, h in enumerate(H) if not h["is_complex"]]
complex_idxs = [i for i, h in enumerate(H) if h["is_complex"]]

triplets = []
used_pairs = set()


def pick_neg(pos_idx):
    """Pick a hard negative for pos_idx."""
    pos = H[pos_idx]
    lvl = pos["level"]
    is_c = pos["is_complex"]

    candidates = []

    # 1. Same level, different name
    same_level = [i for i in by_level.get(lvl, []) if i != pos_idx]
    for i in same_level:
        candidates.append((i, 10))

    # 2. Same complexity type
    if is_c:
        for i in complex_idxs:
            if i != pos_idx:
                candidates.append((i, 5))
    else:
        for i in simple_idxs:
            if i != pos_idx:
                candidates.append((i, 5))

    # 3. Adjacent levels (+/- 1)
    for dlvl in [lvl - 1, lvl + 1]:
        for i in by_level.get(dlvl, []):
            if i != pos_idx:
                candidates.append((i, 3))

    # Filter out already used (pos, neg) pairs
    candidates = [(i, w) for i, w in candidates if (pos_idx, i) not in used_pairs]
    if not candidates:
        # fallback: any other hazard
        candidates = [(i, 1) for i in range(len(H)) if i != pos_idx and (pos_idx, i) not in used_pairs]

    if not candidates:
        return None

    # Weighted random choice
    items = list(candidates)
    weights = [w for _, w in items]
    total = sum(weights)
    r = random.random() * total
    cum = 0
    for idx, w in items:
        cum += w
        if r <= cum:
            return idx

    return items[-1][0] if items else None


# Generate triplets
max_per_hazard = 4  # at most 4 triplets per hazard to spread coverage

# First pass: generate triplets ensuring each hazard appears roughly evenly
attempts = 0
while len(triplets) < 100 and attempts < 500:
    attempts += 1
    # Pick a hazard that hasn't been used too much
    counts = {h["name"]: 0 for h in H}
    for t in triplets:
        counts[t["pos_name"]] = counts.get(t["pos_name"], 0) + 1

    # Weight towards hazards with fewer triplets
    pos_idxs = []
    for i, h in enumerate(H):
        c = counts[h["name"]]
        if c < max_per_hazard:
            pos_idxs.extend([i] * (max_per_hazard - c))
    if not pos_idxs:
        # all saturated, allow more
        pos_idxs = list(range(len(H)))

    pos_idx = random.choice(pos_idxs)
    neg_idx = pick_neg(pos_idx)
    if neg_idx is None:
        continue

    used_pairs.add((pos_idx, neg_idx))
    used_pairs.add((neg_idx, pos_idx))  # symmetric

    pos = H[pos_idx]
    neg = H[neg_idx]
    query = f"What does the {pos['name']} hazard do?"
    triplets.append({
        "query": query,
        "pos": pos["text"],
        "neg": neg["text"],
        "pos_name": pos["name"],
        "neg_name": neg["name"],
    })

triplets = triplets[:100]
print(f"Generated {len(triplets)} triplets")

# Write JSONL
outpath = "/mnt/data/projects/pf2e-codex/training_data/raw/hazards.jsonl"
with open(outpath, "w") as f:
    for t in triplets:
        out = {"query": t["query"], "pos": t["pos"], "neg": t["neg"]}
        f.write(json.dumps(out) + "\n")

print(f"Written to {outpath}")

# Summary stats
from collections import Counter
lvl_counts = Counter()
for t in triplets:
    lvl_counts[t["pos_name"]] += 1
print("\nTriplets per hazard:")
for name, cnt in sorted(lvl_counts.items(), key=lambda x: -x[1]):
    print(f"  {name:40s} x{cnt}")

# Verify no duplicate pairs
pair_set = set()
for t in triplets:
    pair = (t["pos_name"], t["neg_name"])
    if pair in pair_set:
        print(f"WARNING: duplicate pair {pair}")
    pair_set.add(pair)
print(f"\nUnique pairs: {len(pair_set)} / {len(triplets)}")

# Also write a summary text file
with open("/mnt/data/projects/training_data/raw/hazards_out.txt", "w") as f:
    f.write(f"Generated {len(triplets)} training triplets from hazards.json\n\n")
    for i, t in enumerate(triplets):
        f.write(f"=== Triplet {i+1} ===\n")
        f.write(f"Query: {t['query']}\n")
        f.write(f"Positive: {t['pos_name']} (Level {H[[h['name'] for h in H].index(t['pos_name'])]['level']})\n")
        f.write(f"Negative: {t['neg_name']} (Level {H[[h['name'] for h in H].index(t['neg_name'])]['level']})\n")
        f.write(f"Pair: ({t['pos_name']}, {t['neg_name']})\n\n")

print(f"\nSummary written to /mnt/data/projects/training_data/raw/hazards_out.txt")
