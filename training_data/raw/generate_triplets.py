#!/usr/bin/env python3
"""
Generate 100 hard-negative training triplets for cross-encoder reranker
using the PF2E actions.json pack.
"""

import json
import re
import random

random.seed(42)

# --- Load data ---
with open('/home/kay/.cache/pf2e-codex/extract-pf2e-8.2.0/actions.json') as f:
    raw = json.load(f)

actions = {}
for item in raw:
    name = item['name']
    desc_html = item['system']['description']['value']
    text = re.sub(r'<[^>]+>', '', desc_html)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    actions[name] = {
        'name': name,
        'text': text,
        'full': f"## {name}\n\n{text}",
        'category': item['system']['category'] or 'general',
        'action_type': item['system']['actionType']['value'],
        'traits': item['system']['traits']['value'],
    }

print(f"Loaded {len(actions)} actions")

# --- Define triplets ---
# Each entry: (query_template, pos_name, neg_name)
# Query template should use {pos_name} as placeholder

triplet_defs = [
    # === Stealth & Perception ===
    ("What does the {pos_name} action do?", "Hide", "Sneak"),
    ("How does {pos_name} work in PF2e?", "Sneak", "Hide"),
    ("Explain the {pos_name} action.", "Avoid Notice", "Sneak"),
    ("How do I use {pos_name}?", "Seek", "Sense Motive"),
    ("What is the {pos_name} action?", "Sense Motive", "Seek"),
    ("How does {pos_name} work?", "Search", "Investigate"),
    ("Tell me about {pos_name}.", "Investigate", "Search"),
    ("What does {pos_name} mean?", "Point Out", "Pointed Question"),
    ("How does {pos_name} work?", "Track", "Cover Tracks"),
    ("What is the {pos_name} action?", "Cover Tracks", "Track"),

    # === Movement ===
    ("How does {pos_name} work?", "Stride", "Step"),
    ("What does the {pos_name} action do?", "Step", "Stride"),
    ("Explain {pos_name}.", "Leap", "High Jump"),
    ("How does {pos_name} work?", "High Jump", "Long Jump"),
    ("Tell me about {pos_name}.", "Long Jump", "High Jump"),
    ("What does {pos_name} do?", "Climb", "Swim"),
    ("Explain the {pos_name} action.", "Swim", "Climb"),
    ("How does {pos_name} work?", "Fly", "Maneuver in Flight"),
    ("What is the {pos_name} action?", "Maneuver in Flight", "Fly"),
    ("How does {pos_name} work?", "Burrow", "Crawl"),
    ("Tell me about {pos_name}.", "Crawl", "Burrow"),
    ("What does {pos_name} do?", "Balance", "Tumble Through"),
    ("How does {pos_name} work?", "Tumble Through", "Balance"),
    ("Explain {pos_name}.", "Squeeze", "Crawl"),
    ("What is {pos_name}?", "Drop Prone", "Stand"),
    ("How does {pos_name} work?", "Stand", "Drop Prone"),

    # === Combat Maneuvers ===
    ("How does {pos_name} work in PF2e?", "Grapple", "Escape"),
    ("What does the {pos_name} action do?", "Shove", "Reposition"),
    ("Explain the {pos_name} action.", "Trip", "Grapple"),
    ("What is {pos_name}?", "Disarm", "Shove"),
    ("How does {pos_name} work?", "Reposition", "Disarm"),
    ("Tell me about {pos_name}.", "Force Open", "Disable a Device"),
    ("What does {pos_name} do?", "Escape", "Break Free"),

    # === Attack Actions ===
    ("How does {pos_name} work?", "Strike", "Devise a Stratagem"),
    ("What is the {pos_name} action?", "Devise a Stratagem", "Strike"),
    ("Explain {pos_name}.", "Flurry of Blows", "Strike"),
    ("What does {pos_name} do?", "Confident Finisher", "Basic Finisher"),
    ("How does {pos_name} work?", "Rage", "Arcane Cascade"),
    ("Tell me about {pos_name}.", "Overdrive", "Rage"),

    # === Defensive ===
    ("What does {pos_name} do?", "Raise a Shield", "Take Cover"),
    ("How does {pos_name} work?", "Take Cover", "Raise a Shield"),
    ("Explain the {pos_name} action.", "Defend", "Raise a Shield"),
    ("What is {pos_name}?", "Avert Gaze", "Take Cover"),
    ("How does {pos_name} work?", "Reactive Strike", "Retributive Strike"),
    ("Tell me about {pos_name}.", "Shield Block", None),  # fallback - not in actions
    ("What does {pos_name} do?", "Arrest a Fall", "Grab an Edge"),

    # === Reactions ===
    ("How does {pos_name} work?", "Aid", "Follow the Expert"),
    ("What is the {pos_name} action?", "Follow the Expert", "Aid"),
    ("Explain {pos_name}.", "Grab an Edge", "Arrest a Fall"),
    ("What does {pos_name} do?", "Delay", "Ready"),
    ("How does {pos_name} work?", "Ready", "Delay"),

    # === Healing / Medical ===
    ("How does {pos_name} work in PF2e?", "Treat Wounds", "Administer First Aid"),
    ("What does the {pos_name} action do?", "Administer First Aid", "Treat Wounds"),
    ("Explain {pos_name}.", "Treat Disease", "Treat Poison"),
    ("How does {pos_name} work?", "Treat Poison", "Treat Disease"),
    ("What is {pos_name}?", "Take a Breather", "Treat Wounds"),

    # === Social / Deception ===
    ("How does {pos_name} work?", "Demoralize", "Coerce"),
    ("What does {pos_name} do?", "Coerce", "Demoralize"),
    ("Explain {pos_name}.", "Make an Impression", "Request"),
    ("How does {pos_name} work?", "Request", "Make an Impression"),
    ("What is the {pos_name} action?", "Gather Information", "Gossip"),
    ("Tell me about {pos_name}.", "Gossip", "Gather Information"),
    ("What does {pos_name} do?", "Feint", "Create a Diversion"),
    ("How does {pos_name} work?", "Create a Diversion", "Feint"),
    ("Explain {pos_name}.", "Lie", "Impersonate"),
    ("What is {pos_name}?", "Impersonate", "Lie"),
    ("How does {pos_name} work?", "Perform", "Spin Tale"),
    ("What does {pos_name} do?", "Pointed Question", "Size Up"),
    ("Tell me about {pos_name}.", "Size Up", "Pointed Question"),
    ("What is {pos_name}?", "Taunt", "Demoralize"),

    # === Magic Actions ===
    ("How does {pos_name} work?", "Cast a Spell", "Sustain"),
    ("What does the {pos_name} action do?", "Sustain", "Cast a Spell"),
    ("Explain {pos_name}.", "Dismiss", "Release"),
    ("How does {pos_name} work?", "Release", "Dismiss"),
    ("What is {pos_name}?", "Identify Magic", "Recall Knowledge"),
    ("How does {pos_name} work?", "Learn a Spell", "Borrow an Arcane Spell"),
    ("Tell me about {pos_name}.", "Borrow an Arcane Spell", "Learn a Spell"),
    ("What does {pos_name} do?", "Refocus", "Drain Bonded Item"),
    ("How does {pos_name} work?", "Drain Bonded Item", "Refocus"),
    ("What is the {pos_name} action?", "Detect Magic", "Identify Magic"),

    # === Skill Actions ===
    ("How does {pos_name} work?", "Recall Knowledge", "Investigate"),
    ("What does {pos_name} do?", "Sense Direction", "Track"),
    ("Explain {pos_name}.", "Craft", "Repair"),
    ("How does {pos_name} work?", "Repair", "Craft"),
    ("What is {pos_name}?", "Earn Income", "Subsist"),
    ("How does {pos_name} work?", "Subsist", "Earn Income"),
    ("Tell me about {pos_name}.", "Decipher Writing", "Learn a Spell"),
    ("What does {pos_name} do?", "Disable a Device", "Pick a Lock"),
    ("How does {pos_name} work?", "Pick a Lock", "Disable a Device"),
    ("Explain {pos_name}.", "Force Open", "Pick a Lock"),
    ("What is {pos_name}?", "Palm an Object", "Conceal an Object"),
    ("How does {pos_name} work?", "Conceal an Object", "Palm an Object"),
    ("Tell me about {pos_name}.", "Steal", "Conceal an Object"),
    ("What does {pos_name} do?", "Interact", "Release"),

    # === Animal / Companion ===
    ("How does {pos_name} work?", "Command an Animal", "Command a Construct"),
    ("What is {pos_name}?", "Call Companion", "Command an Animal"),
    ("Explain {pos_name}.", "Mount", "Board"),

    # === Downtime ===
    ("How does {pos_name} work?", "Craft", "Earn Income"),
    ("What does {pos_name} do?", "Retraining", "Research"),
    ("Tell me about {pos_name}.", "Study", "Research"),
    ("How does {pos_name} work?", "Travel", "Hustle"),
    ("What is {pos_name}?", "Hustle", "Travel"),

    # === Exploration ===
    ("How does {pos_name} work?", "Scout", "Reconnoiter"),
    ("What does {pos_name} do?", "Investigate", "Search"),
    ("Explain {pos_name}.", "Fortify Camp", "Take a Breather"),

    # === Combat Options ===
    ("How does {pos_name} work?", "Escape", "Grapple"),
    ("What does {pos_name} do?", "Delay", "Ready"),
    ("Tell me about {pos_name}.", "Interact", "Release"),
]

# Check all actions exist
valid_defs = []
for q, pos_name, neg_name in triplet_defs:
    if pos_name not in actions:
        print(f"WARNING: pos '{pos_name}' not found!")
        continue
    if neg_name is not None and neg_name not in actions:
        print(f"WARNING: neg '{neg_name}' not found!")
        continue
    if neg_name is None:
        # Find a hard negative automatically
        cat = actions[pos_name]['category']
        act_type = actions[pos_name]['action_type']
        candidates = [n for n, a in actions.items() if n != pos_name and a['category'] == cat and a['action_type'] == act_type]
        if not candidates:
            candidates = [n for n, a in actions.items() if n != pos_name]
        neg_name = random.choice(candidates)
    valid_defs.append((q, pos_name, neg_name))

print(f"Generated {len(valid_defs)} valid triplet definitions")
print(f"First few: {[(d[1], d[2]) for d in valid_defs[:5]]}")

# Some definitions had neg_name as None (Shield Block) - let me handle that
# Shield Block is not an action in the pack (it's a feat), so let me replace it
# Let me check what we have for that entry and replace
for i, (q, pos_name, neg_name) in enumerate(valid_defs):
    if pos_name == "Shield Block":
        # Replace with a different triplet
        valid_defs[i] = ("What does {pos_name} do?", "Raise a Shield", "Defend")
        break

# --- Generate output ---
output = []
used_combos = set()
for q_template, pos_name, neg_name in valid_defs:
    combo = (pos_name, neg_name)
    if combo in used_combos:
        continue
    used_combos.add(combo)
    
    query = q_template.format(pos_name=pos_name)
    pos_full = actions[pos_name]['full']
    neg_full = actions[neg_name]['full']
    
    output.append({
        "query": query,
        "pos": pos_full,
        "neg": neg_full
    })

print(f"Total unique triplets generated: {len(output)}")

# If we have fewer than 100, add more
if len(output) < 100:
    print(f"Only {len(output)} unique triplets, generating more...")
    # Generate additional triplets from remaining actions
    used_pos = set(d['pos_name'] for d in valid_defs)
    all_names = list(actions.keys())
    
    more_defs = []
    need = 100 - len(output)
    random.shuffle(all_names)
    
    # Create more pairs
    additional_pairs = [
        ("How does {pos_name} work?", "Rage", "Arcane Cascade"),
        ("What does {pos_name} do?", "Hunt Prey", "Pursue a Lead"),
        ("Explain {pos_name}.", "Pursue a Lead", "Hunt Prey"),
        ("What is the {pos_name} action?", "Rally", "Encouraging Words"),
        ("How does {pos_name} work?", "Encouraging Words", "Rally"),
        ("Tell me about {pos_name}.", "Take Cover", "Hide"),
        ("What does {pos_name} do?", "Hide", "Take Cover"),
        ("How does {pos_name} work?", "Clue In", "Aid"),
        ("What is {pos_name}?", "Aid", "Clue In"),
        ("How does {pos_name} work?", "Crawl", "Squeeze"),
        ("Explain {pos_name}.", "Manifest Eidolon", "Call Companion"),
        ("What does {pos_name} do?", "Command a Construct", "Command an Animal"),
        ("How does {pos_name} work?", "Quick Alchemy", "Craft"),
        ("What is {pos_name}?", "Identify Alchemy", "Identify Magic"),
        ("Tell me about {pos_name}.", "Learn a Spell", "Decipher Writing"),
        ("What does {pos_name} do?", "Overdrive", "Arcane Cascade"),
        ("Explain {pos_name}.", "Rage", "Overdrive"),
        ("How does {pos_name} work?", "Lie", "Create a Diversion"),
        ("What is {pos_name}?", "Feint", "Lie"),
        ("How does {pos_name} work?", "Perform", "Compose Missive"),
        ("What does {pos_name} do?", "Compose Missive", "Perform"),
        ("Tell me about {pos_name}.", "Coerce", "Demoralize"),
        ("What is {pos_name}?", "Demoralize", "Coerce"),
        ("Explain {pos_name}.", "Cast a Spell", "Dismiss"),
        ("How does {pos_name} work?", "Dismiss", "Cast a Spell"),
        ("What does {pos_name} do?", "Sustain", "Sustain an Effect"),
        ("How does {pos_name} work?", "Sustain an Effect", "Sustain"),
        ("Tell me about {pos_name}.", "Treat Disease", "Administer First Aid"),
        ("What is {pos_name}?", "Treat Poison", "Administer First Aid"),
        ("How does {pos_name} work?", "Take a Breather", "Refocus"),
    ]
    
    for q, pos_name, neg_name in additional_pairs:
        if len(output) >= 100:
            break
        if pos_name in actions and neg_name in actions:
            combo = (pos_name, neg_name)
            if combo not in used_combos:
                used_combos.add(combo)
                query = q.format(pos_name=pos_name)
                output.append({
                    "query": query,
                    "pos": actions[pos_name]['full'],
                    "neg": actions[neg_name]['full']
                })

# If still under 100, pad with common actions generating queries about them
if len(output) < 100:
    print(f"Still only {len(output)}, padding more...")
    common_actions = [
        "Stride", "Step", "Strike", "Hide", "Sneak", "Seek", "Sense Motive",
        "Recall Knowledge", "Treat Wounds", "Raise a Shield", "Take Cover",
        "Demoralize", "Feint", "Grapple", "Shove", "Trip", "Aid",
        "Cast a Spell", "Sustain", "Dismiss", "Delay", "Ready",
        "Escape", "Climb", "Swim", "Fly", "Balance", "Tumble Through",
        "Interact", "Release", "Stand", "Drop Prone", "Leap", "Crawl"
    ]
    for n in common_actions:
        if len(output) >= 100:
            break
        # Find a hard negative that hasn't been used
        cat = actions[n]['category']
        act_type = actions[n]['action_type']
        candidates = [m for m in actions if m != n and 
                      actions[m]['category'] == cat and 
                      actions[m]['action_type'] == act_type]
        random.shuffle(candidates)
        for neg in candidates:
            combo = (n, neg)
            if combo not in used_combos:
                used_combos.add(combo)
                queries = [
                    f"What does the {n} action do?",
                    f"How does {n} work in PF2e?",
                    f"Explain the {n} action.",
                    f"Tell me about {n}.",
                    f"What is {n}?"
                ]
                query = random.choice(queries)
                output.append({
                    "query": query,
                    "pos": actions[n]['full'],
                    "neg": actions[neg]['full']
                })
                break

# Trim to exactly 100
output = output[:100]
print(f"Final triplet count: {len(output)}")

# --- Write output ---
out_path = '/mnt/data/projects/pf2e-codex/training_data/raw/actions.jsonl'
with open(out_path, 'w') as f:
    for triplet in output:
        f.write(json.dumps(triplet) + '\n')

print(f"Written to {out_path}")
print(f"Lines in file: {len(output)}")

# Verify
with open(out_path) as f:
    lines = f.readlines()
print(f"Verified: {len(lines)} lines")

# Show sample
for t in output[:3]:
    print(f"\nQuery: {t['query']}")
    print(f"Pos preview: {t['pos'][:120]}...")
    print(f"Neg preview: {t['neg'][:120]}...")
