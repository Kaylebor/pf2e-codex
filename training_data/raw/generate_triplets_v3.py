#!/usr/bin/env python3
"""
Generate 100 training triplets from feat-effects.json.

Each triplet: query (natural question), pos (full effect text), neg (full text of a
different feat effect with similar mechanics).

Hard negatives match on: same level + same type of bonus + same selector.
When exact matches aren't available, same level + any related topic.
"""

import json
import re
import random
from collections import defaultdict, Counter

SEED = 42
random.seed(SEED)

# ── Load data ──
with open('/home/kay/.cache/pf2e-codex/extract-pf2e-8.2.0/feat-effects.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")


def clean_desc(desc):
    """Remove HTML, UUID refs, normalize whitespace."""
    text = desc
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'@UUID\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'@Template\[[^\]]*\]', '', text)
    text = re.sub(r'@Check\[[^\]]*\]', '', text)
    text = re.sub(r'@UUID\[[^\]]*\]', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = text.replace('&#x27;', "'").replace('&#x2019;', "'").replace('&#x2014;', '—')
    text = text.replace('&#x2013;', '–')
    text = re.sub(r'Granted by\s*', '', text)
    text = re.sub(r'via\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s]+', '', text)
    return text


# ── Parse entries ──
entries = []
for item in data:
    name = item['name']
    level = item['system']['level']['value']
    desc = clean_desc(item['system']['description']['value'])
    if not desc or len(desc) < 25:
        continue
    # Skip boilerplate-only descriptions
    if desc.lower() in ('ephemeral effect', "you can't be flanked.", ''):
        continue

    rules = item['system'].get('rules', [])

    modifiers = []
    for rule in rules:
        if rule.get('key') == 'FlatModifier':
            sel = rule.get('selector', '')
            if isinstance(sel, list):
                sel = tuple(sorted(sel))
            else:
                sel = (sel,)
            modifiers.append({
                'type': rule.get('type', 'none'),
                'selector': sel,
                'value': rule.get('value', 0),
            })

    entries.append({
        'name': name,
        'level': level,
        'desc': desc,
        'desc_lower': desc.lower(),
        'modifiers': modifiers,
    })

print(f"Filtered to {len(entries)} entries with descriptions")


# ── Detect description topics ──
def detect_topics(desc_lower):
    topics = set()
    if re.search(r'\bac\b', desc_lower): topics.add('ac')
    if re.search(r'saving throw|save', desc_lower): topics.add('saving-throw')
    if re.search(r'\b(attack roll|attack bonus|to hit|strike|attack)\b', desc_lower): topics.add('attack')
    if re.search(r'\b(damage|d\d+)\b', desc_lower): topics.add('damage')
    if re.search(r'\bspeed\b', desc_lower): topics.add('speed')
    if re.search(r'\bperception\b', desc_lower): topics.add('perception')
    if re.search(r'\b(skill check|skill bonus|acrobatics|athletics|stealth|deception|intimidation|survival|diplomacy|occultism|arcana|religion|nature|society|medicine|crafting|performance|thievery)\b', desc_lower): topics.add('skill')
    if re.search(r'\binitiative\b', desc_lower): topics.add('initiative')
    if re.search(r'\btemporary hit points?\b|temp.*hp|temporary hp', desc_lower): topics.add('temphp')
    if re.search(r'\bresistance\b', desc_lower): topics.add('resistance')
    if re.search(r'\bfortitude\b', desc_lower): topics.add('fortitude')
    if re.search(r'\breflex\b', desc_lower): topics.add('reflex')
    if re.search(r'\bwill\b', desc_lower): topics.add('will')
    if re.search(r'\b(fly speed|flight|flying)\b', desc_lower): topics.add('fly')
    if re.search(r'\bfortune\b|roll twice|better result', desc_lower): topics.add('fortune')
    if re.search(r'\bweakness\b', desc_lower): topics.add('weakness')
    if re.search(r'\bfast healing\b', desc_lower): topics.add('fast-healing')
    return topics

for e in entries:
    e['topics'] = detect_topics(e['desc_lower'])


# ── Generate query ──
def generate_query(entry):
    short = re.sub(r'^(Effect|Stance|Action):\s*', '', entry['name'])
    t = entry['topics']
    
    if 'ac' in t and 'saving-throw' in t: return f"What AC and saving throw bonus does {short} grant?"
    if 'ac' in t: return f"What AC bonus does {short} grant?"
    if 'saving-throw' in t: return f"What saving throw bonus does {short} grant?"
    if 'fortitude' in t and 'reflex' in t and 'will' in t: return f"What saving throw bonus does {short} grant?"
    if 'fortitude' in t: return f"What Fortitude save bonus does {short} grant?"
    if 'reflex' in t: return f"What Reflex save bonus does {short} grant?"
    if 'will' in t: return f"What Will save bonus does {short} grant?"
    if 'attack' in t and 'damage' in t: return f"What attack and damage bonus does {short} grant?"
    if 'attack' in t: return f"What attack bonus does {short} grant?"
    if 'damage' in t: return f"What damage bonus does {short} grant?"
    if 'speed' in t: return f"What speed bonus does {short} grant?"
    if 'temphp' in t: return f"How many temporary HP does {short} grant?"
    if 'resistance' in t: return f"What resistance does {short} grant?"
    if 'perception' in t: return f"What perception bonus does {short} grant?"
    if 'skill' in t: return f"What skill bonus does {short} grant?"
    if 'initiative' in t: return f"What initiative bonus does {short} grant?"
    if 'fly' in t: return f"What fly speed does {short} grant?"
    if 'fortune' in t: return f"What fortune effect does {short} grant?"
    if 'fast-healing' in t: return f"How much fast healing does {short} grant?"
    if 'weakness' in t: return f"What weakness does {short} grant?"
    # Check modifiers as fallback
    for mod in entry['modifiers']:
        sel_str = ','.join(mod['selector'])
        if 'ac' in sel_str: return f"What AC bonus does {short} grant?"
        if 'saving-throw' in sel_str or 'fortitude' in sel_str or 'reflex' in sel_str or 'will' in sel_str:
            return f"What saving throw bonus does {short} grant?"
        if 'attack' in sel_str: return f"What attack bonus does {short} grant?"
        if 'damage' in sel_str: return f"What damage bonus does {short} grant?"
        if 'speed' in sel_str: return f"What speed bonus does {short} grant?"
        if 'perception' in sel_str: return f"What perception bonus does {short} grant?"
    # Generic
    return f"What does {short} do?"


# ── Build indices ──
level_idx = defaultdict(list)
for i, e in enumerate(entries):
    level_idx[e['level']].append(i)

topic_idx = defaultdict(list)
for i, e in enumerate(entries):
    for topic in e['topics']:
        topic_idx[topic].append(i)


def score_negative(pos_entry, neg_entry):
    """Score how good a negative this is (higher = better/harder)."""
    score = 0
    # Same level
    if pos_entry['level'] == neg_entry['level']:
        score += 10
    # Topic overlap
    common = pos_entry['topics'] & neg_entry['topics']
    score += len(common) * 5
    # Modifier overlap
    for m1 in pos_entry['modifiers']:
        for m2 in neg_entry['modifiers']:
            if m1['type'] == m2['type'] and m1['type'] != 'none':
                score += 3
            if set(m1['selector']) & set(m2['selector']):
                score += 3
                if isinstance(m1['value'], (int, float)) and isinstance(m2['value'], (int, float)):
                    if abs(m1['value'] - m2['value']) <= 2:
                        score += 2
    return score


# ── Generate 100 triplets ──
triplets = []
used_pairs = set()  # (pos_idx, neg_idx)

# Iterate through entries in random order
all_indices = list(range(len(entries)))
random.shuffle(all_indices)

pos_count = 0
attempts = 0
max_attempts = 5000

while len(triplets) < 100 and attempts < max_attempts:
    attempts += 1
    idx = all_indices[pos_count % len(all_indices)]
    e = entries[idx]
    
    # Build candidate pool
    candidates = set()
    
    # Same level + same topic (hardest)
    for topic in e['topics']:
        for c in level_idx.get(e['level'], []):
            if c != idx and topic in entries[c]['topics']:
                candidates.add(c)
    
    # Same level only
    if not candidates:
        for c in level_idx.get(e['level'], []):
            if c != idx:
                candidates.add(c)
    
    # Any entry
    if not candidates:
        candidates = set(i for i in range(len(entries)) if i != idx)
    
    # Remove already-used pairs
    candidates = [c for c in candidates 
                  if (idx, c) not in used_pairs and (c, idx) not in used_pairs]
    
    if not candidates:
        pos_count += 1
        continue
    
    # Score and pick best
    candidates.sort(key=lambda c: -score_negative(e, entries[c]))
    
    # Pick from top 3 with random weight
    top_n = min(3, len(candidates))
    weights = [score_negative(e, entries[candidates[j]]) + 1 for j in range(top_n)]
    neg_idx = random.choices(candidates[:top_n], weights=weights, k=1)[0]
    neg = entries[neg_idx]
    
    pos_text = e['desc']
    neg_text = neg['desc']
    
    if pos_text == neg_text:
        pos_count += 1
        continue
    
    query = generate_query(e)
    
    used_pairs.add((idx, neg_idx))
    pos_count += 1
    
    triplets.append({
        'query': query,
        'pos': pos_text,
        'neg': neg_text,
        'pos_name': e['name'],
        'neg_name': neg['name'],
        'pos_level': e['level'],
        'neg_level': neg['level'],
        'similarity': score_negative(e, neg),
    })

print(f"Generated {len(triplets)} triplets from {attempts} attempts")


# ── Write JSONL ──
output_path = '/mnt/data/projects/pf2e-codex/training_data/raw/feat-effects.jsonl'
with open(output_path, 'w') as f:
    for t in triplets:
        out = {'query': t['query'], 'pos': t['pos'], 'neg': t['neg']}
        f.write(json.dumps(out, ensure_ascii=False) + '\n')

print(f"Written to {output_path}")


# ── Quality Report ──
same_level = sum(1 for t in triplets if t['pos_level'] == t['neg_level'])
same_topic = 0
for t in triplets:
    # Recompute topic overlap from cached... we don't have it cached
    pass

print(f"\n{'='*50}")
print(f"QUALITY REPORT")
print(f"{'='*50}")
print(f"Total triplets: {len(triplets)}")
print(f"Same-level pairs: {same_level}/{len(triplets)} ({100*same_level/len(triplets):.0f}%)")

# Check descriptions have no @UUID left
uuid_in_pos = sum(1 for t in triplets if '@UUID' in t['pos'])
uuid_in_neg = sum(1 for t in triplets if '@UUID' in t['neg'])
print(f"Triplets with @UUID in pos: {uuid_in_pos}")
print(f"Triplets with @UUID in neg: {uuid_in_neg}")

q_types = Counter()
for t in triplets:
    q = t['query'].lower()
    if 'ac' in q and 'saving' in q: k = 'AC+save'
    elif 'ac' in q: k = 'AC'
    elif any(x in q for x in ['saving', 'fortitude', 'reflex', 'will']): k = 'save'
    elif 'attack' in q and 'damage' in q: k = 'attack+damage'
    elif 'attack' in q: k = 'attack'
    elif 'damage' in q: k = 'damage'
    elif 'speed' in q: k = 'speed'
    elif 'temp' in q: k = 'temp HP'
    elif 'resistance' in q: k = 'resistance'
    elif 'perception' in q: k = 'perception'
    elif 'skill' in q: k = 'skill'
    elif 'fly' in q: k = 'fly'
    elif 'fortune' in q: k = 'fortune'
    elif 'fast healing' in q: k = 'fast healing'
    elif 'weakness' in q: k = 'weakness'
    else: k = 'other'
    q_types[k] += 1

print(f"\nQuery types:")
for k, v in sorted(q_types.items(), key=lambda x: -x[1]):
    print(f"  {k:20s}: {v:3d}")

# Quality checks
issues = []
for i, t in enumerate(triplets):
    if not t['pos'].strip(): issues.append(f"Line {i+1}: empty pos")
    if not t['neg'].strip(): issues.append(f"Line {i+1}: empty neg")
    if t['pos'] == t['neg']: issues.append(f"Line {i+1}: identical")
    if len(t['pos']) < 20: issues.append(f"Line {i+1}: short pos ({len(t['pos'])}c)")
    if len(t['neg']) < 20: issues.append(f"Line {i+1}: short neg ({len(t['neg'])}c)")

if issues:
    print(f"\n⚠ {len(issues)} issue(s):")
    for iss in issues:
        print(f"  {iss}")
else:
    print(f"\n✓ All quality checks passed")

# Print a sample
print(f"\nSample triplets:")
for t in triplets[:10]:
    print(f"\nQ: {t['query']}")
    print(f"P [{t['pos_name']}]: {t['pos'][:120]}...")
    print(f"N [{t['neg_name']}]: {t['neg'][:120]}...")
    print(f"  lvl={t['pos_level']}vs{t['neg_level']}, sim={t['similarity']}")
