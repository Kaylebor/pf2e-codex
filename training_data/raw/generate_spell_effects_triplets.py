#!/usr/bin/env python3
"""
Generate 100 training triplets from spell-effects.json.

Each triplet: query (natural question), pos (full effect text), neg (full text of a
different spell effect with similar mechanics but different effect).

Hard negatives match on: level, bonus type, selector, and topic overlap.
The pos/neg fields use raw HTML descriptions (with @UUID references intact),
matching the format of feat-effects.jsonl.
"""

import json
import re
import random
from collections import defaultdict, Counter

SEED = 42
random.seed(SEED)

# ── Load data ──
with open('/home/kay/.cache/pf2e-codex/extract-pf2e-8.2.0/spell-effects.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")


def clean_desc(desc):
    """Remove HTML, UUID refs, normalize whitespace. Used only for topic detection."""
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
    raw_desc = item['system']['description']['value']  # Keep raw HTML for pos/neg
    cleaned = clean_desc(raw_desc)  # Clean version for topic detection
    if not cleaned or len(cleaned) < 25:
        continue
    # Skip boilerplate-only descriptions
    if cleaned.lower() in ('ephemeral effect', "you can't be flanked.", ''):
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
        'raw_desc': raw_desc,      # Raw HTML for pos/neg output
        'desc': cleaned,            # Cleaned for topic detection
        'desc_lower': cleaned.lower(),
        'modifiers': modifiers,
        'has_damage_dice': any(r.get('key') == 'DamageDice' for r in rules),
        'has_speed': any(r.get('key') == 'BaseSpeed' for r in rules),
        'has_temphp': any(r.get('key') == 'TempHP' for r in rules),
        'has_resistance': any(r.get('key') == 'Resistance' for r in rules),
        'has_weakness': any(r.get('key') == 'Weakness' for r in rules),
    })

print(f"Filtered to {len(entries)} entries with descriptions")


# ── Detect description topics ──
def detect_topics(entry):
    desc = entry['desc_lower']
    topics = set()
    if re.search(r'\bac\b', desc): topics.add('ac')
    if re.search(r'saving throw|save\b', desc): topics.add('saving-throw')
    if re.search(r'\b(attack roll|attack bonus|to hit|strike|attack)\b', desc): topics.add('attack')
    if re.search(r'\b(damage|d\d+)\b', desc): topics.add('damage')
    if re.search(r'\bspeed\b', desc): topics.add('speed')
    if re.search(r'\bperception\b', desc): topics.add('perception')
    if re.search(r'\b(skill check|skill bonus|acrobatics|athletics|stealth|deception|intimidation|survival|diplomacy|occultism|arcana|religion|nature|society|medicine|crafting|performance|thievery)\b', desc): topics.add('skill')
    if re.search(r'\binitiative\b', desc): topics.add('initiative')
    if re.search(r'\btemporary hit points?\b|temp.*hp|temporary hp', desc): topics.add('temphp')
    if re.search(r'\bresistance\b', desc): topics.add('resistance')
    if re.search(r'\bfortitude\b', desc): topics.add('fortitude')
    if re.search(r'\breflex\b', desc): topics.add('reflex')
    if re.search(r'\bwill\b', desc): topics.add('will')
    if re.search(r'\b(fly speed|flight|flying)\b', desc): topics.add('fly')
    if re.search(r'\bfortune\b|roll twice|better result', desc): topics.add('fortune')
    if re.search(r'\bweakness\b', desc): topics.add('weakness')
    if re.search(r'\bfast healing\b', desc): topics.add('fast-healing')
    if re.search(r'\bimmun', desc): topics.add('immunity')
    if re.search(r'\bheal\b|hit point', desc): topics.add('healing')
    if re.search(r'\bspell\b', desc): topics.add('spell')
    if re.search(r'\bcondition\b|blinded|frightened|stunned|paralyzed|slowed|stupified|confused|fascinated|grabbed|prone|invisible', desc): topics.add('condition')
    return topics

for e in entries:
    e['topics'] = detect_topics(e)


# ── Generate query ──
def generate_query(entry):
    short = entry['name']
    # Remove "Spell Effect: " prefix for the query
    short = re.sub(r'^Spell Effect:\s*', '', short)
    t = entry['topics']

    if 'ac' in t and 'saving-throw' in t: return f"What AC and saving throw bonus does {short} grant?"
    if 'ac' in t: return f"What AC bonus does {short} grant?"
    if 'saving-throw' in t: return f"What saving throw bonus does {short} grant?"
    if 'fortitude' in t and 'reflex' in t and 'will' in t: return f"What saving throw bonus does {short} grant?"
    if 'fortitude' in t and 'reflex' in t: return f"What Fortitude and Reflex save bonus does {short} grant?"
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
    if 'immunity' in t: return f"What immunity does {short} grant?"
    if 'healing' in t: return f"What healing does {short} provide?"
    if 'condition' in t: return f"What condition does {short} inflict?"
    if 'spell' in t: return f"What spell effect does {short} grant?"
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
level_topic_idx = defaultdict(lambda: defaultdict(list))
for i, e in enumerate(entries):
    for topic in e['topics']:
        topic_idx[topic].append(i)
        level_topic_idx[e['level']][topic].append(i)


def bonus_similarity(e1, e2):
    """Score how good a negative this is (higher = better/harder)."""
    score = 0
    common = e1['topics'] & e2['topics']
    score += len(common) * 5
    if e1['level'] == e2['level']:
        score += 10
    for m1 in e1['modifiers']:
        for m2 in e2['modifiers']:
            if m1['type'] == m2['type'] and m1['type'] != 'none':
                score += 3
            if set(m1['selector']) & set(m2['selector']):
                score += 3
                if isinstance(m1['value'], (int, float)) and isinstance(m2['value'], (int, float)):
                    if abs(m1['value'] - m2['value']) <= 2:
                        score += 2
    if e1['has_damage_dice'] and e2['has_damage_dice']: score += 2
    if e1['has_speed'] and e2['has_speed']: score += 2
    if e1['has_temphp'] and e2['has_temphp']: score += 2
    if e1['has_resistance'] and e2['has_resistance']: score += 2
    if e1['has_weakness'] and e2['has_weakness']: score += 2
    return score


def find_negatives(entry_idx, count=5):
    """Return top `count` hard-negative candidates (indices) for the entry."""
    e = entries[entry_idx]
    candidates = set()

    # Level + same topic (hardest negatives)
    for topic in e['topics']:
        for c in level_topic_idx.get(e['level'], {}).get(topic, []):
            if c != entry_idx:
                candidates.add(c)

    # Same topic
    for topic in e['topics']:
        for c in topic_idx.get(topic, []):
            if c != entry_idx:
                candidates.add(c)

    # Same level
    for c in level_idx.get(e['level'], []):
        if c != entry_idx:
            candidates.add(c)

    # Any other
    if not candidates:
        candidates = set(i for i in range(len(entries)) if i != entry_idx)

    # Score and pick top
    scored = [(bonus_similarity(e, entries[c]), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])

    return [c for _, c in scored[:count]]


# ── Generate 100 triplets ──
triplets = []
used_pairs = {}
pos_used_counter = Counter()

all_indices = list(range(len(entries)))
random.shuffle(all_indices)

# For each entry, find its top negatives
entry_negatives = {}
for i in all_indices:
    entry_negatives[i] = find_negatives(i, count=10)

attempts = 0
while len(triplets) < 100 and attempts < 2000:
    attempts += 1

    idx = all_indices[len(triplets) % len(all_indices)]
    e = entries[idx]

    candidates = entry_negatives[idx]

    fresh = [c for c in candidates if (idx, c) not in used_pairs]
    if not fresh:
        fresh = [c for c in range(len(entries))
                 if c != idx and (idx, c) not in used_pairs]
        fresh = sorted(fresh, key=lambda c: -bonus_similarity(e, entries[c]))

    if not fresh:
        continue

    neg_idx = fresh[0]
    neg = entries[neg_idx]

    query = generate_query(e)
    # Use RAW HTML descriptions for pos/neg (matching feat-effects.jsonl format)
    pos_text = e['raw_desc']
    neg_text = neg['raw_desc']

    if pos_text == neg_text:
        continue

    used_pairs[(idx, neg_idx)] = bonus_similarity(e, neg)
    pos_used_counter[idx] += 1

    triplets.append({
        'query': query,
        'pos': pos_text,
        'neg': neg_text,
        'pos_name': e['name'],
        'neg_name': neg['name'],
        'pos_level': e['level'],
        'neg_level': neg['level'],
        'pos_topics': list(e['topics']),
        'neg_topics': list(neg['topics']),
        'similarity': bonus_similarity(e, neg),
    })

print(f"Generated {len(triplets)} triplets from {attempts} attempts")

# ── Write JSONL ──
output_path = '/mnt/data/projects/pf2e-codex/training_data/raw/spell-effects.jsonl'
with open(output_path, 'w') as f:
    for t in triplets:
        out = {
            'query': t['query'],
            'pos': t['pos'],
            'neg': t['neg'],
        }
        f.write(json.dumps(out, ensure_ascii=False) + '\n')

print(f"Written to {output_path}")

# ── Write output report ──
report_lines = []
report_lines.append(f"{'='*60}")
report_lines.append(f"SPELL EFFECTS TRAINING TRIPLET GENERATION REPORT")
report_lines.append(f"{'='*60}")
report_lines.append(f"Source: /home/kay/.cache/pf2e-codex/extract-pf2e-8.2.0/spell-effects.json")
report_lines.append(f"Output: {output_path}")
report_lines.append(f"")

same_level = sum(1 for t in triplets if t['pos_level'] == t['neg_level'])
same_topic = sum(1 for t in triplets
                 if t['pos_topics'] and t['neg_topics']
                 and set(t['pos_topics']) & set(t['neg_topics']))

report_lines.append(f"Total entries loaded: {len(data)}")
report_lines.append(f"Entries with valid descriptions: {len(entries)}")
report_lines.append(f"Triplets generated: {len(triplets)}")
report_lines.append(f"Same-level pairs: {same_level}/{len(triplets)} ({100*same_level/max(len(triplets),1):.0f}%)")
report_lines.append(f"Overlapping-topic pairs: {same_topic}/{len(triplets)} ({100*same_topic/max(len(triplets),1):.0f}%)")
report_lines.append(f"Unique positives: {len(pos_used_counter)}")
report_lines.append(f"Avg similarity score: {sum(t['similarity'] for t in triplets)/max(len(triplets),1):.1f}")
report_lines.append(f"Avg pos length: {sum(len(t['pos']) for t in triplets)/max(len(triplets),1):.0f} chars")
report_lines.append(f"Avg neg length: {sum(len(t['neg']) for t in triplets)/max(len(triplets),1):.0f} chars")

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
    elif 'immunity' in q: k = 'immunity'
    elif 'healing' in q: k = 'healing'
    elif 'condition' in q: k = 'condition'
    else: k = 'other'
    q_types[k] += 1

report_lines.append(f"")
report_lines.append(f"Query types:")
for k, v in sorted(q_types.items(), key=lambda x: -x[1]):
    report_lines.append(f"  {k:20s}: {v:3d}")

# Quality checks
issues = []
for i, t in enumerate(triplets):
    if not t['pos'].strip(): issues.append(f"Line {i+1}: empty pos")
    if not t['neg'].strip(): issues.append(f"Line {i+1}: empty neg")
    if t['pos'] == t['neg']: issues.append(f"Line {i+1}: identical")
    if len(t['pos']) < 20: issues.append(f"Line {i+1}: short pos ({len(t['pos'])}c)")
    if len(t['neg']) < 20: issues.append(f"Line {i+1}: short neg ({len(t['neg'])}c)")

if issues:
    report_lines.append(f"")
    report_lines.append(f"⚠ {len(issues)} issue(s):")
    for iss in issues:
        report_lines.append(f"  {iss}")
else:
    report_lines.append(f"")
    report_lines.append(f"✓ All quality checks passed")

report_lines.append(f"")
report_lines.append(f"Sample triplets:")
for t in triplets[:10]:
    report_lines.append(f"")
    report_lines.append(f"Q: {t['query']}")
    report_lines.append(f"P [{t['pos_name']}]: {t['pos'][:130]}...")
    report_lines.append(f"N [{t['neg_name']}]: {t['neg'][:130]}...")
    report_lines.append(f"  sim={t['similarity']}, lvl={t['pos_level']}vs{t['neg_level']}")

report = '\n'.join(report_lines)
print(report)

# Write report to both output locations
with open('/mnt/data/projects/training_data/raw/spell-effects_out.txt', 'w') as f:
    f.write(report)

with open('/mnt/data/projects/pf2e-codex/training_data/raw/spell-effects_out.txt', 'w') as f:
    f.write(report)

print(f"\nReport written to spell-effects_out.txt")
