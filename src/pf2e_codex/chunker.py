"""Rules-aware chunk builder from PF2E Foundry pack entries."""

import hashlib
import html
import json
import os
import re
from typing import Any

# Load aliases (pre-remaster to remaster name mapping)
_ALIASES: dict[str, str] | None = None

def _load_aliases() -> dict[str, str]:
    global _ALIASES
    if _ALIASES is not None:
        return _ALIASES
    alias_path = os.path.join(os.path.dirname(__file__), "aliases.json")
    if os.path.exists(alias_path):
        pairs = json.loads(open(alias_path).read())
        _ALIASES = {old.lower().strip(): new for old, new in pairs}
    else:
        _ALIASES = {}
    return _ALIASES


def resolve_alias(name: str) -> str | None:
    """If name has a remaster alias, return the old (pre-remaster) name or None."""
    aliases = _load_aliases()
    for old_lower, new_name in aliases.items():
        if name.lower() == new_name.lower():
            # Found that this is the remaster name — return original old name
            # But aliases are stored old→new, so we need to find new→old
            return next((o for o, n in aliases.items() if n.lower() == name.lower()), None)
    return None


def entry_hash(entry: dict[str, Any]) -> str:
    """Stable hash of entry content only, excluding metadata that changes every release."""
    content = {
        "name": entry.get("name", ""),
        "type": entry.get("type", ""),
        "system": entry.get("system", {}),
    }
    # Journals store content in pages
    if "pages" in entry and "system" not in entry:
        content["pages"] = [
            {"name": p.get("name", ""), "text": p.get("text", {})}
            for p in entry.get("pages", [])
        ]
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


_RE_UUID_LINK = re.compile(r"@UUID\[([^\]]+)\](?:\{([^}]*)\})?")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTISPACE = re.compile(r"\s+")


def strip_html(raw: str) -> str:
    text = _RE_HTML_TAG.sub(" ", raw)
    text = html.unescape(text)
    return _RE_MULTISPACE.sub(" ", text).strip()


class UUIDResolver:
    """Resolves UUIDs to human-readable names across all packs."""

    def __init__(self, entries_map: dict[str, list[dict[str, Any]]]):
        self._name_by_uuid: dict[str, str] = {}
        for _pack, entries in entries_map.items():
            for e in entries:
                uid = e.get("_id")
                if uid:
                    self._name_by_uuid[uid] = e.get("name", "")
                comp_src = e.get("_stats", {}).get("compendiumSource", "")
                if comp_src and ".Item." in comp_src:
                    _, item_id = comp_src.rsplit(".Item.", 1)
                    self._name_by_uuid.setdefault(item_id, e.get("name", ""))

    def resolve(self, uuid_str: str) -> str | None:
        return self._name_by_uuid.get(uuid_str)


def resolve_text(text: str, resolver: UUIDResolver) -> str:
    def repl(m: re.Match[str]) -> str:
        uuid_full = m.group(1)
        explicit = m.group(2)
        if ".Item." in uuid_full:
            _, item_id = uuid_full.rsplit(".Item.", 1)
        else:
            item_id = uuid_full
        resolved = resolver.resolve(item_id)
        return resolved or explicit or item_id
    return _RE_UUID_LINK.sub(repl, text)


# ── Expression simplification ───────────────────────────────────────────

_VAR_SUBS: dict[str, str] = {
    "@item.badge.value": "the condition value",
    "@actor.level": "character level",
    "@actor.system.proficiencies.defenses.unarmored.rank": "unarmored proficiency rank",
    "@actor.system.proficiencies.defenses.light.rank": "light armor proficiency rank",
    "@actor.system.proficiencies.defenses.medium.rank": "medium armor proficiency rank",
    "@actor.system.proficiencies.defenses.heavy.rank": "heavy armor proficiency rank",
    "@actor.system.proficiencies.classDCs.champion.rank": "champion class DC rank",
    "@spell.rank": "spell rank",
    "@spell.level": "spell level",
    "{item|_id}": "this item",
    "{item|flags.system.rulesSelections.cause}": "selected cause",
    "{item|flags.system.rulesSelections.attribute}": "selected attribute",
}

_PATH_MAP: dict[str, str] = {
    "system.proficiencies.defenses.light.rank": "light armor proficiency",
    "system.proficiencies.defenses.medium.rank": "medium armor proficiency",
    "system.proficiencies.defenses.heavy.rank": "heavy armor proficiency",
    "system.proficiencies.defenses.unarmored.rank": "unarmored defense proficiency",
    "system.proficiencies.classDCs.champion.rank": "champion class DC proficiency",
    "system.proficiencies.classDCs.champion.attribute": "champion class DC attribute",
    "system.skills.religion.rank": "Religion skill proficiency",
    "system.attributes.ac.value": "AC",
    "system.attributes.hp.max": "max HP",
    "system.attributes.perception.rank": "Perception proficiency",
    "system.saves.fortitude.rank": "Fortitude save proficiency",
    "system.saves.reflex.rank": "Reflex save proficiency",
    "system.saves.will.rank": "Will save proficiency",
}


def _simplify_path(path: str) -> str:
    path = path.strip()
    if path in _PATH_MAP:
        return _PATH_MAP[path]
    return path.replace("system.", "").replace(".", " ").replace("_", " ")


def _parse_ternary_args(expr: str) -> tuple[str, str, str] | None:
    if not expr.startswith("ternary(") or not expr.endswith(")"):
        return None
    inner = expr[8:-1]
    depth = 0
    splits = []
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            splits.append(i)
    if len(splits) < 2:
        return None
    return (
        inner[: splits[0]].strip(),
        inner[splits[0] + 1 : splits[1]].strip(),
        inner[splits[1] + 1 :].strip(),
    )


def _simplify_expression(expr: str) -> str:
    expr = expr.strip()
    for var, sub in sorted(_VAR_SUBS.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(var, sub)
    expr = expr.replace("-1 * ", "negative ")
    expr = re.sub(r"(?<=[\w\)])\s*\*\s*(?=[\w\-])", " times ", expr)

    prev = None
    while prev != expr and "ternary(" in expr:
        prev = expr
        match = re.search(r"ternary\(([^()]|\([^()]*\))*\)", expr)
        if match:
            full = match.group(0)
            parsed = _parse_ternary_args(full)
            if parsed:
                cond, high, low = parsed
                lm = re.search(r"gte\s*\(\s*character level\s*,\s*(\d+)\s*\)", cond)
                if lm:
                    level = lm.group(1)
                    low_s = _simplify_expression(low)
                    repl = f"+{high} at level {level}+ (else {low_s})"
                else:
                    repl = (
                        f"{_simplify_expression(high)} if "
                        f"{_simplify_expression(cond)} else {_simplify_expression(low)}"
                    )
                expr = expr[: match.start()] + repl + expr[match.end() :]
            break
        match = re.search(r"ternary\(", expr)
        if match:
            start = match.start() + 8
            depth = 1
            end = start
            while end < len(expr) and depth > 0:
                if expr[end] == "(":
                    depth += 1
                elif expr[end] == ")":
                    depth -= 1
                end += 1
            full = expr[match.start() : end]
            parsed = _parse_ternary_args(full)
            if parsed:
                cond, high, low = parsed
                lm = re.search(r"gte\s*\(\s*character level\s*,\s*(\d+)\s*\)", cond)
                if lm:
                    level = lm.group(1)
                    low_s = _simplify_expression(low)
                    repl = f"+{high} at level {level}+ (else {low_s})"
                else:
                    repl = (
                        f"{_simplify_expression(high)} if "
                        f"{_simplify_expression(cond)} else {_simplify_expression(low)}"
                    )
                expr = expr[: match.start()] + repl + expr[end:]
            break

    def repl_max(m: re.Match[str]) -> str:
        args = [_simplify_expression(a.strip()) for a in m.group(1).split(",")]
        return f"maximum of {', '.join(args)}"
    expr = re.sub(r"max\(([^)]+)\)", repl_max, expr)

    def repl_min(m: re.Match[str]) -> str:
        args = [_simplify_expression(a.strip()) for a in m.group(1).split(",")]
        return f"minimum of {', '.join(args)}"
    expr = re.sub(r"min\(([^)]+)\)", repl_min, expr)

    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


# ── Rule flatteners ──────────────────────────────────────────────────────

def _flatmodifier(rule: dict[str, Any]) -> str:
    val = str(rule.get("value", ""))
    selector = rule.get("selector", "")
    typ = rule.get("type", "")
    label = rule.get("label", "")
    if any(tok in val for tok in ["ternary", "max(", "min(", "@", "{item|"]):
        val = _simplify_expression(val)
    try:
        float(val)
        val_display = f"{float(val):+g}"
    except (ValueError, TypeError):
        val_display = val
    parts = [f"{val_display} {typ} to {selector}"]
    if label:
        parts.append(f"({label})")
    return " ".join(parts)


def _immunity(rule: dict[str, Any]) -> str:
    return f"Immunity to {rule.get('type', 'unknown')}"


def _resistance(rule: dict[str, Any]) -> str:
    return f"Resistance {rule.get('value', '')} to {rule.get('type', '')}"


def _weakness(rule: dict[str, Any]) -> str:
    return f"Weakness {rule.get('value', '')} to {rule.get('type', '')}"


def _rollover(rule: dict[str, Any]) -> str:
    parts = [f"Roll option: {rule.get('option', '')}"]
    if rule.get("toggleable"):
        parts.append("[toggleable]")
    sub = rule.get("suboptions", [])
    if sub:
        parts.append(
            f"choices: {', '.join(s.get('value', s.get('label', '')) for s in sub)}"
        )
    return " ".join(parts)


def _granted(rule: dict[str, Any]) -> str:
    uuid = rule.get("uuid", "")
    level = rule.get("level", "")
    parts = ["Grants"]
    if level:
        parts.append(f"(at level {level})")
    parts.append(f"item: {uuid}")
    return " ".join(parts)


def _choiceset(rule: dict[str, Any]) -> str:
    flag = rule.get("flag", "")
    choices = rule.get("choices", [])
    s = f"Choice set: {flag}"
    if isinstance(choices, list) and choices:
        labels = []
        for c in choices:
            if isinstance(c, dict):
                labels.append(c.get("label", c.get("value", str(c))))
            else:
                labels.append(str(c))
        s += f" (options: {', '.join(labels[:8])})"
    return s


def _note(rule: dict[str, Any]) -> str:
    parts = [rule.get("title", ""), rule.get("text", "")]
    if rule.get("selector"):
        parts.append(f"on {rule['selector']}")
    if rule.get("outcome"):
        parts.append(f"when {'/'.join(rule['outcome'])}")
    return " — ".join(p for p in parts if p)


def _damagedice(rule: dict[str, Any]) -> str:
    num = rule.get("diceNumber", "")
    size = rule.get("dieSize", "")
    dtype = rule.get("damageType", "")
    sel = rule.get("selector", "")
    parts = []
    if num and size:
        parts.append(f"+{num}{size} {dtype} damage")
    if sel:
        parts.append(f"to {sel}")
    return " ".join(parts)


def _strike(rule: dict[str, Any]) -> str:
    label = rule.get("label", "")
    cat = rule.get("category", "")
    group = rule.get("group", "")
    traits = rule.get("traits", [])
    dmg = rule.get("damage", {}).get("base", {})
    parts = [f"Strike: {label or 'unlabeled'}"]
    if cat:
        parts.append(f"({cat})")
    if group:
        parts.append(f"[{group}]")
    if traits:
        parts.append(f"traits: {', '.join(traits)}")
    if dmg:
        parts.append(
            f"damage: {dmg.get('dice', '')}{dmg.get('die', '')} {dmg.get('damageType', '')}"
        )
    return " ".join(parts)


def _base_speed(rule: dict[str, Any]) -> str:
    return f"Base speed: {rule.get('value', '')} ft ({rule.get('selector', '')})"


def _sense(rule: dict[str, Any]) -> str:
    parts = [f"Sense: {rule.get('selector', '')}"]
    if rule.get("acuity"):
        parts.append(f"({rule['acuity']})")
    if rule.get("range"):
        parts.append(f"range {rule['range']} ft")
    return " ".join(parts)


def _activeeffectlike(rule: dict[str, Any]) -> str:
    mode = rule.get("mode", "")
    path = _simplify_path(rule.get("path", ""))
    value = _simplify_expression(str(rule.get("value", "")))
    return f"Effect: {mode} {path} to {value}"


def _itemalteration(rule: dict[str, Any]) -> str:
    return (
        f"Alter {rule.get('itemType', '')}: {rule.get('mode', '')} "
        f"{rule.get('property', '')} = {rule.get('value', '')}"
    )


def _adjust_degree(rule: dict[str, Any]) -> str:
    adj = rule.get("adjustment", {})
    changes = [f"{k} -> {v}" for k, v in adj.items()]
    return f"Adjust degree of success on {rule.get('selector', '')}: {', '.join(changes)}"


def _adjust_modifier(rule: dict[str, Any]) -> str:
    return (
        f"Adjust modifier '{rule.get('slug', '')}' on {rule.get('selector', '')}: "
        f"{rule.get('mode', '')} by {rule.get('value', '')}"
    )


def _martial_proficiency(rule: dict[str, Any]) -> str:
    parts = ["Martial proficiency"]
    if rule.get("sameAs"):
        parts.append(f"same as {rule['sameAs']}")
    if rule.get("rank"):
        parts.append(f"rank {rule['rank']}")
    if rule.get("groups"):
        parts.append(f"groups: {', '.join(rule['groups'])}")
    return " ".join(parts)


def _critical_specialization(rule: dict[str, Any]) -> str:
    groups = rule.get("groups", [])
    return f"Critical specialization: {', '.join(groups) if groups else 'all weapons'}"


def _aura(rule: dict[str, Any]) -> str:
    parts = [f"Aura radius {rule.get('radius', '?')} ft"]
    for eff in rule.get("effects", []):
        parts.append(f"effect: {eff.get('uuid', '')}")
    return " ".join(parts)


def _ephemeral_effect(rule: dict[str, Any]) -> str:
    return f"Ephemeral effect: {rule.get('uuid', '')} on {', '.join(rule.get('selectors', []))}"


def _crafting_ability(rule: dict[str, Any]) -> str:
    parts = ["Crafting ability"]
    if rule.get("isDailyPrep"):
        parts.append("[daily prep]")
    if rule.get("craftableItems"):
        parts.append(f"items: {len(rule['craftableItems'])} entries")
    return " ".join(parts)


def _creature_size(rule: dict[str, Any]) -> str:
    return f"Size: {rule.get('value', '')}"


def _dex_cap(rule: dict[str, Any]) -> str:
    return f"Dexterity modifier cap: {rule.get('value', '')}"


def _token_light(rule: dict[str, Any]) -> str:
    return f"Token light: {rule.get('value', '')}"


def _special_resource(rule: dict[str, Any]) -> str:
    return f"Special resource: {rule.get('slug', '')} max {rule.get('max', '')}"


def _fast_healing(rule: dict[str, Any]) -> str:
    return f"Fast healing {rule.get('value', '')} ({rule.get('details', '')})"


def _multiple_attack_penalty(rule: dict[str, Any]) -> str:
    return f"Multiple attack penalty: {rule.get('value', '')}"


def _lose_hp(rule: dict[str, Any]) -> str:
    return f"Lose {_simplify_expression(str(rule.get('value', '')))} HP"


def _temp_hp(rule: dict[str, Any]) -> str:
    return f"Temporary HP: {_simplify_expression(str(rule.get('value', '')))}"


def _substitute_roll(rule: dict[str, Any]) -> str:
    return f"Substitute roll on {rule.get('selector', '')}"


def _roll_twice(rule: dict[str, Any]) -> str:
    return f"Roll twice on {rule.get('selector', '')} ({rule.get('keep', '')})"


def _actor_traits(rule: dict[str, Any]) -> str:
    parts = []
    if rule.get("add"):
        parts.append(f"Add traits: {', '.join(rule['add'])}")
    if rule.get("remove"):
        parts.append(f"Remove traits: {', '.join(rule['remove'])}")
    return "; ".join(parts)


_RULE_FLATTENERS: dict[str, Any] = {
    "FlatModifier": _flatmodifier,
    "Immunity": _immunity,
    "Resistance": _resistance,
    "Weakness": _weakness,
    "RollOption": _rollover,
    "GrantItem": _granted,
    "ChoiceSet": _choiceset,
    "Note": _note,
    "DamageDice": _damagedice,
    "Strike": _strike,
    "BaseSpeed": _base_speed,
    "Sense": _sense,
    "ActiveEffectLike": _activeeffectlike,
    "ItemAlteration": _itemalteration,
    "AdjustDegreeOfSuccess": _adjust_degree,
    "AdjustModifier": _adjust_modifier,
    "MartialProficiency": _martial_proficiency,
    "CriticalSpecialization": _critical_specialization,
    "Aura": _aura,
    "EphemeralEffect": _ephemeral_effect,
    "CraftingAbility": _crafting_ability,
    "CreatureSize": _creature_size,
    "DexterityModifierCap": _dex_cap,
    "TokenLight": _token_light,
    "SpecialResource": _special_resource,
    "FastHealing": _fast_healing,
    "MultipleAttackPenalty": _multiple_attack_penalty,
    "LoseHitPoints": _lose_hp,
    "TempHP": _temp_hp,
    "SubstituteRoll": _substitute_roll,
    "RollTwice": _roll_twice,
    "ActorTraits": _actor_traits,
}


def flatten_rule(rule: dict[str, Any]) -> str | None:
    key = rule.get("key", "")
    flattener = _RULE_FLATTENERS.get(key)
    if flattener:
        try:
            return flattener(rule)
        except Exception:  # noqa: S110
            return f"[{key}: error]"
    return f"[{key}: {json.dumps(rule, ensure_ascii=False)[:120]}]"


# ── Chunk Builder ──────────────────────────────────────────────────────

class ChunkBuilder:
    def __init__(self, resolver: UUIDResolver):
        self.resolver = resolver

    def build_all(self, entry: dict[str, Any], pack_name: str) -> list[dict[str, Any]]:
        """Build one or more chunks from an entry. Journals produce page chunks."""
        etype = entry.get("type", "")
        h = entry_hash(entry)
        if not etype and "pages" in entry and "system" not in entry:
            chunks = self._build_journal_chunks(entry, pack_name)
            for c in chunks:
                c["source_hash"] = h
            return chunks
        chunk = self._build_single(entry, pack_name)
        if chunk:
            chunk["id"] = f"{pack_name}:{chunk['id']}"
            chunk["source_hash"] = h
            return [chunk]
        return []

    def _build_journal_chunks(self, entry: dict[str, Any], pack_name: str) -> list[dict[str, Any]]:
        journal_name = entry.get("name", "")
        pages = entry.get("pages", [])
        chunks = []
        for i, page in enumerate(pages):
            title = page.get("name", f"Page {i + 1}")
            text_content = ""
            if isinstance(page.get("text"), dict):
                text_content = page["text"].get("content", "")
            elif isinstance(page.get("text"), str):
                text_content = page["text"]
            plain = strip_html(text_content)
            plain = resolve_text(plain, self.resolver)
            if not plain.strip():
                continue
            refs = self._extract_refs(text_content)
            lines = [f"Journal: {journal_name}", f"Page: {title}", "", plain]
            if refs:
                ref_names = sorted({r["name"] for r in refs})
                lines += ["", f"Related: {', '.join(ref_names)}"]
            chunks.append({
                "id": f"{pack_name}:{entry['_id']}_page_{i}",
                "name": f"{journal_name} — {title}",
                "type": "journal_page",
                "pack": pack_name,
                "slug": "",
                "level": None,
                "traits": [],
                "text": "\n".join(lines),
                "raw_rules_count": 0,
                "has_description": bool(plain),
                "refs": refs,
                "source_hash": None,
                "license": entry.get("system", {}).get("publication", {}).get("license", "NONE"),
            })
        return chunks

    def _build_single(self, entry: dict[str, Any], pack_name: str) -> dict[str, Any] | None:
        system = entry.get("system", {})
        name = entry.get("name", "")
        etype = entry.get("type", "")
        slug = system.get("slug", "")
        traits = system.get("traits", {}).get("value", [])

        desc_html = system.get("description", {}).get("value", "")
        desc_text = strip_html(desc_html)
        desc_text = resolve_text(desc_text, self.resolver)

        prereqs = system.get("prerequisites", {}).get("value", [])
        prereq_texts = [
            p.get("value", str(p)) if isinstance(p, dict) else str(p) for p in prereqs
        ]

        level: int | None = None
        if "level" in system:
            lv = system["level"]
            if isinstance(lv, dict):
                level = lv.get("value")
            else:
                level = lv

        action_type = system.get("actionType", {}).get("value", "")
        actions = system.get("actions", {}).get("value", "")
        if action_type == "action" and actions:
            action_cost = f"{actions} action{'s' if actions != 1 else ''}"
        elif action_type:
            action_cost = action_type
        else:
            action_cost = None

        rules = system.get("rules", [])
        rule_texts = []
        for r in rules:
            flat = flatten_rule(r)
            if flat:
                rule_texts.append(flat)
        rule_texts = self._resolve_rule_texts(rule_texts)

        # Include old name as alias if this was renamed in remaster
        old_name = resolve_alias(name)
        display_name = f"{name} (formerly {old_name})" if old_name else name

        lines = [f"{etype}: {display_name}" + (f" ({slug})" if slug else "")]
        if level is not None:
            lines.append(f"Level: {level}")
        if action_cost:
            lines.append(f"Action: {action_cost}")
        if traits:
            lines.append(f"Traits: {', '.join(traits)}")
        if prereq_texts:
            lines.append(f"Prerequisites: {'; '.join(prereq_texts)}")

        self._add_type_specific_fields(lines, etype, system)

        if desc_text:
            lines += ["", "Description:", desc_text]
        if rule_texts:
            lines += ["", "Mechanical Effects:"]
            for rt in rule_texts:
                lines.append(f"- {rt}")

        desc_refs = self._extract_refs(desc_html)
        rule_refs = self._extract_rule_refs(rules)
        all_refs = desc_refs + rule_refs
        if all_refs:
            ref_names = sorted({r["name"] for r in all_refs})
            lines += ["", f"Related: {', '.join(ref_names)}"]

        pub = system.get("publication", {})
        if pub:
            parts = []
            if pub.get("title"):
                parts.append(pub["title"])
            if pub.get("remaster"):
                parts.append("[remaster]")
            if pub.get("license"):
                parts.append(f"({pub['license']})")
            if parts:
                lines += ["", f"Source: {' '.join(parts)}"]

        license_val = system.get("publication", {}).get("license", "NONE")

        return {
            "id": entry.get("_id", ""),
            "name": display_name,
            "type": etype,
            "pack": pack_name,
            "slug": slug,
            "level": level,
            "traits": traits,
            "text": "\n".join(lines),
            "raw_rules_count": len(rules),
            "has_description": bool(desc_text),
            "refs": all_refs,
            "license": license_val,
        }

    def _add_type_specific_fields(self, lines: list[str], etype: str, system: dict[str, Any]) -> None:
        if etype == "condition":
            group = system.get("group", "")
            overrides = system.get("overrides", [])
            if group:
                lines.append(f"Condition Group: {group}")
            if overrides:
                lines.append(f"Overrides: {', '.join(overrides)}")
            duration = system.get("duration", {})
            if duration and duration.get("unit"):
                lines.append(f"Duration: {duration.get('value', '')} {duration['unit']}")
            valued = system.get("value", {})
            if valued.get("isValued"):
                lines.append("Valued: yes (stacks)")

        elif etype == "spell":
            traditions = system.get("traditions", {}).get("value", [])
            if traditions:
                lines.append(f"Traditions: {', '.join(traditions)}")
            time = system.get("time", {})
            if time:
                lines.append(f"Cast: {time.get('value', '')} {time.get('unit', '')}")
            defense = system.get("defense", {})
            if defense:
                save = defense.get("save", {})
                if save:
                    stat = save.get("statistic", "")
                    basic = " (basic)" if save.get("basic") else ""
                    lines.append(f"Save: {stat}{basic}")
            damage = system.get("damage", {})
            if damage:
                dmg_parts = []
                for _k, v in damage.items():
                    if isinstance(v, dict) and v.get("formula"):
                        dmg_parts.append(f"{v['formula']} {v.get('type', '')}")
                if dmg_parts:
                    lines.append(f"Damage: {', '.join(dmg_parts)}")
            heightening = system.get("heightening", {})
            if heightening:
                parts = []
                if heightening.get("type"):
                    parts.append(heightening["type"])
                if heightening.get("interval"):
                    parts.append(f"every {heightening['interval']} rank(s)")
                hdmg = heightening.get("damage", {})
                for slot, val in hdmg.items():
                    parts.append(f"+{val} damage at rank {slot}")
                if heightening.get("area"):
                    parts.append(f"+{heightening['area']} ft area")
                if parts:
                    lines.append(f"Heightening: {'; '.join(parts)}")
            for field, label in [
                ("range", "Range"),
                ("target", "Target"),
                ("area", "Area"),
                ("duration", "Duration"),
            ]:
                val = system.get(field, {})
                if val and val.get("value"):
                    unit = f" {val.get('type', '')}" if field == "area" else ""
                    lines.append(f"{label}: {val['value']}{unit}")

        elif etype == "feat":
            category = system.get("category", "")
            if category:
                lines.append(f"Category: {category}")
            subfeatures = system.get("subfeatures", {})
            if subfeatures:
                for sf_key, sf_val in subfeatures.items():
                    if sf_val:
                        lines.append(
                            f"Subfeature ({sf_key}): "
                            f"{json.dumps(sf_val, ensure_ascii=False)[:200]}"
                        )

        elif etype == "class":
            key_ability = system.get("keyAbility", {}).get("value", [])
            if key_ability:
                lines.append(f"Key Ability: {', '.join(key_ability)}")
            hp = system.get("hp", 0)
            if hp:
                lines.append(f"HP: {hp}")
            perception = system.get("perception", {})
            if isinstance(perception, dict) and perception.get("value"):
                lines.append(f"Perception: {perception['value']}")
            saves = system.get("savingThrows", {})
            if isinstance(saves, dict):
                save_parts = [
                    f"{sk} {sv.get('value', '')}"
                    for sk, sv in saves.items()
                    if isinstance(sv, dict) and sv.get("value")
                ]
                if save_parts:
                    lines.append(f"Saves: {', '.join(save_parts)}")
            trained = system.get("trainedSkills", {})
            if isinstance(trained, dict):
                skills = []
                for sk, sv in trained.items():
                    if sk == "value" and isinstance(sv, list):
                        skills.extend(sv)
                    elif isinstance(sv, dict) and sv.get("value"):
                        skills.append(f"{sk} {sv['value']}")
                if skills:
                    lines.append(f"Trained Skills: {', '.join(str(s) for s in skills)}")

        elif etype == "ancestry":
            hp = system.get("hp", 0)
            if hp:
                lines.append(f"HP: {hp}")
            size = system.get("size", "")
            if size:
                lines.append(f"Size: {size}")
            speed = system.get("speed", 0)
            if speed:
                lines.append(f"Speed: {speed}")
            boosts = system.get("boosts", {})
            if isinstance(boosts, dict) and boosts.get("default"):
                lines.append(f"Ability Boosts: {', '.join(boosts['default'])}")
            flaws = system.get("flaws", {})
            if isinstance(flaws, dict) and flaws.get("default"):
                lines.append(f"Ability Flaws: {', '.join(flaws['default'])}")

        elif etype == "heritage":
            ancestry = system.get("ancestry", {})
            if isinstance(ancestry, dict) and ancestry.get("name"):
                lines.append(f"Ancestry: {ancestry['name']}")

        elif etype == "background":
            boosts = system.get("boosts", {})
            if isinstance(boosts, dict):
                default = boosts.get("default", [])
                if default:
                    lines.append(f"Ability Boosts: {', '.join(default)}")
            skills = system.get("trainedSkills", {})
            if isinstance(skills, dict):
                vals = skills.get("value", [])
                if vals:
                    lines.append(f"Trained Skills: {', '.join(vals)}")
            feat = system.get("items", {})
            if isinstance(feat, dict):
                for _k, v in feat.items():
                    if isinstance(v, dict) and v.get("uuid"):
                        lines.append(f"Granted: {v['uuid']}")

    def _resolve_rule_texts(self, texts: list[str]) -> list[str]:
        resolved = []
        for t in texts:
            t = self._resolve_uuids_in_text(t)
            t = re.sub(r"Grants \(at level \)\s*", "Grants ", t)
            t = t.strip()
            if t:
                resolved.append(t)
        return resolved

    def _resolve_uuids_in_text(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            uuid_full = m.group(1)
            explicit = m.group(2)
            if ".Item." in uuid_full:
                _, item_id = uuid_full.rsplit(".Item.", 1)
            else:
                item_id = uuid_full
            resolved = self.resolver.resolve(item_id)
            return resolved or explicit or item_id
        text = _RE_UUID_LINK.sub(repl, text)

        def repl_bare(m: re.Match[str]) -> str:
            resolved = self.resolver.resolve(m.group(1))
            return resolved or m.group(0)
        text = re.sub(r"Compendium\.[^\s]+\.Item\.([A-Za-z0-9]+)", repl_bare, text)
        return text

    def _extract_refs(self, desc_html: str) -> list[dict[str, str]]:
        """Extract UUID references from HTML description. Returns [{name, uuid, context}]."""
        refs: list[dict[str, str]] = []
        for uuid_full, explicit in _RE_UUID_LINK.findall(desc_html):
            if ".Item." in uuid_full:
                _, item_id = uuid_full.rsplit(".Item.", 1)
            else:
                item_id = uuid_full
            resolved = self.resolver.resolve(item_id)
            name = resolved or explicit or item_id
            # Extract surrounding text for context (first 200 chars of plain desc)
            plain = strip_html(desc_html)[:200]
            refs.append({"name": name, "uuid": item_id, "context": plain})
        return refs

    def _extract_rule_refs(self, rules: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Extract UUID references from rule elements (GrantItem, EphemeralEffect, etc.)."""
        refs: list[dict[str, str]] = []
        for rule in rules:
            # Direct uuid fields
            for key in ("uuid", "compendiumSource"):
                val = rule.get(key)
                if val and isinstance(val, str):
                    item_id = val
                    if ".Item." in item_id:
                        _, item_id = item_id.rsplit(".Item.", 1)
                    resolved = self.resolver.resolve(item_id)
                    name = resolved or item_id
                    refs.append({"name": name, "uuid": item_id, "context": f"rule: {rule.get('key', '')}"})
            # Nested effect UUIDs (Aura, etc.)
            for effect in rule.get("effects", []):
                eff_uuid = effect.get("uuid", "")
                if eff_uuid:
                    item_id = eff_uuid
                    if ".Item." in item_id:
                        _, item_id = item_id.rsplit(".Item.", 1)
                    resolved = self.resolver.resolve(item_id)
                    name = resolved or item_id
                    refs.append({"name": name, "uuid": item_id, "context": f"aura effect: {rule.get('key', '')}"})
        return refs
