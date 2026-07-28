"""Sanitize existing auto_glossary.json — detect and quarantine garbage entries.

Run via:
    python subtranslate.py --sanitize-learning
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
AUTO_GLOSSARY_PATH = CONFIG_DIR / "auto_glossary.json"
BACKUP_DIR = CONFIG_DIR / "learning_backup"

# Common German words that might appear as "source" (should not be there — source must be English)
_GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "eines", "einer", "und", "oder", "aber", "doch", "nicht", "kein", "keine",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten", "sein", "seine", "seiner", "seines",
    "ihr", "ihre", "ihrer", "ihres", "ihnen", "sich", "auch", "nur",
    "noch", "schon", "erst", "sehr", "viel", "wenig", "ganz", "etwas",
    "dann", "dort", "hier", "da", "wo", "wie", "was", "wer", "wem", "wen",
    "zum", "zur", "vom", "beim", "ins", "übers", "unters", "durchs",
    "fürs", "aufs", "hinter", "neben", "zwischen", "über", "unter",
    "vor", "nach", "bei", "mit", "aus", "von", "zu", "an", "in", "auf",
    "um", "durch", "gegen", "ohne", "bis", "entlang", "gemäß", "laut",
    "dank", "trotz", "wegen", "statt", "während", "innerhalb", "außerhalb",
    "oberhalb", "unterhalb", "dieser", "diese", "dieses", "diesem", "diesen",
    "jener", "jene", "jenes", "jenem", "jenen", "solcher", "solche", "solches",
    "mich", "mir", "dich", "dir", "euch", "uns", "man", "jemand", "niemand",
    "kann", "kannst", "können", "konnte", "musste", "muss", "müssen",
    "soll", "sollen", "sollte", "will", "willst", "wollen", "wollte",
    "darf", "darfst", "dürfen", "durfte", "mag", "mögen", "mochte",
    "möchte", "möchtest", "möchten", "bitte", "danke", "hallo", "tschüss",
    "hübsch", "ja", "nein", "doch", "vielleicht", "natürlich",
    "wirklich", "eigentlich", "einfach", "genau", "gerade", "schon",
    "immer", "niemals", "oft", "manchmal", "selten", "täglich",
    "heute", "gestern", "morgen", "jetzt", "sofort", "später", "früher",
    "oben", "unten", "links", "rechts", "drinnen", "draußen",
    "deshalb", "deswegen", "trotzdem", "allerdings", "übrigens",
    "nämlich", "also", "demnach", "folglich", "inzwischen", "mittlerweile",
    "obwohl", "weil", "denn", "dass", "damit", "indem", "während",
    "nachdem", "bevor", "seitdem", "sobald", "solange", "sooft",
    "falls", "wenn", "wann", "ob", "als", "wie", "je", "desto",
    "melden", "meldet", "meldete", "gemeldet",
    "sagen", "sagte", "gesagt", "sagt",
    "machen", "machte", "gemacht", "macht",
    "gehen", "ging", "gegangen", "geht",
    "kommen", "kam", "gekommen", "kommt",
    "sehen", "sah", "gesehen", "sieht",
    "wissen", "wusste", "gewusst", "weiß",
    "geben", "gab", "gegeben", "gibt",
    "nehmen", "nahm", "genommen", "nimmt",
    "lassen", "ließ", "gelassen", "lässt",
    "tun", "tat", "getan", "tust",
    "habe", "hast", "habt", "bin", "bist", "ist", "seid",
    "werde", "wirst", "wird", "werdet",
    "am", "aufs", "durchs", "fürs", "hinters", "übers", "unters", "vors",
    "rüber", "runter", "rauf", "rein", "raus",
}

# Common English words (target should never be English)
_ENGLISH_COMMON = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "down", "left", "right", "all",
    "any", "some", "more", "most", "many", "much", "few", "little",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "done", "doing", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "very", "really", "quite", "just", "only", "also", "too", "not",
    "no", "never", "always", "sometimes", "often", "usually",
    "here", "there", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    "about", "above", "across", "after", "against", "along", "among",
    "around", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "inside", "outside", "throughout",
    "toward", "towards", "underneath", "until", "upon",
    "because", "since", "unless", "although", "though", "while",
    "if", "then", "else", "than", "as", "so", "such",
    "each", "every", "either", "neither", "both", "half",
    "nothing", "everything", "something", "anything",
    "nobody", "everybody", "somebody", "anybody",
    "now", "then", "today", "tomorrow", "yesterday",
    "am", "pm", "ok", "okay", "yes", "no", "please", "thanks",
    "hello", "hi", "hey", "goodbye", "bye",
    "well", "fine", "good", "bad", "great", "nice", "sure",
}

# Patterns that suggest garbage
_SUSPICIOUS_PATTERNS = [
    (re.compile(r"^[^a-zA-Z]"), "starts_with_non_alpha"),
    (re.compile(r"[0-9]{4,}"), "has_long_number"),
    (re.compile(r"^[a-z]$"), "single_lowercase"),
    (re.compile(r"^[A-Z]$"), "single_uppercase"),
    (re.compile(r"[|/\\]"), "contains_special"),
]


def _is_likely_german(word: str) -> bool:
    """Heuristic: is this word more likely German than English?
    
    Uses multiple signals to avoid false positives on English proper nouns.
    """
    w = word.lower().strip(",.!?;:'\"()[]♪-")
    if not w or len(w) < 2:
        return False
    if w in _GERMAN_STOPWORDS:
        return True
    # German-specific letter patterns (strong signal)
    if re.search(r"[äöüß]", w):
        return True
    # German suffixes (strong signal)
    if re.search(r"(chen|lein|heit|keit|tät|schaft|tum|nis|sal|ling)$", w):
        return True
    # German prefixes on longer words (strong signal)
    if re.search(r"^(ge|ver|zer|ent|emp|miss)", w) and len(w) > 5:
        return True
    # Weak signals — require multiple to trigger
    signals = 0
    # Capitalized noun (German capitalizes all nouns)
    if word[0].isupper() and len(word) > 1 and word[1].islower():
        signals += 1
    # Common German suffixes that also appear in English
    if re.search(r"(ung|ion|ling)$", w):
        signals += 1
    # Common German prefixes
    if re.search(r"^(er|be)", w) and len(w) > 4:
        signals += 1
    # Ends with e (common for German adjectives/adverbs)
    if w.endswith("e") and len(w) > 3:
        signals += 1
    return signals >= 2


def _is_likely_english(word: str) -> bool:
    """Heuristic: is this word more likely English than German?
    
    Avoids flagging German words that start with common English-like prefixes
    (e.g. 'Reiseerlaubnis' starts with 're' but is German).
    Requires multiple signals to reduce false positives.
    """
    w = word.lower().strip(",.!?;:'\"()[]♪-")
    if not w or len(w) < 2:
        return False
    if w in _ENGLISH_COMMON:
        return True
    signals = 0
    # English-specific ending patterns (strong signal)
    if re.search(r"(tion|sion|ment|ness|ity|ful|less|able|ible|ous|ive|ly)$", w):
        signals += 2
    # English-specific beginning patterns
    if re.search(r"^(th|sh|ch|wh|wr|kn|ph|ps)", w):
        signals += 2
    # English prefixes — weaker, only count if word doesn't look German
    if re.search(r"^(un|dis|mis|pre|ex)", w) and len(w) > 4:
        if not _is_likely_german(word):
            signals += 1
    return signals >= 2


def _heuristic_classify(source: str, target: str) -> list[str]:
    """Apply heuristics and return list of flags for suspicious entries.

    Uses conservative language detection to avoid false positives on
    proper nouns and English content words.
    """
    flags = []
    s_lower = source.lower().strip()
    t_lower = target.lower().strip()

    # Source is a German stopword (strong signal — these are never English)
    if s_lower in _GERMAN_STOPWORDS:
        flags.append("source_german_stopword")

    # Multi-word source: check if ALL words trigger German detection
    source_words = source.split()
    if len(source_words) > 1:
        german_words = sum(1 for w in source_words if _is_likely_german(w))
        english_words = sum(1 for w in source_words if _is_likely_english(w))
        # Only flag if most words look German AND few look English
        if german_words > len(source_words) / 2 and english_words == 0:
            flags.append("source_is_german")
        # Flag german_to_german only if BOTH sides strongly look German
        target_words = target.split()
        target_german = sum(1 for w in target_words if _is_likely_german(w))
        if german_words > len(source_words) / 2 and target_german > len(target_words) / 2:
            flags.append("german_to_german")
    else:
        # Single-word source
        if _is_likely_german(source):
            flags.append("source_is_german")
        if _is_likely_german(source) and _is_likely_german(target):
            flags.append("german_to_german")

    # Target is English (should be German) — only flag clear English targets
    target_words = target.split()
    english_target_words = sum(1 for w in target_words if _is_likely_english(w))
    german_target_words = sum(1 for w in target_words if _is_likely_german(w))
    if english_target_words > 0 and german_target_words == 0:
        flags.append("target_is_english")

    # Target is very short relative to source
    if len(source) > 5 and len(target) <= 2:
        flags.append("target_too_short")

    # Target is very long relative to source
    if len(source) <= 3 and len(target) > 15:
        flags.append("target_too_long")

    # Source and target are identical
    if source.lower() == target.lower():
        flags.append("identical")

    # Suspicious patterns
    for pattern, flag_name in _SUSPICIOUS_PATTERNS:
        if pattern.search(source) or pattern.search(target):
            flags.append(flag_name)

    # Target contains "//" (likely garbage from NLLB)
    if "//" in target:
        flags.append("contains_double_slash")

    # Source is very short and target is also garbage
    if len(source) <= 2 and len(target) <= 2:
        flags.append("both_too_short")

    return flags


def sanitize_auto_glossary() -> dict:
    """Analyze auto_glossary.json, classify entries, create backup, write clean version.

    Returns report dict with counts and details.
    """
    if not AUTO_GLOSSARY_PATH.exists():
        print("  [SANITIZE] auto_glossary.json not found — nothing to do.")
        return {"status": "not_found", "total": 0}

    # Load raw auto_glossary
    with open(AUTO_GLOSSARY_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print("  [SANITIZE] auto_glossary.json is empty.")
        return {"status": "empty", "total": 0}

    total = len(raw)
    print(f"  [SANITIZE] Loaded {total} entries from auto_glossary.json")

    # Classify each entry
    trusted = {}
    quarantine = {}
    rejected = {}
    entry_reports = []

    for source, target in raw.items():
        flags = _heuristic_classify(source, target)
        flags_str = ", ".join(flags) if flags else "ok"

        entry_report = {
            "source": source,
            "target": target,
            "flags": flags,
            "verdict": "",
        }

        if not flags:
            trusted[source] = target
            entry_report["verdict"] = "trusted"
        elif "source_german_stopword" in flags:
            # Source is a German word — clearly broken entry
            rejected[source] = target
            entry_report["verdict"] = "rejected"
        elif "identical" in flags and len(source) > 2:
            # Source and target are the same word — no translation
            rejected[source] = target
            entry_report["verdict"] = "rejected"
        elif "both_too_short" in flags:
            rejected[source] = target
            entry_report["verdict"] = "rejected"
        elif "contains_double_slash" in flags and "target_too_short" in flags:
            # Double slash + short target is clear garbage
            rejected[source] = target
            entry_report["verdict"] = "rejected"
        else:
            # Put in quarantine — might be valid, needs more evidence
            quarantine[source] = target
            entry_report["verdict"] = "quarantine"

        entry_reports.append(entry_report)

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)

    # Save raw backup
    shutil.copy2(AUTO_GLOSSARY_PATH, backup_path / "auto_glossary.json.bak")
    # Save classified versions
    for name, data in [("trusted", trusted), ("quarantine", quarantine), ("rejected", rejected)]:
        with open(backup_path / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"  [SANITIZE] Backup saved to {backup_path}")

    # Write sanitized auto_glossary.json (only trusted entries)
    with open(AUTO_GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(trusted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Move quarantined entries to learning_db as candidates
    try:
        from translator.learning_db import load_db, save_db
        db = load_db()
        now = datetime.now().isoformat()
        for source, target in quarantine.items():
            # Check if already in learning_db
            exists = any(
                c["source"] == source and c["target"] == target
                for c in db.get("candidates", [])
            )
            if not exists:
                db.setdefault("candidates", []).append({
                    "source": source,
                    "target": target,
                    "evidence_count": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "episodes_seen": [],
                    "qa_improvements": 0,
                    "qa_regressions": 0,
                    "model": "sanitize_migration",
                    "confidence": 0.05,
                    "status": "candidate",
                    "scope": "global",
                    "show": "",
                    "contradictions": [],
                })
        # Move rejected entries to learning_db as rejected
        for source, target in rejected.items():
            exists = any(
                c["source"] == source and c["target"] == target
                for c in db.get("candidates", [])
            )
            if not exists:
                db.setdefault("candidates", []).append({
                    "source": source,
                    "target": target,
                    "evidence_count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "episodes_seen": [],
                    "qa_improvements": 0,
                    "qa_regressions": 0,
                    "model": "sanitize_migration",
                    "confidence": 0.0,
                    "status": "rejected",
                    "scope": "global",
                    "show": "",
                    "contradictions": [],
                })
        save_db(db)
        print(f"  [SANITIZE] Quarantined {len(quarantine)} + rejected {len(rejected)} "
              f"entries migrated to learning_db.json")
    except Exception as e:
        print(f"  [SANITIZE] Could not update learning_db: {e}")
        print(f"  [SANITIZE] Quarantined/rejected data preserved in backup.")

    # Write report
    report = {
        "timestamp": timestamp,
        "total": total,
        "trusted": len(trusted),
        "quarantine": len(quarantine),
        "rejected": len(rejected),
        "backup_path": str(backup_path),
        "entries": entry_reports,
    }
    report_path = backup_path / "sanitize_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"  [SANITIZE] Report: {report_path}")
    print(f"  [SANITIZE] Trusted: {len(trusted)}, Quarantine: {len(quarantine)}, Rejected: {len(rejected)}")

    return report


def print_sanitize_summary(report: dict):
    """Print a human-readable summary of the sanitize report."""
    if "status" in report:
        print(f"  Sanitize: {report['status']}")
        return
    print(f"\n  {'═' * 50}")
    print(f"  LEARNING SANITIZATION REPORT")
    print(f"  {'═' * 50}")
    print(f"  Total entries analyzed: {report['total']}")
    print(f"  Trusted (kept):        {report['trusted']}")
    print(f"  Quarantined:           {report['quarantine']}")
    print(f"  Rejected:              {report['rejected']}")
    print(f"  Backup path:           {report['backup_path']}")
    print(f"  {'─' * 50}")

    if report.get("quarantine") and report.get("entries"):
        print(f"\n  QUARANTINED ENTRIES:")
        for e in report["entries"]:
            if e["verdict"] == "quarantine":
                print(f"    \"{e['source']}\" → \"{e['target']}\"  [{', '.join(e['flags'])}]")

    if report.get("rejected") and report.get("entries"):
        print(f"\n  REJECTED ENTRIES:")
        for e in report["entries"]:
            if e["verdict"] == "rejected":
                print(f"    \"{e['source']}\" → \"{e['target']}\"  [{', '.join(e['flags'])}]")

    print(f"  {'═' * 50}")
