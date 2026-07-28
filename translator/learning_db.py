"""Safe learning database with quarantine, evidence tracking, and controlled promotion.

Candidates accumulate evidence across episodes before promotion to auto_glossary.
This prevents single bad LLM outputs from poisoning the permanent glossary.
"""

import json
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
LEARNING_DB_PATH = CONFIG_DIR / "learning_db.json"
AUTO_GLOSSARY_PATH = CONFIG_DIR / "auto_glossary.json"

STATUS_CANDIDATE = "candidate"
STATUS_TRUSTED = "trusted"
STATUS_REJECTED = "rejected"
STATUS_SUPERSEDED = "superseded"

PROMOTION_MIN_EVIDENCE = 3
PROMOTION_MIN_CONFIDENCE = 0.4


def _load_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default if default is not None else {}


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_db() -> dict:
    db = _load_json(LEARNING_DB_PATH, {"candidates": [], "promoted_count": 0, "rejected_count": 0})
    if "candidates" not in db:
        db["candidates"] = []
    return db


def save_db(db: dict):
    _save_json(LEARNING_DB_PATH, db)


def _find_candidate(db: dict, source: str, target: str) -> dict | None:
    for c in db["candidates"]:
        if c["source"] == source and c["target"] == target:
            return c
    return None


def _find_contradictions(db: dict, source: str, target: str) -> list[str]:
    """Return targets of other active candidates with same source but different target."""
    return [
        c["target"] for c in db["candidates"]
        if c["source"] == source
        and c["target"] != target
        and c["status"] != STATUS_REJECTED
    ]


def _recalc_confidence(candidate: dict):
    """Recalculate confidence score 0.0–1.0 from evidence signals."""
    score = 0.0

    ev = candidate.get("evidence_count", 0)
    if ev >= 5:
        score += 0.35
    elif ev >= 3:
        score += 0.25
    elif ev >= 2:
        score += 0.15
    else:
        score += 0.05

    net_qa = candidate.get("qa_improvements", 0) - candidate.get("qa_regressions", 0)
    if net_qa > 5:
        score += 0.25
    elif net_qa > 2:
        score += 0.15
    elif net_qa > 0:
        score += 0.05
    elif net_qa < 0:
        score -= 0.10

    episodes = len(candidate.get("episodes_seen", []))
    if episodes >= 3:
        score += 0.20
    elif episodes >= 2:
        score += 0.10

    contra = len(candidate.get("contradictions", []))
    if contra:
        score -= 0.15 * min(contra, 3)

    candidate["confidence"] = max(0.0, min(1.0, score))


def add_evidence(
    source: str,
    target: str,
    episode_id: str = "",
    model: str = "",
    qa_before: int | None = None,
    qa_after: int | None = None,
    scope: str = "global",
    show: str = "",
) -> bool:
    """Record evidence for a source→target correction.

    Creates a new candidate or updates an existing one.
    Returns True if new candidate was created.
    """
    db = load_db()
    now = datetime.now().isoformat()

    candidate = _find_candidate(db, source, target)
    if candidate:
        candidate["evidence_count"] += 1
        candidate["last_seen"] = now
        if episode_id and episode_id not in candidate.get("episodes_seen", []):
            candidate.setdefault("episodes_seen", []).append(episode_id)
        if qa_before is not None and qa_after is not None:
            delta = qa_before - qa_after
            if delta > 0:
                candidate["qa_improvements"] = candidate.get("qa_improvements", 0) + delta
            elif delta < 0:
                candidate["qa_regressions"] = candidate.get("qa_regressions", 0) + abs(delta)
        _recalc_confidence(candidate)
        save_db(db)
        return False

    contradictions = _find_contradictions(db, source, target)
    candidate = {
        "source": source,
        "target": target,
        "evidence_count": 1,
        "first_seen": now,
        "last_seen": now,
        "episodes_seen": [episode_id] if episode_id else [],
        "qa_improvements": max(0, (qa_before or 0) - (qa_after or 0)),
        "qa_regressions": max(0, (qa_after or 0) - (qa_before or 0)),
        "model": model,
        "confidence": 0.0,
        "status": STATUS_CANDIDATE,
        "scope": scope,
        "show": show,
        "contradictions": contradictions,
    }
    _recalc_confidence(candidate)
    db["candidates"].append(candidate)
    save_db(db)
    return True


def promote_candidates() -> int:
    """Promote qualified candidates to auto_glossary. Returns count promoted."""
    db = load_db()
    auto = _load_json(AUTO_GLOSSARY_PATH, {})
    promoted = 0

    for c in db.get("candidates", []):
        if c.get("status") != STATUS_CANDIDATE:
            continue
        if c.get("confidence", 0) < PROMOTION_MIN_CONFIDENCE:
            continue
        if c.get("evidence_count", 0) < PROMOTION_MIN_EVIDENCE:
            continue

        # Check contradictions: don't promote if a trusted contradiction exists
        contra = c.get("contradictions", [])
        if contra:
            stronger_conflict = any(
                x.get("status") == STATUS_TRUSTED
                and x.get("confidence", 0) > c["confidence"]
                for x in db["candidates"]
                if x["target"] in contra
            )
            if stronger_conflict:
                continue

        c["status"] = STATUS_TRUSTED
        auto[c["source"]] = c["target"]
        promoted += 1

    if promoted:
        _save_json(AUTO_GLOSSARY_PATH, auto)
        db["promoted_count"] = db.get("promoted_count", 0) + promoted
        save_db(db)

    return promoted


def get_effective_glossary() -> dict:
    """Return merged auto_glossary + trusted candidates (for use during translation)."""
    auto = dict(_load_json(AUTO_GLOSSARY_PATH, {}))
    db = load_db()
    for c in db.get("candidates", []):
        if c.get("status") == STATUS_TRUSTED:
            key = c["source"]
            if key not in auto:
                auto[key] = c["target"]
    return auto


def reject_candidate(source: str, target: str) -> bool:
    """Manually flag a candidate as rejected. Returns True if found."""
    db = load_db()
    c = _find_candidate(db, source, target)
    if c:
        c["status"] = STATUS_REJECTED
        save_db(db)
        return True
    return False


def mark_superseded(source: str, old_target: str, new_target: str) -> bool:
    """When a better translation emerges, mark the old one as superseded."""
    db = load_db()
    c = _find_candidate(db, source, old_target)
    if c:
        c["status"] = STATUS_SUPERSEDED
        c["superseded_by"] = new_target
        save_db(db)
        return True
    return False


def get_stats() -> dict:
    """Return summary statistics about the learning database."""
    db = load_db()
    candidates = db.get("candidates", [])
    total = len(candidates)
    trusted = sum(1 for c in candidates if c.get("status") == STATUS_TRUSTED)
    candidate_count = sum(1 for c in candidates if c.get("status") == STATUS_CANDIDATE)
    rejected = sum(1 for c in candidates if c.get("status") == STATUS_REJECTED)
    superseded = sum(1 for c in candidates if c.get("status") == STATUS_SUPERSEDED)
    return {
        "total_candidates": total,
        "trusted": trusted,
        "candidate": candidate_count,
        "rejected": rejected,
        "superseded": superseded,
        "promoted_count": db.get("promoted_count", 0),
        "quarantine_count": candidate_count,
    }
