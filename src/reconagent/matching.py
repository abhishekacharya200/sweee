"""Reference normalisation and fuzzy matching for mangled bank memos.

Bank memos are the dirtiest input in this whole system. `INV-2026-0142` comes
back as `INV20260142`, `INV 2026 0142`, `/RFB/INV-2026-142`, or OCR'd into
`INV2O26O142`. Everything here is stdlib-only and deterministic so the same
memo always produces the same candidate ranking — a fuzzy matcher that
reshuffles between runs is not auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Character confusions that only ever occur in the numeric tail of a reference.
# Applying these to the alpha prefix would turn "INV" into "1NV".
_DIGIT_LOOKALIKES = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}

_REF_SHAPE = re.compile(r"[A-Z]{2,4}[0-9OQILSBZ]{4,}")


def normalize_ref(raw: str) -> str:
    """Strip separators, upper-case, and repair digit look-alikes in the tail."""
    compact = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    match = re.match(r"^([A-Z]{2,4})(.*)$", compact)
    if not match:
        return compact
    prefix, tail = match.groups()
    return prefix + "".join(_DIGIT_LOOKALIKES.get(char, char) for char in tail)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def memo_ngrams(memo: str, max_n: int = 3) -> list[str]:
    """Candidate reference strings, including ones split across whitespace.

    `INV 2026 0142` is one reference wearing three tokens, so single-token
    scanning misses it entirely.
    """
    tokens = [t for t in re.split(r"[\s,;]+", memo.strip()) if t]
    grams: list[str] = []
    seen: set[str] = set()
    for size in range(1, max_n + 1):
        for start in range(len(tokens) - size + 1):
            gram = " ".join(tokens[start : start + size])
            if gram not in seen:
                seen.add(gram)
                grams.append(gram)
    return grams


def reference_shaped_substrings(gram: str) -> list[str]:
    """Pull reference-shaped runs out of a compacted n-gram.

    Substring search rather than a full match, because rails bolt their own
    prefixes on: `/RFB/INV-2026-0142` compacts to `RFBINV20260142`, which is
    not itself a reference but contains one.
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", gram).upper()
    return [m.group(0) for m in _REF_SHAPE.finditer(compact)]


@dataclass(frozen=True)
class RefCandidate:
    invoice_id: str
    score: float
    matched_text: str


def rank_reference_candidates(
    memo: str, invoice_ids: list[str], top_k: int = 5, min_score: float = 0.6
) -> list[RefCandidate]:
    """Rank known invoice ids by how well they explain the memo text.

    Ties break on invoice_id so the ranking is total and reproducible.
    """
    normalized = {inv_id: normalize_ref(inv_id) for inv_id in invoice_ids}
    best: dict[str, RefCandidate] = {}
    for gram in memo_ngrams(memo):
        for fragment in reference_shaped_substrings(gram):
            fragment_norm = normalize_ref(fragment)
            for inv_id, inv_norm in normalized.items():
                score = similarity(fragment_norm, inv_norm)
                if score < min_score:
                    continue
                current = best.get(inv_id)
                if current is None or score > current.score:
                    best[inv_id] = RefCandidate(inv_id, round(score, 4), fragment)
    ranked = sorted(best.values(), key=lambda c: (-c.score, c.invoice_id))
    return ranked[:top_k]
