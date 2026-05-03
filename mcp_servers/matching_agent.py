"""
Faculty Analysis + Advisor Matching Agent
==========================================

Stages
------
  A  Candidate intake      — from ExtractionOutcome (name, profile_urls,
                             source_sections, role_hint, raw_title)
  B  Early exclusion       — cheap, rule-based; drops definitive non-advisors
                             (emeritus, postdoc, student, staff, alumni)
  C  Profile enrichment    — concurrent _fetch_one_profile per candidate;
                             produces richer bio/research/title/email/website
  D  Scholar acquisition   — profile link → website scan → DDG fallback → author search
  E  Scoring pipeline      — _run_prescreen (keyword filter + batch LLM)
                             → _score_profile → _score_scholar
                             → _overall_match (dynamic scholar weight by confidence)
                             → summary table + JSON report

Scoring weights
---------------
  _score_profile()    research*0.4 + method*0.3 + app*0.2 + style*0.1 + bonus
  _overall_match()    profile*(1-w) + scholar*w
                        w = 0.35 (direct link / high confidence)
                        w = 0.25 (medium confidence)
                        w = 0.15 (low confidence / fallback only)
"""

from __future__ import annotations

import concurrent.futures
import threading as _threading
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Describes the applicant.  Passed to run_matching_agent()."""
    research_interests: str
    keywords: list[str] = field(default_factory=list)
    degree_target: str = "PhD"
    institution_preferences: list[str] = field(default_factory=list)


@dataclass
class MatchingOutcome:
    """Structured result returned by run_matching_agent()."""
    source_url: str
    total_candidates: int
    after_early_exclusion: int
    enriched_count: int
    scholar_searched: int
    excluded: list[dict]
    warnings: list[str]
    top_results: list[dict]

# ── Re-use advisor_server utilities ──────────────────────────────────────────
import mcp_servers.advisor_server as srv

# ── Re-use extraction agent ───────────────────────────────────────────────────
from mcp_servers.extraction_agent import run_extraction_agent, ExtractionOutcome

SEP  = "─" * 74
SEP2 = "═" * 74

# ---------------------------------------------------------------------------
# Kahuna memory — faculty profile debug log
# ---------------------------------------------------------------------------

_KAHUNA_FACULTY_DIR = Path.home() / ".kahuna" / "knowledge" / "faculty"


def _kahuna_record_faculty_profile(
    name: str,
    profile_url: str,
    scholar_url: str,
    scholar_source: str,
    scholar_confidence: str,
    strategies_tried: list[str],
    personal_website: str,
    cv_url: str,
    errors: list[str],
) -> None:
    """Record per-professor Scholar lookup debug info to ~/.kahuna/knowledge/faculty/."""
    _KAHUNA_FACULTY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^\w\-]", "_", name)[:40]
    record = {
        "name": name,
        "profile_url": profile_url,
        "timestamp": ts,
        "scholar": {
            "url": scholar_url,
            "source": scholar_source,
            "confidence": scholar_confidence,
            "strategies_tried": strategies_tried,
        },
        "external_info": {
            "personal_website": personal_website,
            "cv_url": cv_url,
        },
        "errors": errors,
    }
    fname = _KAHUNA_FACULTY_DIR / f"{safe_name}_{ts}.json"
    try:
        fname.write_text(json.dumps(record, indent=2))
        logger.info(
            "[Kahuna/faculty] %s: source=%s confidence=%s errors=%d",
            name, scholar_source, scholar_confidence, len(errors),
        )
    except Exception as exc:
        logger.warning("[Kahuna/faculty] Failed to write for %s: %s", name, exc)

# ---------------------------------------------------------------------------
# Stop words for keyword prescreen
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
    'at', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can', 'this', 'that', 'these', 'those', 'my',
    'our', 'their', 'its', 'as', 'i', 'we', 'you', 'he', 'she', 'it',
    'they', 'me', 'us', 'him', 'her', 'them', 'not', 'no', 'but', 'also',
    'university', 'professor', 'faculty', 'department', 'associate', 'assistant',
    'adjunct', 'research', 'study', 'work', 'include', 'including', 'well',
    'new', 'use', 'using', 'used', 'based', 'provide', 'approach', 'method',
})

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

CV_NORMALIZE_SYSTEM = """\
You are a research keyword extractor for PhD advisor matching.
Given a CV in ANY language, output ONLY a raw JSON object. No markdown. No preamble.

{
  "keywords": ["<English keyword 1>", "<English keyword 2>", ...],
  "research_summary": "<2-3 sentence English summary of the applicant's research background and interests>"
}

Rules:
- keywords: 10-20 specific English terms covering research areas, methods, application domains, tools
- research_summary: written in English regardless of the CV's original language
- Be specific: prefer "reinforcement learning" over "AI", "biomedical NLP" over "healthcare"
- ENGLISH ONLY in the output — translate all terms"""


def _normalize_cv_keywords(
    client: anthropic.Anthropic,
    cv_text: str,
) -> tuple[list[str], str]:
    """
    For non-English CVs: call LLM once to extract English keywords + summary.
    Returns (english_keywords, english_summary).
    Both are empty strings/lists when the CV is already English-dominant.
    """
    if not _is_nonnative_heavy(cv_text):
        return [], ""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=CV_NORMALIZE_SYSTEM,
            messages=[{"role": "user", "content": cv_text[:6000]}],
        )
        result = _extract_json(msg.content[0].text)
        keywords = [str(k) for k in (result.get("keywords") or [])[:20]]
        summary  = str(result.get("research_summary", ""))
        logger.info(
            "[CV normalize] Non-English CV → extracted %d English keywords: %s",
            len(keywords), keywords[:5],
        )
        return keywords, summary
    except Exception as exc:
        logger.warning("[CV normalize] Keyword extraction failed: %s", exc)
        return [], ""


PRESCREEN_SYSTEM = """\
You are a PhD advisor matching assistant.
Given a student CV and a numbered list of faculty profiles, pick the top 10 best matches.
Output ONLY a raw JSON array. No markdown. No preamble.

LANGUAGE RULE: ALL text fields MUST be in English only. No Chinese, no Korean, no Japanese, no other non-Latin scripts.

Schema (return exactly this structure):
[
  {"rank": 1, "index": <faculty number 1-based>, "name": "<exact faculty name from list>", "prescreen_score": <1-10>, "prescreen_reason": "<one sentence English>"},
  ...
]

Rules:
- rank 1 = best match, 10 = weakest of the top 10
- index = the 1-based faculty number EXACTLY as shown in the input list (first faculty = 1, second = 2, ...)
- name = copy the faculty name exactly from the input list
- prescreen_score: integer 1-10
- prescreen_reason: ONE concise sentence in English citing concrete overlap with CV
- Return EXACTLY 10 entries (or fewer if fewer faculty given)"""

PROFILE_SCORE_SYSTEM = """\
You are a PhD advisor matching auditor.
Output ONLY a raw JSON object. Start with { end with }. No markdown fences. No preamble.

LANGUAGE RULE: ALL text fields MUST be in English only. No Chinese, no Korean, no Japanese, no other non-Latin scripts.
IMPORTANT: All score values MUST be integers between 1 and 10 (e.g. 7, not 0.7 or 70%).

DATA PRIORITY: Use evidence in this order (highest to lowest confidence):
  1. OpenAlex research topics — curated labels, use as primary research signal
  2. Faculty personal website excerpt — researcher's own description
  3. Faculty CV excerpt — explicit research interests listed
  4. Bio / Research fields on profile page — use when above are absent
If bio and research fields are empty but OpenAlex topics are present, base your scores on the OpenAlex topics.

MISSING DATA RULE: If ALL sources above are absent or contain only awards/news/credentials,
score ALL four dimensions 2 and set data_quality_note="Insufficient profile data".

SCORING RUBRIC — score SPECIFIC overlap, not generic overlap:
- research_match: Do the professor's SPECIFIC research topics match the student's SPECIFIC research topics?
    1-2 = completely different fields (e.g. cement chemistry vs NLP/LLMs — no overlap)
    3-4 = adjacent domain (e.g. transportation systems vs logistics optimization)
    5-6 = same broad area (e.g. both use ML but for unrelated problems)
    7-8 = same specific area (e.g. both do computer vision for robotics)
    9-10 = near-identical focus (e.g. both work on LLM interpretability)
- method_match: Do they share specific techniques? (e.g. transformers, GNNs, Bayesian methods)
    1-2 = entirely different methods; 5-6 = one overlapping technique; 9-10 = same core toolkit
- application_match: Do their application DOMAINS overlap specifically?
    1-2 = completely different domains; 5-6 = loosely adjacent; 9-10 = same target application
- style_match: Lab culture, collaboration style, industry vs theory orientation.
    1-2 = opposite orientations; 5-6 = neutral; 9-10 = strong alignment

Schema (fill ALL fields):
{
  "research_match":    {"score": <INTEGER 1-10>, "evidence": "<cite CV field AND profile field>"},
  "method_match":      {"score": <INTEGER 1-10>, "evidence": "..."},
  "application_match": {"score": <INTEGER 1-10>, "evidence": "..."},
  "style_match":       {"score": <INTEGER 1-10>, "evidence": "..."},
  "bonus_penalty":     {"value": <float -1.0 to 1.0>, "reason": "..."},
  "faculty_profile_match": <float = research*0.4 + method*0.3 + app*0.2 + style*0.1 + bonus_penalty.value>,
  "score_explanation": "...",
  "data_quality_note": "..."
}
Only cite information present in the provided texts."""

SCHOLAR_SCORE_SYSTEM = """\
You are a PhD advisor matching auditor analyzing recent publications.
Output ONLY a raw JSON object. Start with { end with }. No markdown. No preamble.

LANGUAGE RULE: ALL text fields MUST be in English only. No Chinese, no Korean, no Japanese, no other non-Latin scripts.

Schema:
{
  "recent_themes":       ["<theme1>", "<theme2>", ...],
  "recent_methods":      ["<method1>", ...],
  "recent_applications": ["<app1>", ...],
  "alignment_points":    ["<specific paper/topic that overlaps with CV>", ...],
  "divergence_points":   ["<area in Scholar not in CV, or vice versa>", ...],
  "recent_scholar_match": {
    "score": <1-10>,
    "evidence": "<cite specific paper titles AND CV fields>",
    "strongest_overlap_paper": "<title of paper most relevant to CV>",
    "weakest_area": "<topic in recent work that CV does not cover>"
  },
  "data_quality_note": "..."
}"""

# ---------------------------------------------------------------------------
# JSON / score helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | list:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting the first complete JSON object or array
    for start_ch, end_ch in [('{', '}'), ('[', ']')]:
        start = raw.find(start_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except Exception:
                        break
    return {"error": "no JSON", "raw": raw[:200]}


# Phrases in an evidence or explanation field that signal a LOW score
_CONTRADICTION_SIGNALS = re.compile(
    r"\b(no\s+(overlap|alignment|connection|relation|match|shared|similar|relevant|correspondence)|"
    r"completely\s+different|entirely\s+different|unrelated|not\s+relevant|no\s+meaningful|"
    r"no\s+direct|lacks?\s+overlap|lacks?\s+alignment|"
    r"different\s+(field|domain|area|application|discipline)|"
    r"no\s+specific\s+overlap|not\s+aligned|no\s+significant\s+overlap)\b",
    re.IGNORECASE,
)

# If global score_explanation strongly signals no match, cap all dimension scores
_GLOBAL_NO_MATCH_SIGNALS = re.compile(
    r"\b(no\s+meaningful|completely\s+unrelated|entirely\s+unrelated|"
    r"no\s+meaningful\s+alignment|no\s+significant\s+alignment)\b",
    re.IGNORECASE,
)


def _normalize_scores(sc: dict) -> dict:
    weights = {"research_match": 0.4, "method_match": 0.3,
               "application_match": 0.2, "style_match": 0.1}

    # Check global score_explanation for a blanket "no alignment" statement
    global_no_match = bool(_GLOBAL_NO_MATCH_SIGNALS.search(sc.get("score_explanation", "")))

    for key in weights:
        d = sc.get(key, {})
        s = d.get("score")
        if isinstance(s, (int, float)):
            if s <= 1.5:          # LLM used 0–1 scale; rescale to 1–10
                s = round(s * 10, 1)
            s = min(10.0, max(1.0, float(s)))
            # Contradiction guard: clamp high score when evidence says no overlap
            if s > 4.0:
                evidence = d.get("evidence", "")
                if _CONTRADICTION_SIGNALS.search(evidence) or global_no_match:
                    s = min(s, 3.0)
                    d["_contradiction_clamp"] = True
            d["score"] = s
    bp = sc.get("bonus_penalty") or {}
    raw_bonus = bp.get("value", 0) if isinstance(bp, dict) else 0
    bonus = max(-1.0, min(1.0, float(raw_bonus) if isinstance(raw_bonus, (int, float)) else 0))
    # Always recompute from clamped components to avoid LLM formula errors/overflow
    total = sum(sc.get(k, {}).get("score", 0) * w for k, w in weights.items())
    sc["faculty_profile_match"] = round(min(11.0, max(0.0, total + bonus)), 2)
    return sc

# ---------------------------------------------------------------------------
# Prescreen (two-step: keyword filter → batch LLM)
# ---------------------------------------------------------------------------

def _extract_cv_keywords(cv_text: str, extra_keywords: list[str] | None = None) -> frozenset[str]:
    freq: dict[str, int] = {}
    for w in re.findall(r'\b[a-zA-Z]{3,}\b', cv_text.lower()):
        if w not in _STOP_WORDS:
            freq[w] = freq.get(w, 0) + 1
    # Boost LLM-extracted English keywords for non-English CVs
    for w in (extra_keywords or []):
        for part in re.findall(r'\b[a-zA-Z]{3,}\b', w.lower()):
            if part not in _STOP_WORDS:
                freq[part] = freq.get(part, 0) + 5   # high weight
    top = sorted(freq, key=freq.__getitem__, reverse=True)[:200]
    return frozenset(top)


def _keyword_score(cv_keywords: frozenset[str], profile: dict) -> float:
    text = ' '.join([
        profile.get('name', ''),
        profile.get('title', ''),
        profile.get('bio', '')[:400],
        profile.get('research_interests', '')[:400],
    ]).lower()
    faculty_words = frozenset(re.findall(r'\b[a-z]{3,}\b', text)) - _STOP_WORDS
    if not faculty_words or not cv_keywords:
        return 0.0
    return len(cv_keywords & faculty_words) / len(cv_keywords)


def _prescreen_batch_llm(client: anthropic.Anthropic, cv_text: str, profiles: list[dict]) -> list[dict]:
    lines = []
    for i, p in enumerate(profiles, 1):
        name  = p.get('name', '')
        title = (p.get('title', '') or '')[:60]
        bio   = (p.get('bio', '') or '')[:120]
        ri    = (p.get('research_interests', '') or '')[:120]
        lines.append(f"{i}. {name} | {title} | {bio} {ri}".strip())

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=PRESCREEN_SYSTEM,
        messages=[{"role": "user", "content": (
            f"CV:\n{cv_text[:2000]}\n\n"
            f"Faculty ({len(profiles)} total):\n" + "\n".join(lines)
        )}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    start = raw.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except Exception:
                        break
    logger.warning("[prescreen] LLM output parse failed: %s", raw[:200])
    return []


def _run_prescreen(
    client: anthropic.Anthropic,
    cv_text: str,
    valid_profiles: list[dict],
    top_n: int = 10,
    kw_pool: "int | None" = None,
    cv_keywords_en: list[str] | None = None,
    cv_summary_en: str = "",
) -> list[dict]:
    """Keyword filter → batch LLM. Returns ≤top_n profile dicts with prescreen fields."""
    if kw_pool is None:
        kw_pool = KW_POOL_SIZE
    cv_kw = _extract_cv_keywords(cv_text, extra_keywords=cv_keywords_en)
    kw_ranked = sorted(valid_profiles, key=lambda p: _keyword_score(cv_kw, p), reverse=True)[:kw_pool]
    logger.info("[prescreen] keyword filter: %d → %d", len(valid_profiles), len(kw_ranked))
    if not kw_ranked:
        return []

    # For non-English CVs, prepend English summary to the prescreen prompt
    prescreen_cv = cv_text
    if cv_summary_en:
        prescreen_cv = f"[Applicant profile in English]: {cv_summary_en}\n\n[Full CV]:\n{cv_text}"
    llm_results = _prescreen_batch_llm(client, prescreen_cv, kw_ranked)
    logger.info("[prescreen] LLM returned %d results", len(llm_results))

    # Build name→position lookup for fallback matching
    name_to_pos: dict[str, int] = {}
    for pos, p in enumerate(kw_ranked):
        n = (p.get("name") or "").strip().lower()
        if n:
            name_to_pos[n] = pos

    shortlist: list[dict] = []
    seen_pos: set[int] = set()

    def _resolve_item(item: dict) -> int | None:
        """Return 0-based position in kw_ranked, or None if unresolvable."""
        idx = item.get("index")
        if isinstance(idx, int):
            # Try 1-based (expected)
            if 1 <= idx <= len(kw_ranked):
                return idx - 1
            # Try 0-based (LLM offset-by-one error)
            if 0 <= idx < len(kw_ranked):
                return idx
        # Fallback: match by name field LLM included in output
        iname = (item.get("name") or "").strip().lower()
        if iname:
            # exact match
            if iname in name_to_pos:
                return name_to_pos[iname]
            # partial: check if any key starts with item name or vice versa
            for key, pos in name_to_pos.items():
                if key.startswith(iname[:10]) or iname.startswith(key[:10]):
                    return pos
        return None

    for item in llm_results[:top_n]:
        pos = _resolve_item(item)
        if pos is None or pos in seen_pos:
            continue
        seen_pos.add(pos)
        p = kw_ranked[pos].copy()
        p["prescreen_score"]  = item.get("prescreen_score", 5)
        p["prescreen_reason"] = item.get("prescreen_reason", "")
        shortlist.append(p)

    if not shortlist:
        logger.warning("[prescreen] index mapping failed — falling back to keyword top-N")
        for p in kw_ranked[:top_n]:
            pc = p.copy()
            pc["prescreen_score"]  = 5
            pc["prescreen_reason"] = "Selected by keyword overlap"
            shortlist.append(pc)
    elif len(shortlist) < len(llm_results):
        logger.debug("[prescreen] resolved %d/%d LLM items via index/name matching",
                     len(shortlist), len(llm_results))

    return shortlist

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

_RESEARCH_SIGNAL_WORDS = frozenset({
    "research", "study", "investigat", "focus", "interest", "working on",
    "develop", "model", "algorithm", "machine learning", "deep learning",
    "neural", "optimiz", "design", "analys", "experiment", "laborator",
    "publications", "lab ", "projects", "methods", "technique",
})

# Words that indicate awards/credential text rather than research description
_AWARD_WORDS = ("award", "fellowship", "scholarship", "grant", "prize", "honor", "receipt")


def _profile_has_research_content(profile: dict) -> bool:
    """Return True when ANY data source can support research-fit scoring.

    Checked in priority order:
      1. OA topics  — curated topic labels, always reliable
      2. Personal website content (medium/high confidence)
      3. Faculty CV content (fetched separately)
      4. Profile page bio  — first/third-person description
      5. Profile page research_interests — only if not awards-dominated
    Only returns False when every source is absent or awards-only.
    """
    # 1. OpenAlex topics — curated and reliable even when profile page is sparse
    if profile.get("openalex_topics"):
        return True

    # 2. Personal website with sufficient confidence
    web_conf = profile.get("personal_website_confidence", "")
    if web_conf in ("medium", "high") and profile.get("website_content"):
        return True

    # 3. Faculty CV content (skip binary/unparsed PDF bytes)
    _cv = profile.get("cv_content") or ""
    if _cv and not _cv.startswith("%PDF"):
        return True

    bio      = (profile.get("bio") or "").strip()
    research = (profile.get("research_interests") or "").strip()

    # 4. Bio is most reliable from the profile page itself
    if len(bio) >= 60 and any(sig in bio.lower() for sig in _RESEARCH_SIGNAL_WORDS):
        return True

    # 5. research_interests: accept when signal words outnumber award/credential words.
    #    (Avoids "Scientific and Technological Research Council" false positives.)
    if len(research) >= 60:
        r_lower      = research.lower()
        signal_count = sum(1 for w in _RESEARCH_SIGNAL_WORDS if w in r_lower)
        award_count  = sum(r_lower.count(w) for w in _AWARD_WORDS)
        if signal_count >= 1 and signal_count > award_count:
            return True

    return False


_INSUFFICIENT_PROFILE_SCORE: dict = {
    "research_match":    {"score": 2, "evidence": "No research content on profile page — awards/title only"},
    "method_match":      {"score": 2, "evidence": "No research content on profile page"},
    "application_match": {"score": 2, "evidence": "No research content on profile page"},
    "style_match":       {"score": 5, "evidence": "Cannot assess — no profile content"},
    "bonus_penalty":     {"value": -0.3, "reason": "Profile page lacks research description"},
    "faculty_profile_match": 1.9,
    "score_explanation": "Profile page has no research content; cannot assess fit.",
    "data_quality_note": "Insufficient: bio and research_interests empty or awards-only on profile page.",
}


def _score_profile(
    client: anthropic.Anthropic,
    cv_text: str,
    profile: dict,
    cv_summary_en: str = "",
) -> dict:
    if not _profile_has_research_content(profile):
        return _INSUFFICIENT_PROFILE_SCORE.copy()
    # CV block: prepend English summary when CV is non-English so LLM has
    # clear English context alongside the original text
    cv_block = f"CV:\n{cv_text[:5500]}"
    if cv_summary_en:
        cv_block = f"Applicant profile (English): {cv_summary_en}\n\n" + cv_block

    extra = ""
    cv_text_raw = profile.get("cv_content") or ""
    # Skip binary / unparsed PDF bytes — they confuse the LLM and inflate scores
    if cv_text_raw and not cv_text_raw.startswith("%PDF"):
        extra += f"\n\nFaculty CV (excerpt):\n{cv_text_raw[:1500]}"
    web_conf = profile.get("personal_website_confidence", "")
    if profile.get("website_content") and web_conf in ("medium", "high"):
        extra += f"\n\nFaculty website (excerpt):\n{profile['website_content'][:800]}"
    oa_topics = profile.get("openalex_topics") or []
    if oa_topics:
        extra += f"\n\nOpenAlex research topics: {'; '.join(oa_topics)}"
    oa_stats = profile.get("openalex_stats") or {}
    h_idx = oa_stats.get("h_index")
    if h_idx is not None:
        extra += (
            f"\nAcademic metrics: h-index={h_idx}"
            f", total citations={oa_stats.get('cited_by_count', 0)}"
            f", total works={oa_stats.get('works_count', 0)}"
        )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=PROFILE_SCORE_SYSTEM,
        messages=[{"role": "user", "content": (
            f"{cv_block}\n\n"
            f"FACULTY\nName: {profile.get('name','')}\n"
            f"Title: {profile.get('title','')}\n"
            f"Bio: {profile.get('bio','')[:700]}\n"
            f"Research: {profile.get('research_interests','')[:500]}"
            f"{extra}"
        )}],
    )
    return _normalize_scores(_extract_json(msg.content[0].text))


def _score_scholar(client: anthropic.Anthropic, cv_text: str, scholar_data: dict) -> dict:
    pubs = scholar_data.get("publications", [])
    lines = "\n".join(
        f"  [{p.get('year','?')}] {p.get('title','')} — {p.get('venue','')}"
        for p in pubs
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=SCHOLAR_SCORE_SYSTEM,
        messages=[{"role": "user", "content": (
            f"CV:\n{cv_text[:4000]}\n\nRecent publications ({len(pubs)}):\n{lines}"
        )}],
    )
    return _extract_json(msg.content[0].text)


def _scholar_weight(scholar_confidence: str) -> float:
    """Scholar contribution weight based on how the profile was found."""
    if scholar_confidence == "low":
        return 0.15
    elif scholar_confidence == "medium":
        return 0.25
    else:  # "" (direct link from page) or "high"
        return 0.35


def _overall_match(profile_sc: dict, scholar_sc: dict | None, scholar_confidence: str = "") -> float:
    pm = profile_sc.get("faculty_profile_match", 0.0)
    if scholar_sc and "recent_scholar_match" in scholar_sc:
        sm = scholar_sc["recent_scholar_match"].get("score", 0)
        w  = _scholar_weight(scholar_confidence)
        return round(pm * (1 - w) + sm * w, 2)
    return round(pm, 2)


def _print_faculty(p: dict, rank: int) -> None:
    ps = p.get("profile_scoring", {})
    ss = p.get("scholar_scoring")
    pm = ps.get("faculty_profile_match", "?")
    sm = ss["recent_scholar_match"]["score"] if ss and "recent_scholar_match" in ss else "N/A"
    om = p.get("overall_match", "?")
    print(f"\n  #{rank}  {p['name']}")
    print(f"  {SEP}")
    print(f"  overall={om}  profile={pm}  scholar={sm}")
    print(f"  url: {p.get('profile_url','')}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CANDIDATES       = 100  # Stage A cap
ENRICH_WORKERS       = 10   # Stage C profile-fetch threads (pure I/O, different domains)
OPENALEX_WORKERS     = 10   # kept for benchmark script; main pipeline uses ENRICH_WORKERS
SCHOLAR_URL_WORKERS  = 10   # Stage D part-1: URL discovery (DDG/personal-website, not Scholar)
SCHOLAR_PUB_WORKERS  = 5    # Stage D part-2: scholar.google.com/citations page scraping
SCHOLAR_WORKERS      = 5    # alias kept for benchmark script compatibility
SCORE_WORKERS        = 5    # Stage E — Anthropic Sonnet rate limit (~50 RPM Tier1)
RECOMMEND_WORKERS    = 5    # Stage F — test: if 429 appears, drop to 3
TOP_N_PRESCREEN      = 10   # how many pass LLM pre-screen → all get scored and returned
KW_POOL_SIZE         = 20   # keyword filter pool fed to prescreen LLM (was 40)
TOP_N_PRE_FILTER     = 30   # Stage C.5: keep top-N after profile fetch, before OA/Scholar
TOP_N_SCHOLAR_SEARCH = 30   # Stage D: Scholar search (matches pre-filter size)


# ---------------------------------------------------------------------------
# Stage B — Early exclusion (cheap, no LLM, no network)
# ---------------------------------------------------------------------------

_EXCLUDE_ROLE_HINTS: frozenset[str] = frozenset({
    "student", "staff", "postdoc", "lecturer",
})

_EARLY_EXCLUDE_RE = re.compile(
    r"\b("
    r"emeritus|emerita|emeriti|"
    r"retired|"
    r"lecturer|"
    r"teaching\s+(?:professor|faculty|fellow)|"
    r"postdoc(?:toral)?(?:\s+(?:fellow|researcher|associate))?|"
    r"ph\.?d\.?\s+(?:student|candidate)|"
    r"doctoral\s+(?:student|candidate)|"
    r"graduate\s+student|"
    r"ms\s+student|master.?s?\s+student|"
    r"undergraduate\s+student|"
    r"alumni|alumnus|alumna|"
    r"administrative\s+(?:staff|coordinator|manager)|"
    r"program\s+(?:coordinator|manager)"
    r")\b",
    re.IGNORECASE,
)

# Matches any role that grants PhD advising eligibility.
# Intentionally broad: adjunct/visiting/research/clinical professors all kept.
_PHD_ADVISOR_RE = re.compile(r"\bprofessor\b", re.IGNORECASE)

# Separators between concurrent roles in a single title string.
# Avoid splitting on commas — they appear inside department names.
_ROLE_SEP_RE = re.compile(r"\s*[;|/]\s*|\s+&\s+|\n+", re.IGNORECASE)


def _early_exclude(candidate: dict) -> tuple[bool, str]:
    """
    Return (should_exclude, reason_string).
    No LLM, no network.  Only remove definitive non-advisors.
    If title is missing or ambiguous → keep.

    Multi-role rule: split raw_title into individual role segments. If ANY
    segment grants PhD advising eligibility (contains 'professor' without also
    matching an exclusion pattern in the same segment), keep the person.
    """
    role_hint = candidate.get("role_hint", "")
    raw_title = candidate.get("raw_title", "") or ""

    if role_hint in _EXCLUDE_ROLE_HINTS:
        return True, f"role_hint={role_hint}"

    segments = [s.strip() for s in _ROLE_SEP_RE.split(raw_title) if s.strip()] or [raw_title]

    has_exclude = False
    first_exclude_reason = ""

    for seg in segments:
        m = _EARLY_EXCLUDE_RE.search(seg)
        if m:
            # This segment is a non-advising role
            if not has_exclude:
                first_exclude_reason = f"title contains '{m.group(0).strip()}'"
            has_exclude = True
        elif _PHD_ADVISOR_RE.search(seg):
            # At least one segment is a PhD-advising role → keep regardless
            return False, ""

    if has_exclude:
        return True, first_exclude_reason

    for link in candidate.get("source_links", []):
        section = link.get("source_section", "").lower()
        if any(w in section for w in ("student", "alumni", "emerit")):
            return True, f"source_section '{section}' indicates non-advisor"

    return False, ""


# ---------------------------------------------------------------------------
# Preliminary selection score  (internal only — NOT the final score)
# ---------------------------------------------------------------------------
#
# Used ONLY to rank candidates before expensive Stage C/D calls so we
# prioritise the most-relevant ones.  Never shown in output as a score.

def _tokenize(text: str) -> set[str]:
    _STOP = {
        "the", "and", "for", "with", "that", "this", "are", "was",
        "has", "have", "its", "their", "from", "into", "our", "can",
        "not", "but", "all", "been", "more", "also", "will", "such",
        "than", "when", "use", "used", "how", "new", "using", "data",
    }
    tokens: set[str] = {
        w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP
    }
    # CJK (Chinese/Japanese/Korean): add individual characters + bigrams
    cjk = re.findall(
        r"[一-鿿぀-ゟ゠-ヿ가-힯㐀-䶿]",
        text,
    )
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return tokens


def _is_nonnative_heavy(text: str, threshold: float = 0.20) -> bool:
    """True when >threshold of characters are non-ASCII (CJK, Arabic, Cyrillic, etc.)."""
    if not text:
        return False
    return sum(1 for c in text if ord(c) > 127) / len(text) > threshold


def _preliminary_selection_score(candidate: dict, user_kw: set[str]) -> float:
    """
    Cheap keyword overlap heuristic.
    Used internally to decide who gets expensive enrichment + Scholar search.
    Never exposed as final matching_score.
    """
    if not user_kw:
        return 0.5
    sections = " ".join(
        sl.get("source_section", "")
        for sl in candidate.get("source_links", [])
    )
    candidate_kw = _tokenize(sections)
    if not candidate_kw:
        return 0.3
    inter = len(user_kw & candidate_kw)
    return min(1.0, inter / max(1, len(user_kw)) * 1.5)


def _profile_pre_score(enriched: dict, cv_kw: set[str]) -> float:
    """
    Richer pre-score used after profile fetch (Stage C.5) to select top-N before OA/Scholar.
    Uses bio, research_interests, website_content, and source_sections from the profile page.
    """
    if not cv_kw:
        return 0.5
    texts = [
        enriched.get("bio") or "",
        enriched.get("research_interests") or "",
        enriched.get("website_content") or "",
        " ".join(
            sl.get("source_section", "")
            for sl in enriched.get("source_links", [])
        ),
    ]
    combined = " ".join(texts)
    candidate_kw = _tokenize(combined)
    if not candidate_kw:
        # No profile text at all — small base score; Scholar link is a slight signal
        return 0.05 + (0.05 if enriched.get("google_scholar") else 0.0)
    inter = len(cv_kw & candidate_kw)
    score = min(1.0, inter / max(1, len(cv_kw)) * 2.0)
    # Tiny bonus for a valid, content-rich page
    if enriched.get("page_valid"):
        score = min(1.0, score + 0.03)
    return score


# ---------------------------------------------------------------------------
# Stage C — Profile enrichment
# ---------------------------------------------------------------------------

def _enrich_profile(candidate: dict) -> dict:
    """
    Fetch the best available profile page and return an enriched dict
    in the format expected by _score_profile() and _prescreen().

    Returns a superset of _fetch_one_profile() output, with extra fields:
      - profile_urls     (all URLs tried)
      - source_sections  (research topic headings from extraction page)
      - early_excluded   / early_exclusion_reason
    """
    profile_urls = candidate.get("profile_urls", [])
    if not profile_urls:
        fu = candidate.get("full_profile_url", "") or candidate.get("profile_url", "")
        profile_urls = [fu] if fu else []

    source_sections = [
        sl.get("source_section", "")
        for sl in candidate.get("source_links", [])
        if sl.get("source_section")
    ]

    enriched = None
    for purl in profile_urls[:2]:       # try at most 2 URLs
        if not purl:
            continue
        entry = {**candidate, "full_profile_url": purl}
        result = srv._fetch_one_profile(entry)
        if result.get("page_valid"):
            enriched = result
            break
        enriched = enriched or result   # keep first attempt for error reporting

    if enriched is None:
        # No URLs at all — create a minimal shell
        enriched = {
            "name":               candidate.get("name", ""),
            "title":              "",
            "profile_url":        profile_urls[0] if profile_urls else "",
            "bio":                "",
            "research_interests": "",
            "personal_website":   "",
            "personal_website_confidence": "",
            "google_scholar":     "",
            "cv_url":             "",
            "error":              "no profile URL available",
            "page_valid":         False,
            "page_type":          "unknown",
            "fetch_metadata":     {},
        }

    # Augment with extraction-stage metadata
    enriched["profile_urls"]    = profile_urls
    enriched["source_sections"] = source_sections
    enriched["role_hint"]       = candidate.get("role_hint", "")
    enriched["raw_title"]       = candidate.get("raw_title", "")
    enriched["groups"]          = candidate.get("groups", [])

    # Extraction agent's name overrides profile-scraped name — BUT only when it
    # looks like a real full name (≥2 words).  Single-word extraction names are
    # usually just a last name (e.g. "Wilson", "Lei") and should yield to the
    # full name the profile page actually contains ("Andrew Gordon Wilson").
    candidate_name = candidate.get("name", "")
    if len(candidate_name.split()) >= 2:
        enriched["name"] = candidate_name

    # early_exclusion pass-through (already annotated by Stage B)
    enriched["early_excluded"]          = candidate.get("early_excluded", False)
    enriched["early_exclusion_reason"]  = candidate.get("early_exclusion_reason", "")

    # Fetch extra text for richer scoring (cv + personal website)
    cv_url      = enriched.get("cv_url", "")
    website_url = enriched.get("personal_website", "")
    if cv_url and not enriched.get("cv_content"):
        try:
            enriched["cv_content"] = srv._fetch_extra_content(cv_url, 3000)
        except Exception:
            pass
    if website_url and not enriched.get("website_content"):
        try:
            enriched["website_content"] = srv._fetch_extra_content(website_url, 2000)
        except Exception:
            pass

    logger.info(
        "[Enrichment] %s: page_valid=%s scholar=%s website=%s cv=%s",
        enriched["name"],
        enriched.get("page_valid"),
        bool(enriched.get("google_scholar")),
        bool(enriched.get("personal_website")),
        bool(enriched.get("cv_content")),
    )
    return enriched


# ---------------------------------------------------------------------------
# Stage D — Scholar acquisition
# ---------------------------------------------------------------------------

def _acquire_scholar_url(enriched: dict) -> None:
    """
    Stage D part-1: find the Scholar profile URL.
    Hits university websites, personal pages, DDG — NOT scholar.google.com.
    Safe to run with SCHOLAR_URL_WORKERS (10) concurrent threads.
    """
    if enriched.get("google_scholar"):
        enriched["scholar_source"]           = "profile"
        enriched["scholar_match_confidence"] = "high"
        return

    source_sections = enriched.get("source_sections", [])
    research_kw     = enriched.get("research_interests", "") or ""

    result = srv._scholar_fallback_search(
        name=enriched.get("name", ""),
        profile_url=enriched.get("profile_url", ""),
        research_interests=", ".join(source_sections) + " " + research_kw,
        personal_website=enriched.get("personal_website", ""),
        cv_url=enriched.get("cv_url", ""),
    )

    enriched["google_scholar"]           = result.get("scholar_url", "")
    enriched["scholar_source"]           = result.get("scholar_source", "not_found")
    enriched["scholar_match_confidence"] = result.get("scholar_match_confidence", "")

    logger.info(
        "[Scholar] %s: url=%s source=%s confidence=%s",
        enriched.get("name"),
        bool(enriched["google_scholar"]),
        enriched["scholar_source"],
        enriched["scholar_match_confidence"],
    )

    _kahuna_record_faculty_profile(
        name=enriched.get("name", ""),
        profile_url=enriched.get("profile_url", ""),
        scholar_url=enriched["google_scholar"],
        scholar_source=enriched["scholar_source"],
        scholar_confidence=enriched["scholar_match_confidence"],
        strategies_tried=result.get("strategies_tried", []),
        personal_website=enriched.get("personal_website", ""),
        cv_url=enriched.get("cv_url", ""),
        errors=result.get("errors", []),
    )


def _acquire_scholar_pubs(enriched: dict) -> None:
    """
    Stage D part-2: scrape scholar.google.com/citations?user=XXX for publications.
    Must run with SCHOLAR_PUB_WORKERS (5) to avoid Scholar bot detection.
    """
    gs_url = enriched.get("google_scholar", "")
    if not gs_url or enriched.get("scholar_data"):
        return
    try:
        enriched["scholar_data"] = srv._fetch_scholar_pubs(gs_url)
    except Exception as exc:
        logger.warning("[Scholar] pub fetch error for %s: %s", enriched.get("name"), exc)
        enriched["scholar_data"] = {"fetch_status": "error", "publications": [], "pub_count": 0}


def _acquire_scholar(enriched: dict) -> None:
    """Combined URL + pub fetch. Used by benchmark script."""
    _acquire_scholar_url(enriched)
    _acquire_scholar_pubs(enriched)


# ---------------------------------------------------------------------------
# Stage C.5 — OpenAlex academic enrichment
# ---------------------------------------------------------------------------

def _acquire_openalex(enriched: dict) -> None:
    """
    Query OpenAlex for the faculty member's academic profile.
    Mutates enriched in-place, adding:
      openalex_id, openalex_url, openalex_display_name,
      openalex_topics (list[str]), openalex_stats (dict),
      openalex_works (list of recent pubs), orcid, openalex_confidence
    """
    name = enriched.get("name", "")
    if not name:
        return

    # Derive institution hint and department hint from profile URL domain
    profile_url = enriched.get("profile_url", "") or ""
    inst_hint = ""
    dept_hint = ""
    if profile_url:
        from urllib.parse import urlparse as _urlparse
        parts = _urlparse(profile_url).netloc.lower().replace("www.", "").split(".")
        raw_inst = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
        # Map common university domain abbreviations to a keyword that appears in
        # the full institution name as stored in OpenAlex (e.g. "ufl" → "florida").
        # Map domain abbreviations to a substring that uniquely appears in the
        # full OA institution name.  Use specific phrases so short hints don't
        # match unrelated institutions (e.g. "florida" → "Florida College").
        _DOMAIN_TO_HINT = {
            "ufl":       "university of florida",
            "gatech":    "georgia institute",
            "cmu":       "carnegie mellon",
            "mit":       "massachusetts institute",
            "ucla":      "university of california, los angeles",
            "usc":       "university of southern california",
            "umd":       "university of maryland",
            "uva":       "university of virginia",
            "umn":       "university of minnesota",
            "uiuc":      "university of illinois",
            "upenn":     "university of pennsylvania",
            "psu":       "penn state",
            "osu":       "ohio state",
            "uw":        "university of washington",
            "bu":        "boston university",
            "nyu":       "new york university",
            "jhu":       "johns hopkins",
            "wustl":     "washington university in st",
            "tamu":      "texas a&m",
            "ucsd":      "university of california, san diego",
            "ucsb":      "university of california, santa barbara",
            "ucsc":      "university of california, santa cruz",
            "uci":       "university of california, irvine",
            "ucr":       "university of california, riverside",
            "ucdavis":   "university of california, davis",
            "uchicago":  "university of chicago",
            "umich":     "university of michigan",
            "wisc":      "university of wisconsin",
            "ncsu":      "north carolina state",
            "vt":        "virginia tech",
            "utexas":    "university of texas at austin",
            "utk":       "university of tennessee",
            "uga":       "university of georgia",
            "pitt":      "university of pittsburgh",
            "cwru":      "case western reserve",
            "rpi":       "rensselaer",
            "wpi":       "worcester polytechnic",
            "stevens":   "stevens institute",
            "uwaterloo": "university of waterloo",
            "utoronto":  "university of toronto",
            "mcgill":    "mcgill university",
            "ubc":       "university of british columbia",
            "usyd":      "university of sydney",
            "unsw":      "university of new south wales",
        }
        inst_hint = _DOMAIN_TO_HINT.get(raw_inst, raw_inst)
        # First subdomain component is often a department abbreviation (e.g. "cee", "orc", "cs")
        raw_dept = parts[0] if len(parts) >= 3 else ""
        _GENERIC_SUBDOMAINS = {"web", "faculty", "people", "www", "home", "staff", "sites", "my"}
        dept_hint = "" if raw_dept in _GENERIC_SUBDOMAINS else raw_dept

    orcid = enriched.get("orcid", "") or ""

    # Collect research keywords from the faculty's own profile to help disambiguate
    # same-name same-institution candidates in OpenAlex.  Only non-empty strings matter.
    _ri  = enriched.get("research_interests", "") or ""
    _bio = enriched.get("bio", "") or ""
    research_keywords: list[str] = [s for s in [_ri[:300], _bio[:150]] if s]

    author = srv._openalex_author_search(
        name, inst_hint, orcid=orcid, dept_hint=dept_hint,
        research_keywords=research_keywords or None,
    )
    if not author:
        enriched.setdefault("openalex_id", "")
        enriched.setdefault("openalex_url", "")
        enriched.setdefault("openalex_confidence", "not_found")
        enriched.setdefault("openalex_topics", [])
        enriched.setdefault("openalex_stats", {})
        enriched.setdefault("openalex_works", [])
        return

    author_id = author.get("id", "")
    stats     = author.get("summary_stats", {}) or {}

    enriched["openalex_id"]           = author_id
    enriched["openalex_url"]          = author_id  # OpenAlex IDs are canonical URLs
    enriched["openalex_display_name"] = author.get("display_name", "")
    enriched["openalex_topics"]       = [
        t.get("display_name", "") for t in (author.get("topics") or [])[:8]
    ]
    enriched["openalex_stats"] = {
        "h_index":         stats.get("h_index"),
        "i10_index":       stats.get("i10_index"),
        "cited_by_count":  author.get("cited_by_count", 0),
        "works_count":     author.get("works_count", 0),
        "counts_by_year":  author.get("counts_by_year", []),
    }
    enriched["openalex_confidence"] = author.get("_confidence", "medium")

    # ORCID — propagate only if not already set
    ids   = author.get("ids", {}) or {}
    orcid = ids.get("orcid", "") or ""
    if orcid and not enriched.get("orcid"):
        enriched["orcid"] = orcid

    # Fetch recent works (used as Scholar fallback + recommendation evidence)
    if author_id:
        enriched["openalex_works"] = srv._openalex_works(author_id, per_page=20)
    else:
        enriched["openalex_works"] = []

    logger.info(
        "[OpenAlex] %s: conf=%s h=%s cited=%d topics=%d pubs=%d",
        name,
        enriched["openalex_confidence"],
        enriched["openalex_stats"].get("h_index", "?"),
        enriched["openalex_stats"].get("cited_by_count", 0),
        len(enriched["openalex_topics"]),
        len(enriched["openalex_works"]),
    )


# ---------------------------------------------------------------------------
# Stage E — Feed enriched materials into the ORIGINAL scoring pipeline
# ---------------------------------------------------------------------------

def _run_original_scoring(
    client,
    cv_text: str,
    enriched_profiles: list[dict],
    output_path: Path,
    dept_name: str,
    source_url: str,
    progress_cb=None,
    cv_summary_en: str = "",
) -> list[dict]:
    """
    Calls the ORIGINAL scoring pipeline functions unchanged:
      _prescreen → _score_profile → _score_scholar → _overall_match
      _print_faculty → summary table → JSON report

    Returns the final ranked list (same structure as run_full_pipeline.py).
    """

    # ── Pre-screen: keyword filter → batch LLM ────────────────────────────────
    logger.info("[Scoring] Pre-screening %d profiles...", len(enriched_profiles))
    if len(enriched_profiles) <= TOP_N_PRESCREEN:
        # Already small — skip LLM prescreen to save ~3s; use keyword score order
        cv_kw = _extract_cv_keywords(cv_text)
        top_profiles = sorted(
            [dict(p, prescreen_score=5, prescreen_reason="auto (small cohort)")
             for p in enriched_profiles],
            key=lambda p: _keyword_score(cv_kw, p), reverse=True,
        )
        logger.info("[Scoring] Skipped LLM prescreen (N=%d ≤ %d)",
                    len(enriched_profiles), TOP_N_PRESCREEN)
    else:
        top_profiles = _run_prescreen(
            client, cv_text, enriched_profiles, top_n=TOP_N_PRESCREEN,
            cv_summary_en=cv_summary_en,
        )
    if not top_profiles:
        logger.warning("[Scoring] Pre-screen returned nothing — using all profiles")
        top_profiles = [
            dict(p, prescreen_score=5, prescreen_reason="fallback")
            for p in enriched_profiles[:TOP_N_PRESCREEN]
        ]
    prescreened = top_profiles  # kept for JSON report back-compat

    print(f"\n  Pre-screen top {len(top_profiles)} candidates:")
    for p in top_profiles:
        print(f"    {p.get('prescreen_score','?'):>4}  {p.get('name','?'):<30}  {p.get('prescreen_reason','')[:60]}")

    if progress_cb:
        names = [p.get("name", "?") for p in top_profiles]
        progress_cb({"type": "batch_progress", "step": "shortlisted",
                     "top_n": len(top_profiles), "names": names,
                     "message": f"Pre-screen complete — shortlisted {len(top_profiles)}: "
                                + ", ".join(names[:5]) + ("…" if len(names) > 5 else "")})
        progress_cb({"type": "batch_progress", "step": "full_analysis_start",
                     "message": f"Phase 2 — full analysis on {len(top_profiles)} candidates…"})

    # ── Scholar fetch + full LLM scoring for top candidates ──────────────────
    logger.info("[Scoring] Scholar + profile scoring for %d candidates...", len(top_profiles))

    n_top = len(top_profiles)
    done_count = 0

    def _score_one(p: dict) -> dict:
        gs_url = p.get("google_scholar", "")
        if p.get("scholar_data"):
            # Already pre-fetched in Stage D — skip redundant HTTP fetch
            sd = p["scholar_data"]
            sf = sd.get("fetch_status", "?")
            pc = sd.get("pub_count", 0)
            print(f"  → Scholar (cached): {p['name']} — {sf} ({pc} pubs)")
        elif gs_url:
            print(f"  → Scholar (late fetch): {p['name']}...")
            p["scholar_data"] = srv._fetch_scholar_pubs(gs_url)
            sf = p["scholar_data"]["fetch_status"]
            pc = p["scholar_data"].get("pub_count", 0)
            print(f"     {'✅' if sf=='success' else '❌'} {sf}  ({pc} pubs)")
        else:
            p["scholar_data"] = {"fetch_status": "no_url", "publications": [], "pub_count": 0}
            print(f"  → Scholar: {p['name']} — no URL")

        # OpenAlex fallback: use when Scholar has no pubs
        sd = p.get("scholar_data", {})
        oa_works = p.get("openalex_works") or []
        if oa_works and (sd.get("fetch_status") != "success" or sd.get("pub_count", 0) == 0):
            oa_pubs = [{"year": w["year"], "title": w["title"], "venue": w["venue"]}
                       for w in oa_works]
            p["scholar_data"] = {
                "fetch_status": "openalex",
                "publications":  oa_pubs,
                "pub_count":     len(oa_pubs),
                "source":        "openalex",
            }
            # Use "low" Scholar weight for OpenAlex pub data (fallback source)
            if not p.get("scholar_match_confidence"):
                p["scholar_match_confidence"] = "low"
            print(f"     OpenAlex fallback: {len(oa_pubs)} pubs for {p['name']}")

        print(f"  → Profile score: {p['name']}...")
        p["profile_scoring"] = _score_profile(client, cv_text, p, cv_summary_en=cv_summary_en)

        sd = p.get("scholar_data", {})
        if sd.get("fetch_status") in ("success", "openalex") and sd.get("pub_count", 0) > 0:
            print(f"     Scholar score: {p['name']} ({sd.get('fetch_status')})...")
            p["scholar_scoring"] = _score_scholar(client, cv_text, sd)
        else:
            p["scholar_scoring"] = None

        p["overall_match"] = _overall_match(
            p["profile_scoring"],
            p["scholar_scoring"],
            scholar_confidence=p.get("scholar_match_confidence", ""),
        )
        return p

    _done_lock = _threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
        futs = {pool.submit(_score_one, p): p for p in top_profiles}
        for fut in concurrent.futures.as_completed(futs):
            try:
                scored_p = fut.result()
            except Exception as exc:
                scored_p = futs[fut]
                logger.warning("[Scoring] Error scoring %s: %s", scored_p.get("name"), exc)
            with _done_lock:
                done_count += 1
                n = done_count
            if progress_cb:
                progress_cb({"type": "batch_progress", "step": "detail",
                             "current": n, "total": n_top,
                             "message": f"Scored {n}/{n_top}: {scored_p['name']}…"})

    # Sort; return all scored profiles (prescreen already caps the set size)
    top_profiles.sort(key=lambda p: p.get("overall_match", 0), reverse=True)
    final_top = top_profiles   # no further truncation

    # ── ORIGINAL terminal output ──────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  RESULTS — TOP {len(final_top)}")
    print(SEP2)
    for rank, p in enumerate(final_top, 1):
        _print_faculty(p, rank)

    # ORIGINAL summary table
    print(f"\n{SEP2}")
    print("  SUMMARY TABLE")
    print(SEP2)
    hdr = (
        f"{'Rank':<5} {'Name':<28} {'Overall':>8}  "
        f"{'Profile':>8}  {'Scholar':>8}  {'Pubs':>5}  Pre-screen"
    )
    print(hdr)
    print("─" * len(hdr))
    for rank, p in enumerate(final_top, 1):
        ps   = p.get("profile_scoring", {})
        ss   = p.get("scholar_scoring")
        pm   = ps.get("faculty_profile_match", "?")
        sm   = (ss["recent_scholar_match"]["score"]
                if ss and "recent_scholar_match" in ss else "N/A")
        om   = p.get("overall_match", "?")
        pubs = p.get("scholar_data", {}).get("pub_count", 0)
        pre  = p.get("prescreen_score", "?")
        print(f"{rank:<5} {p['name']:<28} {str(om):>8}  {str(pm):>8}  {str(sm):>8}  {pubs:>5}  {pre}")

    # ── ORIGINAL JSON report ──────────────────────────────────────────────────
    report = {
        "meta": {
            "generated_at":          datetime.now().isoformat(),
            "source_url":            source_url,
            "department":            dept_name,
            "total_faculty_found":   len(enriched_profiles),
            "prescreened_count":     len(prescreened),
            "final_count":           len(final_top),
            "scoring_model":         "claude-haiku-4-5-20251001",
            "scoring_weights": {
                "faculty_profile_match": (
                    "research*0.4 + method*0.3 + app*0.2 + style*0.1 + bonus"
                ),
                "overall_match": (
                    "profile*(1-w) + scholar*w  w=0.35/0.25/0.15 by scholar confidence"
                ),
            },
        },
        "all_prescreened": prescreened,
        "faculty": [
            {
                "final_rank":    rank,
                "prescreen_rank": next(
                    (i + 1 for i, x in enumerate(prescreened)
                     if x.get("name", "").lower() in p["name"].lower()
                     or p["name"].lower() in x.get("name", "").lower()),
                    None,
                ),
                "name":            p["name"],
                "prescreen_score": p.get("prescreen_score"),
                "prescreen_reason": p.get("prescreen_reason", ""),
                "urls": {
                    "faculty_profile":  p.get("profile_url", ""),
                    "personal_website": p.get("personal_website") or None,
                    "google_scholar":   p.get("google_scholar") or None,
                },
                "fetch_status": {
                    "profile_page": p.get("fetch_metadata", {}).get("profile_page", {}),
                    "google_scholar": {
                        "actual_fetch_status": p.get("scholar_data", {}).get("fetch_status"),
                        "http_status":         p.get("scholar_data", {}).get("http_status"),
                        "pubs_retrieved":      p.get("scholar_data", {}).get("pub_count", 0),
                        "error":               p.get("scholar_data", {}).get("error", ""),
                    },
                },
                "extracted_from_profile": {
                    "bio":                p.get("bio", ""),
                    "research_interests": p.get("research_interests", ""),
                },
                "extracted_from_scholar": {
                    "publications": p.get("scholar_data", {}).get("publications", []),
                    "pub_count":    p.get("scholar_data", {}).get("pub_count", 0),
                },
                "scoring": {
                    "profile_scoring":       p.get("profile_scoring", {}),
                    "scholar_scoring":       p.get("scholar_scoring"),
                    "faculty_profile_match": p.get("profile_scoring", {}).get(
                        "faculty_profile_match"
                    ),
                    "recent_scholar_match": (
                        p["scholar_scoring"]["recent_scholar_match"]["score"]
                        if p.get("scholar_scoring")
                        and "recent_scholar_match" in p["scholar_scoring"]
                        else None
                    ),
                    "overall_match": p.get("overall_match"),
                },
            }
            for rank, p in enumerate(final_top, 1)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    size = output_path.stat().st_size
    print(f"\n✅ Saved: {output_path}  ({size:,} bytes)")

    return final_top


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_matching_agent(
    source_url: str,
    user_profile: UserProfile,
    *,
    extraction_outcome: Optional[ExtractionOutcome] = None,
    output_path: Optional[Path] = None,
    progress_cb=None,
    debug_limit: Optional[int] = None,
    skip_scholar: bool = False,
) -> MatchingOutcome:
    """
    Full pipeline: A → B → C → D → E (original scoring).

    Args:
        source_url:         Department page URL (input to extraction agent).
        user_profile:       Applicant profile (research_interests used as CV text).
        extraction_outcome: Pre-computed ExtractionOutcome to skip extraction.
        output_path:        Where to write JSON report.
        progress_cb:        Optional callable(dict) for streaming progress events.

    Returns:
        MatchingOutcome with top_results mapped from the original scoring pipeline.
    """
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Use research_interests as the CV text fed into the original scoring prompts.
    cv_text   = user_profile.research_interests
    dept_name = (
        ", ".join(user_profile.institution_preferences)
        if user_profile.institution_preferences
        else source_url
    )

    # ── CV language normalisation ─────────────────────────────────────────────
    # For non-English CVs, extract English keywords once and reuse throughout.
    # This fixes keyword prescreening while keeping original CV for LLM scoring.
    cv_keywords_en, cv_summary_en = _normalize_cv_keywords(client, cv_text)
    if cv_keywords_en:
        # Use LLM-extracted English keywords for all keyword-matching stages
        user_keywords: list[str] = cv_keywords_en
        if progress_cb:
            progress_cb({"type": "progress", "step": "cv_normalize",
                         "message": f"Non-English CV detected — extracted {len(cv_keywords_en)} English keywords"})
        logger.info(
            "[MatchingAgent] Non-English CV normalised: %d keywords, summary=%d chars",
            len(cv_keywords_en), len(cv_summary_en),
        )
    else:
        user_keywords = user_profile.keywords or []
        cv_summary_en = ""

    if output_path is None:
        output_path = (
            Path(__file__).parent.parent / "outputs" / "matching_agent_report.json"
        )

    warnings: list[str] = []

    # ── Stage A: Candidate intake ─────────────────────────────────────────────
    logger.info("[MatchingAgent] Stage A: extraction for %s", source_url)
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "A", "label": "Candidate Intake", "ts": time.time()})
    if extraction_outcome is None:
        if progress_cb:
            progress_cb({"type": "progress", "step": "extract",
                         "message": f"Extracting faculty from {source_url}…"})
        extraction_outcome = run_extraction_agent(source_url)

    if not extraction_outcome.success or not extraction_outcome.faculty_list:
        logger.error(
            "[MatchingAgent] Extraction failed: %s", extraction_outcome.failure_reason
        )
        warnings.append(f"Extraction failed: {extraction_outcome.failure_reason}")
        return MatchingOutcome(
            source_url=source_url,
            total_candidates=0,
            after_early_exclusion=0,
            enriched_count=0,
            scholar_searched=0,
            excluded=[],
            warnings=warnings,
            top_results=[],
        )

    candidates = extraction_outcome.faculty_list[:MAX_CANDIDATES]
    total_candidates = len(candidates)
    print(f"\n  [A] Extracted: {total_candidates} candidates from {source_url}")
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "A", "label": "Candidate Intake",
                     "ts": time.time(), "out": total_candidates})

    # ── Stage B: Early exclusion ──────────────────────────────────────────────
    logger.info("[MatchingAgent] Stage B: early exclusion")
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "B", "label": "Early Exclusion",
                     "ts": time.time(), "in": total_candidates})
    kept, excluded_list = [], []
    for c in candidates:
        excl, reason = _early_exclude(c)
        if excl:
            c["early_excluded"]         = True
            c["early_exclusion_reason"] = reason
            excluded_list.append(c)
        else:
            c["early_excluded"]         = False
            c["early_exclusion_reason"] = ""
            kept.append(c)

    print(f"  [B] After early exclusion: {len(kept)} kept, {len(excluded_list)} excluded")
    for e in excluded_list:
        print(f"       excluded: {e.get('name')}  ({e['early_exclusion_reason']})")
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "B", "label": "Early Exclusion",
                     "ts": time.time(), "in": total_candidates,
                     "out": len(kept), "excluded": len(excluded_list),
                     "excluded_names": [e.get("name","") for e in excluded_list]})

    if not kept:
        logger.warning("[MatchingAgent] All candidates excluded — nothing to score")
        warnings.append("All candidates were excluded by early exclusion rules")
        return MatchingOutcome(
            source_url=source_url,
            total_candidates=total_candidates,
            after_early_exclusion=0,
            enriched_count=0,
            scholar_searched=0,
            excluded=[{"name": e["name"], "reason": e["early_exclusion_reason"]} for e in excluded_list],
            warnings=warnings,
            top_results=[],
        )

    # debug_limit: cap the candidate list for faster debug runs
    if debug_limit is not None and len(kept) > debug_limit:
        logger.info("[MatchingAgent] debug_limit: capping %d → %d", len(kept), debug_limit)
        warnings.append(f"debug_limit={debug_limit}: kept only first {debug_limit} candidates")
        kept = kept[:debug_limit]
        if progress_cb:
            progress_cb({"type": "progress", "step": "debug_limit",
                         "message": f"debug_limit={debug_limit}: truncated to {debug_limit} candidates"})

    # Optional cheap pre-sort using preliminary_selection_score so we enrich
    # the most-relevant candidates first (helpful when len(kept) > ENRICH_WORKERS*2).
    user_kw: set[str] = _tokenize(" ".join(user_keywords))
    if user_kw:
        kept.sort(
            key=lambda c: _preliminary_selection_score(c, user_kw),
            reverse=True,
        )
        logger.info(
            "[MatchingAgent] Stage B pre-sort by preliminary_selection_score done"
        )

    # ── Stage C: Profile fetch only (all candidates, parallel HTTP) ─────────
    logger.info("[MatchingAgent] Stage C: fetching %d profiles", len(kept))
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "C", "label": "Profile Fetch",
                     "ts": time.time(), "in": len(kept)})
        progress_cb({"type": "progress", "step": "profile_fetch",
                     "message": f"Fetching profile pages for {len(kept)} candidates…"})

    enriched_all: list[dict] = []
    _enrich_lock  = _threading.Lock()
    _enrich_count = [0]

    def _enrich_only(c: dict) -> dict:
        e = _enrich_profile(c)
        with _enrich_lock:
            _enrich_count[0] += 1
            n = _enrich_count[0]
        if progress_cb:
            progress_cb({"type": "batch_progress", "step": "profile_fetch",
                         "current": n, "total": len(kept),
                         "message": f"Profile {n}/{len(kept)}: {e.get('name', '')}…"})
        return e

    with concurrent.futures.ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        futures = {pool.submit(_enrich_only, c): c for c in kept}
        for fut in concurrent.futures.as_completed(futures):
            try:
                enriched_all.append(fut.result())
            except Exception as exc:
                c = futures[fut]
                logger.warning("[MatchingAgent] C error for %s: %s", c.get("name"), exc)
                warnings.append(f"Profile fetch error for {c.get('name')}: {exc}")

    valid_count = sum(1 for e in enriched_all if e.get("page_valid"))
    print(f"  [C] Profile fetch: {valid_count}/{len(enriched_all)} valid")
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "C", "label": "Profile Fetch",
                     "ts": time.time(), "in": len(kept), "out": len(enriched_all),
                     "page_valid": valid_count})
        progress_cb({"type": "batch_progress", "step": "fetch_done",
                     "valid": valid_count, "total": len(enriched_all),
                     "message": f"Profiles fetched: {valid_count}/{len(enriched_all)} valid."})

    # ── Stage C.5: Pre-filter → top-N using profile-page content ─────────────
    # Uses bio + research_interests + source_sections for keyword overlap with CV.
    # Only top-N proceed to the expensive OA + Scholar stages.
    cv_kw_frozen = _extract_cv_keywords(cv_text, extra_keywords=list(user_kw) if user_kw else None)
    logger.info("[MatchingAgent] Stage C.5: pre-filtering %d → top-%d",
                len(enriched_all), TOP_N_PRE_FILTER)
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "C5", "label": "Pre-Filter",
                     "ts": time.time(), "in": len(enriched_all)})

    # Immediately drop candidates whose profile page returned 404 / invalid —
    # no data to score them, and they waste LLM budget.
    invalid_profiles = [e for e in enriched_all if not e.get("page_valid", True)]
    enriched_all     = [e for e in enriched_all if e.get("page_valid", True)]
    if invalid_profiles:
        print(f"  [C.5] Dropped {len(invalid_profiles)} invalid-profile candidate(s): "
              + ", ".join(e["name"] for e in invalid_profiles))

    for e in enriched_all:
        e["_pre_score"] = _profile_pre_score(e, cv_kw_frozen)
    enriched_all.sort(key=lambda e: e["_pre_score"], reverse=True)

    enriched   = enriched_all[:TOP_N_PRE_FILTER]   # proceed to OA + Scholar + LLM
    cut_off    = enriched_all[TOP_N_PRE_FILTER:]    # dropped

    print(f"  [C.5] Pre-filter: kept {len(enriched)}, dropped {len(cut_off)}")
    if cut_off:
        print(f"        Dropped: {', '.join(e['name'] for e in cut_off[:10])}"
              + (f" …+{len(cut_off)-10}" if len(cut_off) > 10 else ""))
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "C5", "label": "Pre-Filter",
                     "ts": time.time(), "in": len(enriched_all), "out": len(enriched),
                     "dropped": len(cut_off)})

    # ── Stage OA: OpenAlex for pre-filtered candidates only ──────────────────
    logger.info("[MatchingAgent] Stage OA: OpenAlex for %d candidates", len(enriched))
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "OA", "label": "OpenAlex",
                     "ts": time.time(), "in": len(enriched)})
        progress_cb({"type": "progress", "step": "openalex",
                     "message": f"Querying OpenAlex for {len(enriched)} candidates…"})

    _oa_lock  = _threading.Lock()
    _oa_count = [0]

    def _oa_only(e: dict) -> None:
        _acquire_openalex(e)
        with _oa_lock:
            _oa_count[0] += 1
            n = _oa_count[0]
        if progress_cb:
            progress_cb({"type": "batch_progress", "step": "openalex",
                         "current": n, "total": len(enriched),
                         "message": f"OpenAlex {n}/{len(enriched)}: {e.get('name', '')}…"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        list(pool.map(_oa_only, enriched))

    oa_found = sum(1 for e in enriched if e.get("openalex_id"))
    print(f"  [OA] OpenAlex: {oa_found}/{len(enriched)} matched")
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "OA", "label": "OpenAlex",
                     "ts": time.time(), "in": len(enriched), "oa_found": oa_found})

    for e in enriched:
        stats   = e.get("openalex_stats", {}) or {}
        bio_len = len(e.get("bio") or "")
        print(
            f"       {e['name']:<28}  page_valid={e.get('page_valid')}  "
            f"bio={bio_len}c  scholar={bool(e.get('google_scholar'))}  "
            f"oa={e.get('openalex_confidence','—')}  h={stats.get('h_index','?')}  "
            f"pubs={len(e.get('openalex_works',[]))}"
        )

    # ── Stage P: Planner — decide per-faculty actions based on OA signal ─────
    logger.info("[MatchingAgent] Stage P: planning execution for %d candidates", len(enriched))
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "P", "label": "Planner",
                     "ts": time.time(), "in": len(enriched)})

    planner_skips = 0
    priority_counts = {"high": 0, "medium": 0, "low": 0}
    for e in enriched:
        # Always search Scholar — recent papers (last 2-3 yrs) matter more than OA coverage.
        # Only skip when explicitly forced by the --skip-scholar CLI flag (debug mode).
        e["_plan_skip_scholar"] = bool(skip_scholar)
        if e["_plan_skip_scholar"]:
            planner_skips += 1

        # Priority: topic overlap between OA topics and CV keywords
        topic_text = " ".join(e.get("openalex_topics", []))
        topic_kw   = frozenset(re.findall(r"[a-z]{3,}", topic_text.lower())) - _STOP_WORDS
        overlap    = len(cv_kw_frozen & topic_kw) if (cv_kw_frozen and topic_kw) else 0
        e["_plan_priority"] = "high" if overlap >= 4 else "medium" if overlap >= 1 else "low"
        priority_counts[e["_plan_priority"]] += 1

    print(
        f"  [P] Planner: {len(enriched) - planner_skips} Scholar searches scheduled"
        + (f" ({planner_skips} forced-skipped by --skip-scholar)" if planner_skips else "")
    )
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "P", "label": "Planner",
                     "ts": time.time(), "skip_scholar": planner_skips,
                     "priority": priority_counts, "skip_scholar_forced": skip_scholar})

    # ── Stage D: Scholar acquisition (all pre-filtered candidates) ────────────
    # Scholar is searched for all TOP_N_PRE_FILTER candidates (same as OA).
    logger.info("[MatchingAgent] Stage D: scholar acquisition for %d candidates", len(enriched))
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "D", "label": "Scholar Acquisition",
                     "ts": time.time()})

    scholar_candidates = [e for e in enriched if not e.get("_plan_skip_scholar")]
    scholar_skipped_planner = [e for e in enriched if e.get("_plan_skip_scholar")]
    scholar_searched_count = len(scholar_candidates)

    for e in scholar_skipped_planner:
        e["scholar_source"]           = "skipped_flag"
        e["scholar_match_confidence"] = ""

    if progress_cb:
        progress_cb({"type": "progress", "step": "scholar",
                     "message": f"Acquiring Scholar profiles for {len(scholar_candidates)} candidates…"})

    _scholar_done  = _threading.Lock()
    _scholar_count = [0]

    # Part 1: URL discovery — hits DDG/personal-sites, safe at 10 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=SCHOLAR_URL_WORKERS) as pool:
        list(pool.map(_acquire_scholar_url, scholar_candidates))

    # Part 2: pub scraping — hits scholar.google.com, hard cap at 5 workers
    def _fetch_pubs_cb(e: dict) -> None:
        _acquire_scholar_pubs(e)
        with _scholar_done:
            _scholar_count[0] += 1
            n = _scholar_count[0]
        if progress_cb:
            progress_cb({"type": "batch_progress", "step": "scholar",
                         "current": n, "total": len(scholar_candidates),
                         "message": f"Scholar {n}/{len(scholar_candidates)}: {e.get('name', '')}…"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=SCHOLAR_PUB_WORKERS) as pool:
        list(pool.map(_fetch_pubs_cb, scholar_candidates))

    found_scholar = sum(1 for e in enriched if e.get("google_scholar"))
    print(
        f"  [D] Scholar: {found_scholar}/{len(enriched)} found  "
        f"({len(scholar_skipped_planner)} planner-skipped)"
    )
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "D", "label": "Scholar Acquisition",
                     "ts": time.time(), "searched": scholar_searched_count,
                     "found": found_scholar, "planner_skipped": len(scholar_skipped_planner),
                     "outside_top_n": 0})
    for e in enriched:
        if e.get("google_scholar"):
            print(
                f"       {e['name']:<28}  source={e.get('scholar_source')}  "
                f"conf={e.get('scholar_match_confidence')}  {e['google_scholar'][:60]}"
            )

    # ── Stage E: ORIGINAL scoring pipeline ───────────────────────────────────
    logger.info("[MatchingAgent] Stage E: original LLM scoring pipeline")
    if progress_cb:
        progress_cb({"type": "stage_start", "stage": "E", "label": "LLM Scoring",
                     "ts": time.time(), "in": len(enriched)})
    final = _run_original_scoring(
        client=client,
        cv_text=cv_text,
        enriched_profiles=enriched,
        output_path=output_path,
        dept_name=dept_name,
        source_url=source_url,
        progress_cb=progress_cb,
        cv_summary_en=cv_summary_en,
    )

    # ── Build MatchingOutcome with full frontend schema (concurrent) ─────────
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "E", "label": "LLM Scoring",
                     "ts": time.time(), "out": len(final)})
        progress_cb({"type": "stage_start", "stage": "F", "label": "Recommendation",
                     "ts": time.time(), "in": len(final)})
        progress_cb({"type": "progress", "step": "recommend",
                     "message": f"Generating recommendations for {len(final)} candidates…"})

    # Pre-assign rank (sort order already fixed) so threads can emit correct rank
    for rank, p in enumerate(final, 1):
        p["_rank"] = rank

    top_results_unsorted: list[dict] = []
    _res_lock = _threading.Lock()

    def _build_one(p: dict) -> dict:
        ps           = p.get("profile_scoring") or {}
        ss           = p.get("scholar_scoring")
        scholar_data = p.get("scholar_data") or {
            "fetch_status": "no_url", "publications": [], "pub_count": 0}
        cv_content      = p.get("cv_content", "")
        website_content = p.get("website_content", "")
        gs_url          = p.get("google_scholar", "")
        scholar_conf    = p.get("scholar_match_confidence", "")
        scholar_src     = p.get("scholar_source", "profile")
        rank            = p["_rank"]

        rec     = _generate_recommendation(client, cv_text, p, ps, ss, scholar_data,
                                           cv_summary_en=cv_summary_en)
        signals = _extract_signals(p, scholar_data, cv_content, website_content)
        result  = _build_result(
            name=p["name"], profile_url=p.get("profile_url", ""), gs_url=gs_url,
            profile=p, profile_sc=ps, scholar_sc=ss, scholar_data=scholar_data,
            rec=rec, signals=signals, scholar_source=scholar_src,
            scholar_confidence=scholar_conf,
        )
        result["rank"] = rank
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=RECOMMEND_WORKERS) as pool:
        futs = {pool.submit(_build_one, p): p for p in final}
        for fut in concurrent.futures.as_completed(futs):
            try:
                result = fut.result()
            except Exception as exc:
                p = futs[fut]
                tb_str = traceback.format_exc()
                logger.warning("[MatchingAgent] Recommendation error for %s: %s\n%s",
                               p.get("name"), exc, tb_str)
                warnings.append(f"Recommendation failed for {p.get('name','?')}: {exc}")
                if progress_cb:
                    progress_cb({"type": "error", "stage": "F",
                                 "message": f"Recommendation failed for {p.get('name','?')}: {exc}",
                                 "traceback": tb_str})
                # Build minimal fallback so this professor stays in the ranking
                try:
                    ps_fb   = p.get("profile_scoring") or {}
                    ss_fb   = p.get("scholar_scoring")
                    sd_fb   = p.get("scholar_data") or {
                        "fetch_status": "no_url", "publications": [], "pub_count": 0}
                    sigs_fb = _extract_signals(
                        p, sd_fb, p.get("cv_content",""), p.get("website_content",""))
                    rec_fb  = {"match_reason": "[recommendation error]",
                               "cold_email": {}, "refined_interests": ""}
                    result  = _build_result(
                        name=p["name"], profile_url=p.get("profile_url",""),
                        gs_url=p.get("google_scholar",""),
                        profile=p, profile_sc=ps_fb, scholar_sc=ss_fb,
                        scholar_data=sd_fb, rec=rec_fb, signals=sigs_fb,
                        scholar_source=p.get("scholar_source",""),
                        scholar_confidence=p.get("scholar_match_confidence",""),
                    )
                    result["rank"] = p["_rank"]
                    with _res_lock:
                        top_results_unsorted.append(result)
                except Exception as exc2:
                    logger.error("[MatchingAgent] Fallback also failed for %s: %s",
                                 p.get("name"), exc2)
                continue
            with _res_lock:
                top_results_unsorted.append(result)

    top_results = sorted(top_results_unsorted, key=lambda r: r["rank"])
    if progress_cb:
        progress_cb({"type": "stage_end", "stage": "F", "label": "Recommendation",
                     "ts": time.time(), "out": len(top_results)})

    return MatchingOutcome(
        source_url=source_url,
        total_candidates=total_candidates,
        after_early_exclusion=len(kept),
        enriched_count=valid_count,
        scholar_searched=scholar_searched_count,
        excluded=[
            {"name": e["name"], "reason": e["early_exclusion_reason"]}
            for e in excluded_list
        ],
        warnings=warnings,
        top_results=top_results,
    )


# ---------------------------------------------------------------------------
# Recommendation generation & signal extraction (frontend output helpers)
# ---------------------------------------------------------------------------

RECOMMENDATION_SYSTEM = """\
You are a PhD application advisor. Output ONLY a raw JSON object. No markdown, no preamble.

STRICT RULES — follow exactly:
1. ENGLISH ONLY. No Chinese, Korean, Japanese, or other non-Latin scripts anywhere.
2. Keep every field SHORT: match_reason ≤ 20 words, each cold_email field ≤ 15 words.
3. PERSONALIZATION — this is the most important rule:
   - Read the faculty's research focus, publications, and bio FIRST.
   - Then ask: "Which specific CV project/skill is closest to WHAT THIS PROFESSOR actually studies?"
   - highlight_experience must cite that specific match — NOT the most impressive CV line overall.
   - entry_point must reference a real paper title or specific topic from THIS professor's work.
   - Do NOT reuse the same highlight for different professors. Each must be uniquely tailored.
4. Generic phrases like "my ML background", "my interpretable ML expertise", or any experience
   that would apply equally to every professor are FORBIDDEN in cold_email fields.

{
  "refined_interests": "<3–5 English keywords/phrases matching CV to THIS faculty's work>",
  "match_reason": "<1 sentence, ≤20 words: the single most specific shared research theme>",
  "cold_email": {
    "entry_point": "<1 sentence, ≤15 words: open with one specific paper title or named project from THIS professor's recent publications>",
    "highlight_experience": "<1 sentence, ≤15 words: the ONE CV project or result most relevant to THIS professor's stated research focus — not the most impressive, the most relevant>",
    "convincing_point": "<1 sentence, ≤15 words: one concrete, specific reason this student fits THIS lab — avoid generic claims>"
  }
}

Only cite information present in the provided texts."""


def _generate_recommendation(
    client: anthropic.Anthropic,
    cv_text: str,
    profile: dict,
    profile_sc: dict,
    scholar_sc: dict | None,
    scholar_data: dict,
    cv_summary_en: str = "",
) -> dict:
    # Publication evidence: Scholar first, OpenAlex as fallback
    pubs = scholar_data.get("publications", [])
    if not pubs:
        pubs = [
            {"year": w["year"], "title": w["title"]}
            for w in (profile.get("openalex_works") or [])[:10]
        ]
    pub_lines     = "\n".join(f"  [{p.get('year','?')}] {p.get('title','')}"
                              for p in pubs[:12])
    scholar_block = f"\nRecent publications:\n{pub_lines}" if pubs else ""

    # OpenAlex structured topics
    oa_topics = profile.get("openalex_topics") or []
    oa_block  = (f"\nOpenAlex research topics: {'; '.join(oa_topics)}" if oa_topics else "")

    # Non-English CV: prepend English summary for clearer context
    cv_block = f"CV:\n{cv_text[:2500]}"
    if cv_summary_en:
        cv_block = f"Applicant profile (English): {cv_summary_en}\n\n" + cv_block

    ps_summary = (
        f"Research {profile_sc.get('research_match',{}).get('score','?')}/10, "
        f"Method {profile_sc.get('method_match',{}).get('score','?')}/10, "
        f"App {profile_sc.get('application_match',{}).get('score','?')}/10"
    )
    ss_summary = ""
    if scholar_sc and "recent_scholar_match" in scholar_sc:
        rm = scholar_sc["recent_scholar_match"]
        ss_summary = (
            f"\nScholar match {rm.get('score','?')}/10. "
            f"Best paper: {rm.get('strongest_overlap_paper','')[:100]}"
        )

    extra = ""
    _cv_raw = profile.get("cv_content") or ""
    if _cv_raw and not _cv_raw.startswith("%PDF"):
        extra += f"\nFaculty CV (excerpt):\n{_cv_raw[:800]}"
    if profile.get("website_content"):
        extra += f"\nFaculty website (excerpt):\n{profile['website_content'][:500]}"
    # Faculty info block — placed FIRST so the LLM anchors on this professor's
    # specific focus before reading the CV, preventing it from defaulting to the
    # most salient CV project regardless of relevance.
    faculty_block = (
        f"=== FACULTY: {profile.get('name','')} ===\n"
        f"Research focus: {'; '.join(oa_topics[:4]) if oa_topics else profile.get('research_interests','')[:200]}\n"
        f"Bio: {profile.get('bio','')[:350]}\n"
        f"Research interests: {profile.get('research_interests','')[:300]}"
        f"{oa_block}"
        f"{scholar_block}"
        f"{extra}"
    )
    # Explicit per-professor anchor at the end to prevent generic highlights
    focus_anchor = (
        f"\n\nThis professor's primary research focus is: "
        f"{'; '.join(oa_topics[:2]) or profile.get('research_interests','')[:120]}\n"
        f"Choose highlight_experience from the CV that matches THIS focus specifically."
    )
    payload = dict(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=RECOMMENDATION_SYSTEM,
        messages=[{"role": "user", "content": (
            f"{faculty_block}\n\n"
            f"=== APPLICANT CV ===\n"
            f"{cv_block}\n\n"
            f"Scores: {ps_summary}{ss_summary}"
            f"{focus_anchor}"
        )}],
    )
    for attempt in range(3):
        try:
            msg = client.messages.create(**payload)
            return _extract_json(msg.content[0].text)
        except anthropic.RateLimitError:
            if attempt == 2:
                raise
            wait = 20 * (attempt + 1)   # 20s → 40s
            logger.warning(
                "[Stage F] 429 rate limit for %s — waiting %ds (attempt %d/3)",
                profile.get("name", "?"), wait, attempt + 1,
            )
            time.sleep(wait)


_FUNDING_STRONG = re.compile(
    r'\b(NSF|NIH|DARPA|DOE|ONR|ARO|AFOSR|DOD|NRC|NSERC|ERC|Wellcome)\b'
    r'|(?:awarded|received|funded by|supported by|grant(?:ed)?)\s+\$?\d',
    re.IGNORECASE,
)
_FUNDING_MEDIUM = re.compile(
    r'\b(grant|funding|award|fellowship|contract|sponsored)\b',
    re.IGNORECASE,
)
_RECRUITING_OPEN = re.compile(
    r'(?:looking for|seeking|recruiting|openings? for|positions? (?:available|open)|'
    r'join (?:my|our) (?:lab|group|team)|'
    r'prospective (?:students?|phd|graduate)|'
    r'phd (?:openings?|positions?)|'
    r'accepting (?:students?|applications?))',
    re.IGNORECASE,
)
_ACTIVITY_YEAR_RE = re.compile(r'\b(202[3-9]|2030)\b')


def _extract_signals(profile: dict, scholar_data: dict,
                     cv_content: str, website_content: str) -> dict:
    """
    Pure keyword/regex signal detection. Returns:
      funding:   "strong" | "medium" | "unknown"
      recruiting: "likely_open" | "unclear"
      activity:  "high" | "medium" | "low" | "unknown"
    """
    all_text = " ".join(filter(None, [
        profile.get("bio", ""),
        profile.get("research_interests", ""),
        cv_content,
        website_content,
    ]))

    if _FUNDING_STRONG.search(all_text):
        funding = "strong"
    elif _FUNDING_MEDIUM.search(all_text):
        funding = "medium"
    else:
        funding = "unknown"

    if _RECRUITING_OPEN.search(all_text):
        recruiting = "likely_open"
    else:
        recruiting = "unclear"

    pubs = scholar_data.get("publications", [])

    # OpenAlex counts_by_year is the most reliable activity signal
    oa_stats = profile.get("openalex_stats") or {}
    counts_by_year = oa_stats.get("counts_by_year") or []
    recent_oa_set  = {2023, 2024, 2025}
    oa_recent_works = sum(
        y.get("works_count", 0) for y in counts_by_year
        if y.get("year") in recent_oa_set
    )
    if counts_by_year:
        if oa_recent_works >= 5:
            activity = "high"
        elif oa_recent_works >= 2:
            activity = "medium"
        else:
            activity = "low"
    else:
        # Fallback: Scholar pub years
        recent_years = {2023, 2024, 2025, 2026}
        recent_count = sum(
            1 for p in pubs
            if p.get("year", "").strip().isdigit()
            and int(p["year"]) in recent_years
        )
        if not pubs:
            year_hits = len(_ACTIVITY_YEAR_RE.findall(all_text))
            activity = "medium" if year_hits >= 3 else "unknown"
        elif recent_count >= 3:
            activity = "high"
        elif recent_count >= 1:
            activity = "medium"
        else:
            activity = "low"

    return {"funding": funding, "recruiting": recruiting, "activity": activity}


def _build_result(name: str, profile_url: str, gs_url: str,
                  profile: dict, profile_sc: dict,
                  scholar_sc: dict | None, scholar_data: dict, rec: dict,
                  signals: dict | None = None,
                  scholar_source: str = "profile",
                  scholar_confidence: str = "") -> dict:
    overall = _overall_match(profile_sc, scholar_sc, scholar_confidence)
    has_scholar = bool(scholar_sc and "recent_scholar_match" in scholar_sc)
    scholar_w = _scholar_weight(scholar_confidence) if has_scholar else 0.0
    logger.info(
        "[build_result] %s | scholar_url=%s | scholar_source=%s | scholar_conf=%s"
        " | overall=%.2f (scholar_w=%.0f%%)",
        name, gs_url or "NONE", scholar_source, scholar_confidence or "direct",
        overall, scholar_w * 100,
    )
    oa_stats = profile.get("openalex_stats") or {}
    return {
        "name": name,
        "profile_url": profile_url,
        "scholar_url": gs_url or None,
        "scholar_source": scholar_source,
        "scholar_confidence": scholar_confidence,
        "scholar_match_confidence": scholar_confidence,
        "cv_url": profile.get("cv_url") or None,
        "website_url": profile.get("personal_website") or None,
        "website_confidence": profile.get("personal_website_confidence") or "",
        # OpenAlex fields
        "openalex_url":        profile.get("openalex_url") or None,
        "openalex_id":         profile.get("openalex_id") or None,
        "orcid":               profile.get("orcid") or None,
        "openalex_topics":     profile.get("openalex_topics") or [],
        "openalex_confidence": profile.get("openalex_confidence") or "",
        "openalex_stats": {
            "h_index":        oa_stats.get("h_index"),
            "cited_by_count": oa_stats.get("cited_by_count", 0),
            "works_count":    oa_stats.get("works_count", 0),
        },
        "refined_interests": rec.get("refined_interests", ""),
        "overall_match": overall,
        "match_reason": rec.get("match_reason", ""),
        "cold_email": rec.get("cold_email", {}),
        "scoring": {
            "faculty_profile_match": profile_sc.get("faculty_profile_match"),
            "recent_scholar_match": (
                scholar_sc["recent_scholar_match"]["score"]
                if has_scholar else None
            ),
            "scholar_weight": scholar_w,
            "scholar_confidence": scholar_confidence,
            "overall_match": overall,
            "dimensions": {
                "research":    profile_sc.get("research_match", {}).get("score"),
                "method":      profile_sc.get("method_match", {}).get("score"),
                "application": profile_sc.get("application_match", {}).get("score"),
                "style":       profile_sc.get("style_match", {}).get("score"),
            },
        },
        "scholar_pubs": scholar_data.get("pub_count", 0),
        "scholar_fetch_status": scholar_data.get("fetch_status"),
        "signals": signals or {"funding": "unknown", "recruiting": "unclear", "activity": "unknown"},
        "bio": profile.get("bio", "")[:300],
        "groups": profile.get("groups", []),
    }
