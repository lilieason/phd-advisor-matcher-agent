"""
Web Extraction Agent
====================

Minimal Planner → Executor → Validator loop for faculty list extraction.

Architecture
------------
  Planner   : inspects page signals + Kahuna memory → ordered strategy list
  Executor  : runs a named strategy → ExtractionResult (structured, never silent)
  Validator : scores result quality with concrete rules → ValidationResult
  Loop      : up to MAX_ATTEMPTS; planner picks fallback on validator failure
  Memory    : _kahuna_load_context before, _kahuna_record_outcome after

Strategy names (executor actions)
----------------------------------
  usc_extractor        — USC-style ?lname=X&fname=Y query-param links
  cbs_extractor        — Columbia Business School m-listing-faculty cards
  card_extractor       — BFS card detection inside a faculty section
  list_extractor       — link-walk inside a faculty section (bypasses cards)
  grouped_extractor    — research-topic-grouped faculty (_extract_grouped_faculty)
  photo_extractor      — Columbia/Drupal "Photo of X" img-alt pattern
  name_alt_extractor   — Georgia Tech style: img alt IS the person name
  interactive_resolver — follows "View Faculty" hrefs to sub-pages
  generic_extractor    — full-page deep profile-link walk (S2 fallback)
  llm_extractor        — Haiku LLM fallback for novel/unrecognized page layouts
                         (fires automatically when name_proportion < 0.60)
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3   # planner + at most 2 fallbacks

# Role/category words that appear as the FIRST word of exactly-2-word phrases
# that look like person names but aren't ("Postdoctoral Scholars", "Student Affairs").
# Applied in generic_extractor and validator.  Does NOT fire for 1-word or 3+-word entries.
_ROLE_PREFIXES: frozenset[str] = frozenset({
    # Role / rank words
    "postdoctoral", "student", "graduate", "visiting", "adjunct",
    "emeritus", "emerita", "staff", "faculty", "research", "undergraduate",
    "doctoral", "phd", "professorial", "honorary", "distinguished",
    # Temporal / category section-header words (e.g. "Previous appointments",
    # "Current projects", "Former members", "Recent graduates")
    "previous", "current", "former", "recent", "incoming", "outgoing",
    "affiliate", "affiliated", "joint", "courtesy", "endowed", "named",
    "new", "past",
})

# ---------------------------------------------------------------------------
# Structured types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionPlan:
    page_representation: str           # card|list|grouped|interactive|mixed|unknown
    strategy_order: list[str]          # strategies to try, in priority order
    reasoning: str                     # one-liner explaining the choice
    mem_hint: dict = field(default_factory=dict)  # from Kahuna


@dataclass
class ExtractionResult:
    strategy_used: str
    page_representation: str
    faculty_list: list[dict]
    faculty_count: int
    raw_signals: dict                  # cards_detected, section_type, etc.


@dataclass
class ValidationResult:
    validator_score: float             # 0.0 – 1.0
    success: bool
    issues: list[str]
    failure_reason: str | None
    name_proportion: float
    topic_leakage_rate: float
    pub_leakage_rate: float


@dataclass
class ExtractionOutcome:
    """
    Unified schema for one extraction run.  Stored in Kahuna and returned
    to the caller (via to_legacy_format() for backward compatibility).
    """
    url: str
    domain: str
    page_representation: str
    strategy_used: str
    faculty_count: int
    faculty_names_sample: list[str]
    validator_score: float
    issues: list[str]
    success: bool
    failure_reason: str | None
    next_best_strategy: str | None
    strategy_trace: list[dict]         # [{strategy, faculty_count, validator_score, issues}]
    timestamp: str

    # The actual faculty list — not stored in Kahuna but returned to callers
    faculty_list: list[dict] = field(default_factory=list, repr=False)
    # Raw error dict if the page could not be scraped at all
    error_payload: dict | None = field(default=None, repr=False)
    # Hidden-content detection results
    hidden_content_detected: bool = False
    requires_browser_interaction: bool = False
    hidden_content_warning: str | None = None

    def to_legacy_format(self) -> "list[dict] | dict":
        """
        Convert to the format expected by _fetch_faculty_list_sync callers:
          - success / partial: list[dict]
          - interactive JS-blocked: {"page_type": ..., "groups": [...], "error": ...}
          - hard error: {"error": ...}
        """
        if self.error_payload is not None:
            return self.error_payload
        return self.faculty_list


# ---------------------------------------------------------------------------
# Lazy imports from advisor_server (avoids circular import at module load)
# ---------------------------------------------------------------------------

def _srv():
    """Return the advisor_server module (imported lazily)."""
    import mcp_servers.advisor_server as _m
    return _m


# ---------------------------------------------------------------------------
# Kahuna memory (richer schema than the earlier simple version)
# ---------------------------------------------------------------------------

_KAHUNA_EXTRACT_DIR = Path.home() / ".kahuna" / "knowledge" / "extractions"


def _kahuna_load_context(url: str) -> dict:
    """Load past extraction outcomes for this URL's domain, newest first."""
    domain = urlparse(url).netloc.replace(":", "_").replace(".", "-")
    _KAHUNA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict] = []
    for f in _KAHUNA_EXTRACT_DIR.glob(f"{domain}_*.json"):
        try:
            outcomes.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not outcomes:
        return {}
    outcomes.sort(key=lambda o: o.get("timestamp", ""), reverse=True)
    recent = outcomes[:5]
    successes = [o for o in recent if o.get("success")]
    failed_strategies = list({o["strategy_used"] for o in recent if not o.get("success")})
    return {
        "domain": domain,
        "past_outcomes": recent,
        "last_strategy": recent[0].get("strategy_used"),
        "last_success": recent[0].get("success", False),
        "best_strategy": successes[0].get("strategy_used") if successes else None,
        "failed_strategies": failed_strategies,
    }


def _kahuna_record_outcome(outcome: ExtractionOutcome) -> None:
    """Persist extraction outcome to Kahuna knowledge base."""
    _KAHUNA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "url": outcome.url,
        "domain": outcome.domain,
        "page_representation": outcome.page_representation,
        "strategy_used": outcome.strategy_used,
        "faculty_count": outcome.faculty_count,
        "faculty_names_sample": outcome.faculty_names_sample,
        "validator_score": outcome.validator_score,
        "issues": outcome.issues,
        "success": outcome.success,
        "failure_reason": outcome.failure_reason,
        "next_best_strategy": outcome.next_best_strategy,
        "strategy_trace": outcome.strategy_trace,
        "timestamp": outcome.timestamp,
    }
    fname = _KAHUNA_EXTRACT_DIR / f"{outcome.domain}_{outcome.timestamp}.json"
    try:
        fname.write_text(json.dumps(record, indent=2))
        logger.info(
            "[Kahuna] Recorded outcome: domain=%s strategy=%s faculty=%d score=%.2f success=%s",
            outcome.domain, outcome.strategy_used, outcome.faculty_count,
            outcome.validator_score, outcome.success,
        )
    except Exception as exc:
        logger.warning("[Kahuna] Failed to write outcome: %s", exc)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

# Map section_type → (page_representation, default_strategy_order)
# llm_extractor leads for static pages; interactive_resolver leads for JS-driven pages.
_SECTION_TYPE_MAP: dict[str, tuple[str, list[str]]] = {
    "faculty_section":                    ("list",        ["llm_extractor", "card_extractor", "list_extractor", "generic_extractor"]),
    "grouped_faculty_section":            ("grouped",     ["llm_extractor", "grouped_extractor", "list_extractor", "generic_extractor"]),
    "mixed_section":                      ("mixed",       ["llm_extractor", "grouped_extractor", "card_extractor", "generic_extractor"]),
    "interactive_grouped_faculty_section":("interactive", ["interactive_resolver", "llm_extractor", "generic_extractor"]),
    "publication_section":                ("unknown",     ["llm_extractor", "generic_extractor"]),
}


def _plan(
    url: str,
    soup: BeautifulSoup,
    page_class: str,
    mem_ctx: dict,
) -> ExtractionPlan:
    """
    Inspect page signals and Kahuna memory to choose an ordered strategy list.

    Decision tree (first matching rule wins):
      1. Faculty section detected → classify section type → map to strategy order
      2. Interactive / modal page (no section) → llm first, then interactive_resolver
      3. Default → llm first, rule-based as fallback

    LLM extractor leads every order. Rule-based extractors are fallbacks.
    USC query-param pages are no longer special-cased — llm_extractor handles them.

    Memory adjustments (after rule selection):
      - Known-failed strategies are pushed to end of order
      - Known-best strategy is moved to front (except interactive_resolver lock)
    """
    srv = _srv()

    # ── Rule 1: Faculty section detected ─────────────────────────────────────
    section = srv._find_faculty_section(soup)
    if section:
        heading_el = section.find(["h2", "h3", "h4", "strong", "b"])
        heading_text = heading_el.get_text(strip=True) if heading_el else ""
        section_type = srv._classify_section_type(section, heading_text)
        page_rep, order = _SECTION_TYPE_MAP.get(
            section_type, ("list", ["llm_extractor", "card_extractor", "list_extractor", "generic_extractor"])
        )
        reasoning = f"section_type={section_type}"
    # ── Rule 2: Interactive / modal page ─────────────────────────────────────
    # interactive_resolver leads: LLM only sees the visible stub, not JS-loaded content.
    elif (page_class == "interactive_grouped_faculty_page"
          or srv._detect_interactive_faculty_page(soup)
          or srv._has_modal_faculty_containers(soup)):
        page_rep, order = "interactive", ["interactive_resolver", "llm_extractor", "generic_extractor"]
        reasoning = "interactive/modal signals — interactive_resolver first, llm_extractor fallback"
    # ── Rule 3: Default — LLM first, rule-based fallback ─────────────────────
    else:
        page_rep, order = "list", ["llm_extractor", "card_extractor", "cbs_extractor", "generic_extractor"]
        reasoning = "llm_extractor (generalist); rule-based fallback"

    # ── Memory adjustments ────────────────────────────────────────────────────
    failed = set(mem_ctx.get("failed_strategies", []))
    best = mem_ctx.get("best_strategy")

    # Push previously-failed strategies to the end
    order = [s for s in order if s not in failed] + [s for s in order if s in failed]

    # Promote best-known strategy to front.
    # Lock interactive_resolver in place — JS-only pages must not swap to LLM.
    _SIGNAL_LOCKED = {"interactive_resolver"}
    kahuna_lock = order and order[0] in _SIGNAL_LOCKED
    if best and best in order and order[0] != best and not kahuna_lock:
        order.remove(best)
        order.insert(0, best)
        reasoning += f"; Kahuna promoted {best} (past success)"

    if mem_ctx:
        logger.info(
            "[Planner] domain=%s last_strategy=%s last_success=%s best=%s",
            mem_ctx.get("domain", "?"), mem_ctx.get("last_strategy"),
            mem_ctx.get("last_success"), best,
        )

    return ExtractionPlan(
        page_representation=page_rep,
        strategy_order=order,
        reasoning=reasoning,
        mem_hint=mem_ctx,
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _execute(
    strategy: str,
    url: str,
    soup: BeautifulSoup,
    page_class: str,
) -> ExtractionResult:
    """
    Dispatch to the named extraction strategy.
    Always returns an ExtractionResult — never raises, never returns None.
    """
    srv = _srv()
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    all_links = soup.find_all("a", href=True)
    raw_signals: dict = {"strategy": strategy}

    # ── photo_extractor (Columbia/Drupal "Photo of X" img alt pattern) ──────
    if strategy == "photo_extractor":
        faculty = srv._extract_photo_alt_faculty(soup, url)
        for e in faculty:
            e.setdefault("_page_class", page_class)
            e.setdefault("groups", [])
        raw_signals["photo_images_found"] = len(faculty)
        return ExtractionResult(
            strategy_used=strategy,
            page_representation="list",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── name_alt_extractor (Georgia Tech style: alt text IS the name) ────────
    if strategy == "name_alt_extractor":
        faculty = srv._extract_name_alt_faculty(soup, url)
        for e in faculty:
            e.setdefault("_page_class", page_class)
            e.setdefault("groups", [])
        raw_signals["name_alt_images_found"] = len(faculty)
        return ExtractionResult(
            strategy_used=strategy,
            page_representation="list",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── cbs_extractor (Columbia Business School m-listing-faculty cards) ─────
    if strategy == "cbs_extractor":
        faculty = srv._extract_cbs_faculty(soup, url)
        for e in faculty:
            e.setdefault("_page_class", page_class)
            e.setdefault("groups", [])
        raw_signals["cbs_cards_found"] = len(faculty)
        return ExtractionResult(
            strategy_used=strategy,
            page_representation="list",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── llm_extractor (LLM-based fallback for novel page structures) ─────────
    if strategy == "llm_extractor":
        faculty = srv._extract_llm_faculty(soup, url)
        for e in faculty:
            e.setdefault("_page_class", page_class)
            e.setdefault("groups", [])
        raw_signals["llm_entries_found"] = len(faculty)
        return ExtractionResult(
            strategy_used=strategy,
            page_representation="list",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── usc_extractor ─────────────────────────────────────────────────────────
    if strategy == "usc_extractor":
        usc_links = [a for a in all_links
                     if "lname=" in a.get("href", "") and "fname=" in a.get("href", "")]
        faculty = []
        seen: set[str] = set()
        for a in usc_links:
            href = a["href"]
            full_url = urljoin(base, href) if not href.startswith("http") else href
            if full_url in seen:
                continue
            seen.add(full_url)
            # Prefer name from query params (lname/fname) — clean, no "Part-Time" noise.
            # Fall back to link text only if URL parsing yields nothing.
            name = srv._name_from_url(href)
            if not name:
                raw_text = a.get_text(separator=" ", strip=True)
                name, _ = srv._split_name_from_title(raw_text)
            faculty.append({
                "name": name or href,
                "profile_url": href,
                "full_profile_url": full_url,
                "_page_class": page_class,
                "groups": [],
            })
        raw_signals["usc_link_count"] = len(usc_links)
        return ExtractionResult(
            strategy_used=strategy,
            page_representation="usc_query",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── card_extractor / list_extractor / grouped_extractor ──────────────────
    if strategy in ("card_extractor", "list_extractor", "grouped_extractor"):
        # Multi-section aggregation: find ALL matching sections and merge.
        all_sections = srv._find_all_person_sections(soup)
        raw_signals["section_found"] = len(all_sections) > 0
        raw_signals["sections_detected"] = len(all_sections)

        if not all_sections:
            return ExtractionResult(
                strategy_used=strategy,
                page_representation="unknown",
                faculty_list=[],
                faculty_count=0,
                raw_signals=raw_signals,
            )

        # Use the first section to determine section_type for page_rep
        first_heading, first_section = all_sections[0]
        heading_el = first_section.find(["h2", "h3", "h4", "strong", "b"])
        heading_text = heading_el.get_text(strip=True) if heading_el else first_heading
        section_type = srv._classify_section_type(first_section, heading_text)
        raw_signals["section_type"] = section_type

        faculty: list[dict] = []
        seen_urls: set[str] = set()
        seen_names: set[str] = set()

        for sec_heading, section in all_sections:
            if strategy == "grouped_extractor":
                sec_entries = srv._extract_grouped_faculty(section, url)
                raw_signals["grouped_headings_found"] = (
                    raw_signals.get("grouped_headings_found", False) or len(sec_entries) > 0
                )
                page_rep = "grouped"

            elif strategy == "card_extractor":
                cards = srv._detect_faculty_cards(section)
                if cards:
                    _link_counts = [len(c.find_all("a", href=True)) for c in cards]
                    _nonempty = [n for n in _link_counts if n > 0]
                    _avg = sum(_nonempty) / len(_nonempty) if _nonempty else 0
                    if _avg > 2.0:
                        logger.info(
                            "[card_extractor] avg_links=%.1f > 2 — columnar layout, "
                            "skipping card extraction (will fall through to list_extractor)",
                            _avg,
                        )
                        cards = []

                # Use a per-section seen set for _extract_card_entry (intra-section dedup).
                # Cross-section dedup is handled below via seen_urls/seen_names.
                _section_seen: set[str] = set()
                sec_entries = []
                for card in cards:
                    entry = srv._extract_card_entry(card, base, _section_seen)
                    if entry is None:
                        continue
                    if entry.get("full_profile_url"):
                        _section_seen.add(entry["full_profile_url"])
                    entry.setdefault("groups", [])
                    sec_entries.append(entry)
                page_rep = "card" if sec_entries else section_type_to_rep(section_type)

            else:  # list_extractor
                sec_entries = srv._extract_links_from_section(section, url)
                raw_signals["cards_detected"] = None
                page_rep = section_type_to_rep(section_type)

            # Deduplicate across sections: prefer profile-URL key, fallback to name
            for e in sec_entries:
                furl = e.get("full_profile_url", "")
                name_key = e.get("name", "").lower().strip()
                if furl and furl in seen_urls:
                    continue
                if not furl and name_key and name_key in seen_names:
                    continue
                if furl:
                    seen_urls.add(furl)
                if name_key:
                    seen_names.add(name_key)
                e.setdefault("_page_class", page_class)
                e.setdefault("groups", [])
                faculty.append(e)

            logger.info(
                "[%s] Section %r → %d entries (running total %d)",
                strategy, sec_heading, len(sec_entries), len(faculty),
            )

        raw_signals["cards_detected"] = raw_signals.get("cards_detected", len(faculty))

        return ExtractionResult(
            strategy_used=strategy,
            page_representation=page_rep,
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # ── interactive_resolver ─────────────────────────────────────────────────
    if strategy == "interactive_resolver":
        interactive = srv._detect_interactive_faculty_page(soup)
        raw_signals["triggers_found"] = len(interactive["groups"]) if interactive else 0

        if not interactive:
            # No triggers found via standard detection — still check for modal containers
            # (e.g. WordPress Spectra blocks where visible_names > 5 suppresses detection).
            modal_faculty = srv._extract_modal_faculty(soup, url)
            if modal_faculty:
                for e in modal_faculty:
                    e.setdefault("groups", e.get("groups", []))
                    e.setdefault("_page_class", page_class)
                raw_signals["modal_extraction"] = True
                logger.info(
                    "[interactive_resolver] Modal extraction (no-trigger branch): %d entries",
                    len(modal_faculty),
                )
                return ExtractionResult(
                    strategy_used=strategy,
                    page_representation="interactive",
                    faculty_list=modal_faculty,
                    faculty_count=len(modal_faculty),
                    raw_signals=raw_signals,
                )
            return ExtractionResult(
                strategy_used=strategy,
                page_representation="interactive",
                faculty_list=[],
                faculty_count=0,
                raw_signals=raw_signals | {"note": "no interactive triggers found"},
            )

        resolved = srv._resolve_interactive_faculty(
            interactive["groups"], url, page_class
        )
        raw_signals["sub_pages_resolved"] = len(resolved) > 0

        if not resolved:
            # All triggers have href="#" — content may be in the DOM (modal/accordion).
            # Try extracting faculty from modal containers in the current page soup.
            modal_faculty = srv._extract_modal_faculty(soup, url)
            if modal_faculty:
                for e in modal_faculty:
                    e.setdefault("groups", e.get("groups", []))
                    e.setdefault("_page_class", page_class)
                raw_signals["modal_extraction"] = True
                raw_signals["sub_pages_resolved"] = True
                logger.info(
                    "[interactive_resolver] Modal extraction succeeded: %d entries",
                    len(modal_faculty),
                )
                return ExtractionResult(
                    strategy_used=strategy,
                    page_representation="interactive",
                    faculty_list=modal_faculty,
                    faculty_count=len(modal_faculty),
                    raw_signals=raw_signals,
                )

            # Truly JS-only triggers — cannot scrape; return sentinel so loop can record it
            raw_signals["js_blocked"] = True
            return ExtractionResult(
                strategy_used=strategy,
                page_representation="interactive",
                faculty_list=[],
                faculty_count=0,
                raw_signals=raw_signals | {
                    "groups": interactive["groups"],
                    "error": (
                        "Faculty visible only after JavaScript interaction — "
                        "automatic extraction not possible"
                    ),
                },
            )

        for e in resolved:
            e.setdefault("groups", [])

        # Also try modal extraction and merge — some pages expose MORE faculty
        # in static modal HTML than sub-page resolution can reach.
        # Union by profile URL → name fallback; keep entries from both sources.
        modal_faculty = srv._extract_modal_faculty(soup, url)
        if modal_faculty:
            seen_merged: set[str] = {
                e.get("full_profile_url", "") or e.get("name", "").lower()
                for e in resolved
                if e.get("full_profile_url") or e.get("name")
            }
            for me in modal_faculty:
                furl = me.get("full_profile_url", "")
                nkey = me.get("name", "").lower()
                key = furl or nkey
                if key and key not in seen_merged:
                    seen_merged.add(key)
                    me.setdefault("groups", [])
                    me.setdefault("_page_class", page_class)
                    resolved.append(me)
            raw_signals["modal_merge"] = True
            logger.info(
                "[interactive_resolver] After modal merge: %d total entries", len(resolved)
            )

        return ExtractionResult(
            strategy_used=strategy,
            page_representation="interactive",
            faculty_list=resolved,
            faculty_count=len(resolved),
            raw_signals=raw_signals,
        )

    # ── generic_extractor (S2 full-page deep link walk) ──────────────────────
    if strategy == "generic_extractor":
        profile_pattern = re.compile(
            # faculty[^/]* catches /faculty/, /facultyfinder/, /faculty-directory/, etc.
            r"/(people|faculty[^/]*|staff|profile[^/]*|directory)/[^/?#]+", re.I
        )
        _SKIP_TEXT = {
            "faculty", "directory", "profile", "staff", "people",
            "home", "read more", "view profile", "more info", "learn more",
        }
        seen: set[str] = set()           # full profile URLs already added
        name_to_idx: dict[str, int] = {} # name_key → index in faculty (for URL merging)
        url_to_key: dict[str, str] = {}  # full_url → name_key (for section updates on URL repeats)
        faculty = []
        _src_host = urlparse(url).netloc

        for a in all_links:
            href = a.get("href", "")
            if not href:
                continue
            if any(x in href for x in ["mailto:", "javascript:", "#", "page=", "filter=", "category"]):
                continue

            # Compute full_url early (needed for cross-dept check and section updates)
            full_url = urljoin(base, href) if not href.startswith("http") else href
            if full_url.rstrip("/") == url.rstrip("/"):
                continue

            # Compute raw text early (needed for section updates on URL repeats)
            raw_text = a.get_text(separator=" ", strip=True)
            if not raw_text or len(raw_text) < 4:
                continue

            # URL already seen — opportunistically record new section for same person
            if full_url in seen:
                existing_key = url_to_key.get(full_url)
                if existing_key and existing_key in name_to_idx:
                    existing = faculty[name_to_idx[existing_key]]
                    source_section = _find_source_section(a)
                    if source_section:
                        existing_sections = {
                            sl.get("source_section", "")
                            for sl in existing.get("source_links", [])
                        }
                        if source_section not in existing_sections:
                            existing["source_links"].append({
                                "url": full_url,
                                "anchor_text": raw_text[:80],
                                "source_section": source_section,
                                "source_context": href,
                            })
                continue

            # Determine if URL is an explicit faculty/profile path
            is_profile_url = bool(profile_pattern.search(href))

            if raw_text.lower() in _SKIP_TEXT:
                continue
            name, _link_title = srv._split_name_from_title(raw_text)
            source_section = _find_source_section(a)
            words = name.split()
            _is_low_confidence = False  # reset per-iteration

            if not is_profile_url:
                # Secondary path: accept cross-department .edu links when the link
                # text is a valid person name (≥2 words) pointing to a different
                # university subdomain.  This handles UW-ISE/ME profiles linked
                # from CEE research pages, etc.
                _tgt_host = urlparse(full_url).netloc
                _tgt_path = urlparse(full_url).path
                if (len(words) >= 2
                        and _tgt_host.endswith(".edu")
                        and _tgt_host != _src_host
                        and _tgt_path.count("/") >= 2
                        and srv._is_valid_person_name(name)
                        and srv._classify_entry_type(name, full_url) == "faculty"):
                    pass  # fall through — treat as accepted person link
                else:
                    continue

            # Fix 3: single-word link text → try to recover full name from parent element
            if len(words) == 1:
                parent = a.parent
                if parent is not None:
                    parent_text = parent.get_text(separator=" ", strip=True)
                    parent_name, _ = srv._split_name_from_title(parent_text)
                    parent_words = parent_name.split()
                    if (len(parent_words) >= 2
                            and srv._is_valid_person_name(parent_name)
                            and words[0].lower() in parent_text.lower()):
                        name = parent_name
                        words = parent_words
                        logger.debug(
                            "[generic_extractor] Expanded single-word link %r → %r",
                            raw_text, name,
                        )
                if len(name.split()) < 2:
                    # Parent lookup failed.  Keep as low-confidence candidate only if the
                    # href itself is a clear individual faculty-profile URL — single surname
                    # links like <a href="/faculty/stoyanovich">Stoyanovich</a> are real people.
                    _profile_strict = re.compile(
                        r"/(faculty|people|profile|directory)/[^/?#]+$", re.I
                    )
                    if _profile_strict.search(href):
                        logger.debug(
                            "[generic_extractor] Keeping single-word %r as low-confidence "
                            "(profile URL confirmed: %s)", name, href,
                        )
                        _is_low_confidence = True
                        # Fall through — entry will be tagged _low_confidence=True below
                    else:
                        continue  # not a clear profile URL — discard
            elif len(words) < 2:
                continue

            # Fix 4: role-prefix blocklist — reject exactly-2-word role phrases
            # e.g. "Postdoctoral Scholars", "Student Affairs", "Graduate Program"
            if len(words) == 2 and words[0].lower() in _ROLE_PREFIXES:
                logger.debug("[generic_extractor] Role-prefix blocked: %r", name)
                continue

            if is_profile_url and srv._classify_entry_type(name, href) in ("publication", "tag_category", "group_topic"):
                # Fix 5: link text is a department profile label (e.g. "ISE profile",
                # "CEE profile", "ME profile") but the href IS a real faculty profile URL.
                # The person's name lives in a <b>/<strong> inside the same container block.
                # UW pattern:
                #   <tr><td><b>Cynthia Chen</b>
                #       <a href="...">CEE profile</a>
                #       <a href="...">ISE profile</a></td></tr>
                # Walk up the DOM through multiple container types (not only <tr>).
                _profile_strict2 = re.compile(
                    r"/(faculty[^/]*|people|profile[^/]*|directory)/[^/?#]+$", re.I
                )
                if not _profile_strict2.search(href):
                    continue

                _BLOCK_TAGS = {"td", "tr", "li", "article", "section", "div", "p"}
                _recovered_name: str | None = None
                node = a.parent
                for _ in range(8):
                    if node is None or not hasattr(node, "name"):
                        break
                    if node.name in _BLOCK_TAGS:
                        for bold in node.find_all(["b", "strong"]):
                            btext = bold.get_text(separator=" ", strip=True)
                            bname, _ = srv._split_name_from_title(btext)
                            if srv._is_valid_person_name(bname):
                                _recovered_name = bname
                                break
                        if _recovered_name:
                            break
                    node = node.parent if hasattr(node, "parent") else None

                if not _recovered_name:
                    continue  # no valid name found in any ancestor container

                name = _recovered_name
                words = name.split()
                logger.info(
                    "[generic_extractor] UW profile-label %r → recovered name %r"
                    " (container=%s, href=%s)",
                    raw_text, name, getattr(node, "name", "?"), href,
                )

            name_key = name.lower().strip()
            if name_key in name_to_idx:
                # Same person, different URL — merge instead of discard
                existing = faculty[name_to_idx[name_key]]
                existing.setdefault("profile_urls", [existing.get("full_profile_url", "")])
                if full_url not in existing["profile_urls"]:
                    existing["profile_urls"].append(full_url)
                    existing.setdefault("source_links", []).append({
                        "url": full_url,
                        "anchor_text": raw_text[:80],
                        "source_section": source_section,
                        "source_context": href,
                    })
                seen.add(full_url)
                url_to_key[full_url] = name_key
                continue
            seen.add(full_url)
            url_to_key[full_url] = name_key
            name_to_idx[name_key] = len(faculty)
            entry: dict = {
                "name": name,
                "profile_url": href,
                "full_profile_url": full_url,
                "profile_urls": [full_url] if full_url else [],
                "source_links": [{
                    "url": full_url,
                    "anchor_text": raw_text[:80],
                    "source_section": source_section,
                    "source_context": href,
                }],
                "_page_class": page_class,
                "groups": [],
                "raw_title": _link_title,
            }
            if _is_low_confidence:
                entry["_low_confidence"] = True
            faculty.append(entry)
        raw_signals["candidate_links_scanned"] = len(all_links)

        # ── External homepage verification pass ───────────────────────────────
        # Collect links whose anchor text looks like a person name but whose
        # href is external (not a university profile path).  Verify each one
        # by fetching and inspecting its content.  Cap at 10 per page.
        _EXT_CAP = 10
        ext_candidates: list[tuple[str, str, str]] = []  # (name, href, full_url)

        for a in all_links:
            if len(ext_candidates) >= _EXT_CAP:
                break
            href = a.get("href", "")
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue
            full_url = urljoin(base, href) if not href.startswith("http") else href
            # Skip if already accepted or it's the same page
            if full_url in seen or full_url.rstrip("/") == url.rstrip("/"):
                continue
            # Skip university-internal links (already handled by profile_pattern above)
            if profile_pattern.search(href):
                continue
            # Only verify links to external domains — same-domain links that
            # don't match profile_pattern are navigation/category pages, not
            # personal homepages.
            from urllib.parse import urlparse as _up
            src_host = _up(url).netloc
            tgt_host = _up(full_url).netloc
            if tgt_host == src_host or not tgt_host:
                continue
            # Skip links inside site navigation / header / footer
            if a.find_parent(["nav", "header", "footer"]):
                continue
            # Skip social / PDF / negative-URL patterns
            if srv._EXT_NEGATIVE_URL_RE.search(full_url):
                continue
            # Require link text that looks like a person name (not a topic/category)
            raw_text = a.get_text(separator=" ", strip=True)
            if not raw_text or len(raw_text) < 4:
                continue
            name, _ = srv._split_name_from_title(raw_text)
            if not srv._is_valid_person_name(name):
                continue
            if srv._classify_entry_type(name, full_url) != "faculty":
                continue
            name_key = name.lower().strip()
            if name_key in name_to_idx:
                continue
            # Avoid duplicate candidates for the same external URL
            if full_url in {c[2] for c in ext_candidates}:
                continue
            ext_candidates.append((name, href, full_url))

        raw_signals["ext_candidates"] = len(ext_candidates)
        ext_verified = 0

        # Derive institution hint from the source page's domain
        parsed_src = urlparse(url)
        institution_hint = parsed_src.netloc  # e.g. "datascience.ucsd.edu"

        for ext_name, ext_href, ext_full_url in ext_candidates:
            result = srv.verify_external_person_homepage(
                ext_full_url, ext_name, institution_hint
            )
            if result["is_person_homepage"]:
                ext_verified += 1
                ext_name_key = ext_name.lower().strip()
                seen.add(ext_full_url)
                name_to_idx[ext_name_key] = len(faculty)
                faculty.append({
                    "name": ext_name,
                    "profile_url": ext_href,
                    "full_profile_url": ext_full_url,
                    "profile_urls": [ext_full_url],
                    "source_links": [{"url": ext_full_url, "anchor_text": ext_name, "source_context": "external_homepage"}],
                    "_page_class": page_class,
                    "groups": [],
                    "raw_title": "",
                    "profile_source": "external_homepage",
                    "profile_confidence": result["confidence"],
                    "verification_signals": result["signals"],
                })
                logger.info(
                    "[generic_extractor] External homepage accepted: %r → %s (%s, %s)",
                    ext_name, ext_full_url[:60],
                    result["confidence"], result["signals"],
                )
            else:
                logger.debug(
                    "[generic_extractor] External homepage rejected: %r → %s (%s)",
                    ext_name, ext_full_url[:60], result["signals"],
                )

        raw_signals["ext_verified"] = ext_verified

        return ExtractionResult(
            strategy_used=strategy,
            page_representation="list",
            faculty_list=faculty,
            faculty_count=len(faculty),
            raw_signals=raw_signals,
        )

    # Unknown strategy name — should not happen
    logger.error("[Executor] Unknown strategy: %s", strategy)
    return ExtractionResult(
        strategy_used=strategy,
        page_representation="unknown",
        faculty_list=[],
        faculty_count=0,
        raw_signals={"error": f"unknown strategy: {strategy}"},
    )


def section_type_to_rep(section_type: str) -> str:
    _MAP = {
        "faculty_section": "list",
        "grouped_faculty_section": "grouped",
        "mixed_section": "mixed",
        "interactive_grouped_faculty_section": "interactive",
        "publication_section": "unknown",
    }
    return _MAP.get(section_type, "unknown")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

# Minimum acceptable validator score for a result to be considered "success"
_PASS_THRESHOLD = 0.60

# Leakage thresholds
_TOPIC_LEAK_MAX = 0.20   # >20% group-topic entries → flag
_PUB_LEAK_MAX   = 0.10   # >10% publication entries → flag
_NAME_PROP_MIN  = 0.70   # <70% valid person names → flag
_LINK_PROP_MIN  = 0.50   # <50% entries with profile URL → flag

# LLM fallback threshold — trigger when name_proportion is this bad
_LLM_FALLBACK_NAME_PROP = 0.60   # <60% valid names → worth trying LLM


def _needs_llm_fallback(validation: ValidationResult) -> bool:
    """
    Return True when the best rule-based result is bad enough to justify
    calling the LLM fallback extractor (cost: ~$0.0005 per page).

    Triggers when:
      - name_proportion < threshold  (extracted entries are mostly not people)
      - OR significant topic/pub leakage  (wrong content type extracted)
    Does NOT trigger when validation already succeeded.
    """
    if validation.success:
        return False
    if validation.name_proportion < _LLM_FALLBACK_NAME_PROP:
        return True
    if validation.topic_leakage_rate > _TOPIC_LEAK_MAX:
        return True
    if validation.pub_leakage_rate > _PUB_LEAK_MAX:
        return True
    return False


def _validate(result: ExtractionResult) -> ValidationResult:
    """
    Score extraction quality using concrete, rule-based checks.

    Scoring weights:
      name_proportion      0.40  — are entries actually people?
      topic_not_leaked     0.25  — group headings not in faculty list?
      pub_not_leaked       0.15  — publication entries absent?
      link_proportion      0.10  — entries have profile URLs?
      count_plausibility   0.10  — is the count believable?
    """
    srv = _srv()
    faculty = result.faculty_list
    n = len(faculty)

    if n == 0:
        # Special case: interactive JS-blocked returns empty but leaves signals
        if result.raw_signals.get("js_blocked"):
            return ValidationResult(
                validator_score=0.0, success=False,
                issues=["js_blocked"],
                failure_reason="interactive page requires JavaScript — cannot scrape",
                name_proportion=0.0, topic_leakage_rate=0.0, pub_leakage_rate=0.0,
            )
        return ValidationResult(
            validator_score=0.0, success=False,
            issues=["empty_result"],
            failure_reason="no faculty extracted",
            name_proportion=0.0, topic_leakage_rate=0.0, pub_leakage_rate=0.0,
        )

    if n == 1:
        return ValidationResult(
            validator_score=0.3, success=False,
            issues=["implausible_count"],
            failure_reason="only 1 result — likely extraction error",
            name_proportion=1.0, topic_leakage_rate=0.0, pub_leakage_rate=0.0,
        )

    # ── Per-entry classification ──────────────────────────────────────────────
    valid_name_count = 0
    topic_count = 0
    pub_count = 0
    link_count = 0

    for f in faculty:
        name = f.get("name", "")
        url_f = f.get("full_profile_url", "")
        low_conf = f.get("_low_confidence", False)

        clean_name, _ = srv._split_name_from_title(name)
        words = clean_name.split()

        # Fix 4: role-prefix blocklist counts as topic leakage in validator
        if len(words) == 2 and words[0].lower() in _ROLE_PREFIXES:
            topic_count += 1
            continue

        if srv._is_valid_person_name(clean_name):
            valid_name_count += 1
        elif low_conf and url_f:
            # Single-word surname entries kept from URL-confirmed profile links
            # (e.g. NYU CDS <a href="/people/stoyanovich">Stoyanovich</a>).
            # The URL already confirms they're real people; count as valid names.
            valid_name_count += 1

        etype = srv._classify_entry_type(name, url_f)
        if etype == "group_topic":
            topic_count += 1
        elif etype == "publication":
            pub_count += 1

        if url_f:
            link_count += 1

    name_proportion     = valid_name_count / n
    topic_leakage_rate  = topic_count / n
    pub_leakage_rate    = pub_count / n
    link_proportion     = link_count / n
    count_plausibility  = min(1.0, n / 5)   # saturates at 5+

    # ── Issues list ───────────────────────────────────────────────────────────
    issues: list[str] = []
    if name_proportion < _NAME_PROP_MIN:
        issues.append("low_name_proportion")
    if topic_leakage_rate > _TOPIC_LEAK_MAX:
        issues.append("topic_leakage")
    if pub_leakage_rate > _PUB_LEAK_MAX:
        issues.append("publication_leakage")
    if link_proportion < _LINK_PROP_MIN:
        issues.append("missing_profile_links")

    # ── Composite score ───────────────────────────────────────────────────────
    score = (
        0.40 * name_proportion
        + 0.25 * (1.0 - topic_leakage_rate)
        + 0.15 * (1.0 - pub_leakage_rate)
        + 0.10 * link_proportion
        + 0.10 * count_plausibility
    )
    score = round(score, 3)

    # Hard-fail on leakage issues regardless of composite score.
    # A 60% topic-leak result must not pass even if other metrics look fine.
    hard_fail = "topic_leakage" in issues or "publication_leakage" in issues
    success = score >= _PASS_THRESHOLD and n >= 2 and not hard_fail

    # ── Failure reason (most specific first) ─────────────────────────────────
    failure_reason: str | None = None
    if not success:
        if "topic_leakage" in issues:
            failure_reason = "group topic headings treated as faculty entries"
        elif "publication_leakage" in issues:
            failure_reason = "publication entries mixed with faculty"
        elif "low_name_proportion" in issues:
            failure_reason = "extracted entries do not look like person names"
        elif "missing_profile_links" in issues:
            failure_reason = "most entries lack profile URLs"
        else:
            failure_reason = f"low extraction score ({score:.2f})"
        if issues:
            issues.append("low_score")

    return ValidationResult(
        validator_score=score,
        success=success,
        issues=issues,
        failure_reason=failure_reason,
        name_proportion=round(name_proportion, 3),
        topic_leakage_rate=round(topic_leakage_rate, 3),
        pub_leakage_rate=round(pub_leakage_rate, 3),
    )


# ---------------------------------------------------------------------------
# Hidden-content detection and recovery
# ---------------------------------------------------------------------------

def _find_source_section(el) -> str:
    """
    Walk up the DOM from el to find the nearest preceding section heading.
    Returns the heading text (≤80 chars) or '' if none found.
    """
    from bs4 import Tag
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5"}
    node = el
    for _ in range(8):
        if node is None or not hasattr(node, "previous_siblings"):
            break
        for sib in node.previous_siblings:
            if not isinstance(sib, Tag):
                continue
            if sib.name in _HEADINGS:
                return sib.get_text(strip=True)[:80]
            h = sib.find(_HEADINGS)
            if h:
                return h.get_text(strip=True)[:80]
        node = node.parent
    return ""


_SHOW_MORE_TEXT_RE = re.compile(
    r"\b(show\s+more|load\s+more|view\s+more|see\s+more|more\s+faculty|"
    r"more\s+people|expand\s+all)\b",
    re.IGNORECASE,
)
_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_HIDDEN_CLASS_RE = re.compile(
    r"\b(is-hidden|js-hidden|d-none|hide(?!-)|hidden(?!-input))\b", re.IGNORECASE
)


def _detect_show_more_signals(soup: BeautifulSoup) -> list[dict]:
    """Return a list of show-more / hidden-content signals found on the page."""
    signals: list[dict] = []
    for tag in soup.find_all(["button", "a", "span", "div"]):
        text = tag.get_text(strip=True)
        if not text or len(text) > 60:
            continue
        if _SHOW_MORE_TEXT_RE.search(text):
            href = tag.get("href", "") or ""
            followable = bool(href and href not in ("#", "javascript:void(0)", "javascript:"))
            signals.append({
                "type": "show_more_button",
                "text": text,
                "href": href,
                "followable": followable,
            })
    for tag in soup.find_all(attrs={"aria-expanded": "false"}):
        href = tag.get("href", "") or ""
        signals.append({
            "type": "aria_expanded_false",
            "tag": tag.name,
            "href": href,
            "followable": bool(href and href not in ("#", "javascript:void(0)", "javascript:")),
        })
    return signals


def _recover_static_hidden_people(soup: BeautifulSoup, url: str) -> list[dict]:
    """
    Try to extract people from DOM elements hidden by CSS/attributes.

    Strategy A: name-as-alt images inside hidden containers (name_alt pattern).
    Strategy B: profile-URL links inside hidden containers.

    Returns a list of additional entries (may be empty).
    """
    srv = _srv()
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    people: list[dict] = []
    seen_names: set[str] = set()

    def _ancestor_is_hidden(el, max_depth: int = 8) -> bool:
        node = el
        for _ in range(max_depth):
            if node is None or not hasattr(node, "get"):
                break
            if _HIDDEN_STYLE_RE.search(node.get("style", "")):
                return True
            if node.has_attr("hidden"):
                return True
            if _HIDDEN_CLASS_RE.search(" ".join(node.get("class", []))):
                return True
            node = node.parent
        return False

    # A. name-as-alt images in hidden containers
    for img in soup.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if not alt or len(alt) < 4:
            continue
        if not _ancestor_is_hidden(img):
            continue
        name, _ = srv._split_name_from_title(alt)
        if not srv._is_valid_person_name(name):
            continue
        name_key = name.lower().strip()
        if name_key in seen_names:
            continue
        link = img.find_parent("a", href=True) or (
            img.parent.find("a", href=True) if img.parent else None
        )
        href = link["href"] if link else ""
        full_url = (urljoin(base, href) if href and not href.startswith("http") else href) or ""
        seen_names.add(name_key)
        people.append({
            "name": name,
            "profile_url": href,
            "full_profile_url": full_url,
            "profile_urls": [full_url] if full_url else [],
            "source_links": [{"url": full_url, "anchor_text": alt, "source_context": "hidden_dom"}],
            "_from_hidden_dom": True,
            "raw_title": "",
        })

    # B. profile-URL links in hidden containers
    _profile_pat = re.compile(r"/(people|faculty[^/]*|profile[^/]*|directory|personnel)/[^/?#]+", re.I)
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not _profile_pat.search(href):
            continue
        if not _ancestor_is_hidden(a):
            continue
        raw_text = a.get_text(separator=" ", strip=True)
        if not raw_text or len(raw_text) < 4:
            continue
        name, _ = srv._split_name_from_title(raw_text)
        if not srv._is_valid_person_name(name):
            continue
        name_key = name.lower().strip()
        if name_key in seen_names:
            continue
        full_url = urljoin(base, href) if not href.startswith("http") else href
        seen_names.add(name_key)
        people.append({
            "name": name,
            "profile_url": href,
            "full_profile_url": full_url,
            "profile_urls": [full_url],
            "source_links": [{"url": full_url, "anchor_text": raw_text[:80],
                              "source_section": "", "source_context": "hidden_dom"}],
            "_from_hidden_dom": True,
            "raw_title": "",
        })

    # C. USC Marshall-style person-list-item cards inside hidden containers
    for li in soup.find_all("li", class_="person-list-item"):
        if not _ancestor_is_hidden(li):
            continue
        # Extract name and link from person card
        name_link = li.find("a", href=True)
        if name_link is None:
            continue
        href = name_link.get("href", "")
        raw_text = name_link.get_text(separator=" ", strip=True)
        if not raw_text or len(raw_text) < 4:
            # Try the li's text directly
            raw_text = li.get_text(separator=" ", strip=True).split("\n")[0][:60]
        name, raw_title = srv._split_name_from_title(raw_text)
        if not srv._is_valid_person_name(name):
            continue
        name_key = name.lower().strip()
        if name_key in seen_names:
            continue
        full_url = (urljoin(base, href) if href and not href.startswith("http") else href) or ""
        seen_names.add(name_key)
        people.append({
            "name": name,
            "profile_url": href,
            "full_profile_url": full_url,
            "profile_urls": [full_url] if full_url else [],
            "source_links": [{"url": full_url, "anchor_text": raw_text[:80],
                              "source_section": "", "source_context": "hidden_person_card"}],
            "_from_hidden_dom": True,
            "raw_title": raw_title,
        })

    return people


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_extraction_agent(url: str, _depth: int = 0) -> ExtractionOutcome:
    """
    Main entry point: fetch page, plan, execute, validate, loop on failure.

    Returns ExtractionOutcome — call .to_legacy_format() for backward compat.

    Loop (max MAX_ATTEMPTS):
      load Kahuna context
      → planner picks strategy order
      → for each strategy (up to MAX_ATTEMPTS):
          executor runs strategy
          validator scores result
          if success → stop
          else       → log issue, try next strategy
      → record final outcome to Kahuna

    _depth: internal recursion guard for linked-expansion (hidden content B path).
            Never pass this argument externally.
    """
    srv = _srv()
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    domain = urlparse(url).netloc.replace(":", "_").replace(".", "-")

    # ── 1. Fetch page ─────────────────────────────────────────────────────────
    try:
        r = srv._get_with_reason(url, timeout=30)
    except srv._FetchError as e:
        outcome = ExtractionOutcome(
            url=url, domain=domain,
            page_representation="unknown", strategy_used="none",
            faculty_count=0, faculty_names_sample=[],
            validator_score=0.0, issues=["fetch_error"],
            success=False, failure_reason=str(e),
            next_best_strategy=None, strategy_trace=[],
            timestamp=ts,
            error_payload={"error": e.reason},
        )
        _kahuna_record_outcome(outcome)
        return outcome

    soup = BeautifulSoup(r.text, "lxml")
    page_class = srv._classify_page_content(soup, url)
    logger.info("[ExtractionAgent] url=%s page_class=%s", url[:80], page_class)

    # ── 1b. Early exit for single-profile pages ───────────────────────────────
    # A single_profile_page is an individual faculty bio, not a directory.
    # BUT: some program pages get misclassified as single_profile_page while
    # actually containing a faculty section (e.g. "TRANSPORTATION SYSTEMS FACULTY").
    # Guard: only early-exit if no faculty section is detectable on the page.
    if page_class == "single_profile_page":
        _section_check = srv._find_faculty_section(soup)
        if _section_check is None:
            logger.info("[ExtractionAgent] Early exit: single_profile_page — no faculty section found")
            outcome = ExtractionOutcome(
                url=url, domain=domain,
                page_representation="single_profile_page", strategy_used="none",
                faculty_count=0, faculty_names_sample=[],
                validator_score=0.0, issues=["not_a_directory"],
                success=False, failure_reason="not a faculty directory (single profile page)",
                next_best_strategy=None, strategy_trace=[],
                timestamp=ts,
                error_payload={"error": "Page is a single faculty profile, not a directory."},
            )
            _kahuna_record_outcome(outcome)
            return outcome
        else:
            # Faculty section found — the single_profile_page classification is wrong.
            # Override and proceed with extraction as a faculty_collection_page.
            logger.info(
                "[ExtractionAgent] single_profile_page overridden: faculty section detected — "
                "proceeding as faculty_collection_page"
            )
            page_class = "faculty_collection_page"

    # ── 2. Load Kahuna memory ─────────────────────────────────────────────────
    mem_ctx = _kahuna_load_context(url)
    if mem_ctx:
        logger.info(
            "[ExtractionAgent] Memory: domain=%s last=%s/%s best=%s failed=%s",
            mem_ctx["domain"], mem_ctx["last_strategy"],
            mem_ctx["last_success"], mem_ctx["best_strategy"],
            mem_ctx["failed_strategies"],
        )

    # ── 3. Plan ───────────────────────────────────────────────────────────────
    plan = _plan(url, soup, page_class, mem_ctx)
    logger.info(
        "[Planner] page_rep=%s order=%s reason=%s",
        plan.page_representation, plan.strategy_order, plan.reasoning,
    )

    # ── 4. Execute → Validate loop ────────────────────────────────────────────
    strategy_trace: list[dict] = []
    best_result: ExtractionResult | None = None
    best_validation: ValidationResult | None = None

    for attempt, strategy in enumerate(plan.strategy_order[:MAX_ATTEMPTS], 1):
        logger.info("[ExtractionAgent] Attempt %d/%d — strategy=%s",
                    attempt, min(len(plan.strategy_order), MAX_ATTEMPTS), strategy)

        result = _execute(strategy, url, soup, page_class)
        validation = _validate(result)

        trace_entry = {
            "strategy": strategy,
            "faculty_count": result.faculty_count,
            "validator_score": validation.validator_score,
            "issues": validation.issues,
            "success": validation.success,
        }
        strategy_trace.append(trace_entry)

        logger.info(
            "[Validator] strategy=%s count=%d score=%.2f success=%s issues=%s",
            strategy, result.faculty_count, validation.validator_score,
            validation.success, validation.issues,
        )

        # Keep the best result seen (highest score), even if not passing threshold
        if best_validation is None or validation.validator_score > best_validation.validator_score:
            best_result = result
            best_validation = validation

        if validation.success:
            break   # good enough — stop trying

        if attempt < min(len(plan.strategy_order), MAX_ATTEMPTS):
            logger.info(
                "[ExtractionAgent] Strategy %s failed (%s) — trying fallback",
                strategy, validation.failure_reason,
            )

    assert best_result is not None
    assert best_validation is not None

    # ── 4b. LLM fallback — triggered when rule-based results have bad name quality ──
    # Runs OUTSIDE the MAX_ATTEMPTS cap (it's expensive; we only try once).
    tried_strategies = {t["strategy"] for t in strategy_trace}
    if "llm_extractor" not in tried_strategies and _needs_llm_fallback(best_validation):
        logger.info(
            "[ExtractionAgent] LLM fallback triggered — name_prop=%.2f leak=%.2f/%.2f",
            best_validation.name_proportion,
            best_validation.topic_leakage_rate,
            best_validation.pub_leakage_rate,
        )
        llm_result     = _execute("llm_extractor", url, soup, page_class)
        llm_validation = _validate(llm_result)
        strategy_trace.append({
            "strategy":       "llm_extractor",
            "faculty_count":  llm_result.faculty_count,
            "validator_score": llm_validation.validator_score,
            "issues":         llm_validation.issues,
            "success":        llm_validation.success,
        })
        logger.info(
            "[Validator] strategy=llm_extractor count=%d score=%.2f success=%s",
            llm_result.faculty_count, llm_validation.validator_score, llm_validation.success,
        )
        if llm_validation.validator_score > best_validation.validator_score:
            best_result     = llm_result
            best_validation = llm_validation

    # ── 5. Handle interactive JS-blocked edge case ─────────────────────────────
    if not best_result.faculty_list and best_result.raw_signals.get("js_blocked"):
        # Build the interactive error payload but still record to Kahuna
        interactive_groups = best_result.raw_signals.get("groups", [])
        error_payload = {
            "page_type": "interactive_grouped_faculty_page",
            "groups": interactive_groups,
            "error": best_result.raw_signals.get("error", "JavaScript-only page"),
        }
        outcome = ExtractionOutcome(
            url=url, domain=domain,
            page_representation="interactive",
            strategy_used=best_result.strategy_used,
            faculty_count=0, faculty_names_sample=[],
            validator_score=best_validation.validator_score,
            issues=best_validation.issues,
            success=False,
            failure_reason=best_validation.failure_reason,
            next_best_strategy=None,
            strategy_trace=strategy_trace,
            timestamp=ts,
            faculty_list=[],
            error_payload=error_payload,
        )
        _kahuna_record_outcome(outcome)
        return outcome

    # ── 6. Build final outcome ────────────────────────────────────────────────
    faculty_list = best_result.faculty_list
    for e in faculty_list:
        e.setdefault("_page_class", page_class)
        e.setdefault("groups", [])

    # ── 6a. Role tagging (metadata only — no filtering) ──────────────────────
    # Attach role_hint to each entry for downstream consumers.
    # No entries are removed here; all extracted people pass through.
    for e in faculty_list:
        name = e.get("name", "")
        raw_title = e.get("raw_title", "") or e.get("title", "")
        href = e.get("full_profile_url", "") or e.get("profile_url", "")
        e["role_hint"] = srv._classify_person_role(name, raw_title, href)
        # Normalise title → raw_title for downstream consumers
        if "title" in e and "raw_title" not in e:
            e["raw_title"] = e.pop("title")
        # Ensure profile_urls exists on every entry
        if "profile_urls" not in e:
            furl = e.get("full_profile_url", "")
            e["profile_urls"] = [furl] if furl else []

    # ── 6b. Hidden-content detection and recovery ─────────────────────────────
    hidden_signals = _detect_show_more_signals(soup)
    hidden_detected = len(hidden_signals) > 0
    requires_browser = False
    hidden_warning: str | None = None

    if hidden_detected:
        logger.info("[hidden_content] %d show-more/hidden signal(s) detected", len(hidden_signals))

        # A. Static hidden DOM recovery (images + links inside hidden containers)
        hidden_people = _recover_static_hidden_people(soup, url)
        if hidden_people:
            existing_names = {e.get("name", "").lower().strip() for e in faculty_list}
            added = 0
            for p in hidden_people:
                nk = p.get("name", "").lower().strip()
                if nk and nk not in existing_names:
                    existing_names.add(nk)
                    p.setdefault("_page_class", page_class)
                    p.setdefault("groups", [])
                    p.setdefault("role_hint", "unknown")
                    faculty_list.append(p)
                    added += 1
            if added:
                logger.info("[hidden_content] Static DOM recovery added %d people", added)
        else:
            # B. Linked expansion: follow href on show-more buttons (depth-limited)
            # Skip if we already have a good-sized result — the "show more" signals are
            # likely navigation links (e.g. "View Faculty →") rather than paginated loaders.
            followable = [s for s in hidden_signals if s.get("followable")]
            linked_added = 0
            if _depth == 0 and followable and len(faculty_list) < 5:
                # Only recurse one level deep to avoid infinite loops
                for sig in followable[:3]:
                    href_sig = sig["href"]
                    link_url = (
                        urljoin(url, href_sig) if not href_sig.startswith("http") else href_sig
                    )
                    if link_url.rstrip("/") == url.rstrip("/"):
                        continue
                    try:
                        sub = run_extraction_agent(link_url, _depth=1)
                        if sub.success and sub.faculty_list:
                            existing_names = {e.get("name", "").lower().strip() for e in faculty_list}
                            for p in sub.faculty_list:
                                nk = p.get("name", "").lower().strip()
                                if nk and nk not in existing_names:
                                    existing_names.add(nk)
                                    p.setdefault("_page_class", page_class)
                                    p.setdefault("groups", [])
                                    faculty_list.append(p)
                                    linked_added += 1
                    except Exception as _exc:
                        logger.warning("[hidden_content] Linked expansion failed: %s", _exc)
                if linked_added:
                    logger.info("[hidden_content] Linked expansion added %d people", linked_added)

            if not linked_added:
                if followable and _depth > 0:
                    pass  # at max depth — skip linked expansion silently
                elif followable:
                    pass  # linked but nothing useful returned
                else:
                    # C. JS-only — cannot resolve statically
                    requires_browser = True
                    hidden_warning = (
                        f"Page has {len(hidden_signals)} show-more/hidden signal(s) that require "
                        "browser interaction to expand. Extraction may be incomplete."
                    )
                    logger.info("[hidden_content] Requires browser interaction — warning set")

    # ── 6c. Filterable directory detection ───────────────────────────────────
    # USC-style pages with query-param faculty links and large result sets are
    # typically full-department directories, not scoped faculty listings.
    # Cap at 200 entries and flag when the raw count exceeds that.
    # Previously gated on usc_extractor only; now fires for any strategy on
    # query-param directory pages (llm_extractor handles USC pages going forward).
    _FILTERABLE_CAP = 200
    page_rep_final = best_result.page_representation
    extra_issues = list(best_validation.issues)
    _is_large_directory = (
        len(faculty_list) > _FILTERABLE_CAP
        and page_rep_final in ("list", "filterable_directory")
    )
    if _is_large_directory:
        logger.info(
            "[filterable_directory] %s returned %d entries — "
            "capping at %d and flagging as filterable_directory",
            best_result.strategy_used, len(faculty_list), _FILTERABLE_CAP,
        )
        faculty_list = faculty_list[:_FILTERABLE_CAP]
        page_rep_final = "filterable_directory"
        if "filterable_directory" not in extra_issues:
            extra_issues.append("filterable_directory")

    # Next-best strategy: first strategy in plan.strategy_order not yet tried or not used
    tried = {t["strategy"] for t in strategy_trace}
    next_best = next(
        (s for s in plan.strategy_order if s not in tried), None
    )

    outcome = ExtractionOutcome(
        url=url, domain=domain,
        page_representation=page_rep_final,
        strategy_used=best_result.strategy_used,
        faculty_count=len(faculty_list),
        faculty_names_sample=[f["name"] for f in faculty_list[:5]],
        validator_score=best_validation.validator_score,
        issues=extra_issues,
        success=best_validation.success,
        failure_reason=best_validation.failure_reason,
        next_best_strategy=next_best,
        strategy_trace=strategy_trace,
        timestamp=ts,
        faculty_list=faculty_list,
        error_payload=None,
        hidden_content_detected=hidden_detected,
        requires_browser_interaction=requires_browser,
        hidden_content_warning=hidden_warning,
    )

    # ── 7. Record to Kahuna ────────────────────────────────────────────────────
    _kahuna_record_outcome(outcome)

    logger.info(
        "[ExtractionAgent] Final: strategy=%s count=%d score=%.2f success=%s trace=%s",
        outcome.strategy_used, outcome.faculty_count,
        outcome.validator_score, outcome.success,
        [(t["strategy"], t["success"]) for t in strategy_trace],
    )

    return outcome
