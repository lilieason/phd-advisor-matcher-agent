"""
MCP Server for Advisor Matching.

Tools:
  - read_cv(file_path)
  - fetch_faculty_list(url)
  - fetch_all_faculty_profiles(faculty_list_json, max_count)
  - fetch_google_scholar_profile(scholar_url)
      Fetch up to 20 recent publications from a Google Scholar profile.
      Returns JSON: {fetch_status, publications: [{title, year, venue, authors}],
                     summary: {recent_themes, methods, applications}}
  - fetch_faculty_page(url)
"""

import asyncio
import datetime
import io
import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

# curl_cffi uses Chrome's TLS fingerprint — bypasses Cloudflare bot detection
# that blocks plain requests/urllib even with browser User-Agent headers.
# We REQUIRE it; if it's missing, fail at import time rather than silently
# degrading to plain requests (which will 403 on Cloudflare-protected pages).
from curl_cffi import requests as _http   # pip install curl-cffi

import requests as _requests_plain   # kept for Scholar (different anti-bot)

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log which HTTP client is active at startup so stale-process issues are
# immediately visible in the server log.
logger.info("HTTP client: curl_cffi/%s (Chrome TLS fingerprint active)",
            getattr(_http, "__version__", "?"))

mcp = FastMCP("Advisor Server")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ── helpers ──────────────────────────────────────────────────────────────── #

class _FetchError(Exception):
    """Carries a human-readable reason for display in the UI."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _is_cf_challenge(resp) -> bool:
    """Return True if the response body looks like a Cloudflare challenge page."""
    if resp.status_code not in (403, 429, 503):
        return False
    head = resp.text[:400].lower()
    return (
        "cf-mitigated" in resp.headers.get("cf-mitigated", "").lower()
        or "just a moment" in head
        or "enable javascript" in head
        or "cloudflare" in resp.headers.get("server", "").lower()
    )


def _raw_get(url: str, timeout: int) -> "_http.Response":
    """Single attempt using curl_cffi with Chrome TLS fingerprint."""
    return _http.get(
        url,
        impersonate="chrome",
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )


def _get(url: str, timeout: int = 25, retries: int = 2) -> "_http.Response | None":
    """
    Fetch *url* via curl_cffi (Chrome TLS fingerprint).
    Retries on transient Cloudflare challenges (403/503) with a short delay.
    Returns the Response on success, None on unrecoverable failure.
    """
    import time
    last_err: str = ""
    for attempt in range(1, retries + 2):   # retries=2 → 3 total attempts
        try:
            r = _raw_get(url, timeout)
            logger.info(
                "GET(%d) %s → %s final=%s len=%d",
                attempt, url[:80], r.status_code, r.url[:80], len(r.content),
            )
            if _is_cf_challenge(r):
                last_err = f"Cloudflare challenge (HTTP {r.status_code})"
                logger.warning("GET(%d) %s — %s; %s",
                               attempt, url[:80], last_err,
                               "retrying…" if attempt <= retries else "giving up")
                if attempt <= retries:
                    time.sleep(2 ** attempt)   # 2s, 4s back-off
                continue
            if r.status_code == 429:
                last_err = "Rate-limited (429)"
                logger.warning("GET(%d) %s — %s; %s",
                               attempt, url[:80], last_err,
                               "retrying…" if attempt <= retries else "giving up")
                if attempt <= retries:
                    time.sleep(5)
                continue
            r.raise_for_status()
            return r

        except _http.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("GET(%d) %s — %s", attempt, url[:80], last_err)

    logger.error("GET gave up after %d attempts for %s — %s", retries + 1, url, last_err)
    return None


def _get_with_reason(url: str, timeout: int = 25, retries: int = 2) -> "_http.Response":
    """
    Like _get() but raises _FetchError with a UI-friendly message on failure.
    Use this wherever the caller needs to surface the error reason.
    """
    import time
    last_err: str = ""
    for attempt in range(1, retries + 2):
        try:
            r = _raw_get(url, timeout)
            logger.info(
                "GET(%d) %s → %s final=%s len=%d",
                attempt, url[:80], r.status_code, r.url[:80], len(r.content),
            )
            if _is_cf_challenge(r):
                snippet = r.text[:120].replace('\n', ' ')
                last_err = (
                    f"HTTP {r.status_code} — Cloudflare challenge page. "
                    f"Snippet: {snippet!r}"
                )
                logger.warning("GET(%d) %s — CF challenge; %s",
                               attempt, url[:80],
                               "retrying…" if attempt <= retries else "giving up")
                if attempt <= retries:
                    time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                last_err = "HTTP 429 — rate-limited, try again later"
                if attempt <= retries:
                    time.sleep(5)
                    continue
                raise _FetchError(last_err)
            if r.status_code >= 400:
                raise _FetchError(f"HTTP {r.status_code} error fetching {url}")
            return r

        except _FetchError:
            raise
        except Exception as e:
            etype = type(e).__name__
            msg = str(e)
            if "timeout" in etype.lower() or "timeout" in msg.lower():
                last_err = f"Timeout after {timeout}s"
            elif "ssl" in etype.lower() or "ssl" in msg.lower():
                # Retry with SSL verification disabled for sites with local CA issues
                try:
                    r2 = _http.get(
                        url, impersonate="chrome", headers=HEADERS,
                        timeout=timeout, allow_redirects=True,
                        verify=False,
                    )
                    logger.warning("GET(%d) %s — SSL fallback (verify=False) → %d",
                                   attempt, url[:80], r2.status_code)
                    if r2.status_code < 400:
                        return r2
                except Exception:
                    pass
                raise _FetchError(f"SSL error fetching {url}: {e}")
            elif "connection" in etype.lower():
                raise _FetchError(f"Connection error fetching {url}: {e}")
            else:
                raise _FetchError(f"Fetch error [{etype}]: {e}")
            logger.warning("GET(%d) %s — %s", attempt, url[:80], last_err)

    raise _FetchError(f"Failed after {retries + 1} attempts — {last_err}")


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove non-content elements from a parsed page."""
    # Remove standard non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]):
        tag.decompose()
    # Remove div/section elements whose class or id signals navigation/chrome
    _NAV_PATTERN = re.compile(
        r"\b(nav|menu|sidebar|breadcrumb|calendar|widget|toolbar|"
        r"search|banner|advertisement|cookie|popup|modal|overlay|"
        r"social|share|related|pagination|skip|utility|top-bar|"
        r"global-header|global-footer|site-header|site-footer)\b",
        re.IGNORECASE,
    )
    for tag in soup.find_all(True):
        if tag.name in ("html", "body", "main", "article", "section", "div", "ul", "li"):
            cls = " ".join(tag.get("class", []))
            tid = tag.get("id", "")
            combined = f"{cls} {tid}"
            if _NAV_PATTERN.search(combined):
                tag.decompose()
    return soup


# ── Name / page validation helpers ───────────────────────────────────────── #

# Keywords that indicate a site title, not a person name
_BAD_NAME_RE = re.compile(
    r"\b(Faculty|Directory|Viterbi|USC|ISE|Department|School|College|"
    r"Home|About|People|Staff|Search|Profile|Page|Menu|Navigation|"
    r"Engineering|University|Error|Loading|Contact|Resources|Research|"
    r"Lab|Group|Center|Institute|Division|Program|Course|Class|"
    r"Login|Admin|Portal|Leadership|Advisory|Board|Committee|"
    r"Students|Alumni|Emeriti|Affiliates|"
    r"View|Complete|Reset|Filters|Filter|Annual|Reports|Report|"
    r"Download|Upload|Submit|Apply|Register|Subscribe|News|Events|"
    r"Calendar|Awards|Grants|Funding|Jobs|Careers|Apply|"
    r"Tech|Headshot|Portrait|Thumbnail|Smiling|Gown|"
    # Navigation / admin page nouns that can look like 2-word person names
    r"Policy|Privacy|Services|Advising|Enrollment|Requirements|"
    r"Evaluation|Admissions|Tutors|Opportunities|Seminar|Workshop|"
    r"Overview|Guidelines|Handbook|Procedures|Syllabus|Curriculum|"
    r"Challenge|Memorial|Memoriam|Inauguration|Commencement|"
    # Organization / institutional nouns (not person names)
    r"Association|Foundation|Network|Society|Organization|Organizational|Consortium|"
    r"Partnership|Federation|Coalition|Alliance|Council|Union|"
    r"Medicine|Today|Card|Boundless|Libraries|Computing|Directories|"
    r"Maps|Tacoma|Bothell|Sitemap|Commencement|Planning|"
    r"Accessibility|Education|Digital|Executive|Engagement|"
    r"Innovation|Transformation|Excellence|Experience|Impact|"
    # Document / image / chart nouns that can trail a name (e.g. "Qianru Guo Pic")
    r"Charts|Chart|Diagram|Diagrams|Slides|Slide|Deck|"
    r"Pic|Pics|Photo|Photos|Image|Images|"
    # Physical venues / places (addresses, buildings, hotels, parking)
    r"Hotel|Garage|Parking|Plaza|Lobby|Suite|Auditorium|Cafeteria|"
    r"Annex|Pavilion|Coliseum|Stadium|Arena|Atrium|"
    # Academic degree / program tokens
    r"Analytics|Logistics|Operations|Supply|Curriculum|"
    r"Bachelor|Master|Doctorate|Degree|Diploma|Certificate|"
    r"Undergraduate|Graduate|Doctoral|"
    # Well-known institution / campus names that appear as non-person text
    r"Berkeley|Stanford|Caltech|Princeton|Harvard|Columbia|Cornell|"
    r"MIT|NYU|UCLA|USC|CMU|Purdue|Rutgers|Yale)\b",
    re.IGNORECASE,
)


def _is_valid_person_name(text: str) -> bool:
    """Return True if text plausibly looks like a human name."""
    words = text.strip().split()
    if len(words) < 2 or len(words) > 6:
        return False
    if len(text) > 70:
        return False
    if _BAD_NAME_RE.search(text):
        return False
    # Reject if any word starts with a digit (e.g. "Level 2", "20988936 771...")
    if any(w[0].isdigit() for w in words if w):
        return False
    # Reject entries whose first word is a degree abbreviation
    # (e.g. "Bs Ieor", "Bs Ms Ieor", "Mba Analytics" are degree+program combos).
    # "Ma" and "Meng" are excluded — common Chinese surnames/given names.
    _DEGREE_ABBREVS = frozenset({"bs", "ba", "mba", "msc", "jd", "md"})
    if words[0].lower().rstrip(".") in _DEGREE_ABBREVS:
        return False
    # First word must start with an uppercase letter (person names are proper nouns).
    # Allows particles like "de", "van", "von" only as non-first words.
    first_alpha = re.sub(r"[^A-Za-z]", "", words[0])
    if not first_alpha:
        # No alphabetic characters in first word (e.g. purely numeric token)
        return False
    if first_alpha[0].islower():
        return False
    # Reject if any word is all-caps and longer than 3 chars (likely acronym/abbrev)
    for w in words:
        alpha = re.sub(r"[^A-Za-z]", "", w)
        if alpha and alpha.isupper() and len(alpha) > 3:
            return False
    # Reject if any word is a bare hyphen or non-letter punctuation token
    if any(w in ("-", "–", "—", "|", "/", "·") for w in words):
        return False
    # Reject breadcrumb / navigation labels: any word ending with ":" (e.g. "You are here:")
    if any(w.endswith(":") for w in words):
        return False
    # Reject if first word is a common English function word (pronouns, articles).
    # These never start a person's name.
    _FUNCTION_WORDS = frozenset({
        "you", "your", "we", "our", "they", "their", "it", "its",
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "all", "any", "some", "no", "not", "both", "each",
    })
    if words[0].lower() in _FUNCTION_WORDS:
        return False
    return True


def _name_from_url(url: str) -> str:
    """
    Try to derive a faculty name from the URL itself.
    Handles USC patterns:  ?lname=Dessouky&fname=Maged  →  'Maged Dessouky'
    """
    parsed = urlparse(url)
    qs = parsed.query

    lname_m = re.search(r"lname=([^&]+)", qs, re.IGNORECASE)
    fname_m = re.search(r"fname=([^&]+)", qs, re.IGNORECASE)
    if lname_m and fname_m:
        return f"{fname_m.group(1)} {lname_m.group(1)}"

    # Path-based: /faculty/LastName or /people/FirstName-LastName
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    for part in reversed(path_parts):
        candidate = part.replace("-", " ").replace("_", " ").title()
        if _is_valid_person_name(candidate):
            return candidate

    return ""


# ── Faculty-section heading patterns ─────────────────────────────────────── #

_FACULTY_SECTION_RE = re.compile(
    r"^\s*(?:our\s+)?(?:affiliated|supervising|participating|core|program|"
    r"research|associated|contributing|primary|faculty|people|"
    r"researchers?|team|members?|scholars?|investigators?|affiliates?|experts?)\s*"
    r"(?:faculty|members?|advisors?|researchers?|team|scholars?)?\s*$",
    re.IGNORECASE,
)

# Secondary: headings that END with "Faculty", "Researchers", "Members", or "Team"
# (up to 3 qualifier words before).
# Catches "Transportation Systems Faculty", "Civil Engineering Researchers", etc.
_FACULTY_SECTION_SUFFIX_RE = re.compile(
    r"^\s*(?:\w[\w\-]*\s+){0,3}(?:faculty|researchers?|members?|team)\s*$",
    re.IGNORECASE,
)


def _find_faculty_section(soup: BeautifulSoup) -> BeautifulSoup | None:
    """
    Scan headings (h2–h4) and prominent <strong>/<b> for faculty-section
    headings (e.g. "Affiliated Faculty", "Supervising Faculty", "People").

    Returns the *tightest* container that (a) contains the heading and
    (b) has at least 2 links AND no more than 200 links (to avoid picking up
    the entire page body; legitimate faculty sections can have 50–100+ links
    when each card contains several interest/topic sub-links).

    Headings that live inside <nav>, <header>, <footer>, or <aside> elements
    are always skipped — they are site-chrome, not content sections.

    Strategy, in order:
      1. Next sibling element after the heading — tightest possible scope.
      2. Nearest parent that is a <section>, <article>, or <li>.
      3. Nearest parent <div> — only if it has ≤ 150 links.

    Returns None if no suitable container is found.
    """
    _CHROME_ANCESTORS = {"nav", "header", "footer", "aside"}

    heading_tags = soup.find_all(["h2", "h3", "h4", "strong", "b"])
    for tag in heading_tags:
        text = tag.get_text(strip=True)
        if not (_FACULTY_SECTION_RE.match(text) or _FACULTY_SECTION_SUFFIX_RE.match(text)):
            continue

        # Skip headings that are inside site navigation/chrome
        if tag.find_parent(_CHROME_ANCESTORS):
            logger.debug("[faculty_section] Skipping %r — inside nav/header/footer", text)
            continue

        logger.debug("[faculty_section] Heading match: %r", text)

        # ── 1. Next sibling: tightest scope, prefer this ──────────────────────
        sib = tag.find_next_sibling()
        if sib:
            sib_links = sib.find_all("a", href=True)
            if 2 <= len(sib_links) <= 200:
                logger.info(
                    "[faculty_section] Heading %r → next sibling <%s> with %d links",
                    text, sib.name, len(sib_links),
                )
                return sib

        # ── 2. Nearest <section>, <article>, <li> parent ─────────────────────
        for parent_tag in ["section", "article", "li"]:
            container = tag.find_parent(parent_tag)
            if container:
                c_links = container.find_all("a", href=True)
                if 2 <= len(c_links) <= 200:
                    logger.info(
                        "[faculty_section] Heading %r → parent <%s> with %d links",
                        text, parent_tag, len(c_links),
                    )
                    return container

        # ── 3. Nearest <div> parent — only if ≤ 150 links ───────────────────
        container = tag.find_parent("div")
        if container:
            c_links = container.find_all("a", href=True)
            if 2 <= len(c_links) <= 150:
                logger.info(
                    "[faculty_section] Heading %r → parent <div> with %d links",
                    text, len(c_links),
                )
                return container
            elif len(c_links) == 0:
                # Heading is in a text-only div (no links); try the grandparent div.
                # Example: UCI — h3 inside div.field.body (0 links), links in div.region-inner.
                grandparent = container.find_parent("div")
                if grandparent:
                    gp_links = grandparent.find_all("a", href=True)
                    if 2 <= len(gp_links) <= 150:
                        logger.info(
                            "[faculty_section] Heading %r → grandparent <div> with %d links",
                            text, len(gp_links),
                        )
                        return grandparent
            else:
                logger.debug(
                    "[faculty_section] Heading %r — parent <div> has %d links "
                    "(out of [2,150] range), skipping",
                    text, len(c_links),
                )

    return None


def _find_all_person_sections(soup: BeautifulSoup) -> list:
    """
    Like _find_faculty_section but returns ALL matching section containers.

    Used for multi-section aggregation — pages like NYU Engineering that have
    multiple faculty sections ("Core Faculty", "Affiliated Faculty", etc.).
    Each returned container satisfies the same 2–200 link bounds used by
    _find_faculty_section.  Containers that are subsets of already-found
    containers are excluded to avoid double-counting.

    Returns a list of (heading_text, container) tuples, or [] if none found.
    """
    _CHROME_ANCESTORS = {"nav", "header", "footer", "aside"}
    results: list[tuple[str, BeautifulSoup]] = []
    seen_ids: set[int] = set()   # id(container) to avoid duplicates

    heading_tags = soup.find_all(["h2", "h3", "h4", "strong", "b"])
    for tag in heading_tags:
        text = tag.get_text(strip=True)
        if not (_FACULTY_SECTION_RE.match(text) or _FACULTY_SECTION_SUFFIX_RE.match(text)):
            continue
        if tag.find_parent(_CHROME_ANCESTORS):
            continue

        container = None

        # 1. Next sibling
        sib = tag.find_next_sibling()
        if sib:
            sib_links = sib.find_all("a", href=True)
            if 2 <= len(sib_links) <= 200:
                container = sib

        # 2. Nearest <section>, <article>, <li> parent
        if container is None:
            for parent_tag in ["section", "article", "li"]:
                p = tag.find_parent(parent_tag)
                if p:
                    c_links = p.find_all("a", href=True)
                    if 2 <= len(c_links) <= 200:
                        container = p
                        break

        # 3. Nearest <div> parent
        if container is None:
            p = tag.find_parent("div")
            if p:
                c_links = p.find_all("a", href=True)
                if 2 <= len(c_links) <= 150:
                    container = p
                elif len(c_links) == 0:
                    gp = p.find_parent("div")
                    if gp:
                        gp_links = gp.find_all("a", href=True)
                        if 2 <= len(gp_links) <= 150:
                            container = gp

        if container is None:
            continue

        cid = id(container)
        if cid in seen_ids:
            continue

        # Skip if this container is an ancestor of an already-found container
        # (would cause superset → double-counting).
        is_ancestor = any(
            container in prev_cont.parents
            for _, prev_cont in results
        )
        if is_ancestor:
            continue

        seen_ids.add(cid)
        results.append((text, container))
        logger.info("[all_person_sections] Found section %r (%d links)", text,
                    len(container.find_all("a", href=True)))

    return results


def _detect_faculty_cards(section: BeautifulSoup) -> list:
    """
    Layout-agnostic faculty card detector.

    Searches breadth-first (3 levels deep) for a set of ≥2 sibling
    block elements (div/li/article) that each look like a faculty card.
    A card is recognised by:
      • having ≤ 40 descendants (card-sized, not a section wrapper), AND
      • containing an <img> with any non-empty alt text, OR a heading
        (h2–h4/strong) with 2–6-word text (plausible person name).

    Headings inside the cards are the PREFERRED name source and are
    always present in Drupal views-row (Columbia) and NYU discover-list
    <li> layouts, so this check is reliably layout-agnostic.

    Returns a list of card elements, or [] when no card structure is found.
    """
    def _looks_like_card(el) -> bool:
        if not hasattr(el, "name") or el.name not in _CARD_TAGS:
            return False
        # Reject containers (section wrappers have many more descendants than cards)
        if len(el.find_all(True)) > 40:
            return False
        # Headshot present (any img with non-empty alt)
        if any(img.get("alt", "").strip() for img in el.find_all("img", limit=3)):
            return True
        # Heading/link with plausible 2–6-word person name text
        for tag in el.find_all(["h2", "h3", "h4", "strong", "a"], limit=8):
            words = tag.get_text(strip=True).split()
            if 2 <= len(words) <= 6:
                return True
        return False

    to_search = [section]
    for _ in range(3):
        next_level: list = []
        for parent in to_search:
            children = [c for c in parent.children if hasattr(c, "name") and c.name]
            card_children = [c for c in children if _looks_like_card(c)]
            if len(card_children) >= 2:
                logger.info(
                    "[detect_cards] %d cards inside <%s class=%s>",
                    len(card_children), parent.name, parent.get("class", []),
                )
                return card_children
            # Continue BFS into container children
            next_level.extend(c for c in children if c.name in _CONTAINER_TAGS)
        to_search = next_level

    return []


def _extract_card_entry(
    card: BeautifulSoup, base: str, seen: set[str]
) -> "dict | None":
    """
    Extract a {name, profile_url, full_profile_url} dict from one faculty card.

    Name extraction priority (most → least reliable):
      1. First <h2>/<h3>/<h4>/<strong> whose text is a valid person name.
         Handles Columbia (h2 inside <a>) and NYU (h4 sibling).
      2. Any <img alt> after stripping a leading "photo of …" prefix.
         Handles Columbia's second img and similar patterns.
      3. Last non-junk text segment of each <a> link (first to last).
         Handles cases where only a link carries the name.

    Profile URL: first href that is not a mailto/anchor/document file and
    has not already been seen.  Returns an empty string when no link exists
    so the entry is still emitted (Task 4 — no href dependency).  The
    downstream pipeline will mark href-less entries as invalid, but they
    are visible in extraction logs.
    """
    name: str = ""

    # ── 1. Heading / strong text ─────────────────────────────────────────
    for tag in card.find_all(["h2", "h3", "h4", "strong"]):
        raw = tag.get_text(strip=True)
        n, _ = _split_name_from_title(raw)
        if _is_valid_person_name(n):
            name = n
            break

    # ── 2. img alt (strip "photo of …" prefix) ──────────────────────────
    if not name:
        for img in card.find_all("img"):
            alt = _PHOTO_PREFIX_RE.sub("", img.get("alt", "")).strip()
            n, _ = _split_name_from_title(alt)
            if _is_valid_person_name(n):
                name = n
                break

    # ── 3. Last clean segment of any link text ───────────────────────────
    if not name:
        for a in card.find_all("a"):
            segments = [
                s.strip()
                for s in a.get_text(separator="|", strip=True).split("|")
                if s.strip() and "photo" not in s.lower()
            ]
            for seg in reversed(segments):
                n, _ = _split_name_from_title(seg)
                if _is_valid_person_name(n):
                    name = n
                    break
            if name:
                break

    if not name:
        return None

    # ── Profile link (optional) ──────────────────────────────────────────
    href = ""
    full_url = ""
    for a in card.find_all("a", href=True):
        h = a["href"].strip()
        if not h or h.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        candidate = urljoin(base, h) if not h.startswith("http") else h
        if candidate in seen:
            continue
        if urlparse(candidate).path.lower().endswith((".pdf", ".doc", ".docx")):
            continue
        href = h
        full_url = candidate
        break

    # ── Title extraction: scan card for a role-keyword-containing element ───
    title_text = ""
    for el in card.find_all(["p", "span", "div", "li"], recursive=True):
        el_text = el.get_text(strip=True)
        if not el_text or el_text == name or len(el_text) > 120:
            continue
        if (_ADVISOR_CORE_RE.search(el_text)
                or _ADVISOR_POSSIBLE_RE.search(el_text)
                or _ADVISOR_UNLIKELY_RE.search(el_text)):
            title_text = el_text
            break

    return {"name": name, "profile_url": href, "full_profile_url": full_url,
            "title": title_text}


def _extract_links_from_section(
    section: BeautifulSoup, base_url: str
) -> list[dict]:
    """
    Extract faculty entries from a detected section element.
    Returns list of {name, profile_url, full_profile_url}.
    profile_url / full_profile_url may be empty strings when a card
    has no outbound link.

    Strategy 1 — card-based (layout-agnostic):
      BFS-detect sibling block elements (div/li/article) that each
      contain a headshot or name-like text.  Handles Columbia Drupal
      views-row grids, NYU Shanghai discover-list <li> items, and any
      other card-based layout.  Does NOT depend on specific CSS classes.

    Strategy 2 — link-walk fallback:
      Walk every <a href> in the section.  Used when no card structure
      is found (plain bulleted lists, text-only faculty rosters).
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    seen: set[str] = set()
    entries: list[dict] = []

    # ── Strategy 1: card-based ────────────────────────────────────────────
    cards = _detect_faculty_cards(section)
    if cards:
        # Heuristic: if each non-empty "card" has >2 links on average, this is
        # a columnar layout (e.g. two-column panels each listing multiple faculty),
        # not a true faculty-card layout.  Fall through to link-walk in that case.
        links_per_card = [len(c.find_all("a", href=True)) for c in cards]
        nonempty_counts = [n for n in links_per_card if n > 0]
        avg_links = sum(nonempty_counts) / len(nonempty_counts) if nonempty_counts else 0
        use_cards = avg_links <= 2.0

        if use_cards:
            for card in cards:
                entry = _extract_card_entry(card, base, seen)
                if entry is None:
                    continue
                if entry.get("full_profile_url"):
                    seen.add(entry["full_profile_url"])
                entries.append(entry)
            if entries:
                return entries
        else:
            logger.info(
                "[extract_links] Card avg_links=%.1f > 2 — columnar layout detected, "
                "skipping card extraction, using link-walk",
                avg_links,
            )

    # ── Strategy 2: link-walk fallback ────────────────────────────────────
    # Tracks whether we've entered a publication sub-section so we can stop.
    in_pub_section = False
    PUB_HEADING_TAGS = frozenset({"h2", "h3", "h4", "h5"})

    for el in section.find_all(True):
        tag = el.name
        if not tag:
            continue

        # Detect publication sub-headings and mark the rest off-limits
        if tag in PUB_HEADING_TAGS:
            heading_text = el.get_text(strip=True)
            if _PUBLICATION_HEADING_RE.match(heading_text):
                in_pub_section = True
                logger.debug(
                    "[link_walk] Stopping at publication heading %r", heading_text
                )
            continue

        if tag != "a" or in_pub_section:
            continue

        href = el.get("href", "").strip()
        if not href or href.startswith(("mailto:", "javascript:", "#")):
            continue

        raw_text = el.get_text(separator=" ", strip=True)
        if not raw_text or len(raw_text) < 4:
            continue

        name, _ = _split_name_from_title(raw_text)
        if len(name.split()) < 2:
            continue

        full_url = urljoin(base, href) if not href.startswith("http") else href
        if full_url in seen:
            continue

        # Semantic filter: skip publications, tags, topic labels
        entry_type = _classify_entry_type(name, full_url)
        if entry_type in ("publication", "tag_category"):
            continue
        # group_topic entries are topic headings, not faculty — skip
        if entry_type == "group_topic":
            continue

        seen.add(full_url)

        # ── Title: look at sibling/parent text for a role keyword ────────────
        title_text = ""
        parent = el.parent
        if parent is not None:
            for sib in parent.children:
                if sib is el:
                    continue
                sib_text = getattr(sib, "get_text", lambda **kw: str(sib))(
                    strip=True
                ) if hasattr(sib, "get_text") else str(sib).strip()
                if not sib_text or sib_text == name or len(sib_text) > 120:
                    continue
                if (_ADVISOR_CORE_RE.search(sib_text)
                        or _ADVISOR_POSSIBLE_RE.search(sib_text)
                        or _ADVISOR_UNLIKELY_RE.search(sib_text)):
                    title_text = sib_text
                    break

        entries.append({
            "name": name,
            "profile_url": href,
            "full_profile_url": full_url,
            "title": title_text,
        })

    return entries


# ── Semantic content classification layer ────────────────────────────────── #

def _classify_entry_type(text: str, href: str = "") -> str:
    """
    Classify a single candidate entry.
    Returns: "faculty" | "group_topic" | "publication" | "tag_category" | "unknown"

    Used to filter out non-faculty entries (article titles, category tags,
    topic labels, DOI links) before they enter the faculty candidate list.
    """
    text = text.strip()
    if not text:
        return "unknown"

    # ── Non-profile URL signals ───────────────────────────────────────────────
    if href:
        # Publication databases — always a paper, never a person
        if any(d in href for d in [
            "doi.org", "arxiv.org", "ssrn.com", "pubmed", "ncbi.nlm.nih.gov",
            "ieeexplore.ieee.org", "dl.acm.org", "springer.com/article",
            "sciencedirect.com/science/article",
        ]):
            return "publication"
        # Research-area / topic page — e.g. /research/supply-chain-systems/
        # Excluded: URLs that also contain a person-profile segment
        _PROFILE_SEG = re.compile(r"/(people|faculty|profile|directory|staff|bio)/", re.I)
        if re.search(r"/research/", href, re.I) and not _PROFILE_SEG.search(href):
            return "group_topic"

    # ── Publication text signals ──────────────────────────────────────────────
    if _PUB_ENTRY_RE.search(text):
        return "publication"

    # Very long text → almost certainly a paper title, not a name
    if len(text) > 120:
        return "publication"

    # ── Single short token → category tag ────────────────────────────────────
    if len(text.split()) == 1 and len(text) < 25:
        return "tag_category"

    # ── Topic heading signals (checked BEFORE name validation) ───────────────
    # Person names don't contain prepositions/conjunctions as interior words,
    # and all words in a person name start with an uppercase letter.
    words = text.split()
    if len(words) >= 2:
        # Connector word in text → research topic phrase (e.g. "Freight and logistics",
        # "Compensation & Classification").  Use alternation because \b does not
        # surround & (a non-word character), so \b&\b never matches.
        if re.search(r'\b(?:and|or|of|for|in|with)\b|(?<!\w)&(?!\w)', text, re.IGNORECASE):
            return "group_topic"
        # Any non-first word starts lowercase → mixed-case topic (e.g. "Transportation safety")
        if any(w[0].islower() for w in words[1:] if w):
            return "group_topic"

    # ── Valid person name → faculty ───────────────────────────────────────────
    name, _ = _split_name_from_title(text)
    if _is_valid_person_name(name):
        return "faculty"

    # ── Multi-word non-name → topic/group heading ─────────────────────────────
    if 2 <= len(words) <= 8:
        return "group_topic"

    return "unknown"


def _classify_section_type(section: BeautifulSoup, heading_text: str) -> str:
    """
    Classify a detected content section.
    Returns one of:
      "faculty_section"                     — direct faculty list / cards
      "grouped_faculty_section"             — faculty under research-topic headings
      "interactive_grouped_faculty_section" — topic cards with "View Faculty" buttons
      "publication_section"                 — publications / papers block
      "mixed_section"                       — faculty + publications intermixed
      "unknown"

    Called after _find_faculty_section() returns a container, so heading_text
    is the matched heading that triggered the section find.
    """
    # ── 0. Canonical faculty headings — never misclassify regardless of content ─
    # Pages with explicit "Faculty", "Our Faculty", or "People" headings are
    # always faculty sections even when surrounding text contains paper references.
    _FACULTY_HEADING_RE = re.compile(
        r"^\s*(?:our\s+|the\s+)?(?:faculty|people|researchers?|members?)\s*$",
        re.IGNORECASE,
    )
    # Check the passed heading AND all headings inside the section.
    # The executor passes section.find("h2/h3/h4/strong/b") which may be the
    # first structural heading (e.g. "All Research"), not the Faculty sub-heading
    # that triggered the section find.  Scanning all avoids misclassifying a
    # section that contains an explicit "Faculty" heading further down.
    _is_canonical_faculty_heading = bool(
        _FACULTY_HEADING_RE.match(heading_text.strip())
        or any(
            _FACULTY_HEADING_RE.match(h.get_text(strip=True))
            for h in section.find_all(["h2", "h3", "h4", "h5"], limit=30)
        )
    )

    if not _is_canonical_faculty_heading:
        # ── 1. Publication heading → definite publication section ─────────────
        if _PUBLICATION_HEADING_RE.match(heading_text.strip()):
            return "publication_section"

    # ── 2. Interactive "View Faculty" triggers ────────────────────────────────
    interactive_hits = sum(
        1 for el in section.find_all(["a", "button"], limit=30)
        if _INTERACTIVE_FACULTY_BTN_RE.search(el.get_text(strip=True))
    )
    if interactive_hits >= 2:
        return "interactive_grouped_faculty_section"

    # ── 3. Publication signal count in section text ───────────────────────────
    section_text = section.get_text(separator=" ", strip=True)
    pub_hits = len(_PUB_ENTRY_RE.findall(section_text))

    # ── 4. Detect grouped faculty: purely structural check ───────────────────
    #
    # Approach: look for h3/h4/h5 headings that are each followed by a sibling
    # element containing ≥2 distinct faculty-name links.  We do NOT filter by
    # heading text (because "Machine Learning", "Data Science", etc. all pass
    # _is_valid_person_name — text-based classification is unreliable here).
    # The structural pattern (heading → list with multiple person-name links)
    # is what uniquely identifies grouped-faculty layout.
    topic_headings = 0
    for sh in section.find_all(["h3", "h4", "h5"], limit=30):
        sh_text = sh.get_text(strip=True)
        if not (1 <= len(sh_text.split()) <= 8):
            continue
        sib = sh.find_next_sibling()
        if sib is None:
            continue
        faculty_under = [
            a for a in sib.find_all("a", limit=8)
            if _is_valid_person_name(
                _split_name_from_title(a.get_text(strip=True))[0]
            )
        ]
        # Require ≥2 distinct person-name links to reduce false positives
        # (single-name sidebars, "see also" boxes, etc.)
        if len(faculty_under) >= 2:
            topic_headings += 1

    if topic_headings >= 2:
        return "mixed_section" if pub_hits >= 2 else "grouped_faculty_section"

    # ── 5. Strong publication signal count ────────────────────────────────────
    # Suppressed when heading is canonical ("Faculty", "People") — we trust the
    # heading over pub signals (the section may embed paper lists alongside names).
    if not _is_canonical_faculty_heading:
        if pub_hits >= 3:
            return "publication_section"
        if pub_hits >= 1:
            return "mixed_section"

    return "faculty_section"


def _extract_grouped_faculty(section: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract faculty from a section organized under research-topic headings.

    Walks the section in document order, tracking h3/h4/h5 headings that
    are NOT person names as group labels.  Faculty name links that follow
    a group heading are attributed to that group.

    Returns a deduplicated list — one entry per faculty member — with an
    optional "groups" list merging all group memberships across the section:

      {"name": "Jeff Ban", "profile_url": "...", "full_profile_url": "...",
       "groups": ["Freight and logistics", "Traffic operations"]}

    Falls back to empty list when no clear grouped structure is found
    (caller then tries _extract_links_from_section as fallback).
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    TOPIC_HEADING_TAGS = frozenset({"h3", "h4", "h5"})

    # ── Pass 1: identify confirmed group headings (structural, not text-based) ─
    #
    # A heading is a "group heading" if its next sibling element contains ≥1
    # faculty-name link.  We use id(element) as the identity key so we can
    # match during the second pass.  Text-based classification is intentionally
    # avoided here (see note in _classify_section_type step 4).
    group_heading_ids: set[int] = set()
    for sh in section.find_all(["h3", "h4", "h5"]):
        t = sh.get_text(strip=True)
        if not (1 <= len(t.split()) <= 8):
            continue
        sib = sh.find_next_sibling()
        if sib is None:
            continue
        has_faculty = any(
            _is_valid_person_name(_split_name_from_title(a.get_text(strip=True))[0])
            for a in sib.find_all("a", limit=8)
        )
        if has_faculty:
            group_heading_ids.add(id(sh))

    if not group_heading_ids:
        # No confirmed group structure — caller will fall back to regular extractor
        return []

    # ── Pass 2: walk in document order, track current group, collect faculty ──
    faculty_map: dict[str, dict] = {}
    current_group: str = ""

    for el in section.find_all(True):
        tag = el.name
        if not tag:
            continue

        # Confirmed group heading → update current group label
        if tag in TOPIC_HEADING_TAGS and id(el) in group_heading_ids:
            current_group = el.get_text(strip=True)
            continue

        if tag != "a":
            continue

        href = el.get("href", "").strip()
        if not href or href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue

        raw_text = el.get_text(separator=" ", strip=True)
        name, _ = _split_name_from_title(raw_text)
        if not _is_valid_person_name(name):
            continue
        if _classify_entry_type(name, href) == "publication":
            continue

        full_url = urljoin(base, href) if not href.startswith("http") else href
        key = full_url or name.lower()

        if key in faculty_map:
            grp_list = faculty_map[key]["groups"]
            if current_group and current_group not in grp_list:
                grp_list.append(current_group)
        else:
            faculty_map[key] = {
                "name": name,
                "profile_url": href,
                "full_profile_url": full_url,
                "groups": [current_group] if current_group else [],
            }

    result = list(faculty_map.values())
    logger.info(
        "[extract_grouped] %d unique faculty, %d confirmed group headings",
        len(result), len(group_heading_ids),
    )
    return result


def _detect_interactive_faculty_page(soup: BeautifulSoup) -> "dict | None":
    """
    Detect pages that show research-group cards with "View Faculty" / "See People"
    triggers but render no actual faculty names in the initial HTML
    (faculty appear only after a click / JS expansion).

    Returns {"groups": [{"label": ..., "href": ...}, ...]}
    when detected, or None otherwise.
    """
    triggers = []
    for el in soup.find_all(["a", "button"]):
        text = el.get_text(strip=True)
        if _INTERACTIVE_FACULTY_BTN_RE.search(text):
            href = el.get("href", "") if el.name == "a" else ""
            triggers.append({"label": text, "href": href})

    if len(triggers) < 2:
        return None

    # Confirm: very few actual faculty names visible in the initial HTML.
    # If faculty is already rendered, this is a normal page.
    # IMPORTANT: exclude links inside <nav>, <header>, <footer>, <aside> — these are
    # site navigation items that pass _is_valid_person_name but aren't faculty.
    _CHROME_TAGS = {"nav", "header", "footer", "aside"}
    visible_names = [
        a.get_text(strip=True)
        for a in soup.find_all("a", href=True)
        if not a.find_parent(_CHROME_TAGS)
        and _is_valid_person_name(
            _split_name_from_title(a.get_text(strip=True))[0]
        )
    ]
    if len(visible_names) > 5:
        return None   # faculty already visible — standard page, not interactive-only

    logger.info(
        "[interactive_detect] %d triggers found, %d visible names — marking interactive",
        len(triggers), len(visible_names),
    )
    return {"groups": triggers}


def _resolve_interactive_faculty(
    triggers: "list[dict]", base_url: str, page_class: str
) -> "list[dict]":
    """
    Attempt to resolve faculty by following the hrefs in "View Faculty" triggers.

    Fetches each linked sub-page and runs faculty extraction (Strategy 1 → 2)
    on it.  Returns deduplicated combined faculty list, or [] if nothing resolves.
    """
    parsed_base = urlparse(base_url)
    origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

    all_entries: list[dict] = []
    seen_keys: set[str] = set()
    seen_sub_urls: set[str] = set()

    _sub_profile_re = re.compile(
        r"/(people|faculty|staff|profile|directory)/[^/?#]+", re.I
    )

    for trigger in triggers:
        href = trigger.get("href", "")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        full_url = urljoin(origin, href) if not href.startswith("http") else href
        if full_url in seen_sub_urls:
            continue
        seen_sub_urls.add(full_url)

        try:
            r = _get(full_url, timeout=20)
            if r is None:
                continue
            sub_soup = BeautifulSoup(r.text, "lxml")

            # Try Strategy 1: faculty-section heading on the sub-page
            sub_section = _find_faculty_section(sub_soup)
            if sub_section:
                entries = _extract_links_from_section(sub_section, full_url)
            else:
                # Fallback: generic deep profile links
                sub_seen: set[str] = set()
                entries = []
                for a in sub_soup.find_all("a", href=True):
                    a_href = a["href"]
                    if not _sub_profile_re.search(a_href):
                        continue
                    raw = a.get_text(separator=" ", strip=True)
                    name, _ = _split_name_from_title(raw)
                    if len(name.split()) < 2:
                        continue
                    sub_full = (
                        urljoin(origin, a_href)
                        if not a_href.startswith("http") else a_href
                    )
                    if sub_full in sub_seen:
                        continue
                    sub_seen.add(sub_full)
                    entries.append({
                        "name": name,
                        "profile_url": a_href,
                        "full_profile_url": sub_full,
                    })

            for e in entries:
                key = e.get("full_profile_url") or e.get("name", "").lower()
                if key not in seen_keys:
                    seen_keys.add(key)
                    e["_page_class"] = page_class
                    all_entries.append(e)

        except Exception as exc:
            logger.warning("[resolve_interactive] %s: %s", full_url[:60], exc)

    logger.info(
        "[resolve_interactive] Resolved %d faculty from %d trigger(s)",
        len(all_entries), len(seen_sub_urls),
    )
    return all_entries


_MODAL_CLASS_RE = re.compile(r"\bmodal\b", re.I)


def _has_modal_faculty_containers(soup: BeautifulSoup) -> bool:
    """
    Return True if the page has modal/accordion containers that each hold
    a 'View X Faculty' trigger AND faculty links in the same static HTML block.
    Used by the planner to route to interactive_resolver (→ modal extraction).
    """
    for el in soup.find_all(True):
        classes = " ".join(el.get("class", []))
        if not _MODAL_CLASS_RE.search(classes):
            continue
        for a in el.find_all("a", href=True, limit=5):
            if _INTERACTIVE_FACULTY_BTN_RE.search(a.get_text(strip=True)):
                return True
    return False


def _extract_modal_faculty(soup: BeautifulSoup, url: str) -> list[dict]:
    """
    Extract faculty from WordPress Spectra modal blocks (and similar patterns)
    where "View X Faculty" triggers are accompanied by the faculty list in the
    same container in the static HTML (JS makes the content *visible* on click,
    but BeautifulSoup can read it directly).

    Pattern detected:
      <div class="...modal...">
        <a href="#">View ML & Perception Faculty</a>  ← trigger (skipped)
        <a href="https://...">Black</a>               ← faculty (kept)
        <a href="https://...">Bowman</a>
        ...
      </div>

    Single-word surnames are accepted as low-confidence entries
    (tagged _low_confidence=True) since they're identified by URL context.
    Multi-word names are accepted normally.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    faculty_map: dict[str, dict] = {}   # keyed by full_url or name
    found_any_modal = False

    for el in soup.find_all(True):
        if not _MODAL_CLASS_RE.search(" ".join(el.get("class", []))):
            continue
        # Only process divs that contain a "View X Faculty" trigger link
        trigger_text = ""
        for a in el.find_all("a", href=True, limit=5):
            if _INTERACTIVE_FACULTY_BTN_RE.search(a.get_text(strip=True)):
                trigger_text = a.get_text(strip=True)
                break
        if not trigger_text:
            continue

        found_any_modal = True
        group_label = trigger_text

        for a in el.find_all("a", href=True):
            text = a.get_text(strip=True)
            if _INTERACTIVE_FACULTY_BTN_RE.search(text):
                continue  # skip the trigger link itself
            href = a.get("href", "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url_a = urljoin(base, href) if not href.startswith("http") else href

            name, _ = _split_name_from_title(text)
            words = name.split()
            low_conf = len(words) < 2

            if low_conf:
                # Single-word entry — use URL slug as candidate name if possible
                if not name:
                    continue
            else:
                if not _is_valid_person_name(name):
                    continue
                if _classify_entry_type(name, full_url_a) in ("publication", "tag_category"):
                    continue

            key = full_url_a or name.lower()
            if key in faculty_map:
                grp_list = faculty_map[key]["groups"]
                if group_label and group_label not in grp_list:
                    grp_list.append(group_label)
            else:
                entry: dict = {
                    "name": name,
                    "profile_url": href,
                    "full_profile_url": full_url_a,
                    "groups": [group_label] if group_label else [],
                }
                if low_conf:
                    entry["_low_confidence"] = True
                faculty_map[key] = entry

    result = list(faculty_map.values())
    logger.info(
        "[modal_extractor] found_modal=%s → %d unique faculty entries",
        found_any_modal, len(result),
    )
    return result


def _classify_page_content(soup: BeautifulSoup, url: str) -> str:
    """
    Content-based page classifier.  Returns one of:
      'single_profile_page'               — one faculty member
      'directory_page'                    — classic full faculty directory (many links)
      'faculty_collection_page'           — non-directory page with embedded faculty section
      'interactive_grouped_faculty_page'  — group cards + "View Faculty" buttons, no faculty rendered
      'unknown'                           — none of the above

    This is the authoritative classifier used at routing time.  It intentionally
    ignores URL keywords so that program/research pages (e.g. Columbia /operations,
    NYU Shanghai /data-science-phd-program) are handled correctly.
    """
    parsed = urlparse(url)
    qs    = parsed.query.lower()
    path  = parsed.path.lower().rstrip("/")

    # ── 1. Explicit profile signals (URL-level, highest confidence) ───────────
    if "lname=" in qs and "fname=" in qs:
        return "single_profile_page"

    # Path ends in a known profile-depth pattern
    for kw in ["/directory/faculty", "/people/faculty", "/faculty/directory",
               "/directory/people", "/directory/staff"]:
        idx = path.find(kw)
        if idx < 0:
            continue
        after_segs = [s for s in path[idx + len(kw):].strip("/").split("/") if s]
        if len(after_segs) == 0:
            return "directory_page"
        return "single_profile_page"

    # ── 2. Count USC-style profile links (strong directory signal) ────────────
    usc_links = [a for a in soup.find_all("a", href=True)
                 if "lname=" in a["href"] and "fname=" in a["href"]]
    if len(usc_links) >= 5:
        return "directory_page"

    # ── 3. Count generic deep profile links ──────────────────────────────────
    _DEEP_PROFILE_RE = re.compile(
        r"/(faculty|people|directory|profile|person)/[^/?#]{2,}", re.I
    )
    deep_links: set[str] = set()
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if _DEEP_PROFILE_RE.search(h):
            deep_links.add(h)
    if len(deep_links) >= 10:
        return "directory_page"

    # ── 4. Faculty-section heading detection → collection page ───────────────
    section = _find_faculty_section(soup)
    if section:
        entries = _extract_links_from_section(section, url)
        if entries:
            return "faculty_collection_page"

    # ── 4.5 Interactive grouped faculty page ──────────────────────────────────
    # Fires only when no faculty was found above — avoids false-positives on
    # normal pages that happen to have a few "View All Faculty" nav links.
    interactive = _detect_interactive_faculty_page(soup)
    if interactive:
        return "interactive_grouped_faculty_page"

    # ── 5. Bare /faculty or /people root URL → directory ─────────────────────
    if re.search(r"/(faculty|people)/?$", path + "/") and not qs:
        return "directory_page"

    # ── 6. Single profile heuristic: page has one plausible person name in h1 ─
    h1 = soup.find("h1")
    if h1 and _is_valid_person_name(h1.get_text(strip=True)):
        return "single_profile_page"

    return "unknown"


def _detect_page_type(soup: BeautifulSoup, url: str) -> str:
    """
    Legacy wrapper kept for _fetch_one_profile compatibility.
    Maps the new _classify_page_content result back to the original labels:
      single_profile_page               → 'faculty_profile'
      directory_page                    → 'faculty_directory'
      faculty_collection_page           → 'faculty_directory'  (triggers batch mode)
      interactive_grouped_faculty_page  → 'faculty_directory'  (triggers batch mode)
      unknown                           → 'unknown'
    """
    cls = _classify_page_content(soup, url)
    mapping = {
        "single_profile_page":              "faculty_profile",
        "directory_page":                   "faculty_directory",
        "faculty_collection_page":          "faculty_directory",
        "interactive_grouped_faculty_page": "faculty_directory",
        "unknown":                          "unknown",
    }
    return mapping.get(cls, "unknown")


def _extract_main_content(soup: BeautifulSoup) -> str:
    """
    Try to isolate the main content area of the page.
    Falls back to full soup text if no container is found.
    """
    # Priority: named profile containers, then semantic HTML5
    for selector in [
        ".directory-profile-content",
        ".profile-content",
        ".faculty-profile",
        "#faculty-profile",
        ".usc-faculty-profile",
        ".profile-details",
        ".field-items",
        "main",
        "article",
        "#main-content",
        "#content",
        ".content-area",
        ".main-content",
        "#main",
        ".content",
    ]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 300:
                return text

    return soup.get_text(separator="\n", strip=True)


# Title keywords used to split "NameTitle" from link text.
# "Dean" intentionally excluded — it is a common surname (Matthew Dean, Howard Dean)
# and rarely appears as a standalone concatenated title in faculty-list links.
_TITLE_KEYWORDS = re.compile(
    r"\b(Professor|Associate|Assistant|Adjunct|Clinical|Research|Chair|Director|"
    r"Lecturer|Instructor|Fellow|Emeritus|Visiting|Faculty)\b"
)

# Trailing academic degree credentials to strip from names.
# Matches patterns like ", Ph.D.", " PhD", " Ph. D.", " M.S.", " Dr.", etc.
_CREDENTIAL_SUFFIX_RE = re.compile(
    r"[,\s]+(?:Ph\.?\s*D\.?|M\.?\s*[SAB]\.?|M\.D\.?|Dr\.?|P\.E\.?|"
    r"J\.D\.?|Ed\.D\.?|Sc\.D\.?|D\.Phil\.?|MBA?|LL[BMD]\.?)\s*$",
    # Note: NO re.IGNORECASE — credentials are uppercase abbreviations.
    # Case-insensitive matching would incorrectly strip surnames like "Ma"
    # (matching M.A.) or "Li" (matching LL.M.) etc.
)


# Detect "Last, First" or "Last, First Middle" name format.
# Must be exactly two comma-separated parts, first part 1–2 words (last name),
# second part 1–3 words (first + optional middle).  Both parts start with uppercase.
_LAST_FIRST_RE = re.compile(
    r"^([A-Z][A-Za-z\-']{1,}(?:\s+[A-Z][A-Za-z\-']{1,})?)"
    r",\s+"
    r"([A-Z][A-Za-z\-']{1,}(?:\s+[A-Z][A-Za-z\-']{1,}){0,2})"
    r"$"
)


def _split_name_from_title(raw_text: str) -> tuple[str, str]:
    """
    Faculty list links often have name+title concatenated, e.g.
    'Ali AbbasProfessor of ISE' → ('Ali Abbas', 'Professor of ISE')

    Also:
    - Strips trailing degree credentials such as ', Ph.D.' or ' M.S.'.
    - Normalises "Last, First" → "First Last" (UW ISE style).
    """
    m = _TITLE_KEYWORDS.search(raw_text)
    if m:
        name = raw_text[: m.start()].strip()
        title = raw_text[m.start():].strip()
    else:
        name = raw_text.strip()
        title = ""

    # Strip trailing degree credentials from the name portion
    name = _CREDENTIAL_SUFFIX_RE.sub("", name).strip().rstrip(",").strip()

    # Normalise "Last, First [Middle]" → "First [Middle] Last"
    lf = _LAST_FIRST_RE.match(name)
    if lf:
        last_part = lf.group(1).strip()
        first_part = lf.group(2).strip()
        name = f"{first_part} {last_part}"

    return name, title


def _extract_section(text: str, *keywords: str, max_chars: int = 600) -> str:
    """
    Find a section in page text that starts with one of the keywords
    and return the next ~max_chars of content.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if any(kw.lower() in stripped for kw in keywords):
            # Collect following non-empty lines
            snippet_lines = []
            total = 0
            for j in range(i + 1, min(i + 30, len(lines))):
                l = lines[j].strip()
                if not l:
                    continue
                # Stop at next section header (short line ending without punctuation)
                if len(l) < 40 and not l[-1] in ".,:;)" and j > i + 2:
                    break
                snippet_lines.append(l)
                total += len(l)
                if total >= max_chars:
                    break
            if snippet_lines:
                return " ".join(snippet_lines)
    return ""


# ── Photo-alt faculty extraction (Columbia Drupal pattern) ──────────────── #

def _has_photo_alt_faculty(soup: BeautifulSoup, min_count: int = 3) -> bool:
    """Return True if the page has ≥ min_count 'Photo of X' alt-text images."""
    imgs = soup.find_all("img", alt=lambda a: a and a.lower().startswith("photo of "))
    return len(imgs) >= min_count


def _has_name_alt_faculty(soup: BeautifulSoup, min_count: int = 5) -> bool:
    """
    Return True if the page has ≥ min_count images whose alt text looks
    directly like a person name (no 'Photo of' prefix, ≥ 2 words, passes
    _is_valid_person_name).  Georgia Tech ISyE style.
    """
    count = 0
    for img in soup.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if alt.lower().startswith("photo of "):
            continue   # handled by _has_photo_alt_faculty
        if _is_valid_person_name(alt):
            count += 1
            if count >= min_count:
                return True
    return False


# Prefixes to strip from image alt text before treating as a person name.
# e.g. "A headshot of Constance Crozier" → "Constance Crozier"
_ALT_PREFIX_RE = re.compile(
    r"^(?:a\s+)?(?:head\s*shot|photo|image|picture|screenshot|portrait|"
    r"recent\s+photo|thumbnail|profile\s+(?:pic|photo|picture|image))"
    r"(?:\s+of\s+(?:prof(?:essor)?\.?\s+|dr\.?\s+)?)?",
    re.IGNORECASE,
)
# Suffixes to strip: "'s headshot", "in a doctoral gown", " headshot", etc.
_ALT_SUFFIX_RE = re.compile(
    r"(?:'s\s+(?:head\s*shot|photo|image|picture)|"
    r"\s+in\s+a\s+\w+\s+gown|\s+head\s*shot|\s+photo|\s+smiling)"
    r"\s*$",
    re.IGNORECASE,
)
# Leading title to strip: "Dr. ", "Prof. ", "Professor "
_ALT_TITLE_PREFIX_RE = re.compile(
    r"^(?:dr\.?|prof(?:essor)?\.?)\s+", re.IGNORECASE
)


def _clean_name_from_alt(alt: str) -> str:
    """Strip headshot/image prefixes, trailing credentials/suffixes, and leading titles."""
    name = alt.strip()
    name = _ALT_PREFIX_RE.sub("", name).strip()
    name = _ALT_SUFFIX_RE.sub("", name).strip()
    name = _ALT_TITLE_PREFIX_RE.sub("", name).strip()
    name, _ = _split_name_from_title(name)   # strips ", Ph.D." etc.
    return name.strip()


def _extract_name_alt_faculty(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract faculty from images whose alt text IS (or contains) the person's
    full name — Georgia Tech ISyE pattern:
      <div class="card"><a href="/users/name"><img alt="First Last"/></a></div>

    Also handles common alt-text patterns like "A headshot of First Last",
    "Image of First Last", "First Last's headshot", "First Last in a doctoral gown".

    The profile URL is taken from the nearest ancestor <a> tag.
    Skips images already covered by _extract_photo_alt_faculty ('Photo of X').
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    entries: list[dict] = []
    seen: set[str] = set()

    for img in soup.find_all("img", alt=True):
        raw_alt = img.get("alt", "").strip()
        if not raw_alt:
            continue

        # Clean the alt text to extract the person name
        name = _clean_name_from_alt(raw_alt)

        if not name or not _is_valid_person_name(name):
            continue

        # Find nearest ancestor <a> with a non-trivial href
        href = ""
        a_tag = img.find_parent("a")
        if a_tag and a_tag.get("href"):
            href = a_tag["href"].strip()
            if href.startswith(("mailto:", "javascript:", "#")):
                href = ""

        full_url = urljoin(base, href) if href and not href.startswith("http") else href

        # Require a profile link: decorative images without a link are skipped.
        if not full_url:
            continue

        # Cross-check: derive candidate name from URL slug and compare to alt-name.
        # If there is NO word overlap between the two, the alt text is likely a
        # photo description (e.g. "Santa Monica photo" for /users/dmitrii-ostrovskii).
        # In that case, fall back to the URL-derived name.
        url_name = _name_from_url(full_url)
        if url_name and name:
            alt_words = {w.lower() for w in name.split() if len(w) > 2}
            url_words = {w.lower() for w in url_name.split() if len(w) > 2}
            if alt_words and url_words and not alt_words & url_words:
                logger.debug(
                    "[name_alt] Alt %r doesn't match URL-name %r — using URL name",
                    name, url_name,
                )
                name = url_name

        if not name or not _is_valid_person_name(name):
            continue

        key = full_url
        if key in seen:
            continue
        seen.add(key)

        # Ensure proper capitalisation (catches "akane fujimoto" → "Akane Fujimoto")
        if name == name.lower():
            name = name.title()

        # ── Title extraction: find role keyword text near the img ────────────
        # Walk up at most 5 levels. At each level, if the container text is
        # small enough (≤ 350 chars — one person's card, not many), search
        # recursively for a short element with a role keyword.
        title_text = ""
        container = img.parent
        for _depth in range(5):
            if container is None or not hasattr(container, "get_text"):
                break
            container_text = container.get_text(separator=" ", strip=True)
            # If container text is very long it spans multiple people — stop.
            if len(container_text) > 350:
                break
            for el in container.find_all(
                ["p", "span", "div", "small", "li", "h3", "h4", "h5"],
                recursive=True,
            ):
                if el is container:
                    continue
                raw = el.get_text(separator=" ", strip=True)
                if not raw or len(raw) > 120:
                    continue
                # Strip the person's name from the text before checking for roles
                check_text = raw.replace(name, "").strip()
                if not check_text:
                    continue
                if (_ADVISOR_CORE_RE.search(check_text)
                        or _ADVISOR_POSSIBLE_RE.search(check_text)
                        or _ADVISOR_UNLIKELY_RE.search(check_text)):
                    title_text = check_text[:80]
                    break
            if title_text:
                break
            container = container.parent

        logger.debug("[name_alt] %r → %r (href=%s title=%r)", raw_alt, name, href, title_text)
        entries.append({
            "name": name,
            "profile_url": href,
            "full_profile_url": full_url,
            "title": title_text,
        })

    logger.info("[name_alt] Extracted %d faculty entries from name-as-alt images", len(entries))
    return entries


def _extract_photo_alt_faculty(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract faculty from img[alt^='Photo of '] pattern.

    Common in Columbia/Drupal sites where each faculty card contains:
      <a href="/content/name"><img alt="Photo of First Last"/><h2>First Last</h2></a>

    The profile URL is taken from the nearest ancestor <a> tag.
    Returns list of {name, profile_url, full_profile_url}.
    """
    from urllib.parse import urljoin as _urljoin
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    entries: list[dict] = []
    seen: set[str] = set()

    imgs = soup.find_all("img", alt=lambda a: a and a.lower().startswith("photo of "))
    for img in imgs:
        raw_name = img["alt"][len("Photo of "):].strip()
        if not _is_valid_person_name(raw_name):
            continue

        # Find the nearest ancestor <a> — profile link
        href = ""
        a_tag = img.find_parent("a")
        if a_tag and a_tag.get("href"):
            href = a_tag["href"].strip()
            if href.startswith(("mailto:", "javascript:", "#")):
                href = ""

        full_url = _urljoin(base, href) if href and not href.startswith("http") else href
        if full_url and full_url in seen:
            continue
        if full_url:
            seen.add(full_url)

        logger.debug("[photo_alt] Extracted %r from alt text (href=%s)", raw_name, href)
        entries.append({
            "name": raw_name,
            "profile_url": href,
            "full_profile_url": full_url,
        })

    logger.info("[photo_alt] Extracted %d faculty entries from Photo-of alt text", len(entries))
    return entries


def _extract_cbs_faculty(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract faculty from Columbia Business School m-listing-faculty card pattern.

    Structure:
      <div class="m-listing-faculty">
        <p class="m-listing-faculty__title">First Last</p>
        <a class="m-listing-faculty__link" href="/faculty/people/first-last"></a>
      </div>

    The anchor text is empty; name comes from the sibling <p> tag.
    """
    from urllib.parse import urljoin as _urljoin
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    entries: list[dict] = []
    seen: set[str] = set()

    cards = soup.find_all("div", class_="m-listing-faculty")
    for card in cards:
        name_tag = card.find("p", class_="m-listing-faculty__title")
        if not name_tag:
            continue
        raw_name = name_tag.get_text(strip=True)
        if not _is_valid_person_name(raw_name):
            continue

        link_tag = card.find("a", class_="m-listing-faculty__link")
        href = ""
        if link_tag and link_tag.get("href"):
            href = link_tag["href"].strip()
            if href.startswith(("mailto:", "javascript:", "#")):
                href = ""

        full_url = _urljoin(base, href) if href and not href.startswith("http") else href
        if full_url and full_url in seen:
            continue
        if full_url:
            seen.add(full_url)

        title_tag = card.find("dt", class_="m-detail-meta__item-title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        logger.debug("[cbs] Extracted %r (href=%s)", raw_name, href)
        entries.append({
            "name":             raw_name,
            "profile_url":      href,
            "full_profile_url": full_url,
            "title":            title,
        })

    logger.info("[cbs] Extracted %d faculty entries from m-listing-faculty cards", len(entries))
    return entries


_FEW_SHOT_DIR = Path.home() / ".kahuna" / "knowledge" / "few_shots"


def _few_shot_save(
    url: str,
    profile_blocks: list[str],
    named_links: list[str],
    img_alts: list[str],
    qparam_links: list[str],
    entries: list[dict],
) -> None:
    """Persist a clean extraction as a few-shot example for future prompts.

    One file per domain — overwrites the previous entry so the library stays
    bounded (one entry per domain, always the most recent clean result).
    """
    _FEW_SHOT_DIR.mkdir(parents=True, exist_ok=True)
    import datetime
    from urllib.parse import urlparse as _up
    domain = _up(url).netloc.replace(".", "-").replace(":", "-")
    record = {
        "url": url,
        "structure": {
            "pb": len(profile_blocks),
            "nl": len(named_links),
            "ia": len(img_alts),
            "qp": len(qparam_links),
        },
        "sources_preview": (profile_blocks or named_links or img_alts or qparam_links)[:5],
        "output_preview":  [{"name": e["name"], "profile_url": e["profile_url"]} for e in entries[:5]],
        "total_count": len(entries),
        "timestamp": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }
    try:
        # One file per domain — overwrite to prevent unbounded growth
        (_FEW_SHOT_DIR / f"{domain}.json").write_text(
            __import__("json").dumps(record, indent=2)
        )
        logger.info("[few_shot] Saved example: url=%s count=%d", url, len(entries))
    except Exception as exc:
        logger.warning("[few_shot] Save failed: %s", exc)


def _few_shot_load(pb: int, nl: int, ia: int, qp: int) -> str | None:
    """
    Find the best-matching saved few-shot example for the current page structure.
    Returns a formatted prompt section, or None if no good match found.
    """
    import json as _json
    _FEW_SHOT_DIR.mkdir(parents=True, exist_ok=True)
    best_score = 0.0
    best_rec: dict | None = None

    for f in _FEW_SHOT_DIR.glob("*.json"):
        try:
            rec = _json.loads(f.read_text())
        except Exception:
            continue
        s = rec.get("structure", {})
        rpb, rnl, ria, rqp = s.get("pb", 0), s.get("nl", 0), s.get("ia", 0), s.get("qp", 0)

        # Cosine-style similarity on the 4 source counts
        def _sim(a: int, b: int) -> float:
            if a == 0 and b == 0:
                return 1.0
            if max(a, b) == 0:
                return 1.0
            return min(a, b) / max(a, b)

        score = (_sim(pb, rpb) * 0.5 + _sim(nl, rnl) * 0.3
                 + _sim(ia, ria) * 0.1 + _sim(qp, rqp) * 0.1)

        if score > best_score:
            best_score = score
            best_rec = rec

    if best_score < 0.6 or best_rec is None:
        return None

    src_lines = "\n".join(f"  {s}" for s in best_rec.get("sources_preview", []))
    out_lines = _json.dumps(best_rec.get("output_preview", []), ensure_ascii=False)
    total = best_rec.get("total_count", "?")
    return (
        f"=== VERIFIED EXAMPLE (similarity={best_score:.2f}, {total} faculty total) ===\n"
        f"Sources looked like:\n{src_lines}\n"
        f"Correct output (first entries shown):\n{out_lines}\n"
        f"Apply the same extraction pattern to the sources below.\n"
        f"==="
    )


def _extract_llm_faculty(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    LLM-based faculty extractor (Haiku). Primary strategy for all page types.

    Six-source strategy — sources ordered from most to least reliable:
      1. Query-param links: ?fname=X&lname=Y — name extracted directly from URL.
         Covers USC-style directory pages (previously handled by usc_extractor).
      2. Profile-URL blocks: links whose path matches /faculty|people|profile|...
         Walk up DOM to grab card container text. Covers empty-text anchors (CBS).
      3. Img alt text: "photo of First Last" and alt-is-name patterns.
         Covers Columbia/Drupal and Georgia Tech conventions.
      4. Named links: <a> text that passes _is_valid_person_name.
         Classic "name in link text" pattern.
      5. Hidden/modal content: links inside CSS-hidden modal or accordion divs.
         Covers NYU CDS UAGb popup blocks and similar.
      6. Page sample: first 1500 chars of body text for context.

    Token budget: ~5-20k input tokens; max_tokens=16384 output (150+ faculty).
    At Haiku pricing: ~$0.001-0.005 per call.
    """
    import anthropic
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parsed = urlparse(base_url)
    # Use full page URL as urljoin base so that relative hrefs like "../faculty/x.html"
    # resolve correctly relative to the page's directory, not just the origin.
    base   = base_url

    _PROFILE_PATH_RE = re.compile(
        r"/(people|faculty|profile|person|staff|researchers?|academics?|content)/",
        re.IGNORECASE,
    )
    # Generic content paths (like /content/first-last) need extra filtering:
    # only accept them if they don't look like non-person pages
    _NON_PERSON_CONTENT_RE = re.compile(
        r"/(admissions|copyright|privacy|nondiscrimination|courses?|events?|"
        r"news|programs?|departments?|centers?|labs?|open-positions|"
        r"job-market|undergraduate|graduate|masters?|phd|careers?)/",
        re.IGNORECASE,
    )

    # ── Resolve all hrefs to absolute URLs in-place ──────────────────────────
    # This ensures the LLM sees absolute URLs in the HTML it processes, so
    # relative paths like "../faculty/x.html" don't get mis-resolved.
    for _a in soup.find_all("a", href=True):
        _h = _a["href"].strip()
        if not _h or _h.startswith(("mailto:", "javascript:", "#", "http")):
            continue
        _a["href"] = urljoin(base, _h)

    # ── Collect all page hrefs for post-LLM URL validation ───────────────────
    # After LLM responds we cross-check returned URLs against this set to catch
    # hallucinated links that were never on the page.
    # We prefer source_urls_norm (URLs from extracted faculty sources) over
    # page_hrefs_norm (every URL on the page) because the LLM may assign
    # hallucinated names to real faculty URLs found in page navigation.
    def _norm_url(u: str) -> str:
        return u.rstrip("/").lower()

    page_hrefs_norm: set[str] = set()
    for _a in soup.find_all("a", href=True):
        _h = _a["href"].strip()
        if _h.startswith(("mailto:", "javascript:", "#")):
            continue
        page_hrefs_norm.add(_norm_url(_h))

    # ── Source 1: profile-URL blocks ─────────────────────────────────────────
    # For each link with a faculty-profile URL, walk up the DOM (up to 5 levels)
    # to find a container whose text is short enough to be a card (< 400 chars)
    # but long enough to contain a name (> 5 chars).
    profile_blocks: list[str] = []
    seen_profile_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not _PROFILE_PATH_RE.search(href):
            continue
        if href.startswith(("mailto:", "javascript:", "#")):
            continue
        # For generic /content/ paths, skip non-person pages (admissions, courses…)
        if "/content/" in href.lower() and _NON_PERSON_CONTENT_RE.search(href):
            continue
        full = urljoin(base, href) if not href.startswith("http") else href
        if full in seen_profile_urls:
            continue
        seen_profile_urls.add(full)

        # Prefer anchor text if it's already a valid person name — avoids
        # multi-name context blobs where the container holds several faculty.
        anchor_text = a.get_text(separator=" ", strip=True)
        if _is_valid_person_name(anchor_text):
            ctx = anchor_text
        else:
            # Walk up DOM collecting candidate containers; pick the smallest that is
            # > 8 chars (contains a real name) and < 300 chars (not a nav/section blob).
            node = a
            ctx  = ""
            for _ in range(7):
                if node.parent is None:
                    break
                node = node.parent
                candidate = node.get_text(separator=" ", strip=True)
                if 8 < len(candidate) < 300:
                    ctx = candidate   # keep updating — want smallest valid container
                elif len(candidate) >= 300:
                    break             # container too large → stop walking up

        profile_blocks.append(f"{ctx or '(no context)'} | {full}")

    # ── Source 2: query-param profile links (USC ?lname=X&fname=Y pattern) ──────
    # USC-style pages embed first/last name directly in the query string.
    # We extract the name from the URL params and pair it with the full URL so
    # the LLM sees clean "First Last | https://..." entries.
    qparam_links: list[str] = []
    seen_qparam: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        href_lower = href.lower()
        if "lname=" not in href_lower or "fname=" not in href_lower:
            continue
        full = urljoin(base, href) if not href.startswith("http") else href
        if full in seen_qparam:
            continue
        seen_qparam.add(full)
        name = _name_from_url(href)   # parses lname/fname params → "First Last"
        if not name:
            # Fall back to anchor text
            name = a.get_text(separator=" ", strip=True)
            name, _ = _split_name_from_title(name) if name else ("", "")
        if name:
            qparam_links.append(f"{name} | {full}")

    # ── Source 3: named links ─────────────────────────────────────────────────
    named_links: list[str] = []
    seen_named: set[str] = set()
    for a in soup.find_all("a", href=True):
        text = a.get_text(separator=" ", strip=True)
        href = a["href"].strip()
        if not text or href.startswith(("mailto:", "javascript:", "#")):
            continue
        if not _is_valid_person_name(text):
            continue
        full = urljoin(base, href) if not href.startswith("http") else href
        if full in seen_named:
            continue
        seen_named.add(full)
        named_links.append(f"{text} | {full}")

    # ── Source 3: img alt text (photo-of / role-prefix / alt-is-name patterns) ──
    # Strips role/title prefixes that precede the actual person name in alt text:
    #   "photo of First Last"          → Columbia/Drupal pattern
    #   "Faculty Member First Last"    → UFL ISE pattern
    #   "Professor First Last"         → generic title prefix
    #   "Headshot of First Last"       → headshot pattern
    # After stripping, the remainder is checked as a person name.
    _ALT_ROLE_PREFIX_RE = re.compile(
        r"^(?:"
        r"photo(?:\s+of)?|"
        r"headshot(?:\s+of)?|"
        r"portrait(?:\s+of)?|"
        r"image(?:\s+of)?|"
        r"(?:faculty|staff|research|clinical|adjunct|visiting|emeritus|emerita|affiliate)"
        r"(?:\s+member)?|"
        r"(?:professor|prof\.?|dr\.?|associate\s+professor|assistant\s+professor)"
        r")\s+",
        re.IGNORECASE,
    )
    img_alts: list[str] = []
    seen_img_hrefs: set[str] = set()
    for img in soup.find_all("img", alt=True):
        raw_alt = img.get("alt", "").strip()
        if not raw_alt:
            continue
        # Strip role/title prefix to isolate the person name
        candidate = _ALT_ROLE_PREFIX_RE.sub("", raw_alt).strip()
        if not _is_valid_person_name(candidate):
            continue
        # Find nearest parent <a> for profile URL
        a_parent = img.find_parent("a")
        href = ""
        if a_parent and a_parent.get("href"):
            href = a_parent["href"].strip()
            if href.startswith(("mailto:", "javascript:", "#")):
                href = ""
        full = urljoin(base, href) if href and not href.startswith("http") else href
        if full and full in seen_img_hrefs:
            continue
        if full:
            seen_img_hrefs.add(full)
        img_alts.append(f"{candidate} | {full}" if full else candidate)

    # ── Source 4: hidden/modal section content ───────────────────────────────────
    # Some pages (e.g. NYU CDS UAGb popups) embed faculty in CSS-hidden divs that are
    # never visible until a button is clicked.  We extract ALL links from these sections
    # so the LLM can see them even though they are hidden from the normal view.
    # Common hidden-content class keywords: modal, popup, collapsed, accordion, hidden.
    _HIDDEN_SECTION_RE = re.compile(
        r"\b(modal|popup|pop-up|collapsed|accordion|toggle|hidden|"
        r"offscreen|off-canvas|drawer|flyout)\b",
        re.IGNORECASE,
    )
    hidden_links: list[str] = []
    seen_hidden: set[str] = set()
    for el in soup.find_all(True):
        cls = " ".join(el.get("class") or [])
        if not _HIDDEN_SECTION_RE.search(cls):
            continue
        for a in el.find_all("a", href=True):
            text = a.get_text(separator=" ", strip=True)
            href = a["href"].strip()
            if not text or href.startswith(("mailto:", "javascript:", "#")):
                continue
            full = urljoin(base, href) if not href.startswith("http") else href
            if full in seen_hidden:
                continue
            seen_hidden.add(full)
            hidden_links.append(f"{text} | {full}")

    # ── Build source_urls_norm: URLs from detected faculty signals only ───────────
    # This is used for post-LLM URL validation instead of all page hrefs.
    # The LLM may assign hallucinated names to real faculty URLs that appear in
    # page navigation; restricting to source-derived URLs prevents false passes.
    source_urls_norm: set[str] = set()
    for _src in (*profile_blocks, *named_links, *img_alts, *hidden_links, *qparam_links):
        if " | " in _src:
            _u = _src.rsplit(" | ", 1)[1].strip()
            if _u:
                source_urls_norm.add(_norm_url(_u))
    # Fall back to all page hrefs if we found no structural signals at all
    _validate_urls_against = source_urls_norm if source_urls_norm else page_hrefs_norm

    # ── Source 5: brief page sample ───────────────────────────────────────────
    # Build full page text (lowercased) for post-LLM name validation.
    # Used to reject empty-URL entries whose name doesn't appear anywhere on the page.
    # Must be built BEFORE soup.decompose() strips scripts/styles.
    page_text_lower = soup.get_text(separator=" ", strip=True).lower()

    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    page_sample = soup.get_text(separator=" ", strip=True)[:1500]

    title_tag  = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""

    # Build prompt sections (only include non-empty sources).
    # Sources are ordered from most-reliable to least-reliable so the LLM
    # encounters the highest-signal data first.
    sections: list[str] = [f"Page: {base_url}\nTitle: {title_text}"]

    # Inject a verified few-shot example if we have one with similar page structure.
    _few_shot_hint = _few_shot_load(len(profile_blocks), len(named_links), len(img_alts), len(qparam_links))
    if _few_shot_hint:
        sections.append(_few_shot_hint)
        logger.info("[few_shot] Injected example into prompt")
    if qparam_links:
        sections.append(
            "=== QUERY-PARAM PROFILE LINKS (name extracted from URL params | profile URL) ===\n"
            "These are the most reliable: name comes directly from ?fname=...&lname=... params.\n"
            + "\n".join(qparam_links[:300])
        )
    if profile_blocks:
        sections.append(
            "=== FACULTY PROFILE LINKS (context text | profile URL) ===\n"
            + "\n".join(profile_blocks[:300])
        )
    if img_alts:
        sections.append(
            "=== IMAGE ALT TEXT (faculty photo captions, name | profile URL) ===\n"
            + "\n".join(img_alts[:200])
        )
    if named_links:
        sections.append(
            "=== NAMED LINKS (link text | URL) ===\n"
            + "\n".join(named_links[:200])
        )
    if hidden_links:
        sections.append(
            "=== HIDDEN/MODAL SECTION LINKS (links inside collapsed or popup sections) ===\n"
            + "\n".join(hidden_links[:300])
        )
    sections.append(f"=== PAGE SAMPLE ===\n{page_sample}")

    prompt = "\n\n".join(sections) + """

Task: extract all faculty members (professors, researchers, lecturers) from the sources above.

Return ONLY a JSON array, no explanation:
[{"name": "First Last", "profile_url": "https://..."}]

Rules:
- QUERY-PARAM PROFILE LINKS are the most reliable — use all of them
- IMAGE ALT TEXT entries are pre-cleaned — use all of them
- FACULTY PROFILE LINKS context text usually starts with the person's name
- HIDDEN/MODAL SECTION LINKS may contain last-name-only entries — skip them unless
  a matching full name appears in another source
- Strip job titles: "Professor Jane Smith" → "Jane Smith"
- profile_url must be a full URL from the sources above (not invented)
- Skip navigation, department pages, courses, events, admin, students"""

    try:
        from mcp_servers.llm_client import make_client
        client = make_client()
        msg = client.messages.create(
            max_tokens=16384,  # 150 faculty × ~56 tokens/entry ≈ 8 400 tokens; 16 384 gives 2× headroom
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            logger.warning("[llm_extractor] No JSON array in response")
            return []
        data = json.loads(m.group())
    except Exception as exc:
        logger.warning("[llm_extractor] API/parse error: %s", exc)
        return []

    entries: list[dict] = []
    seen_names: set[str] = set()
    hallucinated = 0
    for item in data:
        name = (item.get("name") or "").strip()
        purl = (item.get("profile_url") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        if not _is_valid_person_name(name):
            continue
        seen_names.add(name.lower())
        full = urljoin(base, purl) if purl and not purl.startswith("http") else purl
        if full:
            # LLM provided a URL — verify it came from a detected faculty source,
            # not just any link on the page (e.g. navigation links).
            if _norm_url(full) not in _validate_urls_against:
                logger.debug("[llm_extractor] Dropping hallucinated entry %r (URL %s not in faculty sources)", name, full)
                hallucinated += 1
                continue
        else:
            # LLM gave no URL — verify the surname appears in the page text.
            # Surnames are distinctive enough; first names appear too broadly.
            words = name.split()
            last  = words[-1].lower() if words else ""
            if last and last not in page_text_lower:
                logger.debug("[llm_extractor] Dropping hallucinated no-URL entry %r (surname %r not in page)", name, last)
                hallucinated += 1
                continue
        entries.append({
            "name":             name,
            "profile_url":      purl,
            "full_profile_url": full,
        })

    if hallucinated:
        logger.warning("[llm_extractor] Dropped %d hallucinated entries", hallucinated)
    logger.info("[llm_extractor] Extracted %d faculty entries via LLM", len(entries))

    # Save as few-shot example when result is clean (no hallucinations, ≥5 faculty).
    # These examples are injected into future prompts for structurally similar pages.
    if hallucinated == 0 and len(entries) >= 5:
        _few_shot_save(base_url, profile_blocks, named_links, img_alts, qparam_links, entries)

    return entries


# ── Role classification ──────────────────────────────────────────────────── #

# Words in a person's title that indicate their role category.
_ROLE_STUDENT_RE = re.compile(
    r"\b(ph\.?\s*d\.?\s*student|doctoral\s+student|graduate\s+student|"
    r"phd\s+candidate|ms\s+student|master'?s?\s+student|undergraduate|"
    r"postdoc(?:toral)?|post-doc)\b",
    re.IGNORECASE,
)
_ROLE_STAFF_RE = re.compile(
    r"\b(administrator|program\s+coordinator|office\s+manager|"
    r"academic\s+advisor|department\s+manager|staff)\b",
    re.IGNORECASE,
)
_ROLE_LECTURER_RE = re.compile(
    r"\b(lecturer|instructor|teaching\s+professor|teaching\s+faculty)\b",
    re.IGNORECASE,
)
_ROLE_RESEARCHER_RE = re.compile(
    r"\b(research\s+(scientist|engineer|professor|associate)|"
    r"senior\s+researcher|principal\s+researcher|research\s+fellow)\b",
    re.IGNORECASE,
)
_ROLE_AFFILIATED_RE = re.compile(
    r"\b(adjunct|affiliated|associated|visiting|courtesy|emeritus|emerita)\b",
    re.IGNORECASE,
)


def _classify_person_role(name: str, title: str = "", href: str = "") -> str:
    """
    Classify a person entry into a role category.
    Returns one of: faculty_core | faculty_affiliated | researcher |
                    lecturer | student | staff | unknown

    Uses title text (if available) and URL signals.
    Conservative: defaults to faculty_core when signals are ambiguous.
    """
    combined = f"{title} {href}".lower()
    name_lower = name.lower()

    # Student signals (strong — both name context and title)
    if _ROLE_STUDENT_RE.search(combined):
        return "student"
    # Staff
    if _ROLE_STAFF_RE.search(combined):
        return "staff"
    # Lecturer / instructor
    if _ROLE_LECTURER_RE.search(combined):
        return "lecturer"
    # Research scientist / engineer (not faculty)
    if _ROLE_RESEARCHER_RE.search(combined):
        return "researcher"
    # Affiliated / adjunct / visiting / emeritus
    if _ROLE_AFFILIATED_RE.search(combined):
        return "faculty_affiliated"

    return "faculty_core"


# ── Advisor-eligibility classifier ────────────────────────────────────────────

# Core: tenured / tenure-track positions that regularly supervise PhD students
_ADVISOR_CORE_RE = re.compile(
    r"\b("
    r"assistant\s+professor|"
    r"associate\s+professor|"
    r"full\s+professor|"
    r"distinguished\s+professor|"
    r"presidential\s+professor|"
    r"endowed\s+professor|"
    r"chair(?:ed)?\s+professor|"
    r"tenure.track|"
    r"tenured?\s+(?:faculty|professor)|"
    # plain "Professor" only — must NOT be followed by "of Practice"
    r"professor(?!\s+of\s+practice)(?!\s+emeritus)(?!\s+emerita)"
    r")\b",
    re.IGNORECASE,
)

# Possible: can sometimes advise but not tenure-track by default
_ADVISOR_POSSIBLE_RE = re.compile(
    r"\b("
    r"research\s+professor|"
    r"research\s+faculty|"
    r"professor\s+of\s+practice|"
    r"adjunct\s+(?:professor|faculty|associate|instructor)|"
    r"affiliate[d]?\s+(?:professor|faculty)|"
    r"joint\s+(?:professor|faculty|appointment)|"
    r"visiting\s+(?:professor|scholar|faculty|associate)|"
    r"courtesy\s+(?:professor|faculty)|"
    r"clinical\s+(?:professor|faculty)|"
    r"research\s+scientist|"
    r"principal\s+(?:research\s+)?scientist|"
    r"senior\s+(?:research\s+)?scientist"
    r")\b",
    re.IGNORECASE,
)

# Unlikely: not typically primary PhD supervisors
_ADVISOR_UNLIKELY_RE = re.compile(
    r"\b("
    r"lecturer|"
    r"teaching\s+(?:professor|faculty|fellow|assistant)|"
    r"postdoc(?:toral)?(?:\s+(?:fellow|researcher|associate|scholar))?|"
    r"postdoc|"
    r"emeritus|emerita|"
    r"retired|"
    r"ph\.?d\.?\s+(?:student|candidate)|"
    r"doctoral\s+(?:student|candidate)|"
    r"graduate\s+student|"
    r"ms\s+student|master.?s?\s+student|"
    r"undergraduate|"
    r"alumni|alumnus|alumna|"
    r"staff|administrative|coordinator|"
    r"program\s+(?:director|manager|coordinator)|"
    r"instructor(?!\s+professor)"  # "instructor" alone, but not "instructor professor"
    r")\b",
    re.IGNORECASE,
)


def classify_advisor_eligibility(
    title_text: str,
    context_text: str = "",
) -> dict:
    """
    Classify whether a person is eligible to supervise PhD students.

    Args:
        title_text:   The person's title or role string (from extraction or profile).
        context_text: Additional text context (e.g. surrounding card text, href).

    Returns:
        {
            "role_type":           str,   # e.g. "Assistant Professor"
            "advisor_eligibility": str,   # "core" | "possible" | "unlikely"
            "reason":              str,   # matched signal / default explanation
        }
    """
    combined = f"{title_text} {context_text}".strip()

    # Order: unlikely → possible → core.
    # Unlikely and possible patterns are MORE specific (e.g. "Visiting Professor",
    # "Emeritus Professor") and must fire BEFORE the generic "professor" core catch
    # to avoid misclassifying e.g. Visiting Professor as core.

    # ── Unlikely (postdoc, emeritus, lecturer, student, staff — definitive) ──
    m = _ADVISOR_UNLIKELY_RE.search(combined)
    if m:
        matched = m.group(0).strip().title()
        return {
            "role_type": matched,
            "advisor_eligibility": "unlikely",
            "reason": matched,
        }

    # ── Possible (visiting, adjunct, affiliate, research professor…) ──────────
    m = _ADVISOR_POSSIBLE_RE.search(combined)
    if m:
        matched = m.group(0).strip().title()
        return {
            "role_type": matched,
            "advisor_eligibility": "possible",
            "reason": matched,
        }

    # ── Core (tenure-track and tenured faculty) ──────────────────────────────
    m = _ADVISOR_CORE_RE.search(combined)
    if m:
        matched = m.group(0).strip().title()
        return {
            "role_type": matched,
            "advisor_eligibility": "core",
            "reason": matched,
        }

    # ── Default: no title info → treat as core (conservative) ───────────────
    return {
        "role_type": "unknown",
        "advisor_eligibility": "core",
        "reason": "no title signal — defaulting to core",
    }


# ── External homepage verification ───────────────────────────────────────────

# Positive: academic role words appearing on the target page
_EXT_ROLE_RE = re.compile(
    r"\b(professor|assistant\s+professor|associate\s+professor|full\s+professor|"
    r"faculty|researcher|principal\s+investigator|\bPI\b|"
    r"postdoctoral|postdoc|lecturer|instructor|scientist|fellow)\b",
    re.IGNORECASE,
)

# Positive: page has a section/link that personal homepages typically have
_EXT_ACADEMIC_SECTION_RE = re.compile(
    r"\b(publications?|research|teaching|lab|group|"
    r"google\s+scholar|cv|curriculum\s+vitae|biography|bio)\b",
    re.IGNORECASE,
)

# Negative: signals that the page is NOT a personal faculty homepage
_EXT_NEGATIVE_RE = re.compile(
    r"\b(company|corporation|inc\.|llc|stock|nasdaq|nyse|"
    r"news\s+article|press\s+release|tweet|twitter|instagram|"
    r"buy\s+now|shop\s+now|add\s+to\s+cart|pricing)\b",
    re.IGNORECASE,
)

# Negative URL signals (before even fetching)
_EXT_NEGATIVE_URL_RE = re.compile(
    r"twitter\.com|linkedin\.com/in/|facebook\.com|instagram\.com|"
    r"youtube\.com|reddit\.com|doi\.org|arxiv\.org|"
    r"\.pdf$|/pdf/",
    re.IGNORECASE,
)

# Cache: url → verification result dict (process-lifetime)
_EXT_VERIFY_CACHE: dict[str, dict] = {}


def verify_external_person_homepage(
    url: str,
    expected_name: str,
    expected_institution: str | None = None,
) -> dict:
    """
    Heuristically verify whether a URL is a personal academic homepage.

    Returns:
        {
            "is_person_homepage": bool,
            "confidence": "high" | "medium" | "low",
            "signals": [list of str signal names found],
        }

    Only deterministic text signals — no LLM calls.
    Uses a process-level cache keyed by URL.
    """
    if url in _EXT_VERIFY_CACHE:
        return _EXT_VERIFY_CACHE[url]

    def _reject(reason: str) -> dict:
        result = {"is_person_homepage": False, "confidence": "low", "signals": [reason]}
        _EXT_VERIFY_CACHE[url] = result
        logger.info("[ext_verify] REJECTED %s → %s (%s)",
                    expected_name, url[:60], reason)
        return result

    def _accept(confidence: str, signals: list[str]) -> dict:
        result = {"is_person_homepage": True, "confidence": confidence, "signals": signals}
        _EXT_VERIFY_CACHE[url] = result
        return result

    # ── Pre-fetch negative URL signals ───────────────────────────────────────
    if _EXT_NEGATIVE_URL_RE.search(url):
        return _reject("negative_url_pattern")

    # ── Fetch page ────────────────────────────────────────────────────────────
    try:
        resp = _get_with_reason(url, timeout=5, retries=0)
    except Exception as exc:
        return _reject(f"fetch_failed: {type(exc).__name__}")

    ct = resp.headers.get("content-type", "")
    if "pdf" in ct or "octet-stream" in ct:
        return _reject("pdf_or_binary_response")
    if "text/html" not in ct and "text/" not in ct:
        return _reject(f"non_html_content_type: {ct[:40]}")

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return _reject("parse_error")

    page_text = soup.get_text(separator=" ", strip=True)
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    h1_tags = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2"])]
    headings_text = " ".join(h1_tags)
    all_link_texts = " ".join(a.get_text(strip=True) for a in soup.find_all("a"))

    signals: list[str] = []

    # ── Name presence ─────────────────────────────────────────────────────────
    name_lower = expected_name.lower()
    name_words = [w for w in expected_name.split() if len(w) > 1]
    # Check if most name words appear in the page text
    name_words_found = sum(1 for w in name_words if w.lower() in page_text.lower())
    if name_words_found == len(name_words):
        signals.append("name_in_page_text")
    elif name_words_found >= max(1, len(name_words) - 1):
        signals.append("name_partial_in_page_text")

    if name_lower in title_text.lower():
        signals.append("name_in_title")
    if any(name_lower in h.lower() for h in h1_tags):
        signals.append("name_in_heading")

    # ── Academic role ─────────────────────────────────────────────────────────
    if _EXT_ROLE_RE.search(page_text[:5000]):  # search first 5k chars
        signals.append("academic_role_word")

    # ── Academic section links ─────────────────────────────────────────────────
    if _EXT_ACADEMIC_SECTION_RE.search(all_link_texts):
        signals.append("academic_section_link")
    if _EXT_ACADEMIC_SECTION_RE.search(headings_text):
        signals.append("academic_section_heading")

    # ── Institution match ─────────────────────────────────────────────────────
    if expected_institution:
        inst_lower = expected_institution.lower()
        # Use first two words of institution to match loosely
        inst_words = inst_lower.split()[:2]
        if any(w in page_text.lower() for w in inst_words if len(w) > 3):
            signals.append("institution_match")

    # ── Scholar link ─────────────────────────────────────────────────────────
    hrefs = [a.get("href", "") for a in soup.find_all("a", href=True)]
    if any("scholar.google" in h for h in hrefs):
        signals.append("google_scholar_link")
    if any("orcid.org" in h for h in hrefs):
        signals.append("orcid_link")

    # ── Negative signals (soft penalty, not hard reject) ─────────────────────
    # A single social/commercial link shouldn't outweigh name_in_title +
    # google_scholar_link.  Collect as a penalty rather than immediate reject.
    has_negative = bool(_EXT_NEGATIVE_RE.search(page_text[:3000]))
    if has_negative:
        signals.append("negative_commercial_signal")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Require at minimum: name appears on page
    name_present = bool({"name_in_page_text", "name_in_title", "name_in_heading"} & set(signals))
    if not name_present and "name_partial_in_page_text" not in signals:
        return _reject("name_not_found_on_page")

    # Score the signals
    high_value = {"name_in_title", "name_in_heading", "google_scholar_link",
                  "orcid_link", "academic_role_word"}
    medium_value = {"name_in_page_text", "academic_section_link",
                    "academic_section_heading", "institution_match"}

    high_hits = len(high_value & set(signals))
    medium_hits = len(medium_value & set(signals))
    total_score = high_hits * 2 + medium_hits

    # Penalty for negative signals: subtract 2 (one high-value point worth)
    if has_negative:
        total_score -= 2

    if total_score >= 4:
        confidence = "high"
    elif total_score >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    is_homepage = total_score >= 2  # require at least medium confidence
    if is_homepage:
        logger.info("[ext_verify] VERIFIED %s → %s (%d signals: %s)",
                    expected_name, url[:60], len(signals), signals)
        return _accept(confidence, signals)
    else:
        logger.info("[ext_verify] REJECTED %s → %s (score=%d, signals=%s)",
                    expected_name, url[:60], total_score, signals)
        return _reject(f"low_score_{total_score}: {signals}")


_CV_TEXT_RE = re.compile(r'\b(cv|curriculum\s+vitae|r[eé]sum[eé])\b', re.IGNORECASE)
_CV_HREF_RE = re.compile(r'[/_-]cv[._/-]|cv\.pdf|curriculum.vitae', re.IGNORECASE)

# Card detection for faculty section extraction
# Tags that can be individual faculty card elements
_CARD_TAGS: frozenset[str] = frozenset({"div", "li", "article"})
# Tags traversed when BFS-searching for a card parent container
_CONTAINER_TAGS: frozenset[str] = frozenset({"div", "ul", "ol", "article", "section"})
# "photo of …" prefix on img alt text (Columbia pattern) — strip before name check
_PHOTO_PREFIX_RE = re.compile(r"^photo\s+of\s+", re.IGNORECASE)

# ── Semantic content classification ──────────────────────────────────────────

# Headings that signal a publications block — never a faculty section
_PUBLICATION_HEADING_RE = re.compile(
    r"^\s*(?:selected|recent|representative|refereed|peer.reviewed|all|"
    r"latest|our|new)?\s*"
    r"(?:publications?|papers?|articles?|works?|research\s+outputs?|"
    r"preprints?|manuscripts?|books?|book\s+chapters?)\s*$",
    re.IGNORECASE,
)

# Signals within a text/href that mark it as a publication entry (not a person)
_PUB_ENTRY_RE = re.compile(
    r"doi\.org|arxiv\.org|ssrn\.com|pubmed\.ncbi|"
    r"posted\s+in\s*[:\(]|publication\s+date\s*:|"
    r"pp\.\s*\d+|vol(?:ume)?\.?\s*\d+\s*(?:,|no\.)|"
    r"proceedings\s+of\s+(?:the\s+)?[A-Z]|"
    r"journal\s+of\s+\w|conference\s+on\s+\w|workshop\s+on\s+\w|"
    r"(?:19|20)\d{2}\)\s*[,\.]",   # "(2023)." or "(2023)," at end of citation
    re.IGNORECASE,
)

# Button/link text for interactive faculty pages ("View ML Faculty", "See People", etc.)
_INTERACTIVE_FACULTY_BTN_RE = re.compile(
    r"\b(?:view|see|show|explore|browse)\s+"
    r"(?:(?:all|the|our)\s+)?(?:\w+\s+){0,3}"
    r"(?:faculty|people|members?|researchers?)\b",
    re.IGNORECASE,
)



# Anchor text patterns for personal website detection (scored, not keyword-match)
_WEBSITE_TEXT_HIGH = re.compile(
    r'\b(website|home\s*page|lab\s*(?:website|page|site)?|'
    r'personal\s*(?:page|site|website)|my\s*(?:page|site|website|homepage)|'
    r'web\s*page|research\s*(?:group\s*)?(?:page|website)|'
    r'visit\s*(?:my\s*)?(?:site|page|website)?)\b',
    re.IGNORECASE,
)
_WEBSITE_TEXT_MED = re.compile(
    r'\b(home|site|group\s*(?:page|site)|lab\s*page)\b',
    re.IGNORECASE,
)

# Domains that are never personal websites
_EXCLUDED_WEBSITE_DOMAINS = frozenset([
    "scholar.google.com", "linkedin.com", "twitter.com", "x.com",
    "facebook.com", "youtube.com", "researchgate.net", "orcid.org",
    "academia.edu", "semanticscholar.org", "github.com",
    "dblp.org", "acm.org", "ieee.org", "arxiv.org",
    "scopus.com", "webofscience.com", "publons.com",
])

# CSS selectors for contact/sidebar areas — links here get a position bonus
_CONTACT_SELECTORS = [
    ".contact", ".contact-info", ".faculty-contact",
    ".sidebar", ".profile-sidebar", ".bio-sidebar",
    ".profile-info", ".faculty-info", ".person-detail",
    ".contact-details", ".directory-contact", ".vcard",
    "#contact", "#sidebar",
]


def _find_links(soup: BeautifulSoup, base_url: str) -> tuple[str, str, str, str]:
    """
    Return (personal_website, personal_website_confidence, google_scholar, cv_url).

    personal_website_confidence: "high" | "medium" | "low" | ""

    Personal website detection is scored rather than keyword-only:
      +10  anchor text matches _WEBSITE_TEXT_HIGH
      + 5  anchor text matches _WEBSITE_TEXT_MED
      + 4  link found inside a contact/sidebar area
      + 3  URL contains tilde path (~name → personal academic page)
      + 2  URL is on sites.google.com or github.io (personal hosting)
      - 8  link is inside <nav> or <footer>
    Only links with score > 0 are considered. Highest scorer wins.
    """
    base_domain = urlparse(base_url).netloc
    scholar = ""
    cv_url = ""

    # Pre-collect hrefs that appear in contact/sidebar containers
    contact_hrefs: set[str] = set()
    for sel in _CONTACT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            for a in el.find_all("a", href=True):
                contact_hrefs.add(a["href"].strip())

    website_candidates: list[tuple[int, str]] = []  # (score, full_href)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        full_href = urljoin(base_url, href) if not href.startswith("http") else href
        parsed_href = urlparse(full_href)

        if parsed_href.scheme not in ("http", "https"):
            continue

        # ── Google Scholar ────────────────────────────────────────────────────
        # Accept by href (most reliable) OR by any visible label when the href
        # is a direct scholar.google.com URL (catches icon-only sidebar buttons).
        _a_text  = a.get_text(separator=" ", strip=True)
        _title   = (a.get("title")      or "").strip()
        _aria    = (a.get("aria-label") or "").strip()
        _scholar_label = re.search(
            r"\bgoogle\s+scholar\b|\bscholar\b|\bcitations\b",
            f"{_a_text} {_title} {_aria}",
            re.IGNORECASE,
        )
        if not scholar and "scholar.google" in href:
            scholar = full_href
            _how = "href"
            if _scholar_label:
                _how = f"href+label({_scholar_label.group(0)})"
            logger.info("[find_links] Scholar URL from %s: %s", _how, full_href[:100])
            continue
        # If anchor/title/aria says "scholar" but href is a redirect-style URL,
        # log for visibility — we cannot use the URL without following the redirect.
        if not scholar and _scholar_label and href.startswith("http"):
            logger.info(
                "[find_links] Scholar label '%s' on non-Scholar href %s — skipping",
                _scholar_label.group(0), href[:80],
            )

        # ── CV / Resume ───────────────────────────────────────────────────────
        text = a.get_text(separator=" ", strip=True)
        text_lower = text.lower()
        if not cv_url:
            if _CV_TEXT_RE.search(text_lower) or _CV_HREF_RE.search(href):
                cv_url = full_href
                continue

        # ── Personal website candidate scoring ────────────────────────────────
        href_domain = parsed_href.netloc

        # Must be external to the faculty's institution domain
        if base_domain and base_domain in href_domain:
            continue

        # Excluded domains
        if any(ex in href_domain for ex in _EXCLUDED_WEBSITE_DOMAINS):
            continue

        # Exclude PDF files
        if href.lower().endswith(".pdf"):
            continue

        score = 0

        # Anchor text cues
        if _WEBSITE_TEXT_HIGH.search(text):
            score += 10
        elif _WEBSITE_TEXT_MED.search(text):
            score += 5

        # Position bonus: inside a contact/sidebar area
        if href in contact_hrefs:
            score += 4

        # URL structure bonuses
        path = parsed_href.path
        if "~" in path:
            score += 3      # /~username → personal academic page
        if "sites.google" in href_domain:
            score += 2
        if "github.io" in href_domain:
            score += 2

        # Nav/footer penalty
        if a.find_parent(["nav", "footer"]):
            score -= 8

        if score > 0:
            website_candidates.append((score, full_href))

    # Best scoring candidate
    personal = ""
    confidence = ""
    if website_candidates:
        website_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_url = website_candidates[0]
        personal = best_url
        if best_score >= 10:
            confidence = "high"
        elif best_score >= 5:
            confidence = "medium"
        else:
            confidence = "low"

    logger.debug(
        "[find_links] base=%s scholar=%s cv=%s website=%s(%s) candidates=%d",
        base_domain, bool(scholar), bool(cv_url), bool(personal), confidence,
        len(website_candidates),
    )
    return personal, confidence, scholar, cv_url


def _fetch_extra_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch a URL (HTML page or PDF) and return cleaned plain text up to max_chars.
    Used for faculty CV and personal website content enrichment.
    """
    if not url:
        return ""
    try:
        r = _get(url, timeout=12)
        if r is None:
            return ""
        content_type = r.headers.get("Content-Type", "").lower()
        # Detect PDF by Content-Type, URL extension, OR magic bytes in the body.
        # Some servers (e.g. .../curriculum_vitae) serve PDFs without .pdf in the
        # URL and with a generic content-type, so we must check the body too.
        is_pdf = (
            "pdf" in content_type
            or url.lower().endswith(".pdf")
            or r.content[:5] == b"%PDF-"
        )
        if is_pdf:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(r.content))
                pages = [page.extract_text() or "" for page in reader.pages[:6]]
                text = "\n".join(pages)[:max_chars]
                return text if text.strip() else ""
            except Exception as e:
                logger.warning(f"[extra_content] PDF parse error {url}: {e}")
                return ""
        # HTML
        soup = BeautifulSoup(r.text, "lxml")
        _clean_soup(soup)
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception as e:
        logger.warning(f"[extra_content] {url}: {e}")
        return ""


# ── Tool 1: read_cv ───────────────────────────────────────────────────────── #

@mcp.tool()
async def read_cv(file_path: str) -> str:
    """
    Read a CV file (.txt or .pdf) and return its text content.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages).strip()
            return text if text else "Error: could not extract text from PDF (may be scanned)"
        except Exception as e:
            return f"Error reading PDF: {e}"

    return f"Error: unsupported file type '{suffix}'. Use .txt or .pdf."


# ── Tool 2: fetch_faculty_list ────────────────────────────────────────────── #

def _fetch_faculty_list_sync(url: str) -> list[dict] | dict:
    """
    Scrape any page that contains a faculty list.
    Returns list of {name, profile_url, full_profile_url, _page_class},
    or {"error": <reason>} / {"page_type": "interactive_grouped_faculty_page", ...}.

    Delegates to the Web Extraction Agent (extraction_agent.run_extraction_agent)
    which runs a Planner → Executor → Validator loop with Kahuna memory.
    """
    from mcp_servers.extraction_agent import run_extraction_agent
    outcome = run_extraction_agent(url)
    return outcome.to_legacy_format()


@mcp.tool()
async def fetch_faculty_list(url: str) -> str:
    """
    Scrape a department faculty directory page and extract faculty names
    and their profile page URLs.

    Args:
        url: URL of the department faculty list / directory page.

    Returns:
        JSON string: [{"name": "...", "profile_url": "...", "full_profile_url": "..."}, ...]
    """
    result = _fetch_faculty_list_sync(url)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool 3: fetch_all_faculty_profiles ───────────────────────────────────── #

def _fetch_one_profile(entry: dict) -> dict:
    """Synchronous fetch + parse of a single faculty profile page."""
    full_url = entry.get("full_profile_url", "")
    name_hint = entry.get("name", "")

    result = {
        "name": name_hint,
        "title": "",
        "profile_url": full_url,
        "bio": "",
        "research_interests": "",
        "personal_website": "",
        "personal_website_confidence": "",
        "google_scholar": "",
        "cv_url": "",
        "orcid": "",
        "error": "",
        "page_valid": False,
        "page_type": "unknown",
        # ── Audit / provenance metadata ───────────────────────────────────
        "fetch_metadata": {
            "profile_page": {
                "url": full_url,
                "fetched": False,
                "http_status": None,
                "char_count": 0,
                "text_preview": "",       # first 400 chars of cleaned page text
            },
            "personal_website": {
                "url": None,
                "fetched": False,
                "note": "URL extracted from profile page only — page not visited",
            },
            "google_scholar": {
                "url": None,
                "fetched": False,
                "note": "URL extracted from profile page only — page not visited",
            },
        },
    }

    r = _get(full_url, timeout=15)
    if r is None:
        result["error"] = f"Could not fetch {full_url}"
        return result

    # Record actual fetch
    result["fetch_metadata"]["profile_page"]["fetched"] = True
    result["fetch_metadata"]["profile_page"]["http_status"] = r.status_code

    soup_raw = BeautifulSoup(r.text, "lxml")

    # ── Page type detection (before cleaning, so we see all links) ───────────
    page_type = _detect_page_type(soup_raw, full_url)
    result["page_type"] = page_type
    if page_type == "faculty_directory":
        result["error"] = "URL points to a faculty directory page, not an individual profile"
        result["page_valid"] = False
        return result

    # ── External links (Scholar + website + CV) extracted from RAW soup ───────
    # MUST run before _clean_soup: sidebars and <aside> elements that commonly
    # hold Scholar/website buttons are removed by the cleaner.
    personal, web_confidence, scholar, cv = _find_links(soup_raw, full_url)
    result["personal_website"]            = personal
    result["personal_website_confidence"] = web_confidence
    result["google_scholar"]              = scholar
    result["cv_url"]                      = cv
    result["fetch_metadata"]["personal_website"]["url"] = personal or None
    result["fetch_metadata"]["google_scholar"]["url"]   = scholar or None

    # Extract ORCID URL from profile page (e.g. https://orcid.org/0000-0002-XXXX-XXXX)
    _ORCID_HREF_RE = re.compile(r'https://orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])', re.IGNORECASE)
    for a in soup_raw.find_all("a", href=True):
        m = _ORCID_HREF_RE.match(a["href"].strip())
        if m:
            result["orcid"] = a["href"].strip()
            break

    # ── Name: extracted from RAW soup (before cleaning removes <header> tags) ──
    # _clean_soup removes <header>, which is where many university sites place
    # the professor's name. Extracting here prevents section headings (e.g.
    # "Current projects") from being mistaken for a name after cleaning.
    url_name = _name_from_url(full_url)
    if url_name and _is_valid_person_name(url_name):
        result["name"] = url_name
    else:
        # Scan h1 then h2 — only accept if it passes person-name validation
        for tag_name in ("h1", "h2"):
            for el in soup_raw.find_all(tag_name):
                candidate = el.get_text(strip=True)
                if _is_valid_person_name(candidate):
                    result["name"] = candidate
                    break
            if result["name"] and result["name"] != name_hint:
                break
        # If still no valid name from page, fall back to the directory hint
        if not result["name"]:
            result["name"] = name_hint

    # Now clean for content extraction (removes sidebar, nav, aside, header, etc.)
    _clean_soup(soup_raw)

    # ── Page validity: must have a plausible person name ────────────────────
    if not _is_valid_person_name(result["name"]):
        result["error"] = (
            f"Could not extract a valid faculty name from this page "
            f"(got: '{result['name']}'). The URL may point to a listing or generic page."
        )
        result["page_valid"] = False
        return result

    result["page_valid"] = True

    # ── Title: next prominent text block after name heading ──────────────────
    h_tag = soup_raw.find(["h1", "h2"])
    if h_tag:
        sib = h_tag.find_next_sibling()
        if sib:
            t = sib.get_text(strip=True)
            if 0 < len(t) < 120:
                result["title"] = t

    # ── Main content area (minimise nav noise) ───────────────────────────────
    full_text = _extract_main_content(soup_raw)

    # Record text stats for audit
    result["fetch_metadata"]["profile_page"]["char_count"] = len(full_text)
    result["fetch_metadata"]["profile_page"]["text_preview"] = full_text[:400]

    # ── Bio ──────────────────────────────────────────────────────────────────
    bio = _extract_section(
        full_text,
        "biography", "about", "overview", "background",
        max_chars=800,
    )
    result["bio"] = bio

    # ── Research interests ───────────────────────────────────────────────────
    interests = _extract_section(
        full_text,
        "research interest", "research area", "research focus",
        "research summary", "areas of interest", "research topic",
        max_chars=500,
    )
    if not interests:
        interests = bio[:400] if bio else ""
    result["research_interests"] = interests

    logger.info(
        "Parsed profile: %s | valid=%s | scholar=%s | cv=%s | website=%s(%s)",
        result["name"], result["page_valid"],
        bool(scholar), bool(cv), bool(personal), web_confidence,
    )
    return result


@mcp.tool()
async def fetch_all_faculty_profiles(faculty_list_json: str, max_count: int = 25) -> str:
    """
    Given the JSON output from fetch_faculty_list, concurrently fetch every
    faculty profile page and extract structured information.

    Args:
        faculty_list_json: JSON string from fetch_faculty_list.
        max_count:         Maximum number of profiles to fetch (default 25).

    Returns:
        JSON string: list of dicts with keys:
          name, title, profile_url, bio, research_interests,
          personal_website, google_scholar, error
    """
    try:
        faculty_list = json.loads(faculty_list_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON input: {e}"})

    if isinstance(faculty_list, dict) and "error" in faculty_list:
        return faculty_list_json  # propagate upstream error

    entries = faculty_list[:max_count]
    logger.info(f"Fetching {len(entries)} profiles concurrently...")

    # Use threads so requests (sync) works inside async context
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _fetch_one_profile, entry)
        for entry in entries
    ]
    results = await asyncio.gather(*tasks)

    logger.info(f"Done fetching {len(results)} profiles.")
    return json.dumps(list(results), ensure_ascii=False, indent=2)


# ── Tool 4: fetch_faculty_page (generic, kept for one-off use) ────────────── #

@mcp.tool()
async def fetch_faculty_page(url: str) -> str:
    """
    Fetch and return cleaned text from any static web page (faculty page,
    lab page, Google Scholar profile, etc.). Up to 8000 chars.
    """
    r = _get(url)
    if r is None:
        return f"Error: could not fetch {url}"

    soup = BeautifulSoup(r.text, "lxml")
    _clean_soup(soup)

    for selector in ["main", "article", "#content", ".content", "#main", ".main"]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 200:
                break
    else:
        text = soup.get_text(separator="\n", strip=True)

    MAX = 8000
    return text[:MAX] + f"\n\n[truncated at {MAX} chars]" if len(text) > MAX else text


# ── Tool 5: fetch_google_scholar_profile ─────────────────────────────────── #

def _normalize_scholar_url(url: str) -> str:
    """Ensure sortby=pubdate so we get most-recent papers first."""
    if "sortby=pubdate" not in url:
        sep = "&" if "?" in url else "?"
        url = url + sep + "sortby=pubdate"
    # Remove view_op param which can redirect away from the list
    url = re.sub(r"&?view_op=[^&]*", "", url)
    return url


def _fetch_scholar_pubs(scholar_url: str) -> dict:
    """
    Synchronously fetch a Google Scholar profile page and extract
    up to 20 most-recent publications.
    Returns a dict with:
      fetch_status, http_status, publications (list), error (if any)
    """
    result = {
        "fetch_status": "not_attempted",
        "http_status": None,
        "url_used": scholar_url,
        "publications": [],
        "pub_count": 0,
        "error": "",
    }

    url = _normalize_scholar_url(scholar_url)
    result["url_used"] = url

    try:
        r = _requests_plain.get(url, headers=HEADERS, timeout=20)
        result["http_status"] = r.status_code
        r.raise_for_status()
    except _requests_plain.exceptions.Timeout:
        result["fetch_status"] = "timeout"
        result["error"] = "Request timed out after 20s"
        return result
    except _requests_plain.exceptions.HTTPError as e:
        result["fetch_status"] = "http_error"
        result["error"] = f"HTTP {e.response.status_code}"
        return result
    except Exception as e:
        result["fetch_status"] = "error"
        result["error"] = str(e)
        return result

    # Detect bot-blocking
    text_lower = r.text.lower()
    if "captcha" in text_lower or "unusual traffic" in text_lower or "sorry" in text_lower[:500]:
        result["fetch_status"] = "blocked_captcha"
        result["error"] = "Google Scholar returned a CAPTCHA / bot-detection page"
        return result

    if len(r.text) < 2000:
        result["fetch_status"] = "blocked_short_response"
        result["error"] = f"Response too short ({len(r.text)} bytes) — likely blocked"
        return result

    soup = BeautifulSoup(r.text, "lxml")
    rows = soup.select("tr.gsc_a_tr")

    if not rows:
        result["fetch_status"] = "no_publications_found"
        result["error"] = "Page loaded but no publication rows found (structure may have changed)"
        return result

    pubs = []
    for row in rows[:20]:
        title_el = row.select_one("a.gsc_a_at")
        gray_els = row.select("div.gs_gray")
        year_el  = row.select_one("span.gsc_a_h")
        cites_el = row.select_one("a.gsc_a_ac")

        title   = title_el.get_text(strip=True) if title_el else ""
        authors = gray_els[0].get_text(strip=True) if len(gray_els) > 0 else ""
        venue   = gray_els[1].get_text(strip=True) if len(gray_els) > 1 else ""
        year    = year_el.get_text(strip=True)  if year_el  else ""
        cites   = cites_el.get_text(strip=True) if cites_el else ""

        if title:
            pubs.append({
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "citations": cites,
            })

    result["publications"] = pubs
    result["pub_count"] = len(pubs)
    result["fetch_status"] = "success"
    logger.info(f"Scholar: fetched {len(pubs)} pubs from {url[:60]}")
    return result


@mcp.tool()
async def fetch_google_scholar_profile(scholar_url: str) -> str:
    """
    Fetch up to 20 most-recent publications from a Google Scholar profile page.

    Args:
        scholar_url: URL of the Google Scholar profile
                     (e.g. https://scholar.google.com/citations?user=XXX)

    Returns:
        JSON string with:
          fetch_status, http_status, url_used,
          publications: [{title, authors, venue, year, citations}],
          pub_count, error
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_scholar_pubs, scholar_url)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Scholar URL extraction helper ────────────────────────────────────────── #

def _extract_scholar_from_url(url: str, name: str = "") -> str:
    """
    Fetch *url* (HTML page or PDF) and return the first Google Scholar profile
    URL found, or "" if none.

    Handles both:
      • HTML pages  — scans all <a href> with _find_links (checks href,
                       title, aria-label, and anchor text)
      • PDF files   — extracts text with pypdf and regex-searches for
                       scholar.google.com/citations?...

    Returns the full Scholar URL string, or "".
    """
    if not url:
        return ""
    try:
        r = _get(url, timeout=12)
        if r is None:
            return ""
        content_type = r.headers.get("Content-Type", "").lower()
        # Detect PDF by magic bytes in addition to Content-Type / URL suffix.
        # Some servers (e.g. Northwestern NW academic records) serve PDF files
        # with Content-Type: text/html, causing naive header-only checks to miss them.
        is_pdf = (
            "pdf" in content_type
            or url.lower().endswith(".pdf")
            or r.content[:4] == b"%PDF"
        )

        if is_pdf:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(r.content))
                pdf_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages[:8]
                )
                # Scholar URLs appear in text as "scholar.google.com/citations?user=..."
                m = re.search(
                    r"scholar\.google\.com/citations\?[^\s\"'<>\]]+",
                    pdf_text,
                )
                if m:
                    return "https://" + m.group(0).rstrip(".,;)")
            except Exception as _e:
                logger.debug("[_extract_scholar] PDF parse error %s: %s", url[:60], _e)
        else:
            soup = BeautifulSoup(r.text, "lxml")
            _, _, scholar_url, _ = _find_links(soup, url)
            if scholar_url:
                return scholar_url
        return ""
    except Exception as e:
        logger.debug("[_extract_scholar] fetch error %s: %s", url[:60], e)
        return ""


# ── Scholar fallback: DuckDuckGo search for Scholar profile URL ───────────── #

def _search_engine_scholar_lookup(name: str, institution_hint: str = "") -> str:
    """
    Use DuckDuckGo HTML search to find a Google Scholar author profile.
    Searches:  site:scholar.google.com "Name" "institution"

    DDG is used because it does not require login and indexes Scholar pages.
    Returns a scholar.google.com/citations?user=... URL, or "".
    """
    from urllib.parse import quote_plus, parse_qs, urlparse as _up, unquote

    # Query pattern: "{name} scholar.google.com {institution}" works reliably.
    # Avoid including the literal phrase "google scholar" in the query —
    # DDG returns a 202 challenge page when it detects that phrase.
    query_parts = [name, "scholar.google.com"]
    if institution_hint:
        query_parts.append(institution_hint)
    query = " ".join(query_parts)

    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    logger.info("[ddg_scholar] %s: query=%r", name, query)

    r = _get(search_url, timeout=15)
    if r is None:
        logger.warning("[ddg_scholar] %s: fetch returned None", name)
        return ""

    # Method 1: regex over raw HTML — DDG embeds the actual destination URL
    # in <a href="//duckduckgo.com/l/?uddg=ENCODED_URL&...">
    _scholar_re = re.compile(
        r"scholar\.google\.com/citations\?[^\s\"'<>]{5,}",
        re.IGNORECASE,
    )
    for m in _scholar_re.finditer(r.text):
        raw = m.group(0)
        # Skip search / listing result pages — we want author profiles
        if "view_op=search_authors" in raw or "view_op=list_works" in raw:
            continue
        if "user=" in raw:
            url = "https://" + raw.rstrip(".,;)\"'\\")
            # Decode HTML entities that appear in DDG-embedded URLs (e.g. &amp; → &)
            url = url.replace("&amp;", "&").replace("&amp;", "&")
            logger.info("[ddg_scholar] %s: found via DDG regex → %s", name, url[:100])
            return url

    # Method 2: parse <a> hrefs and decode DDG redirect wrappers
    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "uddg=" not in href:
            continue
        try:
            params = parse_qs(_up(href).query)
            uddg = params.get("uddg", [""])[0]
            if not uddg:
                continue
            actual = unquote(uddg)
            if (
                "scholar.google.com/citations" in actual
                and "user=" in actual
                and "view_op=search_authors" not in actual
            ):
                logger.info(
                    "[ddg_scholar] %s: found via DDG redirect → %s", name, actual[:100]
                )
                return actual
        except Exception:
            pass

    logger.info("[ddg_scholar] %s: not found", name)
    return ""


# ── Scholar fallback: scan personal website HTML for Scholar link ─────────── #

def _scholar_fallback_search(
    name: str,
    profile_url: str = "",
    research_interests: str = "",
    email_domain: str = "",
    personal_website: str = "",
    cv_url: str = "",
) -> dict:
    """
    Attempt to find a Google Scholar profile URL for a faculty member whose
    institutional profile page does not link to Scholar directly.

    Strategy (in order, stops at first success):
      1. Fetch the faculty's personal website (if known) and scan its HTML
         for a Scholar link.
      2. Fetch the CV URL (HTML or PDF) and scan for Scholar link.
      3. Google Scholar author search (last resort; often blocked by bot-wall).

    Returns dict with keys:
      scholar_url           — full Scholar URL or ""
      scholar_source        — "personal_website" | "cv" | "author_search" | "not_found"
      scholar_confidence    — "high" | "medium" | ""
      scholar_match_confidence — same as scholar_confidence (alias for UI)
    """
    empty = {
        "scholar_url": "", "scholar_source": "not_found",
        "scholar_confidence": "", "scholar_match_confidence": "",
        "strategies_tried": [], "errors": [],
    }
    if not name:
        return empty
    strategies_tried: list[str] = []
    errors: list[str] = []

    # ── Strategy 1: Scan personal website — direct Scholar link + CV fallback ──
    if personal_website:
        strategies_tried.append("personal_website")
        try:
            r = _get(personal_website, timeout=12)
            if r is not None:
                soup = BeautifulSoup(r.text, "lxml")
                _, _, scholar_url, cv_on_website = _find_links(soup, personal_website)

                if scholar_url:
                    logger.info(
                        "[scholar_fallback] %s: Scholar found on personal website → %s",
                        name, scholar_url,
                    )
                    return {
                        "scholar_url": scholar_url,
                        "scholar_source": "personal_website",
                        "scholar_confidence": "high",
                        "scholar_match_confidence": "high",
                        "strategies_tried": strategies_tried,
                        "errors": errors,
                    }

                if cv_on_website:
                    strategies_tried.append("personal_website_cv")
                    scholar_url = _extract_scholar_from_url(cv_on_website, name)
                    if scholar_url:
                        logger.info(
                            "[scholar_fallback] %s: Scholar found in CV from personal"
                            " website → %s",
                            name, scholar_url,
                        )
                        return {
                            "scholar_url": scholar_url,
                            "scholar_source": "personal_website_cv",
                            "scholar_confidence": "high",
                            "scholar_match_confidence": "high",
                            "strategies_tried": strategies_tried,
                            "errors": errors,
                        }

            logger.info(
                "[scholar_fallback] %s: no Scholar on personal website (%s)",
                name, personal_website[:60],
            )
        except Exception as e:
            logger.warning("[scholar_fallback] %s: personal website error: %s", name, e)
            errors.append(f"personal_website: {e}")

    # ── Strategy 2: Scan CV URL from profile page (HTML or PDF) ──────────────
    if cv_url:
        strategies_tried.append("profile_cv")
        scholar_url = _extract_scholar_from_url(cv_url, name)
        if scholar_url:
            logger.info(
                "[scholar_fallback] %s: Scholar found in profile CV → %s",
                name, scholar_url,
            )
            return {
                "scholar_url": scholar_url,
                "scholar_source": "cv",
                "scholar_confidence": "high",
                "scholar_match_confidence": "high",
                "strategies_tried": strategies_tried,
                "errors": errors,
            }
        logger.info(
            "[scholar_fallback] %s: no Scholar in profile CV (%s)",
            name, cv_url[:60],
        )

    # ── Derive institution hint (used by Strategy 3 + 4) ─────────────────────
    institution_hint = ""
    if profile_url:
        from urllib.parse import urlparse as _urlparse
        host = _urlparse(profile_url).netloc.lower()
        parts = [p for p in host.replace("www.", "").split(".") if len(p) > 2]
        if parts:
            institution_hint = parts[-2] if len(parts) >= 2 else parts[0]

    # ── Strategy 3: DuckDuckGo search for Scholar author profile ─────────────
    strategies_tried.append("ddg_search")
    try:
        scholar_url = _search_engine_scholar_lookup(name, institution_hint)
    except Exception as _e:
        logger.warning("[scholar_fallback] %s: DDG search raised %s — skipping", name, _e)
        errors.append(f"ddg_search: {_e}")
        scholar_url = ""
    if scholar_url:
        logger.info(
            "[scholar_fallback] %s: Scholar found via DDG search → %s",
            name, scholar_url,
        )
        return {
            "scholar_url": scholar_url,
            "scholar_source": "search_engine",
            "scholar_confidence": "medium",
            "scholar_match_confidence": "medium",
            "strategies_tried": strategies_tried,
            "errors": errors,
        }

    # ── No Scholar found ──────────────────────────────────────────────────────
    # (Strategy 4 — Scholar author search — removed: frequently bot-blocked,
    #  slow (5-8s), superseded by OpenAlex as publication fallback.)
    logger.info("[scholar_fallback] %s: Scholar not found after DDG", name)
    return {**empty, "strategies_tried": strategies_tried, "errors": errors}


# ── Level 3: Google Scholar author search ─────────────────────────────────── #

_scholar_author_search_cache: dict[str, dict | None] = {}


def _name_similar(candidate: str, target: str) -> bool:
    """Return True if candidate name is plausibly the same person as target."""
    def _norm(s: str) -> str:
        return s.lower().strip()

    c_parts = _norm(candidate).split()
    t_parts = _norm(target).split()
    if not c_parts or not t_parts:
        return False

    # Last names must match exactly
    if c_parts[-1] != t_parts[-1]:
        return False

    # If only one name token on either side, last-name match is enough
    if len(c_parts) == 1 or len(t_parts) == 1:
        return True

    # First name: allow abbreviated match  (e.g. "J." matches "John")
    c_first = c_parts[0].rstrip(".")
    t_first = t_parts[0].rstrip(".")
    if c_first == t_first:
        return True
    # One is an abbreviation of the other
    if c_first.startswith(t_first) or t_first.startswith(c_first):
        return True

    return False


def _scholar_author_search(
    name: str,
    institution_hint: str = "",
    research_keywords: list[str] | None = None,
) -> dict | None:
    """
    Query Google Scholar's author-search endpoint and return the best-matching
    candidate as a dict with keys ``scholar_url`` and ``scholar_confidence``,
    or ``None`` if no confident match is found.

    Disambiguation scoring (max ~13 pts):
      +2  last-name / abbreviated-first-name match
      +5  affiliation contains institution_hint
      +5  ≥2 research keywords overlap (proportional, capped at 5)
      +1  citation count ≥ 100

    Thresholds: score ≥ 8 → "high", score ≥ 5 → "medium", else → None
    """
    if research_keywords is None:
        research_keywords = []

    cache_key = f"{name}||{institution_hint}||{','.join(research_keywords[:5])}"
    if cache_key in _scholar_author_search_cache:
        return _scholar_author_search_cache[cache_key]

    def _store(result):
        _scholar_author_search_cache[cache_key] = result
        return result

    from urllib.parse import quote_plus
    search_url = (
        "https://scholar.google.com/citations"
        f"?view_op=search_authors&mauthors={quote_plus(name)}&hl=en"
    )

    logger.info("[author_search] %s: fetching %s", name, search_url)
    r = _get(search_url, timeout=15)
    if r is None:
        logger.warning("[author_search] %s: fetch returned None", name)
        return _store(None)

    # Detect login redirect
    final_url = str(r.url) if hasattr(r, "url") else ""
    if "accounts.google.com" in final_url or "ServiceLogin" in final_url:
        logger.warning("[author_search] %s: redirected to Google login — skipping", name)
        return _store(None)

    if r.status_code != 200:
        logger.warning("[author_search] %s: HTTP %d", name, r.status_code)
        return _store(None)

    soup = BeautifulSoup(r.text, "lxml")

    # Each author card: div.gs_ai_chpr (or .gsc_1usr in older layout)
    cards = soup.select("div.gs_ai_chpr")
    if not cards:
        cards = soup.select("div.gsc_1usr")
    logger.info("[author_search] %s: found %d candidate cards", name, len(cards))

    best_score = 0
    best_url   = ""
    best_card_name = ""

    low_kw = [k.lower() for k in research_keywords]

    for card in cards:
        # Candidate name
        name_el = card.select_one(".gs_ai_name a, .gsc_1usr_name a")
        if not name_el:
            continue
        cand_name = name_el.get_text(strip=True)

        # Scholar profile link
        cand_href = name_el.get("href", "")
        if not cand_href.startswith("http"):
            cand_href = "https://scholar.google.com" + cand_href

        # Affiliation text
        aff_el = card.select_one(".gs_ai_aff, .gsc_1usr_aff")
        affiliation = aff_el.get_text(strip=True).lower() if aff_el else ""

        # Research interests
        int_els = card.select(".gs_ai_int a, .gsc_1usr_int a")
        interests_text = " ".join(el.get_text(strip=True).lower() for el in int_els)

        # Citations
        cby_el = card.select_one(".gs_ai_cby, .gsc_1usr_cby")
        citations = 0
        if cby_el:
            import re as _re
            m = _re.search(r"\d+", cby_el.get_text())
            if m:
                citations = int(m.group())

        # ── Score ────────────────────────────────────────────────────────────
        score = 0.0

        # Name similarity
        if _name_similar(cand_name, name):
            score += 2
        else:
            # No name match at all → skip immediately
            logger.debug(
                "[author_search] %s: candidate '%s' name mismatch — skip",
                name, cand_name,
            )
            continue

        # Affiliation
        if institution_hint and institution_hint.lower() in affiliation:
            score += 5

        # Research keyword overlap
        if low_kw:
            combined = interests_text + " " + affiliation
            hits = sum(1 for kw in low_kw if kw in combined)
            kw_score = min(5.0, hits * (5.0 / max(len(low_kw), 2)))
            score += kw_score

        # Citation bonus
        if citations >= 100:
            score += 1

        logger.info(
            "[author_search] %s: candidate='%s' aff='%s' score=%.1f cites=%d",
            name, cand_name, affiliation[:60], score, citations,
        )

        if score > best_score:
            best_score = score
            best_url   = cand_href
            best_card_name = cand_name

    if not best_url or best_score < 5:
        logger.info(
            "[author_search] %s: no confident match (best_score=%.1f best_name='%s')",
            name, best_score, best_card_name,
        )
        return _store(None)

    confidence = "high" if best_score >= 8 else "medium"
    logger.info(
        "[author_search] %s: → %s (score=%.1f conf=%s)",
        name, best_url, best_score, confidence,
    )
    return _store({"scholar_url": best_url, "scholar_confidence": confidence})


# ── OpenAlex academic data ───────────────────────────────────────────────────
# Free, open, no bot-detection, no API key needed.
# Polite-pool: include mailto so OpenAlex prioritises our requests.

_OA_BASE   = "https://api.openalex.org"
_OA_MAILTO = "a1042843517@gmail.com"

# ── OpenAlex field-sanity helpers ────────────────────────────────────────────

def _openalex_field_ok(author: dict) -> bool:
    """Field check is intentionally disabled.
    Disambiguation relies on institution + department matching in
    _openalex_author_search so the system works for any CV domain."""
    return True


def _openalex_author_search(
    name: str,
    institution_hint: str = "",
    orcid: str = "",
    dept_hint: str = "",
    research_keywords: "list[str] | None" = None,
) -> "dict | None":
    """
    Search OpenAlex for a faculty author by name + optional hints.
    Returns the best-matching raw OpenAlex author object (with added _confidence key),
    or None if no plausible match found.

    Layer 1: if orcid is provided, try a direct ORCID filter first — unambiguous.
    Layer 2: name search scored by:
      dept_match > inst_score (float, prefers main campus over sub-schools) >
      research_overlap (topic keywords from profile) > name_score
    """
    import urllib.parse as _up

    # Layer 1 — ORCID direct lookup (no time cost if ORCID not on profile page)
    if orcid:
        orcid_id = orcid.strip()
        if not orcid_id.startswith("https://"):
            orcid_id = f"https://orcid.org/{orcid_id}"
        orcid_url = f"{_OA_BASE}/authors?filter=orcid:{_up.quote(orcid_id, safe='')}&mailto={_OA_MAILTO}"
        try:
            r = _get(orcid_url, timeout=10)
            if r is not None:
                results = r.json().get("results") or []
                if len(results) == 1:
                    author = results[0]
                    author["_confidence"] = "high"
                    logger.info(
                        "[OpenAlex] %r: ORCID match → %r (conf=high)",
                        name, author.get("display_name"),
                    )
                    return author
        except Exception as exc:
            logger.warning("[OpenAlex] ORCID lookup failed for %r: %s", name, exc)
        # Fall through to name search

    # Layer 2 — Name search with institution + department hint scoring
    q   = _up.quote(name)
    url = f"{_OA_BASE}/authors?search={q}&per_page=5&mailto={_OA_MAILTO}"
    try:
        r = _get(url, timeout=12)
        if r is None:
            return None
        data = r.json()
        results = data.get("results") or []
    except Exception as exc:
        logger.warning("[OpenAlex] author search failed for %r: %s", name, exc)
        return None

    if not results:
        return None

    def _name_score(a: dict) -> float:
        import unicodedata as _ud
        _XLAT = str.maketrans("ıİğĞşŞçÇöÖüÜæÆøØåÅñÑ", "iIgGsSçÇoOuUaeoeOaaN")
        _XLAT.update({ord("ç"): "c", ord("Ç"): "C"})
        def _ascii(s: str) -> str:
            s = s.translate(_XLAT)
            return _ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        oa = _ascii(a.get("display_name", "")).lower()
        q_parts = set(_ascii(name).lower().split())
        oa_parts = set(oa.split())
        if not q_parts or not oa_parts:
            return 0.0
        return len(q_parts & oa_parts) / len(q_parts | oa_parts)

    def _inst_score(a: dict) -> float:
        """Return how well the author's institution matches institution_hint.
        1.0 = current institution NAME starts with hint  (e.g. "Cornell University" for "cornell")
        0.5 = hint appears mid-name                      (e.g. "Weill Cornell Medicine" for "cornell")
        0.0 = no match
        Only checks last_known_institutions (current affiliation) — historical affiliations
        are too noisy (past co-authorships can include unrelated "washington" schools etc.).
        """
        if not institution_hint:
            return 0.0
        hint = institution_hint.lower()
        best = 0.0
        for inst in (a.get("last_known_institutions") or []):
            dn = inst.get("display_name", "").lower()
            if dn.startswith(hint):
                best = max(best, 1.0)
            elif hint in dn:
                best = max(best, 0.5)
        return best

    def _dept_match(a: dict) -> bool:
        if not dept_hint:
            return False
        hint = dept_hint.lower()
        for inst in (a.get("last_known_institutions") or []):
            if hint in inst.get("display_name", "").lower():
                return True
        for aff in (a.get("affiliations") or []):
            disp = (aff.get("institution") or {}).get("display_name", "")
            if hint in disp.lower():
                return True
        return False

    # Build a normalised keyword set from the faculty's own research direction.
    # Used as a secondary signal only — absent when profile has no text.
    _kw_set: "frozenset[str]" = frozenset()
    if research_keywords:
        _STOP = {"the", "and", "for", "with", "from", "that", "this", "are",
                 "has", "have", "been", "its", "can", "use", "used", "using"}
        _kw_set = frozenset(
            w.lower() for kw in research_keywords
            for w in re.findall(r"[a-zA-Z]{3,}", kw)
            if w.lower() not in _STOP
        )

    def _research_overlap(a: dict) -> float:
        """Fraction of faculty research keywords matched by any OpenAlex topic name.
        Returns 0.0 when no keywords were provided (graceful no-op)."""
        if not _kw_set:
            return 0.0
        topic_words = frozenset(
            w.lower()
            for t in (a.get("topics") or [])
            for w in re.findall(r"[a-zA-Z]{3,}", t.get("display_name", ""))
        )
        if not topic_words:
            return 0.0
        return len(_kw_set & topic_words) / len(_kw_set)

    # Sort: dept_match > inst_score (float) > research_overlap > name_score
    # inst_score=1.0 for main-campus, 0.5 for sub-school — naturally prefers
    # "Cornell University" over "Weill Cornell Medicine" for hint "cornell".
    # research_overlap is an optional tiebreaker using the faculty's own profile keywords.
    scored = sorted(
        results,
        key=lambda a: (_dept_match(a), _inst_score(a), _research_overlap(a), _name_score(a)),
        reverse=True,
    )
    best = scored[0]
    ns   = _name_score(best)
    ins  = _inst_score(best)

    if ns < 0.5:
        logger.info("[OpenAlex] %r: best name score %.2f too low — skip", name, ns)
        return None

    # When institution_hint is provided and the best match has NO institution
    # connection at all (ins=0.0), reject — we know which school we're looking
    # for and a zero-institution score means it's a different person with the
    # same name at a completely different place.
    if institution_hint and ins == 0.0:
        logger.info(
            "[OpenAlex] %r: rejected — name matches but institution '%s' not found "
            "(best match: %r at %s)",
            name, institution_hint, best.get("display_name"),
            [i.get("display_name") for i in (best.get("last_known_institutions") or [])[:2]],
        )
        return None

    if ns >= 0.8 and ins >= 1.0:
        best["_confidence"] = "high"
    elif ns >= 0.8 and ins >= 0.5:
        # Institution matched but via a sub-school name — slightly less certain
        best["_confidence"] = "medium"
    elif ns >= 0.65 and ins >= 0.5:
        best["_confidence"] = "medium"
    elif ins >= 0.5:
        # Low name score but institution confirms identity
        best["_confidence"] = "medium"
    else:
        # name score < 0.65 AND no institution confirmation — very likely wrong person
        logger.info(
            "[OpenAlex] %r: rejected low-confidence match %r (ns=%.2f, inst=%.1f)",
            name, best.get("display_name"), ns, ins,
        )
        return None

    logger.info(
        "[OpenAlex] %r: matched %r conf=%s h_index=%s works=%d",
        name,
        best.get("display_name"),
        best["_confidence"],
        (best.get("summary_stats") or {}).get("h_index"),
        best.get("works_count", 0),
    )
    return best


def _openalex_works(author_id: str, per_page: int = 20) -> list[dict]:
    """
    Fetch recent works (2018–2025) for an OpenAlex author, sorted by citation count.
    Returns a list of normalised dicts with keys:
      title, year, venue, cited_by_count, doi, oa_url, openalex_id
    """
    import urllib.parse as _up

    url = (
        f"{_OA_BASE}/works"
        f"?filter=authorships.author.id:{_up.quote(author_id)}"
        f",publication_year:2018-2025"
        f"&sort=cited_by_count:desc"
        f"&per_page={per_page}"
        f"&mailto={_OA_MAILTO}"
    )
    try:
        r = _get(url, timeout=15)
        if r is None:
            return []
        works = r.json().get("results") or []
    except Exception as exc:
        logger.warning("[OpenAlex] works fetch failed for %r: %s", author_id, exc)
        return []

    out = []
    for w in works:
        loc   = w.get("primary_location") or {}
        src   = loc.get("source") or {}
        venue = src.get("display_name", "")
        oa    = w.get("open_access") or {}
        out.append({
            "title":          (w.get("title") or "").strip(),
            "year":           str(w.get("publication_year", "")),
            "venue":          venue,
            "cited_by_count": w.get("cited_by_count", 0),
            "doi":            w.get("doi", "") or "",
            "oa_url":         oa.get("oa_url", "") or "",
            "openalex_id":    w.get("id", ""),
        })
    return out


if __name__ == "__main__":
    mcp.run()
