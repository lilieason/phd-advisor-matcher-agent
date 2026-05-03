"""
Matching Agent test — single URL run.

Tests the full pipeline:
  Extraction Agent → Matching Agent → Top-10 ranked advisors
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for name in ("mcp_servers.extraction_agent", "mcp_servers.matching_agent"):
    logging.getLogger(name).setLevel(logging.INFO)

sys.path.insert(0, str(Path(__file__).parent))

from mcp_servers.matching_agent import run_matching_agent, UserProfile
from mcp_servers.extraction_agent import run_extraction_agent

SEP  = "=" * 72
SEP2 = "-" * 72

# ── Test user profile (PhD applicant in transportation) ──────────────────────
TEST_USER = UserProfile(
    research_interests=(
        "transportation network analysis, traffic operations and control, "
        "autonomous vehicles, transportation data science, travel behavior modeling, "
        "freight and logistics optimization"
    ),
    keywords=[
        "transportation", "traffic", "network", "autonomous", "data science",
        "logistics", "freight", "mobility", "travel", "optimization",
    ],
    degree_target="PhD",
    institution_preferences=["University of Washington"],
)

TEST_URL = "https://www.ce.washington.edu/research/transportation"


def print_outcome(outcome):
    print(f"\n{SEP}")
    print(f"MATCHING AGENT — {outcome.source_url}")
    print(SEP)

    print(f"\n[ PIPELINE SUMMARY ]")
    print(f"  total candidates     : {outcome.total_candidates}")
    print(f"  after early exclusion: {outcome.after_early_exclusion}")
    print(f"  profiles enriched    : {outcome.enriched_count}")
    print(f"  scholar searched     : {outcome.scholar_searched}")

    if outcome.excluded:
        print(f"\n[ EARLY EXCLUSIONS ]")
        for e in outcome.excluded:
            print(f"  {e['name']}  →  {e['reason']}")

    if outcome.warnings:
        print(f"\n[ VALIDATOR WARNINGS ]")
        for w in outcome.warnings:
            print(f"  ⚠  {w}")

    print(f"\n[ TOP RESULTS ]")
    for rank, r in enumerate(outcome.top_results, 1):
        print(f"\n  #{rank}  {r['name']}")
        print(f"  {SEP2}")
        print(f"  title              : {r['title']}")
        print(f"  advisor_eligibility: {r['advisor_eligibility']}  ({r['eligibility_reason']})")
        print(f"  matching_score     : {r['matching_score']:.3f}")
        print(f"  profile_score      : {r['profile_score']:.3f}")
        print(f"  scholar_score      : {r['scholar_score']:.3f}")
        print(f"  scholar_url        : {r['scholar_url'] or '(none)'}")
        print(f"  scholar_source     : {r['scholar_source'] or '—'}")
        print(f"  profile_confidence : {r['profile_confidence']}")
        print(f"  source_sections    : {r['source_sections']}")
        ri = r['research_interests']
        print(f"  research_interests : {ri[:120]}{'...' if len(ri)>120 else ''}")
        print(f"  match_reason       : {r['match_reason']}")
        print(f"  outreach_advice    : {r['outreach_advice']}")
        print(f"  profile_url        : {r['profile_url']}")
        if len(r['profile_urls']) > 1:
            print(f"  profile_urls       : {r['profile_urls']}")


if __name__ == "__main__":
    print(f"\n{SEP}")
    print(f"USER PROFILE")
    print(SEP)
    print(f"  research: {TEST_USER.research_interests}")
    print(f"  keywords: {TEST_USER.keywords}")

    print(f"\n{SEP}")
    print(f"STEP 1 — Running Extraction Agent on {TEST_URL}")
    print(SEP)
    t0 = time.time()
    extraction = run_extraction_agent(TEST_URL)
    t1 = time.time()
    print(f"  extracted: {extraction.faculty_count} people  score={extraction.validator_score:.2f}  ({t1-t0:.1f}s)")
    for f in extraction.faculty_list:
        sections = [sl.get("source_section","") for sl in f.get("source_links",[])]
        print(f"  {f['name']:25s}  role={f.get('role_hint','?'):15s}  sections={sections}")

    print(f"\n{SEP}")
    print(f"STEP 2 — Running Matching Agent")
    print(SEP)
    t2 = time.time()
    outcome = run_matching_agent(
        TEST_URL,
        TEST_USER,
        extraction_outcome=extraction,   # pass extraction to avoid re-fetch
    )
    t3 = time.time()
    print(f"  matching done in {t3-t2:.1f}s")

    print_outcome(outcome)

    print(f"\n{SEP}")
    print(f"Total runtime: {t3-t0:.1f}s")
    print(SEP)
