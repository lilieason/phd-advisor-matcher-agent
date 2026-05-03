"""
Extraction Agent validation script.
Runs run_extraction_agent on each test URL and prints a structured report.
No CV pipeline, no LLM calls — extraction only.
"""

import json
import logging
import sys
from pathlib import Path

# Suppress INFO noise from HTTP requests; keep our agent logs
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
# Re-enable only extraction_agent and advisor_server logger at INFO
for name in ("mcp_servers.extraction_agent", "mcp_servers.advisor_server"):
    logging.getLogger(name).setLevel(logging.INFO)

sys.path.insert(0, str(Path(__file__).parent))
from mcp_servers.extraction_agent import run_extraction_agent, _kahuna_load_context

URLS = [
    "https://engineering.uci.edu/dept/cee/academics/graduate/programs/transportation",
    "https://www.mccormick.northwestern.edu/civil-environmental/research/areas/human-systems.html#modeling",
    "https://ieor.berkeley.edu/research/operations-supply-chain-logistics-production/",
    "https://www.ce.washington.edu/research/transportation",
    "https://cds.nyu.edu/phd-areas-faculty/",
    "https://www.cee.ucla.edu/transportation-engineering/",
]

SEP = "=" * 72


def fmt_list(items, limit=5):
    if not items:
        return "(none)"
    shown = items[:limit]
    rest = len(items) - limit
    s = ", ".join(f'"{x}"' for x in shown)
    if rest > 0:
        s += f" … +{rest} more"
    return s


def run_one(url):
    print(f"\n{SEP}")
    print(f"URL: {url}")
    print(SEP)

    # Load memory BEFORE running (shows what the agent will see)
    mem_before = _kahuna_load_context(url)

    print("\n[ MEMORY — before extraction ]")
    if mem_before:
        print(f"  last_strategy  : {mem_before.get('last_strategy')}")
        print(f"  last_success   : {mem_before.get('last_success')}")
        print(f"  best_strategy  : {mem_before.get('best_strategy')}")
        print(f"  failed_strats  : {mem_before.get('failed_strategies')}")
        print(f"  past_runs      : {len(mem_before.get('past_outcomes', []))}")
    else:
        print("  (no prior memory for this domain)")

    # Run the agent
    outcome = run_extraction_agent(url)

    # ── Planner ───────────────────────────────────────────────────────────────
    print("\n[ PLANNER ]")
    print(f"  page_representation : {outcome.page_representation}")
    trace = outcome.strategy_trace
    print(f"  strategies tried    : {[t['strategy'] for t in trace]}")
    print(f"  memory used         : {'yes' if mem_before else 'no (first visit)'}")
    if mem_before and mem_before.get("best_strategy"):
        print(f"  kahuna hint         : promote {mem_before['best_strategy']}")

    # ── Execution + Validation trace ──────────────────────────────────────────
    print("\n[ EXECUTION + VALIDATION TRACE ]")
    for i, t in enumerate(trace, 1):
        status = "✓ PASS" if t["success"] else "✗ FAIL"
        print(f"  Attempt {i}: {t['strategy']}")
        print(f"    faculty_count    : {t['faculty_count']}")
        print(f"    validator_score  : {t['validator_score']:.3f}  {status}")
        if t["issues"]:
            print(f"    issues           : {t['issues']}")

    # ── Final result ──────────────────────────────────────────────────────────
    print("\n[ FINAL RESULT ]")
    print(f"  strategy_used   : {outcome.strategy_used}")
    print(f"  faculty_count   : {outcome.faculty_count}")
    print(f"  validator_score : {outcome.validator_score:.3f}")
    print(f"  success         : {outcome.success}")
    if outcome.failure_reason:
        print(f"  failure_reason  : {outcome.failure_reason}")
    if outcome.next_best_strategy:
        print(f"  next_best       : {outcome.next_best_strategy}")
    if outcome.faculty_names_sample:
        print(f"  sample names    : {fmt_list(outcome.faculty_names_sample)}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n[ DIAGNOSTICS ]")
    has_topic = any("topic_leakage" in t["issues"] for t in trace)
    has_pub   = any("publication_leakage" in t["issues"] for t in trace)
    is_inter  = outcome.page_representation == "interactive"
    js_blocked = any("js_blocked" in t["issues"] for t in trace)

    print(f"  grouped heading leakage detected : {'YES — caught' if has_topic else 'no'}")
    print(f"  publication leakage detected     : {'YES — caught' if has_pub else 'no'}")
    if is_inter:
        print(f"  interactive page classified      : YES")
        print(f"  JS-blocked (unresolvable)        : {'YES' if js_blocked else 'no — triggers resolved'}")
        if outcome.error_payload:
            groups = outcome.error_payload.get("groups", [])
            print(f"  group metadata returned         : {len(groups)} groups")
            for g in groups[:5]:
                print(f"    • {g.get('label','?')} → {g.get('href','')[:60] or '(no href)'}")

    # error payload?
    if outcome.error_payload and not is_inter:
        print(f"  error payload   : {outcome.error_payload.get('error','?')[:100]}")

    # ── Hidden-content detection ──────────────────────────────────────────────
    if outcome.hidden_content_detected:
        print("\n[ HIDDEN CONTENT ]")
        print(f"  hidden_content_detected      : YES")
        print(f"  requires_browser_interaction : {outcome.requires_browser_interaction}")
        if outcome.hidden_content_warning:
            print(f"  warning : {outcome.hidden_content_warning}")

    # ── Memory written ────────────────────────────────────────────────────────
    print("\n[ MEMORY — written ]")
    print(f"  recorded strategy : {outcome.strategy_used}")
    print(f"  recorded score    : {outcome.validator_score:.3f}")
    print(f"  recorded success  : {outcome.success}")
    if outcome.issues:
        print(f"  recorded issues   : {outcome.issues}")
    if outcome.failure_reason:
        print(f"  failure_reason    : {outcome.failure_reason}")


if __name__ == "__main__":
    for url in URLS:
        try:
            run_one(url)
        except Exception as exc:
            print(f"\n  [ERROR] {exc}")

    print(f"\n{SEP}")
    print("All URLs processed.")
    print(SEP)
