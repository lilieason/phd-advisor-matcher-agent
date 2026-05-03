#!/usr/bin/env python3
"""
Advisor Matching Pipeline — Reusable Debug Runner
==================================================
Run after every agent change to get full per-stage timing, counts, and a
saved JSON + Markdown report.

Usage:
  python scripts/debug_pipeline.py --url URL --cv /path/to/cv.txt
  python scripts/debug_pipeline.py --url URL --cv /path/to/cv.pdf
  python scripts/debug_pipeline.py --url URL --cv-text-file /path/to/cv.txt
  python scripts/debug_pipeline.py --url URL --cv cv.txt --limit 5 --skip-scholar
  python scripts/debug_pipeline.py --url URL --cv cv.txt --fast   # limit=5 + skip-scholar

Outputs:
  outputs/debug_runs/debug_<timestamp>.json
  outputs/debug_runs/debug_<timestamp>.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlparse
import textwrap
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Terminal colour helpers ───────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

def ok(s):   return _c("32;1", s)
def err(s):  return _c("31;1", s)
def warn(s): return _c("33;1", s)
def hdr(s):  return _c("36;1", s)
def dim(s):  return _c("2",    s)
def bold(s): return _c("1",    s)

BAR  = "─" * 72
BAR2 = "═" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Stage-event collector
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STAGES = [
    ("A",  "Candidate Intake"),
    ("B",  "Early Exclusion"),
    ("C",  "Profile Fetch"),
    ("C5", "Pre-Filter"),
    ("OA", "OpenAlex"),
    ("P",  "Planner"),
    ("D",  "Scholar Acquisition"),
    ("E",  "LLM Scoring"),
    ("F",  "Recommendation"),
]

class EventCollector:
    """Collects progress_cb events and reconstructs per-stage timing."""

    def __init__(self, verbose: bool = True):
        self.verbose   = verbose
        self.events: list[dict] = []
        self._stage_ts: dict[str, float] = {}   # stage → start ts
        self.stages: dict[str, dict] = {}        # stage → {label, start, end, duration, in, out, ...}

    def __call__(self, ev: dict) -> None:
        self.events.append(ev)
        t    = ev.get("type", "")
        stg  = ev.get("stage", "")

        if t == "stage_start":
            self._stage_ts[stg] = ev.get("ts", time.time())
            self.stages[stg] = {
                "label":    ev.get("label", stg),
                "start_ts": ev.get("ts", 0),
                "end_ts":   None,
                "duration": None,
                "in":       ev.get("in"),
                "out":      None,
            }
        elif t == "stage_end":
            start = self._stage_ts.get(stg, ev.get("ts", 0))
            end   = ev.get("ts", time.time())
            if stg not in self.stages:
                self.stages[stg] = {"label": ev.get("label", stg), "start_ts": start}
            self.stages[stg].update({
                "end_ts":   end,
                "duration": round(end - start, 2),
                "out":      ev.get("out"),
            })
            # Copy any extra fields from the event
            for k, v in ev.items():
                if k not in ("type", "stage", "label", "ts"):
                    self.stages[stg][k] = v

        if not self.verbose:
            return

        if t == "stage_start":
            _lbl = ev.get("label", stg)
            print(f"  {dim(f'[{stg}] {_lbl}  started…')}")
        elif t == "stage_end":
            dur = self.stages.get(stg, {}).get("duration")
            dur_str = f"{dur:.1f}s" if dur is not None else "?"
            _end_msg = f"[{stg}] done  {dur_str}"
            print(f"  {dim(_end_msg)}")
        elif t == "batch_progress":
            cur   = ev.get("current", "")
            total = ev.get("total", "")
            cnt   = f"[{cur}/{total}] " if cur and total else ""
            print(f"    {dim('·')} {cnt}{ev.get('message','')}")
        elif t == "progress":
            print(f"    {dim('·')} {ev.get('message','')}")
        elif t == "error":
            print(f"    {err('ERR')} {ev.get('message','')}")

    def stage_table(self) -> list[dict]:
        rows = []
        for code, label in PIPELINE_STAGES:
            s = self.stages.get(code)
            if s:
                rows.append({
                    "stage":    code,
                    "label":    s.get("label", label),
                    "duration": s.get("duration"),
                    "in":       s.get("in"),
                    "out":      s.get("out"),
                    "extra":    {k: v for k, v in s.items()
                                 if k not in ("label","start_ts","end_ts","duration","in","out")},
                })
            else:
                rows.append({"stage": code, "label": label,
                              "duration": None, "in": None, "out": None, "extra": {}})
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# CV loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_cv(cv_path: Optional[str], cv_text_file: Optional[str]) -> str:
    if cv_text_file:
        p = Path(cv_text_file)
        if not p.exists():
            raise FileNotFoundError(f"--cv-text-file not found: {cv_text_file}")
        return p.read_text(encoding="utf-8")

    if cv_path:
        p = Path(cv_path)
        if not p.exists():
            raise FileNotFoundError(f"--cv not found: {cv_path}")
        if p.suffix.lower() == ".pdf":
            return _extract_pdf_text(p)
        return p.read_text(encoding="utf-8")

    raise ValueError("Provide --cv or --cv-text-file")


def _extract_pdf_text(path: Path) -> str:
    try:
        import pdfminer.high_level as _pdf  # type: ignore
        return _pdf.extract_text(str(path))
    except ImportError:
        pass
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    raise RuntimeError(
        "PDF support requires pdfminer.six or pypdf:\n"
        "  /Users/lisun/aurite_project/venv/bin/pip install pdfminer.six"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Console display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage_header(label: str) -> float:
    t = time.perf_counter()
    print(f"\n{hdr(BAR)}")
    print(f"  {bold(label)}")
    print(hdr(BAR))
    return t


def _print_stage_result(label: str, t0: float, success: bool,
                        note: str = "", error: str = "") -> float:
    dur = time.perf_counter() - t0
    sym = ok("✓") if success else err("✗")
    print(f"  {sym}  {dim(f'{dur:.1f}s')}  {note}")
    if error:
        print(f"  {err('ERROR:')} {error[:200]}")
    return dur


def _print_result(r: dict, rank: int) -> None:
    name    = r.get("name", "?")
    overall = r.get("overall_score") or r.get("overall_match") or "?"
    sc      = r.get("scoring") or {}
    fpm     = sc.get("faculty_profile_match", "?")
    rsm     = sc.get("recent_scholar_match") or "—"
    dims    = sc.get("dimensions") or {}
    sigs    = r.get("signals") or {}
    topics  = (r.get("openalex_topics") or [])[:3]
    reason  = (r.get("match_reason") or "")[:80]

    print(f"\n  {bold(f'#{rank}  {name}')}")
    print(f"     overall={bold(str(overall))}  profile={fpm}  scholar_match={rsm}")
    print(f"     research={dims.get('research','?')}  method={dims.get('method','?')}  "
          f"app={dims.get('application','?')}  style={dims.get('style','?')}")
    print(f"     activity={sigs.get('activity','?')}  recruiting={sigs.get('recruiting','?')}  "
          f"funding={sigs.get('funding','?')}")
    if r.get("profile_url"):
        print(f"     profile:  {dim(r['profile_url'])}")
    gs  = r.get("scholar_url") or r.get("google_scholar")
    src = r.get("scholar_source", "")
    if gs:
        src_tag = f"  [{src}]" if src and src not in ("profile", "") else ""
        print(f"     scholar:  {dim(str(gs))}{dim(src_tag)}")
    elif src in ("not_searched", "openalex_sufficient", "not_found"):
        print(f"     scholar:  {dim(f'({src})')}")
    if r.get("openalex_url"):
        print(f"     openalex: {dim(str(r['openalex_url']))}")
    if topics:
        print(f"     topics:   {dim(', '.join(topics))}")
    if reason:
        print(f"     reason:   {dim(reason)}")


def _print_stage_table(collector: EventCollector, total_s: float) -> None:
    rows = collector.stage_table()
    print(f"\n{BAR}")
    print(f"  {'Stage':<6}  {'Label':<22}  {'Time':>7}  {'In':>6}  {'Out':>6}")
    print(BAR)
    for r in rows:
        dur = f"{r['duration']:.1f}s" if r["duration"] is not None else "—"
        i   = str(r["in"])  if r["in"]  is not None else "—"
        o   = str(r["out"]) if r["out"] is not None else "—"
        bar = ok("✓") if r["duration"] is not None else dim("·")
        print(f"  {bar} {r['stage']:<5}  {r['label']:<22}  {dur:>7}  {i:>6}  {o:>6}")
        if r.get("extra"):
            ex = r["extra"]
            extras = []
            for k, v in ex.items():
                if k not in ("excluded_names",):
                    extras.append(f"{k}={v}")
            if extras:
                print(f"           {dim('  ' + '  '.join(extras))}")
    print(BAR)
    print(f"  {'':6}  {'TOTAL':<22}  {total_s:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(
    url: str,
    cv_text: str,
    verbose: bool = True,
    limit: Optional[int] = None,
    skip_scholar: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Full pipeline debug run. Returns the debug dict (never calls sys.exit).
    """
    from datetime import datetime as _dt
    from mcp_servers.extraction_agent import run_extraction_agent, ExtractionOutcome
    from mcp_servers.matching_agent   import run_matching_agent, UserProfile

    if output_dir is None:
        output_dir = _ROOT / "outputs" / "debug_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts        = _dt.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"debug_{ts}.json"
    md_path   = output_dir / f"debug_{ts}.md"

    t_global  = time.perf_counter()
    errors: list[str] = []
    warnings: list[str] = []

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{BAR2}")
    print(f"  {bold('ADVISOR MATCHING — DEBUG RUN')}")
    print(f"  URL          : {url}")
    print(f"  CV           : {len(cv_text)} chars")
    print(f"  limit        : {limit or 'none'}")
    print(f"  skip_scholar : {skip_scholar}")
    print(f"  time         : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(BAR2)

    debug: dict = {
        "input": {
            "url":          url,
            "cv_len":       len(cv_text),
            "cv_head":      cv_text[:300].replace("\n", " "),
            "limit":        limit,
            "skip_scholar": skip_scholar,
        },
        "environment": {
            "python":    sys.version.split()[0],
            "cwd":       str(Path.cwd()),
            "venv":      os.environ.get("VIRTUAL_ENV", ""),
            "timestamp": ts,
        },
        "timings":    {"total": 0.0, "extraction": 0.0, "matching": 0.0, "stages": []},
        "extraction": {},
        "matching":   {"success": False},
        "stages":     [],
        "per_professor": [],
        "errors":     [],
        "warnings":   [],
    }

    # ── STAGE 1: Extraction ───────────────────────────────────────────────────
    t0 = _print_stage_header("STAGE 1 — Extraction  (run_extraction_agent)")
    extraction_outcome = None
    try:
        extraction_outcome = run_extraction_agent(url)
        fl = extraction_outcome.faculty_list or []

        debug["extraction"] = {
            "success":        extraction_outcome.success,
            "faculty_count":  extraction_outcome.faculty_count,
            "page_class":     extraction_outcome.page_representation,
            "strategy_used":  extraction_outcome.strategy_used,
            "validator_score":extraction_outcome.validator_score,
            "issues":         extraction_outcome.issues,
            "failure_reason": extraction_outcome.failure_reason,
            "faculty_sample": [
                {"name": f.get("name"), "url": f.get("full_profile_url") or f.get("profile_url")}
                for f in fl[:12]
            ],
        }

        if verbose and fl:
            print(f"  Faculty sample (first {min(len(fl), 10)}):")
            for f in fl[:10]:
                u = f.get("full_profile_url") or f.get("profile_url") or "—"
                print(f"    {dim('·')} {(f.get('name') or '?'):<30} {dim(u[:55])}")

        # Single-profile fallback — mirrors web_app.py logic
        if not fl:
            print(f"  {warn('No faculty list → treating as single-profile URL')}")
            extraction_outcome = ExtractionOutcome(
                url=url,
                domain="",
                page_representation="single_profile",
                strategy_used="direct",
                faculty_count=1,
                faculty_names_sample=[],
                validator_score=1.0,
                issues=["single profile URL — no directory extraction"],
                success=True,
                failure_reason=None,
                next_best_strategy=None,
                strategy_trace=[],
                timestamp=_dt.utcnow().strftime("%Y%m%dT%H%M%SZ"),
                faculty_list=[{"full_profile_url": url, "name": ""}],
            )
            debug["extraction"]["mode"] = "single_profile_fallback"

        note = (f"{len(extraction_outcome.faculty_list)} faculty  "
                f"class={extraction_outcome.page_representation}  "
                f"strategy={extraction_outcome.strategy_used}")
        dur1 = _print_stage_result("Extraction", t0, success=True, note=note)

    except Exception as exc:
        tb = traceback.format_exc()
        dur1 = _print_stage_result("Extraction", t0, success=False, error=str(exc))
        errors.append(f"Extraction exception: {exc}")
        debug["extraction"] = {"success": False, "error": str(exc), "traceback": tb}
        print(f"\n{err(tb)}")

    debug["timings"]["extraction"] = round(dur1, 2)

    if not extraction_outcome or not extraction_outcome.success:
        print(f"\n{err('Extraction failed — aborting.')}")
        debug["errors"] = errors
        debug["timings"]["total"] = round(time.perf_counter() - t_global, 2)
        _save_outputs(debug, json_path, md_path)
        return debug

    # ── STAGE 2: Matching pipeline ────────────────────────────────────────────
    t0 = _print_stage_header(
        "STAGE 2 — Matching Pipeline  "
        "(A→B→C+C.5→P→D→E→F)"
    )
    collector = EventCollector(verbose=verbose)
    matching_outcome = None
    try:
        user_profile = UserProfile(research_interests=cv_text)
        matching_outcome = run_matching_agent(
            url,
            user_profile,
            extraction_outcome=extraction_outcome,
            progress_cb=collector,
            debug_limit=limit,
            skip_scholar=skip_scholar,
        )
        n = len(matching_outcome.top_results)
        note = (f"{n} results  "
                f"candidates={matching_outcome.total_candidates}  "
                f"excl={matching_outcome.total_candidates - matching_outcome.after_early_exclusion}  "
                f"enriched={matching_outcome.enriched_count}  "
                f"scholar_searched={matching_outcome.scholar_searched}")
        dur2 = _print_stage_result("Matching", t0, success=True, note=note)

        debug["matching"] = {
            "success":          True,
            "total_candidates": matching_outcome.total_candidates,
            "after_exclusion":  matching_outcome.after_early_exclusion,
            "enriched_count":   matching_outcome.enriched_count,
            "scholar_searched": matching_outcome.scholar_searched,
            "excluded":         matching_outcome.excluded,
        }
        for w in matching_outcome.warnings:
            warnings.append(f"[matching] {w}")

    except Exception as exc:
        tb = traceback.format_exc()
        dur2 = _print_stage_result("Matching", t0, success=False, error=str(exc))
        errors.append(f"Matching exception: {exc}")
        debug["matching"] = {"success": False, "error": str(exc), "traceback": tb}
        print(f"\n{err(tb)}")

    debug["timings"]["matching"] = round(dur2, 2)
    debug["timings"]["stages"]   = collector.stage_table()

    # ── Per-stage timing table ────────────────────────────────────────────────
    total_s = time.perf_counter() - t_global
    _print_stage_table(collector, total_s)

    # ── Results display ───────────────────────────────────────────────────────
    if matching_outcome and matching_outcome.top_results:
        top = matching_outcome.top_results
        print(f"\n{BAR2}")
        print(f"  {bold('TOP ' + str(len(top)) + ' RESULTS')}")
        print(BAR2)
        for i, r in enumerate(top, 1):
            _print_result(r, i)

        # Per-professor structured record
        debug["per_professor"] = []
        for r in top:
            sc   = r.get("scoring") or {}
            sigs = r.get("signals") or {}
            dims = sc.get("dimensions") or {}
            debug["per_professor"].append({
                "rank":            r.get("rank"),
                "name":            r.get("name"),
                "overall_score":   r.get("overall_score") or r.get("overall_match"),
                "faculty_profile_match": sc.get("faculty_profile_match"),
                "recent_scholar_match":  sc.get("recent_scholar_match"),
                "dimensions":      dims,
                "signals":         sigs,
                "openalex_topics": r.get("openalex_topics", []),
                "profile_url":             r.get("profile_url"),
                "scholar_url":             r.get("scholar_url") or r.get("google_scholar"),
                "scholar_source":          r.get("scholar_source", ""),
                "scholar_match_confidence": r.get("scholar_match_confidence") or r.get("scholar_confidence", ""),
                "openalex_url":            r.get("openalex_url"),
                "orcid":                   r.get("orcid"),
                "match_reason":            r.get("match_reason"),
                "cold_email":              r.get("cold_email"),
            })
        debug["matching"]["top_results"] = matching_outcome.top_results
    else:
        print(f"\n{warn('No results returned.')}")

    # ── Errors / warnings ─────────────────────────────────────────────────────
    all_issues = errors + warnings
    if all_issues:
        print(f"\n{BAR}")
        print(f"  {warn('ISSUES (' + str(len(all_issues)) + ')')}")
        print(BAR)
        for e in all_issues[:15]:
            print(f"  {warn('▸')} {e[:160]}")

    # ── Bottleneck detection ──────────────────────────────────────────────────
    bottleneck = _find_bottleneck(collector.stage_table(), total_s)

    # ── Finalize ──────────────────────────────────────────────────────────────
    debug["timings"]["total"]    = round(total_s, 2)
    debug["errors"]              = errors
    debug["warnings"]            = warnings
    debug["bottleneck"]          = bottleneck
    debug["all_events"]          = collector.events  # raw events for deep inspection

    _save_outputs(debug, json_path, md_path)

    print(f"\n{BAR2}")
    print(f"  {bold('DEBUG RUN COMPLETE')}")
    print(f"  Total     : {bold(f'{total_s:.1f}s')}")
    print(f"  Bottleneck: {bottleneck}")
    print(f"  JSON      : {json_path}")
    print(f"  Markdown  : {md_path}")
    print(BAR2 + "\n")

    return debug


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

_SCHOLAR_SOURCE_LABELS = {
    "profile":              "direct link in faculty profile",
    "personal_website":     "found on personal website",
    "personal_website_cv":  "found via CV on personal website",
    "cv":                   "found in faculty CV",
    "search_engine":        "DDG/search engine",
    "skipped_flag":         "skipped (--skip-scholar flag)",
    "not_searched":         "not searched (outside top-N)",
    "not_found":            "not found",
}


def _fmt_scholar_line(r: dict) -> str:
    url  = r.get("scholar_url") or ""
    src  = r.get("scholar_source") or ""
    conf = r.get("scholar_match_confidence") or ""
    label = _SCHOLAR_SOURCE_LABELS.get(src, src)
    if url:
        tag = f"  _(via {label}" + (f", conf={conf}" if conf else "") + ")_"
        return f"- Scholar: {url}{tag}"
    elif src:
        return f"- Scholar: — _(via {label})_"
    return "- Scholar: —"


def _find_bottleneck(stage_rows: list[dict], total_s: float) -> str:
    timed = [(r["label"], r["duration"])
             for r in stage_rows if r["duration"] is not None]
    if not timed:
        return "no stage timing available"
    slowest_label, slowest_dur = max(timed, key=lambda x: x[1])
    pct = round(slowest_dur / total_s * 100) if total_s > 0 else 0
    return f"{slowest_label}  ({slowest_dur:.1f}s = {pct}% of total)"


_KAHUNA_KNOWLEDGE_DIR = Path.home() / ".kahuna" / "knowledge"
_KAHUNA_KEEP_RECENT = 20   # rolling window for timing history


def _save_run_to_kahuna(debug: dict) -> None:
    """Upsert per-URL aggregate stats into Kahuna. One file per hostname, never appending."""
    try:
        _KAHUNA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        inp    = debug.get("input", {})
        mat    = debug.get("matching", {})
        ext    = debug.get("extraction", {})
        profs  = debug.get("per_professor", [])
        stages = debug.get("timings", {}).get("stages", [])
        total_s = debug.get("timings", {}).get("total", 0) or 0

        url       = inp.get("url", "")
        dept_slug = re.sub(r"[^\w]", "-", url.split("//")[-1].split("/")[0])[:40]
        safe_dept = re.sub(r"[^\w\-]", "_", dept_slug)
        date_now  = datetime.now().strftime("%Y-%m-%d")
        ts_now    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # ── Load or initialise aggregate stats ──────────────────────────────────
        sidecar = _KAHUNA_KNOWLEDGE_DIR / f"debug-run-{safe_dept}.json"
        if sidecar.exists():
            agg: dict = json.loads(sidecar.read_text())
        else:
            agg = {
                "url": url, "hostname": dept_slug,
                "first_run": date_now, "run_count": 0,
                "faculty_count": None,
                "timing_total": [],        # rolling list of total_s values
                "stage_timings": {},       # {stage_id: [dur, ...]}
                "extraction_strategies": {},
                "page_classes": {},
                "bottlenecks": {},         # {label: count}
                "planner_skip_rates": [],  # rolling list of skip%
                "scholar_sources": {},     # {source: count}
                "recent_runs": [],         # last 3 summaries
            }

        # ── Merge this run into aggregate ───────────────────────────────────────
        agg["last_run"]     = ts_now
        agg["run_count"]   += 1
        agg["faculty_count"] = mat.get("total_candidates", ext.get("faculty_count"))

        if total_s:
            agg["timing_total"] = (agg["timing_total"] + [round(total_s, 1)])[-_KAHUNA_KEEP_RECENT:]

        for r in stages:
            if r.get("duration") is not None:
                sid = r["stage"]
                agg["stage_timings"].setdefault(sid, [])
                agg["stage_timings"][sid] = (
                    agg["stage_timings"][sid] + [round(r["duration"], 1)]
                )[-_KAHUNA_KEEP_RECENT:]

            # planner skip rate
            if r["stage"] == "P":
                extra = r.get("extra") or {}
                skips   = extra.get("skip_scholar", 0) or 0
                total_p = (extra.get("priority") or {})
                n_cands = sum(total_p.values()) if total_p else agg["faculty_count"] or 1
                rate    = round(skips / n_cands * 100) if n_cands else 0
                agg["planner_skip_rates"] = (
                    agg["planner_skip_rates"] + [rate]
                )[-_KAHUNA_KEEP_RECENT:]

        strat = ext.get("strategy_used", "")
        if strat:
            agg["extraction_strategies"][strat] = agg["extraction_strategies"].get(strat, 0) + 1

        pc = ext.get("page_class", "")
        if pc:
            agg["page_classes"][pc] = agg["page_classes"].get(pc, 0) + 1

        bn_label = (debug.get("bottleneck") or "").split("  (")[0].strip()
        if bn_label:
            agg["bottlenecks"][bn_label] = agg["bottlenecks"].get(bn_label, 0) + 1

        for p in profs:
            src = p.get("scholar_source", "")
            if src:
                agg["scholar_sources"][src] = agg["scholar_sources"].get(src, 0) + 1

        top3 = [p.get("name", "?") for p in profs[:3]]
        agg["recent_runs"] = (agg["recent_runs"] + [{
            "ts": ts_now, "total_s": round(total_s, 1),
            "bottleneck": bn_label, "top3": top3,
        }])[-3:]

        sidecar.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        _write_kahuna_mdc(agg, safe_dept)

    except Exception:
        pass  # never crash the debug run for a Kahuna write failure


def _write_kahuna_mdc(agg: dict, safe_dept: str) -> None:
    n   = agg["run_count"]
    url = agg["url"]

    def _stat(vals: list) -> str:
        if not vals:
            return "—"
        return f"avg={sum(vals)/len(vals):.1f}s  min={min(vals):.1f}s  max={max(vals):.1f}s"

    timing_str = _stat(agg["timing_total"])

    stage_rows = []
    stage_order = ["A", "B", "C", "C5", "OA", "P", "D", "E", "F"]
    for sid in stage_order:
        vals = agg["stage_timings"].get(sid, [])
        if vals:
            stage_rows.append(
                f"| {sid} | {sum(vals)/len(vals):.1f}s"
                f" | {min(vals):.1f}s | {max(vals):.1f}s |"
            )

    bn_items  = sorted(agg["bottlenecks"].items(), key=lambda x: -x[1])
    bn_lines  = [f"- **{k}**: {v}/{n} runs ({round(v/n*100)}%)" for k, v in bn_items]

    ss_total  = sum(agg["scholar_sources"].values())
    ss_items  = sorted(agg["scholar_sources"].items(), key=lambda x: -x[1])
    ss_lines  = [f"- {k}: {v} ({round(v/ss_total*100)}%)" for k, v in ss_items] if ss_total else ["—"]

    sr = agg["planner_skip_rates"]
    skip_str  = f"avg {sum(sr)/len(sr):.0f}%  (min {min(sr)}%  max {max(sr)}%)" if sr else "—"

    strat_str = "  ".join(f"{k}×{v}" for k, v in agg["extraction_strategies"].items()) or "—"
    pc_str    = "  ".join(f"{k}×{v}" for k, v in agg["page_classes"].items()) or "—"

    rec_lines = [
        f"- {r['ts'][:16]}  {r['total_s']}s  [{r['bottleneck']}]  "
        f"top3: {', '.join(r['top3'])}"
        for r in reversed(agg["recent_runs"])
    ]

    dom_bn    = bn_items[0][0] if bn_items else "—"
    ts_iso    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    first_iso = agg.get("first_run", ts_iso[:10]) + "T00:00:00.000Z"

    content = (
        f"---\n"
        f"type: knowledge\n"
        f"title: \"Pipeline History — {agg['hostname']} ({n} runs)\"\n"
        f"summary: \"{url}. {n} runs. Timing: {timing_str}. Main bottleneck: {dom_bn}.\"\n"
        f"created_at: {first_iso}\n"
        f"updated_at: {ts_iso}\n"
        f"source:\n"
        f"  file: \"debug-run-{safe_dept}.mdc\"\n"
        f"  project: \"paper-reader\"\n"
        f"classification:\n"
        f"  category: context\n"
        f"  confidence: 1.0\n"
        f"  topics: [\"pipeline-history\", \"timing\", \"scholar\", \"planner\"]\n"
        f"status: active\n"
        f"---\n\n"
        f"# Pipeline History — {url}\n\n"
        f"**Runs**: {n}  |  **Faculty**: {agg.get('faculty_count','?')}"
        f"  |  **First**: {agg.get('first_run','?')}  |  **Last**: {agg.get('last_run','?')[:10]}\n\n"
        f"## Timing ({n} runs)\n\n"
        f"Overall: {timing_str}\n\n"
        f"| Stage | Avg | Min | Max |\n"
        f"|-------|-----|-----|-----|\n"
        + "\n".join(stage_rows) + "\n\n"
        f"## Bottleneck Distribution\n\n"
        + "\n".join(bn_lines) + "\n\n"
        f"## Scholar Source Distribution ({ss_total} observations)\n\n"
        + "\n".join(ss_lines) + "\n\n"
        f"## Planner Scholar-Skip Rate\n\n"
        f"{skip_str}\n\n"
        f"## Extraction\n\n"
        f"- Strategies: {strat_str}\n"
        f"- Page classes: {pc_str}\n\n"
        f"## Recent Runs\n\n"
        + "\n".join(rec_lines) + "\n"
    )

    out = _KAHUNA_KNOWLEDGE_DIR / f"debug-run-{safe_dept}.mdc"
    out.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _save_outputs(debug: dict, json_path: Path, md_path: Path) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2, default=str)
    md_path.write_text(_build_markdown(debug), encoding="utf-8")
    _save_run_to_kahuna(debug)
    # Auto-open the markdown report on macOS
    try:
        subprocess.Popen(["open", str(md_path)])
    except Exception:
        pass


def _build_markdown(d: dict) -> str:
    ts   = d["environment"]["timestamp"]
    inp  = d["input"]
    tim  = d["timings"]
    ext  = d.get("extraction", {})
    mat  = d.get("matching", {})
    errs = d.get("errors", [])
    warn_list = d.get("warnings", [])
    btl  = d.get("bottleneck", "")
    profs = d.get("per_professor", [])
    stage_rows = tim.get("stages", [])

    lines = [
        f"# Debug Run — {ts}",
        "",
        "## Input",
        f"- **URL**: `{inp.get('url','')}`",
        f"- **CV**: {inp.get('cv_len',0)} chars",
        f"- **limit**: `{inp.get('limit') or 'none'}`",
        f"- **skip_scholar**: `{inp.get('skip_scholar', False)}`",
        f"- **CV preview**: _{inp.get('cv_head','')[:120]}_",
        "",
        "## Timing Overview",
        "",
        f"| Phase | Duration |",
        f"|-------|---------|",
        f"| Extraction | {tim.get('extraction','?')}s |",
        f"| Matching | {tim.get('matching','?')}s |",
        f"| **Total** | **{tim.get('total','?')}s** |",
        "",
        f"> **Bottleneck**: {btl}",
        "",
        "## Pipeline Stage Breakdown",
        "",
        "| Stage | Label | Duration | In | Out | Notes |",
        "|-------|-------|----------|----|-----|-------|",
    ]
    for r in stage_rows:
        dur   = f"{r['duration']:.1f}s" if r["duration"] is not None else "—"
        i     = str(r["in"])  if r["in"]  is not None else "—"
        o     = str(r["out"]) if r["out"] is not None else "—"
        extra = r.get("extra") or {}
        notes_parts = []
        for k, v in extra.items():
            if k not in ("excluded_names",):
                notes_parts.append(f"{k}={v}")
        notes = " · ".join(notes_parts) if notes_parts else ""
        lines.append(f"| {r['stage']} | {r['label']} | {dur} | {i} | {o} | {notes} |")
    lines.append("")

    # Extraction
    lines += [
        "## Extraction Result",
        f"- **Success**: {ext.get('success')}",
        f"- **Faculty found**: {ext.get('faculty_count','?')}",
        f"- **Page class**: {ext.get('page_class','?')}",
        f"- **Strategy**: {ext.get('strategy_used','?')}",
        f"- **Validator score**: {ext.get('validator_score','?')}",
    ]
    if ext.get("mode"):
        lines.append(f"- **Mode**: {ext['mode']}")
    if ext.get("issues"):
        lines.append(f"- **Issues**: {'; '.join(ext['issues'])}")
    fl = ext.get("faculty_sample", [])
    if fl:
        lines += ["", "**Faculty sample:**"]
        for f in fl[:10]:
            lines.append(f"  - `{f.get('name','?')}` — `{f.get('url','')}`")
    lines.append("")

    # Matching summary
    lines += [
        "## Matching Summary",
        f"- **Success**: {mat.get('success')}",
        f"- **Total candidates**: {mat.get('total_candidates','?')}",
        f"- **After exclusion**: {mat.get('after_exclusion','?')}",
        f"- **Enriched**: {mat.get('enriched_count','?')}",
        f"- **Scholar searched**: {mat.get('scholar_searched','?')}",
        "",
    ]

    # Results table
    if profs:
        lines += [
            "## Top Faculty Results",
            "",
            "| # | Name | Overall | Profile | Scholar | Activity | Recruiting | Topics |",
            "|---|------|---------|---------|---------|----------|------------|--------|",
        ]
        for r in profs:
            sigs    = r.get("signals") or {}
            topics  = ", ".join((r.get("openalex_topics") or [])[:2])
            overall = r.get("overall_score") or "?"
            fpm     = r.get("faculty_profile_match") or "?"
            rsm     = r.get("recent_scholar_match") or "—"
            lines.append(
                f"| {r.get('rank','?')} | {r.get('name','?')} | {overall} | {fpm} | {rsm}"
                f" | {sigs.get('activity','?')} | {sigs.get('recruiting','?')} | {topics} |"
            )
        lines.append("")

        lines.append("## Per-Professor Detail")
        for r in profs:
            dims = r.get("dimensions") or {}
            sigs = r.get("signals") or {}
            cold = r.get("cold_email") or {}
            lines += [
                "",
                f"### #{r.get('rank','?')}  {r.get('name','?')}",
                f"- **Overall**: {r.get('overall_score')}",
                f"- **Profile match**: {r.get('faculty_profile_match')}",
                f"- **Scholar match**: {r.get('recent_scholar_match')}",
                f"- Research={dims.get('research')}  Method={dims.get('method')}  "
                f"Application={dims.get('application')}  Style={dims.get('style')}",
                f"- Activity={sigs.get('activity')}  Recruiting={sigs.get('recruiting')}  "
                f"Funding={sigs.get('funding')}",
                f"- Profile: {r.get('profile_url') or '—'}",
                _fmt_scholar_line(r),
                f"- OpenAlex: {r.get('openalex_url') or '—'}",
                f"- ORCID: {r.get('orcid') or '—'}",
                f"- Topics: {', '.join((r.get('openalex_topics') or [])[:5])}",
                f"- **Match reason**: _{r.get('match_reason','')[:200]}_",
            ]
            if isinstance(cold, dict) and cold:
                if cold.get("entry_point"):
                    lines.append(f"- **Email entry point**: _{cold['entry_point']}_")
                if cold.get("highlight_experience"):
                    lines.append(f"- **Highlight**: _{cold['highlight_experience']}_")
                if cold.get("convincing_point"):
                    lines.append(f"- **Key point**: _{cold['convincing_point']}_")
    else:
        lines.append("_No results._")
    lines.append("")

    # Errors + warnings
    all_issues = errs + warn_list
    if all_issues:
        lines += ["## Errors / Warnings", ""]
        for e in all_issues[:20]:
            lines.append(f"- {e[:250]}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advisor Matching Pipeline — reusable debug runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Department directory — full run
          python scripts/debug_pipeline.py \\
            --url "https://ise.usc.edu/faculty/" \\
            --cv sample_papers/sample_cv.txt

          # Single professor profile
          python scripts/debug_pipeline.py \\
            --url "https://www.ce.washington.edu/people/faculty/goodchilda" \\
            --cv sample_papers/sample_cv.txt

          # Fast debug: 5 candidates, no Scholar scraping
          python scripts/debug_pipeline.py \\
            --url "https://ise.usc.edu/faculty/" \\
            --cv sample_papers/sample_cv.txt --fast

          # Custom limit + skip scholar
          python scripts/debug_pipeline.py \\
            --url "https://ise.usc.edu/faculty/" \\
            --cv sample_papers/sample_cv.txt --limit 8 --skip-scholar
        """),
    )
    p.add_argument("--url",          required=True, help="Faculty directory or single profile URL")
    p.add_argument("--cv",           default=None,  help="CV file path (.txt or .pdf)")
    p.add_argument("--cv-text-file", default=None,  dest="cv_text_file",
                   help="Plain-text CV (alias for --cv)")
    p.add_argument("--verbose", "-v", action="store_true", default=True,
                   help="Print per-event progress (default: on)")
    p.add_argument("--quiet",  "-q", action="store_true",
                   help="Suppress per-event lines (overrides --verbose)")
    p.add_argument("--limit",        type=int, default=None,
                   help="Cap candidates after Stage B (e.g. --limit 5 for quick tests)")
    p.add_argument("--skip-scholar", action="store_true",
                   help="Skip all Scholar scraping (Stage D) — uses OpenAlex pubs only")
    p.add_argument("--fast",         action="store_true",
                   help="Shorthand for --limit 5 --skip-scholar")
    p.add_argument("--output-dir",   default=None,
                   help="Override output directory (default: outputs/debug_runs/)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    verbose = args.verbose and not args.quiet
    limit   = args.limit
    skip_s  = args.skip_scholar

    if args.fast:
        limit  = limit or 5
        skip_s = True

    try:
        cv_text = _load_cv(args.cv, args.cv_text_file)
    except Exception as exc:
        print(err(f"CV load error: {exc}"))
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None

    debug = run_debug(
        url=args.url,
        cv_text=cv_text,
        verbose=verbose,
        limit=limit,
        skip_scholar=skip_s,
        output_dir=out_dir,
    )

    sys.exit(0 if debug.get("matching", {}).get("success") else 1)


if __name__ == "__main__":
    main()
