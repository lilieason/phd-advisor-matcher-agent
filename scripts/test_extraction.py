#!/usr/bin/env python3
"""
Extraction Agent Tester
=======================
Run the extraction agent on one or more faculty-list URLs and print a
per-URL summary: strategy, elapsed time, faculty count, validator score,
and the extracted name list.

Usage
-----
  # Interactive — paste / type URLs one per line, blank line to finish
  python scripts/test_extraction.py

  # Non-interactive — pass URLs directly
  python scripts/test_extraction.py URL1 URL2 ...

  # From a file (one URL per line)
  python scripts/test_extraction.py --file urls.txt

  # Keep Kahuna memory between runs (default: cleared before each URL)
  python scripts/test_extraction.py --keep-memory URL ...

Output
------
  Console table with colour + per-URL faculty list.
  Auto-opens the summary table in a temporary HTML file when run from a
  terminal that supports it (--no-open to suppress).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Suppress noisy library logs — only show extraction agent output
import logging
logging.basicConfig(level=logging.WARNING)

from mcp_servers.extraction_agent import run_extraction_agent  # noqa: E402

# ── Colour helpers ────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

def ok(s):   return _c("32;1", s)
def err(s):  return _c("31;1", s)
def warn(s): return _c("33;1", s)
def hdr(s):  return _c("36;1", s)
def dim(s):  return _c("2",    s)
def bold(s): return _c("1",    s)

BAR  = "─" * 76
BAR2 = "═" * 76


# ── Kahuna cache helpers ──────────────────────────────────────────────────────

def _kahuna_key(url: str) -> str:
    """Return the Kahuna file prefix for a given URL (domain slug)."""
    parsed = urlparse(url)
    host = parsed.netloc.replace(".", "-").replace(":", "-")
    return host


def _clear_kahuna(url: str) -> None:
    """Delete cached Kahuna extractions for this domain so each run is fresh."""
    key = _kahuna_key(url)
    pattern = os.path.expanduser(f"~/.kahuna/knowledge/extractions/{key}*")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass


# ── Single URL runner ─────────────────────────────────────────────────────────

def run_one(url: str, clear_memory: bool = True) -> dict:
    """Run extraction on one URL, return result dict."""
    if clear_memory:
        _clear_kahuna(url)

    t0 = time.perf_counter()
    try:
        result = run_extraction_agent(url)
        elapsed = time.perf_counter() - t0
        return {
            "url":      url,
            "strategy": result.strategy_used or "none",
            "count":    result.faculty_count,
            "score":    result.validator_score,
            "success":  result.success,
            "elapsed":  elapsed,
            "names":    [f["name"] for f in result.faculty_list],
            "error":    None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "url":      url,
            "strategy": "ERROR",
            "count":    0,
            "score":    0.0,
            "success":  False,
            "elapsed":  elapsed,
            "names":    [],
            "error":    str(exc),
        }


# ── Console output ────────────────────────────────────────────────────────────

def _score_colour(score: float) -> str:
    if score >= 0.90:
        return ok(f"{score:.3f}")
    if score >= 0.60:
        return warn(f"{score:.3f}")
    return err(f"{score:.3f}")


def print_result(r: dict, index: int, total: int) -> None:
    tag  = ok("✓") if r["success"] else err("✗")
    url_short = r["url"][:70] + ("…" if len(r["url"]) > 70 else "")

    print()
    print(hdr(f"[{index}/{total}] {url_short}"))
    print(BAR)

    # Summary row
    strategy_str = bold(r["strategy"])
    count_str    = bold(str(r["count"]))
    score_str    = _score_colour(r["score"])
    time_str     = dim(f"{r['elapsed']:.1f}s")
    status_str   = ok("success") if r["success"] else err("failed")

    print(f"  {tag}  Strategy : {strategy_str}")
    print(f"     Count    : {count_str}")
    print(f"     Score    : {score_str}")
    print(f"     Time     : {time_str}")
    print(f"     Status   : {status_str}")

    if r["error"]:
        print(f"     {err('Error')}    : {r['error']}")

    if r["names"]:
        print()
        print(dim(f"  Faculty ({r['count']}):"))
        # Print in two columns if count > 10
        names = r["names"]
        if len(names) <= 20:
            for n in names:
                print(f"    • {n}")
        else:
            col = (len(names) + 1) // 2
            left  = names[:col]
            right = names[col:]
            for i, ln in enumerate(left):
                rn = right[i] if i < len(right) else ""
                print(f"    • {ln:<38}  {'• ' + rn if rn else ''}")
    else:
        print()
        print(dim("  (no faculty extracted)"))


def print_summary_table(results: list[dict]) -> None:
    print()
    print(bold(BAR2))
    print(bold(f"  SUMMARY  ({len(results)} URL{'s' if len(results) != 1 else ''})"))
    print(bold(BAR2))

    # Header
    print(f"  {'#':<3} {'S':<1}  {'Strategy':<22} {'Count':>5}  {'Score':>6}  {'Time':>6}  URL")
    _sep = ("  " + dim("─"*3) + "  " + dim("─") + "  " + dim("─"*22)
            + "  " + dim("─"*5) + "  " + dim("─"*6) + "  " + dim("─"*6)
            + "  " + dim("─"*40))
    print(_sep)

    for i, r in enumerate(results, 1):
        tag      = ok("✓") if r["success"] else err("✗")
        strategy = r["strategy"][:22]
        count    = str(r["count"])
        score    = f"{r['score']:.3f}"
        elapsed  = f"{r['elapsed']:.1f}s"
        parsed   = urlparse(r["url"])
        url_disp = (parsed.netloc + parsed.path)[:50]

        score_s  = _score_colour(r["score"])
        print(f"  {i:<3} {tag}  {strategy:<22} {count:>5}  {score_s:>6}  {elapsed:>6}  {dim(url_disp)}")

    print(bold(BAR2))

    # Totals
    total_ok  = sum(1 for r in results if r["success"])
    total_fac = sum(r["count"] for r in results)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    total_t   = sum(r["elapsed"] for r in results)
    print(f"  Passed: {ok(str(total_ok))}/{len(results)}   "
          f"Total faculty: {bold(str(total_fac))}   "
          f"Avg score: {_score_colour(avg_score)}   "
          f"Total time: {dim(f'{total_t:.1f}s')}")
    print()


# ── HTML report (auto-open) ───────────────────────────────────────────────────

def _html_report(results: list[dict]) -> str:
    rows = ""
    for r in results:
        tag   = "✓" if r["success"] else "✗"
        color = "#2a9d2a" if r["success"] else "#cc3333"
        score = r["score"]
        score_color = ("#2a9d2a" if score >= 0.90
                       else "#cc8800" if score >= 0.60
                       else "#cc3333")
        parsed   = urlparse(r["url"])
        url_disp = parsed.netloc + parsed.path

        names_html = ""
        if r["names"]:
            items = "".join(f"<li>{n}</li>" for n in r["names"])
            names_html = f"<details><summary>{r['count']} faculty</summary><ol>{items}</ol></details>"
        else:
            names_html = "<span style='color:#999'>—</span>"

        rows += f"""
        <tr>
          <td style='color:{color};font-weight:bold;font-size:1.2em'>{tag}</td>
          <td title='{r["url"]}'>{url_disp}</td>
          <td><code>{r['strategy']}</code></td>
          <td style='text-align:right'>{r['count']}</td>
          <td style='text-align:right;color:{score_color};font-weight:bold'>{r['score']:.3f}</td>
          <td style='text-align:right'>{r['elapsed']:.1f}s</td>
          <td>{names_html}</td>
        </tr>"""

    total_ok  = sum(1 for r in results if r["success"])
    total_fac = sum(r["count"] for r in results)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extraction Agent Test Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 2rem auto; padding: 0 1rem; background: #fafafa; }}
  h1   {{ color: #222; }}
  .summary {{ display: flex; gap: 2rem; margin: 1rem 0 2rem;
              background: #fff; padding: 1rem 1.5rem; border-radius: 8px;
              box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .metric  {{ text-align: center; }}
  .metric .val {{ font-size: 2rem; font-weight: 700; color: #333; }}
  .metric .lbl {{ font-size: .8rem; color: #888; text-transform: uppercase; }}
  table  {{ width: 100%; border-collapse: collapse; background: #fff;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  th     {{ background: #333; color: #fff; padding: .6rem 1rem; text-align: left; }}
  td     {{ padding: .5rem 1rem; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:hover td {{ background: #f5f5f5; }}
  details summary {{ cursor: pointer; color: #0070c0; }}
  ol {{ margin: .4rem 0 0 1rem; padding: 0; font-size: .9em; }}
  code {{ background: #eee; padding: 2px 5px; border-radius: 3px; font-size: .9em; }}
</style>
</head>
<body>
<h1>Extraction Agent — Test Results</h1>
<div class="summary">
  <div class="metric"><div class="val">{len(results)}</div><div class="lbl">URLs tested</div></div>
  <div class="metric"><div class="val" style="color:#2a9d2a">{total_ok}</div><div class="lbl">Passed</div></div>
  <div class="metric"><div class="val">{total_fac}</div><div class="lbl">Total faculty</div></div>
  <div class="metric"><div class="val">{avg_score:.3f}</div><div class="lbl">Avg score</div></div>
</div>
<table>
  <thead>
    <tr>
      <th></th><th>URL</th><th>Strategy</th>
      <th>Count</th><th>Score</th><th>Time</th><th>Faculty list</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""


def open_html_report(results: list[dict]) -> None:
    html = _html_report(results)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="extraction_test_",
        delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name
    print(dim(f"  HTML report: file://{path}"))
    webbrowser.open(f"file://{path}")


# ── URL collection ────────────────────────────────────────────────────────────

def collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []

    # From --file
    if args.file:
        p = Path(args.file)
        if not p.exists():
            sys.exit(err(f"File not found: {p}"))
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    # From positional args
    urls.extend(args.urls)

    # Interactive fallback
    if not urls:
        print(hdr("Extraction Agent Tester"))
        print(dim("Enter URLs one per line. Blank line or Ctrl-D to start."))
        print()
        try:
            while True:
                try:
                    line = input(dim("  URL: ")).strip()
                except EOFError:
                    break
                if not line:
                    break
                urls.append(line)
        except KeyboardInterrupt:
            print()
            sys.exit(0)

    # Basic sanity
    valid = []
    for u in urls:
        if not u.startswith(("http://", "https://")):
            print(warn(f"  Skipping (not a URL): {u}"))
            continue
        valid.append(u)

    if not valid:
        sys.exit(err("No valid URLs provided."))

    return valid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the extraction agent on one or more faculty-list URLs.",
    )
    parser.add_argument("urls", nargs="*", metavar="URL",
                        help="Faculty list URL(s) to test")
    parser.add_argument("--file", "-f", metavar="FILE",
                        help="Text file with one URL per line")
    parser.add_argument("--keep-memory", action="store_true",
                        help="Do not clear Kahuna cache before each URL (use existing memory)")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the HTML report in a browser")
    args = parser.parse_args()

    urls = collect_urls(args)

    print()
    print(bold(BAR2))
    print(bold(f"  EXTRACTION AGENT TEST  —  {len(urls)} URL{'s' if len(urls) != 1 else ''}"))
    print(bold(BAR2))

    results: list[dict] = []
    for i, url in enumerate(urls, 1):
        print()
        print(dim(f"  Running [{i}/{len(urls)}]: {url}"))
        r = run_one(url, clear_memory=not args.keep_memory)
        results.append(r)
        print_result(r, i, len(urls))

    print_summary_table(results)

    if not args.no_open and _TTY:
        open_html_report(results)


if __name__ == "__main__":
    main()
