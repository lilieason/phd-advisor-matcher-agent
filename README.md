# PhD Advisor Matcher

Upload your CV and a URL → get a ranked list of the best-fit PhD advisors with match scores, research interest alignment, and outreach advice.

## Two modes

**Mode 1 — Faculty directory** (batch ranking)
Paste a department directory URL. The tool extracts all faculty, then for each professor fetches their profile page, OpenAlex publication data, and Google Scholar records to build a comprehensive match score against your CV.
```
https://ise.usc.edu/directory/faculty/
```

**Mode 2 — Individual profile** (per-professor analysis)
Paste one or more direct faculty profile URLs (one per line). Each is analyzed against your CV individually — useful when you already have specific professors in mind.
```
https://ise.usc.edu/directory/faculty/profile/?lname=Dessouky&fname=Maged
```

## Quickstart

**Requirements:** Python 3.9+, an API key from [Anthropic](https://platform.claude.com/settings/keys), [OpenAI](https://platform.openai.com), or [Google Gemini](https://aistudio.google.com) — see **[How to get an API key](get_api_key/README.md)**

```bash
# 1. Clone
git clone https://github.com/lilieason/phd-advisor-matcher-agent.git
cd phd-advisor-matcher-agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the web app
python web_app.py

# 5. Open http://localhost:8001
#    Select provider → enter API key → upload CV → paste URL → Start Analysis
```

## How it works

Upload your CV and paste a URL. Two agents run in sequence — the extraction agent gathers faculty data from the web, then the matching agent scores each professor against your CV and streams results to the frontend in real time.

### Agent design

**Extraction agent — Planner → Executor → Validator loop**
- **Planner**: inspects page signals (HTML structure, link patterns, section layout) and reads Kahuna memory for this domain to select an ordered list of extraction strategies
- **Executor**: runs each strategy in priority order, stopping when the Validator approves the result
- **Validator**: scores extraction quality (faculty count, name validity, hallucination check) and decides pass/retry
- **Kahuna memory**: after each run, records the outcome (strategy used, faculty count, quality score) locally per domain — on the next visit to the same site, the Planner promotes the historically best strategy to the front
- **Few-shot learning**: successful extractions are saved as structured examples; on future runs with structurally similar pages, the example is injected into the LLM prompt to guide extraction

**Matching agent — multi-stage pipeline with live frontend streaming**

The matching agent runs a staged pipeline and streams every step to the frontend in real time via SSE (Server-Sent Events), so the user sees live progress rather than waiting for a final result.

| Stage | What happens | Frontend event |
|-------|-------------|----------------|
| A — CV normalisation | If the CV is non-English, an LLM extracts English keywords and a summary for consistent scoring | `cv_normalize` |
| B — Early exclusion | Keyword overlap + heuristic filters remove obviously off-topic faculty without any LLM calls | — |
| C — Data enrichment | Fetches each professor's profile page (research interests, bio, links), OpenAlex publications, and Google Scholar records in parallel | `profiles_fetched` |
| D — Prescreening (LLM) | Batch LLM call scores all enriched profiles for keyword fit; shortlists the top candidates for deep scoring | `shortlisted` |
| E — Full scoring (LLM) | For each shortlisted professor: separate LLM calls score profile fit and Scholar publication fit; scores are weighted and combined into an overall match score | `detail`, `batch_progress` |
| F — Outreach advice (LLM) | For the top results, generates a personalised entry point (specific paper hook) and a convincing fit statement | — |
| Deliver | Final ranked list sent to frontend; results rendered as an interactive table with score badges, interest pills, and outreach advice | `results` |

## Requirements

See `requirements.txt`. Key dependencies: `fastapi`, `uvicorn`, `anthropic`, `beautifulsoup4`, `curl-cffi`.
