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

**Extraction agent** — fetches the URL, parses faculty names and profile links, visits each profile page to collect research interests, publications, and bio

- Runs a **Planner → Executor → Validator** loop: the Planner reads page signals and Kahuna memory to choose an extraction strategy; the Executor runs it; the Validator checks result quality and triggers a retry with a different strategy if needed
- **Kahuna memory**: records the outcome of every run per domain — on future visits to the same site the Planner promotes the historically best strategy to the front
- **Few-shot learning**: clean extractions are saved locally; on structurally similar pages the saved example is injected into the LLM prompt to improve accuracy

**Matching agent** — scores each faculty member against your CV across research direction, methods, and application domain; generates a ranked list with match reasons and outreach entry points

- **CV normalisation**: non-English CVs are translated to English keywords by an LLM before scoring
- **Early exclusion**: keyword overlap and heuristic filters remove off-topic faculty with no LLM calls
- **Data enrichment**: fetches each professor's profile page, OpenAlex publications, and Google Scholar records in parallel
- **Prescreening**: a single batch LLM call scores all enriched profiles and shortlists the top candidates
- **Deep scoring**: separate LLM calls score profile fit and Scholar publication fit; scores are weighted and combined into an overall match score
- **Outreach advice**: for each top result, generates a personalised paper hook and a specific fit statement
- **Live streaming**: every stage emits SSE events to the frontend so the user sees real-time progress throughout

## Requirements

See `requirements.txt`. Key dependencies: `fastapi`, `uvicorn`, `anthropic`, `beautifulsoup4`, `curl-cffi`.
