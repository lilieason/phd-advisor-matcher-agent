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

1. **Extraction agent** — fetches the URL, parses faculty names and profile links, visits each profile page to collect research interests, publications, and bio
2. **Matching agent** — scores each faculty member against your CV across research direction, methods, and application domain; generates a ranked list with match reasons and outreach entry points

## File structure

```
phd-advisor-matcher-agent/
├── web_app.py                        # FastAPI server, handles CV upload and SSE streaming
├── requirements.txt
├── static/
│   └── index.html                    # Single-page frontend (upload, progress, results table)
├── mcp_servers/
│   ├── extraction_agent.py           # Extraction agent: fetches & parses faculty pages
│   ├── matching_agent.py             # Matching agent: scores fit, generates outreach advice
│   ├── advisor_server.py             # Shared tools: CV reader, HTTP fetcher, OpenAlex/Scholar lookup
│   └── llm_client.py                 # Unified LLM client (Anthropic / OpenAI / Gemini)
└── get_api_key/
    └── README.md                     # Guide to getting an API key
```

## Requirements

See `requirements.txt`. Key dependencies: `fastapi`, `uvicorn`, `anthropic`, `beautifulsoup4`, `curl-cffi`.
