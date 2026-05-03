# PhD Advisor Matcher

Paste a faculty directory URL → get a ranked list of the best-fit PhD advisors for your CV.

## Quickstart

**Requirements:** Python 3.9+, an API key from [Anthropic](https://console.anthropic.com), [OpenAI](https://platform.openai.com), or [Google Gemini](https://aistudio.google.com) — see **[How to get an API key](get_api_key/README.md)**

```bash
# 1. Clone
git clone https://github.com/lilieason/phd-advisor-matcher.git
cd phd-advisor-matcher

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the web app
python web_app.py

# 5. Open http://localhost:8001
#    Select your LLM provider, enter your API key, upload CV, paste URL → done
```

## What it does

1. **Extracts** all faculty from a department directory page
2. **Fetches** each professor's profile + Google Scholar publications
3. **Scores** fit with your CV across research direction, methods, and application domain
4. **Returns** a ranked Top 10 table with match explanations and email entry points

## Supported URL types

- **Faculty directory** (recommended): `https://ise.ufl.edu/people/faculty/`  → full batch ranking
- **Individual profile**: paste one or more direct profile URLs → per-professor analysis

## Requirements

See `requirements.txt`. Key dependencies: `fastapi`, `uvicorn`, `anthropic`, `beautifulsoup4`, `curl-cffi`.
