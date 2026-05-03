# How to Get an API Key

This app supports three LLM providers. Pick any one — **Anthropic is recommended** (best results, free trial credits available).

---

## Option 1: Anthropic (Recommended)

**Model used:** `claude-haiku-4-5` — fast and affordable

1. Go to **https://platform.claude.com/settings/keys**
2. Sign up or log in
3. Click **"Create Key"**, give it a name, copy the key
5. New accounts receive **$5 free credits** (no credit card required initially)

**Cost:** ~$0.001 per professor analyzed (very cheap)

---

## Option 2: OpenAI

**Model used:** `gpt-4o-mini` — fast and cheap

1. Go to **https://platform.openai.com**
2. Sign up or log in
3. Click your profile icon → **"API keys"**
4. Click **"Create new secret key"**, copy it immediately (shown only once)
5. Add a payment method under **Billing** to activate the key

**Cost:** ~$0.001 per professor analyzed

---

## Option 3: Google Gemini

**Model used:** `gemini-1.5-flash` — has a free tier

1. Go to **https://aistudio.google.com**
2. Sign in with your Google account
3. Click **"Get API key"** → **"Create API key"**
4. Copy the key

**Free tier:** 15 requests/minute, 1500 requests/day — enough for normal use

---

## Where to enter the key

Paste your key into the **API Key** field on the web app after selecting your provider. The key is stored only in your browser (localStorage) and never sent to any server other than the provider you chose.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `401 Unauthorized` | Key is wrong or expired — regenerate it |
| `429 Too Many Requests` | Rate limit hit — wait a minute and retry |
| `insufficient_quota` | Add credits to your account |
| Gemini `RESOURCE_EXHAUSTED` | Free tier limit reached — wait until quota resets |
