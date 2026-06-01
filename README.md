---
title: YT Summarizer Pro
emoji: 🎬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# 🎬 VidDigest AI — YouTube Summarizer with RAG

A portfolio-ready AI web app that summarizes any YouTube video and lets you
chat with it using RAG (Retrieval-Augmented Generation).

**100% free stack — no credit card needed.**

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎯 4 Summary Modes | Quick Overview, Detailed Analysis, Bullet Points, Key Quotes |
| 🧠 RAG Q&A | Ask anything about the video — answers from actual transcript chunks |
| 🌙 Dark / Light Mode | Toggle with one click |
| ⚡ Free LLM | Google Gemini 1.5 Flash (15 req/min free) |
| 🔍 Local Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (no API cost) |
| 🗂️ FAISS Vector Search | Fast semantic retrieval |

---

## 🗂️ Project Structure

```
viddigest/
├── backend.py          ← FastAPI server (all logic lives here)
├── requirements.txt    ← Python dependencies
├── .env                ← Your API key (never commit this!)
├── .gitignore          ← Keeps .env off GitHub
├── README.md
└── static/
    ├── index.html      ← App UI
    ├── style.css       ← Styles (dark + light mode)
    └── app.js          ← Frontend logic
```

---

## 🚀 Local Setup (VS Code)

### Step 1 — Clone / open the folder in VS Code

Open the `viddigest` folder in VS Code.  
Open the integrated terminal: **Terminal → New Terminal**

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). This is cached after that.

### Step 4 — Get your free Gemini API key

1. Go to <https://aistudio.google.com/app/apikey>
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key

### Step 5 — Add the key to `.env`

Open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 6 — Run the server

```bash
uvicorn backend:app --reload
```

### Step 7 — Open the app

Go to <http://127.0.0.1:8000> in your browser.

---

## 🧠 How RAG Works Here

```
YouTube URL
    ↓
Transcript fetched (free, no API key)
    ↓
Split into 400-word chunks with 50-word overlap
    ↓
Each chunk → 384-dim vector (all-MiniLM-L6-v2, runs locally)
    ↓
Vectors stored in FAISS index (in memory)
    ↓
On question → embed question → find top 4 closest chunks
    ↓
Chunks + question → Gemini 1.5 Flash → precise answer
```

This means the chatbot answers from **real transcript content**, not guesses.

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|---|---|
| `GEMINI_API_KEY missing` | Check your `.env` file has the key set correctly |
| `Transcript unavailable` | The video has disabled captions. Try a different video |
| `Invalid YouTube URL` | Make sure URL contains a video ID, e.g. `?v=XXXXXXXXXXX` |
| `Module not found` | Run `pip install -r requirements.txt` inside your venv |
| Port already in use | Run `uvicorn backend:app --reload --port 8001` |

---

## 🔒 Security Note

**Never push your `.env` file to GitHub.**  
The `.gitignore` already excludes it. Double-check before every `git push`.

---

## 💡 Free Tier Limits (Gemini)

- 15 requests per minute
- 1,500 requests per day
- No credit card required

More than enough for a portfolio project or demo.
