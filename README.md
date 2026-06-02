# 🎬 VidDigest AI — YouTube Summarizer with RAG

A portfolio-ready AI web app that summarizes any YouTube video and lets you
chat with it using RAG (Retrieval-Augmented Generation).

**100% free stack — no credit card needed.**

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🌐 **Any Language** | Telugu, Hindi, English, or any other language — auto-detected |
| ⚡ **Ultra-Fast LLM** | Groq LLaMA-3.3-70b delivers summaries in ~2 seconds |
| 🧠 **RAG Q&A** | FAISS vector search + Groq gives answers grounded in the actual transcript |
| 🎯 **4 Summary Modes** | Quick Overview, Detailed Analysis, Bullet Points, Key Insights |
| 🎙️ **3-Layer Transcription** | YouTube API → yt-dlp captions → AssemblyAI audio (fallback chain) |
| 📊 **History & Analytics** | Past summaries with thumbnails, usage statistics dashboard |
| 🌙 **Dark / Light Mode** | One-click theme toggle, preference saved locally |
| 🐳 **Docker Ready** | Single command deployment via Dockerfile |
| 💰 **100% Free Stack** | No paid APIs required — all free tiers |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    YT Summarizer Pro                     │
│                                                         │
│  Frontend (Vanilla JS + CSS)                           │
│  ├── Dashboard  → URL input, summary output             │
│  ├── History    → past summaries with thumbnails        │
│  ├── Analytics  → usage stats                           │
│  └── RAG Chat   → how RAG works explained               │
│                                                         │
│  Backend (FastAPI)                                      │
│  │                                                      │
│  ├── Transcript Pipeline (3 Layers)                    │
│  │   ├── Layer 1: youtube-transcript-api  (~1 sec)     │
│  │   ├── Layer 2: yt-dlp auto-captions   (~5 sec)      │
│  │   └── Layer 3: AssemblyAI audio ASR   (~1-2 min)    │
│  │                                                      │
│  ├── RAG Pipeline                                       │
│  │   ├── Chunk transcript (400 words, 60 overlap)      │
│  │   ├── Embed with sentence-transformers              │
│  │   ├── Store in FAISS IndexFlatIP                    │
│  │   └── Retrieve top-4 chunks on question             │
│  │                                                      │
│  └── LLM (Groq LLaMA-3.3-70b)                         │
│      ├── /summarize → full summary                      │
│      └── /ask       → RAG-grounded answer               │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | Async, fast, production-ready |
| **LLM** | Groq LLaMA-3.3-70b | Fastest free LLM (~2 sec response) |
| **Transcript** | youtube-transcript-api, yt-dlp, AssemblyAI | 3-layer fallback — works on any video |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Local, free, 384-dim vectors |
| **Vector Search** | FAISS `IndexFlatIP` | Cosine similarity, instant retrieval |
| **Frontend** | Vanilla JS + CSS (DM Sans) | Zero dependencies, fast load |
| **Deployment** | Docker + Hugging Face Spaces | Free hosting |

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your system (`winget install ffmpeg` on Windows)
- Chrome browser (logged into YouTube — needed for yt-dlp cookie auth)

### 1. Clone the repository

```bash
git clone https://github.com/Satya23BDS0326/YT-Summarizer-Pro.git
cd yt-summarizer-pro
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). Cached after that.

### 4. Get your free API keys

| API | Where to get | Free tier |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | 14,400 req/day, ~2 sec response |
| **AssemblyAI** | [assemblyai.com](https://www.assemblyai.com) | 100 hours/month free |

### 5. Set up environment variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and add your keys:

```env
GROQ_API_KEY=gsk_your_groq_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_key_here
```

### 6. Run the server

```bash
uvicorn backend:app --reload
```

### 7. Open in browser

```
http://127.0.0.1:8000
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t yt-summarizer-pro .

# Run
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e ASSEMBLYAI_API_KEY=your_key \
  yt-summarizer-pro
```

---

## 🤗 Deploy to Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Docker** SDK
2. Push this repository to the Space
3. Add your API keys under **Settings → Repository Secrets**:
   - `GROQ_API_KEY`
   - `ASSEMBLYAI_API_KEY`
4. The Dockerfile handles the rest — your app goes live automatically

---

## 🧠 How RAG Works (Technical)

```
YouTube URL
    │
    ▼
┌─────────────────────────────────┐
│   Transcript Extraction         │
│                                 │
│   1. youtube-transcript-api     │  ← tries first (instant, free)
│   2. yt-dlp auto-captions       │  ← fallback (5 sec, any language)
│   3. AssemblyAI audio ASR       │  ← last resort (any video, ~1 min)
└────────────────┬────────────────┘
                 │ raw transcript text
                 ▼
┌─────────────────────────────────┐
│   RAG Index Building            │
│                                 │
│   • Split into 400-word chunks  │
│     with 60-word overlap        │
│   • Embed each chunk →          │
│     384-dim vector              │
│     (all-MiniLM-L6-v2, local)  │
│   • Store in FAISS IndexFlatIP  │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  /summarize           /ask (RAG Q&A)
        │                 │
   Full transcript    Question → embed
   → Groq LLaMA       → FAISS top-4
   → Summary          → Groq LLaMA
                       → Grounded answer
```

**Why RAG matters:** Without RAG, the LLM would answer from general knowledge (hallucination). With RAG, every answer is retrieved from the actual transcript — factual, verifiable, zero hallucination.

---

## 📁 Project Structure

```
yt-summarizer-pro/
│
├── backend.py          ← FastAPI server — all business logic
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Docker + Hugging Face deployment
├── .env.example        ← Environment variable template
├── .env                ← Your keys (never commit — in .gitignore)
├── .gitignore          ← Excludes .env, venv, __pycache__, etc.
├── cookies.txt         ← Optional: YouTube cookies for yt-dlp auth
├── README.md
│
└── static/
    ├── index.html      ← Single-page app (4 pages: Dashboard, History, Analytics, RAG Chat)
    ├── style.css       ← Dark/light mode, responsive design
    └── app.js          ← All frontend logic, localStorage history
```

---

## 🔌 API Reference

### `POST /summarize`

Summarizes a YouTube video.

**Request:**
```json
{
  "url": "https://youtube.com/watch?v=...",
  "mode": "brief"
}
```

**Modes:** `brief` | `detailed` | `bullet_points` | `key_quotes`

**Response:**
```json
{
  "success": true,
  "video_id": "dQw4w9WgXcQ",
  "title": "Video Title",
  "channel": "Channel Name",
  "thumbnail": "https://img.youtube.com/vi/.../mqdefault.jpg",
  "summary": "## Summary\n...",
  "word_count": 4823
}
```

---

### `POST /ask`

Answers a question using RAG on the indexed transcript.

**Request:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "question": "What is the main point of this video?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "Based on the transcript, the main point is..."
}
```

---

## ⚠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GROQ_API_KEY missing` | Key not in `.env` | Add key to `.env` file |
| `ASSEMBLYAI_API_KEY missing` | Key not in `.env` | Add key to `.env` file |
| `Could not get transcript` | Video private / region-blocked | Try a different video |
| `Module not found` | venv not activated | Run `venv\Scripts\activate` first |
| `Port already in use` | Another server running | Use `--port 8001` |
| `SSL certificate error` | ISP / antivirus intercept | Normal — app handles it automatically |
| `yt-dlp 403 error` | YouTube bot detection | Log into YouTube in Chrome and retry |

---

## 🔒 Security

- **Never commit `.env`** — it's in `.gitignore`
- **`cookies.txt`** is also in `.gitignore` — contains session tokens
- API keys are loaded via `python-dotenv` at runtime only
- No user data is stored server-side — history lives in browser `localStorage`

---

## 💡 Free Tier Limits (Gemini)

- 15 requests per minute
- 1,500 requests per day
- No credit card required

More than enough for a portfolio project or demo.