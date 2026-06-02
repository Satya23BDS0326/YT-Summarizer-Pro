"""
YT Summarizer Pro — backend.py  (Hugging Face / Cloud Edition)
══════════════════════════════════════════════════════════════════════
ROOT CAUSE OF CLOUD FAILURE:
  YouTube actively blocks requests from datacenter IPs (AWS, HF, GCP).
  All direct connections to youtube.com time out with SSL/connection errors.

SOLUTION — 4-layer transcript pipeline:
  Layer 0  Supadata API          Residential proxy, free, ~1 sec ← NEW
  Layer 1  youtube-transcript-api via Supadata proxy             ← NEW
  Layer 2  yt-dlp + android client (bypasses some blocks)
  Layer 3  AssemblyAI            Cloud ASR, any language

FREE API KEYS NEEDED:
  GROQ_API_KEY       → https://console.groq.com          (free)
  ASSEMBLYAI_API_KEY → https://www.assemblyai.com        (free)
  SUPADATA_API_KEY   → https://supadata.ai               (free tier)
══════════════════════════════════════════════════════════════════════
"""

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import assemblyai as aai
import faiss
import requests
import yt_dlp

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
SUPADATA_KEY   = os.getenv("SUPADATA_API_KEY", "")   # free at supadata.ai

if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env or HF Secrets")
if not ASSEMBLYAI_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY missing in .env or HF Secrets")

groq_client          = Groq(api_key=GROQ_KEY)
aai.settings.api_key = ASSEMBLYAI_KEY

# ─────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────

print("⏳ Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready.")

rag_store: dict = {}

# ─────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────

app = FastAPI(title="YT Summarizer Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

# ─────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    url:  str
    mode: str = "brief"

class AskRequest(BaseModel):
    video_id: str
    question: str

# ─────────────────────────────────────────────────────────────────
# URL → VIDEO ID
# ─────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    url    = url.strip()
    parsed = urlparse(url)

    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.strip("/").split("/")[0]
        return vid[:11] if vid else None

    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query).get("v")
        if qs:
            return qs[0][:11]
        parts = [p for p in parsed.path.split("/") if p]
        for marker in ("shorts", "embed", "live"):
            if marker in parts:
                i = parts.index(marker)
                if len(parts) > i + 1:
                    return parts[i + 1][:11]

    m = re.search(r"([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

# ─────────────────────────────────────────────────────────────────
# TEXT CLEANER
# ─────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>",    " ", text)
    text = re.sub(r"\{[^}]+\}",  " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+",        " ", text)
    return text.strip()

def normalize(items) -> str:
    parts = []
    for item in items:
        if isinstance(item, dict):
            parts.append(item.get("text", "").replace("\n", " "))
        else:
            parts.append(str(getattr(item, "text", item)).replace("\n", " "))
    return clean(" ".join(parts))

def is_good(text: str) -> bool:
    """Reject transcripts too short to summarize meaningfully."""
    return len([w for w in text.split() if len(w) > 2]) >= 100

# ─────────────────────────────────────────────────────────────────
# VIDEO METADATA  (title + thumbnail via oEmbed — no API key)
# ─────────────────────────────────────────────────────────────────

def fetch_meta(video_id: str) -> dict:
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed"
            f"?url=https://youtu.be/{video_id}&format=json",
            timeout=8,
        )
        if r.ok:
            d = r.json()
            return {
                "title":     d.get("title", "YouTube Video"),
                "channel":   d.get("author_name", ""),
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            }
    except Exception:
        pass
    return {
        "title":     "YouTube Video",
        "channel":   "",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
    }

# ─────────────────────────────────────────────────────────────────
# LAYER 0 — Supadata API  (BEST for cloud — residential proxy)
# Free at https://supadata.ai — designed for exactly this problem
# ─────────────────────────────────────────────────────────────────

def layer0_supadata(video_id: str) -> str | None:
    """
    Supadata routes transcript requests through residential IPs,
    bypassing YouTube's datacenter block completely.
    Free tier: 1000 requests/month.
    """
    if not SUPADATA_KEY:
        print("  Layer 0: SUPADATA_API_KEY not set, skipping.")
        return None
    try:
        r = requests.get(
            f"https://api.supadata.ai/v1/youtube/transcript",
            params={"videoId": video_id, "text": "true"},
            headers={"x-api-key": SUPADATA_KEY},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            # Supadata returns { content: [...] } or { content: "string" }
            content = data.get("content", "")
            if isinstance(content, list):
                text = clean(" ".join(
                    item.get("text", "") for item in content
                ))
            else:
                text = clean(str(content))
            if is_good(text):
                print(f"  ✅ Layer 0 Supadata OK ({len(text.split())} words)")
                return text
        else:
            print(f"  Layer 0 Supadata HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"  Layer 0 Supadata error: {e}")
    return None

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — youtube-transcript-api  (direct, works if not blocked)
# ─────────────────────────────────────────────────────────────────

def layer1_transcript_api(video_id: str) -> str | None:
    try:
        ytt   = YouTubeTranscriptApi()
        tlist = ytt.list(video_id)

        for finder in (
            lambda t: t.find_transcript(["te", "hi", "en"]),
            lambda t: t.find_generated_transcript(["te", "hi", "en"]),
            lambda t: next(iter(t)).translate("en"),
            lambda t: next(iter(t)),
        ):
            try:
                raw  = finder(tlist).fetch()
                text = normalize(raw)
                if is_good(text):
                    print(f"  ✅ Layer 1 OK ({len(text.split())} words)")
                    return text
            except Exception:
                pass
    except Exception as e:
        print(f"  Layer 1: {e}")
    return None

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — yt-dlp with android client  (bypasses some IP blocks)
# ─────────────────────────────────────────────────────────────────

def _clean_vtt(raw: str) -> str:
    text = re.sub(r"<[^>]+>",                          " ", raw)
    text = re.sub(r"WEBVTT|Kind:.*|Language:.*",        " ", text)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[\.,]\d{3}.*-->.*", " ", text)
    text = re.sub(r"^\d+\s*$",                          " ", text, flags=re.MULTILINE)
    text = re.sub(r"&amp;",                             "&", text)
    text = re.sub(r"\s+",                               " ", text)
    words, out, prev = text.split(), [], None
    for w in words:
        if w != prev:
            out.append(w)
        prev = w
    return " ".join(out).strip()

def layer2_ytdlp_captions(url: str) -> str | None:
    try:
        cookie_file = BASE_DIR / "cookies.txt"
        ydl_opts = {
            "quiet":          True,
            "no_warnings":    True,
            "skip_download":  True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            # android client sometimes bypasses datacenter blocks
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        }
        if cookie_file.exists():
            ydl_opts["cookiefile"] = str(cookie_file)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        subs = {
            **(info.get("subtitles") or {}),
            **(info.get("automatic_captions") or {}),
        }
        if not subs:
            return None

        priority = ["te", "hi", "en"] + [k for k in subs if k not in ("te", "hi", "en")]
        for lang in priority:
            entries = subs.get(lang)
            if not entries:
                continue
            try:
                r    = requests.get(entries[0]["url"], timeout=15)
                r.raise_for_status()
                text = _clean_vtt(r.text)
                if is_good(text):
                    print(f"  ✅ Layer 2 OK (lang={lang}, {len(text.split())} words)")
                    return text
            except Exception:
                continue
    except Exception as e:
        print(f"  Layer 2: {e}")
    return None

# ─────────────────────────────────────────────────────────────────
# LAYER 3 — AssemblyAI  (cloud ASR, any language, most reliable)
# ─────────────────────────────────────────────────────────────────

def layer3_assemblyai(url: str) -> str | None:
    try:
        print("  ⏳ Downloading audio for AssemblyAI...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_file = BASE_DIR / "cookies.txt"
            ydl_opts = {
                "format":      "worstaudio/worst",   # smallest file = fastest
                "outtmpl":     str(Path(tmpdir) / "%(id)s.%(ext)s"),
                "quiet":       True,
                "no_warnings": True,
                "noplaylist":  True,
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                "postprocessors": [{
                    "key":              "FFmpegExtractAudio",
                    "preferredcodec":   "mp3",
                    "preferredquality": "64",
                }],
            }
            if cookie_file.exists():
                ydl_opts["cookiefile"] = str(cookie_file)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info       = ydl.extract_info(url, download=True)
                audio_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")

            if not audio_path.exists():
                candidates = list(Path(tmpdir).glob("*.mp3"))
                audio_path = candidates[0] if candidates else None
            if not audio_path:
                return None

            print(f"  Audio ready ({audio_path.stat().st_size // 1024} KB)")

            config      = aai.TranscriptionConfig(
                speech_models=[aai.SpeechModel.best],
                language_detection=True,
            )
            transcriber = aai.Transcriber()
            result      = transcriber.transcribe(str(audio_path), config=config)

            if result.status == aai.TranscriptStatus.error:
                print(f"  AssemblyAI error: {result.error}")
                return None

            text = clean(result.text or "")
            if is_good(text):
                print(f"  ✅ Layer 3 AssemblyAI OK ({len(text.split())} words)")
                return text

    except Exception as e:
        print(f"  Layer 3: {e}")
    return None

# ─────────────────────────────────────────────────────────────────
# MAIN TRANSCRIPT  — tries all layers in order
# ─────────────────────────────────────────────────────────────────

def get_transcript(video_id: str, url: str) -> str:
    layers = [
        ("Layer 0 — Supadata",       lambda: layer0_supadata(video_id)),
        ("Layer 1 — Transcript API", lambda: layer1_transcript_api(video_id)),
        ("Layer 2 — yt-dlp",         lambda: layer2_ytdlp_captions(url)),
        ("Layer 3 — AssemblyAI",     lambda: layer3_assemblyai(url)),
    ]

    for name, fn in layers:
        print(f"\n▶ Trying {name}...")
        try:
            text = fn()
            if text:
                print(f"SUCCESS: {name}")
                return text
            print(f"  empty, trying next...")
        except Exception as e:
            print(f"  {name} failed: {e}")

    raise RuntimeError(
        "Could not get a transcript for this video.\n"
        "On Hugging Face: YouTube blocks datacenter IPs.\n"
        "Add your SUPADATA_API_KEY in HF Space Secrets for reliable access."
    )

# ─────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────

def chunk(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks

def build_rag(video_id: str, transcript: str) -> None:
    chunks = chunk(transcript)
    embs   = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embs   = embs.astype("float32")
    index  = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    rag_store[video_id] = {"chunks": chunks, "index": index}
    print(f"  RAG index: {len(chunks)} chunks")

def retrieve(video_id: str, question: str, k: int = 4) -> list[str]:
    store = rag_store.get(video_id)
    if not store:
        raise RuntimeError("Summarize a video first, then ask questions.")
    q    = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    q    = q.astype("float32")
    _, I = store["index"].search(q, min(k, len(store["chunks"])))
    return [store["chunks"][i] for i in I[0] if i >= 0]

# ─────────────────────────────────────────────────────────────────
# GROQ  (llama-3.3-70b — ~2 sec, free tier)
# ─────────────────────────────────────────────────────────────────

PROMPTS = {
    "brief":         "Write a short, clear summary in 2-3 paragraphs. Cover the main topic and key takeaway.",
    "detailed":      "Write a detailed analysis with clear sections and headings. Cover: main topic, key points, examples, conclusion.",
    "bullet_points": "Summarize into bullet points grouped under bold headings. Give at least 6 specific bullet points.",
    "key_quotes":    "Extract the 5-7 most important insights and key takeaways. Label each clearly.",
}

def groq_summary(transcript: str, mode: str) -> str:
    prompt = PROMPTS.get(mode, PROMPTS["brief"])
    resp   = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional YouTube video summarizer.\n"
                    "The transcript may be in Telugu, Hindi, English or any language.\n"
                    "ALWAYS write your response in clear, professional English.\n"
                    "Base your summary ONLY on the transcript. Never invent facts.\n"
                    "Ignore subtitle formatting, timestamps, or metadata.\n"
                    "Reply in clean Markdown."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nTRANSCRIPT:\n{transcript[:8000]}",
            },
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return resp.choices[0].message.content

def groq_answer(video_id: str, question: str) -> str:
    chunks  = retrieve(video_id, question)
    context = "\n\n---\n\n".join(chunks)
    resp    = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant for a YouTube video.\n"
                    "Answer using ONLY the transcript context provided.\n"
                    "If the answer is not present, say so clearly.\n"
                    "Always answer in English. Be concise and direct."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nTranscript context:\n{context}",
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────

@app.post("/summarize")
def summarize_video(data: VideoRequest):
    try:
        video_id = extract_video_id(data.url)
        if not video_id:
            return {"success": False, "error": "Invalid YouTube URL."}

        meta       = fetch_meta(video_id)
        transcript = get_transcript(video_id, data.url)
        build_rag(video_id, transcript)
        summary    = groq_summary(transcript, data.mode)

        return {
            "success":    True,
            "video_id":   video_id,
            "title":      meta["title"],
            "channel":    meta["channel"],
            "thumbnail":  meta["thumbnail"],
            "summary":    summary,
            "word_count": len(transcript.split()),
        }

    except Exception as e:
        print(f"❌ SUMMARIZE: {e}")
        return {"success": False, "error": str(e)}


@app.post("/ask")
def ask_video(data: AskRequest):
    try:
        if not data.question.strip():
            return {"success": False, "error": "Please type a question."}
        answer = groq_answer(data.video_id, data.question.strip())
        return {"success": True, "answer": answer}
    except Exception as e:
        print(f"❌ ASK: {e}")
        return {"success": False, "error": str(e)}


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "groq":            bool(GROQ_KEY),
        "assemblyai":      bool(ASSEMBLYAI_KEY),
        "supadata":        bool(SUPADATA_KEY),
        "rag_videos":      len(rag_store),
    }
