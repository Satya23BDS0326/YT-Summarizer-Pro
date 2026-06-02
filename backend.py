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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
SUPADATA_KEY = os.getenv("SUPADATA_API_KEY", "")

if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

if not ASSEMBLYAI_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY missing in .env")

groq_client = Groq(api_key=GROQ_KEY)

aai.settings.api_key = ASSEMBLYAI_KEY

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────

print("⏳ Loading embedding model...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Embedding model ready.")

rag_store = {}

# ─────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class VideoRequest(BaseModel):
    url: str
    mode: str = "brief"

class AskRequest(BaseModel):
    video_id: str
    question: str

# ─────────────────────────────────────────────
# URL → VIDEO ID
# ─────────────────────────────────────────────

def extract_video_id(url: str):

    url = url.strip()

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

# ─────────────────────────────────────────────
# CLEANERS
# ─────────────────────────────────────────────

def clean(text: str):

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize(items):

    parts = []

    for item in items:

        if isinstance(item, dict):
            parts.append(item.get("text", "").replace("\n", " "))
        else:
            parts.append(str(getattr(item, "text", item)).replace("\n", " "))

    return clean(" ".join(parts))

# ─────────────────────────────────────────────
# PRODUCTION API GATEWAY (Supadata Unblocked Layer)
# ─────────────────────────────────────────────

def layer0_supadata_gateway(video_id: str):
    if not SUPADATA_KEY:
        print("⚡ Layer0 Skipped: SUPADATA_API_KEY environment variable is empty.")
        return None
    try:
        url = "https://api.supadata.ai/v1/youtube/transcript"
        params = {"videoId": video_id}
        headers = {"x-api-key": SUPADATA_KEY}
        
        print(f"📡 Querying Supadata Gateway for Video ID: {video_id}...")
        # Increased timeout to 30 seconds to allow internal proxy rotation to finish safely
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"📦 Supadata HTTP Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                content = data.get("content")
                if isinstance(content, list):
                    text = " ".join([item.get("text", "") for item in content if isinstance(item, dict)])
                elif isinstance(content, str):
                    text = content
                else:
                    text = data.get("text", "")
                
                if len(text) > 100:
                    return clean(text)
            elif isinstance(data, list):
                text = " ".join([item.get("text", "") for item in data if isinstance(item, dict)])
                if len(text) > 100:
                    return clean(text)
        else:
            print(f"❌ Supadata Error Log Output: {response.text}")
            
    except Exception as e:
        print("💥 Layer0 Ingestion Exception:", e)
    return None

# ─────────────────────────────────────────────
# LAYER 1
# ─────────────────────────────────────────────

def layer1_transcript_api(video_id):

    try:

        ytt = YouTubeTranscriptApi()

        tlist = ytt.list(video_id)

        for finder in (

            lambda t: t.find_transcript(["te", "hi", "en"]),

            lambda t: t.find_generated_transcript(["te", "hi", "en"]),

            lambda t: next(iter(t)).translate("en"),

            lambda t: next(iter(t)),
        ):

            try:

                raw = finder(tlist).fetch()

                text = normalize(raw)

                if len(text) > 100:
                    return text

            except Exception as e:
                print("Layer1 inner:", e)

    except Exception as e:
        print("Layer1:", e)

    return None

# ─────────────────────────────────────────────
# LAYER 2
# ─────────────────────────────────────────────

def _clean_vtt(raw):

    text = re.sub(r"<[^>]+>", " ", raw)

    text = re.sub(
        r"WEBVTT|Kind:.*|Language:.*",
        " ",
        text
    )

    text = re.sub(
        r"\d{2}:\d{2}:\d{2}[\.,]\d{3}.*-->.*",
        " ",
        text
    )

    text = re.sub(
        r"^\d+\s*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(r"&amp;", "&", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def layer2_ytdlp_captions(url):

    try:

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "cookiefile": str(BASE_DIR / "cookies.txt"),
            "socket_timeout": 3,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(url, download=False)

        subs = {
            **(info.get("subtitles") or {}),
            **(info.get("automatic_captions") or {})
        }

        if not subs:
            return None

        priority = ["te", "hi", "en"] + [
            k for k in subs if k not in ("te", "hi", "en")
        ]

        for lang in priority:

            entries = subs.get(lang)

            if not entries:
                continue

            try:

                r = requests.get(entries[0]["url"], timeout=4)

                r.raise_for_status()

                text = _clean_vtt(r.text)

                if len(text) > 100:
                    return text

            except Exception as e:
                print("Layer2 inner:", e)

    except Exception as e:
        print("Layer2:", e)

    return None

# ─────────────────────────────────────────────
# LAYER 3
# ─────────────────────────────────────────────

def layer3_assemblyai(url):

    try:

        print("⏳ Downloading audio...")

        with tempfile.TemporaryDirectory() as tmpdir:

            ydl_opts = {

                "format": "worstaudio/worst",

                "outtmpl": str(Path(tmpdir) / "%(id)s.%(ext)s"),

                "quiet": True,

                "no_warnings": True,

                "noplaylist": True,

                "cookiefile": str(BASE_DIR / "cookies.txt"),
                "socket_timeout": 3,

                "extractor_args": {
                    "youtube": {
                        "player_client": ["android"]
                    }
                },

                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "96",
            }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:

                info = ydl.extract_info(url, download=True)

                audio_path = Path(
                    ydl.prepare_filename(info)
                ).with_suffix(".mp3")

            if not audio_path.exists():

                candidates = list(Path(tmpdir).glob("*.mp3"))

                audio_path = candidates[0] if candidates else None

            if not audio_path:
                return None

            config = aai.TranscriptionConfig(
                language_detection=True,
            )

            transcriber = aai.Transcriber()

            result = transcriber.transcribe(
                str(audio_path),
                config=config
            )

            if result.status == aai.TranscriptStatus.error:
                print(result.error)
                return None

            text = clean(result.text or "")

            if len(text) > 100:
                return text

    except Exception as e:
        print("Layer3:", e)

    return None

# ─────────────────────────────────────────────
# MAIN TRANSCRIPT
# ─────────────────────────────────────────────

def get_transcript(video_id, url):

    layers = [
        ("Supadata_Gateway", lambda: layer0_supadata_gateway(video_id)),
        ("Layer1", lambda: layer1_transcript_api(video_id)),
        ("Layer2", lambda: layer2_ytdlp_captions(url)),
        ("Layer3", lambda: layer3_assemblyai(url)),
    ]

    for name, fn in layers:

        print(f"▶ Trying {name}")

        try:

            text = fn()

            if text:
                print(f"✅ {name} success")
                return text

        except Exception as e:
            print(f"{name} failed:", e)

    raise RuntimeError(
        "Could not get transcript for this video. On Hugging Face: YouTube blocks datacenter IPs. Add your SUPADATA_API_KEY in HF Space Secrets for reliable access."
    )

# ─────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────

def chunk(text, size=300, overlap=50):

    words = text.split()

    chunks = []

    i = 0

    while i < len(words):

        chunks.append(" ".join(words[i:i+size]))

        if i + size >= len(words):
            break

        i += size - overlap

    return chunks

def build_rag(video_id, transcript):

    chunks = chunk(transcript)

    embs = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embs = embs.astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])

    index.add(embs)

    rag_store[video_id] = {
        "chunks": chunks,
        "index": index,
    }

def retrieve(video_id, question, k=3):

    store = rag_store.get(video_id)

    if not store:
        raise RuntimeError("Summarize video first.")

    q = embed_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    q = q.astype("float32")

    _, I = store["index"].search(
        q,
        min(k, len(store["chunks"]))
    )

    return [
        store["chunks"][i]
        for i in I[0]
        if i >= 0
    ]

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

PROMPTS = {

    "brief": "Give a short summary.",

    "detailed": "Give detailed explanation with headings.",

    "bullet_points": "Summarize in bullet points.",

    "key_quotes": "Extract important insights.",
}

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def groq_summary(transcript, mode):

    prompt = PROMPTS.get(mode, PROMPTS["brief"])

    short_transcript = transcript[:2500]

    response = groq_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content": (
                    "You are a professional YouTube video summarizer. "
                    "The transcript may be in Telugu, Hindi, English or any language. "
                    "ALWAYS generate the final summary ONLY in clear professional English. "
                    "Never answer in Telugu or Hindi. "
                    "Write clean markdown with proper headings and bullet points."
                ),
            },

            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"TRANSCRIPT:\n{short_transcript}"
                ),
            },
        ],

        temperature=0.3,

        max_tokens=400,
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────────
# RAG Q&A
# ─────────────────────────────────────────────

def groq_answer(video_id, question):

    chunks = retrieve(video_id, question)

    context = "\n\n---\n\n".join(chunks)

    response = groq_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content": (
                    "Answer ONLY using the transcript context provided. "
                    "Regardless of transcript language, ALWAYS answer in English."
                ),
            },

            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Transcript:\n{context}"
                ),
            },
        ],

        temperature=0.2,

        max_tokens=300,
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/summarize")
def summarize_video(data: VideoRequest):

    try:

        video_id = extract_video_id(data.url)

        if not video_id:
            return {
                "success": False,
                "error": "Invalid YouTube URL."
            }

        transcript = get_transcript(
            video_id,
            data.url
        )

        build_rag(video_id, transcript)

        summary = groq_summary(
            transcript,
            data.mode
        )

        return {
            "success": True,
            "video_id": video_id,
            "summary": summary,
            "word_count": len(transcript.split()),
        }

    except Exception as e:

        print("❌ SUMMARIZE:", e)

        return {
            "success": False,
            "error": str(e),
        }

@app.post("/ask")
def ask_video(data: AskRequest):

    try:

        if not data.question.strip():

            return {
                "success": False,
                "error": "Please type a question."
            }

        answer = groq_answer(
            data.video_id,
            data.question.strip()
        )

        return {
            "success": True,
            "answer": answer,
        }

    except Exception as e:

        print("❌ ASK:", e)

        return {
            "success": False,
            "error": str(e),
        }
