import os
import sys

# Ensure UTF-8 output encoding on Windows consoles to prevent charmap crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Must be set BEFORE huggingface_hub / sentence_transformers is imported.
# Works around a known hf-xet bug where the Xet CAS backend returns
# 401 Unauthorized when downloading models like all-MiniLM-L6-v2.
# This forces huggingface_hub to fall back to plain HTTP downloads.
os.environ["HF_HUB_DISABLE_XET"] = "1"

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

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = [
    DEFAULT_GROQ_MODEL,
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "qwen/qwen3.6-27b",
]
ACTIVE_GROQ_MODEL = DEFAULT_GROQ_MODEL

def call_groq_chat(**kwargs):
    """
    Calls Groq chat completions with automatic failover and dynamic discovery.
    Guarantees resilience against model deprecations, decommissionings, and 404 model_not_found errors.
    """
    global ACTIVE_GROQ_MODEL
    candidates = []
    for m in [ACTIVE_GROQ_MODEL] + FALLBACK_MODELS:
        if m and m not in candidates:
            candidates.append(m)

    last_err = None
    for model_name in candidates:
        try:
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = model_name
            resp = groq_client.chat.completions.create(**call_kwargs)
            ACTIVE_GROQ_MODEL = model_name
            return resp
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["model_not_found", "does not exist", "decommissioned", "404", "not have access", "cannot be found"]):
                print(f"⚠️ Groq model '{model_name}' unavailable ({e}). Trying fallback...")
                last_err = e
                continue
            raise e

    # Dynamic fallback: query available chat models from the Groq account
    try:
        available_models = groq_client.models.list()
        for m in available_models.data:
            mid = m.id.lower()
            if any(bad in mid for bad in ["whisper", "guard", "safeguard", "orpheus"]):
                continue
            if m.id not in candidates:
                try:
                    call_kwargs = dict(kwargs)
                    call_kwargs["model"] = m.id
                    resp = groq_client.chat.completions.create(**call_kwargs)
                    ACTIVE_GROQ_MODEL = m.id
                    print(f"✅ Recovered with dynamically discovered Groq model: {m.id}")
                    return resp
                except Exception:
                    continue
    except Exception as dyn_err:
        print("⚠️ Dynamic model discovery failed:", dyn_err)

    raise last_err or RuntimeError("No compatible Groq chat model available.")

aai.settings.api_key = ASSEMBLYAI_KEY

# ─────────────────────────────────────────────
# COOKIES — safe runtime loading (never hardcoded)
# ─────────────────────────────────────────────

COOKIE_FILE = None

_cookie_content = os.getenv("YOUTUBE_COOKIES", "")

if _cookie_content:
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    _tmp.write(_cookie_content)
    _tmp.flush()
    COOKIE_FILE = _tmp.name
    print("✅ Cookies loaded from environment secret.")
elif (BASE_DIR / "cookies.txt").exists():
    COOKIE_FILE = str(BASE_DIR / "cookies.txt")
    print("✅ Cookies loaded from local cookies.txt.")
else:
    print("⚠️ No cookies found. yt-dlp will run without authentication.")

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
# LAYER 0 — Supadata Gateway
# ─────────────────────────────────────────────

def layer0_supadata_gateway(video_id: str):
    if not SUPADATA_KEY:
        print("⚡ Layer0 Skipped: SUPADATA_API_KEY variable is empty.")
        return None
    try:
        url = "https://api.supadata.ai/v1/transcript"
        params = {"url": f"https://www.youtube.com/watch?v={video_id}"}
        headers = {"x-api-key": SUPADATA_KEY}

        print(f"📡 Querying Supadata Gateway for Video ID: {video_id}...")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"📦 Supadata HTTP Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            def extract_texts(obj):
                strings = []
                if isinstance(obj, dict):
                    if "text" in obj and isinstance(obj["text"], str):
                        strings.append(obj["text"])
                    for v in obj.values():
                        if isinstance(v, (dict, list)):
                            strings.extend(extract_texts(v))
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, str):
                            strings.append(item)
                        elif isinstance(item, (dict, list)):
                            strings.extend(extract_texts(item))
                return strings

            extracted_strings = extract_texts(data)
            text = clean(" ".join(extracted_strings))

            if len(text) > 100:
                return text
        else:
            print(f"❌ Supadata Error: {response.text}")

    except Exception as e:
        print("💥 Layer0 Exception:", e)
    return None

# ─────────────────────────────────────────────
# METADATA FALLBACK — Supadata
# ─────────────────────────────────────────────

def fetch_supadata_metadata(video_id: str):
    if not SUPADATA_KEY:
        return None
    try:
        url = "https://api.supadata.ai/v1/metadata"
        params = {"url": f"https://www.youtube.com/watch?v={video_id}"}
        headers = {"x-api-key": SUPADATA_KEY}

        print(f"📡 Querying Supadata Metadata for Video ID: {video_id}...")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "Unknown Title")
            description = data.get("description", "")

            tags_list = data.get("tags", [])
            tags = ", ".join(tags_list) if isinstance(tags_list, list) else str(tags_list)

            channel_data = data.get("channel", {})
            channel = channel_data.get("title", "Unknown Channel") if isinstance(channel_data, dict) else str(channel_data)

            metadata_summary = f"Video Title: {title}\nChannel: {channel}\nTags: {tags}\n\nDescription:\n{description}"
            if len(metadata_summary.strip()) > 50:
                return clean(metadata_summary)
    except Exception as e:
        print("💥 Metadata Fallback Exception:", e)
    return None

# ─────────────────────────────────────────────
# OEMBED METADATA
# ─────────────────────────────────────────────

def fetch_oembed_meta(video_id: str):
    try:
        url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "YouTube Video"),
                "author": data.get("author_name", "Unknown Channel")
            }
    except Exception as e:
        print("⚠️ oEmbed Error:", e)
    return None

def generate_semantic_knowledge_base(title: str, author: str):
    try:
        response = call_groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior educator. "
                        "Compile a detailed educational breakdown of the given video topic "
                        "so a vector search model can perform RAG retrieval indexing."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate a knowledge base document for video titled: '{title}' by channel: '{author}'."
                }
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return clean(response.choices[0].message.content)
    except Exception as e:
        print("💥 Knowledge Base Generation Exception:", e)
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
        }

        if COOKIE_FILE:
            ydl_opts["cookiefile"] = COOKIE_FILE

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

                r = requests.get(entries[0]["url"], timeout=20)

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

            if COOKIE_FILE:
                ydl_opts["cookiefile"] = COOKIE_FILE

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

    raise RuntimeError("All transcript layers exhausted.")

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

    response = call_groq_chat(

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
                "content": f"{prompt}\n\nTRANSCRIPT:\n{short_transcript}",
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

    response = call_groq_chat(

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
                "content": f"Question: {question}\n\nTranscript:\n{context}",
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

        transcript = None
        is_metadata_fallback = False

        try:
            transcript = get_transcript(video_id, data.url)
        except Exception as pipe_err:
            print(f"⚠️ All transcript layers failed: {pipe_err}. Trying metadata fallback...")
            transcript = fetch_supadata_metadata(video_id)
            if transcript:
                is_metadata_fallback = True

        if not transcript:
            meta = fetch_oembed_meta(video_id)
            if meta:
                print("🚀 Generating knowledge base from video metadata...")
                transcript = generate_semantic_knowledge_base(meta["title"], meta["author"])
                is_metadata_fallback = True

        if not transcript:
            return {
                "success": False,
                "error": "Could not get transcript for this video. It may be private, region-blocked, or have no captions."
            }

        build_rag(video_id, transcript)

        summary = groq_summary(transcript, data.mode)

        if is_metadata_fallback:
            summary = (
                "### ⚠️ Notice\n"
                "*Could not extract the actual transcript. This summary is based on the video's metadata and description only — not the full content.*\n\n"
                + summary
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
