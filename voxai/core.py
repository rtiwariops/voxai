#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Voice-driven interview helper for **Gemini Flash 2.5**
# ──────────────────────────────────────────────────────────────────────────────
"""
What it does
============
1. Listens on the mic (or any input device containing “blackhole” on macOS).
2. Converts speech → text with Whisper-base (faster-whisper).
3. Pulls a *free* one-paragraph web snippet (DuckDuckGo → Wikipedia fallback)
   to ground the answer.
4. Streams the question + context to Gemini Flash 2.5
   (`gemini-1.5-flash-latest`) and streams tokens back as newline-delimited
   chunks (`CHUNK::...`).
5. Works entirely with free endpoints except for your Google AI key.

Env vars
--------
GENAI_API_KEY   – **required** (Google API key)
GENAI_MODEL     – optional (defaults to `gemini-1.5-flash-latest`)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  Built-ins / stdlib
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, threading, logging, requests
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus

# ──────────────────────────────────────────────────────────────────────────────
#  Third-party deps
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="from-python:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

# ──────────────────────────────────────────────────────────────────────────────
#  ENV + GEMINI INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-2.0-flash-latest")   # Flash 2.5

if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.

Reply *only* in this Markdown layout:

## &lt;Topic&nbsp;Heading&gt;
- **Question**: &lt;repeat the question&gt;
- **Answer**:
  - Bullet 1 (≤ 20 words)
  - Bullet 2 …
  
Ground your answer in the *Context* I provide.  
If—after using Context—no reliable public info exists, reply exactly:
"Unknown: no reliable public source found as of {date.today():%Y-%m-%d}."
""".strip()

custom = Path("system_prompt.txt")
if custom.exists():
    SYSTEM_PROMPT = custom.read_text(encoding="utf-8").strip()
    logger.info(f"STATUS:: Loaded custom prompt from {custom}")

try:
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat()
    logger.info(f"STATUS:: Started chat with model '{MODEL}'")
except Exception as e:
    logger.error(f"CHUNK::[ERROR] Could not start chat: {e}")
    logger.error("CHUNK::[END]")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  WHISPER ASR CONFIG
# ──────────────────────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000  # sample-rate, channels, block-size

try:
    DEV = next(i for i, d in enumerate(sd.query_devices())
               if "blackhole" in d["name"].lower())
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None
    logger.info("STATUS:: No 'blackhole' device found; using default input")

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ──────────────────────────────────────────────────────────────────────────────
chunks: list[np.ndarray] = []
recording = False
last_txt  = ""

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
    """BEGIN recording in a background thread."""
    global recording, chunks
    if recording:
        return
    chunks, recording = [], True
    logger.info("STATUS:: Recording Started")

    def _rec():
        kw = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None:
            kw["device"] = DEV
        with sd.InputStream(**kw):
            while recording:
                sd.sleep(100)

    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe():
    """END recording and run Whisper."""
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")

    if not chunks:
        last_txt = ""
        logger.info("TRANSCRIBED::")
        return

    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    try:
        segments, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        last_txt = ""
        logger.error(f"CHUNK::[ERROR] Whisper transcription failed: {e}")

    logger.info(f"TRANSCRIBED::{last_txt}")

# ──────────────────────────────────────────────────────────────────────────────
#  FREE WEB-SNIPPET RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────
def _duckduckgo(query: str) -> str:
    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1"
        data = requests.get(url, timeout=5).json()
        if (abstract := data.get("AbstractText")):
            return abstract
        for item in data.get("RelatedTopics", [])[:3]:
            if isinstance(item, dict) and item.get("Text"):
                return item["Text"]
    except Exception as e:
        logger.info(f"STATUS:: DuckDuckGo error: {e}")
    return ""

def _wiki_summary(title: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}"
        data = requests.get(url, timeout=5).json()
        if data.get("extract") and data.get("type") != "disambiguation":
            return data["extract"]
    except Exception as e:
        logger.info(f"STATUS:: Wikipedia summary error: {e}")
    return ""

def _wiki_title_search(query: str) -> str:
    try:
        url = f"https://en.wikipedia.org/w/rest.php/v1/search/title?q={quote_plus(query)}&limit=1"
        pages = requests.get(url, timeout=5).json().get("pages", [])
        if pages:
            return _wiki_summary(pages[0]["title"])
    except Exception as e:
        logger.info(f"STATUS:: Wikipedia search error: {e}")
    return ""

def fetch_snippet(query: str) -> str:
    """Return a short web snippet using only free endpoints."""
    aliases = [query]
    q_low = query.lower()

    # Smart aliasing for “Azure Foundry”
    if "azure" in q_low and "foundry" in q_low and "ai" not in q_low:
        aliases += ["Azure AI Foundry", "Microsoft Azure AI Foundry"]

    if not q_low.endswith(" microsoft"):
        aliases.append(query + " Microsoft")

    for q in aliases:
        for func in (_duckduckgo, _wiki_summary, _wiki_title_search):
            if snippet := func(q):
                return snippet
    return ""

# ──────────────────────────────────────────────────────────────────────────────
#  GEMINI STREAMING
# ──────────────────────────────────────────────────────────────────────────────
def _stream_to_logger(prompt: str):
    logger.info("CHUNK::[THINKING]")  # spinner hook
    try:
        it = chat.send_message(
            prompt,
            stream=True,
            generation_config={"temperature": 0.4, "max_output_tokens": 512},
        )
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed: {e}")
        logger.error("CHUNK::[END]")
        return

    buf = ""
    try:
        for part in it:  # stream of partial responses
            text = getattr(part, "text", "")
            if not text:
                continue
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line:
                    logger.info(f"CHUNK::{line}")
        if buf.strip():
            logger.info(f"CHUNK::{buf.strip()}")
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming exception: {e}")
    finally:
        logger.info("CHUNK::[END]")

# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC ask_ai() ENTRY
# ──────────────────────────────────────────────────────────────────────────────
def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript to send to AI.")
        logger.error("CHUNK::[END]")
        return

    context = fetch_snippet(q)
    full_prompt = (
        f"Context:\n{context}\n\nInterview Question: {q}"
        if context else
        f"Interview Question: {q}"
    )
    threading.Thread(target=_stream_to_logger, args=(full_prompt,), daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
#  COMMAND LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main_loop():
    """
    Read simple commands from stdin:
      START   – begin recording
      STOP    – stop & transcribe
      ASK     – send last transcript to Gemini
    Anything else is ignored.
    """
    logger.info("STATUS:: voxai.core ready")
    for line in sys.stdin:
        cmd = line.rstrip("\n").strip().upper()
        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.startswith("ASK"):
            ask_ai()
        else:
            logger.info(f"STATUS:: Unknown command → '{cmd}'")

if __name__ == "__main__":
    main_loop()
