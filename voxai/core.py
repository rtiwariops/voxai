#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Interview helper with *free* web‑snippet retrieval (DuckDuckGo + Wikipedia)
# ──────────────────────────────────────────────────────────────────────────────
"""
What’s in this FREE edition
───────────────────────────
* Uses **DuckDuckGo Instant Answer API** (no key, no cost, ~100 req/min) and
  falls back to **Wikipedia REST summary** if DuckDuckGo returns nothing.
* Removed Bing key and paid tiers entirely.
* All other behaviour (ASR, streaming, interview prompt) unchanged.
"""

import os
import sys
import threading
from pathlib import Path
import logging
from datetime import date
from urllib.parse import quote_plus

import requests  # only free endpoints used
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG & LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT – interview + context
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are TechMentor, a senior technical interviewer.
For **every** question, answer in this Markdown format:

## <Topic>
- **Question**: <repeat question>
- **Answer**:
  - Bullet 1 (≤ 20 words)
  - Bullet 2 …

Ground your answer in the *Context* provided. If, after using the context, no
reliable info exists, reply:
"Unknown: no reliable public source found as of {date.today():%Y-%m-%d}."
""".strip()

if (fp := Path("system_prompt.txt")).exists():
    try:
        SYSTEM_PROMPT = fp.read_text(encoding="utf-8").strip();
        logger.info(f"STATUS:: Loaded custom prompt from {fp}")
    except Exception as e:
        logger.info(f"STATUS:: Failed to read {fp}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  GENAI INIT
# ──────────────────────────────────────────────────────────────────────────────

genai.configure(api_key=API_KEY)
try:
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=SYSTEM_PROMPT)
    chat  = model.start_chat()
    logger.info(f"STATUS:: Started chat with model '{MODEL}'")
except Exception as e:
    logger.error(f"CHUNK::[ERROR] Could not start chat: {e}"); logger.error("CHUNK::[END]"); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  WHISPER ASR
# ──────────────────────────────────────────────────────────────────────────────

whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i, d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None; logger.info("STATUS:: No 'blackhole' device found; using default input")

# ──────────────────────────────────────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────────────────────────────────────

chunks, recording, last_txt = [], False, ""

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    chunks, recording = [], True
    logger.info("STATUS:: Recording Started")
    def _rec():
        args = dict(samplerate=SR, channels=CH, blocksize=BS, callback=audio_cb)
        if DEV is not None:
            args["device"] = DEV
        with sd.InputStream(**args):
            while recording:
                sd.sleep(100)
    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe():
    global recording, last_txt
    recording = False; logger.info("STATUS:: Recording Stopped")
    if not chunks:
        last_txt = ""; logger.info("TRANSCRIBED::"); return
    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    try:
        segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        last_txt = ""; logger.error(f"CHUNK::[ERROR] Whisper transcription failed: {e}")
    logger.info(f"TRANSCRIBED::{last_txt}")

# ──────────────────────────────────────────────────────────────────────────────
#  FREE RETRIEVAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _duckduckgo(query: str) -> str:
    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1"
        data = requests.get(url, timeout=5).json()
        abstract = data.get("AbstractText") or ""
        if abstract:
            return abstract
        rel_topics = data.get("RelatedTopics", [])
        if rel_topics and isinstance(rel_topics[0], dict):
            return rel_topics[0].get("Text", "")
    except Exception as e:
        logger.info(f"STATUS:: DuckDuckGo error: {e}")
    return ""

def _wikipedia_summary(query: str) -> str:
    try:
        title = quote_plus(query.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        data = requests.get(url, timeout=5).json()
        if data.get("extract") and data.get("description") != "Wikimedia disambiguation page":
            return data["extract"]
    except Exception as e:
        logger.info(f"STATUS:: Wikipedia error: {e}")
    return ""

def fetch_snippet(query: str) -> str:
    """Return free web summary, preferring DuckDuckGo, fallback Wikipedia."""
    snippet = _duckduckgo(query)
    if snippet:
        return snippet
    return _wikipedia_summary(query)

# ──────────────────────────────────────────────────────────────────────────────
#  AI STREAM – line‑safe
# ──────────────────────────────────────────────────────────────────────────────

def _stream_to_logger(full_prompt: str):
    logger.info("CHUNK::[THINKING]")
    try:
        iterator = chat.send_message(full_prompt,
                                     stream=True,
                                     generation_config={"temperature": 0.4, "max_output_tokens": 512})
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed: {e}"); logger.error("CHUNK::[END]"); return

    buf = ""
    try:
        for part in iterator:
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


def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript to send to AI."); logger.error("CHUNK::[END]"); return

    context = fetch_snippet(q)
    prefix  = f"Context:\n{context}\n\n" if context else ""
    full_prompt = prefix + f"Interview Question: {q}"
    threading.Thread(target=_stream_to_logger, args=(full_prompt,), daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
#  COMMAND LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main_loop():
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
