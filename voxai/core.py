#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Voice‑to‑Gemini chat loop with non‑blocking live streaming
# ──────────────────────────────────────────────────────────────────────────────

"""
Key changes vs previous version
────────────────────────────────
1. **ask_ai() now streams in a background thread** so the stdin loop never blocks.  
2. **Each token chunk is emitted on its own line** using the original logger – this
   matches the UI’s "one‑line‑per‑chunk" contract so nothing appears to freeze.  
3. Introduced a short “thinking” line when the call starts, letting the UI show
   a spinner immediately.
"""

import os
import sys
import threading
from pathlib import Path
import logging

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG & LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="from-python:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-pro-latest")

if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are TechMentor, a senior software architect.
• Reply **only** with Markdown headings (## Heading) followed by concise bullet points.
• No paragraphs; each bullet ≤ 20 words.
• Use fenced code blocks for code (≤ 10 lines).
""".strip()

custom_prompt = Path("system_prompt.txt")
if custom_prompt.exists():
    try:
        SYSTEM_PROMPT = custom_prompt.read_text(encoding="utf-8").strip()
        logger.info(f"STATUS:: Loaded system prompt from {custom_prompt}")
    except Exception as e:
        logger.info(f"STATUS:: Failed to read {custom_prompt}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  INITIALISE GEMINI CHAT
# ──────────────────────────────────────────────────────────────────────────────

genai.configure(api_key=API_KEY)
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
#  WHISPER ASR SETUP
# ──────────────────────────────────────────────────────────────────────────────

whisper = WhisperModel("base", compute_type="int8")
SR = 16000
CH = 1
BS = 4000

try:
    DEV = next(i for i, d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None
    logger.info("STATUS:: No 'blackhole' device found; using default input")

# ──────────────────────────────────────────────────────────────────────────────
#  STATE VARS
# ──────────────────────────────────────────────────────────────────────────────

chunks      = []
recording   = False
last_txt    = ""

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def audio_cb(indata, frames, time, status):
    if status:
        logger.info(f"STATUS:: Audio warning: {status}")
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    chunks = []
    recording = True
    logger.info("STATUS:: Recording Started")

    def _rec_thread():
        stream_args = dict(samplerate=SR, channels=CH, blocksize=BS, callback=audio_cb)
        if DEV is not None:
            stream_args["device"] = DEV
        with sd.InputStream(**stream_args):
            while recording:
                sd.sleep(100)

    threading.Thread(target=_rec_thread, daemon=True).start()

def stop_and_transcribe():
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")

    if not chunks:
        last_txt = ""
        logger.info("TRANSCRIBED::")
        return

    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    try:
        segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        last_txt = ""
        logger.error(f"CHUNK::[ERROR] Whisper transcription failed: {e}")

    logger.info(f"TRANSCRIBED::{last_txt}")

# ──────────────────────────────────────────────────────────────────────────────
#  AI THREAD – non‑blocking streaming
# ──────────────────────────────────────────────────────────────────────────────

def _ai_stream_thread(prompt: str):
    """Runs in background so UI never freezes."""
    # Emit early notification so UI can show spinner
    logger.info("CHUNK::[THINKING]")

    try:
        iterator = chat.send_message(prompt, stream=True)
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed: {e}")
        logger.error("CHUNK::[END]")
        return

    try:
        for part in iterator:
            text = getattr(part, "text", "")
            if text:
                logger.info(f"CHUNK::{text}")  # newline after each chunk
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming exception: {e}")
    finally:
        logger.info("CHUNK::[END]")


def ask_ai():
    if not last_txt.strip():
        logger.error("CHUNK::[ERROR] No transcript to send to AI.")
        logger.error("CHUNK::[END]")
        return

    threading.Thread(target=_ai_stream_thread, args=(last_txt,), daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
#  COMMAND LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for line in sys.stdin:
        raw = line.rstrip("\n")
        logger.info(f"STATUS:: main_loop got raw line → '{raw}'")
        cmd = raw.strip().upper()

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
