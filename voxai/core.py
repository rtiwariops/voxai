#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Voice-to-Gemini chat loop with live token streaming
# ──────────────────────────────────────────────────────────────────────────────

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
#  CONFIGURATION & STARTUP
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="from-python:%(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

# Load env vars
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-pro-latest")  # sensible default

if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are TechMentor, a senior software architect.
• Reply **only** with Markdown headings (`## Heading`) followed by concise bullet points.
• Never write full prose paragraphs.
• Keep each bullet ≤ 20 words, use plain technical English.
• Use fenced code blocks for code snippets (≤ 10 lines).
""".strip()

# If a `system_prompt.txt` exists in CWD, override:
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
        system_instruction=SYSTEM_PROMPT
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
SR = 16000  # sample rate
CH = 1      # channels
BS = 4000   # block size (samples)

# Auto-select BlackHole or default input
try:
    DEV = next(i for i, d in enumerate(sd.query_devices())
               if "blackhole" in d["name"].lower())
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None
    logger.info("STATUS:: No 'blackhole' device found; using default input")

# ──────────────────────────────────────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────────────────────────────────────

chunks      = []   # list of recorded audio blocks
recording   = False
last_txt    = ""   # most recent ASR text

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO RECORD / TRANSCRIBE
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
#  GEMINI CALL + LIVE STREAM
# ──────────────────────────────────────────────────────────────────────────────

def ask_ai():
    if not last_txt.strip():
        logger.error("CHUNK::[ERROR] No transcript to send to AI.")
        logger.error("CHUNK::[END]")
        return

    try:
        stream = chat.send_message(last_txt, stream=True)  # iterator of chunks
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed: {e}")
        logger.error("CHUNK::[END]")
        return

    # Emit answer in a single teletype line
    sys.stdout.write("CHUNK::")
    sys.stdout.flush()

    any_chunk = False
    try:
        for part in stream:
            text = getattr(part, "text", "")
            if text:
                any_chunk = True
                sys.stdout.write(text)
                sys.stdout.flush()
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming exception: {e}")
    finally:
        if not any_chunk:
            sys.stdout.write("[WARN] Streaming returned zero chunks.")
        sys.stdout.write("\nCHUNK::[END]\n")
        sys.stdout.flush()

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP (stdin commands: START, STOP, ASK)
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
            logger.info(f"STATUS:: Unknown command → '{raw}'")

if __name__ == "__main__":
    main_loop()
