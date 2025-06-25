#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Voice‑to‑Gemini chat loop (interview‑style answers + safe streaming)
# ──────────────────────────────────────────────────────────────────────────────

"""
Changes in this patch
─────────────────────
* Replaced **max_tokens** with **max_output_tokens** in `generation_config` to match
  Gemini API field names and fix the runtime error.
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

logging.basicConfig(level=logging.INFO, format="from-python:%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-pro-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT – interview format
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are TechMentor, a senior technical interviewer.
For *every* user question, follow **exactly** this Markdown layout:

## <Concise Topic Heading>
- **Question**: <repeat the user’s question in one sentence>
- **Answer**:
  - Bullet 1 (≤ 20 words)
  - Bullet 2
  - … (as many as needed)

Rules:
- Only Markdown headings + bullets; never prose paragraphs.
- Keep language clear, technically precise, mid‑to‑senior engineer level.
- Code snippets ≤ 10 lines inside fenced blocks.
""".strip()

custom_prompt = Path("system_prompt.txt")
if custom_prompt.exists():
    try:
        SYSTEM_PROMPT = custom_prompt.read_text(encoding="utf-8").strip()
        logger.info(f"STATUS:: Loaded system prompt from {custom_prompt}")
    except Exception as e:
        logger.info(f"STATUS:: Failed to read {custom_prompt}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  INITIALISE GEMINI
# ──────────────────────────────────────────────────────────────────────────────

genai.configure(api_key=API_KEY)
try:
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat()
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
        kwargs = dict(samplerate=SR, channels=CH, blocksize=BS, callback=audio_cb)
        if DEV is not None:
            kwargs["device"] = DEV
        with sd.InputStream(**kwargs):
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
#  AI STREAM – line‑safe
# ──────────────────────────────────────────────────────────────────────────────

def _stream_to_logger(prompt: str):
    logger.info("CHUNK::[THINKING]")
    try:
        iterator = chat.send_message(
            prompt,
            stream=True,
            generation_config={"temperature": 0.4, "max_output_tokens": 512},
        )
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed: {e}"); logger.error("CHUNK::[END]"); return

    buffer = ""
    try:
        for part in iterator:
            text = getattr(part, "text", "")
            if not text:
                continue
            buffer += text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line:
                    logger.info(f"CHUNK::{line}")
        if buffer.strip():
            logger.info(f"CHUNK::{buffer.strip()}")
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming exception: {e}")
    finally:
        logger.info("CHUNK::[END]")


def ask_ai():
    if not (q := last_txt.strip()):
        logger.error("CHUNK::[ERROR] No transcript to send to AI."); logger.error("CHUNK::[END]"); return
    user_prompt = f"Interview Question: {q}"
    threading.Thread(target=_stream_to_logger, args=(user_prompt,), daemon=True).start()

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
