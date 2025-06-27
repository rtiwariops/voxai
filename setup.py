#!/usr/bin/env python3
#  core.py – Gemini Flash 2.5, reliable streaming
#  Updated 2025-06-27

import os, sys, threading, logging, time, numpy as np
from datetime import date
from pathlib import Path
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# ── ENV / MODEL ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.

Return **only** in this exact Markdown layout:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub-bullet (optional, ≤ 20 words)

There must be a newline after the heading before the first bullet.

Cover purpose, durability/availability, security, pricing, limits, and common
use-cases. If you truly don’t know, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

p = Path("system_prompt.txt")
if p.exists():
    SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()

model = genai.GenerativeModel(model_name=MODEL,
                              system_instruction=SYSTEM_PROMPT)
chat  = model.start_chat()

# ── ASR ──────────────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i,d in enumerate(sd.query_devices())
               if "blackhole" in d["name"].lower())
except StopIteration:
    DEV = None

chunks: list[np.ndarray] = []
recording = False
last_txt  = ""

# ── AUDIO HELPERS ────────────────────────────────────────────────────────────
def _audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
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
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")

    if not chunks:
        last_txt = ""
        logger.info("TRANSCRIBED::")
        return

    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
    last_txt = " ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ── GEMINI STREAM (simple & safe) ────────────────────────────────────────────
def _stream_or_fallback(prompt: str):
    """
    1) Try streaming; flush every chunk exactly as Gemini sends it.
    2) If the stream ends with only a heading (no bullets), immediately
       request the same prompt non-streaming and print the full answer.
    """
    logger.info("CHUNK::[THINKING]")
    buf = ""
    any_bullet = False
    try:
        it = chat.send_message(
            prompt, stream=True,
            generation_config={"temperature":0.4, "max_output_tokens":800},
        )
        for part in it:
            text = getattr(part, "text", "")
            if not text:
                continue
            sys.stdout.flush()              # ensure interleaved order
            logger.info(f"CHUNK::{text}")
            buf += text
            if "- " in text:
                any_bullet = True
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        logger.error("CHUNK::[END]")
        return

    # fallback: no bullet seen → re-ask non-stream
    if not any_bullet:
        try:
            full = chat.send_message(prompt, stream=False).text
            logger.info(f"CHUNK::{full.strip()}")
        except Exception as e:
            logger.error(f"CHUNK::[ERROR] fallback: {e}")

    logger.info("CHUNK::[END]")

def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript")
        logger.error("CHUNK::[END]")
        return
    threading.Thread(target=_stream_or_fallback,
                     args=(f"Interview Question: {q}",),
                     daemon=True).start()

# ── CLI LOOP ─────────────────────────────────────────────────────────────────
def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for line in sys.stdin:
        cmd = line.strip().upper()
        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.startswith("ASK"):
            ask_ai()

if __name__ == "__main__":
    main_loop()
