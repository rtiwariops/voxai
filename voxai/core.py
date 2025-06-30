#!/usr/bin/env python3
#  core.py – Voice → Whisper → Gemini Flash 2.5
#  • answers in “## Heading + bullets” format
#  • streams line-by-line
#  • safety abort → single-shot fallback
#  Updated 2025-06-27

import os, sys, threading, logging, time, numpy as np
from datetime import date
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  ENV + GEMINI INIT
# ─────────────────────────────────────────────────────────────────────────────
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

Return **only** in this Markdown layout:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub-bullet (optional, ≤ 20 words)

(Newline required after the heading before the first bullet.)

Cover purpose, architecture, durability/availability, security, pricing,
limits, typical use-cases.  
If you truly don’t know, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

p = Path("system_prompt.txt")
if p.exists():
    SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()

model = genai.GenerativeModel(model_name=MODEL,
                              system_instruction=SYSTEM_PROMPT)
chat  = model.start_chat()
logger.info(f"STATUS:: Started chat with model '{MODEL}'")

# ─────────────────────────────────────────────────────────────────────────────
#  WHISPER ASR
# ─────────────────────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i,d in enumerate(sd.query_devices())
               if "blackhole" in d["name"].lower())
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None
    logger.info("STATUS:: No 'blackhole' device found; using default input")

# ─────────────────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────────────────
chunks: list[np.ndarray] = []
recording = False
last_txt  = ""

# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    recording, chunks[:] = True, []
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
    try:
        segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        last_txt = ""
        logger.error(f"CHUNK::[ERROR] Whisper transcription failed: {e}")

    logger.info(f"TRANSCRIBED::{last_txt}")

# ─────────────────────────────────────────────────────────────────────────────
#  GEMINI STREAM  (safe, line-by-line)
# ─────────────────────────────────────────────────────────────────────────────
def _stream_safe(prompt: str):
    """
    Streams Gemini line-by-line.
    If a `ValueError` occurs when accessing `part.text`, we assume the
    stream was aborted for safety and retry once without streaming.
    """
    logger.info("CHUNK::[THINKING]")
    buffer = ""
    safety_abort = False

    try:
        iterator = chat.send_message(
            prompt,
            stream=True,
            generation_config={"temperature": 0.4, "max_output_tokens": 800},
        )

        for part in iterator:
            try:
                txt = part.text          # may raise ValueError on SAFETY
            except ValueError:
                safety_abort = True
                break

            if not txt:
                continue
            buffer += txt
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    logger.info(f"CHUNK::{line.strip()}")

        if buffer.strip():
            logger.info(f"CHUNK::{buffer.strip()}")

    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        safety_abort = True

    # Fallback if needed
    if safety_abort:
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
    threading.Thread(
        target=_stream_safe,
        args=(f"Interview Question: {q}",),
        daemon=True,
    ).start()

# ─────────────────────────────────────────────────────────────────────────────
#  CLI LOOP
# ─────────────────────────────────────────────────────────────────────────────
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
        else:
            logger.info(f"STATUS:: Unknown command → '{cmd}'")

if __name__ == "__main__":
    main_loop()
