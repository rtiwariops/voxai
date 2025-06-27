#!/usr/bin/env python3
#  core.py – Voice-to-Gemini Flash 2.5, detailed bullets, fast streaming
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
logging.basicConfig(
    level=logging.INFO,
    format="from-python:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"STATUS:: Running core.py from {__file__}")

# ─────────────────────────────────────────────────────────────────────────────
#  ENV + GEMINI INIT
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-flash-latest")   # Flash 2.5

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
  - Sub-bullet (≤ 20 words, optional)

**There must be a newline after the heading before the first “- ”.**

Cover purpose, architecture, durability/availability, security, pricing, limits,
and typical use-cases. If you truly don’t know, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
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

# ─────────────────────────────────────────────────────────────────────────────
#  WHISPER ASR
# ─────────────────────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000

try:
    DEV = next(i for i, d in enumerate(sd.query_devices())
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
    try:
        segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        last_txt = ""
        logger.error(f"CHUNK::[ERROR] Whisper transcription failed: {e}")

    logger.info(f"TRANSCRIBED::{last_txt}")

# ─────────────────────────────────────────────────────────────────────────────
#  GEMINI STREAM – heading & bullet auto-flush
# ─────────────────────────────────────────────────────────────────────────────
def _stream(prompt: str):
    logger.info("CHUNK::[THINKING]")

    try:
        iterator = chat.send_message(
            prompt,
            stream=True,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 800,  # allow long answers
            },
        )
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        logger.error("CHUNK::[END]")
        return

    buf, last_emit = "", time.time()

    def _flush():
        """Emit buffered text as one CHUNK line."""
        nonlocal buf, last_emit
        if buf.strip():
            logger.info(f"CHUNK::{buf.strip()}")
            buf, last_emit = "", time.time()

    try:
        for part in iterator:
            text = getattr(part, "text", "")
            if not text:
                continue
            buf += text

            # If heading and first bullet are glued, split them
            if "## " in buf and "- " in buf and (
                buf.index("- ") < buf.index("\n") if "\n" in buf else True
            ):
                head, buf = buf.split("- ", 1)
                logger.info(f"CHUNK::{head.strip()}")
                buf = "- " + buf  # keep bullet marker

            # Flush on newline
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.strip():
                    logger.info(f"CHUNK::{line.strip()}")
                    last_emit = time.time()

            # Flush if idle for >2 s
            if time.time() - last_emit > 2:
                _flush()

        _flush()  # leftovers
    finally:
        logger.info("CHUNK::[END]")

# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC: ask_ai
# ─────────────────────────────────────────────────────────────────────────────
def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript")
        logger.error("CHUNK::[END]")
        return
    threading.Thread(
        target=_stream,
        args=(f"Interview Question: {q}",),
        daemon=True,
    ).start()

# ─────────────────────────────────────────────────────────────────────────────
#  CLI LOOP
# ─────────────────────────────────────────────────────────────────────────────
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
