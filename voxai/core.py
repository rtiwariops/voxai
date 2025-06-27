#!/usr/bin/env python3
#  core.py – Voice-to-Gemini Flash 2.5, bullets only, word-by-word streaming
#  Updated 2025-06-27

import os, sys, threading, logging, numpy as np
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
MODEL   = os.getenv("GENAI_MODEL", "gemini-2.5-flash-latest")  # Flash 2.5

if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.

Return **only**:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub-bullet (≤ 20 words, optional)

No question echo, no paragraphs. Cover purpose, architecture, durability/
availability, security, pricing, limits, and typical use-cases in clear,
detailed bullets.

If you truly do not know the answer, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

# Allow overriding via local file
p = Path("system_prompt.txt")
if p.exists():
    SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()
    logger.info(f"STATUS:: Loaded custom prompt from {p}")

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
#  GEMINI STREAM – word-by-word
# ─────────────────────────────────────────────────────────────────────────────
def _stream(prompt: str):
    """
    Streams Gemini output **line-by-line** (fast and readable):

    CHUNK::[THINKING]
    CHUNK::## Blob Storage
    CHUNK::- Blob storage is a highly scalable...
    CHUNK::- It stores unstructured data such as...
    CHUNK::[END]

    – fewer log calls than word-by-word → faster output
    – each bullet arrives as a clean line ready for the UI
    """
    logger.info("CHUNK::[THINKING]")

    try:
        it = chat.send_message(
            prompt,
            stream=True,
            generation_config={"temperature": 0.4, "max_output_tokens": 640},
        )
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        logger.error("CHUNK::[END]")
        return

    buf = ""
    try:
        for part in it:
            txt = getattr(part, "text", "")
            if not txt:
                continue
            buf += txt
            while "\n" in buf:                # flush each complete line
                line, buf = buf.split("\n", 1)
                if line.strip():              # skip bare empty lines
                    logger.info(f"CHUNK::{line.strip()}")
        if buf.strip():                       # leftovers when stream ends
            logger.info(f"CHUNK::{buf.strip()}")
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

    prompt = f"Interview Question: {q}"
    threading.Thread(target=_stream, args=(prompt,), daemon=True).start()

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
