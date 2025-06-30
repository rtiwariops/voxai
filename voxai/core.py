#!/usr/bin/env python3
#  core.py – Voice → Whisper → Gemini Flash 2.5
#  • “## Heading + bullets”, streamed line-by-line
#  • auto-retry if truncated or blocked for safety
#  Updated 2025-06-27

import os, sys, threading, logging, time, numpy as np
from datetime import date
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# ─── ENV + GEMINI ───────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)
genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.

Return **only** in this Markdown layout:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub-bullet (optional, ≤ 20 words)

(Newline required after the heading.)

Cover purpose, architecture, durability/availability, security, pricing, limits
and typical use-cases. If you truly don’t know, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

p = Path("system_prompt.txt")
if p.exists():
    SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()

chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT).start_chat()
logger.info(f"STATUS:: Chat started with model '{MODEL}'")

# ─── WHISPER SETUP ──────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i,d in enumerate(sd.query_devices())
               if "blackhole" in d["name"].lower())
except StopIteration:
    DEV = None

chunks: list[np.ndarray] = []; recording=False; last_txt=""

# ─── AUDIO HELPERS ──────────────────────────────────────────────────────────
def _audio_cb(indata, *_):   # gather audio blocks
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording: return
    chunks, recording = [], True
    logger.info("STATUS:: Recording Started")
    def _rec():
        stream_kw = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None: stream_kw["device"] = DEV
        with sd.InputStream(**stream_kw):
            while recording: sd.sleep(100)
    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe():
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")
    if not chunks:
        last_txt = ""; logger.info("TRANSCRIBED::"); return
    samples = np.concatenate(chunks, axis=0)[:,0].astype(np.float32)
    segs,_ = whisper.transcribe(samples, language="en", beam_size=1)
    last_txt = " ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ─── STREAMING WITH TRUNCATION-CHECK ────────────────────────────────────────
# ── STREAM - flush heading + each bullet immediately ─────────────────────────
def _stream_full(prompt: str):
    """
    Streams Gemini line-by-line AND splits glued bullets:

      CHUNK::## Azure Blob Storage
      CHUNK::- **Purpose**: …
      CHUNK::- **Architecture**: …
      …

    If Gemini stops early (safety or truncation) we retry once non-stream.
    """
    logger.info("CHUNK::[THINKING]")
    buf, last_line, truncated, safety_abort = "", "", False, False

    try:
        it = chat.send_message(
            prompt,
            stream=True,
            generation_config={"temperature": 0.4, "max_output_tokens": 2048},
        )

        for part in it:
            try:
                txt = part.text              # raises ValueError on SAFETY
            except ValueError:
                safety_abort = True
                break

            if not txt:
                continue
            buf += txt

            # ①  If a bullet ("- ") is glued to previous text (no \n yet),
            #    split it so the heading flushes immediately:
            while "- " in buf and (buf.find("- ") < buf.find("\n") if "\n" in buf else True):
                before, after = buf.split("- ", 1)
                if before.strip():
                    logger.info(f"CHUNK::{before.strip()}")
                buf = "- " + after  # keep the "-" with the rest

            # ②  Flush each full line
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.strip():
                    last_line = line.strip()
                    logger.info(f"CHUNK::{last_line}")

        # flush leftovers
        if buf.strip():
            last_line = buf.strip()
            logger.info(f"CHUNK::{last_line}")

        # ③  Heuristic: if the last line doesn’t end with punctuation, assume cut
        if last_line and last_line[-1] not in ".!?":
            truncated = True

    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        truncated = True

    # ④  Fallback on safety abort or truncation
    if safety_abort or truncated:
        try:
            full = chat.send_message(prompt, stream=False).text
            logger.info(f"CHUNK::{full.strip()}")
        except Exception as e2:
            logger.error(f"CHUNK::[ERROR] fallback: {e2}")

    logger.info("CHUNK::[END]")


def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript"); logger.error("CHUNK::[END]"); return
    threading.Thread(target=_stream_full,
                     args=(f"Interview Question: {q}",),
                     daemon=True).start()

# ─── CLI LOOP ────────────────────────────────────────────────────────────────
def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for cmd in map(str.strip, sys.stdin):
        c = cmd.upper()
        if c == "START": start_listening()
        elif c == "STOP": stop_and_transcribe()
        elif c.startswith("ASK"): ask_ai()

if __name__ == "__main__":
    main_loop()
