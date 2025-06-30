#!/usr/bin/env python3
# core.py – Whisper ➜ Gemini Flash 2.5, live bullet streaming
# Updated 2025-06-28

import os, sys, logging, threading, time, re, numpy as np
from datetime import date
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ────────── LOGGING ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ────────── ENV & GEMINI INIT ───────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)
genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.

Return **only** in this layout:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub-bullet (optional)

Cover purpose, architecture, durability, security, pricing, limits, use-cases.  
If unknown, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

p = Path("system_prompt.txt")
if p.exists():
    SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()

chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT).start_chat()
logger.info(f"STATUS:: Chat started with model '{MODEL}'")

# ────────── WHISPER AUDIO SETUP ─────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i,d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
except StopIteration:
    DEV = None

chunks: list[np.ndarray] = []
recording = False
last_txt  = ""

def _audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording: return
    chunks, recording = [], True
    logger.info("STATUS:: Recording Started")
    def _rec():
        cfg = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None: cfg["device"] = DEV
        with sd.InputStream(**cfg):
            while recording: sd.sleep(100)
    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe():
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")
    if not chunks:
        last_txt = ""; logger.info("TRANSCRIBED::"); return
    samples = np.concatenate(chunks, axis=0)[:,0].astype(np.float32)
    segs,_  = whisper.transcribe(samples, language="en", beam_size=1)
    last_txt = " ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ────────── STREAMING with BULLET-SPLIT ─────────────────────────────────────
bullet_fix   = re.compile(r'(?<!\n)( {0,2}- )')   # "- " or "  - " not preceded by \n

def _stream_live(prompt: str):
    """
    • True streaming from Gemini Flash.
    • Inserts a newline before every bullet marker the instant it appears,
      so each bullet is flushed separately.
    • If stream ends without any bullet (safety/truncation), retries once
      with non-stream call and keeps streaming that result.
    """
    logger.info("CHUNK::[THINKING]")
    buf, got_bullet, need_retry = "", False, False

    try:
        it = chat.send_message(prompt, stream=True,
                               generation_config={"temperature":0.4,
                                                  "max_output_tokens":2048})
        for part in it:
            try:
                txt = part.text                       # may raise ValueError on safety
            except ValueError:                        # safety abort
                need_retry = True
                break

            if not txt: continue
            # → instant newline before "- " or "  - "
            txt = bullet_fix.sub(r'\n\1', txt)
            buf += txt

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.strip():
                    logger.info(f"CHUNK::{line.strip()}")
                    if line.lstrip().startswith("-"): got_bullet = True

        if buf.strip():
            logger.info(f"CHUNK::{buf.strip()}")
            if buf.lstrip().startswith("-"): got_bullet = True
        if not got_bullet: need_retry = True

    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}")
        need_retry = True

    if need_retry:
        try:
            full = chat.send_message(prompt, stream=False,
                                     generation_config={"temperature":0.4,
                                                        "max_output_tokens":2048}).text
            for line in bullet_fix.sub(r'\n\1', full).splitlines():
                if line.strip():
                    logger.info(f"CHUNK::{line.strip()}")
        except Exception as e2:
            logger.error(f"CHUNK::[ERROR] fallback: {e2}")

    logger.info("CHUNK::[END]")

# ────────── ASK ENTRY ───────────────────────────────────────────────────────
def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript"); logger.error("CHUNK::[END]"); return
    threading.Thread(target=_stream_live,
                     args=(f"Interview Question: {q}",), daemon=True).start()

# ────────── CLI LOOP ────────────────────────────────────────────────────────
def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for raw in sys.stdin:
        cmd = raw.strip().upper()
        if cmd == "START": start_listening()
        elif cmd == "STOP":  stop_and_transcribe()
        elif cmd.startswith("ASK"): ask_ai()

if __name__ == "__main__":
    main_loop()
