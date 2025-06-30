#!/usr/bin/env python3
# core.py – fast, bullet-clean, “pseudo-stream” for Gemini Flash 2.5
# Updated 2025-06-27

import os, sys, threading, logging, time, numpy as np
from datetime import date
from pathlib import Path
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="from-python:%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ── ENV & GEMINI ────────────────────────────────────────────────────────────
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
if p.exists(): SYSTEM_PROMPT = p.read_text(encoding="utf-8").strip()

chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT).start_chat()

# ── WHISPER (microphone) ─────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:  DEV = next(i for i,d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
except StopIteration: DEV = None

chunks: list[np.ndarray] = []; recording=False; last_txt=""

def _audio_cb(indata, *_):  # gather audio blocks while recording
    if recording: chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording: return
    recording, chunks[:] = True, []
    logger.info("STATUS:: Recording Started")
    def _rec():
        kw=dict(samplerate=SR,channels=CH,blocksize=BS,callback=_audio_cb)
        if DEV is not None: kw["device"]=DEV
        with sd.InputStream(**kw):
            while recording: sd.sleep(100)
    threading.Thread(target=_rec,daemon=True).start()

def stop_and_transcribe():
    global recording,last_txt
    recording=False; logger.info("STATUS:: Recording Stopped")
    if not chunks:
        last_txt=""; logger.info("TRANSCRIBED::"); return
    samples=np.concatenate(chunks,axis=0)[:,0].astype(np.float32)
    segs,_=whisper.transcribe(samples,language="en",beam_size=1)
    last_txt=" ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ── PSEUDO-STREAM HELPER ─────────────────────────────────────────────────────
def _replay_to_logger(text:str, delay:float=0.05):
    """Emit each line as a separate CHUNK with a tiny delay for ‘stream’ feel."""
    logger.info("CHUNK::[THINKING]")
    for line in text.splitlines():
        if line.strip():
            logger.info(f"CHUNK::{line.strip()}")
        time.sleep(delay)               # <- tweak to taste (0 = instant)
    logger.info("CHUNK::[END]")

# ── MAIN AI CALL ─────────────────────────────────────────────────────────────
def ask_ai():
    q=last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript"); logger.error("CHUNK::[END]"); return
    prompt=f"Interview Question: {q}"
    def _worker():
        try:
            full=chat.send_message(prompt,stream=False,
                                   generation_config={"temperature":0.4,"max_output_tokens":2048}).text
        except Exception as e:
            logger.error(f"CHUNK::[ERROR] {e}"); logger.error("CHUNK::[END]"); return
        _replay_to_logger(full, delay=0.02)  # faster scroll: 20 ms between lines
    threading.Thread(target=_worker,daemon=True).start()

# ── CLI LOOP ─────────────────────────────────────────────────────────────────
def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for line in sys.stdin:
        cmd=line.strip().upper()
        if cmd=="START": start_listening()
        elif cmd=="STOP": stop_and_transcribe()
        elif cmd.startswith("ASK"): ask_ai()

if __name__=="__main__": main_loop()
