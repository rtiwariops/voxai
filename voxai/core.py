# core.py

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

# Configure logging to prefix every message with "from-python:" and send to stdout
logging.basicConfig(
    level=logging.INFO,
    format="from-python:%(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info(f"STATUS:: Running core.py from {__file__}")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL")

if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)

if not MODEL:
    logger.error("CHUNK::[ERROR] Missing GENAI_MODEL")
    logger.error("CHUNK::[END]")
    sys.exit(1)

# Initialize Generative AI client
genai.configure(api_key=API_KEY)
try:
    chat = genai.GenerativeModel(MODEL).start_chat()
    logger.info(f"STATUS:: Started chat with model '{MODEL}'")
except Exception as e:
    logger.error(f"CHUNK::[ERROR] Could not start chat: {e}")
    logger.error("CHUNK::[END]")
    sys.exit(1)

# Whisper ASR initialization
whisper = WhisperModel("base", compute_type="int8")
SR = 16000  # sample rate
CH = 1      # channels
BS = 4000   # blocksize

# Choose an input device containing "blackhole"; otherwise default
try:
    DEV = next(
        i for i, d in enumerate(sd.query_devices())
        if "blackhole" in d["name"].lower()
    )
    logger.info(f"STATUS:: Using device index {DEV} for audio capture")
except StopIteration:
    DEV = None
    logger.info("STATUS:: No 'blackhole' device found; using default input")

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD SYSTEM PROMPT (CWD → fallback)
# ──────────────────────────────────────────────────────────────────────────────

cwd_prompt = Path(os.getcwd()) / "system_prompt.txt"
if cwd_prompt.exists():
    try:
        with open(cwd_prompt, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
            logger.info(f"STATUS:: Loaded system prompt from CWD: {cwd_prompt}")
    except Exception as e:
        SYSTEM_PROMPT = ""
        logger.info(f"STATUS:: Failed to read {cwd_prompt}: {e}")
else:
    SYSTEM_PROMPT = (
        "You are a highly seasoned technology leader and expert in tech. Give the answer in simple and concise manner. "
    )
    logger.info("STATUS:: Using built-in fallback prompt")

# ──────────────────────────────────────────────────────────────────────────────
#  STATE VARIABLES
# ──────────────────────────────────────────────────────────────────────────────

chunks = []      # audio buffers
recording = False
last_txt = ""    # holds the most recent transcription

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO CALLBACK & RECORD/STOP
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

    def rec_thread():
        if DEV is not None:
            stream = sd.InputStream(
                device=DEV,
                samplerate=SR,
                channels=CH,
                blocksize=BS,
                callback=audio_cb
            )
        else:
            stream = sd.InputStream(
                samplerate=SR,
                channels=CH,
                blocksize=BS,
                callback=audio_cb
            )
        with stream:
            while recording:
                sd.sleep(100)

    threading.Thread(target=rec_thread, daemon=True).start()

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
#  ASK AI (combine SYSTEM_PROMPT + last_txt)
# ──────────────────────────────────────────────────────────────────────────────

def ask_ai():
    global last_txt

    logger.info("STATUS:: ENTERED ask_ai()")

    if not last_txt.strip():
        logger.error("CHUNK::[ERROR] No transcript to send to AI.")
        logger.error("CHUNK::[END]")
        return

    combined_prompt = SYSTEM_PROMPT + "\n\nUser Transcript: " + last_txt
    logger.info(
        f"STATUS:: ask_ai() invoked; combined_prompt starts with: "
        f"'{combined_prompt[:60]}...'"
    )

    try:
        iterator = chat.send_message(combined_prompt, stream=True)
    except TypeError as e:
        logger.info(f"STATUS:: Streaming not supported ({e}); falling back to non-streaming")
        iterator = None
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] AI call failed at start: {e}")
        logger.error("CHUNK::[END]")
        return

    if iterator is not None:
        any_chunk = False
        try:
            for part in iterator:
                text = getattr(part, "text", None) or ""
                if text:
                    any_chunk = True
                    logger.info(f"CHUNK::{text}")
        except Exception as e:
            logger.error(f"CHUNK::[ERROR] Streaming exception: {e}")
        finally:
            if not any_chunk:
                logger.info("CHUNK::[WARN] Streaming returned zero chunks.")
            logger.info("CHUNK::[END]")
    else:
        try:
            resp = chat.send_message(combined_prompt)  # non-stream
            text = getattr(resp, "text", "")
            if text:
                logger.info(f"CHUNK::{text}")
            else:
                logger.info("CHUNK::[WARN] Non-streaming returned empty text.")
        except Exception as e:
            logger.error(f"CHUNK::[ERROR] Non-streaming call failed: {e}")
        finally:
            logger.info("CHUNK::[END]")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP WITH RAW LINE LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def main_loop():
    logger.info("STATUS:: voxai.core ready")
    for line in sys.stdin:
        raw = line.rstrip("\n")
        logger.info(f"STATUS:: main_loop got raw line → '{raw}'")
        cmd = raw.strip()

        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.upper().startswith("ASK"):
            logger.info(f"STATUS:: main_loop recognized ASK command: '{cmd}'")
            ask_ai()
        else:
            logger.info(f"STATUS:: Unknown command → '{cmd}'")

if __name__ == "__main__":
    main_loop()
