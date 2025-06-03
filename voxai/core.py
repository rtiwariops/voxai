# core.py

import os
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & STARTUP
# ──────────────────────────────────────────────────────────────────────────────

# Print the path to confirm we’re running the intended file
print(f"from-python:STATUS:: Running core.py from {__file__}", flush=True)

# Load environment variables (for GENAI_API_KEY / GENAI_MODEL)
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL")

if not API_KEY:
    print("from-python:CHUNK::[ERROR] Missing GENAI_API_KEY", flush=True)
    print("from-python:CHUNK::[END]", flush=True)
    sys.exit(1)

if not MODEL:
    print("from-python:CHUNK::[ERROR] Missing GENAI_MODEL", flush=True)
    print("from-python:CHUNK::[END]", flush=True)
    sys.exit(1)

# Initialize the Google Generative AI client
genai.configure(api_key=API_KEY)

try:
    chat = genai.GenerativeModel(MODEL).start_chat()
    print(f"from-python:STATUS:: Started chat with model '{MODEL}'", flush=True)
except Exception as e:
    print(f"from-python:CHUNK::[ERROR] Could not start chat: {e}", flush=True)
    print("from-python:CHUNK::[END]", flush=True)
    sys.exit(1)

# Whisper ASR initialization
whisper = WhisperModel("base", compute_type="int8")
SR = 16000  # sample rate
CH = 1      # channels
BS = 4000   # blocksize

# Try to pick an input device containing “blackhole”; otherwise fallback to default
try:
    DEV = next(i for i, d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
    print(f"from-python:STATUS:: Using device index {DEV} for audio capture", flush=True)
except StopIteration:
    DEV = None
    print("from-python:STATUS:: No 'blackhole' device found; using default input", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD SYSTEM PROMPT (CWD → fallback)
# ──────────────────────────────────────────────────────────────────────────────

# 1. First check for 'system_prompt.txt' in the CURRENT WORKING DIRECTORY
cwd_prompt = Path(os.getcwd()) / "system_prompt.txt"

if cwd_prompt.exists():
    try:
        with open(cwd_prompt, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
            print(f"from-python:STATUS:: Loaded system prompt from CWD: {cwd_prompt}", flush=True)
    except Exception as e:
        SYSTEM_PROMPT = ""
        print(f"from-python:STATUS:: Failed to read {cwd_prompt}: {e}", flush=True)
else:
    # 2. No local file → use a built-in fallback
    SYSTEM_PROMPT = (
        "You are an experienced Principal/Chief Engineer or VP, leading technology teams across multi-cloud environments (AWS, Azure, GCP). "
        "Your responsibilities include architecting scalable, reliable systems using container orchestration, serverless functions, "
        "big data processing, ML/AI platforms, observability tools, and Infrastructure-as-Code. You guide engineers, foster collaborative culture, "
        "drive strategic decisions, and advocate best practices in security, performance, and operational excellence. "
        "When responding, maintain a professional, technical tone, offer clear guidance on architecture and leadership practices, "
        "and demonstrate broad knowledge of cloud services, DevOps methodologies, and effective team behaviors."
    )
    print("from-python:STATUS:: Using built-in fallback prompt", flush=True)

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
        print(f"from-python:STATUS:: Audio warning: {status}", flush=True)
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    chunks = []
    recording = True
    print("from-python:STATUS:: Recording Started", flush=True)

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
    print("from-python:STATUS:: Recording Stopped", flush=True)

    if not chunks:
        last_txt = ""
        print("from-python:TRANSCRIBED::", flush=True)
        return

    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    try:
        segs, _ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        last_txt = ""
        print(f"from-python:CHUNK::[ERROR] Whisper transcription failed: {e}", flush=True)

    print(f"from-python:TRANSCRIBED::{last_txt}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
#  ASK AI (combine SYSTEM_PROMPT + last_txt)
# ──────────────────────────────────────────────────────────────────────────────

def ask_ai():
    global last_txt

    # 1) Proof-of-life: show we entered ask_ai()
    print("from-python:STATUS:: ENTERED ask_ai()", flush=True)

    if not last_txt.strip():
        # If no transcript, notify and exit
        print("from-python:CHUNK::[ERROR] No transcript to send to AI.", flush=True)
        print("from-python:CHUNK::[END]", flush=True)
        return

    # 2) Build the combined prompt
    combined_prompt = SYSTEM_PROMPT + "\n\nUser Transcript: " + last_txt
    print(f"from-python:STATUS:: ask_ai() invoked; combined_prompt starts with: '{combined_prompt[:60]}...'", flush=True)

    # 3) Attempt streaming call
    try:
        iterator = chat.send_message(combined_prompt, stream=True)
    except TypeError as e:
        print(f"from-python:STATUS:: Streaming not supported ({e}); falling back to non-streaming", flush=True)
        iterator = None
    except Exception as e:
        print(f"from-python:CHUNK::[ERROR] AI call failed at start: {e}", flush=True)
        print("from-python:CHUNK::[END]", flush=True)
        return

    # 4) If streaming is available, use it
    if iterator is not None:
        any_chunk = False
        try:
            for part in iterator:
                text = getattr(part, "text", None) or ""
                if text:
                    any_chunk = True
                    print(f"from-python:CHUNK::{text}", flush=True)
        except Exception as e:
            print(f"from-python:CHUNK::[ERROR] Streaming exception: {e}", flush=True)
        finally:
            if not any_chunk:
                print("from-python:CHUNK::[WARN] Streaming returned zero chunks.", flush=True)
            print("from-python:CHUNK::[END]", flush=True)
    else:
        # 5) Fallback to non-streaming call
        try:
            resp = chat.send_message(combined_prompt)  # non-stream
            text = getattr(resp, "text", "")
            if text:
                print(f"from-python:CHUNK::{text}", flush=True)
            else:
                print("from-python:CHUNK::[WARN] Non-streaming returned empty text.", flush=True)
        except Exception as e:
            print(f"from-python:CHUNK::[ERROR] Non-streaming call failed: {e}", flush=True)
        finally:
            print("from-python:CHUNK::[END]", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP WITH RAW LINE LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def main_loop():
    print("from-python:STATUS:: voxai.core ready", flush=True)
    for line in sys.stdin:
        raw = line.rstrip("\n")
        print(f"from-python:STATUS:: main_loop got raw line → '{raw}'", flush=True)
        cmd = raw.strip()

        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.upper().startswith("ASK"):
            print(f"from-python:STATUS:: main_loop recognized ASK command: '{cmd}'", flush=True)
            ask_ai()
        else:
            print(f"from-python:STATUS:: Unknown command → '{cmd}'", flush=True)

if __name__ == "__main__":
    main_loop()
