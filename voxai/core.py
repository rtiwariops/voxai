import os
import sys
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerationConfig

# ─── Load configuration ─────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
MODEL = os.getenv("GENAI_MODEL")
if not MODEL:
    print("❌ Set GENAI_MODEL in .env", file=sys.stderr)
    sys.exit(1)
chat = genai.GenerativeModel(MODEL).start_chat()

# ─── Whisper setup ─────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")

# ─── Audio settings ────────────────────────────────────────────────────────────
SR = 16000
CH = 1
BS = 4000
DEV = next(
    i for i, d in enumerate(sd.query_devices())
    if "blackhole" in d["name"].lower() and d["max_input_channels"] > 0
)

# ─── Recording state ────────────────────────────────────────────────────────────
chunks = []
recording = False
last_txt = ""

def audio_cb(indata, frames, time, status):
    if status:
        print(f"⚠️ Audio status: {status}", flush=True)
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    chunks = []
    recording = True

    def _record():
        with sd.InputStream(
            device=DEV, samplerate=SR, channels=CH, blocksize=BS, callback=audio_cb
        ):
            while recording:
                sd.sleep(100)

    threading.Thread(target=_record, daemon=True).start()
    print("from-python:STATUS::Recording Started", flush=True)

def stop_and_transcribe():
    global recording, last_txt
    recording = False

    if chunks:
        samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
        segments, _ = whisper.transcribe(
            samples, language="en", beam_size=1
        )
        last_txt = " ".join(s.text.strip() for s in segments).strip()
    else:
        last_txt = ""

    print(f"from-python:TRANSCRIBED::{last_txt}", flush=True)
    print("from-python:STATUS::Recording Stopped", flush=True)

def ask_ai():
    """Stream the last transcript into Gemini with a conversational prompt."""
    global last_txt
    if not last_txt:
        print("from-python:CHUNK::[ERROR] No transcript.", flush=True)
        print("from-python:CHUNK::[END]", flush=True)
        return

    # Updated system prompt to include VP of Engineering as well
    system_prompt = (
        "You are a Principal/Chief Engineer or VP of Engineering in a technical or leadership interview. "
        "Answer in a single concise paragraph, using precise service names and technical terms. "
        "Explain what the service is and how it fits into an enterprise data pipeline—no bullet points.\n\n"
    )
    full_prompt = system_prompt + last_txt
    gen_config = GenerationConfig(temperature=0.2, max_output_tokens=256)

    for part in chat.send_message(full_prompt, stream=True):
        if part.text:
            print(f"from-python:CHUNK::{part.text}", flush=True)

    print("from-python:CHUNK::[END]", flush=True)

def main_loop():
    print("✅ voxai.core ready", flush=True)
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.startswith("ASK::"):
            ask_ai()

if __name__ == "__main__":
    main_loop()
