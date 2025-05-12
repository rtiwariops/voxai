import os
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
MODEL_NAME = os.getenv("GENAI_MODEL")
if MODEL_NAME is None:
    raise RuntimeError("Please set GENAI_MODEL in your .env")
chat = genai.GenerativeModel(MODEL_NAME).start_chat()
model = WhisperModel("base", compute_type="int8")

# Audio configuration
SAMPLERATE   = 16000
CHANNELS     = 1
BLOCKSIZE    = 4000  # ~0.25s
DEVICE_INDEX = next(
    i for i,d in enumerate(sd.query_devices())
    if "blackhole" in d["name"].lower() and d["max_input_channels"]>0
)

# Recording state
audio_chunks = []
recording = False
last_transcript = ""

def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ Audio status: {status}", flush=True)
    if recording:
        audio_chunks.append(indata.copy())

def start_recording():
    global recording, audio_chunks
    if recording: return
    audio_chunks = []
    recording = True

    def _recorder():
        with sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            blocksize=BLOCKSIZE,
            callback=audio_callback
        ):
            print("🎙️ Recording started.", flush=True)
            while recording:
                sd.sleep(100)

    threading.Thread(target=_recorder, daemon=True).start()
    print("from-python:STATUS::Recording Started", flush=True)

def stop_recording_and_transcribe():
    global recording, last_transcript
    if not recording: return
    recording = False

    if audio_chunks:
        samples = np.concatenate(audio_chunks, axis=0)[:,0].astype(np.float32)
        segments, _ = model.transcribe(samples, language="en", beam_size=1)
        last_transcript = " ".join(s.text.strip() for s in segments).strip()
    else:
        last_transcript = ""

    print(f"from-python:TRANSCRIBED::{last_transcript}", flush=True)
    print("from-python:STATUS::Recording Stopped", flush=True)

def ask_ai():
    global last_transcript
    prompt = last_transcript.strip()
    if not prompt:
        print("from-python:CHUNK::[ERROR] No transcript.", flush=True)
        print("from-python:CHUNK::[END]", flush=True)
        return

    for part in chat.send_message(prompt, stream=True):
        if part.text:
            print(f"from-python:CHUNK::{part.text}", flush=True)
    print("from-python:CHUNK::[END]", flush=True)

if __name__ == "__main__":
    # Print ready so Electron knows we're up
    print("✅ backend.core ready", flush=True)

    # Command loop: listen for START / STOP / ASK:: from Electron
    import sys
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "START":
            start_recording()
        elif cmd == "STOP":
            stop_recording_and_transcribe()
        elif cmd.startswith("ASK::"):
            ask_ai()

