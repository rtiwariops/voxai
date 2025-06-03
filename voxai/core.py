import os, sys, threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load config
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
MODEL = os.getenv("GENAI_MODEL")
if not MODEL:
    print("❌ Set GENAI_MODEL in .env", file=sys.stderr); sys.exit(1)
chat = genai.GenerativeModel(MODEL).start_chat()
whisper = WhisperModel("base", compute_type="int8")

# Audio settings
SR = 16000; CH = 1; BS = 4000
DEV = next(i for i,d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())

# State
chunks = []; recording = False; last_txt = ""

def audio_cb(indata, frames, time, status):
    if status: print(f"⚠️ {status}", flush=True)
    if recording: chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording: return
    chunks = []; recording = True
    def rec():
        with sd.InputStream(device=DEV, samplerate=SR, channels=CH, blocksize=BS, callback=audio_cb):
            while recording: sd.sleep(100)
    threading.Thread(target=rec, daemon=True).start()
    print("from-python:STATUS::Recording Started", flush=True)

def stop_and_transcribe():
    global recording, last_txt
    recording = False
    if chunks:
        samples = np.concatenate(chunks, axis=0)[:,0].astype(np.float32)
        segs,_ = whisper.transcribe(samples, language="en", beam_size=1)
        last_txt = " ".join(s.text.strip() for s in segs).strip()
    else:
        last_txt = ""
    print(f"from-python:TRANSCRIBED::{last_txt}", flush=True)
    print("from-python:STATUS::Recording Stopped", flush=True)

def ask_ai():
    if not last_txt:
        print("from-python:CHUNK::[ERROR] No transcript.", flush=True)
        print("from-python:CHUNK::[END]", flush=True)
        return
    for part in chat.send_message(last_txt, stream=True):
        if part.text: print(f"from-python:CHUNK::{part.text}", flush=True)
    print("from-python:CHUNK::[END]", flush=True)

def main_loop():
    print("✅ voxai.core ready", flush=True)
    for line in sys.stdin:
        c = line.strip()
        if c=="START":      start_listening()
        elif c=="STOP":     stop_and_transcribe()
        elif c.startswith("ASK::"): ask_ai()

if __name__=="__main__":
    main_loop()