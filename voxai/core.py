#!/usr/bin/env python3
#  core.py – answers‑only, detailed bullets, **streams word‑by‑word** for Gemini Flash 2.5
#  (updated 2025‑06‑24 23:59 UTC)

import os, sys, threading, logging, requests, numpy as np
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="from-python:%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ── ENV & MODEL ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-1.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.
Return **only**:

## <Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … (detail)
  - Sub‑bullet (≤ 20 words, optional)

No question echo, no paragraphs. Expand on key facets (purpose, durability,
security, pricing, use‑cases, limits). If, even with Context, nothing reliable
exists, reply exactly:
Unknown: no reliable public source found as of {date.today():%Y-%m-%d}.
""".strip()

custom = Path("system_prompt.txt")
if custom.exists(): SYSTEM_PROMPT = custom.read_text(encoding="utf-8").strip()

model = genai.GenerativeModel(model_name=MODEL, system_instruction=SYSTEM_PROMPT)
chat  = model.start_chat()

# ── ASR CONFIG ───────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i,d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
except StopIteration:
    DEV = None

chunks: list[np.ndarray] = []
recording=False; last_txt=""

# ── AUDIO HELPERS ────────────────────────────────────────────────────────────

def _audio_cb(indata, *_):
    if recording: chunks.append(indata.copy())

def start_listening():
    global recording,chunks
    if recording: return
    chunks=[]; recording=True
    def _rec():
        kw=dict(samplerate=SR,channels=CH,blocksize=BS,callback=_audio_cb)
        if DEV is not None: kw["device"]=DEV
        with sd.InputStream(**kw):
            while recording: sd.sleep(100)
    threading.Thread(target=_rec,daemon=True).start()

def stop_and_transcribe():
    global recording,last_txt
    recording=False
    if not chunks: last_txt=""; logger.info("TRANSCRIBED::"); return
    samples=np.concatenate(chunks,axis=0)[:,0].astype(np.float32)
    segs,_=whisper.transcribe(samples,language="en",beam_size=1)
    last_txt=" ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ── FREE SNIPPET HELPERS (unchanged: DuckDuckGo+Wikipedia) ───────────────────

def _duck(q):
    try:
        j=requests.get(f"https://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_redirect=1&no_html=1",timeout=5).json()
        if j.get("AbstractText"): return j["AbstractText"]
        for t in j.get("RelatedTopics",[])[:3]:
            if isinstance(t,dict) and t.get("Text"): return t["Text"]
    except Exception: pass
    return ""

def _wiki_sum(title):
    try:
        j=requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}",timeout=5).json()
        if j.get("extract") and j.get("type")!="disambiguation": return j["extract"]
    except Exception: pass
    return ""

def _wiki_search(q):
    try:
        pg=requests.get(f"https://en.wikipedia.org/w/rest.php/v1/search/title?q={quote_plus(q)}&limit=1",timeout=5).json().get("pages",[])
        if pg: return _wiki_sum(pg[0]["title"])
    except Exception: pass
    return ""

def fetch_snippet(q):
    aliases=[q]
    if "azure" in q.lower() and "foundry" in q.lower() and "ai" not in q.lower():
        aliases+=["Azure AI Foundry","Microsoft Azure AI Foundry"]
    if not q.lower().endswith(" microsoft"): aliases.append(q+" Microsoft")
    for a in aliases:
        for fn in (_duck,_wiki_sum,_wiki_search):
            s=fn(a)
            if s: return s
    return ""

# ── GEMINI STREAM: **word‑by‑word** ───────────────────────────────────────────

def _stream(prompt:str):
    logger.info("CHUNK::[THINKING]")
    try:
        it=chat.send_message(prompt,stream=True,generation_config={"temperature":0.4,"max_output_tokens":512})
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}"); logger.error("CHUNK::[END]"); return
    word=""
    try:
        for part in it:
            txt=getattr(part,"text","")
            for ch in txt:
                if ch.isspace():
                    if word:
                        logger.info(f"CHUNK::{word}")
                        word=""
                    if ch=="\n":
                        logger.info("CHUNK::\n")  # emit newline marker
                else:
                    word+=ch
        if word: logger.info(f"CHUNK::{word}")
    finally:
        logger.info("CHUNK::[END]")

# ── PUBLIC ask_ai ────────────────────────────────────────────────────────────

def ask_ai():
    q=last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript"); logger.error("CHUNK::[END]"); return
    ctx=fetch_snippet(q)
    prompt=(f"Context:\n{ctx}\n\n" if ctx else "")+f"Interview Question: {q}"
    threading.Thread(target=_stream,args=(prompt,),daemon=True).start()

# ── CLI LOOP ─────────────────────────────────────────────────────────────────

def main_loop():
    for line in sys.stdin:
        c=line.strip().upper()
        if c=="START": start_listening()
        elif c=="STOP": stop_and_transcribe()
        elif c.startswith("ASK"): ask_ai()

if __name__=="__main__": main_loop()
