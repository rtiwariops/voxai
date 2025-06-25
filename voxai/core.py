#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  core.py – Voice‑driven interview helper (answers‑only, detailed) for Gemini Flash 2.5
# ──────────────────────────────────────────────────────────────────────────────
"""
Changes in this build (2025‑06‑24 23:55 UTC)
─────────────────────────────────────────────
1. **SYSTEM_PROMPT** no longer repeats the question. Only an answer section with
   detailed bullets is returned.
2. Bullets may nest once for extra depth (use two‐space indent).
3. Nothing else in logic changed.
"""

import os, sys, threading, logging, requests, numpy as np
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="from-python:%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  ENV + GEMINI INIT
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-2.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY"); logger.error("CHUNK::[END]"); sys.exit(1)

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are **TechMentor**, a senior technical interviewer.
Always reply in **Markdown** using this structure – *no question echo*:

## <Topic Heading>
- Bullet 1 (≤ 25 words)
- Bullet 2 … *(be as detailed as useful)*
  - Sub‑bullet (optional, ≤ 20 words)

Rules:
- Never include the question text.
- No paragraphs, only bullets.
- Expand on key facets: purpose, architecture, durability, security, pricing, common use‑cases, limitations.
- If—even with Context—no authoritative public info exists, reply exactly:
  "Unknown: no reliable public source found as of {date.today():%Y-%m-%d}."
""".strip()

custom = Path("system_prompt.txt")
if custom.exists():
    SYSTEM_PROMPT = custom.read_text(encoding="utf-8").strip()

try:
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=SYSTEM_PROMPT)
    chat  = model.start_chat()
except Exception as e:
    logger.error(f"CHUNK::[ERROR] Could not start chat: {e}"); logger.error("CHUNK::[END]"); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  WHISPER CONFIG (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000
try:
    DEV = next(i for i, d in enumerate(sd.query_devices()) if "blackhole" in d["name"].lower())
except StopIteration:
    DEV = None

# state vars\ nchunks: list[np.ndarray] = []
recording, last_txt = False, ""

# audio helpers
def _audio_cb(indata, *_):
    if recording: chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording: return
    chunks, recording = [], True
    def _rec():
        kw = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None: kw["device"] = DEV
        with sd.InputStream(**kw):
            while recording: sd.sleep(100)
    threading.Thread(target=_rec, daemon=True).start()


def stop_and_transcribe():
    global recording, last_txt
    recording = False
    if not chunks:
        last_txt = ""; logger.info("TRANSCRIBED::"); return
    samples = np.concatenate(chunks, axis=0)[:,0].astype(np.float32)
    segs,_ = whisper.transcribe(samples, language="en", beam_size=1)
    last_txt = " ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# free snippet funcs (same as before) --------------------------------------------------

def _duck(q):
    try:
        url=f"https://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_redirect=1&no_html=1"
        data=requests.get(url,timeout=5).json(); ab=data.get("AbstractText")
        if ab: return ab
        for t in data.get("RelatedTopics",[])[:3]:
            if isinstance(t,dict) and t.get("Text"): return t["Text"]
    except Exception: pass
    return ""

def _wiki_summary(title):
    try:
        j=requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}",timeout=5).json()
        if j.get("extract") and j.get("type")!="disambiguation": return j["extract"]
    except Exception: pass
    return ""

def _wiki_search(q):
    try:
        pg=requests.get(f"https://en.wikipedia.org/w/rest.php/v1/search/title?q={quote_plus(q)}&limit=1",timeout=5).json().get("pages",[])
        if pg: return _wiki_summary(pg[0]["title"])
    except Exception: pass
    return ""

def fetch_snippet(q):
    aliases=[q]
    if "azure" in q.lower() and "foundry" in q.lower() and "ai" not in q.lower():
        aliases+=["Azure AI Foundry","Microsoft Azure AI Foundry"]
    if not q.lower().endswith(" microsoft"): aliases.append(q+" Microsoft")
    for a in aliases:
        for fn in (_duck,_wiki_summary,_wiki_search):
            s=fn(a)
            if s: return s
    return ""

# gemini stream helper ---------------------------------------------------------

def _stream(prompt):
    logger.info("CHUNK::[THINKING]")
    try:
        it=chat.send_message(prompt,stream=True,generation_config={"temperature":0.4,"max_output_tokens":512})
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] {e}"); logger.error("CHUNK::[END]"); return
    buf=""
    for p in it:
        t=getattr(p,"text","")
        if not t: continue
        buf+=t
        while "\n" in buf:
            line,buf=buf.split("\n",1)
            if line: logger.info(f"CHUNK::{line}")
    if buf.strip(): logger.info(f"CHUNK::{buf.strip()}")
    logger.info("CHUNK::[END]")

# public ask_ai ---------------------------------------------------------------

def ask_ai():
    q=last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript"); logger.error("CHUNK::[END]"); return
    ctx=fetch_snippet(q)
    prompt=(f"Context:\n{ctx}\n\n" if ctx else "")+f"Interview Question: {q}"
    threading.Thread(target=_stream,args=(prompt,),daemon=True).start()

# cmd loop --------------------------------------------------------------------

def main_loop():
    for line in sys.stdin:
        cmd=line.strip().upper()
        if cmd=="START": start_listening()
        elif cmd=="STOP": stop_and_transcribe()
        elif cmd.startswith("ASK"): ask_ai()

if __name__=="__main__": main_loop()
