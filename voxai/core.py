#!/usr/bin/env python3
# core.py – Whisper ➜ Gemini Flash 2.5, LIVE bullet streaming
# • heading appears first, each bullet flushes immediately
# • newline injected before every “- ” even when glued to heading text
# Updated 2025-06-28 (“bullet_fix” pattern widened)

import os, sys, logging, threading, time, re, numpy as np, platform
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

# ────────── ENV & GEMINI INIT ────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL", "gemini-2.5-flash-latest")
if not API_KEY:
    logger.error("CHUNK::[ERROR] Missing GENAI_API_KEY")
    logger.error("CHUNK::[END]")
    sys.exit(1)
genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are a **Senior Principal Engineer** providing technical guidance to engineering leadership.

**Response Format:**
- Lead with a concise executive summary
- Follow with structured technical breakdown using bullet points
- Include architectural insights and trade-offs
- Provide implementation considerations
- End with business/strategic implications when relevant

**Technical Depth:**
- Explain at Principal Engineer / VP Engineering level
- Include system design considerations
- Discuss scalability, performance, and operational aspects
- Mention relevant technologies, frameworks, and best practices
- Address both technical and business implications

**Structure Example:**
Brief intro paragraph, then:

## Key Components
• Component 1: Technical details and purpose
• Component 2: How it works and benefits
• Component 3: Integration points and considerations

## Architecture & Design
• System design principles
• Scalability considerations
• Performance characteristics

## Implementation Details
• Technical specifications
• Integration requirements
• Operational considerations

## Business Impact
• Strategic advantages
• Cost implications
• Risk factors

Use this structured approach for all technical explanations.

Current date: {date.today():%Y-%m-%d}
""".strip()

sp = Path("system_prompt.txt")
if sp.exists():
    SYSTEM_PROMPT = sp.read_text(encoding="utf-8").strip()

chat = genai.GenerativeModel(MODEL,
                             system_instruction=SYSTEM_PROMPT).start_chat()
logger.info(f"STATUS:: Chat started with model '{MODEL}'")

# ────────── WHISPER AUDIO SETUP ──────────────────────────────────────────────
whisper = WhisperModel("base", compute_type="int8")
SR, CH, BS = 16000, 1, 4000

def detect_audio_devices():
    """Detect available audio devices - PRIORITIZE SYSTEM AUDIO ONLY"""
    devices = sd.query_devices()
    current_platform = platform.system().lower()
    
    # PRIORITY: System audio loopback devices ONLY
    device_info = {
        'device_id': None,
        'device_name': 'No System Audio Found',
        'source_type': 'system_audio',
        'platform': current_platform
    }
    
    # Search for system audio devices first
    for i, device in enumerate(devices):
        name = device['name'].lower()
        
        # macOS: BlackHole or other loopback devices
        if current_platform == 'darwin':
            if any(term in name for term in ['blackhole', 'loopback', 'soundflower']):
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"STATUS:: Found system audio device: {device['name']}")
                return device_info
        
        # Windows: WASAPI loopback devices
        elif current_platform == 'windows':
            if 'wasapi' in name and 'loopback' in name:
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"STATUS:: Found system audio device: {device['name']}")
                return device_info
        
        # Linux: PulseAudio monitor devices
        elif current_platform == 'linux':
            if 'monitor' in name or 'pulse' in name:
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"STATUS:: Found system audio device: {device['name']}")
                return device_info
    
    # NO FALLBACK TO MICROPHONE - System audio only!
    logger.warning("STATUS:: No system audio device found. Please set up audio loopback:")
    logger.warning("STATUS:: macOS: Install BlackHole and set it as input device")
    logger.warning("STATUS:: Windows: Use WASAPI loopback")
    logger.warning("STATUS:: Linux: Use PulseAudio monitor")
    
    return device_info

# Initialize audio device
AUDIO_DEVICE = detect_audio_devices()
DEV = AUDIO_DEVICE['device_id']

chunks: list[np.ndarray] = []
recording = False
last_txt  = ""

def _audio_cb(indata, *_):
    if recording:
        chunks.append(indata.copy())

def start_listening():
    global recording, chunks
    if recording:
        return
    chunks, recording = [], True
    logger.info("STATUS:: Recording Started")

    def _rec():
        cfg = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None:
            cfg["device"] = DEV
        with sd.InputStream(**cfg):
            while recording:
                sd.sleep(100)

    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe():
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")

    if not chunks:
        last_txt = ""
        logger.info("TRANSCRIBED::")
        return

    samples = np.concatenate(chunks, axis=0)[:, 0].astype(np.float32)
    segs, _  = whisper.transcribe(samples, language="en", beam_size=1)
    last_txt = " ".join(s.text.strip() for s in segs).strip()
    logger.info(f"TRANSCRIBED::{last_txt}")

# ────────── CHATGPT-LIKE STREAMING ──────────────────────────────────────────
def _stream_live(prompt: str):
    global chat
    logger.info("CHUNK::[THINKING]")
    
    try:
        iterator = chat.send_message(
            prompt, stream=True,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40
            }
        )

        accumulated_text = ""
        word_buffer = ""
        
        for part in iterator:
            try:
                chunk = part.text
            except ValueError:
                # Safety filter triggered, try non-streaming
                logger.info("CHUNK::[SAFETY_FILTER]")
                break
                
            if not chunk:
                continue
                
            accumulated_text += chunk
            word_buffer += chunk
            
            # Stream by sentences for better readability
            sentences = re.split(r'([.!?]+\s+)', word_buffer)
            
            if len(sentences) > 1:
                # We have at least one complete sentence
                for i in range(0, len(sentences) - 1, 2):  # Process pairs (sentence + delimiter)
                    if i + 1 < len(sentences):
                        complete_sentence = sentences[i] + sentences[i + 1]
                        if complete_sentence.strip():
                            logger.info(f"CHUNK::{complete_sentence.strip()}")
                
                # Keep the last incomplete sentence in buffer
                word_buffer = sentences[-1] if sentences else ""
            
            # Also stream by meaningful chunks (every ~20 characters)
            elif len(word_buffer) > 20 and (' ' in word_buffer[-10:]):
                # Find last space to avoid breaking words
                last_space = word_buffer.rfind(' ')
                if last_space > 10:
                    chunk_to_send = word_buffer[:last_space + 1]
                    logger.info(f"CHUNK::{chunk_to_send}")
                    word_buffer = word_buffer[last_space + 1:]
        
        # Send remaining buffer
        if word_buffer.strip():
            logger.info(f"CHUNK::{word_buffer.strip()}")
            
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming failed: {e}")
        
        # Reset chat session to recover from broken state
        try:
            logger.info("CHUNK::[RECOVERING]")
            # Try to rewind the broken conversation
            if hasattr(chat, 'rewind'):
                chat.rewind()
            
            # Create fresh chat session as backup
            chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT).start_chat()
            
        except Exception as rewind_error:
            logger.warning(f"CHUNK::[WARNING] Chat rewind failed: {rewind_error}")
            # Create completely new chat session
            chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT).start_chat()
        
        # Fallback to non-streaming with fresh session
        try:
            logger.info("CHUNK::[FALLBACK]")
            response = chat.send_message(
                prompt, stream=False,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048
                }
            )
            
            # Send in paragraphs for better readability
            paragraphs = response.text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    logger.info(f"CHUNK::{paragraph.strip()}")
                    
        except Exception as e2:
            logger.error(f"CHUNK::[ERROR] Fallback failed: {e2}")
            logger.error("CHUNK::[ERROR] Chat session corrupted. Please restart VoxAI.")
    
    logger.info("CHUNK::[END]")

def ask_ai():
    q = last_txt.strip()
    if not q:
        logger.error("CHUNK::[ERROR] No transcript")
        logger.error("CHUNK::[END]")
        return
    
    # Provide better context for the AI
    enhanced_prompt = f"""
Please answer this question directly and thoroughly:

"{q}"

If this is about a specific tool, technology, or concept, explain what it is, how it works, and provide practical examples. Give a complete, informative answer like ChatGPT would.
"""
    
    threading.Thread(target=_stream_live,
                     args=(enhanced_prompt,),
                     daemon=True).start()

# ────────── CLI LOOP ────────────────────────────────────────────────────────
def main_loop():
    logger.info("STATUS:: voxai.core ready")
    # Send audio device info to UI on startup
    logger.info(f"AUDIO_DEVICE::{AUDIO_DEVICE['device_name']}|{AUDIO_DEVICE['source_type']}|{AUDIO_DEVICE['platform']}")
    
    for raw in sys.stdin:
        cmd = raw.strip().upper()
        if cmd == "START":
            start_listening()
        elif cmd == "STOP":
            stop_and_transcribe()
        elif cmd.startswith("ASK"):
            ask_ai()
        elif cmd == "DEVICES":
            # Allow UI to request device info
            logger.info(f"AUDIO_DEVICE::{AUDIO_DEVICE['device_name']}|{AUDIO_DEVICE['source_type']}|{AUDIO_DEVICE['platform']}")

if __name__ == "__main__":
    main_loop()
