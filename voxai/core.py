#!/usr/bin/env python3
# core.py – Whisper ➜ Gemini Flash 2.5, LIVE bullet streaming
# • heading appears first, each bullet flushes immediately
# • newline injected before every “- ” even when glued to heading text
# Updated 2025-06-28 (“bullet_fix” pattern widened)

import os, sys, logging, threading, time, re, numpy as np, platform
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import google.generativeai as genai

from .config import (
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BLOCK_SIZE, MAX_AUDIO_CHUNKS,
    CHUNK_CLEANUP_THRESHOLD, DEFAULT_GEMINI_MODEL, WHISPER_MODELS,
    TRANSCRIPTION_CONFIG, GENERATION_CONFIG, Messages, UIMessages,
    ErrorMessages, MIN_NODE_MAJOR_VERSION, EnvVars, AudioDeviceKeywords,
    STREAMING_CONFIG, get_system_prompt
)

# ────────── LOGGING ──────────────────────────────────────────────────────────
import logging.config

def setup_logging() -> logging.Logger:
    """Set up structured logging configuration.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'voxai': {
                'format': 'from-python:%(message)s',
            },
            'debug': {
                'format': 'from-python:[%(levelname)s] %(name)s: %(message)s',
            }
        },
        'handlers': {
            'stdout': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'formatter': 'voxai' if os.getenv('VOXAI_DEBUG') != '1' else 'debug',
                'level': 'DEBUG' if os.getenv('VOXAI_DEBUG') == '1' else 'INFO',
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['stdout'],
                'level': 'DEBUG' if os.getenv('VOXAI_DEBUG') == '1' else 'INFO',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(log_config)
    return logging.getLogger(__name__)

logger = setup_logging()

# ────────── ENV & GEMINI INIT ────────────────────────────────────────────────
load_dotenv()

def validate_api_key(api_key: str) -> str:
    """Validate and sanitize the API key."""
    if not api_key:
        raise ValueError(ErrorMessages.MISSING_API_KEY)
    
    # Basic sanitization - remove whitespace and control characters
    sanitized_key = ''.join(char for char in api_key.strip() if ord(char) >= 32)
    
    # Basic validation - check format (should start with specific patterns for Google API keys)
    if not sanitized_key or len(sanitized_key) < 10:
        raise ValueError(ErrorMessages.INVALID_API_KEY)
    
    return sanitized_key

def validate_model_name(model: str) -> str:
    """Validate and sanitize the model name."""
    # Allow only alphanumeric, hyphens, and dots for model names
    import re
    if not re.match(r'^[a-zA-Z0-9\-\.]+$', model):
        raise ValueError(f"{ErrorMessages.INVALID_MODEL_NAME}: {model}")
    return model

try:
    API_KEY = validate_api_key(os.getenv(EnvVars.API_KEY))
    MODEL = validate_model_name(os.getenv(EnvVars.MODEL, DEFAULT_GEMINI_MODEL))
    genai.configure(api_key=API_KEY)
except ValueError as e:
    logger.error(f"{Messages.CHUNK_PREFIX}{Messages.ERROR} {ErrorMessages.CONFIG_ERROR}: {e}")
    logger.error(f"{Messages.CHUNK_PREFIX}{Messages.END}")
    sys.exit(1)
except Exception as e:
    logger.error(f"{Messages.CHUNK_PREFIX}{Messages.ERROR} {ErrorMessages.GEMINI_API_ERROR}: {e}")
    logger.error(f"{Messages.CHUNK_PREFIX}{Messages.END}")
    sys.exit(1)

SYSTEM_PROMPT = get_system_prompt()

chat = genai.GenerativeModel(MODEL,
                             system_instruction=SYSTEM_PROMPT).start_chat()
logger.info(f"STATUS:: Chat started with model '{MODEL}'")

# ────────── WHISPER AUDIO SETUP ──────────────────────────────────────────────
# Use tiny model for faster transcription with GPU acceleration if available
try:
    whisper = WhisperModel("tiny", **WHISPER_MODELS["tiny_cuda"])
    logger.info(f"{Messages.STATUS_PREFIX} Using Whisper tiny model with CUDA acceleration")
except (ImportError, RuntimeError, OSError, ValueError) as e:
    logger.info(f"{Messages.STATUS_PREFIX} CUDA not available ({e}), falling back to CPU")
    try:
        whisper = WhisperModel("tiny", **WHISPER_MODELS["tiny"])
        logger.info(f"{Messages.STATUS_PREFIX} Using Whisper tiny model with CPU (int8)")
    except (ImportError, RuntimeError, OSError, MemoryError) as e:
        logger.info(f"{Messages.STATUS_PREFIX} Tiny model failed ({e}), using base model")
        try:
            whisper = WhisperModel("base", **WHISPER_MODELS["base"])
            logger.info(f"{Messages.STATUS_PREFIX} Fallback to Whisper base model")
        except Exception as e:
            logger.error(f"{Messages.CHUNK_PREFIX}{Messages.ERROR} {ErrorMessages.WHISPER_LOAD_ERROR}: {e}")
            logger.error(f"{Messages.CHUNK_PREFIX}{Messages.END}")
            sys.exit(1)

SR, CH, BS = AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BLOCK_SIZE

def detect_audio_devices() -> Dict[str, Union[int, str, None]]:
    """Detect available audio devices - PRIORITIZE SYSTEM AUDIO ONLY.
    
    Returns:
        Dict[str, Union[int, str, None]]: Dictionary containing device information:
            - device_id: Device ID or None if not found
            - device_name: Human-readable device name
            - source_type: Type of audio source ('system_audio')
            - platform: Operating system platform
    """
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
            if any(term in name for term in AudioDeviceKeywords.MACOS_LOOPBACK):
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"{Messages.STATUS_PREFIX} Found system audio device: {device['name']}")
                return device_info
        
        # Windows: WASAPI loopback devices
        elif current_platform == 'windows':
            if all(keyword in name for keyword in AudioDeviceKeywords.WINDOWS_LOOPBACK):
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"{Messages.STATUS_PREFIX} Found system audio device: {device['name']}")
                return device_info
        
        # Linux: PulseAudio monitor devices
        elif current_platform == 'linux':
            if any(keyword in name for keyword in AudioDeviceKeywords.LINUX_MONITOR):
                device_info.update({
                    'device_id': i,
                    'device_name': device['name'],
                    'source_type': 'system_audio'
                })
                logger.info(f"{Messages.STATUS_PREFIX} Found system audio device: {device['name']}")
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

import gc
from collections import deque

# Audio buffer management with memory limits
MAX_CHUNKS = 1000  # Maximum number of audio chunks to keep in memory
CHUNK_CLEANUP_THRESHOLD = 800  # Start cleanup when we reach this many chunks

chunks: deque[np.ndarray] = deque(maxlen=MAX_CHUNKS)
recording = False
last_txt = ""

def _audio_cb(indata: np.ndarray, *_: Any) -> None:
    """Audio callback with memory management.
    
    Args:
        indata: Input audio data from sound device.
        *_: Additional arguments (ignored).
    """
    if recording:
        # Use deque with maxlen for automatic memory management
        chunks.append(indata.copy())
        
        # Periodic memory cleanup if we're approaching limits
        if len(chunks) >= CHUNK_CLEANUP_THRESHOLD:
            logger.debug(f"Audio buffer cleanup: {len(chunks)} chunks")
            # Force garbage collection to free memory
            gc.collect()

def start_listening() -> None:
    """Start audio recording with proper cleanup.
    
    Initializes audio recording session, clears any existing chunks,
    and starts a background thread for audio capture.
    """
    global recording, chunks
    if recording:
        return
    
    # Clear any existing chunks and reset
    chunks.clear()
    recording = True
    logger.info("STATUS:: Recording Started")
    logger.debug(f"Audio buffer initialized, max chunks: {MAX_CHUNKS}")

    def _rec():
        global recording
        cfg = dict(samplerate=SR, channels=CH, blocksize=BS, callback=_audio_cb)
        if DEV is not None:
            cfg["device"] = DEV
        try:
            with sd.InputStream(**cfg):
                while recording:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            recording = False

    threading.Thread(target=_rec, daemon=True).start()

def stop_and_transcribe() -> None:
    """Stop recording and transcribe with memory cleanup.
    
    Stops the audio recording, processes accumulated audio chunks
    using Whisper for transcription, and cleans up memory resources.
    Sets the global last_txt variable with transcription results.
    """
    global recording, last_txt
    recording = False
    logger.info("STATUS:: Recording Stopped")

    if not chunks:
        last_txt = ""
        logger.info("TRANSCRIBED::")
        return

    try:
        # Convert deque to list for concatenation, then immediately clear chunks
        chunks_list = list(chunks)
        chunks.clear()  # Immediate memory cleanup
        
        logger.debug(f"Processing {len(chunks_list)} audio chunks")
        samples = np.concatenate(chunks_list, axis=0)[:, 0].astype(np.float32)
        
        # Clear the chunks list to free memory before transcription
        del chunks_list
        gc.collect()
        
        # Optimized transcription settings for speed
        segs, _ = whisper.transcribe(
            samples, 
            language="en", 
            beam_size=1,           # Fastest beam search
            temperature=0.0,       # No randomness for speed
            compression_ratio_threshold=2.4,  # Skip low-quality audio
            log_prob_threshold=-1.0,          # Accept more transcriptions
            no_speech_threshold=0.6,          # Skip silence faster
            condition_on_previous_text=False, # Don't wait for context
            word_timestamps=False             # Skip word timing for speed
        )
        
        last_txt = " ".join(s.text.strip() for s in segs).strip()
        logger.info(f"TRANSCRIBED::{last_txt}")
        
        # Clean up samples array
        del samples
        gc.collect()
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        last_txt = ""
        logger.info("TRANSCRIBED::")
        # Ensure chunks are cleared even on error
        chunks.clear()
        gc.collect()

# ────────── CHATGPT-LIKE STREAMING ──────────────────────────────────────────

def _process_chunk_by_lines(word_buffer: str) -> Tuple[str, bool]:
    """Process chunks by complete lines first.
    
    Args:
        word_buffer: Current text buffer containing partial content.
        
    Returns:
        Tuple[str, bool]: Remaining buffer content and whether processing occurred.
    """
    if '\n' in word_buffer:
        lines = word_buffer.split('\n')
        for line in lines[:-1]:  # Send all complete lines
            if line.strip():
                logger.info(f"CHUNK::{line}")
        return lines[-1], True  # Return remaining buffer and processed flag
    return word_buffer, False

def _process_chunk_by_sentences(word_buffer: str) -> Tuple[str, bool]:
    """Process chunks by complete sentences.
    
    Args:
        word_buffer: Current text buffer containing partial content.
        
    Returns:
        Tuple[str, bool]: Remaining buffer content and whether processing occurred.
    """
    if any(punct in word_buffer for punct in ['. ', '! ', '? ']):
        sentences = re.split(r'([.!?]+\s+)', word_buffer)
        if len(sentences) > 1:
            # We have at least one complete sentence
            for i in range(0, len(sentences) - 1, 2):  # Process pairs (sentence + delimiter)
                if i + 1 < len(sentences):
                    complete_sentence = sentences[i] + sentences[i + 1]
                    if complete_sentence.strip():
                        logger.info(f"CHUNK::{complete_sentence}")
            
            # Keep the last incomplete sentence in buffer
            return sentences[-1] if sentences else "", True
    return word_buffer, False

def _process_chunk_by_words(word_buffer: str, min_length: int = 30) -> Tuple[str, bool]:
    """Process chunks by meaningful word boundaries.
    
    Args:
        word_buffer: Current text buffer containing partial content.
        min_length: Minimum buffer length before attempting word-based chunking.
        
    Returns:
        Tuple[str, bool]: Remaining buffer content and whether processing occurred.
    """
    if len(word_buffer) > min_length and (' ' in word_buffer[-15:]):
        # Find last space to avoid breaking words
        last_space = word_buffer.rfind(' ')
        if last_space > 15:
            chunk_to_send = word_buffer[:last_space + 1]
            logger.info(f"CHUNK::{chunk_to_send}")
            return word_buffer[last_space + 1:], True
    return word_buffer, False

def _reset_chat_session() -> None:
    """Reset the chat session with error recovery.
    
    Attempts to rewind the current chat session or creates a new one
    if rewind fails. Handles global chat variable updates.
    """
    global chat
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

def _fallback_non_streaming(prompt: str) -> None:
    """Fallback to non-streaming response when streaming fails.
    
    Args:
        prompt: The user prompt to send to the AI model.
        
    Sends a non-streaming request to the AI model and outputs
    the response in paragraph chunks for better readability.
    """
    global chat
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

def _stream_live(prompt: str) -> None:
    """Stream AI responses with intelligent chunking and error recovery.
    
    Args:
        prompt: The user prompt to send to the AI model.
        
    Streams AI responses using intelligent chunking that prioritizes
    complete lines, then sentences, then word boundaries for optimal
    readability. Includes comprehensive error recovery mechanisms.
    """
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
                
            word_buffer += chunk
            
            # Process chunks in order of preference: lines, sentences, then words
            word_buffer, processed = _process_chunk_by_lines(word_buffer)
            if not processed:
                word_buffer, processed = _process_chunk_by_sentences(word_buffer)
                if not processed:
                    word_buffer, _ = _process_chunk_by_words(word_buffer)
        
        # Send remaining buffer
        if word_buffer.strip():
            logger.info(f"CHUNK::{word_buffer.strip()}")
            
    except Exception as e:
        logger.error(f"CHUNK::[ERROR] Streaming failed: {e}")
        _reset_chat_session()
        _fallback_non_streaming(prompt)
    
    logger.info("CHUNK::[END]")

def ask_ai() -> None:
    """Process the last transcribed text and send it to the AI model.
    
    Takes the globally stored transcription result, enhances it with
    context instructions, and starts a background thread to stream
    the AI response back to the UI.
    """
    q = last_txt.strip()
    if not q:
        logger.error(f"{Messages.CHUNK_PREFIX}{Messages.ERROR} {ErrorMessages.NO_TRANSCRIPT}")
        logger.error(f"{Messages.CHUNK_PREFIX}{Messages.END}")
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
def main_loop() -> None:
    """Main command processing loop for VoxAI backend.
    
    Listens for commands from stdin and dispatches them to appropriate
    handlers. Commands include START, STOP, ASK, and DEVICES.
    Sends audio device information to UI on startup.
    """
    logger.info(f"{Messages.STATUS_PREFIX} voxai.core ready")
    # Send audio device info to UI on startup
    logger.info(f"{Messages.AUDIO_DEVICE_PREFIX}{AUDIO_DEVICE['device_name']}|{AUDIO_DEVICE['source_type']}|{AUDIO_DEVICE['platform']}")
    
    for raw in sys.stdin:
        cmd = raw.strip().upper()
        if cmd == Messages.CMD_START:
            start_listening()
        elif cmd == Messages.CMD_STOP:
            stop_and_transcribe()
        elif cmd.startswith(Messages.CMD_ASK):
            ask_ai()
        elif cmd == Messages.CMD_DEVICES:
            # Allow UI to request device info
            logger.info(f"{Messages.AUDIO_DEVICE_PREFIX}{AUDIO_DEVICE['device_name']}|{AUDIO_DEVICE['source_type']}|{AUDIO_DEVICE['platform']}")

if __name__ == "__main__":
    main_loop()
