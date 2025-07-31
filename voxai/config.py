"""Configuration constants for VoxAI."""

import os
from typing import Dict, Any

# Audio configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK_SIZE = 4000
MAX_AUDIO_CHUNKS = 1000
CHUNK_CLEANUP_THRESHOLD = 800

# Model configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-latest"
WHISPER_MODELS = {
    "tiny_cuda": {"compute_type": "float16", "device": "cuda"},
    "tiny": {"compute_type": "int8", "device": "cpu"},
    "base": {"compute_type": "int8", "device": "cpu"}
}

# Transcription settings
TRANSCRIPTION_CONFIG = {
    "language": "en",
    "beam_size": 1,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "word_timestamps": False
}

# AI generation settings
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40
}

# Protocol message constants
class Messages:
    # Status messages
    STATUS_PREFIX = "STATUS::"
    AUDIO_DEVICE_PREFIX = "AUDIO_DEVICE::"
    TRANSCRIBED_PREFIX = "TRANSCRIBED::"
    CHUNK_PREFIX = "CHUNK::"
    
    # Chunk status indicators
    THINKING = "[THINKING]"
    END = "[END]"
    ERROR = "[ERROR]"
    SAFETY_FILTER = "[SAFETY_FILTER]"
    FALLBACK = "[FALLBACK]"
    RECOVERING = "[RECOVERING]"
    WARNING = "[WARNING]"
    
    # User commands
    CMD_START = "START"
    CMD_STOP = "STOP"
    CMD_ASK = "ASK::"
    CMD_DEVICES = "DEVICES"

# UI messages
class UIMessages:
    RECORDING_STARTED = "Recording Started"
    RECORDING_STOPPED = "Recording Stopped"
    NO_SPEECH_DETECTED = "No speech detected. Try again."
    READY_TO_TRANSCRIBE = "Ready to transcribe your voice..."
    WAITING_FOR_QUESTION = "Waiting for your question..."
    RECORDING_SPEAK_NOW = "Recording... Speak now!"
    PROCESSING_RECORDING = "Processing recording..."

# Error messages  
class ErrorMessages:
    MISSING_API_KEY = "Missing GENAI_API_KEY environment variable"
    INVALID_API_KEY = "Invalid GENAI_API_KEY format"
    INVALID_MODEL_NAME = "Invalid model name format"
    CONFIG_ERROR = "Configuration error"
    GEMINI_API_ERROR = "Failed to configure Gemini API"
    WHISPER_LOAD_ERROR = "Failed to load any Whisper model"
    AUDIO_RECORDING_ERROR = "Audio recording error"
    TRANSCRIPTION_ERROR = "Transcription error"
    STREAMING_FAILED = "Streaming failed"
    CHAT_REWIND_FAILED = "Chat rewind failed"
    FALLBACK_FAILED = "Fallback failed"
    CHAT_CORRUPTED = "Chat session corrupted. Please restart VoxAI."
    NO_TRANSCRIPT = "No transcript"

# Node.js configuration
MIN_NODE_MAJOR_VERSION = 14

# Environment variable names
class EnvVars:
    API_KEY = "GENAI_API_KEY"
    MODEL = "GENAI_MODEL"
    DEBUG = "VOXAI_DEBUG"
    PYTHON_EXE = "VOXAI_PYTHON_EXE"

# Audio device detection keywords
class AudioDeviceKeywords:
    MACOS_LOOPBACK = ['blackhole', 'loopback', 'soundflower']
    WINDOWS_LOOPBACK = ['wasapi', 'loopback']
    LINUX_MONITOR = ['monitor', 'pulse']

# Streaming configuration
STREAMING_CONFIG = {
    "word_chunk_min_length": 30,
    "word_break_lookahead": 15,
    "word_break_min_pos": 15
}

def get_system_prompt() -> str:
    """Get the system prompt from file or return default."""
    from pathlib import Path
    from datetime import date
    
    custom_prompt_file = Path("system_prompt.txt")
    if custom_prompt_file.exists():
        return custom_prompt_file.read_text(encoding="utf-8").strip()
    
    return f"""
You are a **Senior Principal Engineer** providing concise technical interview responses for ANY technology topic.

**Response Format:**
- Keep responses to 1-2 short paragraphs maximum (3-4 sentences total)
- Start with a direct, confident statement about your approach
- Include 2-3 specific technical tools/practices/methodologies you use
- End with a brief statement about the outcome or benefit
- Write in first person as if answering an interview question
- MANDATORY: Add brief definitions in brackets for ALL technical terms (2-4 words max)

**Style:**
- Concise and to the point
- Professional but conversational
- Include specific technical terms with short definitions
- Demonstrate practical experience across all technologies
- No lengthy explanations or theory

**Definition Requirements:**
- EVERY technical term, tool, framework, methodology MUST have a definition
- Keep definitions to 2-4 words maximum
- Focus on the core function or purpose
- Use simple, clear language
- Examples: 
  - Docker (container platform)
  - React (UI library)
  - Scrum (agile framework)
  - TDD (test-first development)
  - Redis (in-memory cache)
  - Microservices (distributed architecture)

**Topics Coverage:**
- Software development, DevOps, cloud, databases, frontend, backend, mobile, AI/ML, security, etc.

Current date: {date.today():%Y-%m-%d}
""".strip()