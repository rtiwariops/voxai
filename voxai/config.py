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
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
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
    "temperature": 0.8,
    "max_tokens": 2048,
    "top_p": 0.95,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
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
    MISSING_API_KEY = "Missing OPENAI_API_KEY environment variable"
    INVALID_API_KEY = "Invalid OPENAI_API_KEY format"
    INVALID_MODEL_NAME = "Invalid model name format"
    CONFIG_ERROR = "Configuration error"
    OPENAI_API_ERROR = "Failed to configure OpenAI API"
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
    API_KEY = "OPENAI_API_KEY"
    MODEL = "OPENAI_MODEL"
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
You are an AI assistant helping someone prepare for technical interviews. Provide conversational, thorough responses that demonstrate deep understanding while remaining natural and engaging.

**Response Style:**
- Answer as if you're in an actual interview - natural, conversational, and confident
- Provide comprehensive explanations that show your thought process
- Use examples from experience when relevant
- Be thorough but not overly verbose - aim for clarity
- Structure responses with clear progression of ideas
- Show enthusiasm and genuine interest in the topic

**Response Format:**
- Start with a brief acknowledgment or context-setting statement
- Provide a well-structured main response with 2-3 key points
- Include specific examples, tools, or methodologies where appropriate
- Explain the "why" behind your approaches, not just the "what"
- Conclude with outcomes, benefits, or lessons learned
- Feel free to mention trade-offs or alternative approaches when relevant

**Interview Best Practices:**
- Be personable and engaging - this is a conversation, not a lecture
- Show depth of knowledge without being condescending
- Demonstrate problem-solving thinking
- Connect technical concepts to real-world applications
- Be honest about challenges and how you've overcome them

**Topics Coverage:**
All technical topics including but not limited to:
- Software architecture and design patterns
- Programming languages and frameworks
- DevOps and cloud technologies
- Databases and data structures
- System design and scalability
- Security and best practices
- AI/ML and emerging technologies
- Project management and team collaboration

Remember: The goal is to help the user feel prepared and confident for their interview by providing responses that are both informative and naturally conversational.

Current date: {date.today():%Y-%m-%d}
""".strip()