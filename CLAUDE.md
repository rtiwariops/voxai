# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Development
```bash
# Install package in development mode
pip install -e .

# Install from PyPI
pip install voxai

# Run the application
voxai
```

### Electron UI Development
```bash
# The Electron UI is auto-bootstrapped on first run
# Manual setup (if needed):
cd voxai/electron
npm install
npm start
```

### Testing & Building
```bash
# Build Python package
python setup.py sdist bdist_wheel

# Install dependencies for development
pip install -e .
```

## Architecture Overview

VoxAI is a voice-driven AI assistant with a hybrid Python backend + Electron frontend architecture:

### Core Components

1. **CLI Entry Point** (`cli.py`):
   - Handles Node.js version checking (≥14 required)
   - Auto-bootstraps Electron UI dependencies on first run
   - Launches Python backend and Electron frontend concurrently
   - Manages process lifecycle and cleanup

2. **Python Backend** (`core.py`):
   - **Audio Processing**: Cross-platform system audio capture (no microphone)
   - **Speech-to-Text**: Faster Whisper for transcription
   - **AI Integration**: Google Gemini for streaming responses
   - **Device Detection**: Platform-specific audio loopback detection
   - **Streaming Protocol**: Real-time chunk-based response streaming

3. **Electron Frontend** (`electron/`):
   - Modern gradient UI with glassmorphism effects
   - Real-time status indicators and typing animations
   - Communicates with Python backend via stdout/stdin

### Audio Architecture

**System Audio Only** - VoxAI captures clean computer audio without microphone interference:

- **macOS**: BlackHole/Loopback devices via Ladiocast routing
- **Windows**: WASAPI loopback (built-in)
- **Linux**: PulseAudio monitor devices (built-in)

Device detection prioritizes system audio loopback over microphone input.

### Communication Protocol

Backend communicates with frontend via structured logging:
- `STATUS::` - System status updates
- `TRANSCRIBED::` - Whisper transcription results
- `CHUNK::` - Streaming AI response chunks
- `AUDIO_DEVICE::` - Audio device information
- `[ERROR]`, `[THINKING]`, `[END]` - Special states

### Package Structure

```
voxai/
├── __init__.py          # Package marker
├── cli.py              # Main entry point
├── core.py             # Backend engine
└── electron/           # Frontend UI
    ├── package.json    # Electron dependencies
    ├── main.js         # Electron main process
    └── index.html      # UI interface
```

## Configuration

### Environment Variables
- `GENAI_API_KEY`: Google Gemini API key (required)
- `GENAI_MODEL`: Gemini model variant (default: "gemini-2.5-flash-latest")

### System Prompt Customization
Create `system_prompt.txt` in the working directory to override the default Principal Engineer-level system prompt.

## Development Notes

### Audio Device Handling
The `detect_audio_devices()` function prioritizes system audio loopback devices and provides platform-specific setup guidance when none are found.

### Streaming Implementation
AI responses use sentence-based chunking with fallback to character-based streaming for optimal readability. Includes error recovery and chat session reset mechanisms.

### Packaging
- Uses setuptools with `pyproject.toml` configuration
- Electron files are included as package data
- Entry point: `voxai = voxai.cli:main`

## Troubleshooting

### Audio Setup Issues
If no system audio device is detected, users need platform-specific loopback configuration as detailed in README.md.

### Node.js Requirements
Minimum Node.js version 14 is enforced. The CLI will exit with an error if an older version is detected.